"""MCTS action generator for HeuristicAgent-priored search.

For each owned planet we enumerate a small, structured set of candidate
moves (attack each reachable target at a few ship-size fractions, plus a
"hold" no-op), rank them via the heuristic's existing `_score_target`, and
emit the top-K with softmax-normalized priors.

Why this design (v1):
  * Kaggle RTS action spaces are naturally factored: each owned planet
    independently chooses its launch, and the joint is the product. We
    expose the factored shape directly so the Gumbel-AZ root can either
    sample per-planet independently (cheap, good-enough) or sample joint
    top-K over the product (more faithful, Week-3 upgrade).
  * Ship-size fractions `{0.25, 0.5, 1.0}` replace the heuristic's
    one-size-fits-all: MCTS can discover that a smaller probe is
    preferable against strong defenders, or that full-send is optimal
    against cheap neutrals.
  * "hold" is always included — skipping a planet's turn can be optimal
    (e.g. when incoming enemy fleets force a pure defense).

Deliberately skipped in v1:
  * Defensive intercept angles against inbound enemy fleets. The heuristic
    already credits defense via the arrival-table; MCTS sees the same state
    so defense emerges implicitly from rollouts. Intercept moves are on the
    W3 feature list.
  * Sun-tangent re-routes. Currently handled inside
    `heuristic.route_angle_avoiding_sun` — we inherit that behavior via
    `_score_target`. Explicit tangent move variants can be added if needed.

Test coverage (tests/test_mcts_actions.py):
  * Bounds: max_per_planet is respected; ships > available; prior sums to 1.
  * Hold-move is always present.
  * Against a noop-like opponent state, priors rank reachable enemy targets
    above unreachable ones.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from orbitwars.bots.heuristic import (
    HEURISTIC_WEIGHTS,
    ArrivalTable,
    ParsedObs,
    _score_target,
    _travel_turns,
)
from orbitwars.engine.intercept import (
    is_orbiting_planet,
)
from orbitwars.mcts.bokr_widen import BOKRKernelSelector


# Move kinds — used by the search layer to prune / bias at non-root nodes.
KIND_ATTACK_ENEMY    = "attack_enemy"
KIND_ATTACK_NEUTRAL  = "attack_neutral"
KIND_ATTACK_COMET    = "attack_comet"
KIND_REINFORCE_ALLY  = "reinforce_ally"
KIND_HOLD            = "hold"


@dataclass(frozen=True)
class PlanetMove:
    """One candidate move from one owned planet.

    Immutable so callers can stash it in tree nodes without worrying about
    mutation. `prior` is softmax-normalized inside a planet's move list.
    """
    from_pid: int
    angle: float
    ships: int
    target_pid: int          # -1 for hold
    kind: str
    prior: float = 0.0       # populated by generate_per_planet_moves
    raw_score: float = 0.0   # pre-softmax heuristic score (diagnostic)

    @property
    def is_hold(self) -> bool:
        return self.kind == KIND_HOLD

    def to_move(self) -> List:
        """Kaggle-wire format: [from_pid, angle, ships]. HOLD returns []."""
        if self.is_hold:
            return []
        return [int(self.from_pid), float(self.angle), int(self.ships)]


@dataclass
class ActionConfig:
    """Knobs for the action generator.

    max_per_planet: cap on emitted moves per owned planet (incl. hold).
    ship_fractions: discrete send-sizes as fractions of available ships.
    softmax_temperature: higher → flatter prior (more exploration at root).
    min_launch_size: drop moves below this many ships — matches heuristic.

    Angle refinement (BOKR-style):
      angle_refinement_n_grid: per (target, ship-fraction) pair, instead
          of emitting ONE move at the heuristic's analytic intercept,
          emit ``angle_refinement_n_grid`` moves spread ± ``angle_refinement_range``
          radians around it. ``1`` = current behavior (single base angle
          per target). ``3`` = base ± offset (typical BOKR-mini); ``5`` =
          finer grid. Odd values keep the base angle always represented.
          Upper-bounded by the Gumbel root budget: Gumbel will halve
          across whatever top-k arrives, so more grid points just give
          the search more side-pass structure to discover. Keep
          ``max_per_planet`` in mind — grid × target × fraction explodes
          quickly without the top-K trim.
      angle_refinement_range: half-width of the angle grid in radians.
          ~0.1 rad ≈ 5.7° matches Kore 2022's empirical "pass on either
          side" sweet spot for orbital targets. Wider than ~0.2 rad
          starts aiming at nothing in particular.
    """
    max_per_planet: int = 8
    include_hold: bool = True
    ship_fractions: Tuple[float, ...] = (0.25, 0.5, 1.0)
    softmax_temperature: float = 1.0
    min_launch_size: int = 20
    hold_bonus_score: float = 0.0   # added to HOLD raw score before softmax
    angle_refinement_n_grid: int = 1
    angle_refinement_range: float = 0.1


def _softmax(xs: List[float], temperature: float) -> List[float]:
    """Numerically stable softmax. Returns a probability vector."""
    if not xs:
        return []
    t = max(1e-6, float(temperature))
    m = max(xs)
    exps = [math.exp((x - m) / t) for x in xs]
    z = sum(exps)
    if z <= 0:  # pragma: no cover
        return [1.0 / len(xs)] * len(xs)
    return [e / z for e in exps]


def _kind_for_target(po: ParsedObs, tp: List) -> str:
    pid, owner = tp[0], tp[1]
    if pid in po.comet_planet_ids:
        return KIND_ATTACK_COMET
    if owner == po.player:
        return KIND_REINFORCE_ALLY
    if owner == -1:
        return KIND_ATTACK_NEUTRAL
    return KIND_ATTACK_ENEMY


def generate_per_planet_moves(
    po: ParsedObs,
    table: ArrivalTable,
    weights: Optional[Dict[str, float]] = None,
    cfg: Optional[ActionConfig] = None,
) -> Dict[int, List[PlanetMove]]:
    """For each owned planet, return up to cfg.max_per_planet candidate moves.

    Empty list for any planet with no available ships / no reachable target.
    Always includes a HOLD move when cfg.include_hold is True.
    """
    weights = dict(HEURISTIC_WEIGHTS) if weights is None else dict(weights)
    cfg = cfg or ActionConfig()

    out: Dict[int, List[PlanetMove]] = {}
    for mp in po.my_planets:
        mpid = int(mp[0])
        available = int(mp[5])

        # Enumerate raw candidates first; softmax over them at the end.
        raw: List[Tuple[float, PlanetMove]] = []

        if cfg.include_hold:
            hold = PlanetMove(
                from_pid=mpid, angle=0.0, ships=0, target_pid=-1,
                kind=KIND_HOLD, prior=0.0, raw_score=cfg.hold_bonus_score,
            )
            raw.append((cfg.hold_bonus_score, hold))

        if available < cfg.min_launch_size:
            # Only HOLD is possible — emit it and move on.
            if raw:
                raw[0][1].__dict__  # no-op (frozen dataclass: can't mutate)
                moves = [PlanetMove(
                    from_pid=raw[0][1].from_pid, angle=0.0, ships=0,
                    target_pid=-1, kind=KIND_HOLD, prior=1.0,
                    raw_score=raw[0][0],
                )]
                out[mpid] = moves
            else:
                out[mpid] = []
            continue

        for tp in po.planets:
            tpid = int(tp[0])
            if tpid == mpid:
                continue
            ip = po.initial_planet_by_id.get(tpid, tp)
            kind = _kind_for_target(po, tp)

            for frac in cfg.ship_fractions:
                ships = max(cfg.min_launch_size, int(available * frac))
                if ships > available:
                    continue
                if ships < cfg.min_launch_size:
                    continue

                score, angle, _proj = _score_target(
                    mp, tp, ip, po, table, weights, ships,
                )
                if not math.isfinite(score):
                    continue

                # Emit angle variants around the heuristic's analytic
                # intercept. All variants share the base raw_score
                # because the score is ~angle-invariant at this scale —
                # side-pass discovery is MCTS's job during search.
                # n_grid=1 preserves the legacy single-angle behavior.
                if cfg.angle_refinement_n_grid > 1:
                    sel = BOKRKernelSelector(
                        base_angle=float(angle),
                        angle_range=float(cfg.angle_refinement_range),
                        n_grid=int(cfg.angle_refinement_n_grid),
                    )
                    variant_angles = sel.candidate_angles()
                else:
                    variant_angles = [float(angle)]
                for var_angle in variant_angles:
                    move = PlanetMove(
                        from_pid=mpid, angle=float(var_angle),
                        ships=int(ships), target_pid=tpid, kind=kind,
                        prior=0.0, raw_score=score,
                    )
                    raw.append((score, move))

        # Rank descending by raw_score, keep top-K.
        raw.sort(key=lambda t: t[0], reverse=True)
        raw = raw[: cfg.max_per_planet]

        # Softmax priors.
        scores = [s for (s, _) in raw]
        priors = _softmax(scores, cfg.softmax_temperature)
        out[mpid] = [
            PlanetMove(
                from_pid=m.from_pid, angle=m.angle, ships=m.ships,
                target_pid=m.target_pid, kind=m.kind,
                prior=p, raw_score=m.raw_score,
            )
            for (_, m), p in zip(raw, priors)
        ]

    return out


# ---- Joint-action helpers (used by the root sampler in gumbel_search.py) ----

@dataclass(frozen=True)
class JointAction:
    """One joint per-turn action: a tuple of PlanetMove (one per owned planet).

    The order is stable by planet ID for deterministic hashing inside the
    tree.
    """
    moves: Tuple[PlanetMove, ...]

    def to_wire(self) -> List[List]:
        """Kaggle-wire format. Drops HOLD moves."""
        return [m.to_move() for m in self.moves if not m.is_hold]

    def joint_prior(self) -> float:
        """Product of per-planet priors (independent sampling assumption)."""
        p = 1.0
        for m in self.moves:
            p *= max(m.prior, 1e-9)
        return p


def sample_joint(
    per_planet: Dict[int, List[PlanetMove]],
    rng,  # random.Random — typed loosely to avoid stdlib import cycles
) -> JointAction:
    """Independently sample one PlanetMove from each planet's prior dist."""
    picks: List[PlanetMove] = []
    for pid in sorted(per_planet.keys()):
        moves = per_planet[pid]
        if not moves:
            continue
        weights = [max(m.prior, 0.0) for m in moves]
        total = sum(weights)
        if total <= 0:
            picks.append(moves[0])
            continue
        r = rng.uniform(0.0, total)
        acc = 0.0
        for m, w in zip(moves, weights):
            acc += w
            if r <= acc:
                picks.append(m)
                break
        else:
            picks.append(moves[-1])
    return JointAction(tuple(picks))
