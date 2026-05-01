"""Heuristic bot (Path A).

The "floor" bot: a fast, parameterized heuristic that ranks candidate targets
per owned planet using a linear mix of features and launches exact-plus-one
attacks when a net-positive capture is predicted.

Key ideas (drawn from the Kore 2022 winner's playbook):

  * Fleet-arrival table: for each target planet, we tabulate net incoming
    allied vs enemy ships by arrival time. Scoring factors in both the
    earliest capture window and the steady-state production stream.

  * Exact-plus-one sizing: ship count sent equals projected defender ships at
    arrival time + 1. Under-send is wasted; over-send is merely inefficient.

  * Intercept math: orbital targets are predicted via the Newton-iteration
    solver in engine/intercept.py; comets via the path-indexed solver.

  * Sun-avoidance: direct lines that cross the sun are rerouted to the closest
    tangent angle.

  * Parameterization: HEURISTIC_WEIGHTS is a flat dict of 20-ish floats. It
    feeds TuRBO tuning (Path A) and LLM-evolved mutations (EvoTune). Adding a
    new feature means adding one key here and one term in `_score_target`.

This file is intentionally simple and close to the metal. Do not add clever
caching or search here — that's Path B's job.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from orbitwars.bots.base import Action, Agent, Deadline, Move, no_op, obs_get
from orbitwars.engine.intercept import (
    BOARD_SIZE,
    CENTER,
    SUN_RADIUS,
    OrbitingTarget,
    comet_intercept,
    fleet_speed,
    initial_orbit_params,
    is_orbiting_planet,
    orbiting_intercept,
    path_crosses_sun,
    point_to_segment_distance,
    route_angle_avoiding_sun,
    ships_needed_for_speed,
    static_intercept_angle,
    static_intercept_turns,
    sun_tangent_angles,
)


# ---- Parameter config (tuned by TuRBO / EvoTune) ----

HEURISTIC_WEIGHTS: Dict[str, float] = {
    # Target scoring (higher = stronger preference)
    "w_production": 5.0,          # value per unit production
    "w_ships_cost": 0.02,         # per-ship cost in denominator
    "w_distance_cost": 0.05,      # per-unit Euclidean distance cost
    "w_travel_cost": 0.3,         # per-turn travel cost

    # Target preference multipliers
    "mult_neutral": 1.0,
    "mult_enemy": 1.8,            # bias toward offense over neutrals once in contact
    "mult_comet": 1.5,
    "mult_reinforce_ally": 0.0,   # disabled at v1 (we don't reinforce)

    # Sizing
    "ships_safety_margin": 1.0,   # extra ships beyond exact-plus-one
    "min_launch_size": 20.0,      # don't send fewer than this (matches starter)
    "max_launch_fraction": 0.8,   # leave 20% behind (tuned in W2 via TuRBO)

    # Expansion pacing
    "expand_cooldown_turns": 0.0, # allow every turn
    "keep_reserve_ships": 0.0,    # no forced reserve (exact-plus-one handles this)
    "agg_early_game": 1.0,
    "early_game_cutoff_turn": 100.0,

    # Sun handling
    "sun_avoidance_epsilon": 0.02,

    # Comet engagement
    "comet_max_time_mismatch": 1.5,

    # Search bias (used when MCTS wraps this — harmless here)
    "expand_bias": 0.5,
}


# ---- Observation shape helper ----

@dataclass
class ParsedObs:
    """Typed unpacking of the Kaggle obs for a single agent."""
    player: int
    step: int
    angular_velocity: float
    planets: List[List[Any]]
    initial_planets: List[List[Any]]
    fleets: List[List[Any]]
    next_fleet_id: int
    comet_planet_ids: set
    # Derived
    my_planets: List[List[Any]] = field(default_factory=list)
    enemy_planets: List[List[Any]] = field(default_factory=list)
    neutral_planets: List[List[Any]] = field(default_factory=list)
    planet_by_id: Dict[int, List[Any]] = field(default_factory=dict)
    initial_planet_by_id: Dict[int, List[Any]] = field(default_factory=dict)
    # Comet bookkeeping (per-comet precomputed path + current path index,
    # keyed by the comet's planet pid). Populated from ``obs.comets`` in
    # ``parse_obs``. Used by intercept math and the trajectory-obstruction
    # walk to treat comets as path-indexed moving targets rather than
    # falling through to the static-target branch.
    comet_path_by_pid: Dict[int, List[Tuple[float, float]]] = field(default_factory=dict)
    comet_path_index_by_pid: Dict[int, int] = field(default_factory=dict)
    # Per-turn invariants. ``is_orbiting_planet`` and
    # ``initial_orbit_params`` depend only on the (planet, initial_planet)
    # pair and never change within a single act() call, but the heuristic
    # + arrival-table builder collectively call them ~1M times per
    # 30-rollout MCTS turn. Populating once in ``parse_obs`` eliminates
    # the recomputation hot-path. Keyed by planet pid.
    is_orbiting_by_pid: Dict[int, bool] = field(default_factory=dict)
    orbit_params_by_pid: Dict[int, Tuple[float, float]] = field(default_factory=dict)


_COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)


def _infer_step_from_obs(obs) -> int:
    """Best-effort step inference when ``obs['step']`` is absent.

    Kaggle's orbit_wars engine only populates ``obs.step`` for seat 0
    (player 0). For seat 1 we must infer. Strategies, in order:

    1. **Comet path_index**. Comets spawn at fixed steps
       ``[50,150,250,350,450]``; each group's ``path_index`` directly
       encodes turns since spawn. ``obs.comets`` is append-ordered by
       spawn, so ``comets[i].path_index + COMET_SPAWN_STEPS[i]`` is the
       current step. Works from step 50 onwards.

    2. **Orbital phase**. For any orbiter visible in both
       ``planets`` and ``initial_planets``, ``step ≈ (current_angle
       − initial_angle) / ω``, modular at the orbital period. Unique
       only within the first orbital period (~125-250 turns); beyond
       that needs disambiguation we don't do here. Good enough early-
       game.

    3. **Zero**. Safe default for the fresh-state case.

    Returns 0 if no source agrees. Callers needing monotonicity across
    calls (e.g. for cooldowns) should override via an agent-level
    turn counter.
    """
    # (1) Comet-based: path_index from the first group with idx >= 0.
    comets = obs_get(obs, "comets", None) or []
    for i, g in enumerate(comets):
        if not isinstance(g, dict) or i >= len(_COMET_SPAWN_STEPS):
            continue
        idx = int(g.get("path_index", -1))
        if idx >= 0:
            return _COMET_SPAWN_STEPS[i] + idx

    # (2) Orbital-phase: find any orbiter, compute step from angle delta.
    ω = float(obs_get(obs, "angular_velocity", 0.0))
    if ω > 0.0:
        initial = {int(p[0]): p for p in obs_get(obs, "initial_planets", []) or []}
        comet_pids = set(obs_get(obs, "comet_planet_ids", []) or [])
        for pl in obs_get(obs, "planets", []) or []:
            pid = int(pl[0])
            if pid in comet_pids or pid not in initial:
                continue
            ip = initial[pid]
            # Skip non-orbiters (r + radius >= ROTATION_RADIUS_LIMIT=50).
            dx = float(ip[2]) - 50.0
            dy = float(ip[3]) - 50.0
            r = math.hypot(dx, dy)
            if r + float(ip[4]) >= 50.0:
                continue
            initial_angle = math.atan2(dy, dx)
            cdx = float(pl[2]) - 50.0
            cdy = float(pl[3]) - 50.0
            current_angle = math.atan2(cdy, cdx)
            delta = (current_angle - initial_angle) % (2.0 * math.pi)
            step = int(round(delta / ω))
            # Angle-wrap ambiguity: step modulo period. Early game
            # (step < period) this is exact. Late game it wraps; we
            # accept the modular answer here — agents that need
            # monotonic turn tracking must provide it externally.
            return step

    return 0


def parse_obs(obs, step_override: Optional[int] = None) -> ParsedObs:
    raw_step = obs_get(obs, "step", None)
    if step_override is not None:
        step = int(step_override)
    elif raw_step is not None:
        step = int(raw_step)
    else:
        step = _infer_step_from_obs(obs)
    p = ParsedObs(
        player=obs_get(obs, "player", 0),
        step=step,
        angular_velocity=obs_get(obs, "angular_velocity", 0.0),
        planets=list(obs_get(obs, "planets", [])),
        initial_planets=list(obs_get(obs, "initial_planets", [])),
        fleets=list(obs_get(obs, "fleets", [])),
        next_fleet_id=obs_get(obs, "next_fleet_id", 0),
        comet_planet_ids=set(obs_get(obs, "comet_planet_ids", [])),
    )
    for pl in p.planets:
        p.planet_by_id[pl[0]] = pl
        if pl[1] == p.player:
            p.my_planets.append(pl)
        elif pl[1] == -1:
            p.neutral_planets.append(pl)
        else:
            p.enemy_planets.append(pl)
    for ip in p.initial_planets:
        p.initial_planet_by_id[ip[0]] = ip

    # Cache per-turn orbit invariants for every planet so downstream
    # callers (build_arrival_table, _travel_turns, _intercept_position)
    # can skip recomputing.
    for pl in p.planets:
        pid = pl[0]
        ip = p.initial_planet_by_id.get(pid, pl)
        idx = float(ip[2]) - 50.0
        idy = float(ip[3]) - 50.0
        orb_r = math.hypot(idx, idy)
        init_angle = math.atan2(idy, idx)
        p.orbit_params_by_pid[pid] = (orb_r, init_angle)
        # is_orbiting rule: orb_r + planet_radius < ROTATION_RADIUS_LIMIT (50)
        p.is_orbiting_by_pid[pid] = (orb_r + float(pl[4])) < 50.0

    # Parse obs.comets. Engine schema:
    #   obs.comets: list of groups, each a dict with keys
    #     "planet_ids": [pid, ...]    — comet-planet ids in this group
    #     "paths":      [[[x,y], ...], ...]   — one path per pid (same order)
    #     "path_index": int           — current index shared across the group
    # The comet's current visible position is path[path_index]; the engine
    # increments path_index once per turn in its step-5 comet-move phase.
    for group in obs_get(obs, "comets", []) or []:
        pids = group.get("planet_ids", []) if isinstance(group, dict) else []
        paths = group.get("paths", []) if isinstance(group, dict) else []
        idx = int(group.get("path_index", -1)) if isinstance(group, dict) else -1
        for i, pid in enumerate(pids):
            if i < len(paths):
                path = [(float(pt[0]), float(pt[1])) for pt in paths[i]]
                p.comet_path_by_pid[int(pid)] = path
                p.comet_path_index_by_pid[int(pid)] = idx
    return p


# ---- Fleet-arrival table ----

@dataclass
class ArrivalEvent:
    turn: int
    owner: int
    ships: int


@dataclass
class ArrivalTable:
    """Per-target net ship projections, indexed by arrival turn.

    Used for:
      - Deciding if a reinforce is needed (net-incoming flips negative).
      - Exact-plus-one sizing under concurrent incoming fleets.
      - Blocking attacks on planets already being attacked by a teammate
        (we pass through for now — 2p games only have us).
    """
    events_by_pid: Dict[int, List[ArrivalEvent]] = field(default_factory=dict)

    def add(self, pid: int, turn: int, owner: int, ships: int) -> None:
        self.events_by_pid.setdefault(pid, []).append(ArrivalEvent(turn, owner, ships))

    def events(self, pid: int) -> List[ArrivalEvent]:
        return self.events_by_pid.get(pid, [])

    def projected_defender_at(
        self,
        pid: int,
        defender_owner: int,
        current_ships: int,
        production: int,
        arrival_turn: int,
    ) -> int:
        """Project defender ship count at `arrival_turn`, assuming:
          - Production continues at the given rate (only while owned).
          - Incoming ships flip ownership / decrement per combat rules.

        This is a simplified single-owner projection. For multi-front fights
        the full simulator in fast_engine.py is the ground truth — we use
        this estimate for fast scoring only.
        """
        owner = defender_owner
        ships = current_ships
        events = sorted(self.events(pid), key=lambda e: e.turn)
        last_t = 0
        for e in events:
            if e.turn > arrival_turn:
                break
            # Production between last_t and e.turn (only while owned)
            if owner != -1:
                ships += production * max(0, e.turn - last_t)
            if e.owner == owner:
                ships += e.ships
            else:
                ships -= e.ships
                if ships < 0:
                    owner = e.owner
                    ships = -ships
            last_t = e.turn
        # Production until arrival_turn
        if owner != -1:
            ships += production * max(0, arrival_turn - last_t)
        return ships


def build_arrival_table(
    po: ParsedObs, deadline: Optional[Deadline] = None,
) -> ArrivalTable:
    """Populate arrival events for every in-flight fleet against its target.

    We estimate the target by: the closest planet along the fleet's heading.
    That's imperfect (fleets can target any point in space or a planet that's
    rotated by arrival time), but it's good enough for a first-cut defense
    signal. The MCTS wrapper will replace this with a more precise estimate.

    ``deadline`` (optional) is checked between fleet iterations. This loop
    is O(fleets \u00d7 planets) with a Newton-intercept solve per pair \u2014 on
    dense late-game states (40+ fleets, 40+ planets) it runs 100-300 ms.
    Without a mid-loop check, an in-flight rollout can blow past the outer
    search deadline by the full duration of this function. When the
    deadline fires, we return the partial table accumulated so far; that
    is still a valid input to downstream scoring (just undercounts arrivals
    for the unvisited fleets).
    """
    table = ArrivalTable()
    for f in po.fleets:
        if deadline is not None and deadline.should_stop():
            break
        fid, owner, fx, fy, fangle, from_pid, fships = f
        # Best guess of target: planet whose perpendicular distance from fleet
        # ray is minimal and the planet is ahead of the fleet.
        best_pid = -1
        best_score = float("inf")
        best_turns = 0
        for pl in po.planets:
            pid = pl[0]
            if pid == from_pid:
                continue
            is_orb = po.is_orbiting_by_pid.get(pid, False)
            if is_orb:
                ir, ia = po.orbit_params_by_pid[pid]
                # NOTE: current_step = po.step - 2 matches _travel_turns'
                # engine-phase convention. A fleet observed at obs.step=S
                # has its k-th subsequent collision checked against planet
                # at angle init+ω*(S+k-2); Newton picks that up via
                # position_at(τ) = orbit(step=S-2+τ).
                ot = OrbitingTarget(
                    orbital_radius=ir, initial_angle=ia,
                    angular_velocity=po.angular_velocity,
                    current_step=po.step - 2,
                )
                # Quick check: if we aim at this orbital target, what's the
                # angular difference from the fleet's current heading?
                angle_to, t, _ = orbiting_intercept((fx, fy), ot, fships)
            else:
                angle_to = static_intercept_angle((fx, fy), (pl[2], pl[3]))
                t = static_intercept_turns((fx, fy), (pl[2], pl[3]), fships)
            # Circular distance between angles, in (0, pi]
            da = abs(((angle_to - fangle + math.pi) % (2 * math.pi)) - math.pi)
            # Score: prefer small angle difference, prefer closer.
            score = da * 10.0 + t * 0.1
            if score < best_score:
                best_score = score
                best_pid = pid
                best_turns = t
        if best_pid >= 0:
            table.add(best_pid, int(math.ceil(best_turns)) + po.step, owner, fships)
    return table


# ---- Target scoring ----

def _travel_turns(source: Tuple[float, float], target_pl: List[Any],
                  initial_pl: List[Any], angular_velocity: float,
                  step: int, ships: int,
                  source_radius: float = 0.0,
                  po: Optional[ParsedObs] = None) -> Tuple[float, float]:
    """Return (angle_to_aim, travel_turns_prediction).

    ``source_radius`` is the radius of the source planet. When > 0, the
    Newton is told the fleet actually launches at ``source + (r + 0.1) *
    dir`` — matching the engine's ``process_moves`` offset — so the
    predicted arrival time is correct to ~0.05 turns instead of being
    overestimated by ``(r+0.1)/v`` (up to ~2 turns on small fleets, which
    is exactly long enough for the orbital target to rotate out of the
    aim point and cause a miss).

    Orbit phase offset: at ``obs.step = S`` the observed planet angle is
    ``init + ω*(S-1)`` (verified empirically), and on the k-th fleet-turn
    after launch the engine collision check uses the planet at angle
    ``init + ω*(S+k-2)`` (pre-rotation for that step). Constructing
    ``OrbitingTarget`` with ``current_step = step - 2`` makes the Newton's
    ``position_at(τ) = orbit(step=S-2+τ)`` match the engine's collision
    target at fleet-turn τ.

    Comet branch: comets fail ``is_orbiting_planet`` (their orbital
    radius + radius >= ROTATION_RADIUS_LIMIT by construction) so the
    previous static-fallback aimed at the comet's current position —
    which is where the comet is now, not where it will be at arrival.
    When ``po`` is supplied and the target pid is a comet we use the
    path-indexed ``comet_intercept`` solver instead.
    """
    source_offset = source_radius + 0.1 if source_radius > 0.0 else 0.0
    tpid = int(target_pl[0])
    # Comet branch: target is on a precomputed path; intercept the path,
    # not the current position.
    if po is not None and tpid in po.comet_planet_ids:
        path = po.comet_path_by_pid.get(tpid)
        idx_now = po.comet_path_index_by_pid.get(tpid, -1)
        if path and idx_now >= 0:
            result = comet_intercept(
                source=source, comet_path=path, path_index_now=idx_now,
                ships=ships, source_offset=source_offset,
            )
            if result is None:
                return (0.0, float("inf"))
            angle, t, _ = result
            return (angle, t)
        # Fallthrough to static-aim if comet metadata is missing
        # (shouldn't happen with a well-formed obs).
    # Read orbit metadata from the per-turn cache when available.
    orb_cache = getattr(po, "is_orbiting_by_pid", None) if po is not None else None
    if orb_cache is not None and tpid in orb_cache:
        is_orb = orb_cache[tpid]
        ir_ia = po.orbit_params_by_pid.get(tpid) if is_orb else None
    else:
        is_orb = is_orbiting_planet(target_pl, initial_pl)
        ir_ia = initial_orbit_params(initial_pl) if is_orb else None
    if is_orb:
        ir, ia = ir_ia  # type: ignore[misc]
        ot = OrbitingTarget(
            orbital_radius=ir, initial_angle=ia,
            angular_velocity=angular_velocity, current_step=step - 2,
        )
        angle, t, _ = orbiting_intercept(
            source, ot, ships, source_offset=source_offset,
        )
        return angle, t
    else:
        tx, ty = target_pl[2], target_pl[3]
        angle = static_intercept_angle(source, (tx, ty))
        t = static_intercept_turns(
            source, (tx, ty), ships, source_offset=source_offset,
        )
        return angle, t


def _intercept_position(
    source: Tuple[float, float],
    target_pl: List[Any],
    initial_pl: List[Any],
    angular_velocity: float,
    step: int,
    travel_turns: float,
    po: Optional[ParsedObs] = None,
) -> Tuple[float, float]:
    """Where the target will be at fleet arrival time. Match the same
    engine-phase convention as ``_travel_turns``: collision at fleet-turn
    τ uses planet at angle ``init + ω*(step-2+τ)``.

    For static targets this is just the current observed position.
    For comet targets (when ``po`` supplies path info), we return the
    path position at ``path_index_now + ceil(travel_turns) - 1`` — the
    engine's step-3 collision index at fleet-turn k = ceil(travel_turns).
    """
    tpid = int(target_pl[0])
    if po is not None and tpid in po.comet_planet_ids:
        path = po.comet_path_by_pid.get(tpid)
        idx_now = po.comet_path_index_by_pid.get(tpid, -1)
        if path and idx_now >= 0:
            k = max(1, int(math.ceil(travel_turns)))
            aim_idx = min(idx_now + k - 1, len(path) - 1)
            return (float(path[aim_idx][0]), float(path[aim_idx][1]))
    orb_cache = getattr(po, "is_orbiting_by_pid", None) if po is not None else None
    if orb_cache is not None and tpid in orb_cache:
        is_orb = orb_cache[tpid]
        ir_ia = po.orbit_params_by_pid.get(tpid) if is_orb else None
    else:
        is_orb = is_orbiting_planet(target_pl, initial_pl)
        ir_ia = initial_orbit_params(initial_pl) if is_orb else None
    if is_orb:
        ir, ia = ir_ia  # type: ignore[misc]
        ot = OrbitingTarget(
            orbital_radius=ir, initial_angle=ia,
            angular_velocity=angular_velocity, current_step=step - 2,
        )
        return ot.position_at(travel_turns)
    return (float(target_pl[2]), float(target_pl[3]))


def _score_target(
    mp: List[Any],
    tp: List[Any],
    ip: List[Any],
    po: ParsedObs,
    table: ArrivalTable,
    weights: Dict[str, float],
    ships_to_send: int,
) -> Tuple[float, float, int]:
    """Return (score, aim_angle, defender_projection).

    Higher score = more attractive to launch this attack.
    """
    source_center = (float(mp[2]), float(mp[3]))
    source_radius = float(mp[4])
    angle, turns = _travel_turns(
        source_center, tp, ip,
        po.angular_velocity, po.step, ships_to_send,
        source_radius=source_radius, po=po,
    )
    if turns <= 0 or math.isinf(turns):
        return (-math.inf, 0.0, 0)

    # Avoid sun if needed — use the predicted intercept point (where the
    # target WILL BE at arrival), not the current planet position. For
    # orbiting planets and comets the two can differ substantially (tens
    # of units over a multi-turn flight); mis-routing from the wrong
    # reference point has caused direct sun-kills mid-flight in practice.
    target_pos = _intercept_position(
        source_center, tp, ip, po.angular_velocity, po.step, turns, po=po,
    )
    angle = route_angle_avoiding_sun(source_center, angle, target_pos)

    defender_ships = tp[5]
    defender_owner = tp[1]
    production = tp[6]
    arrival_turn = po.step + int(math.ceil(turns))
    projected = table.projected_defender_at(
        tp[0], defender_owner, defender_ships, production, arrival_turn,
    )

    # Preference multiplier
    if tp[0] in po.comet_planet_ids:
        mult = weights["mult_comet"]
    elif tp[1] == po.player:
        mult = weights["mult_reinforce_ally"]
    elif tp[1] == -1:
        mult = weights["mult_neutral"]
    else:
        mult = weights["mult_enemy"]

    # Core score: production value / (ships cost + travel cost)
    ships_cost = weights["w_ships_cost"] * max(1, ships_to_send)
    travel_cost = weights["w_travel_cost"] * turns
    distance = math.hypot(tp[2] - mp[2], tp[3] - mp[3])
    distance_cost = weights["w_distance_cost"] * distance
    production_value = weights["w_production"] * production

    # Early game aggression multiplier
    if po.step < weights["early_game_cutoff_turn"]:
        mult *= weights["agg_early_game"]

    denom = ships_cost + travel_cost + distance_cost + 1e-6
    score = mult * production_value / denom

    # If we can't actually capture (insufficient ships), penalize heavily.
    needed = projected + int(weights["ships_safety_margin"])
    if ships_to_send < needed and defender_owner != po.player:
        score -= 10.0  # can't capture

    return (score, angle, projected)


# ---- Trajectory obstruction check ----

# Sentinel codes returned by `_trajectory_obstruction`:
#   -1: path is clear — fleet reaches the intended target
#   -2: would cross the sun before hitting any planet
#   -3: would leave the board before hitting any planet
#   -4: walk budget exhausted without hitting anything (treat as waste)
# Any value >= 0 is the pid of an intervening planet the fleet would hit
# *before* reaching the intended target.

_OBSTR_CLEAR = -1
_OBSTR_SUN = -2
_OBSTR_OOB = -3
_OBSTR_WASTED = -4

# Maximum turns to simulate during the obstruction walk. A fleet at
# speed-cap 6 crosses the 100×100 board in ~17 turns; a slow v≈1 fleet
# in ~100 turns. 60 is a compromise: 95% of real intercepts arrive in
# under 30 turns and we reject the long-tail "pointed at nothing" shots
# rather than pay the budget.
_OBSTR_MAX_TURNS = 60


def _trajectory_obstruction(
    source_center: Tuple[float, float],
    source_radius: float,
    angle: float,
    ships: int,
    target_pid: int,
    po: ParsedObs,
) -> int:
    """Simulate the fleet's future trajectory until *something happens*
    and return a code describing what that was.

    CLEAR means the fleet's first collision is with the intended target —
    i.e. the engine's deterministic simulation will hit `target_pid`.
    Any other return value means the launch is a miss: a different
    planet eats the fleet first, the sun destroys it, it flies off the
    board, or it never hits anything within the walk budget.

    Why walk until something happens (instead of stopping at the
    predicted arrival_turn): if the intercept-solved angle is off by a
    fraction of a radian the fleet misses the target and keeps flying
    in a straight line at constant velocity until it dies somewhere.
    That post-target flight is exactly where "wrong planet" collisions
    come from — 223 out of 876 baseline shots in the verifier. Walking
    only to predicted-arrival hides those collisions.

    Engine step order inside a single engine turn S+k-1 (fleet-turn k
    of a fleet launched at step S):
      (a) Fleet movement — segment [pre, post] checked against every
          planet/comet at its pre-this-turn position. First hit in
          planet iteration order eats the fleet.
      (b) Planet rotation — each orbiting planet sweeps the arc
          (pre_rot_pos → post_rot_pos) and the post-fleet-move point is
          point-checked against each planet's swept segment.
      (c) Comet movement — each comet sweeps (path[idx+k-1] → path[idx+k])
          and the post-fleet-move point is checked the same way.

    The walk mirrors all three. Comet positioning is path-indexed, not
    the static ``planet[2],planet[3]`` coords (which for comets is their
    current location at obs.step=S but doesn't tell the walk how the
    comet moves between turns — hence the old walk always missed
    moving-comet collisions).

    Cost: O(walk_turns × n_planets). On dense states with ~30 planets
    and a 20-turn flight, ~600 point-to-segment evals per walk ≈
    150-300 μs. Top-K=5 fallback retries cap the per-my-planet
    obstruction cost at ~1-2 ms, comfortably within budget.
    """
    v = fleet_speed(ships)
    if v <= 0.0:
        return _OBSTR_WASTED

    dirx = math.cos(angle)
    diry = math.sin(angle)
    sx, sy = source_center
    offset = source_radius + 0.1
    # Fleet start position (engine-exact launch point).
    lx = sx + offset * dirx
    ly = sy + offset * diry

    ω = po.angular_velocity
    step = po.step

    # Precompute per-planet metadata so the hot loop does one sin/cos
    # pair per orbiter per turn instead of redoing init-params math.
    # Tuple layout:
    #   (pid, kind, a, b, c, d, radius)
    # where kind is 0=static, 1=orbiter, 2=comet; and (a,b,c,d) is kind
    # specific:
    #   static:  (static_x, static_y, unused, unused)
    #   orbiter: (orbital_radius, initial_angle, unused, unused)
    #   comet:   (comet_pid_as_index_into_po.comet_path_by_pid, 0, 0, 0)
    # Comets store nothing static — path lookup each turn via po dicts.
    _KIND_STATIC = 0
    _KIND_ORBITER = 1
    _KIND_COMET = 2
    planet_meta: List[Tuple[int, int, float, float, float, float, float]] = []
    for pl in po.planets:
        pid = int(pl[0])
        prad = float(pl[4])
        if pid in po.comet_planet_ids:
            planet_meta.append((pid, _KIND_COMET, 0.0, 0.0, 0.0, 0.0, prad))
            continue
        ip = po.initial_planet_by_id.get(pid, pl)
        if is_orbiting_planet(pl, ip):
            ir_o, ia_o = initial_orbit_params(ip)
            planet_meta.append((pid, _KIND_ORBITER, ir_o, ia_o, 0.0, 0.0, prad))
        else:
            planet_meta.append(
                (pid, _KIND_STATIC, float(pl[2]), float(pl[3]), 0.0, 0.0, prad),
            )

    for k in range(1, _OBSTR_MAX_TURNS + 1):
        # Fleet segment during fleet-turn k: [pre, post].
        pre_x = lx + (k - 1) * v * dirx
        pre_y = ly + (k - 1) * v * diry
        post_x = lx + k * v * dirx
        post_y = ly + k * v * diry
        pre = (pre_x, pre_y)
        post = (post_x, post_y)

        # Out-of-bounds check (engine removes fleets that step off-board).
        if not (0.0 <= post_x <= BOARD_SIZE and 0.0 <= post_y <= BOARD_SIZE):
            return _OBSTR_OOB

        # Sun crossing: segment-to-center distance < SUN_RADIUS.
        if point_to_segment_distance((CENTER, CENTER), pre, post) < SUN_RADIUS:
            return _OBSTR_SUN

        pre_rot_step = step + k - 2
        post_rot_step = step + k - 1

        # (a) Fleet-movement collision vs each planet at its pre-this-turn
        # position. First hit in iteration order wins (mirrors engine's
        # break-on-first-hit loop in the fleet-move phase).
        for (pid, kind, a, b, _c, _d, prad) in planet_meta:
            if kind == _KIND_ORBITER:
                theta = b + ω * pre_rot_step
                px = CENTER + a * math.cos(theta)
                py = CENTER + a * math.sin(theta)
            elif kind == _KIND_COMET:
                path = po.comet_path_by_pid.get(pid)
                idx_now = po.comet_path_index_by_pid.get(pid, -1)
                if not path or idx_now < 0:
                    continue
                # Pre-this-turn comet position = path[idx_now + k - 1].
                # Past end-of-path = comet has expired; skip.
                pre_idx = idx_now + k - 1
                if pre_idx >= len(path) or pre_idx < 0:
                    continue
                px = path[pre_idx][0]
                py = path[pre_idx][1]
            else:  # static
                px = a
                py = b
            if point_to_segment_distance((px, py), pre, post) < prad:
                return _OBSTR_CLEAR if pid == target_pid else pid

        # (b) Orbital-planet rotation sweep. Each orbiter moves from its
        # pre-rot position to its post-rot position; the fleet's post-
        # fleet-move point is checked against that arc (segment approx).
        # First hit destroys the fleet.
        for (pid, kind, a, b, _c, _d, prad) in planet_meta:
            if kind != _KIND_ORBITER:
                continue
            theta_pre = b + ω * pre_rot_step
            theta_post = b + ω * post_rot_step
            pre_px = CENTER + a * math.cos(theta_pre)
            pre_py = CENTER + a * math.sin(theta_pre)
            post_px = CENTER + a * math.cos(theta_post)
            post_py = CENTER + a * math.sin(theta_post)
            if point_to_segment_distance(
                (post_x, post_y), (pre_px, pre_py), (post_px, post_py),
            ) < prad:
                return _OBSTR_CLEAR if pid == target_pid else pid

        # (c) Comet movement sweep. Each comet moves from path[idx+k-1]
        # to path[idx+k] (step-5 engine phase). Fleet's post-fleet-move
        # point is checked against that segment.
        for (pid, kind, _a, _b, _c, _d, prad) in planet_meta:
            if kind != _KIND_COMET:
                continue
            path = po.comet_path_by_pid.get(pid)
            idx_now = po.comet_path_index_by_pid.get(pid, -1)
            if not path or idx_now < 0:
                continue
            pre_idx = idx_now + k - 1
            post_idx = idx_now + k
            # Past end-of-path = comet has expired; no more collisions.
            if post_idx >= len(path) or pre_idx < 0 or pre_idx >= len(path):
                continue
            pre_p = path[pre_idx]
            post_p = path[post_idx]
            if point_to_segment_distance(
                (post_x, post_y), (pre_p[0], pre_p[1]), (post_p[0], post_p[1]),
            ) < prad:
                return _OBSTR_CLEAR if pid == target_pid else pid

    # Walked the full budget without hitting anything. Fleet is wasted.
    return _OBSTR_WASTED


# ---- Main agent ----

@dataclass
class LaunchIntent:
    """One planner-emitted launch, with the target the planner *intended* to
    hit and the predicted arrival turn. Written to the agent side-channel
    ``HeuristicAgent.last_launch_intents`` so verifier tools (and future
    miss-logging telemetry) can compare emitted vs. actual without having
    to reverse-engineer target attribution from angle matching.
    """
    turn: int          # po.step at emission
    from_pid: int
    target_pid: int
    angle: float
    ships: int
    predicted_travel_turns: float
    predicted_arrival_turn: int
    score: float


@dataclass
class _LaunchState:
    last_launch_turn: Dict[int, int] = field(default_factory=dict)


class HeuristicAgent(Agent):
    """Path A bot. Parameterized, fast, tournament baseline."""

    name = "heuristic"

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = dict(HEURISTIC_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self._state = _LaunchState()
        # Side-channel: populated by _plan_moves, read by diag tools +
        # the pytest zero-miss gate. One entry per launch emitted this
        # turn, in the same order as the wire `moves` list so
        # `fleet_id_n = next_fleet_id + n` maps 1:1.
        self.last_launch_intents: List[LaunchIntent] = []
        # Full per-game launch history (append-only). Each LaunchIntent
        # has the turn it was emitted plus the predicted target/arrival.
        # Negligible cost during play (~1 list append per launch, a few
        # hundred entries per game). External tooling pairs entries with
        # engine-captured combat_lists to produce the hit/miss report.
        self.launch_log: List[LaunchIntent] = []
        # Monotonic turn counter. Required for seat-1 play: Kaggle's
        # orbit_wars engine omits obs.step for player 1, so parse_obs's
        # inference-from-state is only unique within the first orbital
        # period (~125-250 turns). This counter supplies the unambiguous
        # answer across a full 500-turn game. Reset on game-start
        # detection in act().
        self._turn_counter: Optional[int] = None

    # ---- Public: Kaggle + Agent contract ----

    def act(self, obs, deadline: Deadline) -> Action:
        # Always stage the no-op first so we're safe against timeouts.
        deadline.stage(no_op())

        # Resolve step and detect match-start.
        # Seat 0: obs.step is authoritative. step==0 -> new match.
        # Seat 1: obs.step is None (Kaggle engine quirk). We rely on a
        # monotonic counter, reset when next_fleet_id regresses (or on
        # very first call) — which is the strongest always-available
        # match-start signal.
        raw_step = obs_get(obs, "step", None)
        curr_nfid = int(obs_get(obs, "next_fleet_id", 0))
        if raw_step is not None:
            step = int(raw_step)
            is_match_start = (step == 0)
            self._turn_counter = step
        else:
            prev_nfid = getattr(self, "_prev_next_fleet_id", None)
            first_call = self._turn_counter is None
            # next_fleet_id is monotonically non-decreasing within a
            # match. A drop to 0 (or a first call with nfid==0) is the
            # match-start edge. Using nfid rather than "fleets list
            # empty" is robust to arbitrary early-game turns where no
            # one has launched yet.
            is_match_start = first_call or (
                prev_nfid is not None and prev_nfid > curr_nfid
            )
            if is_match_start:
                step = 1
            else:
                step = (self._turn_counter or 0) + 1
            self._turn_counter = step
        self._prev_next_fleet_id = curr_nfid

        po = parse_obs(obs, step_override=step)

        # Reset per-match state only on true match-start.
        if is_match_start:
            self._state = _LaunchState()
            # New game -> fresh launch log; keeps post-mortem telemetry
            # scoped to the current match.
            self.launch_log = []

        if not po.my_planets:
            return no_op()

        # Outer-deadline check between stages: build_arrival_table scales
        # with fleet count × planet count (intercept math per pair) and
        # on dense late-game states runs 50-200 ms. When the caller
        # (e.g. MCTS rollouts) supplies a Deadline with an absolute
        # hard_stop_at, short-circuit before paying that cost. Returns
        # the no-op staged above so the call is still action-valid.
        if deadline.should_stop():
            return no_op()

        # Thread deadline into build_arrival_table \u2014 on dense states its
        # O(fleets \u00d7 planets) intercept loop dominates (100-300 ms) and
        # must be abortable mid-way or a single in-flight rollout blows
        # past the outer search deadline.
        table = build_arrival_table(po, deadline=deadline)

        if deadline.should_stop():
            return no_op()

        moves: Action = self._plan_moves(po, table, deadline=deadline)

        # Cooldown bookkeeping
        for m in moves:
            self._state.last_launch_turn[int(m[0])] = po.step

        deadline.stage(moves)
        return moves

    # ---- Planning ----

    def _plan_moves(
        self, po: ParsedObs, table: ArrivalTable,
        deadline: Optional[Deadline] = None,
    ) -> Action:
        moves: List[Move] = []
        # Reset the per-turn launch-intent log. Verifier/telemetry reads
        # it straight after act() returns.
        self.last_launch_intents = []
        w = self.weights
        reserve = int(w["keep_reserve_ships"])
        cd = int(w["expand_cooldown_turns"])

        # Build the list of candidate targets once (excludes our own planets
        # for attack; includes them for reinforce consideration).
        candidates = [p for p in po.planets]

        for mp in po.my_planets:
            # Per-my-planet deadline check. The inner loop runs intercept
            # math for every (my_planet, target) pair — 30-100 μs per pair
            # × 40 planets = ~2 ms per outer-iteration. On dense late-game
            # states with 20 my-planets we can still accumulate ~40 ms in
            # the outer loop. Breaking mid-way returns the partial move
            # list built so far (still a valid Action), which is strictly
            # better than overrunning the outer MCTS search deadline.
            if deadline is not None and deadline.should_stop():
                break
            mpid = int(mp[0])
            available = int(mp[5]) - reserve
            if available < int(w["min_launch_size"]):
                continue

            # Defense guard: if enemy ships are inbound and our defenders can't
            # hold, don't drain this planet for offense. Compute the net
            # shortfall and reduce `available` by exactly that much.
            incoming_enemy = 0
            incoming_ally = 0
            for ev in table.events(mpid):
                if ev.owner == po.player:
                    incoming_ally += ev.ships
                else:
                    incoming_enemy += ev.ships
            # Assume production keeps coming while we hold; shortfall estimate
            # is a cheap approximation (production time-integral depends on
            # arrival ordering — handled precisely by fast_engine when MCTS
            # wraps this).
            projected_def = int(mp[5]) + incoming_ally
            shortfall = max(0, incoming_enemy + 1 - projected_def)
            if shortfall > 0:
                # Keep exactly enough to defend; no extra hoarding.
                available = max(0, int(mp[5]) - shortfall)
            if available < int(w["min_launch_size"]):
                continue

            # Cooldown (skip check entirely when cd == 0 — avoids any chance
            # of stale last_launch_turn values blocking launches)
            if cd > 0:
                last = self._state.last_launch_turn.get(mpid, -10_000)
                if po.step - last < cd:
                    continue

            # Score all candidate targets at several possible send sizes.
            # We keep the full ranked list so that when the top-scoring
            # target's trajectory is obstructed (passes through another
            # planet, grazes the sun, leaves the board) we can fall
            # through to the next-best target instead of launching into
            # nothing. Top-K=5 keeps the obstruction-walk cost bounded
            # (each walk is ~50-150 μs, so 5 × 20 my-planets ≈ 15 ms
            # worst case — comfortably under the outer budget).
            ranked: List[Tuple[float, float, int, int, Any]] = []
            for tp in candidates:
                # Inner-loop deadline check. candidates = all planets, so
                # this loop is O(len(planets)) per my-planet with one
                # Newton-intercept solve per iteration via _travel_turns.
                # On dense states it can accumulate 5-15 ms per planet;
                # across 20 my-planets the full outer loop is 100-300 ms.
                # Without this check, an in-flight HeuristicAgent.act call
                # from a rollout can keep running past the search deadline.
                if deadline is not None and deadline.should_stop():
                    break
                if tp[0] == mpid:
                    continue
                ip = po.initial_planet_by_id.get(tp[0], tp)

                # Trial size = exact-plus-one (projected + safety margin).
                # First a cheap estimate of travel turns with a guess ship size:
                _, t_guess = _travel_turns(
                    (mp[2], mp[3]), tp, ip,
                    po.angular_velocity, po.step, max(50, available // 2),
                    source_radius=float(mp[4]), po=po,
                )
                if math.isinf(t_guess) or t_guess <= 0:
                    continue
                arrival = po.step + int(math.ceil(t_guess))
                proj = table.projected_defender_at(
                    tp[0], tp[1], tp[5], tp[6], arrival,
                )
                needed = max(int(w["min_launch_size"]),
                             proj + int(w["ships_safety_margin"]))
                # Allies: send much smaller reinforcement
                if tp[1] == po.player:
                    needed = max(int(w["min_launch_size"]), proj // 5 + 1)

                cap = int(available * w["max_launch_fraction"])
                ships_to_send = min(needed, cap, available)
                if ships_to_send < int(w["min_launch_size"]):
                    continue

                score, angle, _ = _score_target(
                    mp, tp, ip, po, table, self.weights, ships_to_send,
                )
                if not math.isfinite(score):
                    continue
                ranked.append((score, angle, ships_to_send, int(tp[0]), tp))

            if not ranked:
                continue

            ranked.sort(key=lambda t: t[0], reverse=True)

            # Try top-K; launch the first target whose full trajectory is
            # clear (no intervening planets, no sun crossing, no
            # off-board step). If *all* top-K are obstructed, skip this
            # my-planet this turn — better a pass than a wasted fleet.
            chosen: Optional[Tuple[float, float, int, int, float]] = None
            K = 5
            for (score, angle, ships_to_send, target_pid, tp) in ranked[:K]:
                if score <= 0:
                    break
                # Recompute travel time at the *actual* ship count so we
                # can register a correct arrival in the fleet-arrival
                # table once we commit to this launch.
                ip_t = po.initial_planet_by_id.get(target_pid, tp)
                _, t_actual = _travel_turns(
                    (mp[2], mp[3]), tp, ip_t,
                    po.angular_velocity, po.step, ships_to_send,
                    source_radius=float(mp[4]), po=po,
                )
                if math.isinf(t_actual) or t_actual <= 0:
                    continue
                obstr = _trajectory_obstruction(
                    source_center=(float(mp[2]), float(mp[3])),
                    source_radius=float(mp[4]),
                    angle=float(angle),
                    ships=int(ships_to_send),
                    target_pid=int(target_pid),
                    po=po,
                )
                if obstr == _OBSTR_CLEAR:
                    chosen = (score, angle, ships_to_send, target_pid, t_actual)
                    break

            if chosen is None:
                continue

            score, angle, ships_to_send, target_pid, t_actual = chosen
            moves.append([mpid, float(angle), int(ships_to_send)])
            # Side-channel: record the planner's *intent* — (from, target,
            # predicted travel). Verifier tools use this to tell a true
            # miss ("we aimed at planet X and the fleet didn't arrive")
            # from benign outcomes ("we aimed at planet X but X was
            # captured before arrival"). Order matches the wire list so
            # callers can map `fleet_id = next_fleet_id + i`.
            intent = LaunchIntent(
                turn=int(po.step),
                from_pid=int(mpid),
                target_pid=int(target_pid),
                angle=float(angle),
                ships=int(ships_to_send),
                predicted_travel_turns=float(t_actual),
                predicted_arrival_turn=int(po.step) + int(math.ceil(t_actual)),
                score=float(score),
            )
            self.last_launch_intents.append(intent)
            # Also append to the full-game launch log for post-mortem
            # telemetry. Cheap (one list append per launch); no per-turn
            # cost beyond the emission itself.
            self.launch_log.append(intent)
            # Register this launch in the arrival table so subsequent target
            # scoring (in this same turn's planning) sees it.
            table.add(
                target_pid,
                po.step + int(math.ceil(t_actual)),
                po.player, ships_to_send,
            )

        return moves


def build(**overrides) -> HeuristicAgent:
    """Factory for the heuristic agent. Accepts weight overrides."""
    return HeuristicAgent(weights=overrides if overrides else None)
