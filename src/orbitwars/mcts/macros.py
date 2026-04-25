"""Macro-action library: pre-expanded "obvious" joint actions inserted
at the search root as additional candidates.

Plan §6.4 "Macro-action library pre-expanded at the root: hold,
mass-attack-nearest, retreat, reinforce-weakest." Insurance against a
bad prior; documented +Elo trick from microRTS literature.

Each macro is a ``(parsed_obs, per_planet) -> JointAction | None``
function that produces ONE complete joint action (one PlanetMove per
owned planet). The caller (``GumbelRootSearch``) appends them to the
sampled-Gumbel candidate list so Sequential Halving has a chance to
pick them when the prior misses obviously good plays. They are NOT
protected from SH pruning — the heuristic anchor remains the only
protected candidate.

Macros currently shipped:

* ``macro_hold_all`` — every planet holds. Useful when search wants
  to "stand pat and accumulate"; the prior often under-weights HOLD
  because the heuristic's raw_score discounts non-launch actions.
* ``macro_mass_attack_nearest_enemy`` — every planet sends a fraction
  of its garrison at its nearest enemy planet. Catches "overwhelm
  with synchronized strikes" plays the per-planet greedy heuristic
  doesn't compose.
* ``macro_reinforce_weakest_ally`` — every planet sends half its
  garrison toward the globally weakest friendly. Captures defensive
  consolidation plays (frequently right when an enemy opener has us
  spread thin).
* ``macro_retreat_to_largest_ally`` — every non-largest friendly
  sends half its garrison toward the largest ally. Catches the
  "consolidate before getting picked off" pattern.

The macros produce HOLD moves for any source planet with too few
ships (< ``min_launch_size``, default 5), which keeps them legal even
in early-game states where some planets can't afford to launch.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from orbitwars.bots.heuristic import ParsedObs
from orbitwars.mcts.actions import (
    JointAction, KIND_ATTACK_ENEMY, KIND_ATTACK_NEUTRAL,
    KIND_HOLD, KIND_REINFORCE_ALLY, PlanetMove,
)


_DEFAULT_MIN_LAUNCH = 5  # ships; below this, source planet HOLDs


def _direct_angle(src: List[Any], tgt: List[Any]) -> float:
    """Angle from src to tgt in radians, atan2(dy, dx)."""
    dx = float(tgt[2]) - float(src[2])
    dy = float(tgt[3]) - float(src[3])
    return math.atan2(dy, dx)


def _hold_move(pid: int) -> PlanetMove:
    return PlanetMove(
        from_pid=pid, angle=0.0, ships=0, target_pid=-1,
        kind=KIND_HOLD, prior=1.0,
    )


def _launch_move(
    pid: int, target: List[Any], src: List[Any], fraction: float, kind: str,
) -> PlanetMove:
    avail = int(src[5])
    ships = max(1, int(avail * fraction))
    return PlanetMove(
        from_pid=pid, angle=_direct_angle(src, target),
        ships=ships, target_pid=int(target[0]),
        kind=kind, prior=1.0,
    )


def macro_hold_all(
    po: ParsedObs, per_planet: Dict[int, List[PlanetMove]],
) -> Optional[JointAction]:
    """Every owned planet holds.

    Always returns a non-None JointAction (HOLD is always legal), so this
    macro is a guaranteed addition to the candidate set. Useful as a
    floor when other macros can't compose (no enemies / single planet
    edge cases).
    """
    if not per_planet:
        return None
    moves = tuple(_hold_move(int(pid)) for pid in sorted(per_planet.keys()))
    return JointAction(moves=moves)


def macro_mass_attack_nearest_enemy(
    po: ParsedObs, per_planet: Dict[int, List[PlanetMove]],
    *, fraction: float = 1.0, min_launch: int = _DEFAULT_MIN_LAUNCH,
) -> Optional[JointAction]:
    """Every planet attacks its individually-nearest enemy planet.

    Returns None if there are no enemies (so the macro doesn't degenerate
    into HOLD-all, which is already covered).
    """
    if not po.enemy_planets:
        return None
    moves: List[PlanetMove] = []
    has_launch = False
    for pid in sorted(per_planet.keys()):
        src = po.planet_by_id.get(int(pid))
        if src is None:
            moves.append(_hold_move(int(pid)))
            continue
        avail = int(src[5])
        if avail < min_launch:
            moves.append(_hold_move(int(pid)))
            continue
        # Nearest enemy by squared distance.
        nearest = min(
            po.enemy_planets,
            key=lambda t: (float(t[2]) - float(src[2])) ** 2
            + (float(t[3]) - float(src[3])) ** 2,
        )
        moves.append(_launch_move(
            int(pid), nearest, src, fraction, KIND_ATTACK_ENEMY,
        ))
        has_launch = True
    if not has_launch:
        return None
    return JointAction(moves=tuple(moves))


def macro_reinforce_weakest_ally(
    po: ParsedObs, per_planet: Dict[int, List[PlanetMove]],
    *, fraction: float = 0.5, min_launch: int = _DEFAULT_MIN_LAUNCH,
) -> Optional[JointAction]:
    """Every planet (except the weakest itself) sends ~50% of its garrison
    to the globally weakest friendly planet.

    Returns None if we have fewer than 2 friendly planets (no recipient).
    """
    if len(po.my_planets) < 2:
        return None
    weakest = min(po.my_planets, key=lambda p: int(p[5]))
    weakest_pid = int(weakest[0])
    moves: List[PlanetMove] = []
    has_launch = False
    for pid in sorted(per_planet.keys()):
        src = po.planet_by_id.get(int(pid))
        if src is None or int(pid) == weakest_pid:
            moves.append(_hold_move(int(pid)))
            continue
        avail = int(src[5])
        if avail < min_launch:
            moves.append(_hold_move(int(pid)))
            continue
        moves.append(_launch_move(
            int(pid), weakest, src, fraction, KIND_REINFORCE_ALLY,
        ))
        has_launch = True
    if not has_launch:
        return None
    return JointAction(moves=tuple(moves))


def macro_retreat_to_largest_ally(
    po: ParsedObs, per_planet: Dict[int, List[PlanetMove]],
    *, fraction: float = 0.5, min_launch: int = _DEFAULT_MIN_LAUNCH,
) -> Optional[JointAction]:
    """Every non-largest friendly planet sends ~50% of its garrison to
    the largest friendly. Captures the "consolidate before getting
    picked off" pattern when the heuristic is over-aggressive.

    Returns None if we have fewer than 2 friendly planets.
    """
    if len(po.my_planets) < 2:
        return None
    largest = max(po.my_planets, key=lambda p: int(p[5]))
    largest_pid = int(largest[0])
    moves: List[PlanetMove] = []
    has_launch = False
    for pid in sorted(per_planet.keys()):
        src = po.planet_by_id.get(int(pid))
        if src is None or int(pid) == largest_pid:
            moves.append(_hold_move(int(pid)))
            continue
        avail = int(src[5])
        if avail < min_launch:
            moves.append(_hold_move(int(pid)))
            continue
        moves.append(_launch_move(
            int(pid), largest, src, fraction, KIND_REINFORCE_ALLY,
        ))
        has_launch = True
    if not has_launch:
        return None
    return JointAction(moves=tuple(moves))


# Default ordering used by build_macro_anchors. Order matters only for
# de-duplication (earlier macros win when two produce the same wire).
DEFAULT_MACROS = (
    macro_hold_all,
    macro_mass_attack_nearest_enemy,
    macro_reinforce_weakest_ally,
    macro_retreat_to_largest_ally,
)


def build_macro_anchors(
    po: ParsedObs, per_planet: Dict[int, List[PlanetMove]],
    macros: Optional[tuple] = None,
) -> List[JointAction]:
    """Return a list of macro joint actions to use as additional root
    candidates (in the same shape as the heuristic anchor).

    Errors in any individual macro are silently swallowed — a macro
    that throws is just dropped from the candidate list. Macros that
    return None (e.g. ``mass_attack_nearest_enemy`` with no enemies)
    are also dropped. The returned list is de-duplicated by wire.
    """
    if macros is None:
        macros = DEFAULT_MACROS
    seen_wires: set = set()
    out: List[JointAction] = []
    for fn in macros:
        try:
            j = fn(po, per_planet)
        except Exception:
            continue
        if j is None:
            continue
        try:
            wire_key = tuple(tuple(m) for m in j.to_wire())
        except Exception:
            continue
        if wire_key in seen_wires:
            continue
        seen_wires.add(wire_key)
        out.append(j)
    return out
