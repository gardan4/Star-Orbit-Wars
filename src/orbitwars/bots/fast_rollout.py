"""Ultra-cheap rollout policy for MCTS.

The `HeuristicAgent` takes ~18 ms per `act()` call — acceptable for the
one root decision it makes per real turn, but catastrophic inside an
MCTS rollout. At `rollout_depth=15` with 2 players, one rollout is
~560 ms — we can't finish a single rollout inside the 300 ms search
budget.

This file provides `FastRolloutAgent`, which mirrors AlphaGo's
"fast rollout policy" split: the slow/accurate policy drives the tree
and the root decision, and a cheap policy fills in rollout plies. The
fast policy intentionally skips every expensive subroutine:

  * No arrival-table build.
  * No scoring sweep over targets.
  * No Newton intercept for orbiting targets (uses static-intercept,
    which is just an `atan2`).
  * No sun-tangent routing — we accept the rollout-noise cost of the
    occasional fleet flying into the sun. Every candidate gets the
    same bias, so ranking at the root is preserved.
  * No cooldowns, no defense guards, no launch-state tracking.

The one rule: from each of my planets with enough ships, send
`send_fraction × available` at `atan2(nearest_enemy_or_neutral)`.

Expected cost per `act()` call: <500 µs — a 30-50× speedup over the
full heuristic. Net effect at default budget:
    sims/turn goes from <1 (diagnostic only) to ~10-15 (real search).

Invariants preserved:
  * Only my planets launch.
  * `ships <= planet.ships` always.
  * Angle is finite (atan2 well-defined when source != target; the
    guard below rejects self-targets).

This class is used inside MCTS rollouts when
`GumbelConfig.rollout_policy == "fast"`. The root anchor is still
provided by `HeuristicAgent`; only rollout plies swap in the fast
agent.
"""
from __future__ import annotations

import math
from typing import Any, List, Optional

from orbitwars.bots.base import Action, Agent, Deadline, Move, no_op, obs_get


class FastRolloutAgent(Agent):
    """Cheapest-possible rollout policy: nearest-target static push.

    Knobs intentionally minimal — tuning this is not the point. If
    rollouts need to be smarter, promote to a real heuristic; if they
    need to be faster, inline the loop.

    Attributes:
        min_launch_size: do not launch a fleet smaller than this many
            ships (matches HeuristicAgent's default). Prevents single-
            ship dribbles that clutter the fleet count without changing
            the value.
        send_fraction: fraction of available ships to send from a
            launching planet. 0.8 leaves a 20% reserve and matches the
            HeuristicAgent default, so fast and slow rollouts produce
            comparably-sized fleets — only the target-selection logic
            differs.
    """

    name = "fast_rollout"

    def __init__(
        self,
        min_launch_size: int = 20,
        send_fraction: float = 0.8,
    ) -> None:
        self.min_launch_size = int(min_launch_size)
        self.send_fraction = float(send_fraction)

    def act(self, obs: Any, deadline: Deadline) -> Action:
        # Always stage a safe fallback first; if we get interrupted
        # mid-loop the caller still has a valid action.
        deadline.stage(no_op())

        player = obs_get(obs, "player", 0)
        planets = obs_get(obs, "planets", [])
        if not planets:
            return no_op()

        # Single-pass ownership partition. Both lists hold references
        # into the same planet entries, so we avoid copying.
        my_planets: List[Any] = []
        targets: List[Any] = []
        for p in planets:
            if p[1] == player:
                my_planets.append(p)
            else:
                targets.append(p)

        # Either no launchers or no opponents/neutrals to push toward:
        # there is nothing useful to do.
        if not my_planets or not targets:
            return no_op()

        moves: Action = []
        min_size = self.min_launch_size
        frac = self.send_fraction

        for mp in my_planets:
            available = int(mp[5])
            if available < min_size:
                continue

            # Find nearest target by squared-Euclidean — no sqrt needed
            # when we only need the argmin. This is the hot inner loop.
            mx = float(mp[2])
            my_ = float(mp[3])
            best_d2 = float("inf")
            best_tp: Optional[Any] = None
            for tp in targets:
                dx = float(tp[2]) - mx
                dy = float(tp[3]) - my_
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_tp = tp

            if best_tp is None or best_d2 == 0.0:
                # Degenerate: co-located target (shouldn't happen in
                # valid states) or no targets at all.
                continue

            # Static intercept angle — just atan2. We deliberately
            # ignore orbital motion: in 15-ply rollouts the bias is
            # small, and every candidate experiences the same bias,
            # so simple-regret ranking at the root is preserved.
            angle = math.atan2(
                float(best_tp[3]) - my_,
                float(best_tp[2]) - mx,
            )

            ships = int(available * frac)
            if ships < min_size:
                ships = min_size
            if ships > available:
                ships = available

            moves.append([int(mp[0]), float(angle), int(ships)])

        deadline.stage(moves)
        return moves
