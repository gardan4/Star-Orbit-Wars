"""Single-turn trace: does MCTSAgent produce the SAME wire as HeuristicAgent
when its search can't confidently find a better move?

Exercises the anchor+margin-guard path on a tiny synthetic obs. Prints:
  * Heuristic's wire output for this obs.
  * MCTSAgent's wire output for the same obs.
  * SearchResult q_values + visits.

If anchor guard is functioning, MCTS wire should equal heuristic wire
unless SH finds something confidently better (rare on tiny obs).
"""
from __future__ import annotations

import math

from orbitwars.bots.base import Deadline
from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.bots.mcts_bot import MCTSAgent
from orbitwars.mcts.gumbel_search import GumbelConfig


def _mk_obs(step: int, with_fleets: bool = False):
    obs = {
        "player": 0,
        "step": step,
        "angular_velocity": 0.03,
        "planets": [
            [0, 0, 20.0, 50.0, 1.5, 80, 3],
            [1, 1, 80.0, 50.0, 1.5, 60, 3],
            [2, -1, 50.0, 20.0, 1.0, 20, 1],
            [3, -1, 50.0, 80.0, 1.0, 20, 1],
        ],
        "initial_planets": [
            [0, 0, 20.0, 50.0, 1.5, 10, 3],
            [1, 1, 80.0, 50.0, 1.5, 10, 3],
            [2, -1, 50.0, 20.0, 1.0, 10, 1],
            [3, -1, 50.0, 80.0, 1.0, 10, 1],
        ],
        "fleets": [],
        "next_fleet_id": 0,
        "comet_planet_ids": [],
        "comets": [],
    }
    if with_fleets:
        # Add a realistic in-flight fleet mid-game.
        obs["fleets"] = [
            # [fleet_id, owner, x, y, angle, from_planet_id, ships]
            [0, 0, 40.0, 50.0, 0.0, 0, 25],  # my fleet
            [1, 1, 65.0, 52.0, math.pi, 1, 30],  # enemy fleet
        ]
        obs["next_fleet_id"] = 2
    return obs


def main() -> None:
    # Force anchor-only so MCTS ALWAYS returns heuristic's pick.
    # If any "same? False" shows up, the anchor-reconstruction is broken.
    for with_fleets in (False, True):
        print(f"\n===== with_fleets={with_fleets} =====")
        h = HeuristicAgent()
        m = MCTSAgent(
            gumbel_cfg=GumbelConfig(
                num_candidates=2, total_sims=2, rollout_depth=1,
                hard_deadline_ms=50.0,
                anchor_improvement_margin=10.0,  # force anchor
            ),
            rng_seed=0,
        )
        for step in (0, 10, 20, 30, 50):
            obs = _mk_obs(step, with_fleets=with_fleets)
            h_wire = h.act(obs, Deadline())
            m_wire = m.act(obs, Deadline())
            print(f"--- step={step} ---")
            print(f"  HEUR: {h_wire}")
            print(f"  MCTS: {m_wire}")
            print(f"  same? {sorted(h_wire) == sorted(m_wire)}")


if __name__ == "__main__":
    main()
