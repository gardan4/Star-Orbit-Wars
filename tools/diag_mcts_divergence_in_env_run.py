"""Track divergence between MCTS and shadow heur from *inside* env.run.

The manual-env.step diagnostic produced a different game trajectory
than env.run (confirmed via diag_env_step_vs_play_game.py at seed=123).
This version does the comparison inside a wrapped agent that's given
to env.run, so the trajectory matches what play_game sees in the
multi-seed smoke.

Design: a single wrapped agent plays MCTSAgent at seat 1 and ALSO
instantiates a shadow HeuristicAgent that sees the same obs. On each
call, compare MCTS's wire action against the shadow's. Any divergence
is logged.
"""
from __future__ import annotations

import random
import time
from typing import List

from kaggle_environments import make

from orbitwars.bots.base import Deadline
from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.bots.mcts_bot import MCTSAgent


def norm(a) -> list:
    out = []
    for m in a or []:
        if len(m) == 3:
            out.append((int(m[0]), round(float(m[1]), 4), int(m[2])))
    return sorted(out)


def main() -> None:
    for seed in [42, 123, 7]:
        random.seed(seed)
        env = make("orbit_wars", configuration={"actTimeout": 2.0}, debug=False)

        mcts = MCTSAgent(rng_seed=0)
        shadow = HeuristicAgent()
        heur_p0 = HeuristicAgent()

        k_mcts = mcts.as_kaggle_agent()
        k_shadow = shadow.as_kaggle_agent()
        k_heur_p0 = heur_p0.as_kaggle_agent()

        divs: List[tuple] = []
        total = [0]

        def wrapped_p1(obs, cfg):
            # MCTS wire action:
            a_mcts = k_mcts(obs, cfg)
            # Shadow heur action on the SAME obs:
            a_shadow = k_shadow(obs, cfg)
            total[0] += 1
            n_m = norm(a_mcts)
            n_s = norm(a_shadow)
            if n_m != n_s:
                divs.append((total[0] - 1, n_m, n_s))
            return a_mcts  # drive the game with MCTS

        t0 = time.perf_counter()
        env.run([k_heur_p0, wrapped_p1])
        wall = time.perf_counter() - t0

        rewards = [int(env.state[i]["reward"] if env.state[i]["reward"] is not None else 0) for i in range(2)]
        print(f"seed={seed}: {len(divs)}/{total[0]} diverged  "
              f"rewards={rewards}  wall={wall:.0f}s", flush=True)
        for turn, a_m, a_s in divs[:5]:
            extra = set(a_m) - set(a_s)
            missing = set(a_s) - set(a_m)
            print(f"  turn={turn}: mcts-only={len(extra)} heur-only={len(missing)}", flush=True)
            print(f"    mcts: {a_m[:3]}", flush=True)
            print(f"    heur: {a_s[:3]}", flush=True)


if __name__ == "__main__":
    main()
