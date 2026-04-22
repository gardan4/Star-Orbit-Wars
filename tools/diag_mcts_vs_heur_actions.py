"""Diagnostic: does MCTSAgent with margin=2.0 actually produce the same
wire actions as a standalone HeuristicAgent on a live game?

If margin=2.0 is truly anchor-locked, their actions should match
turn-by-turn. Any divergence is a bug in the anchor floor.
"""
from __future__ import annotations

import random

from kaggle_environments import make

from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.bots.mcts_bot import MCTSAgent


def main() -> None:
    random.seed(42)
    env = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
    env.reset(num_agents=2)

    mcts = MCTSAgent(rng_seed=0)
    heur_shadow = HeuristicAgent()
    k_mcts = mcts.as_kaggle_agent()
    k_heur_for_opp = HeuristicAgent().as_kaggle_agent()
    k_heur_shadow = heur_shadow.as_kaggle_agent()

    divergences = 0
    total = 0
    for step in range(30):
        if env.state[0]["status"] != "ACTIVE":
            break
        obs0 = env.state[0]["observation"]
        obs1 = env.state[1]["observation"]
        a_mcts = k_mcts(obs0, env.configuration)
        a_heur = k_heur_shadow(obs0, env.configuration)
        total += 1
        # Canonicalize for order-insensitive comparison.
        def norm(a):
            out = []
            for m in a or []:
                if len(m) == 3:
                    out.append((int(m[0]), round(float(m[1]), 4), int(m[2])))
            return sorted(out)
        n_m = norm(a_mcts)
        n_h = norm(a_heur)
        if n_m != n_h:
            divergences += 1
            extra = set(n_m) - set(n_h)
            missing = set(n_h) - set(n_m)
            print(f"step={step}: DIVERGE mcts+{len(extra)} heur+{len(missing)}")
            print(f"  mcts: {n_m}")
            print(f"  heur: {n_h}")
        # Advance with MCTS action for P0, heuristic-generic for P1
        a_opp = k_heur_for_opp(obs1, env.configuration)
        env.step([a_mcts, a_opp])
    print(f"\n{divergences}/{total} turns diverged")


if __name__ == "__main__":
    main()
