"""Seat-1 variant of diag_mcts_vs_heur_actions.

The seat-0 variant confirmed 0/30 divergence under a generous 60 s budget.
But the multi-seed smoke shows seat-1 MCTS losing 0/3 and some games are
catastrophic blowouts (mcts=0 vs heur=9345). That rules out timing
variance alone — something structural is different at seat 1.

Hypothesis: the step-inference path added for seat 1 (obs.step is None
for seat 1 in the Kaggle engine) introduces a mismatch between what
MCTSAgent's fallback sees and what a fresh standalone HeuristicAgent
would see on the same observation, breaking anchor-lock silently.

If this diagnostic shows >0 divergences at seat 1 while seat 0 showed
zero, the step-override wiring is the culprit.
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

    # MCTS at SEAT 1 this time. Heuristic drives seat 0.
    mcts = MCTSAgent(rng_seed=0)
    heur_shadow = HeuristicAgent()
    k_mcts = mcts.as_kaggle_agent()
    k_heur_p0 = HeuristicAgent().as_kaggle_agent()  # drives P0 in env
    k_heur_shadow = heur_shadow.as_kaggle_agent()   # shadow of MCTS at P1

    divergences = 0
    total = 0
    for step in range(30):
        if env.state[0]["status"] != "ACTIVE":
            break
        obs0 = env.state[0]["observation"]
        obs1 = env.state[1]["observation"]

        # Both MCTS and the shadow heuristic see seat-1's obs (obs1).
        a_mcts = k_mcts(obs1, env.configuration)
        a_heur = k_heur_shadow(obs1, env.configuration)
        total += 1

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
            mcts_fb_tc = getattr(mcts._fallback, "_turn_counter", "?")
            heur_tc = getattr(heur_shadow, "_turn_counter", "?")
            obs_step = obs1.get("step", None) if hasattr(obs1, "get") else getattr(obs1, "step", None)
            print(f"step={step}: DIVERGE mcts-only={len(extra)} heur-only={len(missing)}")
            print(f"  mcts._fallback._turn_counter={mcts_fb_tc}  "
                  f"heur_shadow._turn_counter={heur_tc}  obs.step={obs_step}")
            print(f"  mcts: {n_m}")
            print(f"  heur: {n_h}")

        # Advance: heuristic at P0, MCTS's output at P1.
        a_opp = k_heur_p0(obs0, env.configuration)
        env.step([a_opp, a_mcts])

    print(f"\n{divergences}/{total} turns diverged at SEAT 1")


if __name__ == "__main__":
    main()
