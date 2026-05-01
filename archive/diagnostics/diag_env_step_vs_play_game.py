"""Isolate whether the manual env.step loop differs from play_game.

Baseline diag_heur_vs_heur_by_seed runs via tournaments.harness.play_game
(which calls env.run). Our full-game MCTS diag uses manual env.step loops.
If the outcomes DIFFER for the same seed with identical agents, the
harness vs manual-step is the variable — not MCTSAgent itself.
"""
from __future__ import annotations

import random
import time

from kaggle_environments import make

from orbitwars.bots.heuristic import HeuristicAgent


def run_manual(seed: int) -> tuple:
    random.seed(seed)
    env = make("orbit_wars", configuration={"actTimeout": 2.0}, debug=False)
    env.reset(num_agents=2)
    h0 = HeuristicAgent()
    h1 = HeuristicAgent()
    k0 = h0.as_kaggle_agent()
    k1 = h1.as_kaggle_agent()
    for step in range(500):
        if env.state[0]["status"] != "ACTIVE":
            break
        obs0 = env.state[0]["observation"]
        obs1 = env.state[1]["observation"]
        a0 = k0(obs0, env.configuration)
        a1 = k1(obs1, env.configuration)
        env.step([a0, a1])
    return tuple(env.state[i]["reward"] for i in range(2)), step


def run_env_run(seed: int) -> tuple:
    random.seed(seed)
    env = make("orbit_wars", configuration={"actTimeout": 2.0}, debug=False)
    h0 = HeuristicAgent()
    h1 = HeuristicAgent()
    env.run([h0.as_kaggle_agent(), h1.as_kaggle_agent()])
    return tuple(env.state[i]["reward"] for i in range(2)), int(env.state[0]["observation"]["step"])


def main() -> None:
    for s in [42, 123, 7]:
        r_manual, step_m = run_manual(s)
        r_run, step_r = run_env_run(s)
        match = "MATCH" if r_manual == r_run else "MISMATCH"
        print(f"seed={s}: manual={r_manual} steps={step_m}  |  "
              f"env.run={r_run} steps={step_r}  {match}", flush=True)


if __name__ == "__main__":
    main()
