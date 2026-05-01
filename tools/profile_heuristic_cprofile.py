"""cProfile heuristic.act() inside a realistic rollout.

Builds a turn-200 state via heuristic warmup, then profiles N rollout
plies. Output identifies the hottest functions by cumulative time so
we know exactly which path to optimize.
"""
from __future__ import annotations

import cProfile
import copy
import gc
import pstats
import random
from io import StringIO

from kaggle_environments import make

from orbitwars.bots.base import Deadline
from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.engine.fast_engine import FastEngine


def main() -> None:
    random.seed(42)
    env = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
    env.reset(num_agents=2)
    env.step([[], []])
    heur_warmup = HeuristicAgent().as_kaggle_agent()
    for _ in range(200):
        obs0 = env.state[0].observation
        obs1 = env.state[1].observation
        a0 = heur_warmup(obs0)
        a1 = heur_warmup(obs1)
        env.step([a0, a1])
        if env.done:
            break

    base_eng = FastEngine.from_official_obs(env.state[0].observation, num_agents=2)
    base_state = base_eng.state

    N_ROLLOUTS = 30
    DEPTH = 15

    # Warmup
    for _ in range(3):
        st = copy.deepcopy(base_state)
        eng = FastEngine(st, num_agents=2, rng=random.Random())
        for _ in range(DEPTH):
            if eng.done:
                break
            actions = []
            for i in range(2):
                ob = eng.observation(i)
                actions.append(HeuristicAgent().act(ob, Deadline()))
            eng.step(actions)

    pr = cProfile.Profile()
    gc.collect()
    gc.disable()
    pr.enable()
    try:
        for _ in range(N_ROLLOUTS):
            st = copy.deepcopy(base_state)
            eng = FastEngine(st, num_agents=2, rng=random.Random())
            for _ in range(DEPTH):
                if eng.done:
                    break
                actions = []
                for i in range(2):
                    ob = eng.observation(i)
                    actions.append(HeuristicAgent().act(ob, Deadline()))
                eng.step(actions)
    finally:
        pr.disable()
        gc.enable()

    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(40)
    print(s.getvalue())

    print("\n=== Sorted by tottime (self-time) ===\n")
    s2 = StringIO()
    pstats.Stats(pr, stream=s2).sort_stats("tottime").print_stats(40)
    print(s2.getvalue())


if __name__ == "__main__":
    main()
