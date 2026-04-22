"""Micro-profile a single MCTS rollout to localize where the time goes.

Answers: in the 300-ms MCTS budget, what fraction is engine.step vs
heuristic.act vs observation() vs deepcopy? This determines whether
engine optimization is high- or low-leverage.

Output: per-component mean time over N rollouts at rollout_depth=15.
"""
from __future__ import annotations

import copy
import gc
import random
import time
from statistics import mean, median

from kaggle_environments import make

from orbitwars.bots.fast_rollout import FastRolloutAgent
from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.bots.base import Deadline
from orbitwars.engine.fast_engine import FastEngine


def main() -> None:
    # Build a realistic mid-game state (turn ~200) — hot path for MCTS.
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

    # Seed a FastEngine from this mid-game obs. This is the state MCTS
    # would clone at rollout time.
    base_eng = FastEngine.from_official_obs(env.state[0].observation, num_agents=2)
    base_state = base_eng.state

    # Warmup imports / JIT.
    for _ in range(5):
        _ = copy.deepcopy(base_state)

    N = 30  # 30 rollouts × 15 plies = 900 plies/policy. ~10s total.
    DEPTH = 15

    def run(agent_factory, label):
        """Call agent.act(obs, Deadline()) directly — that's the path
        MCTS rollouts actually use (see gumbel_search._rollout_value).
        `as_kaggle_agent()` wraps every call with a gc.collect() which
        is right for the root turn (cheap over 1s budget) but adds ~14ms
        per call inside rollouts, drowning out real act() cost.
        """
        t_deepcopy = []
        t_obs = []
        t_act = []
        t_step = []
        gc.collect()
        gc.disable()
        try:
            for _ in range(N):
                t0 = time.perf_counter()
                st = copy.deepcopy(base_state)
                eng = FastEngine(st, num_agents=2, rng=random.Random())
                t_deepcopy.append((time.perf_counter() - t0) * 1000.0)

                for _ in range(DEPTH):
                    if eng.done:
                        break
                    actions = [None, None]
                    for i in range(2):
                        t0 = time.perf_counter()
                        ob = eng.observation(i)
                        t_obs.append((time.perf_counter() - t0) * 1e6)

                        agent = agent_factory()
                        t0 = time.perf_counter()
                        actions[i] = agent.act(ob, Deadline())
                        t_act.append((time.perf_counter() - t0) * 1e6)

                    t0 = time.perf_counter()
                    eng.step(actions)
                    t_step.append((time.perf_counter() - t0) * 1e6)
        finally:
            gc.enable()
        return {
            "label": label,
            "deepcopy_ms": t_deepcopy,
            "obs_us": t_obs,
            "act_us": t_act,
            "step_us": t_step,
        }

    heur = run(lambda: HeuristicAgent(), "HeuristicAgent (default)")
    fast = run(lambda: FastRolloutAgent(), "FastRolloutAgent")
    t_deepcopy = heur["deepcopy_ms"]
    t_obs = heur["obs_us"]
    t_act = heur["act_us"]
    t_step = heur["step_us"]

    def summary(label, vals_us, per_rollout_factor):
        total_us_per_rollout = mean(vals_us) * per_rollout_factor
        print(
            f"  {label:<15} mean={mean(vals_us):7.1f}us  "
            f"median={median(vals_us):7.1f}us  "
            f"total/rollout={total_us_per_rollout/1000:6.2f}ms "
            f"(calls/rollout={per_rollout_factor})"
        )

    def report(bucket):
        dc = bucket["deepcopy_ms"]
        ob = bucket["obs_us"]
        ac = bucket["act_us"]
        st = bucket["step_us"]
        print(f"\n--- {bucket['label']}: N={N}, depth={DEPTH} ---")
        print(
            f"  deepcopy+init   mean={mean(dc):7.3f}ms  "
            f"median={median(dc):7.3f}ms  (once per rollout)"
        )
        summary("observation()", ob, DEPTH * 2)
        summary("agent.act()", ac, DEPTH * 2)
        summary("engine.step()", st, DEPTH)

        total_ms = (
            mean(dc)
            + mean(ob) * DEPTH * 2 / 1000
            + mean(ac) * DEPTH * 2 / 1000
            + mean(st) * DEPTH / 1000
        )
        print(f"\n  Sum of means: {total_ms:.2f} ms per rollout")
        print(f"  => ~{max(0, int(300 / total_ms))} rollouts in a 300-ms budget")
        return total_ms

    heur_total = report(heur)
    fast_total = report(fast)

    print()
    print("==================================================")
    print("                 Head-to-head")
    print("==================================================")
    print(f"  Heuristic rollout : {heur_total:7.1f} ms")
    print(f"  Fast rollout      : {fast_total:7.1f} ms")
    if fast_total > 0:
        print(f"  Speedup           : {heur_total/fast_total:6.1f}x")
    print(f"  Sims @ 300 ms budget: heuristic={max(0, int(300/heur_total))} "
          f"fast={max(0, int(300/fast_total))}")


if __name__ == "__main__":
    main()
