"""Micro-profile `ArchetypePosterior.observe()` cost per turn.

Hypothesis under test: in the exploitation smokes, `use_model=True`
ran with ~3 ms/turn extra on HeuristicAgent-observation + archetype
simulation overhead. If that overhead is actually 30-100 ms, it
cannibalizes search budget (300 ms deadline) by 10-30% — which would
explain the weak-negative Run C finding without needing the "archetype
override is the wrong integration" story.

Output: per-call mean/median/p95/p99 over a realistic mid-game obs.

What counts as "expensive": >20 ms would be clear evidence the
observation path is non-trivial against the 300 ms budget. <5 ms
rules out overhead as the mechanism for Run C's −2709-unit drift.
"""
from __future__ import annotations

import random
import time
from statistics import fmean, median

from kaggle_environments import make

from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.opponent.bayes import ArchetypePosterior


def _quantile(xs, q):
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return xs[k]


def main() -> None:
    # Build a mid-game state so the posterior observes a realistic
    # fleet/planet density. Turn-0 obs is degenerate (no launches yet
    # → posterior does almost no work).
    random.seed(42)
    env = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
    env.reset(num_agents=2)
    env.step([[], []])
    warmup = HeuristicAgent().as_kaggle_agent()
    for _ in range(150):
        a0 = warmup(env.state[0].observation)
        a1 = warmup(env.state[1].observation)
        env.step([a0, a1])
        if env.done:
            break

    # Snapshot two consecutive mid-game obs so observe() has a real
    # "prev_obs → curr_obs" transition to score. A single-obs call is
    # a no-op (no launches diff).
    prev_obs = env.state[0].observation
    env.step([
        warmup(env.state[0].observation),
        warmup(env.state[1].observation),
    ])
    curr_obs = env.state[0].observation

    post = ArchetypePosterior()

    # Warmup: first call pays any lazy-import / JIT cost.
    post.observe(prev_obs, opp_player=1)
    post.observe(curr_obs, opp_player=1)

    N = 50
    times_us = []
    for _ in range(N):
        # Re-observe the same transition — cost is deterministic-enough
        # for profiling; this is NOT a correctness measurement.
        t0 = time.perf_counter()
        post.observe(curr_obs, opp_player=1)
        times_us.append((time.perf_counter() - t0) * 1e6)

    mean_us = fmean(times_us)
    med_us = median(times_us)
    p95 = _quantile(times_us, 0.95)
    p99 = _quantile(times_us, 0.99)
    mx = max(times_us)

    print(f"ArchetypePosterior.observe() over N={N}, mid-game obs (turn ~150)")
    print(f"  mean   = {mean_us/1000:7.2f} ms ({mean_us:.1f} us)")
    print(f"  median = {med_us/1000:7.2f} ms ({med_us:.1f} us)")
    print(f"  p95    = {p95/1000:7.2f} ms")
    print(f"  p99    = {p99/1000:7.2f} ms")
    print(f"  max    = {mx/1000:7.2f} ms")
    print()

    # Express as fraction of the MCTS 300 ms deadline.
    pct_budget = mean_us / 1000.0 / 300.0 * 100.0
    print(
        f"Fraction of 300 ms MCTS deadline: mean {pct_budget:.1f}% "
        f"(median {med_us/1000/300*100:.1f}%, p95 {p95/1000/300*100:.1f}%)"
    )
    if mean_us / 1000.0 > 20.0:
        print(
            "  => STRONG evidence for 'observation overhead cannibalizes "
            "search budget' mechanism (>20 ms mean)."
        )
    elif mean_us / 1000.0 > 5.0:
        print(
            "  => WEAK evidence for overhead mechanism "
            "(5-20 ms; non-trivial but absorbable)."
        )
    else:
        print(
            "  => RULES OUT overhead as primary mechanism for Run C drift "
            "(<5 ms, <1.7% of budget)."
        )


if __name__ == "__main__":
    main()
