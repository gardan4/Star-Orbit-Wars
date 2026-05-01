"""Multi-seed heuristic-vs-heuristic regression check.

Caches in heuristic.py / intercept.py must be lossless. Run several
seeds; outcomes must match the previously-observed baseline values
recorded inline below. If any seed diverges, a cache change has
introduced a behavior shift.
"""
from __future__ import annotations

import time

from orbitwars.bots.heuristic import HeuristicAgent
from tournaments.harness import play_game


# Baseline outcomes recorded BEFORE the cache changes at the same seeds.
# Re-run after any cache-related edit; values must match.
BASELINE = {
    42:  {"rewards": [1, -1],  "scores": [1880, 1834], "steps": 499},
    123: None,  # to be observed
    7:   None,  # to be observed
}


def main() -> None:
    for seed in (42, 123, 7):
        h1 = HeuristicAgent()
        h2 = HeuristicAgent()
        t0 = time.perf_counter()
        result = play_game(
            [h1.as_kaggle_agent(), h2.as_kaggle_agent()],
            seed=seed,
            players=2,
            step_timeout=2.0,
        )
        wall = time.perf_counter() - t0
        line = (f"seed={seed}: rewards={list(result.rewards)} "
                f"scores={list(result.final_scores)} steps={result.steps} "
                f"wall={wall:.1f}s")
        baseline = BASELINE.get(seed)
        if baseline is not None:
            ok = (list(result.rewards) == baseline["rewards"]
                  and list(result.final_scores) == baseline["scores"]
                  and result.steps == baseline["steps"])
            line += f"  match_baseline={ok}"
        print(line)


if __name__ == "__main__":
    main()
