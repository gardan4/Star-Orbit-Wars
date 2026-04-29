"""A/B test the neutral-exact-plus-one heuristic fix vs legacy.

Both sides use the same HeuristicAgent class — the only difference is
``legacy_neutral_floor`` weight (0 = fix enabled, 1 = pre-fix behavior).
Plays paired games on identical seeds, reports W-L-T and Elo delta.

Why this is needed: spectator observation (2026-04-26) noted that the
heuristic over-sends ships to neutral captures (sending 30 to a 5-ship
neutral wastes 25 ships per capture, killing the early-game snowball).
The fix in heuristic.py differentiates neutral targets (exact-plus-one
sizing) from enemy targets (keep min_launch_size floor). This script
quantifies the effect.

Run:
    python -m tools.compare_neutral_fix --games 30 --seed 100 --workers 7
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import random as _r
import time
from pathlib import Path
from typing import List, Tuple

from kaggle_environments import make

from orbitwars.bots.heuristic import HEURISTIC_WEIGHTS, HeuristicAgent


def _play_one(seed: int, swap: bool) -> Tuple[int, int, int]:
    """Return (a_reward, b_reward, steps).
    A = fixed heuristic (race+engage+counter+speed minimum-viable),
    B = legacy heuristic (static min_launch_size floor).
    If swap, B plays seat 0."""
    _r.seed(seed)
    fixed_w = dict(HEURISTIC_WEIGHTS)
    legacy_w = dict(HEURISTIC_WEIGHTS)
    # Full legacy: BOTH neutral and enemy use static min_launch_size floor.
    legacy_w["legacy_neutral_floor"] = 1.0
    legacy_w["legacy_enemy_floor"] = 1.0
    fixed_agent = HeuristicAgent(weights=fixed_w).as_kaggle_agent()
    legacy_agent = HeuristicAgent(weights=legacy_w).as_kaggle_agent()
    if swap:
        agents = [legacy_agent, fixed_agent]
    else:
        agents = [fixed_agent, legacy_agent]
    env = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
    env.run(agents)
    rewards = [int(s.reward if s.reward is not None else 0) for s in env.state]
    steps = int(env.state[0].observation.step)
    if swap:
        a_r = rewards[1]
        b_r = rewards[0]
    else:
        a_r = rewards[0]
        b_r = rewards[1]
    return a_r, b_r, steps


def _worker(args):
    seed, swap = args
    return _play_one(seed, swap)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30,
                    help="number of seeds (each plays 2 mirror matches)")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    tasks: List[Tuple[int, bool]] = []
    for g in range(args.games):
        s = args.seed + 1000 * g
        tasks.append((s, False))  # fixed=seat0
        tasks.append((s, True))   # fixed=seat1

    print(f"A/B fix vs legacy: {args.games} seeds × 2 mirrors = {len(tasks)} games  workers={args.workers}", flush=True)
    t0 = time.perf_counter()
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            results = pool.map(_worker, tasks)
    else:
        results = [_worker(t) for t in tasks]
    wall = time.perf_counter() - t0

    a_w = a_l = a_t = 0
    for (a_r, b_r, steps), (seed, swap) in zip(results, tasks):
        if a_r == 1:
            a_w += 1
            outcome = "WIN"
        elif a_r == -1:
            a_l += 1
            outcome = "LOSS"
        else:
            a_t += 1
            outcome = "TIE"
        seat = "seat0" if not swap else "seat1"
        print(f"  s{seed:6d} fixed={seat} steps={steps:3d} {outcome}", flush=True)

    total = a_w + a_l + a_t
    a_score = a_w + 0.5 * a_t
    a_wr = a_score / max(total, 1)
    se = math.sqrt(a_wr * (1 - a_wr) / max(total, 1))
    if 0.0 < a_wr < 1.0:
        elo = -400.0 * math.log10(1.0 / a_wr - 1.0)
    elif a_wr == 0.0:
        elo = -800.0
    else:
        elo = 800.0
    print(f"\n=== Summary ({total} games, {wall:.0f}s) ===", flush=True)
    print(f"fixed vs legacy:  W-L-T = {a_w}-{a_l}-{a_t}  wr={a_wr:.3f} ± {se:.3f}", flush=True)
    print(f"Elo delta (fixed - legacy): {elo:+.1f}", flush=True)


if __name__ == "__main__":
    main()
