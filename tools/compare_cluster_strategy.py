"""A/B test cluster-aware target selection vs legacy nearest-target.

The user spectator-observed (2026-04-26) that the top ladder player
opens by sending exact-plus-one fleets to a DISTANT cluster of
neutrals (not the nearest), then expands within that cluster, then
takes nearby planets — playing for cluster control rather than
nearest-target greedy.

This tool tests whether cluster-aware target selection
(cluster_strategy_weight > 0) beats legacy greedy nearest-target
selection in heuristic-vs-heuristic play.

Run:
    python -m tools.compare_cluster_strategy --games 30 --workers 7
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import random as _r
import time
from typing import List, Tuple

from kaggle_environments import make

from orbitwars.bots.heuristic import HEURISTIC_WEIGHTS, HeuristicAgent


def _play_one(seed: int, swap: bool) -> Tuple[int, int, int]:
    """A = cluster-aware (cluster_strategy_weight=1.0).
    B = legacy (cluster_strategy_weight=0.0, default)."""
    _r.seed(seed)
    cluster_w = dict(HEURISTIC_WEIGHTS)
    cluster_w["cluster_strategy_weight"] = 1.0
    legacy_w = dict(HEURISTIC_WEIGHTS)  # default cluster_strategy_weight=0
    cluster_agent = HeuristicAgent(weights=cluster_w).as_kaggle_agent()
    legacy_agent = HeuristicAgent(weights=legacy_w).as_kaggle_agent()
    if swap:
        agents = [legacy_agent, cluster_agent]
    else:
        agents = [cluster_agent, legacy_agent]
    env = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
    env.run(agents)
    rewards = [int(s.reward if s.reward is not None else 0) for s in env.state]
    steps = int(env.state[0].observation.step)
    if swap:
        a_r = rewards[1]
    else:
        a_r = rewards[0]
    return a_r, 0, steps


def _worker(args):
    return _play_one(*args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cluster-weight", type=float, default=1.0,
                    help="cluster_strategy_weight for the cluster-aware side")
    args = ap.parse_args()

    tasks = []
    for g in range(args.games):
        s = args.seed + 1000 * g
        tasks.append((s, False))
        tasks.append((s, True))

    print(f"Cluster-aware vs legacy: {args.games} seeds × 2 mirrors = {len(tasks)} games", flush=True)
    t0 = time.perf_counter()
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            results = pool.map(_worker, tasks)
    else:
        results = [_worker(t) for t in tasks]
    wall = time.perf_counter() - t0

    a_w = a_l = a_t = 0
    for (a_r, _b, steps), (seed, swap) in zip(results, tasks):
        if a_r == 1: a_w += 1; outcome = "WIN"
        elif a_r == -1: a_l += 1; outcome = "LOSS"
        else: a_t += 1; outcome = "TIE"
        seat = "seat0" if not swap else "seat1"
        print(f"  s{seed:6d} cluster={seat} steps={steps:3d} {outcome}", flush=True)

    total = a_w + a_l + a_t
    a_score = a_w + 0.5 * a_t
    a_wr = a_score / max(total, 1)
    se = math.sqrt(a_wr * (1 - a_wr) / max(total, 1))
    if 0.0 < a_wr < 1.0:
        elo = -400.0 * math.log10(1.0 / a_wr - 1.0)
    else:
        elo = -800.0 if a_wr == 0.0 else 800.0
    print(f"\n=== Summary ({total} games, {wall:.0f}s) ===", flush=True)
    print(f"cluster vs legacy:  W-L-T = {a_w}-{a_l}-{a_t}  wr={a_wr:.3f} ± {se:.3f}", flush=True)
    print(f"Elo delta: {elo:+.1f}", flush=True)


if __name__ == "__main__":
    main()
