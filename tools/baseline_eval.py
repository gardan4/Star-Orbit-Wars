"""Evaluate a fixed weights dict against a TuRBO fitness pool.

Use before launching TuRBO-v2 so we have a reference point to beat. The
last run completed only 2 of 25 trials; we never recorded what *default*
HEURISTIC_WEIGHTS actually score against the w2 pool. Without that
baseline, a trial reporting "0.35 win rate" is ambiguous — better or
worse than what ships on the ladder?

Usage:
    python tools/baseline_eval.py \
        --pool w2 --games-per-opp 5 --step-timeout 1.0 \
        --out runs/baseline_w2_defaults_20260424.json

Writes the full FitnessResult (per-game records + per-opp breakdown) so
it's directly comparable to turbo_runner output.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from orbitwars.bots.heuristic import HEURISTIC_WEIGHTS
from orbitwars.tune.fitness import (
    FitnessConfig,
    evaluate,
    starter_pool,
    w1_pool,
    w2_pool,
    w3_pool,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["starter", "w1", "w2", "w3"], default="w2")
    ap.add_argument("--games-per-opp", type=int, default=5)
    ap.add_argument("--step-timeout", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers for per-game eval (1=serial).",
    )
    args = ap.parse_args()

    if args.pool == "starter":
        opps = starter_pool()
    elif args.pool == "w1":
        opps = w1_pool()
    elif args.pool == "w3":
        opps = w3_pool()
    else:
        opps = w2_pool()

    cfg = FitnessConfig(
        opponents=opps,
        games_per_opponent=args.games_per_opp,
        step_timeout=args.step_timeout,
        seed_base=args.seed,
        workers=args.workers,
    )

    t0 = time.perf_counter()
    result = evaluate(dict(HEURISTIC_WEIGHTS), cfg)
    wall = time.perf_counter() - t0

    by_opp = result.by_opponent()
    print(f"\n=== Baseline eval ===")
    print(f"pool={args.pool}  games/opp={args.games_per_opp}  step_timeout={args.step_timeout}")
    print(f"overall win_rate={result.win_rate:.3f}  hard_win_rate={result.hard_win_rate:.3f}")
    print(f"n_games={result.n_games}  wall={wall:.0f}s")
    print(f"\nBy opponent:")
    for opp_name in sorted(by_opp.keys()):
        wr, n = by_opp[opp_name]
        print(f"  {opp_name:<15} {wr:.3f} ({int(wr*n)}/{n})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "weights": dict(HEURISTIC_WEIGHTS),
                "pool": args.pool,
                "games_per_opp": args.games_per_opp,
                "step_timeout": args.step_timeout,
                "seed": args.seed,
                "win_rate": result.win_rate,
                "hard_win_rate": result.hard_win_rate,
                "n_games": result.n_games,
                "wall_seconds": wall,
                "by_opponent": {k: list(v) for k, v in by_opp.items()},
                "games": [
                    {
                        "opp_name": g.opp_name,
                        "seed": g.seed,
                        "hero_seat": g.hero_seat,
                        "reward": g.reward,
                        "steps": g.steps,
                        "wall_seconds": g.wall_seconds,
                    }
                    for g in result.games
                ],
            },
            indent=2,
        )
    )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
