"""Measure per-turn wall time for MCTSAgent so the overage-bank plan
has real numbers.

Question we're answering: under the default `hard_deadline_ms=300` search
budget, do we have significant headroom below the Kaggle 1-second ceiling?
If p95 is already near 1s, no overage-bank will help us run more sims;
if p95 is well under, a 10-s opening-turn investment + conservative
tail is genuinely available.

Output: p50/p90/p95/p99/max per player across one full game, plus a
count of turns that exceeded the internal SEARCH_DEADLINE_MS (850 ms).
Runs one game (~200s wall-clock, seeded) and exits.
"""
from __future__ import annotations

import argparse
import statistics

from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.bots.mcts_bot import MCTSAgent
from orbitwars.mcts.gumbel_search import GumbelConfig
from tournaments.harness import play_game


def _percentiles(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return {}
    def pct(p):
        # Nearest-rank — robust for small-to-medium samples.
        k = max(0, min(n - 1, int(round(p * (n - 1)))))
        return xs[k]
    return {
        "n": n,
        "mean": statistics.fmean(xs),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": xs[-1],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mcts-seat", type=int, choices=(0, 1), default=0)
    ap.add_argument(
        "--hard-deadline-ms", type=float, default=300.0,
        help="MCTS internal hard deadline. Kaggle actTimeout is 1000 ms.",
    )
    args = ap.parse_args()

    mcts = MCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=4, total_sims=32, rollout_depth=15,
            hard_deadline_ms=args.hard_deadline_ms,
            anchor_improvement_margin=2.0,
        ),
        rng_seed=0,
    )
    heur = HeuristicAgent()

    agents = [mcts.as_kaggle_agent(), heur.as_kaggle_agent()]
    if args.mcts_seat == 1:
        agents = [heur.as_kaggle_agent(), mcts.as_kaggle_agent()]

    print(f"Running one game: MCTS(seat={args.mcts_seat}) vs HEUR, seed={args.seed}")
    print(f"MCTS hard_deadline_ms={args.hard_deadline_ms}", flush=True)

    result = play_game(agents, seed=args.seed, players=2, step_timeout=2.0)
    print(f"Steps: {result.steps}")
    print(f"Final scores: {result.final_scores}")
    print()

    mcts_times = result.turn_times_ms[args.mcts_seat]
    heur_times = result.turn_times_ms[1 - args.mcts_seat]

    for label, times in (("MCTS", mcts_times), ("HEUR", heur_times)):
        pct = _percentiles(times)
        print(f"--- {label} ---")
        print(
            f"  n={pct['n']:>4d}  mean={pct['mean']:>7.1f} ms  "
            f"p50={pct['p50']:>6.1f}  p90={pct['p90']:>6.1f}  "
            f"p95={pct['p95']:>6.1f}  p99={pct['p99']:>6.1f}  "
            f"max={pct['max']:>7.1f}"
        )
        # Guard against losing to timeout-style forfeit.
        over_900 = sum(1 for t in times if t >= 900.0)
        over_850 = sum(1 for t in times if t >= 850.0)
        if over_900 or over_850:
            print(
                f"  WARN: {over_850} turns >= 850 ms, "
                f"{over_900} turns >= 900 ms — near the Kaggle 1-s ceiling"
            )
        headroom_p95 = 1000.0 - pct["p95"]
        print(f"  budget headroom at p95: {headroom_p95:+.0f} ms below 1000 ms")


if __name__ == "__main__":
    main()
