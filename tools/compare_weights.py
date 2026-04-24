"""Compare two weight dicts head-to-head over a fitness pool.

**Ship-or-shelve gate** for any tuning run. After TuRBO finishes we
must decide whether the best weights are a statistically-real
improvement over the shipped defaults — not just a noisy Sobol outlier.
This tool takes a weights JSON (e.g. from TuRBO's best trial) and a
reference label (``defaults`` or another JSON), plays both against the
same opponent pool with IDENTICAL seeds, and emits:

  * win_rate, hard_win_rate, and per-opp breakdown for each side.
  * Paired-game delta vs opponent pool — since we use the same
    seed_base and seat-alternation, every seed is played by both
    sides, giving a paired comparison stronger than independent
    samples of equal N.
  * Binomial SE and a crude 95% CI on the delta.

Usage:
    python tools/compare_weights.py \\
        --weights-a runs/turbo_v2_20260424.json --label-a turbo_v2 \\
        --weights-b defaults \\
        --pool w2 --games-per-opp 10 --step-timeout 1.0 --workers 7 \\
        --out runs/compare_turbo_v2_vs_defaults.json

Decision rule (plan §W3 gate): ship TuRBO weights iff
``turbo_wr - defaults_wr >= 0.07`` (one binomial SE over the 0.667
defaults baseline) AND no per-opp regression worse than -0.10.

--weights-a / --weights-b accept:
  * ``defaults`` — the shipped ``HEURISTIC_WEIGHTS`` dict.
  * A path to a JSON file shaped like TuRBO's output (``best`` or
    ``best_point`` key) or a raw weights dict.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Tuple

from orbitwars.bots.heuristic import HEURISTIC_WEIGHTS
from orbitwars.tune.fitness import (
    FitnessConfig,
    FitnessResult,
    evaluate,
    starter_pool,
    w1_pool,
    w2_pool,
)


POOL_FACTORIES = {
    "starter": starter_pool,
    "w1": w1_pool,
    "w2": w2_pool,
}


def _load_weights(spec: str) -> Dict[str, float]:
    """Resolve a CLI weight spec to a concrete weights dict.

    ``defaults`` — ship the in-tree default weights.
    path to .json — parse. If the JSON has ``best``/``best_point``/
    ``best_weights`` at top-level we unwrap it; otherwise we assume
    the top-level IS the weights dict.
    """
    if spec == "defaults":
        return dict(HEURISTIC_WEIGHTS)
    p = Path(spec)
    if not p.exists():
        raise FileNotFoundError(f"weights file not found: {spec}")
    data = json.loads(p.read_text(encoding="utf-8"))
    for key in ("best_weights", "best_point", "best", "weights"):
        if key in data and isinstance(data[key], dict):
            return {k: float(v) for k, v in data[key].items()}
    # TuRBO runner shape: {best_index, trials: [{trial, weights, ...}]}
    # — unwrap the best trial's weights.
    if (
        isinstance(data, dict)
        and "best_index" in data
        and isinstance(data.get("trials"), list)
        and data["best_index"] is not None
    ):
        best_i = int(data["best_index"])
        for tr in data["trials"]:
            if int(tr.get("trial", -1)) == best_i and isinstance(tr.get("weights"), dict):
                return {k: float(v) for k, v in tr["weights"].items()}
    # Assume top-level IS the weights dict
    if isinstance(data, dict) and all(isinstance(v, (int, float)) for v in data.values()):
        return {k: float(v) for k, v in data.items()}
    raise ValueError(
        f"couldn't find a weights dict in {spec!r} — expected a top-level dict, "
        f"a TuRBO runner output (best_index + trials), or one of "
        f"{{best_weights, best_point, best, weights}}."
    )


def _run_side(
    weights: Dict[str, float],
    cfg: FitnessConfig,
    label: str,
) -> Tuple[FitnessResult, float]:
    t0 = time.perf_counter()
    result = evaluate(weights, cfg)
    wall = time.perf_counter() - t0
    print(f"[{label}] n={result.n_games}  win_rate={result.win_rate:.3f}  "
          f"hard={result.hard_win_rate:.3f}  wall={wall:.0f}s", flush=True)
    return result, wall


def _paired_delta(a: FitnessResult, b: FitnessResult) -> Tuple[float, float, float]:
    """Paired per-game delta (a_reward - b_reward), assuming same seed
    ordering. Returns (mean_delta, se, wins_for_a).

    Deltas are in {-2, -1, 0, +1, +2} (reward is {-1, 0, +1}, difference
    on same seed is at most ±2). We report mean + binomial-ish SE on the
    indicator ``a_reward > b_reward`` which is the conventional read of
    head-to-head win-rate.
    """
    n = min(a.n_games, b.n_games)
    if n == 0:
        return (0.0, 0.0, 0.5)
    deltas: list = []
    wins_for_a = 0
    for ga, gb in zip(a.games[:n], b.games[:n]):
        if ga.seed != gb.seed or ga.opp_name != gb.opp_name:
            # Not paired — fall back to mean-of-rewards.
            continue
        d = ga.reward - gb.reward
        deltas.append(d)
        if ga.reward > gb.reward:
            wins_for_a += 1
        elif ga.reward == gb.reward:
            wins_for_a += 0.5  # tie counts half
    if not deltas:
        mean_reward_a = sum(g.reward for g in a.games) / max(1, a.n_games)
        mean_reward_b = sum(g.reward for g in b.games) / max(1, b.n_games)
        return (mean_reward_a - mean_reward_b, float("nan"), float("nan"))

    mean_d = sum(deltas) / len(deltas)
    p = wins_for_a / len(deltas)
    se = math.sqrt(max(0.0, p * (1 - p) / len(deltas)))
    return (mean_d, se, p)


def _diff_by_opp(a: FitnessResult, b: FitnessResult) -> Dict[str, Dict[str, float]]:
    a_by = a.by_opponent()
    b_by = b.by_opponent()
    names = sorted(set(a_by) | set(b_by))
    out: Dict[str, Dict[str, float]] = {}
    for n in names:
        wr_a, n_a = a_by.get(n, (0.0, 0))
        wr_b, n_b = b_by.get(n, (0.0, 0))
        out[n] = {
            "a_wr": wr_a, "a_n": int(n_a),
            "b_wr": wr_b, "b_n": int(n_b),
            "delta": wr_a - wr_b,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-a", required=True,
                    help="weights spec: 'defaults' or path to .json")
    ap.add_argument("--weights-b", required=True,
                    help="weights spec: 'defaults' or path to .json")
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    ap.add_argument("--pool", choices=list(POOL_FACTORIES), default="w2")
    ap.add_argument("--games-per-opp", type=int, default=10)
    ap.add_argument("--step-timeout", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    weights_a = _load_weights(args.weights_a)
    weights_b = _load_weights(args.weights_b)
    pool = POOL_FACTORIES[args.pool]()
    cfg = FitnessConfig(
        opponents=pool,
        games_per_opponent=args.games_per_opp,
        step_timeout=args.step_timeout,
        seed_base=args.seed,
        workers=args.workers,
    )

    print(f"\n=== compare_weights ===")
    print(f"pool={args.pool}  games/opp={args.games_per_opp}  "
          f"step_timeout={args.step_timeout}  workers={args.workers}")
    print(f"A={args.label_a}  B={args.label_b}")

    # Run in series — each uses the full worker pool.
    a_res, a_wall = _run_side(weights_a, cfg, args.label_a)
    b_res, b_wall = _run_side(weights_b, cfg, args.label_b)

    mean_d, se_pair, wins_for_a = _paired_delta(a_res, b_res)
    delta_wr = a_res.win_rate - b_res.win_rate
    per_opp = _diff_by_opp(a_res, b_res)

    # Wilson-ish CI on the win_rate delta using a conservative binomial
    # SE at each side, combined.
    n_a = max(1, a_res.n_games)
    n_b = max(1, b_res.n_games)
    se_a = math.sqrt(a_res.win_rate * (1 - a_res.win_rate) / n_a)
    se_b = math.sqrt(b_res.win_rate * (1 - b_res.win_rate) / n_b)
    combined_se = math.sqrt(se_a ** 2 + se_b ** 2)
    ci95 = (delta_wr - 1.96 * combined_se, delta_wr + 1.96 * combined_se)

    print(f"\n=== Summary ===")
    print(f"{args.label_a}_wr - {args.label_b}_wr = {delta_wr:+.3f}  "
          f"(95% CI: {ci95[0]:+.3f}, {ci95[1]:+.3f})")
    print(f"paired-game: {args.label_a} > {args.label_b} on "
          f"{wins_for_a:.1%} of seeds  (SE {se_pair:.3f})")
    print(f"\nPer-opponent delta ({args.label_a} - {args.label_b}):")
    for name in sorted(per_opp):
        row = per_opp[name]
        print(f"  {name:<15} {row['a_wr']:.3f} vs {row['b_wr']:.3f} "
              f"= {row['delta']:+.3f}  (n={row['a_n']}/{row['b_n']})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "args": vars(args),
        "a": {
            "label": args.label_a,
            "win_rate": a_res.win_rate,
            "hard_win_rate": a_res.hard_win_rate,
            "n_games": a_res.n_games,
            "wall_seconds": a_wall,
            "weights": weights_a,
        },
        "b": {
            "label": args.label_b,
            "win_rate": b_res.win_rate,
            "hard_win_rate": b_res.hard_win_rate,
            "n_games": b_res.n_games,
            "wall_seconds": b_wall,
            "weights": weights_b,
        },
        "delta_win_rate": delta_wr,
        "ci95": list(ci95),
        "paired_win_rate_for_a": wins_for_a,
        "paired_se": se_pair,
        "per_opp": per_opp,
    }, indent=2), encoding="utf-8")
    print(f"\nWritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
