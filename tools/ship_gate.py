"""Ship-or-shelve decision helper for a compare_weights JSON.

Reads the output of ``tools/compare_weights.py`` and applies the plan
§W3 gate:

  1. ``delta_wr >= min_delta`` (default 0.07 — one binomial SE above the
     0.667 defaults baseline at N≈90 games).
  2. No per-opp regression worse than ``-max_regression`` (default
     -0.10).

Usage::

    .\\.venv\\Scripts\\python.exe -m tools.ship_gate \\
        --compare runs/compare_turbo_v2_vs_defaults.json \\
        --label-a turbo_v2 --label-b defaults

Exits 0 on PASS, 1 on FAIL. Intended to be called from scripts that
auto-ship on success; for manual review, run with ``--verbose`` to see
which gate failed and by how much.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple


def evaluate_gate(
    compare_json: Dict,
    min_delta: float = 0.07,
    max_regression: float = 0.10,
) -> Tuple[bool, Dict]:
    """Apply the two gates and return (pass, detail)."""
    delta = float(compare_json["delta_win_rate"])
    per_opp = compare_json.get("per_opp", {})
    worst_regression_name = None
    worst_regression_delta = 0.0
    for opp, row in per_opp.items():
        d = float(row["delta"])
        if d < worst_regression_delta:
            worst_regression_delta = d
            worst_regression_name = opp

    gate_delta = delta >= min_delta
    gate_regression = worst_regression_delta >= -max_regression

    # ci95 lower bound is a stronger signal — if it's positive, the
    # improvement is "statistically distinguishable from noise" at ~95%.
    ci_lo, ci_hi = compare_json.get("ci95", [float("nan"), float("nan")])
    ci_clears_zero = ci_lo > 0.0

    detail = {
        "delta_wr": delta,
        "min_delta": min_delta,
        "gate_delta": gate_delta,
        "worst_regression_opp": worst_regression_name,
        "worst_regression_delta": worst_regression_delta,
        "max_regression": max_regression,
        "gate_regression": gate_regression,
        "ci95": [ci_lo, ci_hi],
        "ci_clears_zero": ci_clears_zero,
    }
    return (gate_delta and gate_regression), detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", required=True, type=Path,
                    help="Path to compare_weights JSON output.")
    ap.add_argument("--min-delta", type=float, default=0.07,
                    help="Required win_rate delta (default 0.07 = 1 SE).")
    ap.add_argument("--max-regression", type=float, default=0.10,
                    help="Max allowed per-opp regression (default 0.10).")
    ap.add_argument("--quiet", action="store_true",
                    help="Print only PASS/FAIL.")
    args = ap.parse_args()

    data = json.loads(args.compare.read_text(encoding="utf-8"))
    ok, d = evaluate_gate(data, args.min_delta, args.max_regression)
    label_a = data.get("a", {}).get("label", "A")
    label_b = data.get("b", {}).get("label", "B")

    if args.quiet:
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1

    print(f"=== Ship gate: {label_a} vs {label_b} ===")
    print(f"delta_wr = {d['delta_wr']:+.3f}  (min required: +{args.min_delta:.3f})  "
          f"-> {'PASS' if d['gate_delta'] else 'FAIL'}")
    print(f"95% CI on delta: [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]  "
          f"{'(clears zero)' if d['ci_clears_zero'] else '(spans zero)'}")
    if d["worst_regression_opp"] is None:
        print(f"worst per-opp regression: none (all opps >= {label_b})")
    else:
        print(f"worst per-opp regression: {d['worst_regression_opp']} "
              f"= {d['worst_regression_delta']:+.3f}  "
              f"(max allowed: {-args.max_regression:+.3f})  "
              f"-> {'PASS' if d['gate_regression'] else 'FAIL'}")
    print()
    print(f"VERDICT: {'SHIP' if ok else 'SHELVE'} {label_a}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
