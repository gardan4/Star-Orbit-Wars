"""Backup ship path: v11.1 = v11 with TuRBO-v4 weights instead of v3 weights.

Use this if:
* v12 (NN prior) fails the H2H gate vs v11.
* TuRBO-v4 has produced a clear winner (best trial wr > 0.95 vs the
  9-archetype pool — same bar v3 trial 42 cleared at 1.000).

The default config matches v11 except for the weights JSON path:
  weights:           runs/turbo_v4_*.json (best trial)
  sim_move_variant:  exp3 (eta=0.3) — same as v11
  rollout_policy:    heuristic (default) — same as v11
  anchor_margin:     2.0 (default) — same as v11
  use_opponent_model: True (default) — same as v11

This is the conservative-defense ship: if NN prior didn't pan out, at
least we've tested wider TuRBO bounds and may have a stronger weights
configuration to defend the v11 ladder floor.

Run:
  $env:PYTHONPATH="src;."; .venv\\Scripts\\python.exe tools\\build_v11_1.py \\
      --weights-json runs\\turbo_v4_20260425d.json \\
      --out submissions\\v11_1.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from tools.bundle import bundle_bot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-json", required=True,
                    help="TuRBO-v4 weights JSON")
    ap.add_argument("--out", default="submissions/v11_1.py")
    ap.add_argument("--no-smoke", action="store_true")
    ap.add_argument("--exp3-eta", type=float, default=0.3)
    args = ap.parse_args()

    wp = Path(args.weights_json)
    if not wp.exists():
        print(f"ERROR: weights JSON not found at {wp}", file=sys.stderr)
        return 2

    # Same recipe as v11 build, but pointing at v4 weights instead of v3.
    import json
    raw = json.loads(wp.read_text(encoding="utf-8"))
    for key in ("best_weights", "best_point", "best", "weights"):
        if isinstance(raw, dict) and key in raw and isinstance(raw[key], dict):
            raw = raw[key]
            break
    if isinstance(raw, dict) and "best_index" in raw and isinstance(raw.get("trials"), list):
        best_i = int(raw["best_index"])
        for tr in raw["trials"]:
            if int(tr.get("trial", -1)) == best_i and isinstance(tr.get("weights"), dict):
                raw = tr["weights"]
                break
    if not isinstance(raw, dict):
        print(f"ERROR: couldn't unwrap weights from {wp}", file=sys.stderr)
        return 3
    weights_override = {k: float(v) for k, v in raw.items()}
    print(f"weights override: {len(weights_override)} keys from {wp}")

    out_path = Path(args.out)
    bundle_bot(
        "mcts_bot", out_path,
        weights_override=weights_override,
        sim_move_variant="exp3",
        exp3_eta=args.exp3_eta,
    )
    print(f"bundle: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")

    if not args.no_smoke:
        from tools.bundle import _smoke_test
        _smoke_test(out_path)

    print(
        f"\n=== v11.1 ready at {out_path} ===\n"
        "Next: H2H vs v11.py and ship if wr >= 0.55:\n"
        f"  python -m tools.diag_bundle_round_robin --bundles {out_path},submissions/v11.py --games 20 --seed 42"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
