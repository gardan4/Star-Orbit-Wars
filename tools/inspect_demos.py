"""Quick statistics on a collected demo .npz.

Inspects:
  * # demos and field shapes
  * visit_dist concentration (max-channel mean, entropy, fraction one-hot)
  * terminal_value distribution (win/loss/tie balance)

Use to decide whether a demo set has the right entropy for policy
distillation. Rule of thumb: if `fraction max>0.9` is above ~0.3,
distilling raw visit_dist will not improve over BC's argmax-imitation.
Use `tools.train_az_head --policy-tau 1.5 --policy-eps 0.5` to soften
the targets at training time, OR re-collect with a less aggressive
SH (smaller `total_sims`, larger `num_candidates`).

Run:
    python -m tools.inspect_demos runs/closed_loop_iter1_postfix/demos_iter1_big.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("demos", type=Path)
    args = ap.parse_args()

    if not args.demos.exists():
        print(f"ERROR: {args.demos} not found", file=sys.stderr)
        return 1

    print(f"\n=== {args.demos} ===")
    d = np.load(args.demos)
    print(f"size: {args.demos.stat().st_size / 1e6:.1f} MB")
    print("fields:")
    for k in d.keys():
        print(f"  {k}: shape={d[k].shape} dtype={d[k].dtype}")

    if "visit_dist" in d:
        v = d["visit_dist"].astype(np.float32)
        print(f"\nvisit_dist concentration:")
        print(f"  max-channel mean: {v.max(axis=1).mean():.4f}  "
              f"(1.0=one-hot, 0.125=uniform)")
        print(f"  entropy mean: {(-v * np.log(v + 1e-9)).sum(axis=1).mean():.4f}  "
              f"(max log(8)={np.log(8.0):.4f})")
        print(f"  fraction max > 0.5: {(v.max(axis=1) > 0.5).mean():.4f}")
        print(f"  fraction max > 0.9: {(v.max(axis=1) > 0.9).mean():.4f}  "
              f"(>0.3 = policy distillation will not help over BC)")
        # Per-channel utilization (which channels see any visits at all).
        any_mass = (v > 0.0).any(axis=0)
        print(f"  channels with any visit anywhere: {int(any_mass.sum())}/8")

    if "terminal_value" in d:
        tv = d["terminal_value"].astype(np.float32)
        n = len(tv)
        wins = int((tv > 0.5).sum())
        losses = int((tv < -0.5).sum())
        ties = n - wins - losses
        print(f"\nterminal_value distribution:")
        print(f"  n: {n:,}  mean: {tv.mean():.4f}  "
              f"std: {tv.std():.4f}")
        print(f"  win:loss:tie = {wins}:{losses}:{ties}  "
              f"(win-rate={wins/max(1, n):.3f})")
        print(f"  balance score: {1 - abs(2*wins/max(1, n) - 1):.3f}  "
              f"(1.0=perfectly balanced 50/50, 0.0=fully one-sided)")

    if "gy" in d and "gx" in d:
        gy = d["gy"]
        gx = d["gx"]
        print(f"\ngrid coverage:")
        print(f"  gy range: [{gy.min()}, {gy.max()}]  unique: {len(np.unique(gy))}")
        print(f"  gx range: [{gx.min()}, {gx.max()}]  unique: {len(np.unique(gx))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
