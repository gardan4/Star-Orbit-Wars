"""Two-stage trainer for the bigger-backbone AZ head.

Stage 1: BC warm-start on heuristic demos with cfg.backbone_channels=48,
         n_blocks=4 (~250 K params, ~1 MB raw, fits in Kaggle bundle
         without the Dataset path).
Stage 2: AZ refinement (joint policy + value with tau-smoothed targets)
         on post-Phantom MCTS demos, with backbone unfreezing scheduled
         after enough heads-only epochs that the value head is stable.

The 64 K-param "small" model couldn't sustain backbone unfreeze
(training instability at val_mse 0.48 → 1.18). The 250 K target is
chosen because:
  * fits in <1 MB raw torch state dict → fits in <1.4 MB base64
    inline, well under the Kaggle 1.5 MB bundle threshold without
    the Dataset path.
  * 4× the parameter count of the small model. Empirically (see
    bc_warmstart_v2 results: 64-channel/6-block at val_acc 0.600 vs
    32-channel/3-block at val_acc 0.568) the policy head's accuracy
    scales sub-linearly with params here, but the value head's
    coexistence with the policy head DOES stabilize meaningfully
    once backbone capacity passes ~150 K params.
  * RTX 3070 trains it in ~10-15 min (BC) + 5-8 min (AZ), tractable
    in a single session.

Usage
-----
    PYTHONPATH=src .venv-gpu/Scripts/python.exe -m tools.train_az_bigger \\
        --bc-demos runs/bc_demos.npz \\
        --az-demos runs/closed_loop_iter1_postfix/demos_iter1_big.npz \\
        --out runs/az_v39_bigger.pt \\
        --device cuda

The default args train a v39-bigger checkpoint suitable for
``bundle.py --rollout-policy=nn_value --value-mix-alpha=0.5
--nn-checkpoint runs/az_v39_bigger.pt``.

If wall time is tight, use ``--bc-epochs 5 --az-epochs 8`` for a
~10-min smoke; final training should be 15+8 epochs minimum.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _run(label: str, cmd: list) -> int:
    print(f"\n{'=' * 60}\n[{label}] {' '.join(str(c) for c in cmd)}\n{'=' * 60}", flush=True)
    proc = subprocess.run(cmd, cwd=_ROOT)
    if proc.returncode != 0:
        print(f"[{label}] FAILED with exit code {proc.returncode}", file=sys.stderr)
        sys.exit(proc.returncode)
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-demos", type=Path,
                    default=Path("runs/bc_demos.npz"),
                    help="Heuristic-target demo npz for BC stage")
    ap.add_argument("--az-demos", type=Path,
                    default=Path("runs/closed_loop_iter1_postfix/demos_iter1_big.npz"),
                    help="MCTS visit-distribution demos for AZ stage")
    ap.add_argument("--out", type=Path,
                    default=Path("runs/az_v39_bigger.pt"))
    ap.add_argument("--bc-out", type=Path,
                    default=Path("runs/bc_v39_bigger.pt"))
    ap.add_argument("--backbone-channels", type=int, default=48)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--bc-epochs", type=int, default=15)
    ap.add_argument("--az-epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--policy-tau", type=float, default=1.5)
    ap.add_argument("--policy-eps", type=float, default=0.5)
    ap.add_argument("--unfreeze-after", type=int, default=5,
                    help="Unfreeze backbone after this many AZ epochs")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--skip-bc", action="store_true",
                    help="Skip BC warm-start; --bc-out must already exist")
    args = ap.parse_args()

    python_exe = sys.executable

    if not args.skip_bc:
        if not args.bc_demos.exists():
            print(f"BC demos not found: {args.bc_demos}", file=sys.stderr)
            return 1
        _run("BC warm-start", [
            python_exe, "-m", "tools.bc_warmstart",
            "--cache-path", str(args.bc_demos),
            "--out", str(args.bc_out),
            "--epochs", str(args.bc_epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--device", args.device,
            "--backbone-channels", str(args.backbone_channels),
            "--n-blocks", str(args.n_blocks),
        ])
    else:
        if not args.bc_out.exists():
            print(
                f"--skip-bc set but {args.bc_out} doesn't exist",
                file=sys.stderr,
            )
            return 1

    if not args.az_demos.exists():
        print(f"AZ demos not found: {args.az_demos}", file=sys.stderr)
        return 1

    _run("AZ refinement", [
        python_exe, "-m", "tools.train_az_head",
        "--demos", str(args.az_demos),
        "--bc-checkpoint", str(args.bc_out),
        "--out", str(args.out),
        "--epochs", str(args.az_epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--policy-tau", str(args.policy_tau),
        "--policy-eps", str(args.policy_eps),
        "--lambda-p", "1.0",
        "--lambda-v", "1.0",
        "--unfreeze-backbone-after-epoch", str(args.unfreeze_after),
        "--device", args.device,
    ])

    print(f"\nFinal AZ checkpoint: {args.out}")
    print(
        f"\nNext: bundle v39 with mixed leaf eval at α=0.5:\n"
        f"  python -m tools.bundle --bot mcts_bot \\\n"
        f"    --weights-json runs/turbo_v3_20260424.json \\\n"
        f"    --sim-move-variant exp3 --exp3-eta 0.3 \\\n"
        f"    --rollout-policy nn_value --value-mix-alpha 0.5 \\\n"
        f"    --anchor-margin 0.5 --nn-checkpoint {args.out} \\\n"
        f"    --total-sims 64 --hard-deadline-ms 850 --num-candidates 4 \\\n"
        f"    --out submissions/v39.py --smoke-test"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
