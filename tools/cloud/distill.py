"""Distill a large teacher ConvPolicy into a smaller student.

Why: a 5M-param teacher (64ch x 16 blocks) is the right size for AZ
self-play to converge to a strong policy/value, but it's too big to ship
inline (~20 MB fp32 -> 27 MB base64 -> exceeds Kaggle's 1 MB notebook
push limit). The Kaggle Dataset path is also unreliable for big NNs
(see docs/KAGGLE_DATASET_BIG_AZ_BUG.md). The clean ship path is to
distill the teacher into a smaller student (e.g. 48ch x 8, ~1M params)
that fits inline cleanly when int8-quantized.

Distillation loss: KL on the teacher's policy logits + MSE on the
teacher's value output. The student trains on the teacher's outputs at
the same input states; ground-truth labels (visit_dist + terminal_value
in the demos) are NOT used directly — the teacher already integrated them.

This typically retains ~80-90% of the teacher's strength at 1/5th the
parameter count, per the standard Hinton distillation literature.

Run:
    python -m tools.cloud.distill \\
        --teacher runs/cloud_az/iter10/checkpoint.pt \\
        --demos runs/cloud_az/iter10/demos.npz \\
        --student-channels 48 --student-blocks 8 \\
        --out runs/cloud_az_distilled.pt \\
        --epochs 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from orbitwars.nn.conv_policy import ConvPolicy, ConvPolicyCfg


def _load_ckpt(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state" in ckpt and "cfg" in ckpt:
        cfg = ConvPolicyCfg(**ckpt["cfg"])
        m = ConvPolicy(cfg).to(device)
        m.load_state_dict(ckpt["model_state"])
        return m, cfg
    raise RuntimeError(f"Unrecognized checkpoint format at {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, type=Path,
                    help="Trained teacher .pt to distill from")
    ap.add_argument("--demos", required=True, type=Path,
                    help=".npz with x grids (visit_dist + terminal_value not used)")
    ap.add_argument("--out", required=True, type=Path,
                    help="Where to save the distilled student .pt")
    ap.add_argument("--student-channels", type=int, default=48,
                    help="Student backbone_channels (smaller than teacher)")
    ap.add_argument("--student-blocks", type=int, default=8,
                    help="Student n_blocks")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=2.0,
                    help="Softmax temperature for distillation (>1 softens)")
    ap.add_argument("--lambda-policy", type=float, default=1.0)
    ap.add_argument("--lambda-value", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"device: {device}", flush=True)

    # Load teacher (frozen, eval mode)
    print(f"loading teacher from {args.teacher}", flush=True)
    teacher, t_cfg = _load_ckpt(args.teacher, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    n_t = sum(p.numel() for p in teacher.parameters())
    print(f"  teacher: {n_t:,} params, cfg={t_cfg}", flush=True)

    # Build student
    s_cfg = ConvPolicyCfg(
        backbone_channels=args.student_channels,
        n_blocks=args.student_blocks,
    )
    student = ConvPolicy(s_cfg).to(device)
    n_s = sum(p.numel() for p in student.parameters())
    compress = n_t / max(1, n_s)
    print(f"  student: {n_s:,} params (compression {compress:.1f}×), cfg={s_cfg}", flush=True)

    # Load demos
    print(f"loading demos from {args.demos}", flush=True)
    data = np.load(args.demos)
    if "x" not in data:
        print(f"ERROR: demos missing 'x' field", file=sys.stderr); return 1
    x_all = data["x"].astype(np.float32)
    n = x_all.shape[0]
    print(f"  {n:,} state grids", flush=True)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = int(args.val_frac * n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"  train={len(train_idx):,}  val={len(val_idx):,}", flush=True)

    x_train = torch.from_numpy(x_all[train_idx])
    x_val = torch.from_numpy(x_all[val_idx]).to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr,
                                 weight_decay=1e-4)

    T = args.temperature
    print(f"\nDistilling {args.epochs} epochs   T={T}   "
          f"lambda_p={args.lambda_policy}  lambda_v={args.lambda_value}", flush=True)

    t_start = time.perf_counter()
    history: dict[str, list[float]] = {
        "train_kl": [], "train_mse": [], "val_kl": [], "val_mse": [],
    }

    for epoch in range(args.epochs):
        student.train()
        order = torch.randperm(len(train_idx))
        sum_kl = sum_mse = 0.0
        n_seen = 0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            x_b = x_train[idx].to(device, non_blocking=True)

            # Teacher forward (no grad)
            with torch.no_grad():
                t_logits, t_value = teacher(x_b)
                # Softmax with temperature on flat per-cell channels.
                # ConvPolicy logits are (B, A, H, W); we treat each cell
                # independently. KL is computed across the action axis.
                t_log_probs = F.log_softmax(t_logits / T, dim=1)
                t_probs = t_log_probs.exp()

            # Student forward
            s_logits, s_value = student(x_b)
            s_log_probs = F.log_softmax(s_logits / T, dim=1)

            # KL(teacher || student) — pixel-wise then mean
            kl = (t_probs * (t_log_probs - s_log_probs)).sum(dim=1).mean()
            mse = F.mse_loss(s_value, t_value)
            loss = args.lambda_policy * kl * (T * T) + args.lambda_value * mse
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bsz = len(idx)
            sum_kl += float(kl.item()) * bsz
            sum_mse += float(mse.item()) * bsz
            n_seen += bsz

        train_kl = sum_kl / max(1, n_seen)
        train_mse = sum_mse / max(1, n_seen)

        # Val
        student.eval()
        with torch.no_grad():
            v_kl = v_mse = 0.0
            n_v = len(val_idx)
            for s in range(0, n_v, args.batch_size):
                e = min(n_v, s + args.batch_size)
                x_b = x_val[s:e]
                t_l, t_v = teacher(x_b)
                s_l, s_v = student(x_b)
                tlp = F.log_softmax(t_l / T, dim=1)
                tp = tlp.exp()
                slp = F.log_softmax(s_l / T, dim=1)
                v_kl += float((tp * (tlp - slp)).sum(dim=1).mean().item()) * (e - s)
                v_mse += float(F.mse_loss(s_v, t_v).item()) * (e - s)
            val_kl = v_kl / max(1, n_v)
            val_mse = v_mse / max(1, n_v)

        history["train_kl"].append(train_kl)
        history["train_mse"].append(train_mse)
        history["val_kl"].append(val_kl)
        history["val_mse"].append(val_mse)
        wall = int(time.perf_counter() - t_start)
        print(f"  epoch {epoch + 1:>2}/{args.epochs}  "
              f"train: kl={train_kl:.4f} mse={train_mse:.4f}  "
              f"val: kl={val_kl:.4f} mse={val_mse:.4f}  wall={wall}s",
              flush=True)

    # Save (Path-sanitized for cross-OS unpickle)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sanitized_args = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
    }
    torch.save({
        "model_state": {k: v.detach().cpu() for k, v in student.state_dict().items()},
        "cfg": asdict(s_cfg),
        "distilled_from": str(args.teacher),
        "distillation_args": sanitized_args,
        "history": history,
    }, args.out)
    print(f"\nsaved {args.out} ({n_s:,} params)", flush=True)

    json_path = args.out.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "cfg": asdict(s_cfg),
            "distilled_from": str(args.teacher),
            "history": history,
            "args": sanitized_args,
            "compression_ratio": compress,
        }, f, indent=2)
    print(f"sidecar {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
