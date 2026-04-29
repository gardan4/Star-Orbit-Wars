"""Joint policy+value training (AlphaZero-style) on MCTS demos.

The unified version of ``tools/train_value_head.py`` (value-only) and
``tools/train_policy_head.py`` (policy-only). Trains both heads
together with the standard AlphaZero loss:

    L = lambda_p * CE(visit_dist, softmax(cell_logits))
      + lambda_v * MSE(terminal_value, value_pred)
      + lambda_l2 * ||policy_head_params + value_head_params||^2

By default the backbone is FROZEN (only the two heads update). Pass
``--unfreeze-backbone-after-epoch N`` to unfreeze the conv stem +
residual blocks after epoch N — useful for full AlphaZero policy
iteration where the backbone needs to learn features that better
support both heads. The policy & value heads' losses can be
imbalanced, so the lambda weights are exposed.

Why a joint trainer
-------------------
Iterative head-only training (run train_value_head, then
train_policy_head) is fine for prototyping but suboptimal: each head
sees the OLD backbone, so neither can request features the backbone
should learn. Joint training lets the backbone evolve toward
features useful for BOTH the value and policy outputs simultaneously
— the AlphaZero recipe in the original paper.

Why post-hoc temperature/smoothing on the policy target
-------------------------------------------------------
The 2026-04-29 iter-1 demos showed 85% of visit_dist rows are
essentially one-hot, meaning policy distillation from raw visits
gives the policy head no signal beyond what BC already provides.
This script applies AlphaZero-style temperature ``tau`` and
optional epsilon-smoothing AT TRAIN TIME so the same demos can be
re-used with different target shapes — no need to re-collect.

Run:
    python -m tools.train_az_head \\
        --demos runs/closed_loop_iter1_postfix/demos_iter1.npz \\
        --bc-checkpoint runs/bc_warmstart_small_cpu.pt \\
        --out runs/closed_loop_iter1_postfix/az_iter1.pt \\
        --epochs 30 --batch-size 256 --lr 1e-3 \\
        --policy-tau 1.5 --policy-eps 0.5 \\
        --lambda-p 1.0 --lambda-v 1.0 \\
        --unfreeze-backbone-after-epoch 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from orbitwars.nn.conv_policy import ConvPolicy, ConvPolicyCfg


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            print(f"device: cuda ({torch.cuda.get_device_name(0)})", flush=True)
            return torch.device("cuda")
        print("device: cpu", flush=True)
        return torch.device("cpu")
    return torch.device(arg)


def _load_bc_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state" in ckpt and "cfg" in ckpt:
        cfg = ConvPolicyCfg(**ckpt["cfg"])
        model = ConvPolicy(cfg)
        model.load_state_dict(ckpt["model_state"])
    elif "model_state_dict" in ckpt:
        cfg = ConvPolicyCfg()
        model = ConvPolicy(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise RuntimeError(f"unrecognized checkpoint format at {path}")
    return model.to(device), cfg


def _shape_target(visit_dist: np.ndarray, tau: float, eps: float) -> np.ndarray:
    """Apply AlphaZero-style temperature + smoothing to the visit
    distribution targets. Operates on (N, 8) arrays.

    target = (visits + eps) ** (1/tau) / sum
    """
    out = visit_dist.copy()
    if eps > 0.0:
        out = out + eps
    if tau != 1.0 and tau > 0.0:
        out = np.power(np.maximum(out, 0.0), 1.0 / tau)
    out_sum = out.sum(axis=1, keepdims=True)
    out_sum = np.where(out_sum > 0, out_sum, 1.0)
    return out / out_sum


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True, type=Path,
                    help=".npz with x + gy + gx + visit_dist + terminal_value")
    ap.add_argument("--bc-checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)

    # Loss weights.
    ap.add_argument("--lambda-p", type=float, default=1.0,
                    help="Policy CE loss weight")
    ap.add_argument("--lambda-v", type=float, default=1.0,
                    help="Value MSE loss weight")
    ap.add_argument("--lambda-l2", type=float, default=1e-4,
                    help="L2 regularization on trainable params")

    # Policy-target shaping (post-hoc on saved visit_dist).
    ap.add_argument("--policy-tau", type=float, default=1.0,
                    help="Visit-distribution temperature applied at train "
                         "time. tau>1.0 spreads the target.")
    ap.add_argument("--policy-eps", type=float, default=0.0,
                    help="Epsilon added to each channel before tau "
                         "transform. Lifts zero-visit channels off the "
                         "floor; necessary if visit_dist is one-hot.")

    # Unfreeze schedule.
    ap.add_argument("--unfreeze-backbone-after-epoch", type=int, default=-1,
                    help="-1 (default): backbone frozen for entire run. "
                         ">=0: unfreeze stem + blocks at the start of "
                         "epoch N+1. Use sparingly; backbone changes can "
                         "destabilize both heads.")

    args = ap.parse_args()

    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load demos.
    print(f"loading demos from {args.demos}", flush=True)
    data = np.load(args.demos)
    required = ("x", "gy", "gx", "visit_dist", "terminal_value")
    missing = [k for k in required if k not in data]
    if missing:
        print(f"ERROR: demos missing fields: {missing}", file=sys.stderr)
        return 1

    x_all = data["x"].astype(np.float32)
    gy_all = data["gy"].astype(np.int64)
    gx_all = data["gx"].astype(np.int64)
    visit_all = data["visit_dist"].astype(np.float32)
    v_all = data["terminal_value"].astype(np.float32)
    n = x_all.shape[0]

    # Inspect target shape.
    rows_one_hot = (visit_all.max(axis=1) > 0.9).mean()
    print(
        f"  {n:,} demos. visit_dist max-channel mean={visit_all.max(axis=1).mean():.3f} "
        f"(one-hot fraction={rows_one_hot:.3f})  "
        f"v_mix: win={int((v_all > 0.5).sum())} "
        f"loss={int((v_all < -0.5).sum())} mean={v_all.mean():.3f}",
        flush=True,
    )

    # Apply policy target shaping.
    visit_shaped = _shape_target(visit_all, args.policy_tau, args.policy_eps)
    rows_one_hot_shaped = (visit_shaped.max(axis=1) > 0.9).mean()
    print(
        f"  policy-shaped (tau={args.policy_tau}, eps={args.policy_eps}): "
        f"max-channel mean={visit_shaped.max(axis=1).mean():.3f} "
        f"(one-hot fraction={rows_one_hot_shaped:.3f})",
        flush=True,
    )

    # Train/val split.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = int(args.val_frac * n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"  train={len(train_idx):,}  val={len(val_idx):,}", flush=True)

    # Load model.
    print(f"loading checkpoint from {args.bc_checkpoint}", flush=True)
    model, cfg = _load_bc_checkpoint(args.bc_checkpoint, device)
    print(f"  cfg: backbone={cfg.backbone_channels} blocks={cfg.n_blocks}",
          flush=True)

    # Initial freeze: only the two heads train.
    def _set_freeze(unfreeze_backbone: bool):
        for name, p in model.named_parameters():
            if name.startswith("policy_head.") or name.startswith("value_head."):
                p.requires_grad = True
            else:
                p.requires_grad = unfreeze_backbone

    _set_freeze(unfreeze_backbone=False)
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total_params = sum(p.numel() for p in model.parameters())
    print(
        f"  trainable: {n_train_params:,} / {n_total_params:,} "
        f"({100 * n_train_params / max(1, n_total_params):.1f}%)  "
        f"[heads only initially]",
        flush=True,
    )

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.lambda_l2,
    )

    # Move data to tensors.
    x_train = torch.from_numpy(x_all[train_idx])
    gy_train = torch.from_numpy(gy_all[train_idx])
    gx_train = torch.from_numpy(gx_all[train_idx])
    p_train = torch.from_numpy(visit_shaped[train_idx])
    v_train = torch.from_numpy(v_all[train_idx])
    x_val = torch.from_numpy(x_all[val_idx]).to(device, non_blocking=True)
    gy_val = torch.from_numpy(gy_all[val_idx]).to(device, non_blocking=True)
    gx_val = torch.from_numpy(gx_all[val_idx]).to(device, non_blocking=True)
    p_val = torch.from_numpy(visit_shaped[val_idx]).to(device, non_blocking=True)
    v_val = torch.from_numpy(v_all[val_idx]).to(device, non_blocking=True)

    def _losses(logits_full, gy_b, gx_b, p_target, value_pred, v_target):
        B = logits_full.shape[0]
        idx_b = torch.arange(B, device=logits_full.device)
        cell_logits = logits_full[idx_b, :, gy_b, gx_b]
        log_probs = F.log_softmax(cell_logits, dim=-1)
        eps = 1e-9
        p_safe = p_target + eps
        p_safe = p_safe / p_safe.sum(dim=-1, keepdim=True)
        ce = -(p_safe * log_probs).sum(dim=-1).mean()
        value_pred = value_pred.squeeze(-1)
        mse = F.mse_loss(value_pred, v_target)
        return ce, mse

    history = {
        "train_total": [], "train_ce": [], "train_mse": [],
        "val_ce": [], "val_mse": [],
        "wall_seconds": [],
    }
    t_start = time.perf_counter()
    print(f"\ntraining {args.epochs} epochs  "
          f"lambda_p={args.lambda_p}  lambda_v={args.lambda_v}  "
          f"lambda_l2={args.lambda_l2}", flush=True)

    for epoch in range(args.epochs):
        # Unfreeze backbone if scheduled.
        if (
            args.unfreeze_backbone_after_epoch >= 0
            and epoch == args.unfreeze_backbone_after_epoch + 1
        ):
            print(
                f"  >>> epoch {epoch + 1}: UNFREEZING backbone <<<",
                flush=True,
            )
            _set_freeze(unfreeze_backbone=True)
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr * 0.5,  # lower LR for backbone fine-tune
                weight_decay=args.lambda_l2,
            )

        model.train()
        order = torch.randperm(len(train_idx))
        sum_total = sum_ce = sum_mse = 0.0
        n_seen = 0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            x_b = x_train[idx].to(device, non_blocking=True)
            gy_b = gy_train[idx].to(device, non_blocking=True)
            gx_b = gx_train[idx].to(device, non_blocking=True)
            p_b = p_train[idx].to(device, non_blocking=True)
            v_b = v_train[idx].to(device, non_blocking=True)

            logits_full, value_pred = model(x_b)
            ce, mse = _losses(logits_full, gy_b, gx_b, p_b, value_pred, v_b)
            total = args.lambda_p * ce + args.lambda_v * mse
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()

            bsz = len(idx)
            sum_total += float(total.item()) * bsz
            sum_ce += float(ce.item()) * bsz
            sum_mse += float(mse.item()) * bsz
            n_seen += bsz

        train_total = sum_total / max(1, n_seen)
        train_ce = sum_ce / max(1, n_seen)
        train_mse = sum_mse / max(1, n_seen)

        # Val in batches.
        model.eval()
        with torch.no_grad():
            n_v = len(val_idx)
            v_ce = v_mse = 0.0
            for s in range(0, n_v, args.batch_size):
                e = min(n_v, s + args.batch_size)
                logits_v, value_v = model(x_val[s:e])
                ce_v, mse_v = _losses(
                    logits_v, gy_val[s:e], gx_val[s:e],
                    p_val[s:e], value_v, v_val[s:e],
                )
                v_ce += float(ce_v.item()) * (e - s)
                v_mse += float(mse_v.item()) * (e - s)
            val_ce = v_ce / max(1, n_v)
            val_mse = v_mse / max(1, n_v)

        elapsed = time.perf_counter() - t_start
        history["train_total"].append(train_total)
        history["train_ce"].append(train_ce)
        history["train_mse"].append(train_mse)
        history["val_ce"].append(val_ce)
        history["val_mse"].append(val_mse)
        history["wall_seconds"].append(elapsed)
        print(
            f"  epoch {epoch + 1:>2}/{args.epochs}  "
            f"train: total={train_total:.4f} ce={train_ce:.4f} mse={train_mse:.4f}  "
            f"val: ce={val_ce:.4f} mse={val_mse:.4f}  wall={elapsed:.0f}s",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict
    torch.save({
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "cfg": asdict(cfg),
        "az_trained_jointly": True,
        "training_history": history,
        "training_args": vars(args),
        "source_bc_checkpoint": str(args.bc_checkpoint),
        "source_demos": str(args.demos),
    }, args.out)
    print(f"\nsaved {args.out}", flush=True)

    json_path = args.out.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({
            "cfg": asdict(cfg),
            "training_history": history,
            "training_args": {k: (str(v) if isinstance(v, Path) else v)
                              for k, v in vars(args).items()},
            "n_demos": int(n),
        }, f, indent=2)
    print(f"sidecar {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
