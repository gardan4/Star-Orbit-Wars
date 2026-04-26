"""Train the value head of a ConvPolicy on (state, terminal_value) demos.

Phase 2 step 2 of the NN value-head Q work
(see ``docs/NN_VALUE_HEAD_DESIGN.md``).

The BC training in ``bc_warmstart.py`` discards the value-head output
(``logits, _value = model(x_b)``) so the value head sits at random
init in shipped checkpoints. To make ``rollout_policy='nn_value'``
useful we need the value head to actually predict expected outcome
from a given state — not noise.

This tool:
  1. Loads a demo .npz produced by ``tools/collect_mcts_demos.py``
     that includes the new ``terminal_value`` column
     (+1/-1/0 from the decision-maker's perspective).
  2. Loads an existing BC checkpoint (frozen policy + backbone).
  3. Freezes ``stem``, ``blocks``, ``policy_head`` and trains ONLY
     ``value_head`` via MSE on terminal_value targets.
  4. Saves a new checkpoint with the trained value head.

Why freeze the backbone:
  * Preserves the BC policy that's known to work as a move_prior_fn
    in v22 — we don't want value training to corrupt the policy.
  * Faster convergence (fewer parameters, smaller gradient noise).
  * Decoupled iteration: if the value head's MSE loss isn't
    converging, that's a value-target problem (demo quality) not a
    backbone-feature problem.

Joint policy+value training (Path B) is a follow-up — once we
validate that frozen-backbone value head produces ladder-positive
results.

Run:
    python -m tools.train_value_head \\
        --demos runs/mcts_demos_v6_with_outcomes.npz \\
        --bc-checkpoint runs/bc_warmstart_small_cpu.pt \\
        --out runs/bc_v_v1.pt \\
        --epochs 10 --batch-size 256 --lr 1e-3
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
    """Load a ConvPolicy checkpoint produced by tools/bc_warmstart.py.
    Same logic as nn_prior.load_conv_policy but inlined to avoid the
    extra import-time cost (this script is short-running)."""
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
    model = model.to(device)
    return model, cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True, type=Path,
                    help="path to .npz with x + terminal_value (from collect_mcts_demos.py)")
    ap.add_argument("--bc-checkpoint", required=True, type=Path,
                    help="BC checkpoint to initialize from (frozen)")
    ap.add_argument("--out", required=True, type=Path,
                    help="output checkpoint path")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction held out for val MSE")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load demos.
    print(f"loading demos from {args.demos}", flush=True)
    data = np.load(args.demos)
    if "terminal_value" not in data:
        print(
            f"ERROR: {args.demos} has no 'terminal_value' column. "
            f"Re-collect demos with the updated tools/collect_mcts_demos.py "
            f"that records terminal_value per state.", file=sys.stderr,
        )
        return 1
    x_all = data["x"].astype(np.float32)
    v_all = data["terminal_value"].astype(np.float32)
    n = x_all.shape[0]
    print(
        f"  {n:,} demos  v_mix: "
        f"win={int((v_all > 0.5).sum())} "
        f"loss={int((v_all < -0.5).sum())} "
        f"tie={int(((v_all >= -0.5) & (v_all <= 0.5)).sum())} "
        f"mean={v_all.mean():.3f}",
        flush=True,
    )

    # Train/val split.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = int(args.val_frac * n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"  train={len(train_idx):,}  val={len(val_idx):,}", flush=True)

    # Load model and freeze backbone+policy.
    print(f"loading BC checkpoint from {args.bc_checkpoint}", flush=True)
    model, cfg = _load_bc_checkpoint(args.bc_checkpoint, device)
    print(f"  cfg: backbone={cfg.backbone_channels} blocks={cfg.n_blocks} value_hidden={cfg.value_hidden}", flush=True)

    for name, p in model.named_parameters():
        if name.startswith("value_head."):
            p.requires_grad = True
        else:
            p.requires_grad = False
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total_params = sum(p.numel() for p in model.parameters())
    print(
        f"  trainable params: {n_train_params:,} / {n_total_params:,} "
        f"({100 * n_train_params / max(1, n_total_params):.1f}%)",
        flush=True,
    )

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
    )

    # Move data to device-friendly tensors. For small demo sets we can
    # keep everything in RAM; for large ones a streaming DataLoader is
    # the upgrade. n=11k-200k fits in RAM at fp32 (n × 12 × 50 × 50 × 4
    # bytes = 120k × 120k = 14.4 GB worst case; typical is ~700 MB).
    x_train = torch.from_numpy(x_all[train_idx])
    v_train = torch.from_numpy(v_all[train_idx])
    x_val = torch.from_numpy(x_all[val_idx]).to(device, non_blocking=True)
    v_val = torch.from_numpy(v_all[val_idx]).to(device, non_blocking=True)

    history = {"train_mse": [], "val_mse": [], "wall_seconds": []}
    t_start = time.perf_counter()
    print(f"\ntraining {args.epochs} epochs", flush=True)
    for epoch in range(args.epochs):
        model.train()
        # value_head isn't typically affected by train/eval mode (no
        # dropout/BN), but call it anyway in case future arches add them.
        order = torch.randperm(len(train_idx))
        total_loss = 0.0
        n_seen = 0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            x_b = x_train[idx].to(device, non_blocking=True)
            v_b = v_train[idx].to(device, non_blocking=True)
            _logits, value_pred = model(x_b)
            value_pred = value_pred.squeeze(-1)
            loss = F.mse_loss(value_pred, v_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(idx)
            n_seen += len(idx)
        train_mse = total_loss / max(1, n_seen)

        # Val MSE.
        model.eval()
        with torch.no_grad():
            _logits_v, value_pred_v = model(x_val)
            value_pred_v = value_pred_v.squeeze(-1)
            val_mse = float(F.mse_loss(value_pred_v, v_val).item())

        elapsed = time.perf_counter() - t_start
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["wall_seconds"].append(elapsed)
        print(
            f"  epoch {epoch + 1:>2}/{args.epochs}  "
            f"train_mse={train_mse:.4f}  val_mse={val_mse:.4f}  "
            f"wall={elapsed:.0f}s",
            flush=True,
        )

    # Save updated checkpoint.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict
    torch.save({
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "cfg": asdict(cfg),
        "value_head_trained": True,
        "training_history": history,
        "source_bc_checkpoint": str(args.bc_checkpoint),
        "source_demos": str(args.demos),
    }, args.out)
    print(f"\nsaved {args.out}", flush=True)

    # Sidecar JSON for ease of inspection (no torch dependency to read).
    json_path = args.out.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({
            "cfg": asdict(cfg),
            "training_history": history,
            "n_demos": int(n),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "source_bc_checkpoint": str(args.bc_checkpoint),
            "source_demos": str(args.demos),
        }, f, indent=2)
    print(f"sidecar {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
