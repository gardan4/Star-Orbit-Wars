"""Train the policy head of a ConvPolicy on (state, visit_dist) demos.

The complement to ``tools/train_value_head.py``. Where the value head
fits terminal_value via MSE, the policy head fits the MCTS visit
distribution via KL divergence — the standard AlphaZero policy
distillation target.

Why this matters
----------------
``bc_warmstart.py`` trains the policy head to imitate the heuristic's
single-action picks. That bounds the policy at "as good as the
heuristic" in the limit. After Phantom 4/5/6 fixes (2026-04-28),
self-play with real MCTS produces visit distributions that reflect
search-improved decisions — strictly better than the heuristic's
pick where search overrode the anchor. Distilling those visits into
the policy head gives MCTS a stronger PRIOR, which tightens search
on the right candidates and frees rollouts to evaluate good moves
deeper instead of wasting visits on bad ones.

This is the "train policy on visit distributions" half of AlphaZero's
joint loss. Combining it with the value-head training in
``tools/train_value_head.py`` gives a full AlphaZero-style policy
iteration loop. Per the iter-1 result (v33 with frozen-prior + v4
value head lost 0-8 to v32b), we hypothesize that the value head
alone isn't sufficient; the prior's quality bounds search efficiency,
and frozen-BC priors are too weak to surface good candidates at the
top of Gumbel sampling.

Frozen vs joint
---------------
Like train_value_head, this script freezes everything except the
target head. That decouples failure modes:
  * If train_policy_head's KL doesn't go down, that's a demo / target
    problem (bad visit distributions, not enough data).
  * If train_value_head's MSE doesn't go down, that's a value-target
    problem.
  * If both train cleanly but the resulting agent loses H2H, we know
    the issue is in MCTS plumbing, not in any one head's quality.

Joint policy+value training is a follow-up once we validate
frozen-head iteration produces ladder-positive agents.

Run:
    python -m tools.train_policy_head \\
        --demos runs/closed_loop_iter1_postfix/demos_iter1.npz \\
        --bc-checkpoint runs/closed_loop_iter1_postfix/value_head_iter1.pt \\
        --out runs/closed_loop_iter1_postfix/policy_head_iter1.pt \\
        --epochs 30 --batch-size 256 --lr 1e-3
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
    """Same loader as train_value_head.py."""
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
                    help="path to .npz with x + gy + gx + visit_dist (from collect_mcts_demos.py)")
    ap.add_argument("--bc-checkpoint", required=True, type=Path,
                    help="checkpoint to initialize from (frozen except policy_head)")
    ap.add_argument("--out", required=True, type=Path,
                    help="output checkpoint path")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
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
    if "visit_dist" not in data or "gy" not in data or "gx" not in data:
        print(
            f"ERROR: {args.demos} missing visit_dist / gy / gx. Re-collect "
            f"demos with the updated tools/collect_mcts_demos.py.",
            file=sys.stderr,
        )
        return 1
    x_all = data["x"].astype(np.float32)
    gy_all = data["gy"].astype(np.int64)
    gx_all = data["gx"].astype(np.int64)
    visit_all = data["visit_dist"].astype(np.float32)  # (N, 8)
    n = x_all.shape[0]
    # Sanity check: visit_dist rows should be ~ probability simplex.
    row_sums = visit_all.sum(axis=1)
    print(
        f"  {n:,} demos  visit_dist row-sums: "
        f"min={row_sums.min():.3f} mean={row_sums.mean():.3f} max={row_sums.max():.3f} "
        f"(expected ~1.0 by construction in collect_mcts_demos.py)",
        flush=True,
    )

    # Train/val split.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = int(args.val_frac * n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"  train={len(train_idx):,}  val={len(val_idx):,}", flush=True)

    # Load model and freeze everything except policy_head.
    print(f"loading checkpoint from {args.bc_checkpoint}", flush=True)
    model, cfg = _load_bc_checkpoint(args.bc_checkpoint, device)
    print(f"  cfg: backbone={cfg.backbone_channels} blocks={cfg.n_blocks} "
          f"action_channels={cfg.n_action_channels}", flush=True)

    for name, p in model.named_parameters():
        if name.startswith("policy_head."):
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

    # Move data to tensors. Keep x in pinned-host memory and stream
    # (12k × 12 × 50 × 50 × 4 = 1.4 GB at fp32 — borderline for GPU).
    x_train = torch.from_numpy(x_all[train_idx])
    gy_train = torch.from_numpy(gy_all[train_idx])
    gx_train = torch.from_numpy(gx_all[train_idx])
    v_train = torch.from_numpy(visit_all[train_idx])
    x_val = torch.from_numpy(x_all[val_idx]).to(device, non_blocking=True)
    gy_val = torch.from_numpy(gy_all[val_idx]).to(device, non_blocking=True)
    gx_val = torch.from_numpy(gx_all[val_idx]).to(device, non_blocking=True)
    v_val = torch.from_numpy(visit_all[val_idx]).to(device, non_blocking=True)

    def _kl_loss(logits_full, gy_b, gx_b, target_dist):
        """KL(target_dist || softmax(cell_logits)) per demo, averaged.

        logits_full: (B, 8, H, W) — full grid logits.
        gy_b, gx_b: (B,) — source planet's grid cell.
        target_dist: (B, 8) — MCTS visit distribution at that cell.
        """
        B = logits_full.shape[0]
        # Gather the (B, 8) cell-logits at (gy, gx).
        idx_b = torch.arange(B, device=logits_full.device)
        cell_logits = logits_full[idx_b, :, gy_b, gx_b]  # (B, 8)
        log_probs = F.log_softmax(cell_logits, dim=-1)
        # KL(target || pred) = sum_i target * (log target - log pred). Drop
        # the constant entropy term — F.kl_div does this when reduction='batchmean'.
        # Note F.kl_div expects log-probs as input. Add a small eps to target
        # to avoid log(0) when computing entropy externally; KL is finite
        # regardless because target * log(target) -> 0 as target -> 0.
        eps = 1e-9
        target_safe = target_dist + eps
        target_safe = target_safe / target_safe.sum(dim=-1, keepdim=True)
        # Cross-entropy (= KL up to a target-only constant) with soft targets.
        # CE = -sum_i target * log_pred. This is the AlphaZero policy loss.
        ce = -(target_safe * log_probs).sum(dim=-1).mean()
        return ce

    history = {"train_loss": [], "val_loss": [], "wall_seconds": []}
    t_start = time.perf_counter()
    print(f"\ntraining {args.epochs} epochs (cross-entropy on visit_dist)",
          flush=True)
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train_idx))
        total_loss = 0.0
        n_seen = 0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            x_b = x_train[idx].to(device, non_blocking=True)
            gy_b = gy_train[idx].to(device, non_blocking=True)
            gx_b = gx_train[idx].to(device, non_blocking=True)
            v_b = v_train[idx].to(device, non_blocking=True)
            logits_full, _value = model(x_b)
            loss = _kl_loss(logits_full, gy_b, gx_b, v_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(idx)
            n_seen += len(idx)
        train_loss = total_loss / max(1, n_seen)

        # Val loss in batches (the 1.4 GB limit forces this).
        model.eval()
        with torch.no_grad():
            n_v = len(val_idx)
            v_loss_sum = 0.0
            for s in range(0, n_v, args.batch_size):
                e = min(n_v, s + args.batch_size)
                logits_v, _ = model(x_val[s:e])
                v_loss_sum += float(
                    _kl_loss(logits_v, gy_val[s:e], gx_val[s:e], v_val[s:e]).item()
                ) * (e - s)
            val_loss = v_loss_sum / max(1, n_v)

        elapsed = time.perf_counter() - t_start
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["wall_seconds"].append(elapsed)
        print(
            f"  epoch {epoch + 1:>2}/{args.epochs}  "
            f"train_ce={train_loss:.4f}  val_ce={val_loss:.4f}  "
            f"wall={elapsed:.0f}s",
            flush=True,
        )

    # Save updated checkpoint.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict
    torch.save({
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "cfg": asdict(cfg),
        "policy_head_trained_on_visits": True,
        "training_history": history,
        "source_bc_checkpoint": str(args.bc_checkpoint),
        "source_demos": str(args.demos),
    }, args.out)
    print(f"\nsaved {args.out}", flush=True)

    # Sidecar JSON.
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
