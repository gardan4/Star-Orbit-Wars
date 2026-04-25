r"""Full-size behavior-cloning warm-start training on GPU.

**Goal**: produce a `runs/bc_warmstart.pt` checkpoint that W5 PPO can load
as initialization so self-play doesn't have to discover basic "launch at
enemies" from random weights.

Differences from `tools/bc_prototype.py`:
  * Full-size `ConvPolicyCfg` (~460 K params) instead of the tiny 45 K
    plumbing model.
  * Runs on CUDA by default (falls back to CPU with a warning).
  * Train / val split with best-val checkpointing (not last-epoch).
  * Demo cache: collected demos are pickled to `runs/bc_demos.npz` so
    subsequent runs skip the self-play step. Pass `--regen` to force
    re-collection (e.g. after a heuristic change).
  * Saves checkpoint to `runs/bc_warmstart.pt` with a header dict
    capturing config, demo hash, accuracy curve, and torch version
    so we can audit what went into a particular student.

Run (GPU venv):
    .\.venv-gpu\Scripts\python.exe -m tools.bc_warmstart \
        --games 60 --epochs 15 --batch-size 256

Run (CPU fallback — slow; for smoke only):
    .\.venv\Scripts\python.exe -m tools.bc_warmstart \
        --games 10 --epochs 3 --device cpu

Outputs:
  runs/bc_demos.npz    — (x, gy, gx, labels) cache across runs
  runs/bc_warmstart.pt — best-val checkpoint + metadata
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

# Reuse the prototype's demo-collection + label logic so we don't fork
# two slightly-different encoders (bugs would diverge silently).
from tools.bc_prototype import (  # type: ignore[import-not-found]
    ANGLE_BUCKET_CENTERS,
    Demo,
    _angle_to_bucket,
    _frac_to_bucket,
    build_tensors,
    collect_demos,
)

from orbitwars.nn.conv_policy import ACTION_LOOKUP, ConvPolicy, ConvPolicyCfg


def _resolve_device(arg: str) -> torch.device:
    """Resolve --device arg to a torch.device, printing what we chose.

    'auto' prefers CUDA if available. Any explicit value is honored
    verbatim so a CPU-only run on a GPU machine is possible for
    apples-to-apples benchmarks.
    """
    if arg == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            print(f"device: cuda ({torch.cuda.get_device_name(0)})", flush=True)
        else:
            dev = torch.device("cpu")
            print(
                "device: cpu (CUDA not available — "
                "this will be slow; pass --device cpu explicitly to suppress "
                "this warning)",
                flush=True,
            )
    else:
        dev = torch.device(arg)
        if arg == "cuda" and not torch.cuda.is_available():
            print(
                "ERROR: --device cuda requested but torch.cuda.is_available() "
                "returned False. Install CUDA torch: "
                "pip install torch --index-url "
                "https://download.pytorch.org/whl/cu124",
                flush=True,
            )
            sys.exit(2)
        print(f"device: {arg}", flush=True)
    return dev


def _demos_hash(x: np.ndarray, labels: np.ndarray) -> str:
    """Short hash over demo tensors so checkpoints record *which* demos
    they were trained on. Used for provenance, not correctness."""
    h = hashlib.sha256()
    h.update(x.shape.__str__().encode())
    h.update(x[:16].tobytes())  # just a sample — full hash would be slow
    h.update(labels.shape.__str__().encode())
    h.update(labels[:256].tobytes())
    return h.hexdigest()[:16]


def _load_or_collect_demos(
    cache_path: Path,
    games: int,
    cfg: ConvPolicyCfg,
    regen: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, gy, gx, labels) as numpy arrays, from cache if possible.

    Cache key isn't perfect — it doesn't notice if the heuristic or
    feature encoder changed between runs. Callers flag `--regen` after
    meaningful code changes. We do cheap shape sanity:
      * cached n_channels must equal cfg.n_channels
      * cached grid shape must equal (cfg.grid_h, cfg.grid_w)
    If those drift, re-collect automatically.
    """
    if cache_path.exists() and not regen:
        data = np.load(cache_path)
        x = data["x"]
        gy = data["gy"]
        gx = data["gx"]
        labels = data["labels"]
        shape_ok = (
            x.shape[1] == cfg.n_channels
            and x.shape[2] == cfg.grid_h
            and x.shape[3] == cfg.grid_w
        )
        if shape_ok:
            print(
                f"loaded {len(labels):,} cached demos from {cache_path} "
                f"(x={tuple(x.shape)})",
                flush=True,
            )
            return x, gy, gx, labels
        print(
            f"cache at {cache_path} has wrong shape "
            f"(x={tuple(x.shape)} vs cfg {cfg.n_channels}x"
            f"{cfg.grid_h}x{cfg.grid_w}) — re-collecting",
            flush=True,
        )

    print(f"collecting demos: {games} self-play games", flush=True)
    demos: List[Demo] = collect_demos(games, cfg, verbose=True)
    if not demos:
        raise RuntimeError("no demos collected — heuristic never launched?")
    x_t, gy_t, gx_t, labels_t = build_tensors(demos, cfg)
    x = x_t.numpy()
    gy = gy_t.numpy()
    gx = gx_t.numpy()
    labels = labels_t.numpy()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=x, gy=gy, gx=gx, labels=labels)
    print(f"cached {len(labels):,} demos to {cache_path}", flush=True)
    return x, gy, gx, labels


def _train_val_split(
    n: int, val_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Random permutation split; `val_frac` in [0, 0.5]."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    return perm[n_val:], perm[:n_val]


def _eval(
    model: nn.Module,
    x: torch.Tensor,
    gy: torch.Tensor,
    gx: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> Tuple[float, float]:
    """Return (loss, acc) — no grad, eval mode."""
    model.eval()
    n = x.shape[0]
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            x_b = x[start : start + batch_size]
            gy_b = gy[start : start + batch_size]
            gx_b = gx[start : start + batch_size]
            y_b = labels[start : start + batch_size]
            logits, _ = model(x_b)
            b = x_b.shape[0]
            per_cell = logits[torch.arange(b, device=logits.device), :, gy_b, gx_b]
            loss = F.cross_entropy(per_cell, y_b, reduction="sum")
            total_loss += loss.item()
            total_correct += (per_cell.argmax(dim=-1) == y_b).sum().item()
    return total_loss / max(1, n), total_correct / max(1, n)


def train(
    model: nn.Module,
    x: torch.Tensor,
    gy: torch.Tensor,
    gx: torch.Tensor,
    labels: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    verbose: bool = True,
    eager_save_path: Optional[Path] = None,
    eager_save_cfg: Optional[dict] = None,
) -> dict:
    """Train with AdamW + cosine LR decay; return curve dict.

    Keeps the best-val model weights on CPU so the caller can save the
    actually-best checkpoint rather than just the last-epoch state.

    If ``eager_save_path`` is given, the model state_dict is saved at
    every val-acc improvement so a mid-run crash still leaves a usable
    checkpoint on disk. Provenance is filled in by the caller after
    training completes.
    """
    device = next(model.parameters()).device
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Cosine from `lr` down to `lr * 0.01` over total epochs.
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * 0.01,
    )

    x_tr, gy_tr, gx_tr, y_tr = x[train_idx], gy[train_idx], gx[train_idx], labels[train_idx]
    x_va, gy_va, gx_va, y_va = x[val_idx],   gy[val_idx],   gx[val_idx],   labels[val_idx]

    curve = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = -math.inf
    best_state = None

    n_tr = x_tr.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        total_loss = 0.0
        total_correct = 0
        for start in range(0, n_tr, batch_size):
            idx = perm[start : start + batch_size]
            x_b = x_tr[idx]
            gy_b = gy_tr[idx]
            gx_b = gx_tr[idx]
            y_b = y_tr[idx]
            logits, _value = model(x_b)
            b = x_b.shape[0]
            per_cell = logits[torch.arange(b, device=device), :, gy_b, gx_b]
            loss = F.cross_entropy(per_cell, y_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * b
            total_correct += (per_cell.argmax(dim=-1) == y_b).sum().item()

        tr_loss = total_loss / max(1, n_tr)
        tr_acc = total_correct / max(1, n_tr)
        va_loss, va_acc = _eval(model, x_va, gy_va, gx_va, y_va, batch_size)

        curve["train_loss"].append(tr_loss)
        curve["train_acc"].append(tr_acc)
        curve["val_loss"].append(va_loss)
        curve["val_acc"].append(va_acc)
        curve["lr"].append(opt.param_groups[0]["lr"])

        is_best = va_acc > best_val_acc
        if is_best:
            best_val_acc = va_acc
            # CPU copy so we don't waste GPU RAM tracking it.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if eager_save_path is not None:
                eager_save_path.parent.mkdir(parents=True, exist_ok=True)
                # Sidecar-only state_dict; full provenance is written after
                # the loop. This is the crash-safety fallback. We include
                # cfg so the loader can reconstruct the model with the
                # right backbone — without it, partial checkpoints with
                # non-default --backbone-channels / --n-blocks fail to
                # load. Save in BOTH the partial-format key
                # ('model_state_dict') AND the full-format keys
                # ('model_state' + 'cfg') so the loader's full-format
                # branch wins, ensuring the cfg is honored.
                eager_dict: dict = {
                    "model_state_dict": best_state,  # legacy key
                    "best_val_acc": float(best_val_acc),
                    "epoch": int(epoch + 1),
                    "_partial": True,
                }
                if eager_save_cfg is not None:
                    eager_dict["model_state"] = best_state  # full-format key
                    eager_dict["cfg"] = eager_save_cfg
                torch.save(eager_dict, eager_save_path)

        if verbose:
            mark = " *" if is_best else "  "
            print(
                f"  epoch {epoch + 1:>2}/{epochs}{mark} "
                f"lr={opt.param_groups[0]['lr']:.2e}  "
                f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f}  "
                f"va_loss={va_loss:.4f} va_acc={va_acc:.3f}  "
                f"best_va={best_val_acc:.3f}",
                flush=True,
            )
        sched.step()

    curve["best_val_acc"] = best_val_acc
    curve["best_state"] = best_state
    return curve


def main() -> int:
    ap = argparse.ArgumentParser(description="Full-size BC warm-start")
    ap.add_argument("--games", type=int, default=60,
                    help="Self-play games for demo collection (when cache miss)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="Validation split fraction")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--cache-path", type=str, default="runs/bc_demos.npz")
    ap.add_argument("--regen", action="store_true",
                    help="Force re-collection of demos even if cache exists")
    ap.add_argument("--out", type=str, default="runs/bc_warmstart.pt",
                    help="Output checkpoint path")
    ap.add_argument("--seed", type=int, default=0)
    # Full-size ConvPolicyCfg defaults (don't override unless benchmarking).
    ap.add_argument("--backbone-channels", type=int, default=None,
                    help="Default = ConvPolicyCfg default")
    ap.add_argument("--n-blocks", type=int, default=None,
                    help="Default = ConvPolicyCfg default")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _resolve_device(args.device)

    # Build cfg (full-size default unless user overrides).
    cfg_kwargs = {}
    if args.backbone_channels is not None:
        cfg_kwargs["backbone_channels"] = args.backbone_channels
    if args.n_blocks is not None:
        cfg_kwargs["n_blocks"] = args.n_blocks
    cfg = ConvPolicyCfg(**cfg_kwargs)

    print(f"=== BC warm-start ===", flush=True)
    print(
        f"cfg: backbone_channels={cfg.backbone_channels}  "
        f"n_blocks={cfg.n_blocks}  n_action_channels={cfg.n_action_channels}",
        flush=True,
    )

    # Load or collect demos.
    cache_path = Path(args.cache_path)
    x_np, gy_np, gx_np, labels_np = _load_or_collect_demos(
        cache_path, args.games, cfg, regen=args.regen,
    )
    demo_hash = _demos_hash(x_np, labels_np)
    n = len(labels_np)

    # Class distribution sanity.
    counts = np.bincount(labels_np, minlength=len(ACTION_LOOKUP))
    print("label histogram:", flush=True)
    for i, (ab, fr) in enumerate(ACTION_LOOKUP):
        print(
            f"  {i}: bucket={ab} frac={fr:<4}  "
            f"n={counts[i]:<6} ({counts[i] / n:.1%})",
            flush=True,
        )
    majority_class_acc = float(counts.max()) / n
    print(
        f"majority-class baseline: {majority_class_acc:.3f}  "
        f"random-guess baseline: {1.0 / len(ACTION_LOOKUP):.3f}",
        flush=True,
    )

    # Move to device and split.
    x = torch.from_numpy(x_np).to(device)
    gy = torch.from_numpy(gy_np).to(device)
    gx = torch.from_numpy(gx_np).to(device)
    labels = torch.from_numpy(labels_np).to(device)
    train_idx, val_idx = _train_val_split(n, args.val_frac, args.seed)
    print(
        f"tensors: x={tuple(x.shape)}  labels={tuple(labels.shape)}  "
        f"device={x.device}  train={len(train_idx):,} val={len(val_idx):,}",
        flush=True,
    )

    # Build model.
    model = ConvPolicy(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}", flush=True)

    # Train. Pass the final out_path as eager-save target so partial
    # checkpoints land on disk at every val-acc improvement (crash-safety).
    # Also pass cfg so the partial checkpoint carries enough info for
    # nn_prior.load_conv_policy to reconstruct the model with the right
    # backbone (otherwise non-default --backbone-channels / --n-blocks
    # checkpoints can't be loaded).
    t0 = time.perf_counter()
    curve = train(
        model, x, gy, gx, labels, train_idx, val_idx,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        eager_save_path=Path(args.out),
        eager_save_cfg=asdict(cfg),
    )
    dt_train = time.perf_counter() - t0
    print(
        f"train wall: {dt_train:.0f}s  "
        f"best_val_acc: {curve['best_val_acc']:.3f}",
        flush=True,
    )

    # Save best-val checkpoint with provenance.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_state = curve.pop("best_state")
    checkpoint = {
        "model_state": best_state,
        "cfg": asdict(cfg),
        "curve": curve,                       # train/val loss + acc per epoch
        "demo_hash": demo_hash,
        "n_demos": int(n),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "seed": args.seed,
        "hparams": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "val_frac": args.val_frac,
        },
    }
    torch.save(checkpoint, out_path)
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)", flush=True)

    # Also write a sidecar JSON with just the curve for easy plotting.
    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(json.dumps(
        {
            "cfg": asdict(cfg),
            "curve": curve,
            "n_demos": int(n),
            "hparams": checkpoint["hparams"],
            "best_val_acc": curve["best_val_acc"],
        },
        indent=2,
    ))
    print(f"wrote {sidecar}", flush=True)

    # Gate — the W4 plan calls for "usefully guide MCTS", operationalized
    # here as "≥ random + 2×". With 8 classes that's ≥ 0.25. Majority
    # class is typically ~0.14, so this is a low bar; a real warm-start
    # should hit 0.5+ on val.
    random_guess = 1.0 / len(ACTION_LOOKUP)
    gate = max(2 * random_guess, majority_class_acc + 0.05)
    if curve["best_val_acc"] < gate:
        print(
            f"WARN: best_val_acc {curve['best_val_acc']:.3f} below gate "
            f"{gate:.3f} — PPO warm-start is unlikely to help. "
            f"Check plumbing or increase --games.",
            flush=True,
        )
        return 2
    print(
        f"PASS: best_val_acc {curve['best_val_acc']:.3f} ≥ gate {gate:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
