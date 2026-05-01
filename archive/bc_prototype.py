"""Behavior-cloning prototype for the W4 conv policy.

**Goal**: end-to-end sanity check that the plumbing between
``features/obs_encode.encode_grid`` and ``nn/conv_policy.ConvPolicy``
works, and that a tiny conv can learn a non-trivial signal from
heuristic demonstrations.

This is **not** the full W4 BC warm-start — that happens on GPU with a
much larger model and dataset. This is a 5-minute-CPU sanity run that
tells us: does the gradient flow, does the label space match, is the
per-cell gather indexing correct? The gate is "accuracy > random
(12.5% for 8 classes) after a few epochs".

Pipeline:
  1. Play N self-play games (HeuristicAgent vs HeuristicAgent) with
     FastEngine. Seeds 0..N-1.
  2. Every turn, for each owned planet that the heuristic launched from,
     record a training example:
       - input: ``encode_grid(obs, player_id)``  (C, H, W) float32
       - location: (gy, gx) grid cell of the launching planet
       - label: ACTION_LOOKUP index (0..7) that best matches the
         heuristic's chosen (angle, ship_fraction).
  3. Train a *tiny* conv (4 channels, 2 blocks — ~50k params) for
     K epochs with cross-entropy loss gathered at the (gy, gx) cells.
  4. Report per-epoch train accuracy.

A full training run will use the default ConvPolicyCfg (~460k params)
on GPU with 100x more data and MCTS-search targets rather than raw
heuristic labels. This script stays CPU-tractable.

Run:
    python tools/bc_prototype.py --games 40 --epochs 8

Expected: accuracy climbs from ~12.5% (random) to 40%+ within 5 epochs.
If it stalls at 12.5%, something is wrong upstream — probably the
label mapping or the grid-cell lookup.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from orbitwars.bots.base import Deadline, obs_get
from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.engine.fast_engine import FastEngine
from orbitwars.features.obs_encode import encode_grid
from orbitwars.nn.conv_policy import (
    ACTION_LOOKUP,
    ConvPolicy,
    ConvPolicyCfg,
    planet_to_grid_coords,
)

# Center angle of each of the 4 buckets. 0=East, pi/2=North, pi=West,
# 3pi/2=South. We wrap to pick the closest bucket below; this is not a
# magic constant but a direct read of ACTION_LOOKUP.
ANGLE_BUCKET_CENTERS = (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)


def _angle_to_bucket(angle: float) -> int:
    """Snap a continuous angle in radians to the nearest of 4 cardinal
    buckets. Handles wraparound correctly (e.g. 5.8 rad snaps to East).
    """
    # Normalize to [0, 2pi)
    a = angle % (2.0 * math.pi)
    # Compute circular distance to each bucket center.
    best_i = 0
    best_d = float("inf")
    for i, c in enumerate(ANGLE_BUCKET_CENTERS):
        d = abs(a - c)
        d = min(d, 2.0 * math.pi - d)  # wraparound
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _frac_to_bucket(frac: float) -> int:
    """Snap a ship fraction in [0, 1] to 0 (50%) or 1 (100%).

    Threshold at 0.75 so moderate launches (60-70%) count as "50%" and
    all-in launches count as "100%". Matches ACTION_LOOKUP's two
    ship-fraction anchors.
    """
    return 0 if frac < 0.75 else 1


@dataclass
class Demo:
    """One (state, location, label) training example."""

    grid: np.ndarray   # (C, H, W) float32
    gy: int
    gx: int
    label: int         # 0..7 (ACTION_LOOKUP index)


def collect_demos(
    n_games: int,
    cfg: ConvPolicyCfg,
    step_timeout: float = 1.0,
    verbose: bool = True,
) -> List[Demo]:
    """Play n_games self-play matches and extract per-launch demos.

    Both seats are HeuristicAgent so we get fully-in-distribution demos
    (no off-policy correction issues for a sanity check). Each turn
    each seat may emit multiple launches; each launch becomes one Demo
    record from that seat's perspective.

    Note: ``Deadline()`` takes an *absolute* perf_counter time when
    given an argument; passing nothing uses the default
    ``SEARCH_DEADLINE_MS`` elapsed-time path — which is what we want.
    The ``step_timeout`` arg is kept for signature compat with other
    tools but is unused here.
    """
    del step_timeout  # Reserved for future per-turn budgeting.
    demos: List[Demo] = []
    t0 = time.perf_counter()
    for game_seed in range(n_games):
        random.seed(game_seed)
        h0 = HeuristicAgent()
        h1 = HeuristicAgent()
        fe = FastEngine.from_scratch(num_agents=2, seed=game_seed)
        # Use the inner .act(obs, deadline) path directly so we don't
        # pay the as_kaggle_agent() wrapper cost (it's the same logic).
        for _ in range(500):
            if fe.done:
                break
            obs0 = fe.observation(0)
            obs1 = fe.observation(1)
            moves0 = h0.act(obs0, Deadline())
            moves1 = h1.act(obs1, Deadline())

            # Collect demos from both seats.
            for (obs, moves, player_id) in (
                (obs0, moves0, 0),
                (obs1, moves1, 1),
            ):
                if not moves:
                    continue
                # Index planets by id for fast lookup.
                planets = obs_get(obs, "planets", [])
                planet_by_id = {int(p[0]): p for p in planets}
                grid = encode_grid(obs, player_id=player_id, cfg=cfg)
                for mv in moves:
                    pid = int(mv[0])
                    angle = float(mv[1])
                    ships_sent = int(mv[2])
                    p = planet_by_id.get(pid)
                    if p is None:
                        continue
                    px, py = float(p[2]), float(p[3])
                    ships_avail = float(p[5])
                    if ships_avail <= 0:
                        continue
                    gy, gx = planet_to_grid_coords(px, py, cfg)
                    ab = _angle_to_bucket(angle)
                    fb = _frac_to_bucket(ships_sent / ships_avail)
                    label = ab * 2 + fb
                    demos.append(Demo(grid=grid, gy=gy, gx=gx, label=label))

            fe.step([moves0, moves1])

        if verbose and (game_seed + 1) % 10 == 0:
            dt = time.perf_counter() - t0
            print(
                f"  game {game_seed + 1}/{n_games}  demos={len(demos)}  "
                f"wall={dt:.0f}s"
            )
    return demos


def build_tensors(
    demos: List[Demo], cfg: ConvPolicyCfg
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert Demo list -> contiguous tensors for batched training."""
    n = len(demos)
    x = np.empty((n, cfg.n_channels, cfg.grid_h, cfg.grid_w), dtype=np.float32)
    gy = np.empty(n, dtype=np.int64)
    gx = np.empty(n, dtype=np.int64)
    labels = np.empty(n, dtype=np.int64)
    for i, d in enumerate(demos):
        x[i] = d.grid
        gy[i] = d.gy
        gx[i] = d.gx
        labels[i] = d.label
    return (
        torch.from_numpy(x),
        torch.from_numpy(gy),
        torch.from_numpy(gx),
        torch.from_numpy(labels),
    )


def train(
    model: nn.Module,
    x: torch.Tensor,
    gy: torch.Tensor,
    gx: torch.Tensor,
    labels: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    verbose: bool = True,
) -> List[float]:
    """Minimal training loop. Returns per-epoch accuracy."""
    n = x.shape[0]
    opt = optim.Adam(model.parameters(), lr=lr)
    accs: List[float] = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            x_b = x[idx]
            gy_b = gy[idx]
            gx_b = gx[idx]
            y_b = labels[idx]
            # (B, n_action_channels, H, W)
            policy_logits, _value = model(x_b)
            b = x_b.shape[0]
            # Gather per-cell logits: output[i, :, gy[i], gx[i]]
            per_cell = policy_logits[
                torch.arange(b), :, gy_b, gx_b
            ]  # (B, n_action_channels)
            loss = F.cross_entropy(per_cell, y_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * b
            total_correct += (per_cell.argmax(dim=-1) == y_b).sum().item()
            total_seen += b
        acc = total_correct / max(total_seen, 1)
        accs.append(acc)
        if verbose:
            print(
                f"  epoch {epoch + 1:>2}/{epochs}  "
                f"loss={total_loss / max(total_seen, 1):.4f}  "
                f"acc={acc:.3f}"
            )
    return accs


def main() -> int:
    ap = argparse.ArgumentParser(description="W4 BC prototype (CPU)")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--step-timeout", type=float, default=1.0)
    # Tiny model for CPU tractability — not the default ConvPolicyCfg.
    ap.add_argument("--backbone-channels", type=int, default=32)
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = ConvPolicyCfg(
        backbone_channels=args.backbone_channels,
        n_blocks=args.n_blocks,
    )
    print(f"=== BC prototype ===")
    print(
        f"cfg: backbone_channels={cfg.backbone_channels}  "
        f"n_blocks={cfg.n_blocks}  n_action_channels={cfg.n_action_channels}"
    )

    t0 = time.perf_counter()
    demos = collect_demos(args.games, cfg, step_timeout=args.step_timeout)
    dt_collect = time.perf_counter() - t0
    print(
        f"collected {len(demos):,} demos from {args.games} games "
        f"in {dt_collect:.0f}s"
    )

    if not demos:
        print("No demos collected — heuristic never launched. Abort.")
        return 1

    # Class distribution — useful sanity on label mapping.
    labels_np = np.asarray([d.label for d in demos])
    counts = np.bincount(labels_np, minlength=len(ACTION_LOOKUP))
    print("label histogram:")
    for i, (ab, fr) in enumerate(ACTION_LOOKUP):
        share = counts[i] / len(demos)
        print(
            f"  {i}: bucket={ab} frac={fr:<4}  "
            f"n={counts[i]:<6} ({share:.1%})"
        )
    majority_class_acc = float(counts.max()) / len(demos)
    print(f"majority-class baseline accuracy: {majority_class_acc:.3f}")
    print(f"random-guess baseline: {1.0 / len(ACTION_LOOKUP):.3f}")

    x, gy, gx, labels = build_tensors(demos, cfg)
    print(f"tensors: x={tuple(x.shape)}  labels={tuple(labels.shape)}")

    model = ConvPolicy(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    t0 = time.perf_counter()
    accs = train(
        model, x, gy, gx, labels,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )
    dt_train = time.perf_counter() - t0
    print(f"train wall: {dt_train:.0f}s")
    print(f"final acc: {accs[-1]:.3f}  best acc: {max(accs):.3f}")

    # Gate: accuracy must beat random-guess (1/8 = 0.125) by a wide
    # margin. Majority-class beats random too, so the real check is
    # whether the model learns BEYOND the class-prior — only possible if
    # the grid encoding + gather is wired correctly.
    if max(accs) <= majority_class_acc + 0.02:
        print(
            "WARN: model did not beat majority-class baseline. Could be "
            "label imbalance or a plumbing bug (check grid indexing)."
        )
        return 2
    print("PASS: model beats majority-class baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
