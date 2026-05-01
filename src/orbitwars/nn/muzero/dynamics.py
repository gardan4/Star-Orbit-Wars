"""Dynamics network ``g(latent, action) -> (next_latent, reward)``.

This is the load-bearing module — it replaces ``engine.step`` in MCTS
rollouts. Quality of search depends on this being a good model of
game dynamics over the rollout horizon (~15 plies).

Action encoding: see ``docs/MUZERO_SPEC.md`` §"Action encoding". The
chosen scheme is **spatial action map**:
  action: (B, n_action_channels, H, W) one-hot at firing-planet cells.
  Concat with latent -> (B, n_latent_channels + n_action_channels, H, W).
  3x3 conv mixes them spatially.

Architecture:
  * stem conv merges latent + action_map
  * N residual blocks (LayerNorm-stabilized for unroll consistency)
  * residual output gated to next_latent (latent + delta, NOT
    overwriting — keeps representation aligned across unroll steps)
  * scalar reward head (avg-pool + small MLP, tanh to [-1, 1])

Default: 64ch x 4 blocks ~ 500k params.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DynamicsCfg:
    n_latent_channels: int = 64
    n_action_channels: int = 8
    n_blocks: int = 4
    reward_hidden: int = 64


class _GroupNormResBlock(nn.Module):
    """Pre-act ResBlock with GroupNorm (matches ConvPolicy.ResBlock)."""

    def __init__(self, c: int, num_groups: int = 8) -> None:
        super().__init__()
        groups = min(num_groups, c)
        self.gn1 = nn.GroupNorm(groups, c)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.gn2 = nn.GroupNorm(groups, c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.relu(self.gn1(x)))
        y = self.conv2(F.relu(self.gn2(y)))
        return x + y


class DynamicsNet(nn.Module):
    """One-step dynamics: ``g(latent_t, action_t) -> (latent_{t+1}, reward_t)``.

    Spec contract:
        in:  latent (B, n_latent_channels, H, W)
             action_map (B, n_action_channels, H, W)
                — per-cell one-hot encoding of "which action fires at
                this planet's grid cell". Cells with no firing planet
                are zero across all action channels.
        out: (next_latent (B, n_latent_channels, H, W),
              reward (B, 1) in [-1, 1] after tanh)

    The "residual gating" pattern (next = latent + delta) is critical
    for multi-step unroll stability — straight overwrite caused mode
    collapse in MuZero ablations.

    Implementation status: STUB. The forward() method is structured;
    fill in delta computation per the paper's recipe (residual conv
    block stack).
    """

    def __init__(self, cfg: DynamicsCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.stem = nn.Conv2d(
            cfg.n_latent_channels + cfg.n_action_channels,
            cfg.n_latent_channels, 3, padding=1,
        )
        self.blocks = nn.ModuleList(
            [_GroupNormResBlock(cfg.n_latent_channels) for _ in range(cfg.n_blocks)]
        )
        self.reward_pool = nn.AdaptiveAvgPool2d(1)
        self.reward_mlp = nn.Sequential(
            nn.Linear(cfg.n_latent_channels, cfg.reward_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.reward_hidden, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        latent: torch.Tensor,
        action_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concat action map onto latent, run residual stack.
        x = torch.cat([latent, action_map], dim=1)
        delta = self.stem(x)
        for block in self.blocks:
            delta = block(delta)
        # Residual gating: next_latent = latent + learned delta.
        # Critical for unroll stability — see MUZERO_SPEC.md.
        next_latent = latent + delta
        # Reward from the next latent via avg-pool + MLP.
        v = self.reward_pool(next_latent).flatten(1)
        reward = self.reward_mlp(v)
        return next_latent, reward
