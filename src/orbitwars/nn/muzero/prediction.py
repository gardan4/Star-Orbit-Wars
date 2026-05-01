"""Prediction network ``f(latent) -> (policy_logits, value)``.

Two-head MLP/conv on top of the latent: per-cell action logits (8
channels, mirroring ACTION_LOOKUP) + global state value (scalar).

Used at:
  * Search root (initial_inference): h(obs) -> latent -> (policy, value).
    Policy biases the candidate-joint sampling; value bootstraps the
    Q-estimate (no rollout needed for the root itself).
  * Every leaf in MuZero rollouts (recurrent_inference): the dynamics
    module produces a sequence of latents; we run prediction on each
    to get rollout-internal value estimates.

Architecture: 1x1 conv head for policy (matches ConvPolicy) + global
average pool + 2-layer MLP for value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PredictionCfg:
    n_latent_channels: int = 64
    n_action_channels: int = 8
    value_hidden: int = 128


class PredictionNet(nn.Module):
    """Two-head readout from latent.

    Spec contract:
        in:  latent (B, n_latent_channels, H, W)
        out: (policy_logits (B, n_action_channels, H, W),
              value (B, 1) in [-1, 1] after tanh)

    Implementation status: STUB. Heads as described in PredictionCfg.
    Mirror ``ConvPolicy``'s policy + value heads byte-for-byte so a
    trained ConvPolicy ckpt can warmstart this module.
    """

    def __init__(self, cfg: PredictionCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.policy_head = nn.Conv2d(
            cfg.n_latent_channels, cfg.n_action_channels, 1
        )
        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value_mlp = nn.Sequential(
            nn.Linear(cfg.n_latent_channels, cfg.value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.value_hidden, 1),
            nn.Tanh(),
        )

    def forward(
        self, latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        policy_logits = self.policy_head(latent)
        v = self.value_pool(latent).flatten(1)  # (B, C)
        value = self.value_mlp(v)               # (B, 1)
        return policy_logits, value
