"""Representation network ``h(obs) -> latent``.

Maps a 12-channel observation grid to a wider latent grid that the
prediction head + dynamics consume. Same spatial dims as the input
(50x50), wider channel count.

Architecture: identical to the existing ``ConvPolicy`` backbone (stem
+ N residual blocks with GroupNorm), but emits the latent BEFORE the
policy/value heads. Default 64ch x 6 blocks ~ 600k params.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the existing ResBlock to keep the latent space layout aligned
# with what the heuristic-trained backbone produces. Lets us warmstart
# the representation from a BC checkpoint.
from orbitwars.nn.conv_policy import ResBlock


@dataclass(frozen=True)
class RepresentationCfg:
    """Hyperparameters for the representation encoder.

    Defaults match a "medium" MuZero scale: ~600k params, fits inline
    after distillation + int8.
    """
    grid_h: int = 50
    grid_w: int = 50
    n_input_channels: int = 12     # matches obs_encode.encode_grid output
    n_latent_channels: int = 64    # latent width
    n_blocks: int = 6


class RepresentationNet(nn.Module):
    """Encode observation grid into a latent grid.

    Spec contract:
        in:  (B, n_input_channels, grid_h, grid_w) float32
        out: (B, n_latent_channels, grid_h, grid_w) float32 — the latent
              that prediction + dynamics operate on.

    Implementation status: STUB. Spec docstring is load-bearing; fill in
    forward() per the architecture mirrored from ConvPolicy.
    """

    def __init__(self, cfg: RepresentationCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.stem = nn.Conv2d(
            cfg.n_input_channels, cfg.n_latent_channels, 3, padding=1,
        )
        self.blocks = nn.ModuleList(
            [ResBlock(cfg.n_latent_channels) for _ in range(cfg.n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard pre-act conv + residual blocks, no head.
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return h
