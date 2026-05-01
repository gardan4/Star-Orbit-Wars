"""Composed MuZero model — entry points used by MCTS.

MCTS interacts with the model through two operations:

* ``initial_inference(obs) -> (policy, value, latent)``
  Called at the search root. Encodes the real observation into latent
  space and produces root-level policy + value. The latent is cached
  for ``recurrent_inference`` calls during rollouts.

* ``recurrent_inference(latent, action_map) -> (next_latent, reward, policy, value)``
  Called at every rollout step. Runs dynamics + prediction in one go.

The split mirrors the official MuZero pseudocode and lets the search
loop stay agnostic to the specific representation/dynamics
architectures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.nn as nn

from orbitwars.nn.muzero.dynamics import DynamicsCfg, DynamicsNet
from orbitwars.nn.muzero.prediction import PredictionCfg, PredictionNet
from orbitwars.nn.muzero.representation import RepresentationCfg, RepresentationNet


@dataclass(frozen=True)
class MuZeroCfg:
    """Top-level MuZero hyperparameters. Defaults are a sensible
    starting point; tune via overrides at training time."""
    repr: RepresentationCfg = field(default_factory=RepresentationCfg)
    pred: PredictionCfg = field(default_factory=PredictionCfg)
    dyn: DynamicsCfg = field(default_factory=DynamicsCfg)
    discount: float = 0.997  # per-ply discount applied during search


class MuZeroNet(nn.Module):
    """Full MuZero model = representation + prediction + dynamics.

    Compatible with the standard joint-training loss; see
    ``tools/cloud/train_dynamics.py`` for the recipe.

    Implementation status: SCAFFOLD. Each submodule is real; the joint
    training loop must be added in ``train_dynamics.py``.
    """

    def __init__(self, cfg: MuZeroCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MuZeroCfg()
        self.representation = RepresentationNet(self.cfg.repr)
        self.prediction = PredictionNet(self.cfg.pred)
        self.dynamics = DynamicsNet(self.cfg.dyn)

    # ---- MCTS entry points ----------------------------------------------

    def initial_inference(
        self, obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Root-level inference. Returns (policy_logits, value, latent)."""
        latent = self.representation(obs)
        policy_logits, value = self.prediction(latent)
        return policy_logits, value, latent

    def recurrent_inference(
        self, latent: torch.Tensor, action_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One MCTS rollout step.

        Returns (next_latent, reward, policy_logits, value).
        """
        next_latent, reward = self.dynamics(latent, action_map)
        policy_logits, value = self.prediction(next_latent)
        return next_latent, reward, policy_logits, value

    # ---- training utilities ---------------------------------------------

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
