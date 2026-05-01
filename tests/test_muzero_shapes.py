"""Shape-contract tests for the MuZero scaffold.

The implementation is a SCAFFOLD (per ``docs/MUZERO_SPEC.md``); it
already runs end-to-end, but training + MCTS integration are not yet
landed. These tests pin the API contracts so subsequent implementation
work doesn't drift.
"""
from __future__ import annotations

import torch

from orbitwars.nn.muzero import (
    DynamicsCfg,
    DynamicsNet,
    MuZeroCfg,
    MuZeroNet,
    PredictionCfg,
    PredictionNet,
    RepresentationCfg,
    RepresentationNet,
)


def test_representation_shape():
    cfg = RepresentationCfg()
    net = RepresentationNet(cfg)
    obs = torch.zeros(2, cfg.n_input_channels, cfg.grid_h, cfg.grid_w)
    latent = net(obs)
    assert latent.shape == (2, cfg.n_latent_channels, cfg.grid_h, cfg.grid_w)


def test_prediction_shape():
    cfg = PredictionCfg()
    net = PredictionNet(cfg)
    latent = torch.zeros(3, cfg.n_latent_channels, 50, 50)
    policy, value = net(latent)
    assert policy.shape == (3, cfg.n_action_channels, 50, 50)
    assert value.shape == (3, 1)
    # tanh range
    assert value.abs().max().item() <= 1.0


def test_dynamics_shape_and_residual_gating():
    cfg = DynamicsCfg()
    net = DynamicsNet(cfg)
    latent = torch.randn(2, cfg.n_latent_channels, 50, 50)
    action_map = torch.zeros(2, cfg.n_action_channels, 50, 50)
    next_latent, reward = net(latent, action_map)
    assert next_latent.shape == latent.shape
    assert reward.shape == (2, 1)
    # Residual gating: at init the dynamics module's output is
    # latent + small_delta. The delta won't be zero but should be
    # bounded.
    delta = (next_latent - latent).abs().mean().item()
    assert 0.0 < delta < 50.0  # finite, non-trivial


def test_muzero_full_initial_inference():
    m = MuZeroNet()
    obs = torch.zeros(4, 12, 50, 50)
    policy, value, latent = m.initial_inference(obs)
    assert policy.shape == (4, 8, 50, 50)
    assert value.shape == (4, 1)
    assert latent.shape == (4, 64, 50, 50)


def test_muzero_full_recurrent_inference():
    m = MuZeroNet()
    latent = torch.randn(2, 64, 50, 50)
    action_map = torch.zeros(2, 8, 50, 50)
    nl, reward, policy, value = m.recurrent_inference(latent, action_map)
    assert nl.shape == latent.shape
    assert reward.shape == (2, 1)
    assert policy.shape == (2, 8, 50, 50)
    assert value.shape == (2, 1)


def test_muzero_param_budget_in_range():
    """Default cfg targets ~1M params (post-distill ship envelope)."""
    m = MuZeroNet()
    n = m.num_params()
    # Sanity: between 500k (too small for AZ Go-style) and 5M (too big
    # to ship inline). Default should sit ~800k-1.2M.
    assert 500_000 < n < 2_500_000, f"unexpected param count {n:,}"
