"""Forward-pass smoke tests for the W4 torch policies.

Each test pins a narrow contract so a future refactor that breaks the
training plumbing fails here *before* burning compute on a bad run:

  * **Shape + dtype**: output tensors have the shapes the PPO loop and
    MCTS leaf-eval will index into.
  * **Finiteness**: no NaN/inf from uninitialized paths (e.g. masks
    accidentally set to all-False, division-by-zero in masked pool).
  * **Value range**: tanh on the value head is not accidentally removed.
  * **Parameter count**: actual ``.numel()`` sum is within 20% of the
    static estimate used in the plan's <2M-param gate.
  * **Determinism** (eval mode): same input twice → same output (no
    stray dropout left on).
  * **End-to-end with obs_encode**: encode a real mid-game FastEngine
    observation and push it through both models.

These tests instantiate the models on CPU. The GPU build switch is
`pip install torch --index-url https://download.pytorch.org/whl/cu121`
whenever we're ready to train — nothing in this file assumes a device.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from orbitwars.engine.fast_engine import FastEngine
from orbitwars.features.obs_encode import encode_entities, encode_grid
from orbitwars.nn.conv_policy import (
    ConvPolicy,
    ConvPolicyCfg,
    param_count_estimate as conv_param_estimate,
)
from orbitwars.nn.set_transformer import (
    SetTransformerCfg,
    SetTransformerPolicy,
    entity_feature_schema,
    param_count_estimate as set_param_estimate,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mid_game_obs():
    """A mid-game observation pair (player 0 and player 1 POV)."""
    random.seed(42)
    from orbitwars.bots.heuristic import HeuristicAgent

    h0 = HeuristicAgent().as_kaggle_agent()
    h1 = HeuristicAgent().as_kaggle_agent()
    fe = FastEngine.from_scratch(num_agents=2, seed=42)
    for _ in range(100):
        obs0 = fe.observation(0)
        obs1 = fe.observation(1)
        fe.step([h0(obs0), h1(obs1)])
        if fe.done:
            break
    return fe.observation(0), fe.observation(1)


# ---------------------------------------------------------------------------
# ConvPolicy
# ---------------------------------------------------------------------------


def test_conv_policy_forward_shapes():
    cfg = ConvPolicyCfg()
    model = ConvPolicy(cfg).eval()
    x = torch.zeros(2, cfg.n_channels, cfg.grid_h, cfg.grid_w)
    with torch.no_grad():
        policy, value = model(x)
    assert policy.shape == (2, cfg.n_action_channels, cfg.grid_h, cfg.grid_w)
    assert value.shape == (2, 1)
    assert policy.dtype == torch.float32
    assert value.dtype == torch.float32


def test_conv_policy_no_nan():
    cfg = ConvPolicyCfg()
    model = ConvPolicy(cfg).eval()
    # Non-zero input exercises the residual path + GroupNorm non-trivially.
    x = torch.randn(2, cfg.n_channels, cfg.grid_h, cfg.grid_w)
    with torch.no_grad():
        policy, value = model(x)
    assert torch.isfinite(policy).all()
    assert torch.isfinite(value).all()


def test_conv_policy_value_in_tanh_range():
    cfg = ConvPolicyCfg()
    model = ConvPolicy(cfg).eval()
    x = torch.randn(4, cfg.n_channels, cfg.grid_h, cfg.grid_w) * 10.0
    with torch.no_grad():
        _policy, value = model(x)
    assert value.abs().max().item() <= 1.0 + 1e-6, (
        "value head missing tanh — will break MCTS value bootstrap"
    )


def test_conv_policy_param_count_within_20_percent():
    cfg = ConvPolicyCfg()
    model = ConvPolicy(cfg)
    actual = sum(p.numel() for p in model.parameters())
    est = conv_param_estimate(cfg)
    # Estimate is rough (~10% slack for biases/norms); allow 20% either way.
    ratio = actual / est
    assert 0.8 <= ratio <= 1.2, (
        f"conv param estimate drifted: actual={actual:,} est={est:,} ratio={ratio:.2f}"
    )


def test_conv_policy_deterministic_in_eval_mode():
    cfg = ConvPolicyCfg()
    model = ConvPolicy(cfg).eval()
    x = torch.randn(1, cfg.n_channels, cfg.grid_h, cfg.grid_w)
    with torch.no_grad():
        out_a = model(x)
        out_b = model(x)
    torch.testing.assert_close(out_a[0], out_b[0])
    torch.testing.assert_close(out_a[1], out_b[1])


def test_conv_policy_from_real_observation(mid_game_obs):
    """End-to-end: FastEngine → encode_grid → ConvPolicy.forward."""
    obs, _ = mid_game_obs
    cfg = ConvPolicyCfg()
    grid_np = encode_grid(obs, player_id=0, cfg=cfg)
    assert grid_np.shape == (cfg.n_channels, cfg.grid_h, cfg.grid_w)
    x = torch.from_numpy(grid_np).unsqueeze(0)  # (1, C, H, W)
    model = ConvPolicy(cfg).eval()
    with torch.no_grad():
        policy, value = model(x)
    assert policy.shape == (1, cfg.n_action_channels, cfg.grid_h, cfg.grid_w)
    assert value.shape == (1, 1)
    assert torch.isfinite(policy).all()
    assert torch.isfinite(value).all()


# ---------------------------------------------------------------------------
# SetTransformerPolicy
# ---------------------------------------------------------------------------


def test_set_transformer_forward_shapes():
    cfg = SetTransformerCfg()
    model = SetTransformerPolicy(cfg).eval()
    n_feat = len(entity_feature_schema())
    features = torch.zeros(2, cfg.n_max_entities, n_feat)
    mask = torch.zeros(2, cfg.n_max_entities, dtype=torch.bool)
    # Mark first 20 entities as valid so masked-softmax has something to
    # attend to (all-False mask → softmax(-inf) = NaN).
    mask[:, :20] = True
    features[:, :20, 0] = 1.0  # is_valid
    with torch.no_grad():
        policy, value = model(features, mask)
    assert policy.shape == (2, cfg.n_max_entities, cfg.n_action_channels)
    assert value.shape == (2, 1)
    assert policy.dtype == torch.float32


def test_set_transformer_no_nan():
    cfg = SetTransformerCfg()
    model = SetTransformerPolicy(cfg).eval()
    n_feat = len(entity_feature_schema())
    features = torch.randn(2, cfg.n_max_entities, n_feat)
    mask = torch.zeros(2, cfg.n_max_entities, dtype=torch.bool)
    mask[:, :40] = True
    with torch.no_grad():
        policy, value = model(features, mask)
    # Padded-row policy logits are allowed to be anything (decoder
    # ignores them via mask); the valid rows must be finite.
    valid_policy = policy[mask]
    assert torch.isfinite(valid_policy).all()
    assert torch.isfinite(value).all()


def test_set_transformer_value_in_tanh_range():
    cfg = SetTransformerCfg()
    model = SetTransformerPolicy(cfg).eval()
    n_feat = len(entity_feature_schema())
    features = torch.randn(4, cfg.n_max_entities, n_feat) * 5.0
    mask = torch.zeros(4, cfg.n_max_entities, dtype=torch.bool)
    mask[:, :30] = True
    with torch.no_grad():
        _policy, value = model(features, mask)
    assert value.abs().max().item() <= 1.0 + 1e-6


def test_set_transformer_param_count_within_20_percent():
    cfg = SetTransformerCfg()
    n_feat = len(entity_feature_schema())
    model = SetTransformerPolicy(cfg, n_features=n_feat)
    actual = sum(p.numel() for p in model.parameters())
    est = set_param_estimate(cfg, n_features=n_feat)
    ratio = actual / est
    assert 0.8 <= ratio <= 1.2, (
        f"set-transformer param estimate drifted: actual={actual:,} est={est:,} ratio={ratio:.2f}"
    )


def test_set_transformer_deterministic_in_eval_mode():
    cfg = SetTransformerCfg()
    model = SetTransformerPolicy(cfg).eval()
    n_feat = len(entity_feature_schema())
    features = torch.randn(1, cfg.n_max_entities, n_feat)
    mask = torch.zeros(1, cfg.n_max_entities, dtype=torch.bool)
    mask[:, :25] = True
    with torch.no_grad():
        a = model(features, mask)
        b = model(features, mask)
    torch.testing.assert_close(a[0], b[0])
    torch.testing.assert_close(a[1], b[1])


def test_set_transformer_padding_doesnt_leak(mid_game_obs):
    """Changing padding-row *features* must not change valid-row outputs.

    The entity mask is the main correctness contract — if a padding row
    with a huge feature value shifts the valid-row logits, the policy
    is silently broken the moment we train on variable-N batches.
    """
    obs, _ = mid_game_obs
    cfg = SetTransformerCfg()
    features_np, mask_np = encode_entities(obs, player_id=0, cfg=cfg)
    f = torch.from_numpy(features_np).unsqueeze(0)
    m = torch.from_numpy(mask_np).unsqueeze(0)
    # Sanity: at least one valid entity and at least one padding row.
    assert m.any()
    assert (~m).any()

    model = SetTransformerPolicy(cfg).eval()
    with torch.no_grad():
        policy_a, value_a = model(f, m)

    # Perturb padding rows with large random values.
    f2 = f.clone()
    pad_mask = ~m[0]  # (N,) bool
    f2[0, pad_mask] = torch.randn_like(f2[0, pad_mask]) * 100.0
    with torch.no_grad():
        policy_b, value_b = model(f2, m)

    # Valid-row policy logits must be identical; value (masked-mean pool)
    # must also be identical.
    valid = m[0]
    torch.testing.assert_close(policy_a[0, valid], policy_b[0, valid])
    torch.testing.assert_close(value_a, value_b)


def test_set_transformer_from_real_observation(mid_game_obs):
    """End-to-end: FastEngine → encode_entities → SetTransformerPolicy."""
    obs, _ = mid_game_obs
    cfg = SetTransformerCfg()
    features_np, mask_np = encode_entities(obs, player_id=0, cfg=cfg)
    f = torch.from_numpy(features_np).unsqueeze(0)
    m = torch.from_numpy(mask_np).unsqueeze(0)
    model = SetTransformerPolicy(cfg).eval()
    with torch.no_grad():
        policy, value = model(f, m)
    assert policy.shape == (1, cfg.n_max_entities, cfg.n_action_channels)
    assert value.shape == (1, 1)
    # At least one valid-row logit finite.
    assert torch.isfinite(policy[0, m[0]]).all()
    assert torch.isfinite(value).all()
