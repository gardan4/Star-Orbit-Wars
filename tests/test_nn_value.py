"""Tests for orbitwars.nn.nn_value (value-fn bridge from ConvPolicy)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from orbitwars.nn.conv_policy import ConvPolicy, ConvPolicyCfg
from orbitwars.nn.nn_value import (
    make_constant_value_fn,
    make_nn_value_fn,
    make_random_value_fn,
)


def _mk_obs(my_count: int = 2, enemy_count: int = 1, my_ships: int = 50):
    planets = []
    for i in range(my_count):
        planets.append([i, 0, 20.0 + i * 5, 50.0, 1.5, my_ships, 3])
    for j in range(enemy_count):
        planets.append([my_count + j, 1, 80.0 - j * 5, 50.0, 1.5, 30, 3])
    initial = [[p[0], p[1], p[2], p[3], p[4], 10, p[6]] for p in planets]
    return {
        "player": 0,
        "step": 50,
        "angular_velocity": 0.03,
        "planets": planets,
        "initial_planets": initial,
        "fleets": [],
        "next_fleet_id": 0,
        "comet_planet_ids": [],
        "comets": [],
    }


def _mk_model(seed: int = 0):
    cfg = ConvPolicyCfg(
        grid_h=50, grid_w=50, n_channels=12,
        backbone_channels=32, n_blocks=3,
        n_action_channels=8, value_hidden=128,
    )
    torch.manual_seed(seed)
    model = ConvPolicy(cfg)
    model.eval()
    return model, cfg


def test_make_nn_value_fn_returns_finite_scalar():
    model, cfg = _mk_model(seed=0)
    fn = make_nn_value_fn(model, cfg)
    obs = _mk_obs()
    v = fn(obs, my_player=0)
    assert isinstance(v, float)
    assert np.isfinite(v)
    assert -1.0 <= v <= 1.0


def test_make_nn_value_fn_consistent_across_calls():
    """Same model + same obs → same value (no per-call randomness)."""
    model, cfg = _mk_model(seed=0)
    fn = make_nn_value_fn(model, cfg)
    obs = _mk_obs()
    v1 = fn(obs, my_player=0)
    v2 = fn(obs, my_player=0)
    assert v1 == pytest.approx(v2, abs=1e-7)


def test_make_nn_value_fn_player_swap_changes_value():
    """Encoding flips perspective by player_id, so player 0 and player 1
    on the same obs should generally NOT give the same value."""
    model, cfg = _mk_model(seed=0)
    fn = make_nn_value_fn(model, cfg)
    obs = _mk_obs(my_count=3, enemy_count=1)
    v0 = fn(obs, my_player=0)
    v1 = fn(obs, my_player=1)
    # Asymmetric position (3 owned vs 1 enemy) — value should differ.
    # Fresh-init model is noisy but the encoded grid IS different
    # between perspectives, so forward outputs should differ.
    assert v0 != pytest.approx(v1, abs=1e-3), (
        "Perspective swap should change encoded grid → value."
    )


def test_make_nn_value_fn_handles_encoding_errors_gracefully(monkeypatch):
    """If encode_grid raises, value_fn returns 0.0 instead of propagating."""
    model, cfg = _mk_model(seed=0)
    fn = make_nn_value_fn(model, cfg)
    # Monkey-patch the imported encode_grid in nn_value to raise.
    from orbitwars.nn import nn_value
    def boom(*args, **kwargs):
        raise RuntimeError("simulated encode failure")
    monkeypatch.setattr(nn_value, "encode_grid", boom)
    v = fn(_mk_obs(), my_player=0)
    assert v == 0.0  # safe fallback — search will anchor-lock instead


def test_make_nn_value_fn_clips_values():
    """Even if value head outputs out-of-range, clip enforces [-1, 1]."""
    model, cfg = _mk_model(seed=0)
    # Force value head to output very large values by scaling weights.
    with torch.no_grad():
        for p in model.value_head.parameters():
            p.mul_(1000.0)
    fn = make_nn_value_fn(model, cfg, clip=1.0)
    obs = _mk_obs()
    v = fn(obs, my_player=0)
    assert -1.0 <= v <= 1.0


def test_make_constant_value_fn_returns_same_value():
    fn = make_constant_value_fn(value=0.5)
    obs1 = _mk_obs(my_count=2)
    obs2 = _mk_obs(my_count=5)
    assert fn(obs1, 0) == 0.5
    assert fn(obs2, 1) == 0.5
    assert fn({}, 0) == 0.5  # even malformed obs


def test_make_constant_value_fn_default_zero():
    fn = make_constant_value_fn()
    assert fn(_mk_obs(), 0) == 0.0


def test_make_random_value_fn_in_range_and_varies():
    fn = make_random_value_fn(seed=42)
    obs = _mk_obs()
    values = [fn(obs, 0) for _ in range(20)]
    assert all(-1.0 <= v <= 1.0 for v in values)
    # Should vary across calls (even with same obs).
    assert len(set(values)) > 1


def test_make_random_value_fn_deterministic_given_seed():
    """Same seed → same sequence."""
    fn_a = make_random_value_fn(seed=42)
    fn_b = make_random_value_fn(seed=42)
    obs = _mk_obs()
    seq_a = [fn_a(obs, 0) for _ in range(10)]
    seq_b = [fn_b(obs, 0) for _ in range(10)]
    assert seq_a == seq_b


def test_make_nn_value_fn_runs_on_obs_with_fleets_and_comets():
    """Value fn should handle a richer obs (fleets, comets present)."""
    model, cfg = _mk_model(seed=0)
    fn = make_nn_value_fn(model, cfg)
    obs = _mk_obs()
    obs["fleets"] = [[0, 0, 50.0, 50.0, 0.0, 0, 20]]  # one ally fleet
    obs["comet_planet_ids"] = [3]
    obs["comets"] = [[100, 50.0, 50.0, 0.0, 0.0, 0.5]]
    v = fn(obs, my_player=0)
    assert np.isfinite(v)
    assert -1.0 <= v <= 1.0
