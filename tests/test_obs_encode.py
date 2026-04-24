"""Tests for the W4 observation encoders (features/obs_encode.py).

These tests pin the contract that both W4 NN candidates depend on:
  * ``encode_grid(obs, player_id)`` returns (C=12, H=50, W=50) float32.
  * ``encode_entities(obs, player_id)`` returns (320, 19) + bool mask.
  * Perspective normalization swaps owner_me / owner_enemy under
    ``player_id`` change (self-consistent symmetry on a symmetric map).
  * Schema offsets in conv/set-transformer stay in sync with the encoder.
  * Output is NaN-free and in reasonable numeric ranges.

Regression goal: if a future refactor changes the feature schema or
the encoder layout, one of these tests breaks loudly before a training
run wastes compute on misaligned tensors.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from orbitwars.engine.fast_engine import FastEngine
from orbitwars.features.obs_encode import (
    encode_entities,
    encode_grid,
    owned_planet_positions,
)
from orbitwars.nn.conv_policy import ConvPolicyCfg, feature_channels
from orbitwars.nn.set_transformer import (
    SetTransformerCfg,
    entity_feature_schema,
    feature_offsets,
)


@pytest.fixture
def mid_game_engine():
    """A 2-player engine advanced 100 steps with heuristic play."""
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
    return fe


def test_grid_shape_and_dtype(mid_game_engine):
    obs = mid_game_engine.observation(0)
    cfg = ConvPolicyCfg()
    g = encode_grid(obs, player_id=0, cfg=cfg)
    assert g.shape == (cfg.n_channels, cfg.grid_h, cfg.grid_w)
    assert g.dtype == np.float32
    assert len(feature_channels()) == cfg.n_channels


def test_grid_no_nans_or_infs(mid_game_engine):
    obs = mid_game_engine.observation(0)
    g = encode_grid(obs, player_id=0)
    assert np.isfinite(g).all(), "encode_grid produced NaN or inf"


def test_grid_turn_phase_is_broadcast(mid_game_engine):
    obs = mid_game_engine.observation(0)
    g = encode_grid(obs, player_id=0)
    ch = {name: i for i, name in enumerate(feature_channels())}
    turn_phase_plane = g[ch["turn_phase"]]
    # All cells should carry the same value.
    assert np.all(turn_phase_plane == turn_phase_plane[0, 0])
    # Value == step / 500.
    assert abs(turn_phase_plane[0, 0] - obs["step"] / 500.0) < 1e-6


def test_grid_perspective_swap(mid_game_engine):
    """On a symmetric map, ship_count_p0(player=0) equals
    ship_count_p1(player=1) and vice versa."""
    obs0 = mid_game_engine.observation(0)
    obs1 = mid_game_engine.observation(1)
    g0 = encode_grid(obs0, player_id=0)
    g1 = encode_grid(obs1, player_id=1)
    ch = {name: i for i, name in enumerate(feature_channels())}
    # p0's "me" channel should sum to same as p1's "enemy" channel.
    np.testing.assert_allclose(
        g0[ch["ship_count_p0"]].sum(),
        g1[ch["ship_count_p1"]].sum(),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        g0[ch["ship_count_p1"]].sum(),
        g1[ch["ship_count_p0"]].sum(),
        atol=1e-4,
    )


def test_entity_shape_and_mask(mid_game_engine):
    obs = mid_game_engine.observation(0)
    cfg = SetTransformerCfg()
    features, mask = encode_entities(obs, player_id=0, cfg=cfg)
    assert features.shape == (cfg.n_max_entities, len(entity_feature_schema()))
    assert mask.shape == (cfg.n_max_entities,)
    assert features.dtype == np.float32
    assert mask.dtype == np.bool_

    # Valid rows should have is_valid=1; padding rows is_valid=0.
    offs = feature_offsets()
    for i in range(cfg.n_max_entities):
        if mask[i]:
            assert features[i, offs["is_valid"]] == 1.0
        else:
            assert features[i, offs["is_valid"]] == 0.0
            # Padding rows stay all-zero (no leakage from trailing real data).
            assert np.all(features[i] == 0.0)


def test_entity_no_nans_or_infs(mid_game_engine):
    obs = mid_game_engine.observation(0)
    features, _mask = encode_entities(obs, player_id=0)
    assert np.isfinite(features).all(), "encode_entities produced NaN or inf"


def test_entity_type_one_hot_exclusive(mid_game_engine):
    """Every valid row should have exactly one of type_planet /
    type_fleet set (type_comet is allowed co-set with type_planet since
    comets ARE planet-backed)."""
    obs = mid_game_engine.observation(0)
    features, mask = encode_entities(obs, player_id=0)
    offs = feature_offsets()
    for i in np.where(mask)[0]:
        t_planet = features[i, offs["type_planet"]]
        t_fleet = features[i, offs["type_fleet"]]
        # Exactly one of planet/fleet (comet is allowed alongside planet).
        assert (t_planet + t_fleet) == 1.0


def test_entity_owner_mutually_exclusive(mid_game_engine):
    """Every valid row has exactly one owner flag set."""
    obs = mid_game_engine.observation(0)
    features, mask = encode_entities(obs, player_id=0)
    offs = feature_offsets()
    for i in np.where(mask)[0]:
        s = (
            features[i, offs["owner_me"]]
            + features[i, offs["owner_enemy"]]
            + features[i, offs["owner_neutral"]]
        )
        assert s == 1.0, f"row {i} has owner sum {s}"


def test_entity_perspective_swap(mid_game_engine):
    """On a symmetric map, p0's owner_me count == p1's owner_enemy count."""
    obs0 = mid_game_engine.observation(0)
    obs1 = mid_game_engine.observation(1)
    f0, m0 = encode_entities(obs0, player_id=0)
    f1, m1 = encode_entities(obs1, player_id=1)
    offs = feature_offsets()
    p0_me = int((f0[m0, offs["owner_me"]] == 1).sum())
    p0_enemy = int((f0[m0, offs["owner_enemy"]] == 1).sum())
    p1_me = int((f1[m1, offs["owner_me"]] == 1).sum())
    p1_enemy = int((f1[m1, offs["owner_enemy"]] == 1).sum())
    assert p0_me == p1_enemy
    assert p0_enemy == p1_me


def test_owned_planet_positions_matches_owner_me(mid_game_engine):
    """owned_planet_positions and entity-encoder owner_me planets agree."""
    obs = mid_game_engine.observation(0)
    mine = owned_planet_positions(obs, player_id=0)
    features, mask = encode_entities(obs, player_id=0)
    offs = feature_offsets()
    # Count entity rows that are (type_planet=1, owner_me=1).
    n_owned_rows = 0
    for i in np.where(mask)[0]:
        if (
            features[i, offs["type_planet"]] == 1.0
            and features[i, offs["owner_me"]] == 1.0
        ):
            n_owned_rows += 1
    assert n_owned_rows == len(mine)


def test_schema_offsets_stable():
    """feature_offsets() must match the canonical entity_feature_schema()."""
    offs = feature_offsets()
    schema = entity_feature_schema()
    for i, name in enumerate(schema):
        assert offs[name] == i, f"offset drift for {name}: {offs[name]} != {i}"


def test_pos_feature_offsets_are_what_model_expects():
    """The commented-out torch forward pass in set_transformer.py uses
    hardcoded POS_X=7, POS_Y=8 — verify these still match the schema."""
    offs = feature_offsets()
    assert offs["pos_x"] == 7, (
        "pos_x offset changed — update SetTransformerPolicy.forward constants"
    )
    assert offs["pos_y"] == 8, (
        "pos_y offset changed — update SetTransformerPolicy.forward constants"
    )
