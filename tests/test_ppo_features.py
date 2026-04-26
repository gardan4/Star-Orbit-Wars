"""Tests for orbitwars.nn.ppo_features (rollout-collection helpers)."""
from __future__ import annotations

import numpy as np
import pytest

from orbitwars.nn.conv_policy import ConvPolicyCfg
from orbitwars.nn.ppo_features import (
    DecisionRow,
    encode_decisions,
    stack_decisions,
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


def test_encode_decisions_returns_one_row_per_owned_planet():
    obs = _mk_obs(my_count=3, enemy_count=2, my_ships=50)
    rows = encode_decisions(obs, player_id=0)
    assert len(rows) == 3
    assert all(isinstance(r, DecisionRow) for r in rows)
    assert all(r.source_planet_id in {0, 1, 2} for r in rows)
    assert all(r.available_ships == 50 for r in rows)


def test_encode_decisions_skips_zero_ship_planets():
    """A planet with 0 ships can't launch — drop it from decisions."""
    obs = _mk_obs(my_count=2)
    obs["planets"][0][5] = 0  # zero ships on planet 0
    rows = encode_decisions(obs, player_id=0)
    assert len(rows) == 1
    assert rows[0].source_planet_id == 1


def test_encode_decisions_returns_empty_when_no_owned():
    """Player with no owned planets gets no decision rows."""
    obs = _mk_obs(my_count=0, enemy_count=2)
    rows = encode_decisions(obs, player_id=0)
    assert rows == []


def test_encode_decisions_grid_cell_matches_planet_position():
    """A planet at (x=20, y=50) → gy = 50/2 = 25, gx = 20/2 = 10
    (with 50×50 grid spanning [0, 100))."""
    obs = _mk_obs(my_count=1)
    rows = encode_decisions(obs, player_id=0)
    assert len(rows) == 1
    # Planet 0 is at (20, 50) by _mk_obs setup. With grid 50×50:
    # gy = int(50 * 50 / 100) = 25, gx = int(20 * 50 / 100) = 10.
    assert rows[0].gy == 25
    assert rows[0].gx == 10


def test_encode_decisions_obs_x_shape_matches_cfg():
    cfg = ConvPolicyCfg()
    obs = _mk_obs(my_count=2)
    rows = encode_decisions(obs, player_id=0, cfg=cfg)
    assert rows[0].obs_x.shape == (cfg.n_channels, cfg.grid_h, cfg.grid_w)
    assert rows[0].obs_x.dtype == np.float32


def test_encode_decisions_shares_obs_x_across_rows():
    """Same turn → same obs_x reference. Memory efficiency check."""
    obs = _mk_obs(my_count=3)
    rows = encode_decisions(obs, player_id=0)
    # Identity check — encode_grid is called once.
    assert rows[0].obs_x is rows[1].obs_x is rows[2].obs_x


def test_stack_decisions_concatenates_along_batch_dim():
    obs_a = _mk_obs(my_count=2)
    obs_b = _mk_obs(my_count=1)
    rows = encode_decisions(obs_a, 0) + encode_decisions(obs_b, 0)
    obs_x, gy, gx = stack_decisions(rows)
    assert obs_x.shape == (3, 12, 50, 50)
    assert gy.shape == (3,)
    assert gx.shape == (3,)


def test_stack_decisions_handles_empty_input():
    obs_x, gy, gx = stack_decisions([])
    assert obs_x.shape[0] == 0
    assert gy.shape == (0,)
    assert gx.shape == (0,)
