"""NNRolloutAgent + nn_rollout_factory dispatch tests.

The agent itself is a thin wrapper around a ConvPolicy forward pass +
ACTION_LOOKUP decode. Tests verify:
  * It produces valid wire actions (correct shape, ships ≤ available,
    finite angle).
  * It tolerates an empty obs / no own planets without crashing.
  * GumbelRootSearch dispatches to the factory when rollout_policy='nn'.
"""
from __future__ import annotations

import torch

from orbitwars.bots.base import Deadline
from orbitwars.bots.nn_rollout import NNRolloutAgent
from orbitwars.mcts.gumbel_search import GumbelConfig, GumbelRootSearch
from orbitwars.nn.conv_policy import ConvPolicy, ConvPolicyCfg


def _fake_obs(planets, player=0):
    return {
        "player": player,
        "step": 0,
        "planets": planets,
        "initial_planets": planets,
        "fleets": [],
        "comets": [],
        "next_fleet_id": 0,
        "comet_planet_ids": [],
        "angular_velocity": 0.04,
    }


def _make_model(seed: int = 0) -> tuple:
    torch.manual_seed(seed)
    cfg = ConvPolicyCfg(backbone_channels=8, n_blocks=1)
    model = ConvPolicy(cfg)
    return model, cfg


def test_nn_rollout_returns_valid_actions():
    model, cfg = _make_model()
    agent = NNRolloutAgent(model, cfg)
    # 3 my planets with enough ships, 2 enemy planets.
    planets = [
        [0, 0, 20.0, 30.0, 1.0, 50, 1],
        [1, 0, 70.0, 70.0, 1.0, 100, 1],
        [2, 0, 50.0, 80.0, 1.0, 30, 1],
        [3, 1, 30.0, 60.0, 1.0, 40, 1],
        [4, 1, 80.0, 20.0, 1.0, 35, 1],
    ]
    obs = _fake_obs(planets, player=0)
    moves = agent.act(obs, Deadline())
    assert isinstance(moves, list)
    # Should produce up to 3 launches (one per my-planet with ships ≥ 20).
    assert 0 < len(moves) <= 3
    for m in moves:
        assert len(m) == 3
        pid, angle, ships = m
        assert isinstance(pid, int)
        assert isinstance(ships, int)
        assert ships >= 20  # min_launch_size default
        # Angle is one of the 4 bucket centers (or any valid float).
        assert -10.0 < float(angle) < 10.0
        # Source planet must be one of ours.
        src = next((p for p in planets if p[0] == pid), None)
        assert src is not None and src[1] == 0
        assert ships <= int(src[5])


def test_nn_rollout_returns_noop_on_empty_planets():
    model, cfg = _make_model()
    agent = NNRolloutAgent(model, cfg)
    obs = _fake_obs([], player=0)
    moves = agent.act(obs, Deadline())
    assert moves == []


def test_nn_rollout_skips_planets_below_min_launch():
    model, cfg = _make_model()
    agent = NNRolloutAgent(model, cfg, min_launch_size=30)
    planets = [
        [0, 0, 20.0, 30.0, 1.0, 25, 1],   # 25 < 30 → skipped
        [1, 0, 70.0, 70.0, 1.0, 50, 1],   # 50 ≥ 30 → eligible
        [2, 1, 30.0, 60.0, 1.0, 40, 1],
    ]
    obs = _fake_obs(planets, player=0)
    moves = agent.act(obs, Deadline())
    pids = {m[0] for m in moves}
    assert 0 not in pids  # below min_launch_size
    assert 1 in pids


def test_gumbel_search_dispatches_to_nn_rollout_factory():
    """When rollout_policy='nn' and nn_rollout_factory is set, both
    _opp_factory and _my_future_factory must return the NN rollout agent.
    """
    model, cfg = _make_model()

    calls = {"factory": 0}

    def factory():
        calls["factory"] += 1
        return NNRolloutAgent(model, cfg)

    gcfg = GumbelConfig(
        num_candidates=2, total_sims=4, rollout_depth=1,
        rollout_policy="nn", hard_deadline_ms=5000.0,
    )
    search = GumbelRootSearch(
        gumbel_cfg=gcfg, rng_seed=0, nn_rollout_factory=factory,
    )
    # Both factories should call into the supplied factory.
    a = search._opp_factory()
    assert isinstance(a, NNRolloutAgent)
    b = search._my_future_factory()
    assert isinstance(b, NNRolloutAgent)
    assert calls["factory"] == 2


def test_gumbel_search_falls_back_when_factory_missing():
    """rollout_policy='nn' with factory=None should fall through to the
    HeuristicAgent default (the safe path; never forfeits a turn).
    """
    from orbitwars.bots.heuristic import HeuristicAgent
    gcfg = GumbelConfig(
        num_candidates=2, total_sims=4, rollout_depth=1,
        rollout_policy="nn", hard_deadline_ms=5000.0,
    )
    search = GumbelRootSearch(
        gumbel_cfg=gcfg, rng_seed=0, nn_rollout_factory=None,
    )
    # No factory → falls through to fast-or-heuristic path.
    a = search._opp_factory()
    # rollout_policy="nn" with factory=None matches neither nn nor fast,
    # so we land on default heuristic.
    assert isinstance(a, HeuristicAgent)
