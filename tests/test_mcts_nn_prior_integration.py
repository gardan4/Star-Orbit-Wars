"""Integration smoke: MCTSAgent + NN prior end-to-end.

These tests build a *random-init* ConvPolicy (no BC checkpoint required)
and route its priors into the search via ``move_prior_fn``. The point is
NOT that random-init priors win — they obviously won't. The point is:

* The wiring round-trips: ConvPolicy logits → bucket lookup → softmax →
  PlanetMove.prior overwrite → search reads them at the root.
* The search returns a valid wire-format action without crashing.
* If the NN forward fails for any reason, the search falls back to the
  heuristic priors that ``generate_per_planet_moves`` already attached
  (defensive: a buggy NN must NEVER forfeit a turn).

Once the BC checkpoint at ``runs/bc_warmstart.pt`` lands, the same
fixture transparently loads real weights via ``load_conv_policy``. The
test marks itself ``skipif`` when torch isn't installed in the venv
running the suite (CPU-only ladder venv).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from orbitwars.bots.base import Deadline
from orbitwars.bots.mcts_bot import MCTSAgent
from orbitwars.mcts.gumbel_search import GumbelConfig
from orbitwars.nn.conv_policy import ConvPolicy, ConvPolicyCfg
from orbitwars.nn.nn_prior import make_nn_prior_fn


def _mk_obs(step: int = 10, my_ships: int = 50, enemy_ships: int = 30):
    return {
        "player": 0,
        "step": step,
        "angular_velocity": 0.03,
        "planets": [
            [0, 0, 20.0, 50.0, 1.5, my_ships, 3],
            [1, 1, 80.0, 50.0, 1.5, enemy_ships, 3],
            [2, -1, 50.0, 20.0, 1.0, 10, 1],
        ],
        "initial_planets": [
            [0, 0, 20.0, 50.0, 1.5, 10, 3],
            [1, 1, 80.0, 50.0, 1.5, 10, 3],
            [2, -1, 50.0, 20.0, 1.0, 10, 1],
        ],
        "fleets": [],
        "next_fleet_id": 0,
        "comet_planet_ids": [],
        "comets": [],
    }


@pytest.fixture
def random_conv_prior_fn():
    """Random-init ConvPolicy + matching prior closure.

    Uses default ConvPolicyCfg so it matches what ``load_conv_policy``
    constructs for partial checkpoints (which omit cfg).
    """
    cfg = ConvPolicyCfg()
    model = ConvPolicy(cfg)
    model.eval()
    return make_nn_prior_fn(model, cfg, hold_neutral_prob=0.05, temperature=1.0)


def test_mcts_with_nn_prior_returns_wire_format(random_conv_prior_fn):
    agent = MCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=4, total_sims=8, rollout_depth=2,
            hard_deadline_ms=2000.0,
        ),
        rng_seed=0,
        move_prior_fn=random_conv_prior_fn,
    )
    dl = Deadline()
    action = agent.act(_mk_obs(), dl)
    assert isinstance(action, list)
    for move in action:
        assert len(move) == 3
        pid, angle, ships = move
        assert isinstance(pid, int)
        assert isinstance(angle, float)
        assert isinstance(ships, int)


def test_mcts_with_broken_prior_fn_falls_back():
    """A prior_fn that always raises must NOT forfeit the turn."""
    def boom(obs, my_player, moves_by_planet, available_by_planet):
        raise RuntimeError("simulated NN failure")

    agent = MCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=4, total_sims=8, rollout_depth=2,
            hard_deadline_ms=2000.0,
        ),
        rng_seed=0,
        move_prior_fn=boom,
    )
    dl = Deadline()
    action = agent.act(_mk_obs(), dl)
    # Same shape contract as the heuristic path.
    assert isinstance(action, list)
    for move in action:
        assert len(move) == 3


def test_mcts_with_nn_prior_matches_no_prior_when_disabled(random_conv_prior_fn):
    """Sanity: setting move_prior_fn=None should be byte-identical to
    not threading the parameter at all (default constructor)."""
    cfg_kwargs = dict(
        gumbel_cfg=GumbelConfig(
            num_candidates=4, total_sims=8, rollout_depth=2,
            hard_deadline_ms=2000.0,
        ),
        rng_seed=0,
    )
    a1 = MCTSAgent(**cfg_kwargs)
    a2 = MCTSAgent(**cfg_kwargs, move_prior_fn=None)
    obs = _mk_obs()
    act1 = a1.act(obs, Deadline())
    act2 = a2.act(obs, Deadline())
    # Same RNG seed + same obs + same (None) prior → byte-equal action.
    assert act1 == act2
