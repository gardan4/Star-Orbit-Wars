"""Tests for MCTSAgent — the Path B bot.

Covers the Agent-contract integration:
  * act() returns a valid wire-format action within the deadline.
  * act() falls back to heuristic on search failure.
  * act() respects the outer Deadline (returns early when budget is tight).
  * Agent can be used as a kaggle_agent callable.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from orbitwars.bots.base import Deadline
from orbitwars.bots.mcts_bot import MCTSAgent, build
from orbitwars.mcts.actions import ActionConfig
from orbitwars.mcts.gumbel_search import GumbelConfig


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


def test_mcts_agent_returns_wire_format():
    agent = MCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=4, total_sims=8, rollout_depth=2,
            hard_deadline_ms=2000.0,
        ),
        rng_seed=0,
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


def test_mcts_agent_falls_back_on_no_legal_moves():
    """All planets owned by the opponent → heuristic staged empty
    action → MCTSAgent returns empty action (no crash)."""
    agent = MCTSAgent(rng_seed=0)
    obs = _mk_obs()
    obs["planets"][0][1] = 1
    dl = Deadline()
    action = agent.act(obs, dl)
    assert action == []


def test_mcts_agent_falls_back_on_search_exception(monkeypatch):
    """If the inner search raises, we still return a valid action
    (the heuristic's)."""
    from orbitwars.mcts import gumbel_search as gs

    class _Boom:
        def __getattr__(self, _name):
            raise RuntimeError("injected failure")
        def search(self, *a, **kw):
            raise RuntimeError("injected search failure")

    agent = MCTSAgent(rng_seed=0)
    agent._search = _Boom()  # type: ignore[assignment]

    dl = Deadline()
    action = agent.act(_mk_obs(), dl)
    # Should be the heuristic's output — a list, may be empty or populated.
    assert isinstance(action, list)


def test_mcts_agent_respects_tight_outer_deadline():
    """When the outer Deadline has almost no budget left, skip search
    and return the heuristic action."""
    agent = MCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=4, total_sims=64, rollout_depth=2,
            hard_deadline_ms=2000.0,
        ),
        rng_seed=0,
    )
    dl = Deadline()
    # Burn the budget before act() is called.
    dl._t0 -= 10.0  # pretend 10s have already passed

    t0 = time.perf_counter()
    action = agent.act(_mk_obs(), dl)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(action, list)
    # Should bypass search entirely → fast.
    assert elapsed_ms < 200.0


def test_mcts_agent_resets_state_on_step_zero():
    """Turn 0 refreshes the heuristic fallback so stale cooldowns from
    a previous match don't block launches."""
    agent = MCTSAgent(rng_seed=0)
    old_fallback = agent._fallback
    old_search = agent._search

    agent.act(_mk_obs(step=0), Deadline())
    assert agent._fallback is not old_fallback
    assert agent._search is not old_search


def test_build_factory():
    agent = build(rng_seed=1)
    assert isinstance(agent, MCTSAgent)
    assert agent.name == "mcts"


def test_mcts_agent_as_kaggle_agent_is_callable():
    agent = MCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=3, total_sims=4, rollout_depth=1,
            hard_deadline_ms=1000.0,
        ),
        rng_seed=0,
    )
    k_agent = agent.as_kaggle_agent()
    # Kaggle passes obs + cfg; we accept both.
    result = k_agent(_mk_obs())
    assert isinstance(result, list)


def test_mcts_agent_with_object_style_obs():
    """SimpleNamespace obs (Kaggle ladder style) works end-to-end."""
    obs_dict = _mk_obs()
    obs_obj = SimpleNamespace(**obs_dict)
    agent = MCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=3, total_sims=4, rollout_depth=1,
            hard_deadline_ms=2000.0,
        ),
        rng_seed=0,
    )
    action = agent.act(obs_obj, Deadline())
    assert isinstance(action, list)


def test_mcts_agent_matches_heuristic_when_guard_is_tight():
    """With a very high improvement margin, MCTS should degenerate to
    the heuristic's action exactly. This is the key "heuristic floor"
    guarantee — if search can't find a confidently better move, we
    return what the heuristic picked."""
    from orbitwars.bots.heuristic import HeuristicAgent

    agent = MCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=8, total_sims=16, rollout_depth=2,
            hard_deadline_ms=2000.0,
            anchor_improvement_margin=10.0,  # impossible to clear
        ),
        rng_seed=0,
    )
    heur = HeuristicAgent()

    for step in (0, 10, 20, 30):
        obs = _mk_obs(step=step)
        h_action = heur.act(obs, Deadline())
        m_action = agent.act(obs, Deadline())
        assert sorted(m_action) == sorted(h_action), (
            f"step={step}: MCTS={m_action} vs HEUR={h_action}"
        )


def test_mcts_agent_preserves_margin_in_tight_cfg(monkeypatch):
    """Regression test: when mcts_bot rebuilds GumbelConfig to tighten
    the deadline, it MUST carry over `anchor_improvement_margin` from
    the original config. Dropping it silently reverts to the default
    (0.15), which would let rollout noise overwrite the anchor and
    cause the bot to lose the heuristic floor guarantee."""
    observed: list = []

    original_search = None

    def _capturing_search(self, obs, my_player, start_time=None, anchor_action=None):
        observed.append(self.gumbel_cfg.anchor_improvement_margin)
        # Return None → agent falls back to heuristic. We only care that
        # the cfg in effect during search carried the margin through.
        return None

    from orbitwars.mcts import gumbel_search as gs
    monkeypatch.setattr(gs.GumbelRootSearch, "search", _capturing_search)

    agent = MCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=2, total_sims=2, rollout_depth=1,
            hard_deadline_ms=2000.0,
            anchor_improvement_margin=7.77,  # unique sentinel
        ),
        rng_seed=0,
    )
    agent.act(_mk_obs(step=10), Deadline())
    assert observed, "search() was not invoked"
    assert observed[0] == pytest.approx(7.77), (
        f"tight_cfg stripped anchor_improvement_margin: saw {observed[0]}"
    )


def test_mcts_agent_wires_opponent_posterior():
    """When ``use_opponent_model=True`` (default), the agent builds an
    ArchetypePosterior on turn 0 and calls observe() on subsequent turns.
    """
    from orbitwars.opponent.bayes import ArchetypePosterior

    agent = MCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=2, total_sims=2, rollout_depth=1,
            hard_deadline_ms=2000.0,
            anchor_improvement_margin=10.0,
        ),
        rng_seed=0,
    )
    # Turn 0: should construct the posterior fresh.
    agent.act(_mk_obs(step=0), Deadline())
    assert isinstance(agent.opp_posterior, ArchetypePosterior)
    # turns_observed is 0 since the first observe() just snapshots state.
    assert agent.opp_posterior.turns_observed() == 0

    # Further turns accumulate evidence.
    for step in (1, 2, 3):
        agent.act(_mk_obs(step=step), Deadline())
    # After 4 total observe() calls, 3 of them contribute evidence.
    assert agent.opp_posterior.turns_observed() == 3


def test_mcts_agent_routes_posterior_to_search_override():
    """When the posterior is concentrated past the threshold, MCTSAgent
    installs an archetype-building override on the search."""
    import numpy as np

    agent = MCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=2, total_sims=2, rollout_depth=1,
            hard_deadline_ms=2000.0,
            anchor_improvement_margin=10.0,
        ),
        rng_seed=0,
    )
    # Turn 0 bootstrap: sets up the posterior.
    agent.act(_mk_obs(step=0), Deadline())
    # Artificially force the posterior into a concentrated state.
    post = agent.opp_posterior
    assert post is not None
    # Accumulate fake evidence meeting MIN_TURNS.
    post._turns_observed = agent._POSTERIOR_MIN_TURNS
    # Force log_alpha so softmax puts >0.35 on one archetype.
    post.log_alpha = np.full(post.K, -5.0)
    target_idx = post.names.index("rusher")
    post.log_alpha[target_idx] = 3.0  # dominates

    # Run another turn — the routing logic should kick in.
    agent._maybe_route_posterior_to_search()
    assert agent._search.opp_policy_override is not None

    # Calling the override yields a rusher-flavored HeuristicAgent.
    opp_built = agent._search.opp_policy_override()
    assert opp_built.name == "rusher"


def test_mcts_agent_no_override_under_weak_posterior():
    """With a near-uniform posterior, the search's opp_policy_override
    stays None — we don't exploit when we're uncertain."""
    import numpy as np

    agent = MCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=2, total_sims=2, rollout_depth=1,
            hard_deadline_ms=2000.0,
            anchor_improvement_margin=10.0,
        ),
        rng_seed=0,
    )
    agent.act(_mk_obs(step=0), Deadline())
    post = agent.opp_posterior
    assert post is not None
    post._turns_observed = 50  # enough turns
    # Near-uniform → max < 0.35
    post.log_alpha = np.zeros(post.K)

    agent._maybe_route_posterior_to_search()
    assert agent._search.opp_policy_override is None


def test_mcts_agent_can_disable_opponent_model():
    """`use_opponent_model=False` — no posterior ever constructed."""
    agent = MCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=2, total_sims=2, rollout_depth=1,
            hard_deadline_ms=2000.0,
            anchor_improvement_margin=10.0,
        ),
        rng_seed=0,
        use_opponent_model=False,
    )
    agent.act(_mk_obs(step=0), Deadline())
    agent.act(_mk_obs(step=1), Deadline())
    assert agent.opp_posterior is None


def test_mcts_agent_opponent_model_failure_does_not_break_turn(monkeypatch):
    """A raising observe() must be caught so agents remain tournament-safe."""
    agent = MCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=2, total_sims=2, rollout_depth=1,
            hard_deadline_ms=2000.0,
            anchor_improvement_margin=10.0,
        ),
        rng_seed=0,
    )
    agent.act(_mk_obs(step=0), Deadline())

    from orbitwars.opponent.bayes import ArchetypePosterior

    def _boom(self, obs, opp_player):  # pragma: no cover - injected failure
        raise RuntimeError("intentional posterior failure")

    monkeypatch.setattr(ArchetypePosterior, "observe", _boom)

    # Must not raise.
    action = agent.act(_mk_obs(step=5), Deadline())
    assert isinstance(action, list)


def test_mcts_agent_telemetry_tracks_override_events():
    """The `telemetry` dict must track (a) turns the posterior observed,
    (b) how often the search override was installed, (c) how often it
    was cleared. This is what smoke harnesses read to understand *why*
    a use-model vs no-model run did or didn't show a delta."""
    import numpy as np

    agent = MCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=2, total_sims=2, rollout_depth=1,
            hard_deadline_ms=2000.0,
            anchor_improvement_margin=10.0,
        ),
        rng_seed=0,
    )
    # Fresh-start: all counters at zero.
    assert agent.telemetry["override_fires"] == 0
    assert agent.telemetry["override_clears"] == 0
    assert agent.telemetry["last_top_name"] is None
    assert agent.telemetry["last_top_prob"] == 0.0

    agent.act(_mk_obs(step=0), Deadline())
    post = agent.opp_posterior
    assert post is not None

    # Force concentration above threshold → override should fire,
    # telemetry should record it.
    post._turns_observed = agent._POSTERIOR_MIN_TURNS
    post.log_alpha = np.full(post.K, -5.0)
    post.log_alpha[post.names.index("rusher")] = 3.0
    agent._maybe_route_posterior_to_search()
    assert agent.telemetry["override_fires"] == 1
    assert agent.telemetry["last_top_name"] == "rusher"
    assert agent.telemetry["last_top_prob"] > 0.35

    # Now collapse posterior below threshold → override should clear,
    # telemetry records the transition.
    post.log_alpha = np.zeros(post.K)
    agent._maybe_route_posterior_to_search()
    assert agent.telemetry["override_clears"] == 1
    assert agent._search.opp_policy_override is None


def test_mcts_agent_telemetry_resets_on_step_zero():
    """Telemetry must reset at the start of each match so back-to-back
    games in a smoke harness don't see cumulative counts from prior matches."""
    import numpy as np

    agent = MCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=2, total_sims=2, rollout_depth=1,
            hard_deadline_ms=2000.0,
            anchor_improvement_margin=10.0,
        ),
        rng_seed=0,
    )
    # Match 1: pump some fake telemetry.
    agent.act(_mk_obs(step=0), Deadline())
    post = agent.opp_posterior
    post._turns_observed = agent._POSTERIOR_MIN_TURNS
    post.log_alpha = np.full(post.K, -5.0)
    post.log_alpha[post.names.index("rusher")] = 3.0
    agent._maybe_route_posterior_to_search()
    assert agent.telemetry["override_fires"] == 1

    # Match 2 starts at step 0 — telemetry must be reset.
    agent.act(_mk_obs(step=0), Deadline())
    assert agent.telemetry["override_fires"] == 0
    assert agent.telemetry["override_clears"] == 0
    assert agent.telemetry["last_top_name"] is None


def test_fast_engine_step_does_not_touch_global_random():
    """Regression test for the global-random pollution bug: FastEngine
    must use its instance RNG for comet-ship generation so MCTS rollouts
    don't desynchronize the outer Kaggle judge's RNG stream."""
    import random as _pyr
    from orbitwars.engine.fast_engine import FastEngine

    # Build obs at step=49 (comet spawns at step 50), then step once
    # past a comet spawn boundary. Capture global-random state before
    # and after; they must match.
    obs = _mk_obs(step=49)
    eng = FastEngine.from_official_obs(obs, num_agents=2)

    _pyr.seed(12345)
    pre_state = _pyr.getstate()
    eng.step([[], []])  # no-op actions; _maybe_spawn_comets runs
    post_state = _pyr.getstate()

    assert pre_state == post_state, (
        "FastEngine.step() consumed entropy from the global random "
        "module — this desynchronizes the Kaggle judge's RNG during "
        "MCTS rollouts. Use self._rng in _maybe_spawn_comets."
    )
