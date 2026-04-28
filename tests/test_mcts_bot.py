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


def test_mcts_agent_reserves_rollout_overshoot_budget(monkeypatch):
    """Regression test for the W2 time-budget audit failure: when outer
    remaining time is just above the wrap-up floor, `safe_budget` MUST
    reserve enough headroom for a one-rollout overshoot (~260 ms on
    dense mid-game states) so the total turn stays under HARD_DEADLINE_MS.

    Concretely: with remaining ≈ 400 ms, safe_budget should be at most
    ~60 ms (not 300). The pre-fix code set safe_budget = min(300, remaining
    - 40) = 300, which blew past 900 ms when pre-search had already eaten
    most of the budget and a rollout overshoot landed at 260 ms on top.
    """
    observed: list = []

    def _capturing_search(self, obs, my_player, start_time=None, anchor_action=None, **kwargs):
        # **kwargs absorbs outer_hard_stop_at (added in the audit tail
        # fix) so this regression remains focused on the safe_budget
        # arithmetic it's actually testing.
        observed.append(self.gumbel_cfg.hard_deadline_ms)
        return None

    from orbitwars.mcts import gumbel_search as gs
    monkeypatch.setattr(gs.GumbelRootSearch, "search", _capturing_search)

    agent = MCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=2, total_sims=2, rollout_depth=1,
            hard_deadline_ms=300.0,  # shipped default
            anchor_improvement_margin=10.0,
        ),
        rng_seed=0,
    )
    # Construct a Deadline that has ~400 ms of remaining budget against
    # HARD_DEADLINE_MS=900. elapsed=500 → remaining=400.
    dl = Deadline()
    dl._t0 -= 0.500  # pretend 500 ms have passed
    agent.act(_mk_obs(step=10), dl)

    assert observed, "search() was not invoked"
    safe_budget = observed[0]
    # With remaining ≈ 400 and overshoot reserve 260 + wrap-up 40, the
    # effective safe_budget is remaining - 300 = ~100. Strict upper bound
    # 110 leaves some wiggle room for the timing of _t0.
    assert safe_budget <= 110.0, (
        f"safe_budget {safe_budget} does not reserve rollout-overshoot "
        f"headroom — with remaining≈400 ms and overshoot+wrap-up≈300 ms, "
        f"safe_budget should be ≤110 ms, not the full 300."
    )


def test_mcts_agent_skips_search_when_budget_wont_cover_overshoot():
    """When outer remaining < overshoot reserve, search is skipped and
    we return the staged heuristic action. This prevents the case where
    a 50 ms search + 260 ms overshoot bolts onto an already-burned turn
    and breaches HARD_DEADLINE_MS."""
    agent = MCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=4, total_sims=64, rollout_depth=2,
            hard_deadline_ms=300.0,
            anchor_improvement_margin=10.0,
        ),
        rng_seed=0,
    )
    dl = Deadline()
    # 0.7s already elapsed → 200 ms remaining — below 260 ms overshoot reserve.
    dl._t0 -= 0.700

    t0 = time.perf_counter()
    action = agent.act(_mk_obs(step=10), dl)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(action, list)
    # Skipped search → heuristic-only latency. Should be well under 100 ms.
    assert elapsed_ms < 150.0, f"search not skipped; elapsed={elapsed_ms:.1f} ms"


def test_mcts_agent_preserves_margin_in_tight_cfg(monkeypatch):
    """Regression test: when mcts_bot rebuilds GumbelConfig to tighten
    the deadline, it MUST carry over `anchor_improvement_margin` from
    the original config. Dropping it silently reverts to the default
    (0.15), which would let rollout noise overwrite the anchor and
    cause the bot to lose the heuristic floor guarantee."""
    observed: list = []

    original_search = None

    def _capturing_search(self, obs, my_player, start_time=None, anchor_action=None, **kwargs):
        # **kwargs absorbs outer_hard_stop_at (added in the audit tail fix).
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


# ---- Decoupled sim-move wiring (Path B / W3) ---------------------------

def test_mcts_agent_populates_opp_candidate_builder_on_split_posterior():
    """With a posterior split ~60/30 across two archetypes, MCTSAgent
    populates ``opp_candidate_builder`` so the search's decoupled UCB
    branch fires. When the builder is called, it must return >=2
    distinct wire actions — otherwise the search falls back to SH."""
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
    post._turns_observed = agent._POSTERIOR_MIN_TURNS

    # Force a top-2 split: rusher ~0.6, turtler ~0.3, rest ~noise.
    # Concrete log-alphas with softmax > _POSTERIOR_DECOUPLED_MIN_SECOND_PROB
    # on both positions.
    post.log_alpha = np.full(post.K, -5.0)
    post.log_alpha[post.names.index("rusher")] = 2.0
    post.log_alpha[post.names.index("turtler")] = 1.3

    agent._maybe_route_posterior_to_search()
    assert agent._search.opp_candidate_builder is not None, (
        "split posterior should populate the decoupled builder"
    )
    assert agent.telemetry["builder_fires"] == 1

    # Builder must produce >=2 distinct wires.
    wires = agent._search.opp_candidate_builder(_mk_obs(step=10), opp_player=1)
    assert isinstance(wires, list)
    assert len(wires) >= 2


def test_mcts_agent_no_opp_candidate_builder_on_winner_take_all():
    """Winner-take-all posterior (top ~0.9): the 2nd archetype is below
    the decoupled gate, so no builder is installed. Single-archetype
    ``opp_policy_override`` is strictly stronger in that regime."""
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
    post._turns_observed = agent._POSTERIOR_MIN_TURNS

    # Single dominant archetype — 2nd-top well below threshold.
    post.log_alpha = np.full(post.K, -10.0)
    post.log_alpha[post.names.index("rusher")] = 5.0

    agent._maybe_route_posterior_to_search()
    # Override set (single-archetype path), but builder stays None.
    assert agent._search.opp_policy_override is not None
    assert agent._search.opp_candidate_builder is None


def test_mcts_agent_clears_opp_candidate_builder_on_posterior_collapse():
    """When the posterior de-concentrates (match ends, new opp), the
    builder must clear along with the override so the next match starts
    from a clean slate."""
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
    post._turns_observed = agent._POSTERIOR_MIN_TURNS

    # First: arm the builder with a split posterior.
    post.log_alpha = np.full(post.K, -5.0)
    post.log_alpha[post.names.index("rusher")] = 2.0
    post.log_alpha[post.names.index("turtler")] = 1.3
    agent._maybe_route_posterior_to_search()
    assert agent._search.opp_candidate_builder is not None

    # Now collapse to near-uniform — both override and builder clear.
    post.log_alpha = np.zeros(post.K)
    agent._maybe_route_posterior_to_search()
    assert agent._search.opp_policy_override is None
    assert agent._search.opp_candidate_builder is None


def test_mcts_agent_decoupled_flag_DIAGNOSTIC_disabled_default():
    """DIAGNOSTIC (2026-04-28): under Phantom 4.0 bug,
    ``use_decoupled_sim_move=True`` was silently dropped from tight_cfg
    on every act() call, so it had no runtime effect despite being
    set in MCTSAgent.__init__. After the bugfix it actually fires —
    and Kaggle errors. Diagnostic state: __init__ does NOT force
    True; default is GumbelConfig's default False. Caller can set
    True via gumbel_cfg arg explicitly. Restore True default once
    the decoupled-path Kaggle bug is isolated."""
    agent = MCTSAgent(rng_seed=0)
    assert agent.gumbel_cfg.use_decoupled_sim_move is False
    assert agent._search.gumbel_cfg.use_decoupled_sim_move is False
    # Caller-explicit True still works.
    explicit = MCTSAgent(
        gumbel_cfg=GumbelConfig(use_decoupled_sim_move=True), rng_seed=0,
    )
    assert explicit.gumbel_cfg.use_decoupled_sim_move is True


def test_mcts_agent_bokr_refinement_defaults_off():
    """Shipped MCTSAgent defaults to BOKR angle refinement OFF
    (single-angle per target, n_grid=1) because the 3-variant expansion
    pushed the turn-time tail past Kaggle's 1s actTimeout in audit
    (seed=42 @ 300ms deadline: max=1156ms, 2 turns >= 900ms).

    The bokr_widen module is still wired into generate_per_planet_moves
    — callers can opt in by passing an ActionConfig with
    ``angle_refinement_n_grid > 1``. This test pins the default to OFF
    so we can't silently regress submission safety."""
    agent = MCTSAgent(rng_seed=0)
    assert agent.action_cfg.angle_refinement_n_grid == 1
    # Explicit opt-in still works:
    from orbitwars.mcts.actions import ActionConfig
    agent_opt = MCTSAgent(
        action_cfg=ActionConfig(
            angle_refinement_n_grid=3,
            angle_refinement_range=0.1,
            max_per_planet=10,
        ),
        rng_seed=0,
    )
    assert agent_opt.action_cfg.angle_refinement_n_grid == 3


def test_mcts_agent_telemetry_resets_builder_counters_on_step_zero():
    """Turn-0 reset zeroes builder_fires and builder_clears so smoke
    harnesses running multiple games see clean counts per-match."""
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
    post._turns_observed = agent._POSTERIOR_MIN_TURNS
    post.log_alpha = np.full(post.K, -5.0)
    post.log_alpha[post.names.index("rusher")] = 2.0
    post.log_alpha[post.names.index("turtler")] = 1.3
    agent._maybe_route_posterior_to_search()
    assert agent.telemetry["builder_fires"] == 1

    # New match at step 0 → builder_fires resets.
    agent.act(_mk_obs(step=0), Deadline())
    assert agent.telemetry["builder_fires"] == 0
    assert agent.telemetry["builder_clears"] == 0
    # And the underlying builder is cleared on the search too.
    assert agent._search.opp_candidate_builder is None


def test_fallback_turn_counter_tracks_shadow_at_seat_1():
    """Regression: MCTSAgent._fallback._turn_counter must advance in
    lockstep with a standalone HeuristicAgent at seat 1 (obs.step None).

    The original bug: act() called self._fallback.act() BEFORE the
    fresh-game reset that replaced self._fallback. On turn 1 the old
    fallback's counter advanced 0→1, then got discarded. On turn 2 the
    new fallback's counter advanced None→1 instead of 1→2, leaving it
    permanently one behind a parallel shadow HeuristicAgent. Because
    MCTSAgent threads ``step_override = fallback._turn_counter`` into
    search, the anchor heuristic_move was computed at step N-1 while
    the engine observed it at step N — silently breaking anchor-lock
    at seat 1 (3/30 turns diverged vs. 0/30 at seat 0; confirmed by
    tools/diag_mcts_vs_heur_actions_seat1.py prior to the fix).

    This test simulates a seat-1 stream (step stripped from obs) and
    asserts counter-parity across 10 turns.
    """
    from orbitwars.bots.heuristic import HeuristicAgent

    mcts = MCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=2, total_sims=2, rollout_depth=1,
            hard_deadline_ms=2000.0,
            anchor_improvement_margin=10.0,
        ),
        rng_seed=0,
    )
    shadow = HeuristicAgent()

    def _strip_step(o):
        # Emulate the Kaggle seat-1 obs shape (obs.step is None / absent).
        o2 = dict(o)
        o2.pop("step", None)
        return o2

    # 10 turns with obs.step omitted. next_fleet_id stays 0 (no launches
    # in the synthetic obs), which is fine — match-start is detected on
    # the first call (prev_nfid is None), not on nfid regression.
    for _ in range(10):
        obs = _strip_step(_mk_obs())
        mcts.act(obs, Deadline())
        shadow.act(obs, Deadline())
        assert mcts._fallback._turn_counter == shadow._turn_counter, (
            f"Counter drift: mcts._fallback={mcts._fallback._turn_counter} "
            f"vs shadow={shadow._turn_counter}"
        )


def test_phantom4_tight_cfg_preserves_all_configured_fields(monkeypatch):
    """Regression test for PHANTOM 4.0 (2026-04-27).

    Before the fix, ``mcts_bot.act()`` rebuilt a ``tight_cfg`` for each
    search call to inject the safe-budget deadline, but the constructor
    only copied 5 fields and silently reverted ``rollout_policy``,
    ``sim_move_variant``, ``exp3_eta``, ``use_decoupled_sim_move``,
    ``use_macros``, ``per_rollout_budget_ms``, ``num_opp_candidates``
    to their GumbelConfig defaults. Every Kaggle ladder submission since
    v9 had been running with default config instead of bundle-time
    settings.

    This test pins the fix by intercepting the search call and asserting
    the cfg used for search has the configured (non-default) values.
    """
    from orbitwars.mcts.gumbel_search import GumbelRootSearch

    cfg = GumbelConfig(
        # Configure values that are DIFFERENT from the GumbelConfig defaults.
        rollout_policy="nn_value",        # default: "heuristic"
        sim_move_variant="exp3",           # default: "ucb"
        exp3_eta=0.7,                       # default: 0.3
        use_decoupled_sim_move=False,      # default: True (also overwritten in __init__ to True!)
        use_macros=True,                    # default: False
        num_opp_candidates=99,              # default: 5
        per_rollout_budget_ms=42.0,         # default: None
        num_candidates=4, total_sims=4, rollout_depth=1, hard_deadline_ms=2000.0,
        anchor_improvement_margin=2.0,
    )
    agent = MCTSAgent(gumbel_cfg=cfg, rng_seed=0)
    # Intercept the search call to capture the cfg actually used.
    captured = {}
    original_search = agent._search.search

    def capture_search(*args, **kwargs):
        captured["gumbel_cfg"] = agent._search.gumbel_cfg
        # Snapshot the fields at the moment of search invocation.
        c = agent._search.gumbel_cfg
        captured["snapshot"] = {
            "rollout_policy": c.rollout_policy,
            "sim_move_variant": c.sim_move_variant,
            "exp3_eta": c.exp3_eta,
            "use_macros": c.use_macros,
            "num_opp_candidates": c.num_opp_candidates,
            "per_rollout_budget_ms": c.per_rollout_budget_ms,
            "anchor_improvement_margin": c.anchor_improvement_margin,
            "total_sims": c.total_sims,
        }
        return original_search(*args, **kwargs)

    monkeypatch.setattr(agent._search, "search", capture_search)
    agent.act(_mk_obs(), Deadline())

    # All configured fields should be preserved at search time.
    snap = captured.get("snapshot", {})
    assert snap.get("rollout_policy") == "nn_value", (
        f"rollout_policy was reverted to default: got {snap.get('rollout_policy')!r}"
    )
    assert snap.get("sim_move_variant") == "exp3", (
        f"sim_move_variant was reverted: got {snap.get('sim_move_variant')!r}"
    )
    assert snap.get("exp3_eta") == 0.7, (
        f"exp3_eta was reverted: got {snap.get('exp3_eta')!r}"
    )
    assert snap.get("use_macros") is True, (
        f"use_macros was reverted: got {snap.get('use_macros')!r}"
    )
    assert snap.get("num_opp_candidates") == 99, (
        f"num_opp_candidates was reverted: got {snap.get('num_opp_candidates')!r}"
    )
    assert snap.get("per_rollout_budget_ms") == 42.0, (
        f"per_rollout_budget_ms was reverted: got {snap.get('per_rollout_budget_ms')!r}"
    )
    # Anchor margin and total_sims were preserved by the original code,
    # but pin them too as a sanity check.
    assert snap.get("anchor_improvement_margin") == 2.0
    assert snap.get("total_sims") == 4


def test_phantom5_fresh_game_preserves_move_prior_fn_and_value_fn():
    """Regression test for PHANTOM 5.0 (2026-04-28).

    Before the fix, ``mcts_bot.act()`` detected ``fresh_game`` (turn 0)
    and rebuilt ``self._search = GumbelRootSearch(...)`` WITHOUT
    threading ``move_prior_fn`` or ``value_fn`` from the previous
    instance. Both fields default to None on the dataclass, so the NN
    prior + NN value head were silently DISABLED at the start of every
    match. This is the second Phantom-class bug: an internal rebuild
    quietly drops configured behavior.

    Symptom in the wild: ``rollout_policy='nn_value'`` bundles fell back
    to heuristic rollouts via the
    ``rollout_policy='nn_value' but no value_fn supplied`` path. The
    warning fires once per process so multi-game self-play didn't
    surface it after game 1.
    """
    sentinel_prior = object()
    sentinel_value = object()
    # Stub the dataclass so we can assert without a real model.
    agent = MCTSAgent(rng_seed=0)
    agent._search.move_prior_fn = sentinel_prior  # type: ignore[assignment]
    agent._search.value_fn = sentinel_value  # type: ignore[assignment]

    # Drive one act() cycle. Turn 0 obs => fresh_game branch fires.
    agent.act(_mk_obs(step=0), Deadline())

    assert agent._search.move_prior_fn is sentinel_prior, (
        "fresh_game rebuild dropped move_prior_fn — Phantom 5.0 regression"
    )
    assert agent._search.value_fn is sentinel_value, (
        "fresh_game rebuild dropped value_fn — Phantom 5.0 regression"
    )
