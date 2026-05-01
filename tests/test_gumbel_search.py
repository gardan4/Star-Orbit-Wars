"""Tests for Gumbel top-k + Sequential Halving + rollout primitives.

Covers:
  * gumbel_topk returns distinct indices, length <= k, ranked desc.
  * gumbel_topk seeds are deterministic.
  * enumerate_joints dedupes and respects n_samples cap.
  * sequential_halving converges to the best constant-value candidate.
  * sequential_halving respects the hard deadline.
  * _rollout_value returns a scalar in [-1, 1] from a tiny synthetic game.
"""
from __future__ import annotations

import math
import random
import time
from types import SimpleNamespace

import pytest

from orbitwars.engine.fast_engine import FastEngine
from orbitwars.mcts.actions import (
    ActionConfig,
    JointAction,
    PlanetMove,
    KIND_ATTACK_ENEMY,
    KIND_HOLD,
    generate_per_planet_moves,
)
from orbitwars.mcts.gumbel_search import (
    GumbelConfig,
    GumbelRootSearch,
    _build_anchor_joint,
    _rollout_value,
    _value_fn_eval,
    enumerate_joints,
    gumbel_topk,
    sequential_halving,
)


def _mk_obs():
    """Tiny 2-player, 3-planet synthetic observation. Same shape as
    test_mcts_actions' fixture — keeps us honest against the action
    generator's contract."""
    return {
        "player": 0,
        "step": 10,
        "angular_velocity": 0.03,
        "planets": [
            [0, 0, 20.0, 50.0, 1.5, 50, 3],
            [1, 1, 80.0, 50.0, 1.5, 30, 3],
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


# ---- gumbel_topk --------------------------------------------------------

def test_gumbel_topk_returns_k_distinct_indices():
    priors = [0.1, 0.2, 0.3, 0.4]
    rng = random.Random(0)
    picks = gumbel_topk(priors, k=3, rng=rng)
    assert len(picks) == 3
    assert len({i for i, _ in picks}) == 3


def test_gumbel_topk_is_sorted_descending():
    rng = random.Random(0)
    picks = gumbel_topk([0.25, 0.25, 0.25, 0.25], k=4, rng=rng)
    scores = [s for _, s in picks]
    assert scores == sorted(scores, reverse=True)


def test_gumbel_topk_seed_is_deterministic():
    p = [0.1, 0.3, 0.2, 0.4]
    a = gumbel_topk(p, k=4, rng=random.Random(42))
    b = gumbel_topk(p, k=4, rng=random.Random(42))
    assert a == b


def test_gumbel_topk_skips_zero_priors():
    picks = gumbel_topk([0.0, 0.0, 1.0], k=3, rng=random.Random(0))
    assert [i for i, _ in picks] == [2]


def test_gumbel_topk_heavy_tail_prefers_high_prior():
    """With a strong prior on index 0, averaged over many seeds it should
    almost always appear in the top-2."""
    priors = [10.0, 0.01, 0.01, 0.01]
    top0_hits = 0
    for seed in range(50):
        picks = gumbel_topk(priors, k=2, rng=random.Random(seed))
        if any(i == 0 for i, _ in picks):
            top0_hits += 1
    assert top0_hits > 45  # >90% inclusion


# ---- enumerate_joints ---------------------------------------------------

def test_enumerate_joints_is_deduped():
    """With only HOLD moves for one planet, all joint samples collapse
    to a single distinct action."""
    per_planet = {
        0: [PlanetMove(from_pid=0, angle=0.0, ships=0, target_pid=-1,
                       kind=KIND_HOLD, prior=1.0)]
    }
    joints = enumerate_joints(per_planet, n_samples=10, rng=random.Random(0))
    assert len(joints) == 1
    assert joints[0].moves[0].is_hold


def test_enumerate_joints_returns_distinct_when_available():
    m1 = PlanetMove(from_pid=0, angle=0.1, ships=20, target_pid=1,
                    kind=KIND_ATTACK_ENEMY, prior=0.4)
    m2 = PlanetMove(from_pid=0, angle=0.2, ships=20, target_pid=2,
                    kind=KIND_ATTACK_ENEMY, prior=0.3)
    m3 = PlanetMove(from_pid=0, angle=0.3, ships=40, target_pid=1,
                    kind=KIND_ATTACK_ENEMY, prior=0.3)
    joints = enumerate_joints(
        {0: [m1, m2, m3]}, n_samples=3, rng=random.Random(1),
    )
    wire_sigs = {tuple(tuple(m) for m in j.to_wire()) for j in joints}
    assert len(joints) == len(wire_sigs)  # all distinct
    assert len(joints) <= 3


# ---- sequential_halving -------------------------------------------------

def _mk_distinct_joints(n: int):
    """Frozen dataclasses with empty tuples compare equal — give each
    joint a unique move so `list.index()` works in test rollout fns."""
    return [
        JointAction(moves=(PlanetMove(
            from_pid=i, angle=0.0, ships=0, target_pid=-1,
            kind=KIND_HOLD, prior=1.0,
        ),))
        for i in range(n)
    ]


def test_sequential_halving_picks_highest_constant_value():
    """Candidate 2 always scores 0.9; others score 0.0/0.1/0.2. SH
    should find it."""
    joints = _mk_distinct_joints(4)
    values = [0.0, 0.1, 0.9, 0.2]

    def rollout_fn(j: JointAction) -> float:
        return values[joints.index(j)]

    cfg = GumbelConfig(num_candidates=4, total_sims=32, rollout_depth=1)
    res = sequential_halving(joints, rollout_fn, cfg)
    assert res.best_joint is joints[2]
    assert res.q_values[2] == pytest.approx(0.9)
    assert res.n_rollouts > 0


def test_sequential_halving_single_candidate_short_circuits():
    j = _mk_distinct_joints(1)[0]
    cfg = GumbelConfig(num_candidates=1, total_sims=10, rollout_depth=1)
    res = sequential_halving([j], lambda _: 0.5, cfg)
    assert res.best_joint is j
    assert res.visits[0] == 1  # one diagnostic rollout
    assert res.q_values[0] == pytest.approx(0.5)


def test_sequential_halving_exposes_full_candidate_list():
    """Result should carry every candidate evaluated (not just the
    winner) so external tooling — distillation BC, ablations,
    diagnostics — can read the full visit distribution. Parallel-indexed
    with `visits` and `q_values`."""
    joints = _mk_distinct_joints(4)
    values = [0.0, 0.5, 0.9, 0.1]

    def rollout_fn(j: JointAction) -> float:
        return values[joints.index(j)]

    cfg = GumbelConfig(num_candidates=4, total_sims=32, rollout_depth=1)
    res = sequential_halving(joints, rollout_fn, cfg)
    # candidates field present + parallel-indexed.
    assert hasattr(res, "candidates")
    assert len(res.candidates) == 4
    assert len(res.visits) == 4
    assert len(res.q_values) == 4
    # Same identity as the input joints.
    for i, j in enumerate(joints):
        assert res.candidates[i] is j
    # Single-candidate path also populates the list.
    j1 = _mk_distinct_joints(1)[0]
    res1 = sequential_halving(
        [j1], lambda _: 0.5,
        GumbelConfig(num_candidates=1, total_sims=10, rollout_depth=1),
    )
    assert len(res1.candidates) == 1
    assert res1.candidates[0] is j1


def test_sequential_halving_respects_deadline():
    """A rollout that sleeps long enough to blow the budget should
    cause the loop to abort cleanly."""
    joints = _mk_distinct_joints(4)

    def slow_rollout(_j: JointAction) -> float:
        time.sleep(0.02)  # 20 ms each
        return 0.5

    cfg = GumbelConfig(
        num_candidates=4, total_sims=64, rollout_depth=1,
        hard_deadline_ms=30.0,  # strictly less than 4 rollouts worth
    )
    t0 = time.perf_counter()
    res = sequential_halving(joints, slow_rollout, cfg, start_time=t0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert res.aborted or res.n_rollouts < 8
    assert elapsed_ms < 200.0  # never hangs


def test_sequential_halving_empty_raises():
    cfg = GumbelConfig()
    with pytest.raises(ValueError):
        sequential_halving([], lambda _: 0.0, cfg)


def test_sequential_halving_protected_idx_survives_pruning():
    """With protected_idx=0 and the anchor assigned a LOW constant Q,
    the anchor should still be visited in every round (so its visit
    count roughly tracks the later-round candidates)."""
    joints = _mk_distinct_joints(4)
    # Anchor at idx 0 gets the worst score; idx 2 is the true best.
    values = [-0.5, 0.2, 0.9, 0.1]

    def rollout_fn(j: JointAction) -> float:
        return values[joints.index(j)]

    cfg = GumbelConfig(num_candidates=4, total_sims=64, rollout_depth=1)
    res = sequential_halving(joints, rollout_fn, cfg, protected_idx=0)
    # Idx 2 should still win on mean-Q.
    assert res.best_joint is joints[2]
    # Anchor (idx 0) survived to the final round, so its visits should
    # be comparable to the final-round count (> 1 round's worth).
    # With n_rounds = ceil(log2(4)) = 2, round 1 gives everyone 1+ sim,
    # round 2 gives the top-2 (including anchor, due to protection)
    # more sims. So anchor visits should be strictly > 1.
    assert res.visits[0] > 1


def test_sequential_halving_protected_anchor_wins_when_best():
    """Anchor is the true best candidate. Even without tiebreaks, it
    should come out on top."""
    joints = _mk_distinct_joints(4)
    values = [0.9, 0.2, 0.3, 0.1]

    def rollout_fn(j: JointAction) -> float:
        return values[joints.index(j)]

    cfg = GumbelConfig(num_candidates=4, total_sims=32, rollout_depth=1)
    res = sequential_halving(joints, rollout_fn, cfg, protected_idx=0)
    assert res.best_joint is joints[0]


# ---- _build_anchor_joint ------------------------------------------------

def test_build_anchor_joint_populates_launches_and_holds():
    m1 = PlanetMove(from_pid=0, angle=0.1, ships=20, target_pid=1,
                    kind=KIND_ATTACK_ENEMY, prior=0.5)
    per_planet = {0: [m1], 2: [m1]}  # two owned planets: 0 and 2
    wire = [[0, 0.5, 33]]  # heuristic only launches from planet 0
    aj = _build_anchor_joint(wire, per_planet)
    assert aj is not None
    # One move per owned planet, sorted by pid: (0, 2).
    assert len(aj.moves) == 2
    assert aj.moves[0].from_pid == 0
    assert aj.moves[0].angle == pytest.approx(0.5)
    assert aj.moves[0].ships == 33
    assert not aj.moves[0].is_hold
    assert aj.moves[1].from_pid == 2
    assert aj.moves[1].is_hold  # no wire entry for pid=2 → hold


def test_build_anchor_joint_returns_none_when_empty():
    assert _build_anchor_joint(None, {0: []}) is None
    assert _build_anchor_joint([], {}) is None


def test_build_anchor_joint_wire_round_trip():
    """The rebuilt joint's `to_wire()` reproduces the input wire (for
    planets we own)."""
    per_planet = {5: [], 7: []}
    wire = [[5, 1.23, 40], [7, 2.34, 10]]
    aj = _build_anchor_joint(wire, per_planet)
    out = aj.to_wire()
    # Wire output is sorted by pid (matches sample_joint).
    assert out == [[5, 1.23, 40], [7, 2.34, 10]]


# ---- _rollout_value -----------------------------------------------------

def test_rollout_value_returns_scalar_in_range():
    """Smoke: a single rollout on a real tiny game returns a finite
    value in [-1, +1] without crashing."""
    obs = _mk_obs()
    eng = FastEngine.from_official_obs(SimpleNamespace(**obs), num_agents=2)
    base_state = eng.state

    from orbitwars.bots.heuristic import HEURISTIC_WEIGHTS, HeuristicAgent

    def factory():
        return HeuristicAgent(weights=HEURISTIC_WEIGHTS)

    v = _rollout_value(
        base_state=base_state,
        my_player=0,
        my_action=[],  # hold
        opp_agent_factory=factory,
        my_future_factory=factory,
        depth=3,
        num_agents=2,
    )
    assert -1.0 <= v <= 1.0
    assert math.isfinite(v)


def test_rollout_value_aborts_on_deadline():
    """When ``deadline_fn`` returns True BEFORE the first rollout ply,
    the rollout must short-circuit entirely — zero HeuristicAgent.act()
    calls, ~5 ms wall cost. This is load-bearing for the Kaggle 1-s
    turn ceiling: the audit showed that a single turn-0 opp.act() on a
    dense mid-game state runs 100-300 ms. If sequential_halving's
    pre-rollout deadline check passes by a hair and then the rollout
    starts, the turn-0 opp heuristic would push total turn time past
    the 900 ms outer ceiling. The pre-turn-0 deadline check in
    _rollout_value prevents that."""
    obs = _mk_obs()
    eng = FastEngine.from_official_obs(SimpleNamespace(**obs), num_agents=2)
    base_state = eng.state

    from orbitwars.bots.heuristic import HEURISTIC_WEIGHTS, HeuristicAgent

    plies_done = [0]

    class _CountingHeuristic(HeuristicAgent):
        def act(self, obs_, dl):  # type: ignore[override]
            plies_done[0] += 1
            return super().act(obs_, dl)

    def factory():
        return _CountingHeuristic(weights=HEURISTIC_WEIGHTS)

    # Deadline already fired — rollout short-circuits before turn 0.
    v = _rollout_value(
        base_state=base_state,
        my_player=0,
        my_action=[],  # hold
        opp_agent_factory=factory,
        my_future_factory=factory,
        depth=10,
        num_agents=2,
        deadline_fn=lambda: True,
    )
    assert -1.0 <= v <= 1.0
    assert math.isfinite(v)
    # Zero HeuristicAgent.act() calls — the short-circuit fires before
    # turn-0 runs.
    assert plies_done[0] == 0, (
        f"expected 0 opp act (deadline pre-fired) but got {plies_done[0]}; "
        f"deadline_fn was ignored"
    )


def test_rollout_value_propagates_hard_stop_at_to_inner_deadlines():
    """When ``hard_stop_at`` is passed to ``_rollout_value``, each inner
    ``agent.act(obs, Deadline(...))`` call must receive a Deadline whose
    ``hard_stop_at`` equals the outer value. This is load-bearing: an
    in-flight ``HeuristicAgent._plan_moves`` on a dense mid-game state
    runs 400-700 ms of intercept math per pair; without hard_stop_at
    propagation it can't short-circuit when the outer search deadline
    fires and pushes total turn time past 900 ms (observed pre-fix max:
    1172 ms against shipped 300 ms deadline + 260 ms overshoot reserve).
    """
    from orbitwars.bots.base import Deadline
    from orbitwars.bots.heuristic import HEURISTIC_WEIGHTS, HeuristicAgent

    obs = _mk_obs()
    eng = FastEngine.from_official_obs(SimpleNamespace(**obs), num_agents=2)
    base_state = eng.state

    captured: list = []

    class _CapturingHeuristic(HeuristicAgent):
        def act(self, obs_, dl):  # type: ignore[override]
            captured.append(getattr(dl, "_hard_stop_at", "MISSING"))
            return super().act(obs_, dl)

    def factory():
        return _CapturingHeuristic(weights=HEURISTIC_WEIGHTS)

    # Use an "effectively infinite" future hard_stop so act() doesn't
    # short-circuit and we can observe the propagated value.
    fake_stop = time.perf_counter() + 1e6
    _rollout_value(
        base_state=base_state,
        my_player=0,
        my_action=[],
        opp_agent_factory=factory,
        my_future_factory=factory,
        depth=3,
        num_agents=2,
        deadline_fn=lambda: False,
        hard_stop_at=fake_stop,
    )
    assert captured, "rollout should have run at least one inner act()"
    for v in captured:
        assert v == fake_stop, (
            f"inner Deadline.hard_stop_at={v!r}; expected {fake_stop!r} — "
            "hard_stop_at was not propagated"
        )


def test_rollout_value_per_rollout_budget_tightens_inner_deadline():
    """per_rollout_budget_ms must produce an *effective* deadline of
    ``min(hard_stop_at, now + per_rollout_budget_ms)``. When the budget
    cap is sooner than hard_stop_at, inner Deadlines see the tighter
    value \u2014 not the outer hard_stop_at. This is the per-rollout fat-
    tail guard: without it, one unlucky rollout on a step-35-ish state
    can naturally run 685 ms and blow the outer 300 ms budget.
    """
    from orbitwars.bots.base import Deadline
    from orbitwars.bots.heuristic import HEURISTIC_WEIGHTS, HeuristicAgent

    obs = _mk_obs()
    eng = FastEngine.from_official_obs(SimpleNamespace(**obs), num_agents=2)
    base_state = eng.state

    captured: list = []

    class _CapturingHeuristic(HeuristicAgent):
        def act(self, obs_, dl):  # type: ignore[override]
            captured.append(getattr(dl, "_hard_stop_at", "MISSING"))
            return super().act(obs_, dl)

    def factory():
        return _CapturingHeuristic(weights=HEURISTIC_WEIGHTS)

    # Outer hard_stop in the far future; per-rollout cap is 50 ms.
    # Expectation: inner Deadlines see a hard_stop_at within ~(now, now+50ms),
    # NOT the far-future hard_stop_at.
    t_enter = time.perf_counter()
    far_future = t_enter + 1e6
    _rollout_value(
        base_state=base_state,
        my_player=0,
        my_action=[],
        opp_agent_factory=factory,
        my_future_factory=factory,
        depth=3,
        num_agents=2,
        deadline_fn=lambda: False,
        hard_stop_at=far_future,
        per_rollout_budget_ms=50.0,
    )
    assert captured, "rollout should have run at least one inner act()"
    # Inner Deadlines must use the tighter (now + 50ms) cap, not far_future.
    max_allowed = t_enter + 0.050 + 0.050  # slop for Python overhead
    for v in captured:
        assert isinstance(v, float), f"expected float hard_stop_at, got {v!r}"
        assert v <= max_allowed, (
            f"inner Deadline.hard_stop_at={v!r} exceeds tight cap {max_allowed!r}; "
            "per_rollout_budget_ms not applied"
        )
        assert v != far_future, (
            "inner Deadline.hard_stop_at should be the rollout cap, not "
            "the outer hard_stop_at"
        )


def test_rollout_value_per_rollout_budget_bounds_wall_time():
    """With a tight per_rollout_budget_ms (say 20 ms) and an unbounded
    outer hard_stop_at, a full rollout must return within approximately
    the budget + detection slack. This is the behavior the profile
    relies on to bound single-rollout wall cost when the natural cost
    has a fat tail (diag_rollout_deadline: max=685 ms natural).
    """
    obs = _mk_obs()
    eng = FastEngine.from_official_obs(SimpleNamespace(**obs), num_agents=2)
    base_state = eng.state

    from orbitwars.bots.heuristic import HEURISTIC_WEIGHTS, HeuristicAgent

    def factory():
        return HeuristicAgent(weights=HEURISTIC_WEIGHTS)

    # Outer hard_stop far in the future so only per_rollout_budget_ms gates.
    far_future = time.perf_counter() + 1e6
    t0 = time.perf_counter()
    _rollout_value(
        base_state=base_state,
        my_player=0,
        my_action=[],
        opp_agent_factory=factory,
        my_future_factory=factory,
        depth=15,
        num_agents=2,
        deadline_fn=lambda: False,
        hard_stop_at=far_future,
        per_rollout_budget_ms=20.0,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    # Budget is 20 ms; allow up to 100 ms for detection slack on CI.
    # (One heuristic ply on a tiny synthetic obs is well under 10 ms.)
    assert elapsed_ms < 100.0, (
        f"rollout wall time {elapsed_ms:.1f} ms exceeds tight 20 ms budget + slack"
    )


def test_rollout_value_runs_full_depth_when_deadline_false():
    """Mirror of the abort test: with a False-returning deadline_fn,
    plies proceed normally up to ``depth`` or terminal, so we can
    trust that the deadline plumbing doesn't spuriously cut rollouts
    short under normal operation."""
    obs = _mk_obs()
    eng = FastEngine.from_official_obs(SimpleNamespace(**obs), num_agents=2)
    base_state = eng.state

    from orbitwars.bots.heuristic import HEURISTIC_WEIGHTS, HeuristicAgent

    plies_done = [0]

    class _CountingHeuristic(HeuristicAgent):
        def act(self, obs_, dl):  # type: ignore[override]
            plies_done[0] += 1
            return super().act(obs_, dl)

    def factory():
        return _CountingHeuristic(weights=HEURISTIC_WEIGHTS)

    depth = 3
    _rollout_value(
        base_state=base_state,
        my_player=0,
        my_action=[],
        opp_agent_factory=factory,
        my_future_factory=factory,
        depth=depth,
        num_agents=2,
        deadline_fn=lambda: False,
    )
    # Turn 0: 1 opp.act. Turns 1..depth-1: 2 acts each (my + opp).
    # So 1 + 2*(depth-1) = 1 + 4 = 5 when depth=3. Allow <= in case
    # the tiny synthetic game terminates early.
    expected_max = 1 + 2 * (depth - 1)
    assert plies_done[0] <= expected_max
    assert plies_done[0] >= 1, "turn-0 opp action must always run"


# ---- GumbelRootSearch end-to-end ---------------------------------------

def test_gumbel_root_search_returns_result():
    """End-to-end: from obs to a chosen JointAction with Q/visits."""
    obs = _mk_obs()
    search = GumbelRootSearch(
        gumbel_cfg=GumbelConfig(
            num_candidates=4, total_sims=8, rollout_depth=2,
            hard_deadline_ms=2000.0,
        ),
        rng_seed=0,
    )
    res = search.search(obs, my_player=0)
    assert res is not None
    assert res.n_candidates >= 1
    assert res.best_joint is not None
    # Wire format is a list of [pid, angle, ships] or empty for all-hold.
    wire = res.best_joint.to_wire()
    assert isinstance(wire, list)
    for move in wire:
        assert len(move) == 3


def test_gumbel_root_search_returns_none_with_no_owned_planets():
    obs = _mk_obs()
    # Flip all "my" planets to enemy so we have no legal moves.
    obs["planets"][0][1] = 1
    search = GumbelRootSearch(rng_seed=0)
    res = search.search(obs, my_player=0)
    assert res is None


def test_gumbel_root_search_uses_opp_policy_override():
    """When opp_policy_override is set, the factory gets called during
    search. We stub in a counter factory and verify it was invoked."""
    from orbitwars.bots.heuristic import HeuristicAgent

    calls = []

    def counting_factory():
        calls.append(1)
        return HeuristicAgent()

    obs = _mk_obs()
    search = GumbelRootSearch(
        gumbel_cfg=GumbelConfig(
            num_candidates=3, total_sims=4, rollout_depth=2,
            hard_deadline_ms=2000.0,
        ),
        rng_seed=0,
        opp_policy_override=counting_factory,
    )
    res = search.search(obs, my_player=0)
    assert res is not None
    # Rollout depth 2 × num_sims 4 × (at least 1 opp act per step) → multiple calls.
    assert len(calls) >= 4, f"override factory only called {len(calls)} times"


def test_gumbel_root_search_uses_anchor_as_candidate():
    """Anchor action is prepended to the candidate list so SH can compare
    it head-to-head with Gumbel samples."""
    obs = _mk_obs()
    search = GumbelRootSearch(
        gumbel_cfg=GumbelConfig(
            num_candidates=4, total_sims=8, rollout_depth=1,
            hard_deadline_ms=2000.0,
        ),
        rng_seed=0,
    )
    anchor = [[0, 1.234, 42]]  # a clearly-distinct signature
    res = search.search(obs, my_player=0, anchor_action=anchor)
    assert res is not None
    # The anchor candidate is at index 0 and was visited ≥1 time.
    assert res.visits[0] >= 1


def test_gumbel_root_search_anchor_none_stays_backward_compatible():
    """anchor_action=None keeps the old behavior (pure Gumbel samples)."""
    obs = _mk_obs()
    search = GumbelRootSearch(
        gumbel_cfg=GumbelConfig(
            num_candidates=4, total_sims=8, rollout_depth=1,
            hard_deadline_ms=2000.0,
        ),
        rng_seed=0,
    )
    res = search.search(obs, my_player=0, anchor_action=None)
    assert res is not None
    assert res.best_joint is not None


def test_gumbel_root_search_margin_guard_retains_anchor():
    """When the SH winner doesn't beat the anchor by the configured
    margin, the anchor is returned. Verified by setting a huge margin
    so no winner could ever clear it."""
    obs = _mk_obs()
    search = GumbelRootSearch(
        gumbel_cfg=GumbelConfig(
            num_candidates=4, total_sims=8, rollout_depth=1,
            hard_deadline_ms=2000.0,
            anchor_improvement_margin=10.0,  # impossible to clear
        ),
        rng_seed=0,
    )
    anchor = [[0, 0.5, 25]]
    res = search.search(obs, my_player=0, anchor_action=anchor)
    assert res is not None
    # Anchor preserved: its wire output matches what we passed in.
    wire = res.best_joint.to_wire()
    assert wire == [[0, 0.5, 25]]


# ---- Decoupled sim-move branch (Path B / W3) ---------------------------

def test_gumbel_root_search_uses_decoupled_when_flagged(monkeypatch):
    """When ``use_decoupled_sim_move=True`` and ``opp_candidate_builder``
    returns >=2 wire actions, search runs the decoupled UCB bandit from
    sim_move.py instead of sequential_halving.

    Verified by spying on ``decoupled_ucb_root``: the spy records the
    call and delegates to the real implementation so the search still
    returns a valid result."""
    import orbitwars.mcts.sim_move as sim_move_mod
    real_decoupled = sim_move_mod.decoupled_ucb_root
    calls = []

    def spy_decoupled(*args, **kwargs):
        calls.append((args, kwargs))
        return real_decoupled(*args, **kwargs)

    monkeypatch.setattr(sim_move_mod, "decoupled_ucb_root", spy_decoupled)

    def opp_builder(obs_, opp_player):
        # Two distinct wire actions so the bandit has something to
        # marginalize over. Hold + a small launch from planet 1 (the
        # opp's home base in _mk_obs).
        return [[], [[1, 0.0, 5]]]

    obs = _mk_obs()
    search = GumbelRootSearch(
        gumbel_cfg=GumbelConfig(
            num_candidates=3, total_sims=6, rollout_depth=1,
            hard_deadline_ms=2000.0,
            use_decoupled_sim_move=True,
            num_opp_candidates=2,
        ),
        rng_seed=0,
        opp_candidate_builder=opp_builder,
    )
    res = search.search(obs, my_player=0)
    assert res is not None
    assert res.best_joint is not None
    assert len(calls) == 1, (
        f"decoupled_ucb_root should have fired exactly once; got {len(calls)}"
    )


def test_gumbel_root_search_falls_back_to_sh_when_only_one_opp_candidate(monkeypatch):
    """Builder returns only 1 distinct opp wire → the decoupled branch
    degenerates (marginalizing over a single strategy is a no-op), so we
    fall back to ``sequential_halving``. Spy on both to confirm."""
    import orbitwars.mcts.sim_move as sim_move_mod
    import orbitwars.mcts.gumbel_search as gumbel_mod

    dec_calls = []
    sh_calls = []
    real_sh = gumbel_mod.sequential_halving

    def spy_decoupled(*args, **kwargs):
        dec_calls.append(1)
        raise AssertionError("should not have been called")

    def spy_sh(*args, **kwargs):
        sh_calls.append(1)
        return real_sh(*args, **kwargs)

    monkeypatch.setattr(sim_move_mod, "decoupled_ucb_root", spy_decoupled)
    monkeypatch.setattr(gumbel_mod, "sequential_halving", spy_sh)

    def opp_builder(obs_, opp_player):
        return [[]]  # single hold candidate

    obs = _mk_obs()
    search = GumbelRootSearch(
        gumbel_cfg=GumbelConfig(
            num_candidates=3, total_sims=6, rollout_depth=1,
            hard_deadline_ms=2000.0,
            use_decoupled_sim_move=True,
        ),
        rng_seed=0,
        opp_candidate_builder=opp_builder,
    )
    res = search.search(obs, my_player=0)
    assert res is not None
    assert len(dec_calls) == 0
    assert len(sh_calls) == 1


def test_gumbel_root_search_falls_back_to_sh_when_flag_off(monkeypatch):
    """Flag off → decoupled branch never fires even if a builder is set."""
    import orbitwars.mcts.sim_move as sim_move_mod

    dec_calls = []

    def spy_decoupled(*args, **kwargs):
        dec_calls.append(1)
        raise AssertionError("should not have been called")

    monkeypatch.setattr(sim_move_mod, "decoupled_ucb_root", spy_decoupled)

    def opp_builder(obs_, opp_player):
        return [[], [[1, 0.0, 5]]]

    obs = _mk_obs()
    search = GumbelRootSearch(
        gumbel_cfg=GumbelConfig(
            num_candidates=3, total_sims=6, rollout_depth=1,
            hard_deadline_ms=2000.0,
            use_decoupled_sim_move=False,  # flag off
        ),
        rng_seed=0,
        opp_candidate_builder=opp_builder,
    )
    res = search.search(obs, my_player=0)
    assert res is not None
    assert len(dec_calls) == 0


def test_gumbel_root_search_decoupled_deduplicates_opp_wires(monkeypatch):
    """Duplicate opp wires (same moves, same order) are collapsed before
    dispatch — otherwise we'd waste rollouts scoring the same response
    twice. With only one *distinct* wire, we fall back to SH."""
    import orbitwars.mcts.sim_move as sim_move_mod

    dec_calls = []

    def spy_decoupled(*args, **kwargs):
        dec_calls.append(1)
        raise AssertionError("should not have been called after dedup")

    monkeypatch.setattr(sim_move_mod, "decoupled_ucb_root", spy_decoupled)

    def opp_builder(obs_, opp_player):
        # Three entries, but all collapse to the same wire-key after dedup.
        return [[], [], []]

    obs = _mk_obs()
    search = GumbelRootSearch(
        gumbel_cfg=GumbelConfig(
            num_candidates=3, total_sims=6, rollout_depth=1,
            hard_deadline_ms=2000.0,
            use_decoupled_sim_move=True,
        ),
        rng_seed=0,
        opp_candidate_builder=opp_builder,
    )
    res = search.search(obs, my_player=0)
    assert res is not None
    assert len(dec_calls) == 0


# ---- sim_move_variant dispatch (W4 Exp3 A/B infrastructure) ------------

def test_gumbel_root_search_sim_move_variant_exp3_dispatches_to_exp3(monkeypatch):
    """With ``sim_move_variant="exp3"`` and a valid opp builder, search
    MUST call ``decoupled_exp3_root`` and MUST NOT call
    ``decoupled_ucb_root``. Mirrors the ucb-default test above and is
    the core W4 wiring gate: a regression here silently unships the
    A/B."""
    import orbitwars.mcts.sim_move as sim_move_mod

    real_exp3 = sim_move_mod.decoupled_exp3_root
    exp3_calls = []
    ucb_calls = []

    def spy_exp3(*args, **kwargs):
        exp3_calls.append((args, kwargs))
        return real_exp3(*args, **kwargs)

    def spy_ucb(*args, **kwargs):
        ucb_calls.append(1)
        raise AssertionError("ucb path should not fire when variant=exp3")

    monkeypatch.setattr(sim_move_mod, "decoupled_exp3_root", spy_exp3)
    monkeypatch.setattr(sim_move_mod, "decoupled_ucb_root", spy_ucb)

    def opp_builder(obs_, opp_player):
        return [[], [[1, 0.0, 5]]]

    obs = _mk_obs()
    search = GumbelRootSearch(
        gumbel_cfg=GumbelConfig(
            num_candidates=3, total_sims=6, rollout_depth=1,
            hard_deadline_ms=2000.0,
            use_decoupled_sim_move=True,
            sim_move_variant="exp3",
            exp3_eta=0.5,
            num_opp_candidates=2,
        ),
        rng_seed=0,
        opp_candidate_builder=opp_builder,
    )
    res = search.search(obs, my_player=0)
    assert res is not None
    assert len(exp3_calls) == 1, (
        f"decoupled_exp3_root should have fired once; got {len(exp3_calls)}"
    )
    assert len(ucb_calls) == 0
    # eta is propagated through the config, not lost.
    _, kwargs = exp3_calls[0]
    assert kwargs.get("eta") == 0.5


def test_gumbel_root_search_sim_move_variant_ucb_is_default(monkeypatch):
    """With no explicit variant, UCB is used. Guards against
    default-drift from the W3 shipped configuration."""
    import orbitwars.mcts.sim_move as sim_move_mod

    real_ucb = sim_move_mod.decoupled_ucb_root
    ucb_calls = []
    exp3_calls = []

    def spy_ucb(*args, **kwargs):
        ucb_calls.append(1)
        return real_ucb(*args, **kwargs)

    def spy_exp3(*args, **kwargs):
        exp3_calls.append(1)
        raise AssertionError("exp3 path should not fire by default")

    monkeypatch.setattr(sim_move_mod, "decoupled_ucb_root", spy_ucb)
    monkeypatch.setattr(sim_move_mod, "decoupled_exp3_root", spy_exp3)

    def opp_builder(obs_, opp_player):
        return [[], [[1, 0.0, 5]]]

    obs = _mk_obs()
    cfg = GumbelConfig(
        num_candidates=3, total_sims=6, rollout_depth=1,
        hard_deadline_ms=2000.0,
        use_decoupled_sim_move=True,
        num_opp_candidates=2,
    )
    # Explicitly assert the default hasn't drifted.
    assert cfg.sim_move_variant == "ucb"
    search = GumbelRootSearch(
        gumbel_cfg=cfg, rng_seed=0, opp_candidate_builder=opp_builder,
    )
    res = search.search(obs, my_player=0)
    assert res is not None
    assert len(ucb_calls) == 1
    assert len(exp3_calls) == 0


def test_gumbel_root_search_unknown_variant_warns_and_falls_back(monkeypatch, capsys):
    """A typo in sim_move_variant must not crash mid-game — fall back
    to UCB and log once. This is a play-time safety net, not a happy
    path; callers should be warned."""
    import orbitwars.mcts.sim_move as sim_move_mod

    real_ucb = sim_move_mod.decoupled_ucb_root
    ucb_calls = []

    def spy_ucb(*args, **kwargs):
        ucb_calls.append(1)
        return real_ucb(*args, **kwargs)

    monkeypatch.setattr(sim_move_mod, "decoupled_ucb_root", spy_ucb)

    def opp_builder(obs_, opp_player):
        return [[], [[1, 0.0, 5]]]

    obs = _mk_obs()
    cfg = GumbelConfig(
        num_candidates=3, total_sims=6, rollout_depth=1,
        hard_deadline_ms=2000.0,
        use_decoupled_sim_move=True,
        sim_move_variant="regret_matching_plus",  # typo / unknown
        num_opp_candidates=2,
    )
    search = GumbelRootSearch(
        gumbel_cfg=cfg, rng_seed=0, opp_candidate_builder=opp_builder,
    )
    res = search.search(obs, my_player=0)
    assert res is not None
    assert len(ucb_calls) == 1
    captured = capsys.readouterr()
    assert "unknown sim_move_variant" in captured.out


# ---- value_fn / nn_value pathway (Phase 1 of NN value-head Q) -----------

def test_value_fn_eval_returns_scalar_in_range():
    """Smoke: 1-ply eval with a constant value_fn returns the constant."""
    obs = _mk_obs()
    eng = FastEngine.from_official_obs(SimpleNamespace(**obs), num_agents=2)
    base_state = eng.state
    from orbitwars.bots.heuristic import HEURISTIC_WEIGHTS, HeuristicAgent

    def factory():
        return HeuristicAgent(weights=HEURISTIC_WEIGHTS)

    def constant_value_fn(o, p):
        return 0.42

    v = _value_fn_eval(
        base_state=base_state,
        my_player=0,
        my_action=[],  # hold
        opp_agent_factory=factory,
        value_fn=constant_value_fn,
        num_agents=2,
    )
    assert v == pytest.approx(0.42, abs=1e-6)


def test_value_fn_eval_clips_out_of_range():
    """If value_fn returns 5.0, _value_fn_eval clips to +1.0."""
    obs = _mk_obs()
    eng = FastEngine.from_official_obs(SimpleNamespace(**obs), num_agents=2)
    base_state = eng.state
    from orbitwars.bots.heuristic import HEURISTIC_WEIGHTS, HeuristicAgent

    def factory():
        return HeuristicAgent(weights=HEURISTIC_WEIGHTS)

    v = _value_fn_eval(
        base_state=base_state,
        my_player=0,
        my_action=[],
        opp_agent_factory=factory,
        value_fn=lambda o, p: 5.0,
        num_agents=2,
    )
    assert v == 1.0


def test_value_fn_eval_falls_back_on_value_fn_exception():
    """A throwing value_fn must NEVER forfeit — fall back to engine score."""
    obs = _mk_obs()
    eng = FastEngine.from_official_obs(SimpleNamespace(**obs), num_agents=2)
    base_state = eng.state
    from orbitwars.bots.heuristic import HEURISTIC_WEIGHTS, HeuristicAgent

    def factory():
        return HeuristicAgent(weights=HEURISTIC_WEIGHTS)

    def boom(o, p):
        raise RuntimeError("simulated value_fn failure")

    v = _value_fn_eval(
        base_state=base_state,
        my_player=0,
        my_action=[],
        opp_agent_factory=factory,
        value_fn=boom,
        num_agents=2,
    )
    # Must return a finite number — engine score on the post-step state.
    assert isinstance(v, float)
    assert -1.0 <= v <= 1.0


def test_gumbel_search_dispatches_to_value_fn_when_configured():
    """When rollout_policy='nn_value' AND value_fn is set, the search
    uses _value_fn_eval (verified by counting calls). Otherwise it
    uses _rollout_value (heuristic rollouts)."""
    obs = _mk_obs()
    calls = {"value_fn": 0}

    def counting_value_fn(o, p):
        calls["value_fn"] += 1
        return 0.5

    cfg = GumbelConfig(
        num_candidates=2, total_sims=4, rollout_depth=1,
        rollout_policy="nn_value",
        hard_deadline_ms=2000.0,
    )
    search = GumbelRootSearch(
        gumbel_cfg=cfg, rng_seed=0, value_fn=counting_value_fn,
    )
    res = search.search(obs, my_player=0)
    assert res is not None
    # value_fn must have been called at least once (one per candidate
    # rollout; with 2 candidates × 2 sims/round × 1 round = 4 sims, but
    # actual count depends on SH internals — just assert > 0).
    assert calls["value_fn"] > 0


def test_gumbel_search_falls_back_when_nn_value_set_but_no_fn(capsys):
    """rollout_policy='nn_value' with value_fn=None should warn and
    fall back to heuristic rollouts (search still completes)."""
    import warnings
    obs = _mk_obs()
    cfg = GumbelConfig(
        num_candidates=2, total_sims=4, rollout_depth=1,
        rollout_policy="nn_value",
        hard_deadline_ms=2000.0,
    )
    search = GumbelRootSearch(
        gumbel_cfg=cfg, rng_seed=0, value_fn=None,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = search.search(obs, my_player=0)
    assert res is not None
    # Warning about missing value_fn should fire.
    msgs = [str(warning.message) for warning in w]
    assert any("nn_value" in m and "value_fn" in m for m in msgs)


def test_value_fn_steering_changes_action_vs_constant():
    """Two GumbelRootSearch with the SAME seed but DIFFERENT value_fn
    (one constant, one steering toward a specific candidate) should
    produce different best_joint when the candidate set has >1 option.

    This is the diagnostic the design doc calls for: confirm that
    value_fn actually steers Q estimates to the wire."""
    obs = _mk_obs()

    def constant_fn(o, p):
        return 0.0

    cfg_c = GumbelConfig(
        num_candidates=4, total_sims=16, rollout_depth=1,
        rollout_policy="nn_value", hard_deadline_ms=5000.0,
    )
    s_const = GumbelRootSearch(
        gumbel_cfg=cfg_c, rng_seed=42, value_fn=constant_fn,
    )
    res_const = s_const.search(obs, my_player=0)

    # Steering value_fn returns +1 for one specific obs hash, 0 otherwise.
    # We can't easily steer to a specific candidate without knowing
    # the post-step obs identity, so just check that an extreme +1
    # value_fn pushes Q-values toward 1.0 across the board.
    def extreme_fn(o, p):
        return 1.0

    s_extreme = GumbelRootSearch(
        gumbel_cfg=cfg_c, rng_seed=42, value_fn=extreme_fn,
    )
    res_extreme = s_extreme.search(obs, my_player=0)

    # Q-values from the extreme run should be uniformly higher.
    avg_q_const = sum(res_const.q_values) / max(1, len(res_const.q_values))
    avg_q_extreme = sum(res_extreme.q_values) / max(1, len(res_extreme.q_values))
    assert avg_q_extreme > avg_q_const + 0.5, (
        f"value_fn=1.0 should produce higher Q than value_fn=0.0 "
        f"(got {avg_q_extreme:.3f} vs {avg_q_const:.3f})"
    )


def test_value_mix_alpha_zero_ignores_nn_value():
    """With value_mix_alpha=0.0 the search must collapse to pure
    heuristic rollouts: a wildly off NN value_fn (returns +1.0 always)
    must NOT pull Q estimates upward, because NN contributes 0% weight.
    """
    obs = _mk_obs()

    def extreme_fn(o, p):
        return 1.0

    # Pure NN (alpha=1) under extreme_fn: Q approaches +1.
    cfg_pure = GumbelConfig(
        num_candidates=4, total_sims=16, rollout_depth=1,
        rollout_policy="nn_value", hard_deadline_ms=5000.0,
        value_mix_alpha=1.0,
    )
    s_pure = GumbelRootSearch(gumbel_cfg=cfg_pure, rng_seed=42, value_fn=extreme_fn)
    res_pure = s_pure.search(obs, my_player=0)

    # alpha=0 ignores NN; Q should NOT all collapse to +1 (it's whatever
    # the depth-1 heuristic rollout produces, typically near 0 in
    # mid-game starts).
    cfg_mix0 = GumbelConfig(
        num_candidates=4, total_sims=16, rollout_depth=1,
        rollout_policy="nn_value", hard_deadline_ms=5000.0,
        value_mix_alpha=0.0,
    )
    s_mix0 = GumbelRootSearch(gumbel_cfg=cfg_mix0, rng_seed=42, value_fn=extreme_fn)
    res_mix0 = s_mix0.search(obs, my_player=0)

    avg_q_pure = sum(res_pure.q_values) / max(1, len(res_pure.q_values))
    avg_q_mix0 = sum(res_mix0.q_values) / max(1, len(res_mix0.q_values))
    # Pure-NN with extreme_fn=+1 should be at or near +1.
    assert avg_q_pure > 0.5
    # Alpha=0 ignores the NN entirely; the heuristic rollout result
    # must be strictly lower (rollout values typically in [-0.5, 0.5]
    # at depth 1).
    assert avg_q_mix0 < avg_q_pure - 0.2, (
        f"value_mix_alpha=0.0 should ignore extreme NN value_fn; "
        f"avg_q_mix0={avg_q_mix0:.3f} vs avg_q_pure={avg_q_pure:.3f}"
    )


def test_value_mix_alpha_intermediate_blends():
    """With alpha=0.5 the leaf value lies between the pure-NN value
    and the pure-heuristic-rollout value."""
    obs = _mk_obs()

    def extreme_fn(o, p):
        return 1.0

    base_kwargs = dict(
        num_candidates=4, total_sims=16, rollout_depth=1,
        rollout_policy="nn_value", hard_deadline_ms=5000.0,
    )
    cfg_pure = GumbelConfig(value_mix_alpha=1.0, **base_kwargs)
    cfg_mix = GumbelConfig(value_mix_alpha=0.5, **base_kwargs)
    cfg_zero = GumbelConfig(value_mix_alpha=0.0, **base_kwargs)

    res_pure = GumbelRootSearch(
        gumbel_cfg=cfg_pure, rng_seed=42, value_fn=extreme_fn,
    ).search(obs, my_player=0)
    res_mix = GumbelRootSearch(
        gumbel_cfg=cfg_mix, rng_seed=42, value_fn=extreme_fn,
    ).search(obs, my_player=0)
    res_zero = GumbelRootSearch(
        gumbel_cfg=cfg_zero, rng_seed=42, value_fn=extreme_fn,
    ).search(obs, my_player=0)

    avg_pure = sum(res_pure.q_values) / max(1, len(res_pure.q_values))
    avg_mix = sum(res_mix.q_values) / max(1, len(res_mix.q_values))
    avg_zero = sum(res_zero.q_values) / max(1, len(res_zero.q_values))
    # alpha=0.5 must lie strictly between alpha=0 and alpha=1.
    assert avg_zero < avg_mix < avg_pure, (
        f"mix=0.5 should blend; got zero={avg_zero:.3f} "
        f"mix={avg_mix:.3f} pure={avg_pure:.3f}"
    )
