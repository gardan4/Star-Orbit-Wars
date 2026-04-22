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
