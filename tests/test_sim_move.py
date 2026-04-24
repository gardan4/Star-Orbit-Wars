"""Tests for `orbitwars.mcts.sim_move` — decoupled root bandits.

Covers:
  * Warm-up phase: every (my, opp) cell gets visited before UCB picks in.
  * UCB convergence: with a clear zero-sum winner, the optimal my-candidate
    accrues the most visits and the best marginal mean-Q.
  * Protected-my-idx guarantee: the anchor's visits >= n_opp_candidates
    even at a tiny sim budget.
  * Deadline short-circuits mid-bandit with ``aborted=True``.
  * Zero-sum bookkeeping: my_q_sum[i] + opp_q_sum[j] summed over all
    rollouts of cell (i, j) equals zero (within float noise).
  * Exp3 sketch returns a valid candidate and non-degenerate stats.
"""
from __future__ import annotations

import math
import time

import pytest

from orbitwars.mcts.actions import JointAction, PlanetMove, KIND_HOLD
from orbitwars.mcts.sim_move import (
    DecoupledSearchResult,
    decoupled_exp3_root,
    decoupled_ucb_root,
)


def _mk_joint(tag: int) -> JointAction:
    """Tiny JointAction stand-in: one PlanetMove, tag stored in from_pid."""
    return JointAction(moves=(PlanetMove(
        from_pid=tag, angle=0.0, ships=0, target_pid=-1,
        kind=KIND_HOLD, prior=1.0,
    ),))


# ---- decoupled_ucb_root -------------------------------------------------

def test_decoupled_ucb_returns_best_my_candidate():
    """With my[0] dominant (v=+1 always) and others losing (v=-1), the
    returned best_my_joint is my[0]."""
    my = [_mk_joint(i) for i in range(3)]
    opp = [_mk_joint(10), _mk_joint(11)]

    def rollout(mi, oi):
        return +1.0 if mi is my[0] else -1.0

    result = decoupled_ucb_root(
        my, opp, rollout,
        total_sims=40, hard_deadline_ms=5000.0,
    )
    assert isinstance(result, DecoupledSearchResult)
    assert result.best_my_joint is my[0]
    assert result.my_q_values[0] == pytest.approx(1.0, abs=1e-6)
    # Lost candidates converge to -1 (they are always losing).
    assert result.my_q_values[1] == pytest.approx(-1.0, abs=1e-6)


def test_decoupled_ucb_warmup_visits_all_cells():
    """Phase-1 warm-up plays every (i, j) cell at least once before UCB
    selection begins — guarantees every arm has some data."""
    my = [_mk_joint(i) for i in range(2)]
    opp = [_mk_joint(j) for j in range(3)]
    visited_cells = set()

    def rollout(mi, oi):
        i = mi.moves[0].from_pid
        j = oi.moves[0].from_pid
        visited_cells.add((i, j))
        return 0.0

    # Budget exactly covers the 6 warm-up cells; no UCB phase should run.
    decoupled_ucb_root(
        my, opp, rollout,
        total_sims=6, hard_deadline_ms=5000.0,
    )
    assert len(visited_cells) == 6
    # And budget < 6 visits fewer cells.
    visited_cells.clear()
    decoupled_ucb_root(
        my, opp, rollout,
        total_sims=4, hard_deadline_ms=5000.0,
    )
    assert len(visited_cells) == 4


def test_decoupled_ucb_budget_is_respected():
    """total_sims is the exact upper bound on rollouts."""
    my = [_mk_joint(i) for i in range(2)]
    opp = [_mk_joint(j) for j in range(2)]
    calls = [0]

    def rollout(mi, oi):
        calls[0] += 1
        return 0.5

    result = decoupled_ucb_root(
        my, opp, rollout,
        total_sims=20, hard_deadline_ms=5000.0,
    )
    assert result.n_rollouts == 20
    assert calls[0] == 20
    assert sum(result.my_visits) == 20
    assert sum(result.opp_visits) == 20


def test_decoupled_ucb_zero_sum_bookkeeping():
    """Under zero-sum updates, my_q_sum[i] == -opp_q_sum_averaged[...]
    modulo interleaving. Strictest invariant: the TOTAL mean Q across my
    candidates equals -TOTAL mean Q across opp candidates when the
    bandit has converged."""
    my = [_mk_joint(i) for i in range(2)]
    # Use tags 10+j so we can distinguish my-indices from opp-indices.
    opp = [_mk_joint(10 + j) for j in range(2)]

    def rollout(mi, oi):
        # 4 distinct cells, known payoffs: (0,0)=+0.5 (0,1)=-0.5 (1,0)=-0.3 (1,1)=+0.3
        i = mi.moves[0].from_pid
        j = oi.moves[0].from_pid - 10
        return [[0.5, -0.5], [-0.3, 0.3]][i][j]

    # Large budget → full convergence to cell means.
    result = decoupled_ucb_root(
        my, opp, rollout,
        total_sims=200, hard_deadline_ms=5000.0,
    )
    # Aggregate check: mean(my_q) + mean(opp_q) ≈ 0 when rollouts span
    # all cells equitably. In practice UCB concentrates on the best
    # cell so this is a loose check.
    total_my = sum(q * v for q, v in zip(result.my_q_values, result.my_visits))
    total_opp = sum(q * v for q, v in zip(result.opp_q_values, result.opp_visits))
    assert total_my + total_opp == pytest.approx(0.0, abs=1e-6)


def test_decoupled_ucb_respects_deadline():
    """When the hard deadline expires mid-bandit, ``aborted=True`` and
    we return with whatever visits we've accumulated."""
    my = [_mk_joint(i) for i in range(4)]
    opp = [_mk_joint(j) for j in range(4)]
    t_start = time.perf_counter()

    def rollout(mi, oi):
        # Slow rollouts so we hit the deadline fast.
        time.sleep(0.02)
        return 0.0

    result = decoupled_ucb_root(
        my, opp, rollout,
        total_sims=1000, hard_deadline_ms=60.0,  # 60 ms hard cap
        start_time=t_start,
    )
    assert result.aborted is True
    assert result.n_rollouts < 1000  # couldn't finish full budget
    assert result.duration_ms >= 60.0


def test_decoupled_ucb_protected_my_idx_visits_all_opp():
    """protected_my_idx plays against every opp candidate first — this
    gives the anchor a low-variance Q estimate before UCB kicks in."""
    my = [_mk_joint(i) for i in range(3)]
    opp = [_mk_joint(j) for j in range(4)]

    def rollout(mi, oi):
        return 0.0

    # Budget = n_opp exactly: only the protected anchor should get visits.
    result = decoupled_ucb_root(
        my, opp, rollout,
        total_sims=4, hard_deadline_ms=5000.0,
        protected_my_idx=1,
    )
    assert result.my_visits[1] == 4  # anchor hit every opp
    assert result.my_visits[0] == 0
    assert result.my_visits[2] == 0
    # Each opp got exactly one visit (from the anchor).
    assert result.opp_visits == [1, 1, 1, 1]


def test_decoupled_ucb_rejects_empty_lists():
    def rollout(mi, oi):
        return 0.0
    with pytest.raises(ValueError):
        decoupled_ucb_root([], [_mk_joint(0)], rollout, 10, 5000.0)
    with pytest.raises(ValueError):
        decoupled_ucb_root([_mk_joint(0)], [], rollout, 10, 5000.0)


def test_decoupled_ucb_single_candidate_each_side():
    """Degenerate n=1 per side → one rollout if budget >= 1."""
    my = [_mk_joint(0)]
    opp = [_mk_joint(10)]
    calls = [0]

    def rollout(mi, oi):
        calls[0] += 1
        return 0.42

    result = decoupled_ucb_root(my, opp, rollout, total_sims=5, hard_deadline_ms=5000.0)
    assert calls[0] == 5
    assert result.best_my_joint is my[0]
    assert result.my_q_values[0] == pytest.approx(0.42)
    assert result.opp_q_values[0] == pytest.approx(-0.42)


# ---- decoupled_exp3_root ------------------------------------------------

def test_decoupled_exp3_returns_valid_result():
    """Smoke test: the Exp3 variant runs, returns a valid my-candidate,
    and consumes its budget."""
    my = [_mk_joint(i) for i in range(3)]
    opp = [_mk_joint(j) for j in range(2)]

    def rollout(mi, oi):
        return +1.0 if mi is my[0] else -0.5

    result = decoupled_exp3_root(
        my, opp, rollout,
        total_sims=60, hard_deadline_ms=5000.0,
    )
    assert result.n_rollouts == 60
    # my[0] is the dominant strategy → should be picked.
    assert result.best_my_joint is my[0]
    # my[0] should accumulate the most visits (Exp3's softmax may
    # concentrate fully on it at a large-enough eta — we don't require
    # strict positivity on every arm).
    assert result.my_visits[0] >= max(result.my_visits[1:])


def test_decoupled_exp3_respects_deadline():
    my = [_mk_joint(i) for i in range(3)]
    opp = [_mk_joint(j) for j in range(2)]

    def rollout(mi, oi):
        time.sleep(0.02)
        return 0.0

    result = decoupled_exp3_root(
        my, opp, rollout,
        total_sims=500, hard_deadline_ms=40.0,
    )
    assert result.aborted is True
    assert result.n_rollouts < 500


def test_decoupled_exp3_protected_my_idx_visits_all_opp():
    """Anchor-lock contract (W4 A/B): the protected my candidate must
    be paired with every opp at least once, BEFORE the softmax is free
    to collapse onto a single my arm. Mirrors
    test_decoupled_ucb_protected_my_idx_visits_all_opp — same guarantee
    for the Exp3 variant so swapping variants doesn't break the
    anchor_guard upstream."""
    my = [_mk_joint(i) for i in range(4)]
    opp = [_mk_joint(j) for j in range(3)]

    # Make the protected arm (index 0) WORSE than the others so Exp3 would
    # never pick it after warm-up — if warm-up is missing, visits[0] stays
    # at 0.
    def rollout(mi, oi):
        if mi is my[0]:
            return -1.0
        return +0.5

    import random as _r
    rng = _r.Random(42)  # deterministic for CI
    result = decoupled_exp3_root(
        my, opp, rollout,
        total_sims=30, hard_deadline_ms=5000.0,
        protected_my_idx=0, rng=rng,
    )
    # Protected arm paired with every opp at least once.
    assert result.my_visits[0] >= len(opp), (
        f"protected arm under-visited: visits[0]={result.my_visits[0]} < "
        f"n_opp={len(opp)}; warm-up phase missing?"
    )
    # And every opp has at least one visit too (warm-up covers them).
    for j in range(len(opp)):
        assert result.opp_visits[j] >= 1, (
            f"opp[{j}] never visited in warm-up"
        )


def test_decoupled_exp3_rng_seed_is_deterministic():
    """Two identical runs with a seeded RNG must produce identical
    histograms. Load-bearing for deterministic tests downstream that
    want to assert on Exp3 output without flakiness."""
    my = [_mk_joint(i) for i in range(3)]
    opp = [_mk_joint(j) for j in range(2)]

    def rollout(mi, oi):
        return 0.1 * (hash((id(mi), id(oi))) % 7 - 3)

    import random as _r
    a = decoupled_exp3_root(
        my, opp, rollout,
        total_sims=40, hard_deadline_ms=5000.0,
        rng=_r.Random(123),
    )
    b = decoupled_exp3_root(
        my, opp, rollout,
        total_sims=40, hard_deadline_ms=5000.0,
        rng=_r.Random(123),
    )
    assert a.my_visits == b.my_visits
    assert a.opp_visits == b.opp_visits
