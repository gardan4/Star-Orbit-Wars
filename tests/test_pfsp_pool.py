"""Tests for orbitwars.nn.pfsp_pool — opponent sampling + win-rate tracking."""
from __future__ import annotations

import collections
import pytest

from orbitwars.nn.pfsp_pool import OpponentEntry, PFSPPool


def _mk_dummy_pool(names):
    """Build a pool of dummy opponents (factories return a stub)."""
    entries = [
        OpponentEntry(name=n, factory=lambda n=n: (lambda obs: []))
        for n in names
    ]
    return PFSPPool(opponents=entries, seed=0)


def test_pool_initializes_win_rates_at_one_half():
    pool = _mk_dummy_pool(["a", "b", "c"])
    assert pool.win_rate("a") == 0.5
    assert pool.win_rate("b") == 0.5
    assert pool.win_rate("c") == 0.5


def test_update_win_rate_ema_smoothing():
    """EMA: prev=0.5, alpha=0.05, won=True → new = 0.95*0.5 + 0.05*1.0 = 0.525."""
    pool = _mk_dummy_pool(["a"])
    pool.update_win_rate("a", won=True)
    assert pool.win_rate("a") == pytest.approx(0.525, abs=1e-9)
    # Five wins push toward 1.0 but not fully (heavy smoothing).
    for _ in range(5):
        pool.update_win_rate("a", won=True)
    assert pool.win_rate("a") < 1.0
    assert pool.win_rate("a") > 0.6


def test_update_win_rate_value_accepts_ties():
    pool = _mk_dummy_pool(["a"])
    pool.update_win_rate_value("a", 0.5)  # tie → no change from 0.5
    assert pool.win_rate("a") == pytest.approx(0.5, abs=1e-9)


def test_sample_returns_opponent_in_pool():
    pool = _mk_dummy_pool(["a", "b", "c"])
    for _ in range(20):
        opp = pool.sample()
        assert opp.name in {"a", "b", "c"}


def test_sample_overweights_low_winrate_opponents():
    """If we beat 'a' a lot but lose to 'b', PFSP should sample 'b'
    much more often."""
    pool = _mk_dummy_pool(["a", "b", "c"])
    # Make 'b' look hard (low win rate) and 'a' look easy.
    for _ in range(40):
        pool.update_win_rate("a", won=True)
    for _ in range(40):
        pool.update_win_rate("b", won=False)
    # 'c' stays at 0.5 (default).
    counts = collections.Counter()
    for _ in range(2000):
        counts[pool.sample().name] += 1
    # 'b' (loss-heavy) should be sampled far more than 'a' (win-heavy).
    assert counts["b"] > counts["a"] * 2, (
        f"PFSP not overweighting hard opponent: {dict(counts)}"
    )


def test_min_weight_floor_keeps_solved_opponents_in_rotation():
    pool = _mk_dummy_pool(["solved"])
    pool._win_rate["solved"] = 1.0  # we always beat it.
    counts = collections.Counter()
    for _ in range(50):
        counts[pool.sample().name] += 1
    # Even at win_rate=1.0, the min_weight floor keeps it sampled.
    assert counts["solved"] == 50  # only opponent in pool


def test_base_weight_overrides_pfsp():
    """A high base_weight on a high-win-rate opponent should still
    keep it in rotation — useful for "always train against the
    current ladder floor"."""
    entries = [
        OpponentEntry(name="strong", factory=lambda: (lambda o: []), base_weight=10.0),
        OpponentEntry(name="weak", factory=lambda: (lambda o: []), base_weight=1.0),
    ]
    pool = PFSPPool(opponents=entries, seed=0)
    # Make both look easy.
    pool._win_rate["strong"] = 0.9
    pool._win_rate["weak"] = 0.9
    # Weights: strong = 10 * 0.01 = 0.1, weak = 1 * 0.01 = 0.01
    counts = collections.Counter()
    for _ in range(2000):
        counts[pool.sample().name] += 1
    # 'strong' should be sampled ~10x more often than 'weak'.
    assert counts["strong"] > counts["weak"] * 5


def test_snapshot_returns_independent_copy():
    pool = _mk_dummy_pool(["a"])
    snap = pool.snapshot()
    pool.update_win_rate("a", won=True)
    assert snap["a"] == pytest.approx(0.5, abs=1e-9), (
        "snapshot should not be affected by later updates"
    )
    assert pool.win_rate("a") != snap["a"]


def test_seed_reproducible_sampling():
    pool_a = _mk_dummy_pool(["x", "y", "z"])
    pool_b = _mk_dummy_pool(["x", "y", "z"])  # same seed=0 default
    seq_a = [pool_a.sample().name for _ in range(20)]
    seq_b = [pool_b.sample().name for _ in range(20)]
    assert seq_a == seq_b


def test_factory_called_per_match():
    """factory() should produce a fresh agent — pre-sampling pinning
    not done. PPO trainer relies on this for stateful agents like
    HeuristicAgent / MCTSAgent."""
    counter = {"calls": 0}

    def factory():
        counter["calls"] += 1
        return lambda obs: []

    pool = PFSPPool(
        opponents=[OpponentEntry(name="x", factory=factory)],
        seed=0,
    )
    # Each `sample().factory()` call increments.
    for _ in range(5):
        agent = pool.sample().factory()
        assert callable(agent)
    assert counter["calls"] == 5
