"""Tests for opponent pool factories in orbitwars.tune.fitness.

These are cheap — they construct the pool and verify shape. They do
NOT play games, so they're fast (<1 s total). The real validation that
TuRBO benefits from w2_pool is a full tuning run, which is separately
kicked off from the CLI.
"""
from __future__ import annotations

import pytest

from orbitwars.tune.fitness import Opponent, starter_pool, w1_pool, w2_pool


def test_starter_pool_shape():
    pool = starter_pool()
    assert len(pool) == 1
    name, factory = pool[0]
    assert name == "starter"
    assert factory() == "starter"


def test_w1_pool_contains_starter_random_noop():
    pool = w1_pool()
    names = {name for name, _ in pool}
    assert {"starter", "random", "noop"}.issubset(names)


def test_w2_pool_includes_all_archetypes():
    from orbitwars.opponent.archetypes import ARCHETYPE_NAMES

    pool = w2_pool()
    names = [name for name, _ in pool]
    # All 7 archetypes plus starter + random.
    assert "starter" in names
    assert "random" in names
    for arch in ARCHETYPE_NAMES:
        assert arch in names


def test_w2_pool_factories_yield_working_callables():
    """Each archetype factory must return a fresh kaggle-agent callable
    (not the same instance reused). This guarantees per-game state is
    isolated."""
    from orbitwars.opponent.archetypes import ARCHETYPE_NAMES

    pool = w2_pool()
    for name, factory in pool:
        if name in ("starter", "random"):
            continue
        assert name in ARCHETYPE_NAMES
        a1 = factory()
        a2 = factory()
        # Each call must produce a fresh callable — otherwise an agent's
        # per-match state would leak into the next game in the eval.
        assert a1 is not a2
        assert callable(a1)
        assert callable(a2)


def test_w2_pool_archetype_factories_close_over_name_correctly():
    """Regression check for the late-binding lambda gotcha: every
    archetype factory must produce an agent with the correct name."""
    pool = w2_pool()
    for name, factory in pool:
        if name in ("starter", "random"):
            continue
        agent = factory()
        # `as_kaggle_agent()` sets __name__ to self.name which for
        # ArchetypeAgent is the archetype name.
        assert agent.__name__ == name, (
            f"Factory for {name!r} produced agent named {agent.__name__!r} "
            f"— late-binding lambda bug?"
        )
