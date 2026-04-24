"""Tests for opponent pool factories in orbitwars.tune.fitness.

These are cheap — they construct the pool and verify shape. They do
NOT play games, so they're fast (<1 s total). The real validation that
TuRBO benefits from w2_pool is a full tuning run, which is separately
kicked off from the CLI.

Pools now hold ``OpponentSpec`` records (picklable) rather than closures,
so the ``multiprocessing.Pool`` path in ``evaluate`` works on Windows.
"""
from __future__ import annotations

import pickle

from orbitwars.tune.fitness import (
    Opponent,
    OpponentSpec,
    starter_pool,
    w1_pool,
    w2_pool,
)


def test_starter_pool_shape():
    pool = starter_pool()
    assert len(pool) == 1
    name, spec = pool[0]
    assert name == "starter"
    assert isinstance(spec, OpponentSpec)
    # kaggle_builtin reconstructs to the pass-through string.
    assert spec.make_kaggle_agent() == "starter"


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


def test_w2_pool_specs_yield_fresh_callables():
    """Each archetype spec must return a fresh kaggle-agent callable
    (not the same instance reused). This guarantees per-game state is
    isolated."""
    from orbitwars.opponent.archetypes import ARCHETYPE_NAMES

    pool = w2_pool()
    for name, spec in pool:
        if name in ("starter", "random"):
            continue
        assert name in ARCHETYPE_NAMES
        a1 = spec.make_kaggle_agent()
        a2 = spec.make_kaggle_agent()
        # Each call must produce a fresh callable — otherwise an agent's
        # per-match state would leak into the next game in the eval.
        assert a1 is not a2
        assert callable(a1)
        assert callable(a2)


def test_w2_pool_archetype_specs_bind_name_correctly():
    """Regression check for the late-binding lambda gotcha: every
    archetype spec must reconstruct an agent with the correct name.
    OpponentSpec stores `param` by-value so late-binding is impossible
    by construction, but the test remains as a contract guard."""
    pool = w2_pool()
    for name, spec in pool:
        if name in ("starter", "random"):
            continue
        agent = spec.make_kaggle_agent()
        # `as_kaggle_agent()` sets __name__ to self.name which for
        # ArchetypeAgent is the archetype name.
        assert agent.__name__ == name, (
            f"Spec for {name!r} produced agent named {agent.__name__!r} "
            f"— late-binding bug?"
        )


def test_pool_specs_are_picklable():
    """The whole point of OpponentSpec: every entry must pickle so
    ``multiprocessing.Pool`` on Windows (spawn) can ship tasks into
    workers without the ``can't pickle local object`` error we'd
    hit with the old lambda-based factories."""
    for pool in (starter_pool(), w1_pool(), w2_pool()):
        for name, spec in pool:
            blob = pickle.dumps(spec)
            round_trip: OpponentSpec = pickle.loads(blob)
            assert round_trip == spec, f"pickle round-trip failed for {name}"
            # And the reconstructed spec still builds a live agent.
            agent = round_trip.make_kaggle_agent()
            assert callable(agent) or isinstance(agent, str)
