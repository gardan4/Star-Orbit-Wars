"""Tests for the archetype portfolio.

These are fast unit-level checks — does the module import, are all
archetypes constructible, do their overrides hit real weight keys, and
can they produce an action on a synthetic observation without crashing?

The expensive "archetype beats no-op" integration check lives in
``tools/smoke_archetypes.py`` — 7 archetypes × a 500-turn game each is
too slow for the pytest budget.
"""
from __future__ import annotations

from orbitwars.bots.heuristic import HEURISTIC_WEIGHTS, HeuristicAgent
from orbitwars.bots.base import Deadline
from orbitwars.opponent.archetypes import (
    ARCHETYPE_NAMES,
    ARCHETYPE_WEIGHTS,
    ArchetypeAgent,
    all_archetypes,
    make_archetype,
)


# --- Catalogue shape ----------------------------------------------------

def test_all_archetypes_constructible():
    agents = all_archetypes()
    assert len(agents) == len(ARCHETYPE_NAMES)
    for a, n in zip(agents, ARCHETYPE_NAMES):
        assert isinstance(a, ArchetypeAgent)
        assert isinstance(a, HeuristicAgent)  # shares Path A contract
        assert a.name == n
        assert a.archetype == n


def test_archetype_names_unique():
    assert len(set(ARCHETYPE_NAMES)) == len(ARCHETYPE_NAMES)


def test_archetype_names_canonical_order():
    """Posterior tracking relies on a stable index ordering."""
    assert ARCHETYPE_NAMES == list(ARCHETYPE_WEIGHTS.keys())


def test_unknown_archetype_raises():
    try:
        make_archetype("nonexistent")
    except KeyError as e:
        assert "nonexistent" in str(e)
    else:
        raise AssertionError("expected KeyError")


# --- Weight override hygiene -------------------------------------------

def test_every_override_key_is_real():
    """The assertion at module-import time guards this, but be explicit —
    a silently-ignored override is a debugging nightmare."""
    known = set(HEURISTIC_WEIGHTS.keys())
    for name, overrides in ARCHETYPE_WEIGHTS.items():
        unknown = set(overrides) - known
        assert not unknown, f"{name} has unknown keys: {unknown}"


def test_every_archetype_has_some_overrides():
    """An archetype with no overrides is a duplicate of default heuristic."""
    for name, overrides in ARCHETYPE_WEIGHTS.items():
        assert overrides, f"archetype {name!r} has no overrides"


def test_archetypes_are_stylistically_distinct():
    """The weights dicts must be pairwise different — the posterior
    can't separate identical policies."""
    seen: dict = {}
    for name, overrides in ARCHETYPE_WEIGHTS.items():
        # Freeze as a sorted tuple of items for hashability
        frozen = tuple(sorted(overrides.items()))
        assert frozen not in seen, (
            f"{name!r} has identical weights to {seen[frozen]!r}"
        )
        seen[frozen] = name


# --- Effective-weight composition --------------------------------------

def test_effective_weights_layer_correctly():
    """HeuristicAgent.__init__ merges archetype overrides on top of
    HEURISTIC_WEIGHTS. Unspecified keys fall through to default, specified
    keys win."""
    ag = make_archetype("rusher")
    # A key the rusher overrides
    assert ag.weights["mult_enemy"] == ARCHETYPE_WEIGHTS["rusher"]["mult_enemy"]
    # A key the rusher doesn't touch → falls through to default
    assert ag.weights["sun_avoidance_epsilon"] == HEURISTIC_WEIGHTS[
        "sun_avoidance_epsilon"
    ]


# --- Behavioral smoke on a real observation ----------------------------

def test_archetypes_act_on_real_obs_without_crashing():
    """One step of a fresh kaggle env — every archetype must produce a
    valid (possibly empty) action list in finite time."""
    import random

    from kaggle_environments import make  # slow import; keep scoped

    random.seed(17)  # orbit_wars map generation reads global random
    env = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
    env.reset(num_agents=2)
    obs = env.state[0]["observation"]

    for name in ARCHETYPE_NAMES:
        ag = make_archetype(name)
        dl = Deadline()
        action = ag.act(obs, dl)
        # Contract: a list of [planet_id, angle, ships] triples, or empty.
        assert isinstance(action, list), f"{name}: action not list"
        for mv in action:
            assert len(mv) == 3, f"{name}: malformed move {mv}"
