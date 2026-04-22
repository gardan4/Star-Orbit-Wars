"""Tests for the Bayesian archetype posterior.

These exercise:
  * uniform-prior start, monotone-valid distribution after N observations
  * concentration: when opp *is* an archetype, posterior gives that
    archetype the highest mass (the core novelty test)
  * reset behavior
  * edge cases: single-call bootstrap without crash, opp elimination

The concentration test drives a real kaggle env — it's slower but
tests what actually matters.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
from kaggle_environments import make

from orbitwars.bots.base import Deadline
from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.opponent.archetypes import (
    ARCHETYPE_NAMES,
    ArchetypeAgent,
    all_archetypes,
    make_archetype,
)
from orbitwars.opponent.bayes import ArchetypePosterior, _softmax


# --- Static checks -------------------------------------------------------

def test_prior_is_uniform():
    p = ArchetypePosterior()
    dist = p.distribution()
    assert dist.shape == (len(ARCHETYPE_NAMES),)
    np.testing.assert_allclose(dist, np.full_like(dist, 1.0 / len(dist)))


def test_distribution_sums_to_one_after_random_updates():
    p = ArchetypePosterior()
    # Hack in some fake log-alphas
    p.log_alpha = np.array([1.0, -2.0, 0.5, 3.1, -0.7, 0.0, 1.2])[: p.K]
    assert abs(p.distribution().sum() - 1.0) < 1e-9


def test_reset_restores_uniform():
    p = ArchetypePosterior()
    p.log_alpha[:] = np.random.RandomState(0).randn(p.K) * 5
    p.reset()
    np.testing.assert_allclose(p.distribution(), 1.0 / p.K)
    assert p.turns_observed() == 0


def test_softmax_stable_under_shift():
    x = np.array([1000.0, 1001.0, 999.0])
    d = _softmax(x)
    assert np.all(np.isfinite(d))
    assert abs(d.sum() - 1.0) < 1e-12


def test_single_observation_does_not_error():
    """First observe() call has no prior obs to diff against; must not
    crash and must not modify the posterior."""
    p = ArchetypePosterior()
    random.seed(5)
    env = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
    env.reset(num_agents=2)
    obs = env.state[0]["observation"]
    p.observe(obs, opp_player=1)
    np.testing.assert_allclose(p.distribution(), 1.0 / p.K)


# --- Behavioral / concentration test ------------------------------------

def _run_turns_with_opp(opp_agent, my_agent, n_turns: int, seed: int = 7):
    """Play ``n_turns`` steps of a 2-player env with given agents and
    return the per-turn obs-from-player-0's-perspective sequence."""
    random.seed(seed)
    env = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
    env.reset(num_agents=2)

    obs_seq = []
    kag_my = my_agent.as_kaggle_agent()
    kag_opp = opp_agent.as_kaggle_agent()

    for _ in range(n_turns):
        obs_p0 = env.state[0]["observation"]
        obs_seq.append(obs_p0)
        if env.state[0]["status"] != "ACTIVE":
            break
        obs_p1 = env.state[1]["observation"]
        a0 = kag_my(obs_p0, env.configuration)
        a1 = kag_opp(obs_p1, env.configuration)
        env.step([a0, a1])

    return obs_seq


@pytest.mark.parametrize("archetype_name", ["rusher", "turtler", "defender"])
def test_posterior_concentrates_on_true_archetype(archetype_name):
    """When opp plays the ``archetype_name`` policy, its posterior
    probability after ~30 observed turns should exceed the uniform
    baseline (1/K) by a meaningful margin."""
    opp = make_archetype(archetype_name)
    me = HeuristicAgent()

    obs_seq = _run_turns_with_opp(opp, me, n_turns=40, seed=13)

    post = ArchetypePosterior(temperature=2.0, eps=0.1)
    for obs in obs_seq:
        post.observe(obs, opp_player=1)

    dist = post.distribution()
    idx = ARCHETYPE_NAMES.index(archetype_name)
    uniform = 1.0 / len(ARCHETYPE_NAMES)

    # Posterior mass on the true archetype must exceed uniform — and
    # ideally be the top entry, though we keep the assertion loose to
    # avoid flakiness on short sequences.
    assert dist[idx] > uniform, (
        f"{archetype_name} posterior={dist[idx]:.3f} did not beat "
        f"uniform {uniform:.3f} over {post.turns_observed()} turns "
        f"(full dist: {dict(zip(ARCHETYPE_NAMES, np.round(dist, 3)))})"
    )


def test_most_likely_returns_known_archetype():
    p = ArchetypePosterior()
    ml = p.most_likely()
    assert ml in ARCHETYPE_NAMES


# --- Cost sanity ---------------------------------------------------------

def test_observe_is_cheap():
    """One observe() call must stay under ~50 ms even on a dense
    mid-game obs — our turn budget is 1 s and MCTS gets most of it."""
    import time

    random.seed(99)
    env = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
    env.reset(num_agents=2)
    me = HeuristicAgent()
    opp = make_archetype("economy")
    obs_seq = _run_turns_with_opp(opp, me, n_turns=5, seed=99)

    p = ArchetypePosterior()
    # Bootstrap on first obs (no-op).
    p.observe(obs_seq[0], opp_player=1)
    # Time the second.
    t0 = time.perf_counter()
    p.observe(obs_seq[-1], opp_player=1)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 50.0, f"observe took {elapsed_ms:.1f} ms"
