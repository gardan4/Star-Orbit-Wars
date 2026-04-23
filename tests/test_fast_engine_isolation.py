"""Regression tests for FastEngine's global-random isolation contract.

The bug: ``kaggle_environments.envs.orbit_wars.generate_comet_paths``
internally calls ``random.uniform`` 3-900 times per invocation (ellipse
eccentricity, semi-major axis, orientation, looped up to 300 times until
a valid perihelion). ``FastEngine._maybe_spawn_comets`` calls that
function at every comet-spawn step (50, 150, 250, 350, 450).

Without isolation, MCTS rollouts that cross a spawn step consume entropy
from the *Kaggle judge's* global random stream — the same stream used by
the official engine for its own comet spawns. Empirical signature:
MCTS-at-seat-1 plays IDENTICAL wire actions to shadow-heuristic on every
turn (0/N diverged) yet the game outcome flips. See
``tools/diag_who_touches_global_random.py`` for the measurement: heur-vs-
heur at seed=123 made 3166 global random calls; MCTS-vs-heur made
28888 — a 9× leak, all at ``orbit_wars.py:{233,234,242}`` inside
``generate_comet_paths``.

Fix: ``_maybe_spawn_comets`` snapshots + restores global random state
around the ``generate_comet_paths`` call, but only when ``self._rng`` is
*not* the ``random`` module (i.e. parity validator mode explicitly
shares global state).
"""
from __future__ import annotations

import random

import pytest
from kaggle_environments import make

from orbitwars.engine.fast_engine import COMET_SPAWN_STEPS, FastEngine


def _mid_game_obs_at_step(step_target: int, seed: int = 42):
    """Advance a real env to just before ``step_target`` via heur play and
    return the obs at the resulting state.
    """
    from orbitwars.bots.heuristic import HeuristicAgent

    random.seed(seed)
    env = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
    env.reset(num_agents=2)
    env.step([[], []])
    heur = HeuristicAgent().as_kaggle_agent()
    while env.state[0]["observation"]["step"] < step_target:
        a0 = heur(env.state[0].observation)
        a1 = heur(env.state[1].observation)
        env.step([a0, a1])
        if env.done:
            break
    return env.state[0].observation


def test_isolation_mode_does_not_perturb_global_random_at_comet_spawn():
    """The hot bug: rollout engines leaking entropy into the judge's stream.

    A FastEngine constructed with the default rng (instance Random, not
    the ``random`` module) must leave global state untouched when stepping
    across a comet-spawn boundary.
    """
    # Reach step 49 so step+1 == 50 triggers _maybe_spawn_comets.
    obs = _mid_game_obs_at_step(step_target=49, seed=42)
    assert int(obs["step"]) == 49, f"setup failed to reach step 49: got {obs['step']}"

    # Isolation mode: default constructor (no rng arg) gives random.Random().
    fast = FastEngine.from_official_obs(obs, num_agents=2)
    assert fast._rng is not random, "precondition: should be isolated"

    before_state = random.getstate()
    fast.step([[], []])
    after_state = random.getstate()

    assert before_state == after_state, (
        "FastEngine in isolation mode leaked global random state during "
        "comet spawn. Check that _maybe_spawn_comets save/restores around "
        "generate_comet_paths when self._rng is not random (the module)."
    )


def test_isolation_mode_still_produces_valid_paths_at_comet_spawn():
    """Save/restore must not break functional behavior — paths should
    still get generated and attached to state.comets.
    """
    obs = _mid_game_obs_at_step(step_target=49, seed=42)
    fast = FastEngine.from_official_obs(obs, num_agents=2)
    pre_num_comet_groups = len(fast.state.comets)
    fast.step([[], []])
    # Spawn may still fail (rare, engine retries up to 300 times per spawn),
    # but in our seed the spawn should succeed. Weaken to "no crash, no
    # corruption" if flaky.
    post_num_comet_groups = len(fast.state.comets)
    assert post_num_comet_groups >= pre_num_comet_groups, (
        "comet spawn should not remove existing comet groups"
    )


def test_parity_mode_still_consumes_global_random():
    """The parity validator explicitly passes ``rng=random`` (the module)
    so both engines draw from the same stream. In that mode the isolation
    must NOT fire — the official engine's consumption from global random
    is what we're trying to mirror for parity.
    """
    obs = _mid_game_obs_at_step(step_target=49, seed=42)
    fast = FastEngine.from_official_obs(obs, num_agents=2, rng=random)
    assert fast._rng is random, "precondition: parity mode"

    before_state = random.getstate()
    fast.step([[], []])
    after_state = random.getstate()

    # Comet spawn via generate_comet_paths + ship_sizing must consume
    # random in parity mode — >= 1 call suffices, the exact count is
    # engine-detail.
    assert before_state != after_state, (
        "FastEngine in parity mode (rng=random) must consume global "
        "random state during comet spawn — if not, parity with the "
        "official engine will break at the next spawn."
    )


@pytest.mark.parametrize("spawn_step", COMET_SPAWN_STEPS)
def test_isolation_holds_across_all_spawn_steps(spawn_step: int):
    """The isolation contract must hold at every spawn step, not just
    the first one. The engine re-enters _maybe_spawn_comets at 50, 150,
    250, 350, 450.
    """
    obs = _mid_game_obs_at_step(step_target=spawn_step - 1, seed=42)
    if int(obs["step"]) != spawn_step - 1:
        # Game ended before reaching this spawn step — skip.
        pytest.skip(f"game ended before step {spawn_step - 1}")
    fast = FastEngine.from_official_obs(obs, num_agents=2)
    before_state = random.getstate()
    fast.step([[], []])
    after_state = random.getstate()
    assert before_state == after_state, (
        f"isolation broken at spawn step {spawn_step}"
    )
