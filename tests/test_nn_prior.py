"""Unit tests for the ConvPolicy -> PlanetMove prior bridge.

These tests build a TINY ConvPolicy with hand-set weights (or just a
random init we don't care about) and verify the bridge:
  * Maps continuous angles to the correct discrete bucket.
  * Maps ship-fraction to the closer of {0.5, 1.0}.
  * Picks the correct ACTION_LOOKUP channel for a (bucket, frac) pair.
  * Per-planet softmax sums to 1.
  * HOLD moves get the configured neutral mass.

We deliberately don't load a real bc_warmstart.pt — the bridge logic is
checkpoint-independent and these tests should run in any environment.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from orbitwars.mcts.actions import (
    KIND_ATTACK_ENEMY,
    KIND_ATTACK_NEUTRAL,
    KIND_HOLD,
    PlanetMove,
)
from orbitwars.nn.conv_policy import (
    ACTION_LOOKUP,
    ConvPolicy,
    ConvPolicyCfg,
)
from orbitwars.nn.nn_prior import (
    N_ACTION_CHANNELS,
    N_ANGLE_BUCKETS,
    angle_to_bucket,
    candidate_to_channel,
    make_nn_prior_fn,
    nn_priors_for_planet,
    ship_fraction_to_bucket,
)


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------


def test_angle_buckets_at_centers():
    """Angles exactly at bucket centers (0, pi/2, pi, 3pi/2) hit
    buckets (0, 1, 2, 3)."""
    assert angle_to_bucket(0.0) == 0
    assert angle_to_bucket(math.pi / 2) == 1
    assert angle_to_bucket(math.pi) == 2
    assert angle_to_bucket(3 * math.pi / 2) == 3


def test_angle_bucket_wrapping():
    """Negative angles and angles >= 2*pi wrap correctly."""
    assert angle_to_bucket(-math.pi / 2) == 3  # -90deg = 270deg
    assert angle_to_bucket(2 * math.pi) == 0   # full revolution
    assert angle_to_bucket(2 * math.pi + 0.01) == 0


def test_angle_bucket_boundaries():
    """At the boundary between buckets, ties go consistently. Doesn't
    matter which way as long as it's stable — we just don't want NaN."""
    # Boundary between bucket 0 (centered at 0) and bucket 1 (centered
    # at pi/2) is at pi/4.
    b = angle_to_bucket(math.pi / 4)
    assert b in (0, 1)


def test_ship_fraction_snap():
    """Continuous fractions snap to nearest of {0.5, 1.0}."""
    assert ship_fraction_to_bucket(50, 100) == pytest.approx(0.5)
    assert ship_fraction_to_bucket(100, 100) == pytest.approx(1.0)
    # 25% is closer to 0.5 than 1.0.
    assert ship_fraction_to_bucket(25, 100) == pytest.approx(0.5)
    # 75% is closer to 1.0 than 0.5 (0.25 vs 0.25 — tied; default to
    # one of them is fine).
    f = ship_fraction_to_bucket(75, 100)
    assert f in (0.5, 1.0)


def test_ship_fraction_zero_available_safe():
    """No crash when source has 0 ships — degenerate case the heuristic
    filters out, but the bridge must be defensive."""
    f = ship_fraction_to_bucket(0, 0)
    assert f in (0.5, 1.0)


# ---------------------------------------------------------------------------
# Channel mapping
# ---------------------------------------------------------------------------


def test_candidate_to_channel_known_pairs():
    """Hand-checked: angle=0 (East) + 100% ships -> ACTION_LOOKUP[1]
    which is ``(0, 1.0)``."""
    m = PlanetMove(
        from_pid=0, angle=0.0, ships=100, target_pid=5,
        kind=KIND_ATTACK_ENEMY, prior=0.0, raw_score=0.0,
    )
    assert candidate_to_channel(m, available=100) == 1

    # angle=pi/2 (North) + 50% ships -> ACTION_LOOKUP[2] = (1, 0.5)
    m2 = PlanetMove(
        from_pid=0, angle=math.pi / 2, ships=50, target_pid=5,
        kind=KIND_ATTACK_ENEMY, prior=0.0, raw_score=0.0,
    )
    assert candidate_to_channel(m2, available=100) == 2


def test_candidate_to_channel_hold_returns_sentinel():
    """HOLD moves get -1 — caller must handle them separately."""
    m = PlanetMove(
        from_pid=0, angle=0.0, ships=0, target_pid=-1,
        kind=KIND_HOLD, prior=0.0, raw_score=0.0,
    )
    assert candidate_to_channel(m, available=100) == -1


def test_action_lookup_constants_match():
    """N_ACTION_CHANNELS and N_ANGLE_BUCKETS match ACTION_LOOKUP."""
    assert N_ACTION_CHANNELS == len(ACTION_LOOKUP)
    assert N_ANGLE_BUCKETS == 4  # current shipped action factorization


# ---------------------------------------------------------------------------
# Per-planet prior assignment
# ---------------------------------------------------------------------------


def _make_obs_with_planet(pid: int = 0, x: float = 50.0, y: float = 50.0):
    """Minimal obs dict with one planet at (x, y) — enough for
    ``nn_priors_for_planet`` to find it."""
    return {
        "planets": [[pid, 0, x, y, 2.0, 100.0, 3.0]],
        "fleets": [],
        "comet_planet_ids": [],
        "step": 0,
    }


def test_nn_prior_softmax_sums_to_one():
    """Three candidate moves on one planet -> priors sum to 1.0."""
    cfg = ConvPolicyCfg()
    model = ConvPolicy(cfg)
    obs = _make_obs_with_planet()
    moves = [
        PlanetMove(from_pid=0, angle=0.0, ships=100, target_pid=1,
                   kind=KIND_ATTACK_ENEMY, raw_score=0.0),
        PlanetMove(from_pid=0, angle=math.pi / 2, ships=50, target_pid=2,
                   kind=KIND_ATTACK_NEUTRAL, raw_score=0.0),
        PlanetMove(from_pid=0, angle=math.pi, ships=100, target_pid=3,
                   kind=KIND_ATTACK_ENEMY, raw_score=0.0),
    ]
    priors = nn_priors_for_planet(
        obs, player_id=0, moves=moves, available_ships=100,
        model=model, cfg=cfg,
    )
    assert len(priors) == 3
    assert sum(priors) == pytest.approx(1.0, abs=1e-5)
    # All non-negative.
    assert all(p >= 0 for p in priors)


def test_nn_prior_with_hold_gets_neutral_mass():
    """HOLD candidate doesn't get squashed to zero even when the NN is
    very confident on a non-hold move."""
    # Pin torch RNG so this test is deterministic regardless of pytest
    # ordering plugins that seed random/numpy but not torch.
    torch.manual_seed(0)
    cfg = ConvPolicyCfg()
    model = ConvPolicy(cfg)
    obs = _make_obs_with_planet()
    moves = [
        PlanetMove(from_pid=0, angle=0.0, ships=100, target_pid=1,
                   kind=KIND_ATTACK_ENEMY, raw_score=0.0),
        PlanetMove(from_pid=0, angle=0.0, ships=0, target_pid=-1,
                   kind=KIND_HOLD, raw_score=0.0),
    ]
    priors = nn_priors_for_planet(
        obs, player_id=0, moves=moves, available_ships=100,
        model=model, cfg=cfg, hold_neutral_prob=0.10,
    )
    # HOLD prior should be NEAR (within softmax noise) the configured
    # 0.10 floor. We allow a wide band because random init + softmax
    # does NOT exactly produce 0.10 — but it should not collapse to 0.
    assert priors[1] > 0.01


def test_nn_prior_empty_moves_is_empty_list():
    """Defensive: zero candidates -> empty prior list, no crash."""
    cfg = ConvPolicyCfg()
    model = ConvPolicy(cfg)
    obs = _make_obs_with_planet()
    priors = nn_priors_for_planet(
        obs, player_id=0, moves=[], available_ships=0,
        model=model, cfg=cfg,
    )
    assert priors == []


def test_nn_prior_unknown_planet_falls_back_to_uniform():
    """If the move references a planet that isn't in obs.planets (a
    stale state), the bridge returns a uniform distribution rather
    than crashing or NaN-ing."""
    cfg = ConvPolicyCfg()
    model = ConvPolicy(cfg)
    obs = _make_obs_with_planet(pid=0)  # planet 0 exists
    # ...but the move is from planet 999 (not in obs).
    moves = [
        PlanetMove(from_pid=999, angle=0.0, ships=10, target_pid=1,
                   kind=KIND_ATTACK_ENEMY, raw_score=0.0),
        PlanetMove(from_pid=999, angle=math.pi, ships=10, target_pid=2,
                   kind=KIND_ATTACK_ENEMY, raw_score=0.0),
    ]
    priors = nn_priors_for_planet(
        obs, player_id=0, moves=moves, available_ships=10,
        model=model, cfg=cfg,
    )
    assert priors == [0.5, 0.5]


def test_nn_prior_temperature_flattens():
    """Higher temperature -> more uniform distribution (entropy up)."""
    cfg = ConvPolicyCfg()
    model = ConvPolicy(cfg)
    obs = _make_obs_with_planet()
    moves = [
        PlanetMove(from_pid=0, angle=0.0, ships=100, target_pid=1,
                   kind=KIND_ATTACK_ENEMY, raw_score=0.0),
        PlanetMove(from_pid=0, angle=math.pi / 2, ships=50, target_pid=2,
                   kind=KIND_ATTACK_ENEMY, raw_score=0.0),
        PlanetMove(from_pid=0, angle=math.pi, ships=100, target_pid=3,
                   kind=KIND_ATTACK_ENEMY, raw_score=0.0),
    ]
    p_cold = nn_priors_for_planet(
        obs, 0, moves, 100, model, cfg, temperature=0.1,
    )
    p_hot = nn_priors_for_planet(
        obs, 0, moves, 100, model, cfg, temperature=10.0,
    )
    # Hot distribution should have higher entropy.
    h_cold = -sum(p * math.log(p + 1e-12) for p in p_cold)
    h_hot = -sum(p * math.log(p + 1e-12) for p in p_hot)
    assert h_hot > h_cold


# ---------------------------------------------------------------------------
# Closure factory
# ---------------------------------------------------------------------------


def test_make_nn_prior_fn_returns_dict():
    """The factory should return a callable that accepts the moves-by-
    planet dict shape produced by ``generate_per_planet_moves`` and
    returns the same shape with priors filled in."""
    cfg = ConvPolicyCfg()
    model = ConvPolicy(cfg)
    obs = {
        "planets": [
            [0, 0, 30.0, 30.0, 2.0, 100.0, 3.0],
            [1, 0, 70.0, 70.0, 2.0, 50.0, 3.0],
        ],
        "fleets": [],
        "comet_planet_ids": [],
        "step": 0,
    }
    moves_by_planet = {
        0: [
            PlanetMove(from_pid=0, angle=0.0, ships=100, target_pid=2,
                       kind=KIND_ATTACK_ENEMY, raw_score=0.0),
            PlanetMove(from_pid=0, angle=math.pi, ships=50, target_pid=3,
                       kind=KIND_ATTACK_ENEMY, raw_score=0.0),
        ],
        1: [
            PlanetMove(from_pid=1, angle=0.0, ships=50, target_pid=2,
                       kind=KIND_ATTACK_ENEMY, raw_score=0.0),
        ],
    }
    available = {0: 100, 1: 50}
    fn = make_nn_prior_fn(model, cfg)
    out = fn(obs, 0, moves_by_planet, available)

    assert set(out.keys()) == {0, 1}
    assert len(out[0]) == 2
    assert len(out[1]) == 1
    # Per-planet softmax invariant.
    assert sum(m.prior for m in out[0]) == pytest.approx(1.0, abs=1e-5)
    assert sum(m.prior for m in out[1]) == pytest.approx(1.0, abs=1e-5)
    # PlanetMove is frozen — the originals should be unchanged.
    assert moves_by_planet[0][0].prior == 0.0
    # New ones have NN-derived priors (typically not equal to 0).
    assert out[0][0].prior > 0.0


def test_make_nn_prior_fn_preserves_raw_score():
    """Heuristic raw_score is diagnostic — keep it through the wrap."""
    cfg = ConvPolicyCfg()
    model = ConvPolicy(cfg)
    obs = {"planets": [[0, 0, 50.0, 50.0, 2.0, 100.0, 3.0]],
           "fleets": [], "comet_planet_ids": [], "step": 0}
    moves_by_planet = {
        0: [
            PlanetMove(from_pid=0, angle=0.0, ships=100, target_pid=1,
                       kind=KIND_ATTACK_ENEMY, raw_score=42.0),
        ],
    }
    fn = make_nn_prior_fn(model, cfg)
    out = fn(obs, 0, moves_by_planet, {0: 100})
    assert out[0][0].raw_score == 42.0
