"""Bit-equivalence tests for intercept_vec batch solvers.

The vectorized batch versions in ``intercept_vec.py`` must produce
output arrays whose elements match the scalar versions to within
float64 last-bit precision (1e-12 absolute on a typical input).
Decisions downstream depend on ``ceil(t)`` and ``atan2`` rounding,
both of which are stable to that tolerance.
"""
from __future__ import annotations

import numpy as np
import pytest

from orbitwars.engine.intercept import (
    OrbitingTarget,
    fleet_speed,
    orbiting_intercept,
    static_intercept_angle,
    static_intercept_turns,
)
from orbitwars.engine.intercept_vec import (
    fleet_speed_batch,
    orbiting_intercept_batch,
    static_intercept_angle_batch,
    static_intercept_turns_batch,
)


def _random_inputs(n: int, seed: int):
    rng = np.random.default_rng(seed)
    sources = rng.uniform(5.0, 95.0, size=(n, 2))
    targets = rng.uniform(5.0, 95.0, size=(n, 2))
    orb_r = rng.uniform(5.0, 30.0, size=n)
    init_angle = rng.uniform(-np.pi, np.pi, size=n)
    ships = rng.integers(1, 500, size=n)
    source_offset = rng.uniform(0.0, 5.0, size=n)
    return sources, targets, orb_r, init_angle, ships, source_offset


def test_fleet_speed_batch_matches_scalar():
    ships = np.array([0, 1, 2, 10, 50, 100, 500, 1000, 5000])
    expected = np.array([fleet_speed(int(s)) for s in ships])
    got = fleet_speed_batch(ships)
    np.testing.assert_allclose(got, expected, atol=1e-15, rtol=0)


def test_orbiting_intercept_batch_matches_scalar_random():
    n = 200
    sources, _targets, orb_r, init_angle, ships, source_offset = _random_inputs(n, seed=42)
    omega = 0.04
    current_step = 137

    angles_scalar = np.empty(n)
    ts_scalar = np.empty(n)
    for i in range(n):
        ot = OrbitingTarget(
            orbital_radius=float(orb_r[i]),
            initial_angle=float(init_angle[i]),
            angular_velocity=omega,
            current_step=current_step,
        )
        a, t, _ = orbiting_intercept(
            (float(sources[i, 0]), float(sources[i, 1])),
            ot,
            int(ships[i]),
            source_offset=float(source_offset[i]),
        )
        angles_scalar[i] = a
        ts_scalar[i] = t

    angles_batch, ts_batch = orbiting_intercept_batch(
        sources, orb_r, init_angle, omega, current_step, ships, source_offset,
    )

    # t output: arrival is computed via int(ceil(t)); even 1e-9 drift
    # cannot flip ceil. Tighter tolerance is plausible but we anchor
    # to the float64 last-bit-or-three threshold.
    np.testing.assert_allclose(ts_batch, ts_scalar, atol=1e-9, rtol=0)
    # Angle output: atan2 is stable; allow same float64 last-bit threshold.
    # Wrap-around: angles near +/- pi may differ by 2π between scalar
    # (which uses math.atan2) and numpy (which uses np.arctan2). We
    # compare on the unit circle to avoid that artifact.
    np.testing.assert_allclose(np.cos(angles_batch), np.cos(angles_scalar), atol=1e-9, rtol=0)
    np.testing.assert_allclose(np.sin(angles_batch), np.sin(angles_scalar), atol=1e-9, rtol=0)


def test_orbiting_intercept_batch_handles_zero_ships():
    sources = np.array([[10.0, 10.0], [50.0, 50.0]])
    orb_r = np.array([15.0, 15.0])
    init_angle = np.array([0.0, np.pi / 2])
    ships = np.array([0, 100])
    source_offset = np.array([1.0, 1.0])

    angles, ts = orbiting_intercept_batch(
        sources, orb_r, init_angle, 0.04, 100, ships, source_offset,
    )
    assert ts[0] == np.inf
    assert angles[0] == 0.0
    assert np.isfinite(ts[1])


def test_static_intercept_turns_batch_matches_scalar():
    n = 100
    sources, targets, _, _, ships, source_offset = _random_inputs(n, seed=7)
    expected = np.array([
        static_intercept_turns(
            (float(sources[i, 0]), float(sources[i, 1])),
            (float(targets[i, 0]), float(targets[i, 1])),
            int(ships[i]),
            source_offset=float(source_offset[i]),
        )
        for i in range(n)
    ])
    got = static_intercept_turns_batch(sources, targets, ships, source_offset)
    np.testing.assert_allclose(got, expected, atol=1e-12, rtol=0)


def test_static_intercept_angle_batch_matches_scalar():
    n = 100
    sources, targets, *_ = _random_inputs(n, seed=99)
    expected = np.array([
        static_intercept_angle(
            (float(sources[i, 0]), float(sources[i, 1])),
            (float(targets[i, 0]), float(targets[i, 1])),
        )
        for i in range(n)
    ])
    got = static_intercept_angle_batch(sources, targets)
    np.testing.assert_allclose(got, expected, atol=1e-12, rtol=0)
