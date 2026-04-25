"""Tests for BC 4-fold rotational augmentation."""
from __future__ import annotations

import numpy as np
import pytest

from orbitwars.nn.bc_augment import (
    augment_4fold,
    rotate_demo,
    _FLEET_COS_CH,
    _FLEET_SIN_CH,
)


def _mk_demo(N: int = 50) -> np.ndarray:
    """Build a deterministic (C=12, H=N, W=N) tensor with a single
    "marker" planet at a known cell."""
    x = np.zeros((12, N, N), dtype=np.float32)
    return x


def test_identity_rotation_is_noop():
    x = _mk_demo()
    x[0, 10, 20] = 7.0
    rx, gy, gx, lb = rotate_demo(x, 10, 20, 3, k=0)
    np.testing.assert_array_equal(rx, x)
    assert (gy, gx, lb) == (10, 20, 3)


def test_90_ccw_moves_corner_correctly():
    """Plant a marker at (gy=5, gx=10). After 90° CCW (k=1),
    expect new (gy', gx') = (49 - 10, 5) = (39, 5)."""
    x = _mk_demo(50)
    x[0, 5, 10] = 1.0
    rx, gy, gx, lb = rotate_demo(x, 5, 10, 0, k=1)
    assert (gy, gx) == (39, 5)
    # The marker should now sit at (39, 5) in the rotated tensor.
    assert rx[0, 39, 5] == 1.0
    assert rx[0, 5, 10] == 0.0


def test_180_inverts_position():
    x = _mk_demo(50)
    x[0, 7, 13] = 1.0
    rx, gy, gx, lb = rotate_demo(x, 7, 13, 0, k=2)
    assert (gy, gx) == (42, 36)
    assert rx[0, 42, 36] == 1.0


def test_270_ccw_flips_sides():
    x = _mk_demo(50)
    x[0, 10, 20] = 1.0
    rx, gy, gx, lb = rotate_demo(x, 10, 20, 0, k=3)
    assert (gy, gx) == (20, 39)
    assert rx[0, 20, 39] == 1.0


def test_action_label_shifts_by_2k_per_quarter():
    """Action label = bucket*2 + frac. Under k * 90° CCW, label shifts
    by 2*k modulo 8."""
    x = _mk_demo()
    for orig in range(8):
        for k in range(4):
            _, _, _, lb = rotate_demo(x, 0, 0, orig, k=k)
            assert lb == (orig + 2 * k) % 8


def test_angle_vector_channels_rotate_correctly():
    """Plant a fleet at (gy=10, gx=20) with angle 0 (cos=1, sin=0,
    pointing East). After 90° CCW, the cell moves to
    (49-20, 10) = (29, 10) and the vector should now point North:
    (cos=0, sin=1)."""
    x = _mk_demo(50)
    x[_FLEET_COS_CH, 10, 20] = 1.0
    x[_FLEET_SIN_CH, 10, 20] = 0.0
    rx, gy, gx, lb = rotate_demo(x, 10, 20, 0, k=1)
    assert (gy, gx) == (29, 10)
    np.testing.assert_almost_equal(rx[_FLEET_COS_CH, 29, 10], 0.0, decimal=5)
    np.testing.assert_almost_equal(rx[_FLEET_SIN_CH, 29, 10], 1.0, decimal=5)


def test_angle_vector_180_negates():
    """A vector pointing E (cos=1, sin=0) under 180° points W
    (cos=-1, sin=0). Cell (10, 20) → (39, 29)."""
    x = _mk_demo(50)
    x[_FLEET_COS_CH, 10, 20] = 1.0
    rx, gy, gx, _ = rotate_demo(x, 10, 20, 0, k=2)
    assert (gy, gx) == (39, 29)
    np.testing.assert_almost_equal(rx[_FLEET_COS_CH, 39, 29], -1.0, decimal=5)
    np.testing.assert_almost_equal(rx[_FLEET_SIN_CH, 39, 29], 0.0, decimal=5)


def test_angle_vector_at_pi_2_rotates_to_pi():
    """A vector pointing N (cos=0, sin=1) under 90° CCW points W
    (cos=-1, sin=0). Cell (10, 20) → (29, 10)."""
    x = _mk_demo(50)
    x[_FLEET_COS_CH, 10, 20] = 0.0
    x[_FLEET_SIN_CH, 10, 20] = 1.0
    rx, gy, gx, _ = rotate_demo(x, 10, 20, 0, k=1)
    assert (gy, gx) == (29, 10)
    np.testing.assert_almost_equal(rx[_FLEET_COS_CH, 29, 10], -1.0, decimal=5)
    np.testing.assert_almost_equal(rx[_FLEET_SIN_CH, 29, 10], 0.0, decimal=5)


def test_4_rotations_compose_to_identity():
    """Rotating 4 times by 90° must return the original demo."""
    x = _mk_demo(50)
    x[0, 5, 10] = 1.0
    x[_FLEET_COS_CH, 7, 13] = 0.6
    x[_FLEET_SIN_CH, 7, 13] = 0.8
    cur_x, cur_gy, cur_gx, cur_lb = x.copy(), 5, 10, 3
    for _ in range(4):
        cur_x, cur_gy, cur_gx, cur_lb = rotate_demo(cur_x, cur_gy, cur_gx, cur_lb, k=1)
    np.testing.assert_allclose(cur_x, x, atol=1e-5)
    assert (cur_gy, cur_gx, cur_lb) == (5, 10, 3)


def test_augment_4fold_quadruples_dataset():
    N = 5
    x = np.random.RandomState(0).randn(N, 12, 50, 50).astype(np.float32)
    gy = np.array([3, 7, 11, 15, 25], dtype=np.int64)
    gx = np.array([10, 20, 30, 40, 5], dtype=np.int64)
    labels = np.array([0, 1, 2, 5, 7], dtype=np.int64)

    ax, agy, agx, alb = augment_4fold(x, gy, gx, labels)
    assert ax.shape == (4 * N, 12, 50, 50)
    assert agy.shape == (4 * N,)
    assert agx.shape == (4 * N,)
    assert alb.shape == (4 * N,)
    # First N should be identity.
    np.testing.assert_array_equal(ax[:N], x)
    np.testing.assert_array_equal(agy[:N], gy)
    np.testing.assert_array_equal(agx[:N], gx)
    np.testing.assert_array_equal(alb[:N], labels)


def test_augment_4fold_matches_rotate_demo_per_row():
    """The vectorized augment must match the per-demo rotate_demo
    for every (row, k) combination."""
    N = 3
    rng = np.random.RandomState(42)
    x = rng.randn(N, 12, 50, 50).astype(np.float32)
    gy = np.array([5, 15, 25], dtype=np.int64)
    gx = np.array([10, 20, 35], dtype=np.int64)
    labels = np.array([0, 3, 6], dtype=np.int64)

    ax, agy, agx, alb = augment_4fold(x, gy, gx, labels)

    for k in range(4):
        for i in range(N):
            ref_x, ref_gy, ref_gx, ref_lb = rotate_demo(
                x[i], gy[i], gx[i], labels[i], k=k,
            )
            row = k * N + i
            np.testing.assert_allclose(ax[row], ref_x, atol=1e-5)
            assert agy[row] == ref_gy
            assert agx[row] == ref_gx
            assert alb[row] == ref_lb


def test_augment_4fold_preserves_dtypes():
    N = 3
    x = np.zeros((N, 12, 50, 50), dtype=np.float32)
    gy = np.zeros(N, dtype=np.int64)
    gx = np.zeros(N, dtype=np.int64)
    labels = np.zeros(N, dtype=np.int64)
    ax, agy, agx, alb = augment_4fold(x, gy, gx, labels)
    assert ax.dtype == np.float32
    assert agy.dtype == np.int64
    assert agx.dtype == np.int64
    assert alb.dtype == np.int64


def test_augment_4fold_handles_soft_target_labels():
    """Soft visit-distribution labels (N, 8 floats) should rotate by
    rolling the 8 channels by 2*k under k * 90° CCW."""
    N = 2
    rng = np.random.RandomState(0)
    x = rng.randn(N, 12, 50, 50).astype(np.float32)
    gy = np.array([5, 15], dtype=np.int64)
    gx = np.array([10, 20], dtype=np.int64)
    # Soft labels: deterministic distributions, two distinct shapes.
    labels = np.array([
        [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],  # row 0: E + W
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # row 1: pure N-50%
    ], dtype=np.float32)
    ax, agy, agx, alb = augment_4fold(x, gy, gx, labels)
    assert alb.shape == (4 * N, 8)
    assert alb.dtype == np.float32

    # k=0 (identity) preserves rows.
    np.testing.assert_array_equal(alb[:N], labels)

    # k=1 (90° CCW): channels shift by 2. Row 0's [0.5, 0, 0, 0, 0.5, 0, 0, 0]
    # becomes [0, 0, 0.5, 0, 0, 0, 0.5, 0].
    np.testing.assert_array_almost_equal(
        alb[N + 0],
        np.array([0, 0, 0.5, 0, 0, 0, 0.5, 0]),
    )
    # Row 1's [0, 0, 1, 0, 0, 0, 0, 0] becomes [0, 0, 0, 0, 1, 0, 0, 0].
    np.testing.assert_array_almost_equal(
        alb[N + 1],
        np.array([0, 0, 0, 0, 1, 0, 0, 0]),
    )

    # k=2 (180°): channels shift by 4. Row 0 unchanged (E + W is its own
    # 180° image); row 1 [0,0,1,0,0,0,0,0] → [0,0,0,0,0,0,1,0].
    np.testing.assert_array_almost_equal(alb[2 * N + 0], labels[0])
    np.testing.assert_array_almost_equal(
        alb[2 * N + 1],
        np.array([0, 0, 0, 0, 0, 0, 1, 0]),
    )

    # k=3 (270° CCW): channels shift by 6. Row 1 [0,0,1,0,0,0,0,0]
    # → [0, 0, 0, 0, 0, 0, 0, 0] then add the 1 at (2+6)%8 = 0:
    #   [1, 0, 0, 0, 0, 0, 0, 0].
    np.testing.assert_array_almost_equal(
        alb[3 * N + 1],
        np.array([1, 0, 0, 0, 0, 0, 0, 0]),
    )
