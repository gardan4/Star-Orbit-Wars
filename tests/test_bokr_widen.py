"""Tests for `orbitwars.mcts.bokr_widen` — kernel-UCB sub-action selector.

Covers:
  * Grid construction: odd size, symmetric around base, correct spacing.
  * Kernel is circularly-wrap-aware (angles at 0 and 2pi-eps are close).
  * Unvisited arms return +inf UCB so ``select`` tries each grid point
    before revisiting.
  * Kernel mean converges to the true value on a noiseless surface.
  * UCB converges to the argmax on a noisy surface with enough budget.
  * Edge: empty history → ``best_angle`` returns ``base_angle``.
  * Edge: ``n_grid=1`` still works (degenerate, but legal).
  * Off-grid observations still contribute via the kernel.
"""
from __future__ import annotations

import math
import random

import pytest

from orbitwars.mcts.bokr_widen import (
    BOKRKernelSelector,
    _angular_diff,
    _gaussian_kernel,
)


# ---- Kernel math --------------------------------------------------------

def test_angular_diff_is_symmetric():
    assert _angular_diff(0.1, 0.5) == pytest.approx(0.4, abs=1e-9)
    assert _angular_diff(0.5, 0.1) == pytest.approx(0.4, abs=1e-9)


def test_angular_diff_wraps_at_2pi():
    """0 and 2pi-eps should be eps apart, not 2pi-eps apart."""
    eps = 0.05
    assert _angular_diff(0.0, 2.0 * math.pi - eps) == pytest.approx(eps, abs=1e-9)
    # And symmetrically.
    assert _angular_diff(2.0 * math.pi - eps, 0.0) == pytest.approx(eps, abs=1e-9)


def test_angular_diff_bounded_by_pi():
    """Max circular angular diff is pi."""
    assert _angular_diff(0.0, math.pi) == pytest.approx(math.pi, abs=1e-9)
    # Near-antipodal
    assert _angular_diff(0.0, math.pi - 0.01) == pytest.approx(math.pi - 0.01, abs=1e-9)
    assert _angular_diff(0.0, math.pi + 0.01) == pytest.approx(math.pi - 0.01, abs=1e-9)


def test_gaussian_kernel_peaks_at_zero_distance():
    """K(a, a) = 1 for any bandwidth."""
    for h in [0.01, 0.1, 1.0]:
        assert _gaussian_kernel(0.3, 0.3, h) == pytest.approx(1.0, abs=1e-9)


def test_gaussian_kernel_decays_with_distance():
    """Further angles get smaller weight."""
    w_close = _gaussian_kernel(0.0, 0.05, 0.1)
    w_far = _gaussian_kernel(0.0, 0.5, 0.1)
    assert w_close > w_far
    # Far past 3 bandwidths → near zero.
    assert w_far < 1e-4


# ---- Grid construction --------------------------------------------------

def test_selector_grid_is_symmetric_around_base():
    sel = BOKRKernelSelector(base_angle=1.0, angle_range=0.2, n_grid=5)
    grid = sel.candidate_angles()
    assert len(grid) == 5
    # Base is at the middle index.
    assert grid[2] == pytest.approx(1.0, abs=1e-9)
    # Endpoints at base ± range.
    assert grid[0] == pytest.approx(0.8, abs=1e-9)
    assert grid[4] == pytest.approx(1.2, abs=1e-9)


def test_selector_grid_odd_size_enforced():
    """Even n_grid → bumped to odd (so base is always a grid point)."""
    sel = BOKRKernelSelector(base_angle=0.0, angle_range=0.1, n_grid=4)
    grid = sel.candidate_angles()
    assert len(grid) == 5
    # Base at middle.
    assert grid[2] == pytest.approx(0.0, abs=1e-9)


def test_selector_degenerate_n_grid_one():
    """n_grid=1 → grid is just [base_angle]. Select always returns it."""
    sel = BOKRKernelSelector(base_angle=0.7, angle_range=0.5, n_grid=1)
    assert sel.candidate_angles() == [pytest.approx(0.7, abs=1e-9)]
    assert sel.select() == pytest.approx(0.7, abs=1e-9)


def test_selector_rejects_bad_params():
    with pytest.raises(ValueError):
        BOKRKernelSelector(base_angle=0.0, angle_range=-0.1, n_grid=5)
    with pytest.raises(ValueError):
        BOKRKernelSelector(base_angle=0.0, angle_range=0.1, n_grid=0)


def test_selector_default_bandwidth_is_half_grid_spacing():
    """kernel_h=None → defaults to 0.5 × grid spacing."""
    sel = BOKRKernelSelector(base_angle=0.0, angle_range=0.2, n_grid=5)
    # 5 points across 0.4 rad → spacing = 0.1; half = 0.05.
    assert sel.kernel_h == pytest.approx(0.05, abs=1e-9)


# ---- UCB selection ------------------------------------------------------

def test_select_visits_each_arm_before_revisiting():
    """With no data, every grid point has UCB=+inf; after one visit
    that point's UCB drops. ``select`` must prefer unvisited arms
    until the whole grid has been touched at least once."""
    sel = BOKRKernelSelector(base_angle=0.0, angle_range=0.2, n_grid=5)
    visited = set()
    for _ in range(5):
        theta = sel.select()
        visited.add(round(theta, 6))
        sel.update(theta, 0.0)
    assert len(visited) == 5


def test_best_angle_empty_history_returns_base():
    sel = BOKRKernelSelector(base_angle=0.42, angle_range=0.1, n_grid=3)
    assert sel.best_angle() == pytest.approx(0.42, abs=1e-9)


def test_kernel_mean_converges_on_noiseless_surface():
    """On v(θ) = 1 - (θ-θ*)^2 with θ* = base, the kernel mean at base
    should approach 1.0 as we pour in observations."""
    sel = BOKRKernelSelector(base_angle=0.0, angle_range=0.2, n_grid=9)
    for _ in range(5):
        for theta in sel.candidate_angles():
            v = 1.0 - theta * theta
            sel.update(theta, v)
    mean_at_base, n_eff = sel.kernel_mean(0.0)
    # Noiseless, base is the true max → kernel mean should be very close to 1.
    assert mean_at_base == pytest.approx(1.0, abs=0.05)
    assert n_eff > 0.0


def test_ucb_converges_to_offset_optimum():
    """Offset the true optimum to +0.1 rad from base. With enough
    budget, ``best_angle`` should return an angle close to +0.1, not
    base=0.0."""
    rng = random.Random(17)
    sel = BOKRKernelSelector(
        base_angle=0.0, angle_range=0.2, n_grid=9, c_ucb=1.4,
    )
    true_peak = 0.1
    for _ in range(300):
        theta = sel.select()
        # Quadratic payoff peaked at true_peak with small noise.
        v = 1.0 - (theta - true_peak) ** 2 + rng.gauss(0.0, 0.05)
        sel.update(theta, v)
    pick = sel.best_angle()
    assert abs(pick - true_peak) < 0.05, (
        f"best_angle={pick:.3f} should be near true_peak={true_peak}"
    )


def test_update_and_kernel_mean_share_value_across_neighbors():
    """Record at angle A only; kernel mean at A+ε should still be close
    to that value (value sharing — the whole point of BOKR)."""
    sel = BOKRKernelSelector(base_angle=0.0, angle_range=0.2, n_grid=9)
    sel.update(angle=0.0, value=0.8)
    mean_at_0, _ = sel.kernel_mean(0.0)
    # 0.05 rad away — should still be close to 0.8 (within a kernel width).
    mean_at_eps, n_eff_eps = sel.kernel_mean(0.05)
    assert mean_at_0 == pytest.approx(0.8, abs=1e-9)
    # Exactly 0.8 (single observation → weighted mean is that value).
    assert mean_at_eps == pytest.approx(0.8, abs=1e-9)
    # n_eff at epsilon is strictly < 1 (kernel weight < 1).
    assert 0.0 < n_eff_eps < 1.0


def test_ucb_exploration_bonus_decays_with_visits():
    """Repeated visits at a single grid point inflate its n_eff, which
    shrinks the bonus. Other (unvisited) points stay at +inf until the
    bonus at the visited point drops below the mean-based competition."""
    sel = BOKRKernelSelector(base_angle=0.0, angle_range=0.2, n_grid=5)
    # Force a visited point far from base so the kernel doesn't spill
    # too much to its neighbors.
    sel.update(angle=-0.2, value=0.5)
    ucb_first = sel.ucb_score(-0.2, n_total=1)
    for _ in range(50):
        sel.update(angle=-0.2, value=0.5)
    ucb_after = sel.ucb_score(-0.2, n_total=51)
    assert ucb_after < ucb_first, (
        f"UCB bonus should shrink with more visits; "
        f"before={ucb_first:.3f}, after={ucb_after:.3f}"
    )


def test_off_grid_observations_contribute_via_kernel():
    """update(angle) with angle not on the grid still influences future
    kernel_mean queries at nearby grid points."""
    sel = BOKRKernelSelector(base_angle=0.0, angle_range=0.2, n_grid=5)
    # Off-grid: grid points are at {-0.2, -0.1, 0, 0.1, 0.2} so 0.03
    # isn't on the grid.
    sel.update(angle=0.03, value=0.9)
    mean_at_0, _ = sel.kernel_mean(0.0)
    # With one obs the kernel-weighted mean IS that obs value.
    assert mean_at_0 == pytest.approx(0.9, abs=1e-9)


def test_n_observations_counts_all_updates():
    sel = BOKRKernelSelector(base_angle=0.0, angle_range=0.1, n_grid=3)
    assert sel.n_observations() == 0
    for _ in range(7):
        sel.update(angle=0.05, value=0.3)
    assert sel.n_observations() == 7


# ---- Circular wrap behavior --------------------------------------------

def test_selector_grid_wraps_into_symmetric_interval():
    """A grid near +pi should wrap cleanly into [-pi, pi] (cosmetic —
    kernel math is already wrap-aware)."""
    sel = BOKRKernelSelector(
        base_angle=math.pi - 0.05, angle_range=0.2, n_grid=5,
    )
    grid = sel.candidate_angles()
    for theta in grid:
        assert -math.pi <= theta <= math.pi + 1e-9


def test_kernel_treats_wrapping_angles_as_neighbors():
    """Observation at 2pi - 0.01 contributes to kernel_mean at +0.01:
    circular distance is 0.02, which is well inside a typical
    bandwidth."""
    sel = BOKRKernelSelector(
        base_angle=0.0, angle_range=0.05, n_grid=3, kernel_h=0.05,
    )
    sel.update(angle=2.0 * math.pi - 0.01, value=0.7)
    mean, n_eff = sel.kernel_mean(0.01)
    # Weight close to 1 because circular distance is tiny.
    assert mean == pytest.approx(0.7, abs=1e-9)
    assert n_eff > 0.8
