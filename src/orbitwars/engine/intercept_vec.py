"""Batch (vectorized) variants of the intercept solvers.

The scalar versions in ``intercept.py`` are called ~600k times per
30-rollout MCTS turn from ``heuristic._plan_moves``; cProfile tags the
Newton inner loop in ``orbiting_intercept`` as the largest single
self-time consumer (~16% of heuristic.act). This module re-implements
the same algorithm in numpy so all (my_planet × target) intercepts for
one planning loop execute in O(1) numpy calls instead of O(N) Python
calls.

**Bit-equivalence contract**: the scalar ``orbiting_intercept`` uses
a freeze-on-convergence loop — once an element's ``|t_new - t| < tol``
or ``|df| < 1e-12``, that element's ``t`` is frozen and no further
updates apply. ``orbiting_intercept_batch`` mirrors this freeze
semantics via an ``active`` mask so the per-element output is the
same value the scalar version would have produced. Tests pin this
to <1e-12 absolute on a 200-pair random suite (see
``tests/test_intercept_vec.py``).

**When NOT bit-equal**: numpy's vectorized libm calls ``np.cos`` /
``np.sin`` are platform-dependent; on systems using SVML / SIMD libm
they may differ from CPython ``math.*`` in the last few bits. The
test suite enforces a generous absolute tolerance (1e-9) on output
``t`` and ``angle``; downstream callers in heuristic.py round ``t``
to integer arrival turns, so sub-1e-9 drift cannot change decisions.

API mirrors the scalar version: caller supplies arrays of source
positions, target orbital params, ship counts, and source offsets;
gets back arrays of ``(angle, t)``.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from orbitwars.engine.intercept import CENTER, DEFAULT_MAX_SPEED


def fleet_speed_batch(
    ships: np.ndarray,
    max_speed: float = DEFAULT_MAX_SPEED,
) -> np.ndarray:
    """Vectorized fleet_speed. Returns float64 array shape == ships.shape.

    Mirrors the scalar formula exactly:
        ships <= 0 -> 0.0
        ships == 1 -> 1.0
        else -> min(max_speed,
                    1 + (max_speed - 1) * (log(ships)/log(1000))^1.5)
    """
    ships = np.asarray(ships)
    # Guard log against ships <= 0 by computing on max(ships, 2) and
    # masking the zero/one-ship cases at the end. (log(2) is a finite
    # safe stand-in; the value is overwritten by the np.where below.)
    safe_ships = np.maximum(ships.astype(np.float64), 2.0)
    log_ratio = np.log(safe_ships) / np.log(1000.0)
    raw = 1.0 + (max_speed - 1.0) * np.power(log_ratio, 1.5)
    capped = np.minimum(raw, max_speed)
    out = np.where(
        ships <= 0,
        0.0,
        np.where(ships == 1, 1.0, capped),
    )
    return out.astype(np.float64)


def orbiting_intercept_batch(
    sources_xy: np.ndarray,         # (N, 2) float
    orb_r: np.ndarray,              # (N,) float
    init_angle: np.ndarray,         # (N,) float
    angular_velocity: float,        # scalar
    current_step: int,              # scalar
    ships: np.ndarray,              # (N,) int
    source_offset: np.ndarray,      # (N,) float
    max_iters: int = 8,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Newton-iteration intercept of N orbiting targets.

    Returns ``(angles, ts)`` arrays of shape ``(N,)``. Elements with
    ``ships <= 0`` get ``angle=0.0, t=inf`` to match scalar behavior.

    Iteration freezes per-element on:
      * ``|t_new - t| < tol``: element's ``t`` is updated to ``t_new``
        and frozen (matches scalar's "set t = t_new; break").
      * ``|df| < 1e-12``: element's ``t`` is left unchanged and frozen
        (matches scalar's "break before update").
    Other active elements continue iterating up to ``max_iters``.
    """
    N = ships.shape[0]
    if N == 0:
        return (np.empty(0), np.empty(0))

    v = fleet_speed_batch(ships)
    sx = sources_xy[:, 0].astype(np.float64)
    sy = sources_xy[:, 1].astype(np.float64)
    orb_r = np.asarray(orb_r, dtype=np.float64)
    init_angle = np.asarray(init_angle, dtype=np.float64)
    source_offset = np.asarray(source_offset, dtype=np.float64)
    omega = float(angular_velocity)
    step = float(current_step)

    # Initial guess: straight-line time to current target position.
    theta_now = init_angle + omega * step
    cx = CENTER + orb_r * np.cos(theta_now)
    cy = CENTER + orb_r * np.sin(theta_now)
    d0 = np.hypot(cx - sx, cy - sy)
    # Avoid div-by-zero for v=0 elements; they're masked out below.
    safe_v = np.where(v > 0.0, v, 1.0)
    t = np.maximum(0.0, (d0 - source_offset) / safe_v)
    # Per-element activity: starts True, freezes on convergence/df-zero.
    bad = v <= 0.0
    active = ~bad

    for _ in range(max_iters):
        if not active.any():
            break
        theta = init_angle + omega * (step + t)
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        tx = CENTER + orb_r * cos_th
        ty = CENTER + orb_r * sin_th
        dx = tx - sx
        dy = ty - sy
        rhs = source_offset + v * t
        f = dx * dx + dy * dy - rhs * rhs
        df = (
            2.0 * dx * (-orb_r * omega * sin_th)
            + 2.0 * dy * (orb_r * omega * cos_th)
            - 2.0 * rhs * v
        )
        df_zero = active & (np.abs(df) < 1e-12)
        # Compute dt for elements where df is non-zero. df_zero elements
        # get dt=0 effectively (we won't apply the update to them).
        safe_df = np.where(np.abs(df) < 1e-12, 1.0, df)
        dt = -f / safe_df
        t_new = np.maximum(0.0, t + dt)
        # Convergence detection (only meaningful for currently-active,
        # non-df-zero elements).
        converged_now = active & (~df_zero) & (np.abs(t_new - t) < tol)
        # Update rules:
        #   df_zero: keep old t (scalar broke before update)
        #   converged_now or otherwise active: take t_new
        #   inactive (already frozen): keep old t
        update_to_new = active & (~df_zero)
        t = np.where(update_to_new, t_new, t)
        # Freeze converged + df_zero elements.
        active = active & (~converged_now) & (~df_zero)

    # Final angle from final t.
    theta_final = init_angle + omega * (step + t)
    tx_f = CENTER + orb_r * np.cos(theta_final)
    ty_f = CENTER + orb_r * np.sin(theta_final)
    angle = np.arctan2(ty_f - sy, tx_f - sx)

    # Mask out bad-velocity elements.
    t_out = np.where(bad, np.inf, t)
    angle_out = np.where(bad, 0.0, angle)
    return angle_out, t_out


def static_intercept_turns_batch(
    sources_xy: np.ndarray,         # (N, 2)
    targets_xy: np.ndarray,         # (N, 2)
    ships: np.ndarray,              # (N,) int
    source_offset: np.ndarray,      # (N,)
) -> np.ndarray:
    """Batch ``static_intercept_turns``. Returns ``ts`` array."""
    v = fleet_speed_batch(ships)
    dx = targets_xy[:, 0] - sources_xy[:, 0]
    dy = targets_xy[:, 1] - sources_xy[:, 1]
    d = np.hypot(dx, dy)
    travel = np.maximum(0.0, d - source_offset)
    safe_v = np.where(v > 0.0, v, 1.0)
    t = travel / safe_v
    return np.where(v > 0.0, t, np.inf)


def static_intercept_angle_batch(
    sources_xy: np.ndarray,
    targets_xy: np.ndarray,
) -> np.ndarray:
    """Batch ``static_intercept_angle``. Returns angle array."""
    return np.arctan2(
        targets_xy[:, 1] - sources_xy[:, 1],
        targets_xy[:, 0] - sources_xy[:, 0],
    )
