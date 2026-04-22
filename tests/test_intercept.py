"""Tests for intercept solvers. Run with: python -m pytest tests/test_intercept.py -q
(or the ad-hoc runner at the bottom if pytest isn't available.)
"""
from __future__ import annotations

import math

from orbitwars.engine.intercept import (
    CENTER,
    DEFAULT_MAX_SPEED,
    OrbitingTarget,
    comet_intercept,
    fleet_speed,
    is_orbiting_planet,
    orbiting_intercept,
    path_crosses_sun,
    route_angle_avoiding_sun,
    ships_needed_for_speed,
    static_intercept_angle,
    static_intercept_turns,
    sun_tangent_angles,
)


def test_fleet_speed_boundaries():
    assert fleet_speed(1) == 1.0
    # 1000 ships exactly should be max speed
    assert abs(fleet_speed(1000) - DEFAULT_MAX_SPEED) < 1e-9
    # Monotone non-decreasing in ships
    prev = 0.0
    for s in [1, 2, 5, 10, 50, 100, 250, 500, 750, 1000, 2000]:
        v = fleet_speed(s)
        assert v >= prev - 1e-9
        assert v <= DEFAULT_MAX_SPEED + 1e-9
        prev = v


def test_ships_needed_inverse():
    for s in [1, 5, 50, 500, 999]:
        v = fleet_speed(s)
        need = ships_needed_for_speed(v)
        # round-trip within a few ships (ceiling noise)
        assert need <= s + 1 and fleet_speed(need) >= v - 1e-6


def test_static_intercept_angle():
    # East
    a = static_intercept_angle((50, 50), (60, 50))
    assert abs(a - 0.0) < 1e-9
    # North-ish in screen coords (y+ is down? engine uses +y one direction;
    # we just match atan2 convention).
    a = static_intercept_angle((50, 50), (50, 60))
    assert abs(a - math.pi / 2) < 1e-9


def test_orbiting_intercept_zero_omega_matches_static():
    # With ω = 0 the target doesn't move; intercept should match static.
    src = (10.0, 10.0)
    ot = OrbitingTarget(
        orbital_radius=20.0,
        initial_angle=0.0,        # target at (70, 50)
        angular_velocity=0.0,
        current_step=0,
    )
    a, t, _ = orbiting_intercept(src, ot, ships=50)
    # Static intercept to (70, 50)
    a0 = static_intercept_angle(src, (70.0, 50.0))
    t0 = static_intercept_turns(src, (70.0, 50.0), ships=50)
    assert abs(a - a0) < 1e-6
    assert abs(t - t0) < 1e-4


def test_orbiting_intercept_nonzero_omega_lands_on_target():
    src = (5.0, 5.0)
    ot = OrbitingTarget(
        orbital_radius=20.0,
        initial_angle=0.3,
        angular_velocity=0.04,
        current_step=10,
    )
    ships = 100
    a, t, _ = orbiting_intercept(src, ot, ships)
    # Verify: at time t, the fleet position = source + v*t*(cos a, sin a)
    v = fleet_speed(ships)
    fx = src[0] + v * t * math.cos(a)
    fy = src[1] + v * t * math.sin(a)
    tx, ty = ot.position_at(t)
    assert abs(fx - tx) < 1e-3, f"x: {fx} vs {tx}"
    assert abs(fy - ty) < 1e-3, f"y: {fy} vs {ty}"


def test_comet_intercept_linear_path():
    # Comet moves in a straight line east at 1 unit/turn
    path = [(i * 1.0, 20.0) for i in range(100)]
    src = (0.0, 0.0)
    # At path_index_now = 0, comet is at (0, 20)
    # With a fast fleet (ships=1000, v=6), we should easily catch it.
    result = comet_intercept(src, path, path_index_now=0, ships=1000)
    assert result is not None
    angle, dt, idx = result
    # Verify geometry
    v = fleet_speed(1000)
    fx = src[0] + v * dt * math.cos(angle)
    fy = src[1] + v * dt * math.sin(angle)
    tx, ty = path[idx]
    assert abs(fx - tx) < 1e-3 and abs(fy - ty) < 1e-3


def test_path_crosses_sun_direct_line_through_center():
    assert path_crosses_sun((30, 50), (70, 50)) is True
    assert path_crosses_sun((0, 0), (100, 100)) is True  # through center
    # Off-axis, misses the sun
    assert path_crosses_sun((0, 0), (100, 0)) is False
    assert path_crosses_sun((0, 95), (100, 95)) is False


def test_sun_tangent_angles_symmetric():
    src = (20.0, 50.0)  # due west of center
    left, right = sun_tangent_angles(src)
    # Both tangents should be to the east (angle near 0), symmetric about 0
    assert abs(left + right) < 0.05  # symmetric about 0 modulo epsilon
    # And they should NOT cross the sun for a target slightly beyond the sun
    tx = 80.0
    for angle in (left, right):
        # Trace 60 units; check sun crossing
        ex = src[0] + 60 * math.cos(angle)
        ey = src[1] + 60 * math.sin(angle)
        # Allow a small numerical margin
        assert not path_crosses_sun(src, (ex, ey), clearance=-0.1)


def test_route_angle_avoiding_sun_passthrough():
    # If direct path doesn't cross sun, return direct angle unchanged.
    a = route_angle_avoiding_sun((0, 0), 0.0, (100, 0))
    assert abs(a - 0.0) < 1e-12


def test_route_angle_avoiding_sun_deflects():
    src = (20.0, 50.0)
    tgt = (80.0, 50.0)
    direct = static_intercept_angle(src, tgt)  # 0 rad, straight through sun
    rerouted = route_angle_avoiding_sun(src, direct, tgt)
    assert rerouted != direct
    # Trace rerouted a distance equal to src->tgt; verify no sun crossing
    dist = math.hypot(tgt[0] - src[0], tgt[1] - src[1])
    ex = src[0] + dist * math.cos(rerouted)
    ey = src[1] + dist * math.sin(rerouted)
    assert not path_crosses_sun(src, (ex, ey))


def test_is_orbiting_planet_classification():
    # Orbiting: close to center, small radius
    # planet = [id, owner, x, y, radius, ships, production]
    # initial @ (60, 50), r=1.0 -> orbital_radius = 10, 10+1=11 < 50 => orbiting
    p = [0, -1, 60.0, 50.0, 1.0, 5, 1]
    ip = [0, -1, 60.0, 50.0, 1.0, 5, 1]
    assert is_orbiting_planet(p, ip)
    # Static: far from center. (99, 50, r=2) -> orbital_radius=49, 49+2=51 >= 50 => static.
    p = [1, -1, 99.0, 50.0, 2.0, 5, 1]
    ip = [1, -1, 99.0, 50.0, 2.0, 5, 1]
    assert not is_orbiting_planet(p, ip)
    # Also check an edge-orbiting planet just under the limit.
    # (80, 50, r=1) -> orbital_radius=30, 30+1=31 < 50 => orbiting
    p = [2, -1, 80.0, 50.0, 1.0, 5, 1]
    ip = [2, -1, 80.0, 50.0, 1.0, 5, 1]
    assert is_orbiting_planet(p, ip)


if __name__ == "__main__":
    # Ad-hoc runner so we don't need pytest in the Week 1 minimum environment.
    import sys, traceback
    failures = 0
    tests = [(name, obj) for name, obj in globals().items()
             if name.startswith("test_") and callable(obj)]
    for name, fn in tests:
        try:
            fn()
            print(f"OK   {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {name}: {e}")
        except Exception:
            failures += 1
            print(f"ERR  {name}:")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
