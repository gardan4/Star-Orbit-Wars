# Auto-generated Orbit Wars submission. Do not edit by hand.
# Built by tools/bundle.py on 2026-04-23 14:55:49.
# Bot: heuristic
#
# Single-file submission: Kaggle will import this and call agent(obs, cfg).
from __future__ import annotations


# --- inlined: orbitwars/engine/intercept.py ---

"""Intercept solvers for straight-line fleets against static, orbiting, and
comet-path targets in Orbit Wars.

Game facts driving the math (cross-checked against
kaggle_environments/envs/orbit_wars/orbit_wars.py v1.0.9):

  * Fleets travel in straight lines at constant speed for their lifetime.
    Speed depends only on fleet size at launch:
        speed = 1 + (max_speed - 1) * (log(ships) / log(1000)) ** 1.5
        speed = min(speed, max_speed)           # default max_speed = 6
    Single ship -> 1.0/turn; 1000-ship fleet -> 6.0/turn (the cap).
  * Planets with orbital_radius + planet_radius < ROTATION_RADIUS_LIMIT (50)
    rotate around the board center at a fixed, game-global angular velocity ω
    in (0.025, 0.05) rad/turn. Position at time t (turns from now):
        θ(t) = θ0 + ω*t
        pos(t) = (cx + r*cos θ(t), cy + r*sin θ(t))
  * Comets move along a precomputed path (list of (x, y)) with `path_index`
    advancing by 1 each turn.
  * Sun at (50, 50) radius 10 destroys any fleet whose path segment comes
    within 10 of the center. When the direct line crosses the sun we route
    via a tangent angle ±ε.

No gravity — fleets are straight-line. (The engine has no force model.)
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# Mirror the engine constants so we don't depend on importing the engine here.
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
BOARD_SIZE = 100.0
DEFAULT_MAX_SPEED = 6.0

TWO_PI = 2.0 * math.pi


# ---- Fleet speed (exact match to engine) ----

def fleet_speed(ships: int, max_speed: float = DEFAULT_MAX_SPEED) -> float:
    """Engine's fleet speed formula. ships >= 1."""
    if ships <= 0:
        return 0.0
    if ships == 1:
        return 1.0
    s = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5
    return min(s, max_speed)


def ships_needed_for_speed(target_speed: float, max_speed: float = DEFAULT_MAX_SPEED) -> int:
    """Inverse of fleet_speed. Ceiling ships count to reach `target_speed`."""
    if target_speed <= 1.0:
        return 1
    if target_speed >= max_speed:
        # max_speed is hit at 1000 ships exactly.
        return 1000
    frac = (target_speed - 1.0) / (max_speed - 1.0)
    log_ships = math.log(1000.0) * frac ** (1.0 / 1.5)
    return max(1, int(math.ceil(math.exp(log_ships))))


# ---- Static intercept ----

def static_intercept_angle(
    source: Tuple[float, float],
    target: Tuple[float, float],
) -> float:
    """Angle (radians) pointing from source directly at target."""
    return math.atan2(target[1] - source[1], target[0] - source[0])


def static_intercept_turns(
    source: Tuple[float, float],
    target: Tuple[float, float],
    ships: int,
    source_offset: float = 0.0,
) -> float:
    """Turns for a fleet of `ships` ships from `source` to reach `target`.

    ``source_offset`` accounts for the engine's launch offset: on the launch
    turn, the engine places the fleet at ``source + (source_planet_radius +
    0.1) * dir(angle)`` — not at the planet centre. Pass
    ``source_offset = source_planet_radius + 0.1`` to produce an arrival-time
    estimate that matches what the engine will observe. Default 0.0 keeps
    backward compatibility for callers that already pass launch-offset
    positions as ``source`` (e.g. in-flight fleets in ``build_arrival_table``).
    """
    dx = target[0] - source[0]
    dy = target[1] - source[1]
    d = math.hypot(dx, dy)
    v = fleet_speed(ships)
    if v <= 0:
        return float("inf")
    # Fleet already has the offset distance "covered" by the launch-placement;
    # travel time is the remaining straight-line distance at speed v.
    travel = max(0.0, d - source_offset)
    return travel / v


# ---- Orbiting-planet intercept (Newton iteration) ----

@dataclass
class OrbitingTarget:
    """Target orbiting the board center at (cx, cy) with fixed angular velocity.

    initial_angle and orbital_radius come from the observation's
    `initial_planets` entry. The current observed position is
        (cx + r cos(θ0 + ω t_now), cy + r sin(θ0 + ω t_now))
    where t_now is the current step count.
    """
    orbital_radius: float
    initial_angle: float       # θ0, radians
    angular_velocity: float    # ω, rad/turn
    current_step: int          # t_now (so we compute θ at t = t_now + Δt)

    def position_at(self, delta_t: float) -> Tuple[float, float]:
        θ = self.initial_angle + self.angular_velocity * (self.current_step + delta_t)
        return (CENTER + self.orbital_radius * math.cos(θ),
                CENTER + self.orbital_radius * math.sin(θ))


def orbiting_intercept(
    source: Tuple[float, float],
    orbit: OrbitingTarget,
    ships: int,
    max_iters: int = 8,
    tol: float = 1e-4,
    source_offset: float = 0.0,
) -> Tuple[float, float, int]:
    """Solve for time-of-flight Δt such that
    |orbit(Δt) - source|² = (source_offset + v·Δt)².

    Returns (angle_to_intercept, delta_t_turns, iters_used).

    ``source_offset`` accounts for the engine launching the fleet at
    ``source + (r + 0.1) * dir(angle)`` rather than at ``source`` itself.
    For a fleet launched from a planet of radius ``r``, pass
    ``source_offset = r + 0.1`` so the Newton matches engine trajectory.
    Callers who already pass a launch-offset-adjusted source (e.g.
    mid-flight fleets) should leave it 0.0.

    Uses Newton on f(t) = (orbit.x(t) - sx)² + (orbit.y(t) - sy)² -
                          (source_offset + v·t)².

    Initial guess: time for straight-line intercept of the *current*
    orbit position with the offset subtracted — exact for ω = 0 and a
    good start otherwise.
    """
    v = fleet_speed(ships)
    if v <= 0.0:
        return (0.0, float("inf"), 0)

    sx, sy = source
    r = orbit.orbital_radius
    ω = orbit.angular_velocity
    # Current position of the target, used only for initial guess.
    cur = orbit.position_at(0.0)
    d0 = math.hypot(cur[0] - sx, cur[1] - sy)
    t = max(0.0, (d0 - source_offset) / v)

    for i in range(max_iters):
        θ = orbit.initial_angle + ω * (orbit.current_step + t)
        tx = CENTER + r * math.cos(θ)
        ty = CENTER + r * math.sin(θ)
        dx = tx - sx
        dy = ty - sy
        rhs = source_offset + v * t
        # f(t) = dx² + dy² - (source_offset + v·t)²
        f = dx * dx + dy * dy - rhs * rhs
        # df/dt = 2 dx * (-r ω sin θ) + 2 dy * (r ω cos θ) - 2·rhs·v
        df = 2.0 * dx * (-r * ω * math.sin(θ)) \
             + 2.0 * dy * (r * ω * math.cos(θ)) \
             - 2.0 * rhs * v
        if abs(df) < 1e-12:
            break
        dt = -f / df
        t_new = max(0.0, t + dt)
        if abs(t_new - t) < tol:
            t = t_new
            break
        t = t_new

    θ_final = orbit.initial_angle + ω * (orbit.current_step + t)
    tx = CENTER + r * math.cos(θ_final)
    ty = CENTER + r * math.sin(θ_final)
    angle = math.atan2(ty - sy, tx - sx)
    return (angle, t, i + 1)


# ---- Point-to-segment distance (engine-parity util) ----

def point_to_segment_distance(
    pt: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    """Distance from ``pt`` to the segment [a, b]. Matches the engine's
    ``point_to_segment_distance`` helper byte-for-byte, so using it for
    obstruction / sun-crossing predictions mirrors what the engine will
    actually compute at collision time.
    """
    px, py = pt
    ax, ay = a
    bx, by = b
    abx = bx - ax
    aby = by - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / ab_len_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


# ---- Comet intercept (path-indexed, linear scan) ----

def comet_intercept(
    source: Tuple[float, float],
    comet_path: Sequence[Tuple[float, float]],
    path_index_now: int,
    ships: int,
    max_time_mismatch: float = 1.0,
    source_offset: float = 0.0,
) -> Optional[Tuple[float, float, int]]:
    """Find the earliest future path index where fleet arrival time matches
    the comet's arrival time at that index.

    Returns (angle, delta_t_to_fleet_arrival, path_index) or None.

    Phase convention (matches engine v1.0.9 step order):
      * ``path_index_now`` = ``obs.comets[*].path_index`` — the comet's
        current position is ``comet_path[path_index_now]``.
      * On engine turn S (= fleet-turn 1 for a freshly launched fleet),
        fleet-vs-planet collision runs in step 3 with the comet STILL at
        ``path[path_index_now]`` (the comet doesn't move until step 5
        of the same turn). On fleet-turn k, the step-3 collision sees
        the comet at ``path[path_index_now + k - 1]``.
      * Therefore: if we aim at ``path[idx]`` and want fleet-turn k
        collision to hit, we need ``k = idx - path_index_now + 1`` AND
        the fleet to have traveled ``source_offset + k*v`` units of
        distance. Equating: fleet travel time (``dist - source_offset) /
        v``) should equal ``idx - path_index_now + 1``.

    ``source_offset`` accounts for the engine launch offset
    ``(source_radius + 0.1)`` — pass it in so the fleet-travel distance
    matches the engine's actual fleet position. Default 0.0 matches
    legacy callers that supply a launch-adjusted source.

    Algorithm: scan forward from ``path_index_now`` and return the first
    index whose mismatch ``|t_arrive - (idx - path_index_now + 1)|`` is
    within ``max_time_mismatch``. The engine's continuous sweep will
    trigger a collision when trajectories cross inside the band.
    """
    v = fleet_speed(ships)
    if v <= 0.0:
        return None

    sx, sy = source
    # Start scanning at the current comet position. For fleet-turn 1 the
    # comet is still at path[path_index_now] during step-3 collision, so
    # aiming at path[path_index_now] CAN hit if the comet is within v
    # units (rare but valid).
    start_idx = max(0, path_index_now)
    best_idx = None
    best_mismatch = float("inf")
    # Monotonicity: ``mismatch = t_arrive - k_engine`` is monotonically
    # non-increasing in idx whenever comet speed (≈1/turn) ≤ fleet speed
    # (≥1/turn). Increasing idx adds at most ~1 to dist (so ≤ 1/v to
    # t_arrive) but adds exactly 1 to k_engine. So the sequence starts
    # large positive (fleet very late, comet close) and decreases
    # through 0 to negative (fleet very early, comet far future). The
    # acceptable band is the middle chunk; we want the FIRST idx in it.
    for idx in range(start_idx, len(comet_path)):
        tx, ty = comet_path[idx]
        dist = math.hypot(tx - sx, ty - sy)
        # Effective fleet travel time from launch to path[idx], i.e. the
        # number of fleet-turns (at constant speed v) needed to cover
        # the straight-line distance minus the launch-offset prefix.
        t_arrive = max(0.0, (dist - source_offset) / v)
        # Turn number on which the engine's step-3 collision sees the
        # comet at path[idx]. Fleet-turn numbering starts at 1.
        k_engine = float(idx - path_index_now + 1)
        mismatch = t_arrive - k_engine  # +ve = fleet late, -ve = fleet early
        # If fleet arrives much later than comet at this idx (comet is
        # still close to source, fleet hasn't caught up yet), try next
        # (further) idx — k_engine grows faster than t_arrive, so the
        # mismatch will come down into band shortly.
        if mismatch > max_time_mismatch:
            continue
        # If fleet would arrive much earlier than comet at this idx,
        # every further idx will be even earlier (since mismatch is
        # monotonically decreasing). No intercept possible — stop.
        if mismatch < -max_time_mismatch:
            break
        # In-band: record and stop at the first acceptable index.
        if abs(mismatch) < best_mismatch:
            best_mismatch = abs(mismatch)
            best_idx = idx
        break

    if best_idx is None:
        return None
    tx, ty = comet_path[best_idx]
    angle = math.atan2(ty - sy, tx - sx)
    t_arrive = max(0.0, (math.hypot(tx - sx, ty - sy) - source_offset) / v)
    return (angle, t_arrive, best_idx)


# ---- Sun-tangent routing ----

def path_crosses_sun(
    source: Tuple[float, float],
    target: Tuple[float, float],
    sun_radius: float = SUN_RADIUS,
    clearance: float = 0.5,
) -> bool:
    """True if the straight segment source->target comes within sun_radius
    (+clearance) of the board center.
    """
    sx, sy = source
    tx, ty = target
    dx, dy = tx - sx, ty - sy
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return False
    # Projection of center onto line, clamped to segment
    t = ((CENTER - sx) * dx + (CENTER - sy) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    px = sx + t * dx
    py = sy + t * dy
    d = math.hypot(px - CENTER, py - CENTER)
    return d < (sun_radius + clearance)


def sun_tangent_angles(
    source: Tuple[float, float],
    sun_radius: float = SUN_RADIUS,
    epsilon: float = 0.02,
) -> Tuple[float, float]:
    """Return (left_tangent_angle, right_tangent_angle) — the two angles from
    source tangent to the sun (plus a small safety epsilon).

    If source is inside the sun, this is undefined; we return two angles straight
    outward and let the caller decide.
    """
    sx, sy = source
    dx = CENTER - sx
    dy = CENTER - sy
    d = math.hypot(dx, dy)
    if d <= sun_radius + 1e-6:
        # Inside the sun — return opposite-directions radially outward.
        a = math.atan2(-dy, -dx)
        return (a - 0.1, a + 0.1)
    theta = math.atan2(dy, dx)      # angle toward sun center
    phi = math.asin(min(1.0, sun_radius / d))  # half-angle of sun disk
    return (theta + phi + epsilon, theta - phi - epsilon)


def route_angle_avoiding_sun(
    source: Tuple[float, float],
    direct_angle: float,
    target: Tuple[float, float],
) -> float:
    """If the direct path crosses the sun, return the better tangent angle;
    otherwise return direct_angle unchanged.

    "Better" = the tangent closer in angular distance to `direct_angle`.
    """
    if not path_crosses_sun(source, target):
        return direct_angle
    left, right = sun_tangent_angles(source)

    def _ang_dist(a, b):
        d = (a - b) % TWO_PI
        return d if d <= math.pi else TWO_PI - d

    return left if _ang_dist(left, direct_angle) <= _ang_dist(right, direct_angle) else right


# ---- Helper: detect if a planet is orbiting from the current observation ----

def is_orbiting_planet(planet: Sequence, initial_planet: Sequence) -> bool:
    """Engine rule: rotates if orbital_radius + radius < ROTATION_RADIUS_LIMIT.

    Uses initial_planet[x, y] for the static orbital radius reference
    (planet[x, y] may already have rotated).
    """
    r = planet[4]
    dx = initial_planet[2] - CENTER
    dy = initial_planet[3] - CENTER
    orb_r = math.hypot(dx, dy)
    return (orb_r + r) < ROTATION_RADIUS_LIMIT


def initial_orbit_params(initial_planet: Sequence) -> Tuple[float, float]:
    """Return (orbital_radius, initial_angle) from an `initial_planets` entry."""
    dx = initial_planet[2] - CENTER
    dy = initial_planet[3] - CENTER
    return (math.hypot(dx, dy), math.atan2(dy, dx))



# --- inlined: orbitwars/bots/base.py ---

"""Base agent interface with hard timing guard and fallback action.

All bots in this project inherit `Agent` and implement `act(obs) -> list`. The
wrapper enforces:
  * A valid no-op fallback is always available.
  * Per-turn wall-clock is audited; if `act` overruns, the wrapper returns the
    best-so-far action (if the bot staged one) or the fallback.
  * gc is disabled at module load; one manual `gc.collect()` between turns keeps
    latency spikes out of the 1-second budget.

Kaggle's official agent contract is a plain callable `agent(obs, cfg=None) -> list`.
`Agent.as_kaggle_agent()` produces such a callable so bots can be submitted
as-is.
"""

import gc
import math
import time
from typing import Callable, List

# Action type: list of [from_planet_id, angle_rad, num_ships]
Move = List[float]
Action = List[Move]

# Disable gc once at module import. Individual agents explicitly collect between
# turns to avoid latency spikes during search.
gc.disable()

# Safety margins. actTimeout is 1s; we stop search at 850ms, return by 900ms.
HARD_DEADLINE_MS = 900.0
SEARCH_DEADLINE_MS = 850.0
EARLY_FALLBACK_MS = 200.0  # Must have a valid action staged by this time.


def no_op() -> Action:
    """Always-valid null action."""
    return []


class Deadline:
    """Per-turn wall-clock timer with best-so-far action buffer.

    Usage inside an agent:
        dl = Deadline()
        dl.stage(fallback_action_here)           # by EARLY_FALLBACK_MS
        while dl.remaining_ms() > slack:
            improved = search_one_step()
            dl.stage(improved)
        return dl.best()

    ``hard_stop_at`` (optional, in ``time.perf_counter()`` seconds) is an
    *external* absolute deadline. When set, ``should_stop()`` fires at
    that instant regardless of per-call elapsed time. Used by MCTS
    rollouts to propagate the search's outer deadline into the rollout's
    inner ``HeuristicAgent.act`` calls — without this, a single in-flight
    heuristic.act on a dense mid-game state (400-700 ms observed) can
    blow past the outer deadline while its own per-call ``Deadline()``
    still shows "plenty of time left".
    """

    __slots__ = ("_t0", "_best", "_hard_stop_at")

    def __init__(self, hard_stop_at: float | None = None) -> None:
        self._t0 = time.perf_counter()
        self._best: Action = no_op()
        self._hard_stop_at = hard_stop_at

    def stage(self, action: Action) -> None:
        """Mark this action as the current best; returned if deadline hits."""
        self._best = action

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def remaining_ms(self, deadline_ms: float = SEARCH_DEADLINE_MS) -> float:
        return deadline_ms - self.elapsed_ms()

    def should_stop(self, deadline_ms: float = SEARCH_DEADLINE_MS) -> bool:
        # An external hard stop (e.g. outer MCTS deadline) always wins.
        if self._hard_stop_at is not None:
            if time.perf_counter() >= self._hard_stop_at:
                return True
        return self.elapsed_ms() >= deadline_ms

    def best(self) -> Action:
        return self._best


class Agent:
    """Base class for all bots in this project.

    Subclass and implement `act(obs, deadline) -> Action`. The `deadline`
    argument is supplied by the wrapper; call `deadline.stage(...)` as soon as
    you have a valid action, then improve it until `deadline.should_stop()`.
    """

    name: str = "base"

    def act(self, obs, deadline: Deadline) -> Action:  # pragma: no cover — abstract
        raise NotImplementedError

    # ---- Kaggle submission wrapper ----
    def as_kaggle_agent(self) -> Callable:
        """Return a plain callable usable as a Kaggle submission.

        The callable honors the hard deadline: if `act` runs long we return
        the staged best-so-far; if it raises, we return no_op.
        """

        def kaggle_agent(obs, cfg=None):
            dl = Deadline()
            try:
                result = self.act(obs, dl)
                if dl.elapsed_ms() > HARD_DEADLINE_MS:
                    return dl.best()
                return result if isinstance(result, list) else dl.best()
            except Exception:
                return dl.best()
            finally:
                # One explicit collection between turns, cheap and keeps us
                # off the critical path next turn.
                gc.collect()

        kaggle_agent.__name__ = self.name
        return kaggle_agent


# ---- Built-in trivial agents for baselines and pipeline testing ----

class NoOpAgent(Agent):
    """Does nothing. Used for pipeline validation (dry-run submission)."""

    name = "no_op"

    def act(self, obs, deadline: Deadline) -> Action:
        deadline.stage(no_op())
        return no_op()


class RandomAgent(Agent):
    """Random valid launches. Used as a weak baseline."""

    name = "random"

    def __init__(self, seed: int | None = None):
        import random as _r
        self._r = _r.Random(seed)

    def act(self, obs, deadline: Deadline) -> Action:
        deadline.stage(no_op())
        player = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0)
        planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
        moves: Action = []
        for p in planets:
            if p[1] == player and p[5] > 0:
                angle = self._r.uniform(0, 2 * math.pi)
                ships = p[5] // 2
                if ships >= 20:
                    moves.append([p[0], angle, ships])
        deadline.stage(moves)
        return moves


def obs_get(obs, key, default=None):
    """Observation accessor that works for both dict and object-style obs.

    Kaggle hands agents a dict-like obs; kaggle_environments's internal
    state uses a SimpleNamespace. This indirection lets us write one code path.
    """
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)



# --- inlined: orbitwars/bots/heuristic.py ---

"""Heuristic bot (Path A).

The "floor" bot: a fast, parameterized heuristic that ranks candidate targets
per owned planet using a linear mix of features and launches exact-plus-one
attacks when a net-positive capture is predicted.

Key ideas (drawn from the Kore 2022 winner's playbook):

  * Fleet-arrival table: for each target planet, we tabulate net incoming
    allied vs enemy ships by arrival time. Scoring factors in both the
    earliest capture window and the steady-state production stream.

  * Exact-plus-one sizing: ship count sent equals projected defender ships at
    arrival time + 1. Under-send is wasted; over-send is merely inefficient.

  * Intercept math: orbital targets are predicted via the Newton-iteration
    solver in engine/intercept.py; comets via the path-indexed solver.

  * Sun-avoidance: direct lines that cross the sun are rerouted to the closest
    tangent angle.

  * Parameterization: HEURISTIC_WEIGHTS is a flat dict of 20-ish floats. It
    feeds TuRBO tuning (Path A) and LLM-evolved mutations (EvoTune). Adding a
    new feature means adding one key here and one term in `_score_target`.

This file is intentionally simple and close to the metal. Do not add clever
caching or search here — that's Path B's job.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple



# ---- Parameter config (tuned by TuRBO / EvoTune) ----

HEURISTIC_WEIGHTS: Dict[str, float] = {
    # Target scoring (higher = stronger preference)
    "w_production": 5.0,          # value per unit production
    "w_ships_cost": 0.02,         # per-ship cost in denominator
    "w_distance_cost": 0.05,      # per-unit Euclidean distance cost
    "w_travel_cost": 0.3,         # per-turn travel cost

    # Target preference multipliers
    "mult_neutral": 1.0,
    "mult_enemy": 1.8,            # bias toward offense over neutrals once in contact
    "mult_comet": 1.5,
    "mult_reinforce_ally": 0.0,   # disabled at v1 (we don't reinforce)

    # Sizing
    "ships_safety_margin": 1.0,   # extra ships beyond exact-plus-one
    "min_launch_size": 20.0,      # don't send fewer than this (matches starter)
    "max_launch_fraction": 0.8,   # leave 20% behind (tuned in W2 via TuRBO)

    # Expansion pacing
    "expand_cooldown_turns": 0.0, # allow every turn
    "keep_reserve_ships": 0.0,    # no forced reserve (exact-plus-one handles this)
    "agg_early_game": 1.0,
    "early_game_cutoff_turn": 100.0,

    # Sun handling
    "sun_avoidance_epsilon": 0.02,

    # Comet engagement
    "comet_max_time_mismatch": 1.5,

    # Search bias (used when MCTS wraps this — harmless here)
    "expand_bias": 0.5,
}


# ---- Observation shape helper ----

@dataclass
class ParsedObs:
    """Typed unpacking of the Kaggle obs for a single agent."""
    player: int
    step: int
    angular_velocity: float
    planets: List[List[Any]]
    initial_planets: List[List[Any]]
    fleets: List[List[Any]]
    next_fleet_id: int
    comet_planet_ids: set
    # Derived
    my_planets: List[List[Any]] = field(default_factory=list)
    enemy_planets: List[List[Any]] = field(default_factory=list)
    neutral_planets: List[List[Any]] = field(default_factory=list)
    planet_by_id: Dict[int, List[Any]] = field(default_factory=dict)
    initial_planet_by_id: Dict[int, List[Any]] = field(default_factory=dict)
    # Comet bookkeeping (per-comet precomputed path + current path index,
    # keyed by the comet's planet pid). Populated from ``obs.comets`` in
    # ``parse_obs``. Used by intercept math and the trajectory-obstruction
    # walk to treat comets as path-indexed moving targets rather than
    # falling through to the static-target branch.
    comet_path_by_pid: Dict[int, List[Tuple[float, float]]] = field(default_factory=dict)
    comet_path_index_by_pid: Dict[int, int] = field(default_factory=dict)


_COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)


def _infer_step_from_obs(obs) -> int:
    """Best-effort step inference when ``obs['step']`` is absent.

    Kaggle's orbit_wars engine only populates ``obs.step`` for seat 0
    (player 0). For seat 1 we must infer. Strategies, in order:

    1. **Comet path_index**. Comets spawn at fixed steps
       ``[50,150,250,350,450]``; each group's ``path_index`` directly
       encodes turns since spawn. ``obs.comets`` is append-ordered by
       spawn, so ``comets[i].path_index + COMET_SPAWN_STEPS[i]`` is the
       current step. Works from step 50 onwards.

    2. **Orbital phase**. For any orbiter visible in both
       ``planets`` and ``initial_planets``, ``step ≈ (current_angle
       − initial_angle) / ω``, modular at the orbital period. Unique
       only within the first orbital period (~125-250 turns); beyond
       that needs disambiguation we don't do here. Good enough early-
       game.

    3. **Zero**. Safe default for the fresh-state case.

    Returns 0 if no source agrees. Callers needing monotonicity across
    calls (e.g. for cooldowns) should override via an agent-level
    turn counter.
    """
    # (1) Comet-based: path_index from the first group with idx >= 0.
    comets = obs_get(obs, "comets", None) or []
    for i, g in enumerate(comets):
        if not isinstance(g, dict) or i >= len(_COMET_SPAWN_STEPS):
            continue
        idx = int(g.get("path_index", -1))
        if idx >= 0:
            return _COMET_SPAWN_STEPS[i] + idx

    # (2) Orbital-phase: find any orbiter, compute step from angle delta.
    ω = float(obs_get(obs, "angular_velocity", 0.0))
    if ω > 0.0:
        initial = {int(p[0]): p for p in obs_get(obs, "initial_planets", []) or []}
        comet_pids = set(obs_get(obs, "comet_planet_ids", []) or [])
        for pl in obs_get(obs, "planets", []) or []:
            pid = int(pl[0])
            if pid in comet_pids or pid not in initial:
                continue
            ip = initial[pid]
            # Skip non-orbiters (r + radius >= ROTATION_RADIUS_LIMIT=50).
            dx = float(ip[2]) - 50.0
            dy = float(ip[3]) - 50.0
            r = math.hypot(dx, dy)
            if r + float(ip[4]) >= 50.0:
                continue
            initial_angle = math.atan2(dy, dx)
            cdx = float(pl[2]) - 50.0
            cdy = float(pl[3]) - 50.0
            current_angle = math.atan2(cdy, cdx)
            delta = (current_angle - initial_angle) % (2.0 * math.pi)
            step = int(round(delta / ω))
            # Angle-wrap ambiguity: step modulo period. Early game
            # (step < period) this is exact. Late game it wraps; we
            # accept the modular answer here — agents that need
            # monotonic turn tracking must provide it externally.
            return step

    return 0


def parse_obs(obs, step_override: Optional[int] = None) -> ParsedObs:
    raw_step = obs_get(obs, "step", None)
    if step_override is not None:
        step = int(step_override)
    elif raw_step is not None:
        step = int(raw_step)
    else:
        step = _infer_step_from_obs(obs)
    p = ParsedObs(
        player=obs_get(obs, "player", 0),
        step=step,
        angular_velocity=obs_get(obs, "angular_velocity", 0.0),
        planets=list(obs_get(obs, "planets", [])),
        initial_planets=list(obs_get(obs, "initial_planets", [])),
        fleets=list(obs_get(obs, "fleets", [])),
        next_fleet_id=obs_get(obs, "next_fleet_id", 0),
        comet_planet_ids=set(obs_get(obs, "comet_planet_ids", [])),
    )
    for pl in p.planets:
        p.planet_by_id[pl[0]] = pl
        if pl[1] == p.player:
            p.my_planets.append(pl)
        elif pl[1] == -1:
            p.neutral_planets.append(pl)
        else:
            p.enemy_planets.append(pl)
    for ip in p.initial_planets:
        p.initial_planet_by_id[ip[0]] = ip

    # Parse obs.comets. Engine schema:
    #   obs.comets: list of groups, each a dict with keys
    #     "planet_ids": [pid, ...]    — comet-planet ids in this group
    #     "paths":      [[[x,y], ...], ...]   — one path per pid (same order)
    #     "path_index": int           — current index shared across the group
    # The comet's current visible position is path[path_index]; the engine
    # increments path_index once per turn in its step-5 comet-move phase.
    for group in obs_get(obs, "comets", []) or []:
        pids = group.get("planet_ids", []) if isinstance(group, dict) else []
        paths = group.get("paths", []) if isinstance(group, dict) else []
        idx = int(group.get("path_index", -1)) if isinstance(group, dict) else -1
        for i, pid in enumerate(pids):
            if i < len(paths):
                path = [(float(pt[0]), float(pt[1])) for pt in paths[i]]
                p.comet_path_by_pid[int(pid)] = path
                p.comet_path_index_by_pid[int(pid)] = idx
    return p


# ---- Fleet-arrival table ----

@dataclass
class ArrivalEvent:
    turn: int
    owner: int
    ships: int


@dataclass
class ArrivalTable:
    """Per-target net ship projections, indexed by arrival turn.

    Used for:
      - Deciding if a reinforce is needed (net-incoming flips negative).
      - Exact-plus-one sizing under concurrent incoming fleets.
      - Blocking attacks on planets already being attacked by a teammate
        (we pass through for now — 2p games only have us).
    """
    events_by_pid: Dict[int, List[ArrivalEvent]] = field(default_factory=dict)

    def add(self, pid: int, turn: int, owner: int, ships: int) -> None:
        self.events_by_pid.setdefault(pid, []).append(ArrivalEvent(turn, owner, ships))

    def events(self, pid: int) -> List[ArrivalEvent]:
        return self.events_by_pid.get(pid, [])

    def projected_defender_at(
        self,
        pid: int,
        defender_owner: int,
        current_ships: int,
        production: int,
        arrival_turn: int,
    ) -> int:
        """Project defender ship count at `arrival_turn`, assuming:
          - Production continues at the given rate (only while owned).
          - Incoming ships flip ownership / decrement per combat rules.

        This is a simplified single-owner projection. For multi-front fights
        the full simulator in fast_engine.py is the ground truth — we use
        this estimate for fast scoring only.
        """
        owner = defender_owner
        ships = current_ships
        events = sorted(self.events(pid), key=lambda e: e.turn)
        last_t = 0
        for e in events:
            if e.turn > arrival_turn:
                break
            # Production between last_t and e.turn (only while owned)
            if owner != -1:
                ships += production * max(0, e.turn - last_t)
            if e.owner == owner:
                ships += e.ships
            else:
                ships -= e.ships
                if ships < 0:
                    owner = e.owner
                    ships = -ships
            last_t = e.turn
        # Production until arrival_turn
        if owner != -1:
            ships += production * max(0, arrival_turn - last_t)
        return ships


def build_arrival_table(
    po: ParsedObs, deadline: Optional[Deadline] = None,
) -> ArrivalTable:
    """Populate arrival events for every in-flight fleet against its target.

    We estimate the target by: the closest planet along the fleet's heading.
    That's imperfect (fleets can target any point in space or a planet that's
    rotated by arrival time), but it's good enough for a first-cut defense
    signal. The MCTS wrapper will replace this with a more precise estimate.

    ``deadline`` (optional) is checked between fleet iterations. This loop
    is O(fleets \u00d7 planets) with a Newton-intercept solve per pair \u2014 on
    dense late-game states (40+ fleets, 40+ planets) it runs 100-300 ms.
    Without a mid-loop check, an in-flight rollout can blow past the outer
    search deadline by the full duration of this function. When the
    deadline fires, we return the partial table accumulated so far; that
    is still a valid input to downstream scoring (just undercounts arrivals
    for the unvisited fleets).
    """
    table = ArrivalTable()
    for f in po.fleets:
        if deadline is not None and deadline.should_stop():
            break
        fid, owner, fx, fy, fangle, from_pid, fships = f
        # Best guess of target: planet whose perpendicular distance from fleet
        # ray is minimal and the planet is ahead of the fleet.
        best_pid = -1
        best_score = float("inf")
        best_turns = 0
        for pl in po.planets:
            pid = pl[0]
            if pid == from_pid:
                continue
            is_orb = is_orbiting_planet(pl, po.initial_planet_by_id.get(pid, pl))
            if is_orb:
                ir, ia = initial_orbit_params(po.initial_planet_by_id[pid])
                # NOTE: current_step = po.step - 2 matches _travel_turns'
                # engine-phase convention. A fleet observed at obs.step=S
                # has its k-th subsequent collision checked against planet
                # at angle init+ω*(S+k-2); Newton picks that up via
                # position_at(τ) = orbit(step=S-2+τ).
                ot = OrbitingTarget(
                    orbital_radius=ir, initial_angle=ia,
                    angular_velocity=po.angular_velocity,
                    current_step=po.step - 2,
                )
                # Quick check: if we aim at this orbital target, what's the
                # angular difference from the fleet's current heading?
                angle_to, t, _ = orbiting_intercept((fx, fy), ot, fships)
            else:
                angle_to = static_intercept_angle((fx, fy), (pl[2], pl[3]))
                t = static_intercept_turns((fx, fy), (pl[2], pl[3]), fships)
            # Circular distance between angles, in (0, pi]
            da = abs(((angle_to - fangle + math.pi) % (2 * math.pi)) - math.pi)
            # Score: prefer small angle difference, prefer closer.
            score = da * 10.0 + t * 0.1
            if score < best_score:
                best_score = score
                best_pid = pid
                best_turns = t
        if best_pid >= 0:
            table.add(best_pid, int(math.ceil(best_turns)) + po.step, owner, fships)
    return table


# ---- Target scoring ----

def _travel_turns(source: Tuple[float, float], target_pl: List[Any],
                  initial_pl: List[Any], angular_velocity: float,
                  step: int, ships: int,
                  source_radius: float = 0.0,
                  po: Optional[ParsedObs] = None) -> Tuple[float, float]:
    """Return (angle_to_aim, travel_turns_prediction).

    ``source_radius`` is the radius of the source planet. When > 0, the
    Newton is told the fleet actually launches at ``source + (r + 0.1) *
    dir`` — matching the engine's ``process_moves`` offset — so the
    predicted arrival time is correct to ~0.05 turns instead of being
    overestimated by ``(r+0.1)/v`` (up to ~2 turns on small fleets, which
    is exactly long enough for the orbital target to rotate out of the
    aim point and cause a miss).

    Orbit phase offset: at ``obs.step = S`` the observed planet angle is
    ``init + ω*(S-1)`` (verified empirically), and on the k-th fleet-turn
    after launch the engine collision check uses the planet at angle
    ``init + ω*(S+k-2)`` (pre-rotation for that step). Constructing
    ``OrbitingTarget`` with ``current_step = step - 2`` makes the Newton's
    ``position_at(τ) = orbit(step=S-2+τ)`` match the engine's collision
    target at fleet-turn τ.

    Comet branch: comets fail ``is_orbiting_planet`` (their orbital
    radius + radius >= ROTATION_RADIUS_LIMIT by construction) so the
    previous static-fallback aimed at the comet's current position —
    which is where the comet is now, not where it will be at arrival.
    When ``po`` is supplied and the target pid is a comet we use the
    path-indexed ``comet_intercept`` solver instead.
    """
    source_offset = source_radius + 0.1 if source_radius > 0.0 else 0.0
    tpid = int(target_pl[0])
    # Comet branch: target is on a precomputed path; intercept the path,
    # not the current position.
    if po is not None and tpid in po.comet_planet_ids:
        path = po.comet_path_by_pid.get(tpid)
        idx_now = po.comet_path_index_by_pid.get(tpid, -1)
        if path and idx_now >= 0:
            result = comet_intercept(
                source=source, comet_path=path, path_index_now=idx_now,
                ships=ships, source_offset=source_offset,
            )
            if result is None:
                return (0.0, float("inf"))
            angle, t, _ = result
            return (angle, t)
        # Fallthrough to static-aim if comet metadata is missing
        # (shouldn't happen with a well-formed obs).
    if is_orbiting_planet(target_pl, initial_pl):
        ir, ia = initial_orbit_params(initial_pl)
        ot = OrbitingTarget(
            orbital_radius=ir, initial_angle=ia,
            angular_velocity=angular_velocity, current_step=step - 2,
        )
        angle, t, _ = orbiting_intercept(
            source, ot, ships, source_offset=source_offset,
        )
        return angle, t
    else:
        tx, ty = target_pl[2], target_pl[3]
        angle = static_intercept_angle(source, (tx, ty))
        t = static_intercept_turns(
            source, (tx, ty), ships, source_offset=source_offset,
        )
        return angle, t


def _intercept_position(
    source: Tuple[float, float],
    target_pl: List[Any],
    initial_pl: List[Any],
    angular_velocity: float,
    step: int,
    travel_turns: float,
    po: Optional[ParsedObs] = None,
) -> Tuple[float, float]:
    """Where the target will be at fleet arrival time. Match the same
    engine-phase convention as ``_travel_turns``: collision at fleet-turn
    τ uses planet at angle ``init + ω*(step-2+τ)``.

    For static targets this is just the current observed position.
    For comet targets (when ``po`` supplies path info), we return the
    path position at ``path_index_now + ceil(travel_turns) - 1`` — the
    engine's step-3 collision index at fleet-turn k = ceil(travel_turns).
    """
    tpid = int(target_pl[0])
    if po is not None and tpid in po.comet_planet_ids:
        path = po.comet_path_by_pid.get(tpid)
        idx_now = po.comet_path_index_by_pid.get(tpid, -1)
        if path and idx_now >= 0:
            k = max(1, int(math.ceil(travel_turns)))
            aim_idx = min(idx_now + k - 1, len(path) - 1)
            return (float(path[aim_idx][0]), float(path[aim_idx][1]))
    if is_orbiting_planet(target_pl, initial_pl):
        ir, ia = initial_orbit_params(initial_pl)
        ot = OrbitingTarget(
            orbital_radius=ir, initial_angle=ia,
            angular_velocity=angular_velocity, current_step=step - 2,
        )
        return ot.position_at(travel_turns)
    return (float(target_pl[2]), float(target_pl[3]))


def _score_target(
    mp: List[Any],
    tp: List[Any],
    ip: List[Any],
    po: ParsedObs,
    table: ArrivalTable,
    weights: Dict[str, float],
    ships_to_send: int,
) -> Tuple[float, float, int]:
    """Return (score, aim_angle, defender_projection).

    Higher score = more attractive to launch this attack.
    """
    source_center = (float(mp[2]), float(mp[3]))
    source_radius = float(mp[4])
    angle, turns = _travel_turns(
        source_center, tp, ip,
        po.angular_velocity, po.step, ships_to_send,
        source_radius=source_radius, po=po,
    )
    if turns <= 0 or math.isinf(turns):
        return (-math.inf, 0.0, 0)

    # Avoid sun if needed — use the predicted intercept point (where the
    # target WILL BE at arrival), not the current planet position. For
    # orbiting planets and comets the two can differ substantially (tens
    # of units over a multi-turn flight); mis-routing from the wrong
    # reference point has caused direct sun-kills mid-flight in practice.
    target_pos = _intercept_position(
        source_center, tp, ip, po.angular_velocity, po.step, turns, po=po,
    )
    angle = route_angle_avoiding_sun(source_center, angle, target_pos)

    defender_ships = tp[5]
    defender_owner = tp[1]
    production = tp[6]
    arrival_turn = po.step + int(math.ceil(turns))
    projected = table.projected_defender_at(
        tp[0], defender_owner, defender_ships, production, arrival_turn,
    )

    # Preference multiplier
    if tp[0] in po.comet_planet_ids:
        mult = weights["mult_comet"]
    elif tp[1] == po.player:
        mult = weights["mult_reinforce_ally"]
    elif tp[1] == -1:
        mult = weights["mult_neutral"]
    else:
        mult = weights["mult_enemy"]

    # Core score: production value / (ships cost + travel cost)
    ships_cost = weights["w_ships_cost"] * max(1, ships_to_send)
    travel_cost = weights["w_travel_cost"] * turns
    distance = math.hypot(tp[2] - mp[2], tp[3] - mp[3])
    distance_cost = weights["w_distance_cost"] * distance
    production_value = weights["w_production"] * production

    # Early game aggression multiplier
    if po.step < weights["early_game_cutoff_turn"]:
        mult *= weights["agg_early_game"]

    denom = ships_cost + travel_cost + distance_cost + 1e-6
    score = mult * production_value / denom

    # If we can't actually capture (insufficient ships), penalize heavily.
    needed = projected + int(weights["ships_safety_margin"])
    if ships_to_send < needed and defender_owner != po.player:
        score -= 10.0  # can't capture

    return (score, angle, projected)


# ---- Trajectory obstruction check ----

# Sentinel codes returned by `_trajectory_obstruction`:
#   -1: path is clear — fleet reaches the intended target
#   -2: would cross the sun before hitting any planet
#   -3: would leave the board before hitting any planet
#   -4: walk budget exhausted without hitting anything (treat as waste)
# Any value >= 0 is the pid of an intervening planet the fleet would hit
# *before* reaching the intended target.

_OBSTR_CLEAR = -1
_OBSTR_SUN = -2
_OBSTR_OOB = -3
_OBSTR_WASTED = -4

# Maximum turns to simulate during the obstruction walk. A fleet at
# speed-cap 6 crosses the 100×100 board in ~17 turns; a slow v≈1 fleet
# in ~100 turns. 60 is a compromise: 95% of real intercepts arrive in
# under 30 turns and we reject the long-tail "pointed at nothing" shots
# rather than pay the budget.
_OBSTR_MAX_TURNS = 60


def _trajectory_obstruction(
    source_center: Tuple[float, float],
    source_radius: float,
    angle: float,
    ships: int,
    target_pid: int,
    po: ParsedObs,
) -> int:
    """Simulate the fleet's future trajectory until *something happens*
    and return a code describing what that was.

    CLEAR means the fleet's first collision is with the intended target —
    i.e. the engine's deterministic simulation will hit `target_pid`.
    Any other return value means the launch is a miss: a different
    planet eats the fleet first, the sun destroys it, it flies off the
    board, or it never hits anything within the walk budget.

    Why walk until something happens (instead of stopping at the
    predicted arrival_turn): if the intercept-solved angle is off by a
    fraction of a radian the fleet misses the target and keeps flying
    in a straight line at constant velocity until it dies somewhere.
    That post-target flight is exactly where "wrong planet" collisions
    come from — 223 out of 876 baseline shots in the verifier. Walking
    only to predicted-arrival hides those collisions.

    Engine step order inside a single engine turn S+k-1 (fleet-turn k
    of a fleet launched at step S):
      (a) Fleet movement — segment [pre, post] checked against every
          planet/comet at its pre-this-turn position. First hit in
          planet iteration order eats the fleet.
      (b) Planet rotation — each orbiting planet sweeps the arc
          (pre_rot_pos → post_rot_pos) and the post-fleet-move point is
          point-checked against each planet's swept segment.
      (c) Comet movement — each comet sweeps (path[idx+k-1] → path[idx+k])
          and the post-fleet-move point is checked the same way.

    The walk mirrors all three. Comet positioning is path-indexed, not
    the static ``planet[2],planet[3]`` coords (which for comets is their
    current location at obs.step=S but doesn't tell the walk how the
    comet moves between turns — hence the old walk always missed
    moving-comet collisions).

    Cost: O(walk_turns × n_planets). On dense states with ~30 planets
    and a 20-turn flight, ~600 point-to-segment evals per walk ≈
    150-300 μs. Top-K=5 fallback retries cap the per-my-planet
    obstruction cost at ~1-2 ms, comfortably within budget.
    """
    v = fleet_speed(ships)
    if v <= 0.0:
        return _OBSTR_WASTED

    dirx = math.cos(angle)
    diry = math.sin(angle)
    sx, sy = source_center
    offset = source_radius + 0.1
    # Fleet start position (engine-exact launch point).
    lx = sx + offset * dirx
    ly = sy + offset * diry

    ω = po.angular_velocity
    step = po.step

    # Precompute per-planet metadata so the hot loop does one sin/cos
    # pair per orbiter per turn instead of redoing init-params math.
    # Tuple layout:
    #   (pid, kind, a, b, c, d, radius)
    # where kind is 0=static, 1=orbiter, 2=comet; and (a,b,c,d) is kind
    # specific:
    #   static:  (static_x, static_y, unused, unused)
    #   orbiter: (orbital_radius, initial_angle, unused, unused)
    #   comet:   (comet_pid_as_index_into_po.comet_path_by_pid, 0, 0, 0)
    # Comets store nothing static — path lookup each turn via po dicts.
    _KIND_STATIC = 0
    _KIND_ORBITER = 1
    _KIND_COMET = 2
    planet_meta: List[Tuple[int, int, float, float, float, float, float]] = []
    for pl in po.planets:
        pid = int(pl[0])
        prad = float(pl[4])
        if pid in po.comet_planet_ids:
            planet_meta.append((pid, _KIND_COMET, 0.0, 0.0, 0.0, 0.0, prad))
            continue
        ip = po.initial_planet_by_id.get(pid, pl)
        if is_orbiting_planet(pl, ip):
            ir_o, ia_o = initial_orbit_params(ip)
            planet_meta.append((pid, _KIND_ORBITER, ir_o, ia_o, 0.0, 0.0, prad))
        else:
            planet_meta.append(
                (pid, _KIND_STATIC, float(pl[2]), float(pl[3]), 0.0, 0.0, prad),
            )

    for k in range(1, _OBSTR_MAX_TURNS + 1):
        # Fleet segment during fleet-turn k: [pre, post].
        pre_x = lx + (k - 1) * v * dirx
        pre_y = ly + (k - 1) * v * diry
        post_x = lx + k * v * dirx
        post_y = ly + k * v * diry
        pre = (pre_x, pre_y)
        post = (post_x, post_y)

        # Out-of-bounds check (engine removes fleets that step off-board).
        if not (0.0 <= post_x <= BOARD_SIZE and 0.0 <= post_y <= BOARD_SIZE):
            return _OBSTR_OOB

        # Sun crossing: segment-to-center distance < SUN_RADIUS.
        if point_to_segment_distance((CENTER, CENTER), pre, post) < SUN_RADIUS:
            return _OBSTR_SUN

        pre_rot_step = step + k - 2
        post_rot_step = step + k - 1

        # (a) Fleet-movement collision vs each planet at its pre-this-turn
        # position. First hit in iteration order wins (mirrors engine's
        # break-on-first-hit loop in the fleet-move phase).
        for (pid, kind, a, b, _c, _d, prad) in planet_meta:
            if kind == _KIND_ORBITER:
                theta = b + ω * pre_rot_step
                px = CENTER + a * math.cos(theta)
                py = CENTER + a * math.sin(theta)
            elif kind == _KIND_COMET:
                path = po.comet_path_by_pid.get(pid)
                idx_now = po.comet_path_index_by_pid.get(pid, -1)
                if not path or idx_now < 0:
                    continue
                # Pre-this-turn comet position = path[idx_now + k - 1].
                # Past end-of-path = comet has expired; skip.
                pre_idx = idx_now + k - 1
                if pre_idx >= len(path) or pre_idx < 0:
                    continue
                px = path[pre_idx][0]
                py = path[pre_idx][1]
            else:  # static
                px = a
                py = b
            if point_to_segment_distance((px, py), pre, post) < prad:
                return _OBSTR_CLEAR if pid == target_pid else pid

        # (b) Orbital-planet rotation sweep. Each orbiter moves from its
        # pre-rot position to its post-rot position; the fleet's post-
        # fleet-move point is checked against that arc (segment approx).
        # First hit destroys the fleet.
        for (pid, kind, a, b, _c, _d, prad) in planet_meta:
            if kind != _KIND_ORBITER:
                continue
            theta_pre = b + ω * pre_rot_step
            theta_post = b + ω * post_rot_step
            pre_px = CENTER + a * math.cos(theta_pre)
            pre_py = CENTER + a * math.sin(theta_pre)
            post_px = CENTER + a * math.cos(theta_post)
            post_py = CENTER + a * math.sin(theta_post)
            if point_to_segment_distance(
                (post_x, post_y), (pre_px, pre_py), (post_px, post_py),
            ) < prad:
                return _OBSTR_CLEAR if pid == target_pid else pid

        # (c) Comet movement sweep. Each comet moves from path[idx+k-1]
        # to path[idx+k] (step-5 engine phase). Fleet's post-fleet-move
        # point is checked against that segment.
        for (pid, kind, _a, _b, _c, _d, prad) in planet_meta:
            if kind != _KIND_COMET:
                continue
            path = po.comet_path_by_pid.get(pid)
            idx_now = po.comet_path_index_by_pid.get(pid, -1)
            if not path or idx_now < 0:
                continue
            pre_idx = idx_now + k - 1
            post_idx = idx_now + k
            # Past end-of-path = comet has expired; no more collisions.
            if post_idx >= len(path) or pre_idx < 0 or pre_idx >= len(path):
                continue
            pre_p = path[pre_idx]
            post_p = path[post_idx]
            if point_to_segment_distance(
                (post_x, post_y), (pre_p[0], pre_p[1]), (post_p[0], post_p[1]),
            ) < prad:
                return _OBSTR_CLEAR if pid == target_pid else pid

    # Walked the full budget without hitting anything. Fleet is wasted.
    return _OBSTR_WASTED


# ---- Main agent ----

@dataclass
class LaunchIntent:
    """One planner-emitted launch, with the target the planner *intended* to
    hit and the predicted arrival turn. Written to the agent side-channel
    ``HeuristicAgent.last_launch_intents`` so verifier tools (and future
    miss-logging telemetry) can compare emitted vs. actual without having
    to reverse-engineer target attribution from angle matching.
    """
    turn: int          # po.step at emission
    from_pid: int
    target_pid: int
    angle: float
    ships: int
    predicted_travel_turns: float
    predicted_arrival_turn: int
    score: float


@dataclass
class _LaunchState:
    last_launch_turn: Dict[int, int] = field(default_factory=dict)


class HeuristicAgent(Agent):
    """Path A bot. Parameterized, fast, tournament baseline."""

    name = "heuristic"

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = dict(HEURISTIC_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self._state = _LaunchState()
        # Side-channel: populated by _plan_moves, read by diag tools +
        # the pytest zero-miss gate. One entry per launch emitted this
        # turn, in the same order as the wire `moves` list so
        # `fleet_id_n = next_fleet_id + n` maps 1:1.
        self.last_launch_intents: List[LaunchIntent] = []
        # Full per-game launch history (append-only). Each LaunchIntent
        # has the turn it was emitted plus the predicted target/arrival.
        # Negligible cost during play (~1 list append per launch, a few
        # hundred entries per game). External tooling pairs entries with
        # engine-captured combat_lists to produce the hit/miss report.
        self.launch_log: List[LaunchIntent] = []
        # Monotonic turn counter. Required for seat-1 play: Kaggle's
        # orbit_wars engine omits obs.step for player 1, so parse_obs's
        # inference-from-state is only unique within the first orbital
        # period (~125-250 turns). This counter supplies the unambiguous
        # answer across a full 500-turn game. Reset on game-start
        # detection in act().
        self._turn_counter: Optional[int] = None

    # ---- Public: Kaggle + Agent contract ----

    def act(self, obs, deadline: Deadline) -> Action:
        # Always stage the no-op first so we're safe against timeouts.
        deadline.stage(no_op())

        # Resolve step and detect match-start.
        # Seat 0: obs.step is authoritative. step==0 -> new match.
        # Seat 1: obs.step is None (Kaggle engine quirk). We rely on a
        # monotonic counter, reset when next_fleet_id regresses (or on
        # very first call) — which is the strongest always-available
        # match-start signal.
        raw_step = obs_get(obs, "step", None)
        curr_nfid = int(obs_get(obs, "next_fleet_id", 0))
        if raw_step is not None:
            step = int(raw_step)
            is_match_start = (step == 0)
            self._turn_counter = step
        else:
            prev_nfid = getattr(self, "_prev_next_fleet_id", None)
            first_call = self._turn_counter is None
            # next_fleet_id is monotonically non-decreasing within a
            # match. A drop to 0 (or a first call with nfid==0) is the
            # match-start edge. Using nfid rather than "fleets list
            # empty" is robust to arbitrary early-game turns where no
            # one has launched yet.
            is_match_start = first_call or (
                prev_nfid is not None and prev_nfid > curr_nfid
            )
            if is_match_start:
                step = 1
            else:
                step = (self._turn_counter or 0) + 1
            self._turn_counter = step
        self._prev_next_fleet_id = curr_nfid

        po = parse_obs(obs, step_override=step)

        # Reset per-match state only on true match-start.
        if is_match_start:
            self._state = _LaunchState()
            # New game -> fresh launch log; keeps post-mortem telemetry
            # scoped to the current match.
            self.launch_log = []

        if not po.my_planets:
            return no_op()

        # Outer-deadline check between stages: build_arrival_table scales
        # with fleet count × planet count (intercept math per pair) and
        # on dense late-game states runs 50-200 ms. When the caller
        # (e.g. MCTS rollouts) supplies a Deadline with an absolute
        # hard_stop_at, short-circuit before paying that cost. Returns
        # the no-op staged above so the call is still action-valid.
        if deadline.should_stop():
            return no_op()

        # Thread deadline into build_arrival_table \u2014 on dense states its
        # O(fleets \u00d7 planets) intercept loop dominates (100-300 ms) and
        # must be abortable mid-way or a single in-flight rollout blows
        # past the outer search deadline.
        table = build_arrival_table(po, deadline=deadline)

        if deadline.should_stop():
            return no_op()

        moves: Action = self._plan_moves(po, table, deadline=deadline)

        # Cooldown bookkeeping
        for m in moves:
            self._state.last_launch_turn[int(m[0])] = po.step

        deadline.stage(moves)
        return moves

    # ---- Planning ----

    def _plan_moves(
        self, po: ParsedObs, table: ArrivalTable,
        deadline: Optional[Deadline] = None,
    ) -> Action:
        moves: List[Move] = []
        # Reset the per-turn launch-intent log. Verifier/telemetry reads
        # it straight after act() returns.
        self.last_launch_intents = []
        w = self.weights
        reserve = int(w["keep_reserve_ships"])
        cd = int(w["expand_cooldown_turns"])

        # Build the list of candidate targets once (excludes our own planets
        # for attack; includes them for reinforce consideration).
        candidates = [p for p in po.planets]

        for mp in po.my_planets:
            # Per-my-planet deadline check. The inner loop runs intercept
            # math for every (my_planet, target) pair — 30-100 μs per pair
            # × 40 planets = ~2 ms per outer-iteration. On dense late-game
            # states with 20 my-planets we can still accumulate ~40 ms in
            # the outer loop. Breaking mid-way returns the partial move
            # list built so far (still a valid Action), which is strictly
            # better than overrunning the outer MCTS search deadline.
            if deadline is not None and deadline.should_stop():
                break
            mpid = int(mp[0])
            available = int(mp[5]) - reserve
            if available < int(w["min_launch_size"]):
                continue

            # Defense guard: if enemy ships are inbound and our defenders can't
            # hold, don't drain this planet for offense. Compute the net
            # shortfall and reduce `available` by exactly that much.
            incoming_enemy = 0
            incoming_ally = 0
            for ev in table.events(mpid):
                if ev.owner == po.player:
                    incoming_ally += ev.ships
                else:
                    incoming_enemy += ev.ships
            # Assume production keeps coming while we hold; shortfall estimate
            # is a cheap approximation (production time-integral depends on
            # arrival ordering — handled precisely by fast_engine when MCTS
            # wraps this).
            projected_def = int(mp[5]) + incoming_ally
            shortfall = max(0, incoming_enemy + 1 - projected_def)
            if shortfall > 0:
                # Keep exactly enough to defend; no extra hoarding.
                available = max(0, int(mp[5]) - shortfall)
            if available < int(w["min_launch_size"]):
                continue

            # Cooldown (skip check entirely when cd == 0 — avoids any chance
            # of stale last_launch_turn values blocking launches)
            if cd > 0:
                last = self._state.last_launch_turn.get(mpid, -10_000)
                if po.step - last < cd:
                    continue

            # Score all candidate targets at several possible send sizes.
            # We keep the full ranked list so that when the top-scoring
            # target's trajectory is obstructed (passes through another
            # planet, grazes the sun, leaves the board) we can fall
            # through to the next-best target instead of launching into
            # nothing. Top-K=5 keeps the obstruction-walk cost bounded
            # (each walk is ~50-150 μs, so 5 × 20 my-planets ≈ 15 ms
            # worst case — comfortably under the outer budget).
            ranked: List[Tuple[float, float, int, int, Any]] = []
            for tp in candidates:
                # Inner-loop deadline check. candidates = all planets, so
                # this loop is O(len(planets)) per my-planet with one
                # Newton-intercept solve per iteration via _travel_turns.
                # On dense states it can accumulate 5-15 ms per planet;
                # across 20 my-planets the full outer loop is 100-300 ms.
                # Without this check, an in-flight HeuristicAgent.act call
                # from a rollout can keep running past the search deadline.
                if deadline is not None and deadline.should_stop():
                    break
                if tp[0] == mpid:
                    continue
                ip = po.initial_planet_by_id.get(tp[0], tp)

                # Trial size = exact-plus-one (projected + safety margin).
                # First a cheap estimate of travel turns with a guess ship size:
                _, t_guess = _travel_turns(
                    (mp[2], mp[3]), tp, ip,
                    po.angular_velocity, po.step, max(50, available // 2),
                    source_radius=float(mp[4]), po=po,
                )
                if math.isinf(t_guess) or t_guess <= 0:
                    continue
                arrival = po.step + int(math.ceil(t_guess))
                proj = table.projected_defender_at(
                    tp[0], tp[1], tp[5], tp[6], arrival,
                )
                needed = max(int(w["min_launch_size"]),
                             proj + int(w["ships_safety_margin"]))
                # Allies: send much smaller reinforcement
                if tp[1] == po.player:
                    needed = max(int(w["min_launch_size"]), proj // 5 + 1)

                cap = int(available * w["max_launch_fraction"])
                ships_to_send = min(needed, cap, available)
                if ships_to_send < int(w["min_launch_size"]):
                    continue

                score, angle, _ = _score_target(
                    mp, tp, ip, po, table, self.weights, ships_to_send,
                )
                if not math.isfinite(score):
                    continue
                ranked.append((score, angle, ships_to_send, int(tp[0]), tp))

            if not ranked:
                continue

            ranked.sort(key=lambda t: t[0], reverse=True)

            # Try top-K; launch the first target whose full trajectory is
            # clear (no intervening planets, no sun crossing, no
            # off-board step). If *all* top-K are obstructed, skip this
            # my-planet this turn — better a pass than a wasted fleet.
            chosen: Optional[Tuple[float, float, int, int, float]] = None
            K = 5
            for (score, angle, ships_to_send, target_pid, tp) in ranked[:K]:
                if score <= 0:
                    break
                # Recompute travel time at the *actual* ship count so we
                # can register a correct arrival in the fleet-arrival
                # table once we commit to this launch.
                ip_t = po.initial_planet_by_id.get(target_pid, tp)
                _, t_actual = _travel_turns(
                    (mp[2], mp[3]), tp, ip_t,
                    po.angular_velocity, po.step, ships_to_send,
                    source_radius=float(mp[4]), po=po,
                )
                if math.isinf(t_actual) or t_actual <= 0:
                    continue
                obstr = _trajectory_obstruction(
                    source_center=(float(mp[2]), float(mp[3])),
                    source_radius=float(mp[4]),
                    angle=float(angle),
                    ships=int(ships_to_send),
                    target_pid=int(target_pid),
                    po=po,
                )
                if obstr == _OBSTR_CLEAR:
                    chosen = (score, angle, ships_to_send, target_pid, t_actual)
                    break

            if chosen is None:
                continue

            score, angle, ships_to_send, target_pid, t_actual = chosen
            moves.append([mpid, float(angle), int(ships_to_send)])
            # Side-channel: record the planner's *intent* — (from, target,
            # predicted travel). Verifier tools use this to tell a true
            # miss ("we aimed at planet X and the fleet didn't arrive")
            # from benign outcomes ("we aimed at planet X but X was
            # captured before arrival"). Order matches the wire list so
            # callers can map `fleet_id = next_fleet_id + i`.
            intent = LaunchIntent(
                turn=int(po.step),
                from_pid=int(mpid),
                target_pid=int(target_pid),
                angle=float(angle),
                ships=int(ships_to_send),
                predicted_travel_turns=float(t_actual),
                predicted_arrival_turn=int(po.step) + int(math.ceil(t_actual)),
                score=float(score),
            )
            self.last_launch_intents.append(intent)
            # Also append to the full-game launch log for post-mortem
            # telemetry. Cheap (one list append per launch); no per-turn
            # cost beyond the emission itself.
            self.launch_log.append(intent)
            # Register this launch in the arrival table so subsequent target
            # scoring (in this same turn's planning) sees it.
            table.add(
                target_pid,
                po.step + int(math.ceil(t_actual)),
                po.player, ships_to_send,
            )

        return moves


def build(**overrides) -> HeuristicAgent:
    """Factory for the heuristic agent. Accepts weight overrides."""
    return HeuristicAgent(weights=overrides if overrides else None)




# --- agent entry point ---

agent = HeuristicAgent().as_kaggle_agent()
