# Auto-generated Orbit Wars submission. Do not edit by hand.
# Built by tools/bundle.py on 2026-04-24 20:44:14.
# Bot: mcts_bot
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

    __slots__ = ("_t0", "_best", "_hard_stop_at", "_extra_budget_ms")

    def __init__(
        self,
        hard_stop_at: float | None = None,
        extra_budget_ms: float = 0.0,
    ) -> None:
        self._t0 = time.perf_counter()
        self._best: Action = no_op()
        self._hard_stop_at = hard_stop_at
        # Per-turn boost drawn from ``obs.remainingOverageTime``. When the
        # Kaggle overage bank is fat, the agent wrapper can pass a
        # positive value here; every ``remaining_ms`` / ``should_stop``
        # call then treats the caller's base deadline as lifted by this
        # many milliseconds. Zero keeps behavior identical to turns that
        # don't (or can't) use the bank. See ``Agent.deadline_boost_ms``.
        self._extra_budget_ms = float(max(0.0, extra_budget_ms))

    def stage(self, action: Action) -> None:
        """Mark this action as the current best; returned if deadline hits."""
        self._best = action

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    @property
    def extra_budget_ms(self) -> float:
        """Read-only accessor for the overage-bank boost applied this turn."""
        return self._extra_budget_ms

    def remaining_ms(self, deadline_ms: float = SEARCH_DEADLINE_MS) -> float:
        return (deadline_ms + self._extra_budget_ms) - self.elapsed_ms()

    def should_stop(self, deadline_ms: float = SEARCH_DEADLINE_MS) -> bool:
        # An external hard stop (e.g. outer MCTS deadline) always wins.
        if self._hard_stop_at is not None:
            if time.perf_counter() >= self._hard_stop_at:
                return True
        return self.elapsed_ms() >= deadline_ms + self._extra_budget_ms

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

    # ---- Overage-bank hook ---------------------------------------------
    #
    # The Kaggle simulator draws from ``obs.remainingOverageTime`` whenever
    # a turn overshoots ``actTimeout`` (1 s). Most agents don't need that
    # — trivial baselines finish in <10 ms. But search-heavy agents can
    # benefit from spending the bank on the opening turns, where a deeper
    # look-ahead pays off before the map has diverged. Subclasses opt in
    # by overriding ``deadline_boost_ms``. The default is 0 (no boost),
    # which preserves existing behavior for every shipped agent.
    #
    # Safety: the boost is read INSIDE ``kaggle_agent``'s try/except, so
    # a misbehaving override can't forfeit the match — it at worst
    # returns 0 and we fall back to the standard 850 ms deadline.
    def deadline_boost_ms(self, obs, step: int) -> float:  # pragma: no cover — default
        """Extra per-turn budget in ms drawn from ``obs.remainingOverageTime``.

        Returns 0.0 by default. Subclasses that want to exploit the
        overage bank should override and return a positive number on
        turns where a longer search is worth the bank draw. The wrapper
        adds this to both the search deadline and the hard-timeout guard.
        """
        return 0.0

    # ---- Kaggle submission wrapper ----
    def as_kaggle_agent(self) -> Callable:
        """Return a plain callable usable as a Kaggle submission.

        The callable honors the hard deadline: if `act` runs long we return
        the staged best-so-far; if it raises, we return no_op.
        """

        def kaggle_agent(obs, cfg=None):
            # Compute the per-turn overage boost first so Deadline knows
            # its true ceiling before ``act`` does anything expensive. Any
            # exception here degrades gracefully to zero-boost behavior —
            # we'd rather run under the default 850 ms ceiling than forfeit
            # on a malformed override.
            try:
                step = int(obs_get(obs, "step", 0))
                boost_ms = float(self.deadline_boost_ms(obs, step))
                if not math.isfinite(boost_ms) or boost_ms < 0.0:
                    boost_ms = 0.0
            except Exception:
                boost_ms = 0.0
            dl = Deadline(extra_budget_ms=boost_ms)
            try:
                result = self.act(obs, dl)
                # The hard-timeout guard lifts by the same boost so the
                # wrapper doesn't reject an otherwise-legal overage turn.
                if dl.elapsed_ms() > HARD_DEADLINE_MS + boost_ms:
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



# --- inlined: orbitwars/bots/fast_rollout.py ---

"""Ultra-cheap rollout policy for MCTS.

The `HeuristicAgent` takes ~18 ms per `act()` call — acceptable for the
one root decision it makes per real turn, but catastrophic inside an
MCTS rollout. At `rollout_depth=15` with 2 players, one rollout is
~560 ms — we can't finish a single rollout inside the 300 ms search
budget.

This file provides `FastRolloutAgent`, which mirrors AlphaGo's
"fast rollout policy" split: the slow/accurate policy drives the tree
and the root decision, and a cheap policy fills in rollout plies. The
fast policy intentionally skips every expensive subroutine:

  * No arrival-table build.
  * No scoring sweep over targets.
  * No Newton intercept for orbiting targets (uses static-intercept,
    which is just an `atan2`).
  * No sun-tangent routing — we accept the rollout-noise cost of the
    occasional fleet flying into the sun. Every candidate gets the
    same bias, so ranking at the root is preserved.
  * No cooldowns, no defense guards, no launch-state tracking.

The one rule: from each of my planets with enough ships, send
`send_fraction × available` at `atan2(weighted_nearest_target)`.

Expected cost per `act()` call: <500 µs — a 30-50× speedup over the
full heuristic. Net effect at default budget:
    sims/turn goes from <1 (diagnostic only) to ~10-15 (real search).

Archetype flavoring:
  The four knobs (``min_launch_size``, ``send_fraction``,
  ``enemy_bias``, ``keep_reserve_ships``) expose enough surface that
  ``from_weights(HEURISTIC_WEIGHTS-style dict)`` can build a fast
  rollout agent whose aggregate launch cadence and target preference
  track any of the archetype configs. This is used by MCTSAgent to
  run opp rollouts under the posterior's most-likely archetype
  *without* paying the ~18 ms/ply HeuristicAgent cost.

Invariants preserved:
  * Only my planets launch.
  * `ships <= planet.ships` always.
  * Angle is finite (atan2 well-defined when source != target; the
    guard below rejects self-targets).

This class is used inside MCTS rollouts when
`GumbelConfig.rollout_policy == "fast"`. The root anchor is still
provided by `HeuristicAgent`; only rollout plies swap in the fast
agent.
"""

import math
from typing import Any, Dict, List, Optional



class FastRolloutAgent(Agent):
    """Cheapest-possible rollout policy: nearest-target static push.

    Knobs intentionally minimal — tuning this is not the point. If
    rollouts need to be smarter, promote to a real heuristic; if they
    need to be faster, inline the loop.

    Attributes:
        min_launch_size: do not launch a fleet smaller than this many
            ships (matches HeuristicAgent's default). Prevents single-
            ship dribbles that clutter the fleet count without changing
            the value.
        send_fraction: fraction of available ships to send from a
            launching planet. 0.8 leaves a 20% reserve and matches the
            HeuristicAgent default, so fast and slow rollouts produce
            comparably-sized fleets — only the target-selection logic
            differs.
        enemy_bias: distance multiplier for enemy targets. <1.0 biases
            toward enemies (rusher/harasser flavor); >1.0 biases toward
            neutrals (economy/comet_camper flavor). Applied as
            ``effective_d2 = d2 * enemy_bias^2`` for enemy targets so
            an enemy at distance D "competes" with a neutral at
            ``D * enemy_bias``. 1.0 recovers the original behavior
            (pure nearest-target).
        keep_reserve_ships: extra ship reserve held back beyond
            ``min_launch_size``. Defender-style archetypes set this
            high; rusher-style set it to 0. A planet launches only
            when ``available >= min_launch_size + keep_reserve_ships``,
            and sends at most ``available - keep_reserve_ships``.
    """

    name = "fast_rollout"

    def __init__(
        self,
        min_launch_size: int = 20,
        send_fraction: float = 0.8,
        enemy_bias: float = 1.0,
        keep_reserve_ships: int = 0,
    ) -> None:
        self.min_launch_size = int(min_launch_size)
        self.send_fraction = float(send_fraction)
        # Clamp to avoid pathological 0/negative multipliers that would
        # make every enemy instantly dominate or disappear.
        self.enemy_bias = float(max(0.1, min(10.0, enemy_bias)))
        self.keep_reserve_ships = int(max(0, keep_reserve_ships))

    @classmethod
    def from_weights(cls, weights: Dict[str, float]) -> "FastRolloutAgent":
        """Build a fast-rollout flavor matching a HEURISTIC_WEIGHTS dict.

        Pulls the four knob-equivalents out of the weights and clamps
        to sane ranges:
          * ``min_launch_size`` (direct copy; default 20)
          * ``max_launch_fraction`` → send_fraction (direct; default 0.8)
          * ``mult_enemy`` / ``mult_neutral`` → enemy_bias, inverted so
            a HIGHER mult_enemy becomes a LOWER distance multiplier
            (i.e. stronger enemy preference). Computed as
            ``mult_neutral / max(mult_enemy, eps)``.
          * ``keep_reserve_ships`` (direct copy; default 0)

        Unspecified keys fall back to FastRolloutAgent's own defaults.
        """
        mult_enemy = float(weights.get("mult_enemy", 1.0))
        mult_neutral = float(weights.get("mult_neutral", 1.0))
        # Inverse: lower bias = stronger enemy preference. Clamp to
        # avoid division-by-zero if mult_enemy is plausibly 0.
        enemy_bias = mult_neutral / max(mult_enemy, 1e-3)
        return cls(
            min_launch_size=int(weights.get("min_launch_size", 20)),
            send_fraction=float(weights.get("max_launch_fraction", 0.8)),
            enemy_bias=enemy_bias,
            keep_reserve_ships=int(weights.get("keep_reserve_ships", 0)),
        )

    def act(self, obs: Any, deadline: Deadline) -> Action:
        # Always stage a safe fallback first; if we get interrupted
        # mid-loop the caller still has a valid action.
        deadline.stage(no_op())

        player = obs_get(obs, "player", 0)
        planets = obs_get(obs, "planets", [])
        if not planets:
            return no_op()

        # Single-pass ownership partition. Both lists hold references
        # into the same planet entries, so we avoid copying. Enemy
        # flagging is precomputed once so the inner loop just reads a
        # bool rather than re-comparing owners.
        my_planets: List[Any] = []
        targets: List[Any] = []
        target_is_enemy: List[bool] = []
        for p in planets:
            owner = p[1]
            if owner == player:
                my_planets.append(p)
            else:
                targets.append(p)
                # Any non-ours-and-non-neutral owner is an enemy.
                target_is_enemy.append(owner != -1 and owner != player)

        # Either no launchers or no opponents/neutrals to push toward:
        # there is nothing useful to do.
        if not my_planets or not targets:
            return no_op()

        moves: Action = []
        min_size = self.min_launch_size
        frac = self.send_fraction
        reserve = self.keep_reserve_ships
        # Apply the bias as a squared multiplier in the distance
        # comparison — equivalent to scaling effective distance by
        # ``enemy_bias`` while avoiding a sqrt.
        enemy_d2_mult = self.enemy_bias * self.enemy_bias

        for mp in my_planets:
            available = int(mp[5])
            # Don't launch unless we can afford min_size AND still hold
            # the reserve. A reserve of 0 recovers the original gate.
            if available < min_size + reserve:
                continue

            # Find nearest target by squared-Euclidean — no sqrt needed
            # when we only need the argmin. This is the hot inner loop.
            # enemy targets' effective distance is scaled by
            # ``enemy_bias`` so enemy-leaning archetypes prefer enemies
            # even when a neutral is marginally closer.
            mx = float(mp[2])
            my_ = float(mp[3])
            best_d2 = float("inf")
            best_tp: Optional[Any] = None
            for tp, is_enemy in zip(targets, target_is_enemy):
                dx = float(tp[2]) - mx
                dy = float(tp[3]) - my_
                d2 = dx * dx + dy * dy
                if is_enemy:
                    d2 *= enemy_d2_mult
                if d2 < best_d2:
                    best_d2 = d2
                    best_tp = tp

            if best_tp is None or best_d2 == 0.0:
                # Degenerate: co-located target (shouldn't happen in
                # valid states) or no targets at all.
                continue

            # Static intercept angle — just atan2. We deliberately
            # ignore orbital motion: in 15-ply rollouts the bias is
            # small, and every candidate experiences the same bias,
            # so simple-regret ranking at the root is preserved.
            angle = math.atan2(
                float(best_tp[3]) - my_,
                float(best_tp[2]) - mx,
            )

            # Ship count respects both send_fraction and the reserve
            # floor. send_fraction on (available - reserve) so the
            # reserve is literally set aside.
            launchable = available - reserve
            ships = int(launchable * frac)
            if ships < min_size:
                ships = min_size
            if ships > launchable:
                ships = launchable

            moves.append([int(mp[0]), float(angle), int(ships)])

        deadline.stage(moves)
        return moves



# --- inlined: orbitwars/engine/fast_engine.py ---

"""Numpy SoA re-implementation of the orbit_wars engine.

Goal: state-equal parity with kaggle_environments v1.0.9 over 1000 random seeds,
while running materially faster (target 10-100x) than the stock engine.

Key design choices:

  * Planets and fleets are stored as parallel numpy arrays (Structure-of-Arrays).
    The three hot loops — fleet movement + OOB/sun/planet collision, planet
    rotation + sweep, comet movement + sweep — are vectorized via broadcasting
    (O(F*P) with F,P <= ~300 is 100k flops per turn, negligible).
  * Comet groups (which carry precomputed paths) stay as list-of-dicts: few of
    them, branchy logic, path lookups are dict reads — not hot.
  * Combat events are stored as lists of (owner, ships) tuples per planet,
    captured at collision time so the order of fleet removal vs combat
    resolution doesn't matter.
  * Ship counts are stored as int64 to mirror Python's unbounded ints;
    positions as float64 to avoid drift accumulating over 500 turns (important
    for parity with the reference engine).

Parity target: integer-equal ship counts and owner IDs for every planet/fleet
at every turn; positions within 1e-9 of reference (pure float-math match is
achievable since we compute each quantity the same way).
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- Engine constants (must mirror kaggle_environments/envs/orbit_wars) ---

BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1
COMET_SPAWN_STEPS = [50, 150, 250, 350, 450]
DEFAULT_MAX_SPEED = 6.0
DEFAULT_COMET_SPEED = 4.0
DEFAULT_EPISODE_STEPS = 500


# --- Vectorized geometry helpers ---

def _pt_seg_dist_pairs(
    pts_x: np.ndarray, pts_y: np.ndarray,       # shape (P,)
    seg_v_x: np.ndarray, seg_v_y: np.ndarray,   # shape (F,)
    seg_w_x: np.ndarray, seg_w_y: np.ndarray,   # shape (F,)
) -> np.ndarray:
    """All-pairs distance: point[i] to segment[j]. Returns shape (P, F)."""
    px = pts_x[:, None]
    py = pts_y[:, None]
    vx = seg_v_x[None, :]
    vy = seg_v_y[None, :]
    wx = seg_w_x[None, :]
    wy = seg_w_y[None, :]
    dx = wx - vx
    dy = wy - vy
    l2 = dx * dx + dy * dy
    safe_l2 = np.where(l2 == 0.0, 1.0, l2)
    t = ((px - vx) * dx + (py - vy) * dy) / safe_l2
    t = np.clip(t, 0.0, 1.0)
    proj_x = vx + t * dx
    proj_y = vy + t * dy
    d = np.hypot(px - proj_x, py - proj_y)
    if np.any(l2 == 0.0):
        d_zero = np.hypot(px - vx, py - vy)
        d = np.where(l2 == 0.0, d_zero, d)
    return d


def _seg_dist_single_point_many_segs(
    px: float, py: float,
    seg_v_x: np.ndarray, seg_v_y: np.ndarray,
    seg_w_x: np.ndarray, seg_w_y: np.ndarray,
) -> np.ndarray:
    """One point, many segments. Returns shape (F,)."""
    dx = seg_w_x - seg_v_x
    dy = seg_w_y - seg_v_y
    l2 = dx * dx + dy * dy
    safe_l2 = np.where(l2 == 0.0, 1.0, l2)
    t = ((px - seg_v_x) * dx + (py - seg_v_y) * dy) / safe_l2
    t = np.clip(t, 0.0, 1.0)
    proj_x = seg_v_x + t * dx
    proj_y = seg_v_y + t * dy
    d = np.hypot(px - proj_x, py - proj_y)
    if np.any(l2 == 0.0):
        d_zero = np.hypot(px - seg_v_x, py - seg_v_y)
        d = np.where(l2 == 0.0, d_zero, d)
    return d


def _seg_dist_many_points_single_seg(
    pts_x: np.ndarray, pts_y: np.ndarray,
    v_x: float, v_y: float,
    w_x: float, w_y: float,
) -> np.ndarray:
    """Many points, one segment. Returns shape (P,)."""
    dx = w_x - v_x
    dy = w_y - v_y
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        return np.hypot(pts_x - v_x, pts_y - v_y)
    t = ((pts_x - v_x) * dx + (pts_y - v_y) * dy) / l2
    t = np.clip(t, 0.0, 1.0)
    proj_x = v_x + t * dx
    proj_y = v_y + t * dy
    return np.hypot(pts_x - proj_x, pts_y - proj_y)


def _fleet_speed_batched(ships: np.ndarray, max_speed: float) -> np.ndarray:
    """Vectorized fleet speed. Clamps to max_speed."""
    ships_f = ships.astype(np.float64)
    safe = np.maximum(ships_f, 1.0)
    out = 1.0 + (max_speed - 1.0) * (np.log(safe) / math.log(1000.0)) ** 1.5
    np.clip(out, 0.0, max_speed, out=out)
    out[ships <= 0] = 0.0
    return out


# --- State ---

@dataclass
class GameConfig:
    ship_speed: float = DEFAULT_MAX_SPEED
    comet_speed: float = DEFAULT_COMET_SPEED
    episode_steps: int = DEFAULT_EPISODE_STEPS


@dataclass
class GameState:
    """Full game state, mirrors the reference engine's observation shape.

    All arrays are dense and indexed contiguously — we rebuild on every
    insert/remove to avoid alive-mask bookkeeping. Planet/fleet counts stay
    small (<300 fleets, <60 planets) so compact-on-mutate is fine here and
    keeps semantics identical to the list-based reference.
    """
    config: GameConfig
    step: int = 0
    angular_velocity: float = 0.0
    next_fleet_id: int = 0

    # Planets (including comets)
    p_id: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    p_owner: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    p_x: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    p_y: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    p_radius: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    p_ships: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    p_production: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    # Initial positions for rotation math
    p_init_x: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    p_init_y: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    # Fleets
    f_id: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    f_owner: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    f_x: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    f_y: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    f_angle: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    f_from_pid: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    f_ships: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    # Comet bookkeeping. Each group: {planet_ids, paths, path_index}
    comets: List[Dict[str, Any]] = field(default_factory=list)
    comet_planet_ids: List[int] = field(default_factory=list)

    # ---- Structural helpers ----
    def num_planets(self) -> int:
        return int(self.p_id.shape[0])

    def num_fleets(self) -> int:
        return int(self.f_id.shape[0])

    def _comet_pid_set(self) -> set:
        return set(self.comet_planet_ids)

    def planet_index(self, pid: int) -> int:
        """Return current dense array index for planet id, or -1."""
        idx = np.where(self.p_id == pid)[0]
        return int(idx[0]) if idx.size else -1

    def to_official_planets(self) -> List[List[Any]]:
        """Emit planets as the engine's list-of-lists form."""
        return [
            [
                int(self.p_id[i]),
                int(self.p_owner[i]),
                float(self.p_x[i]),
                float(self.p_y[i]),
                float(self.p_radius[i]),
                int(self.p_ships[i]),
                int(self.p_production[i]),
            ]
            for i in range(self.num_planets())
        ]

    def to_official_initial_planets(self) -> List[List[Any]]:
        return [
            [
                int(self.p_id[i]),
                -1,
                float(self.p_init_x[i]),
                float(self.p_init_y[i]),
                float(self.p_radius[i]),
                int(self.p_ships[i]),
                int(self.p_production[i]),
            ]
            for i in range(self.num_planets())
        ]

    def to_official_fleets(self) -> List[List[Any]]:
        return [
            [
                int(self.f_id[i]),
                int(self.f_owner[i]),
                float(self.f_x[i]),
                float(self.f_y[i]),
                float(self.f_angle[i]),
                int(self.f_from_pid[i]),
                int(self.f_ships[i]),
            ]
            for i in range(self.num_fleets())
        ]


# --- Ingest/append helpers ---

def _ingest_planets(
    state: GameState,
    planets_list: List[List[Any]],
    initial_planets: Optional[List[List[Any]]] = None,
) -> None:
    n = len(planets_list)
    state.p_id = np.array([int(p[0]) for p in planets_list], dtype=np.int64) if n else np.zeros(0, dtype=np.int64)
    state.p_owner = np.array([int(p[1]) for p in planets_list], dtype=np.int64) if n else np.zeros(0, dtype=np.int64)
    state.p_x = np.array([float(p[2]) for p in planets_list], dtype=np.float64) if n else np.zeros(0, dtype=np.float64)
    state.p_y = np.array([float(p[3]) for p in planets_list], dtype=np.float64) if n else np.zeros(0, dtype=np.float64)
    state.p_radius = np.array([float(p[4]) for p in planets_list], dtype=np.float64) if n else np.zeros(0, dtype=np.float64)
    state.p_ships = np.array([int(p[5]) for p in planets_list], dtype=np.int64) if n else np.zeros(0, dtype=np.int64)
    state.p_production = np.array([int(p[6]) for p in planets_list], dtype=np.int64) if n else np.zeros(0, dtype=np.int64)

    if initial_planets is not None and len(initial_planets) == n and n:
        state.p_init_x = np.array([float(p[2]) for p in initial_planets], dtype=np.float64)
        state.p_init_y = np.array([float(p[3]) for p in initial_planets], dtype=np.float64)
    else:
        state.p_init_x = state.p_x.copy()
        state.p_init_y = state.p_y.copy()


def _append_planets(state: GameState, new_planets: List[List[Any]]) -> None:
    if not new_planets:
        return
    ids = np.array([int(p[0]) for p in new_planets], dtype=np.int64)
    owners = np.array([int(p[1]) for p in new_planets], dtype=np.int64)
    xs = np.array([float(p[2]) for p in new_planets], dtype=np.float64)
    ys = np.array([float(p[3]) for p in new_planets], dtype=np.float64)
    rs = np.array([float(p[4]) for p in new_planets], dtype=np.float64)
    ships = np.array([int(p[5]) for p in new_planets], dtype=np.int64)
    prods = np.array([int(p[6]) for p in new_planets], dtype=np.int64)
    state.p_id = np.concatenate([state.p_id, ids])
    state.p_owner = np.concatenate([state.p_owner, owners])
    state.p_x = np.concatenate([state.p_x, xs])
    state.p_y = np.concatenate([state.p_y, ys])
    state.p_radius = np.concatenate([state.p_radius, rs])
    state.p_ships = np.concatenate([state.p_ships, ships])
    state.p_production = np.concatenate([state.p_production, prods])
    # Newly spawned comets: initial_x/y recorded as the spawn position
    # (the engine stores the first off-board placeholder in initial_planets).
    state.p_init_x = np.concatenate([state.p_init_x, xs.copy()])
    state.p_init_y = np.concatenate([state.p_init_y, ys.copy()])


def _ingest_fleets(state: GameState, fleets_list: List[List[Any]]) -> None:
    n = len(fleets_list)
    state.f_id = np.array([int(f[0]) for f in fleets_list], dtype=np.int64) if n else np.zeros(0, dtype=np.int64)
    state.f_owner = np.array([int(f[1]) for f in fleets_list], dtype=np.int64) if n else np.zeros(0, dtype=np.int64)
    state.f_x = np.array([float(f[2]) for f in fleets_list], dtype=np.float64) if n else np.zeros(0, dtype=np.float64)
    state.f_y = np.array([float(f[3]) for f in fleets_list], dtype=np.float64) if n else np.zeros(0, dtype=np.float64)
    state.f_angle = np.array([float(f[4]) for f in fleets_list], dtype=np.float64) if n else np.zeros(0, dtype=np.float64)
    state.f_from_pid = np.array([int(f[5]) for f in fleets_list], dtype=np.int64) if n else np.zeros(0, dtype=np.int64)
    state.f_ships = np.array([int(f[6]) for f in fleets_list], dtype=np.int64) if n else np.zeros(0, dtype=np.int64)


def _append_fleet(
    state: GameState,
    fid: int, owner: int,
    x: float, y: float, angle: float,
    from_pid: int, ships: int,
) -> None:
    state.f_id = np.append(state.f_id, fid)
    state.f_owner = np.append(state.f_owner, owner)
    state.f_x = np.append(state.f_x, x)
    state.f_y = np.append(state.f_y, y)
    state.f_angle = np.append(state.f_angle, angle)
    state.f_from_pid = np.append(state.f_from_pid, from_pid)
    state.f_ships = np.append(state.f_ships, ships)


# --- Engine ---

class FastEngine:
    """Deterministic, numpy-accelerated orbit_wars engine.

    Usage:
        eng = FastEngine.from_scratch(num_agents=2, seed=42)
        while not eng.done:
            actions = [agent(obs) for agent in ...]
            eng.step(actions)
    """

    def __init__(
        self,
        state: GameState,
        num_agents: int = 2,
        rng: Optional[Any] = None,
    ):
        self.state = state
        self.num_agents = num_agents
        self.done: bool = False
        self.rewards: List[int] = [0] * num_agents
        # IMPORTANT: use an instance-local RNG for all step-time randomness
        # (comet ship counts, etc). If we used the global `random` module,
        # MCTS rollouts would consume entropy from the same stream that the
        # real Kaggle judge uses for its own engine, desynchronizing the
        # outer game. See `_maybe_spawn_comets` — line 455-458 of the
        # reference engine and our mirror at the same logical site. A fresh
        # `random.Random()` seeds from os.urandom, decoupling from global.
        #
        # The parity validator EXPLICITLY passes `rng=random` (the module
        # itself) to share the global stream with the reference engine.
        # Duck typing: both the module and `random.Random()` expose
        # `.randint(a, b)`.
        self._rng = rng if rng is not None else random.Random()

    # ---- Construction ----

    @classmethod
    def from_scratch(
        cls,
        num_agents: int = 2,
        seed: Optional[int] = None,
        config: Optional[GameConfig] = None,
    ) -> "FastEngine":
        """Generate a fresh game using the reference engine's map generator.

        We import the reference generator to guarantee identical map layouts.
        """
        if seed is not None:
            random.seed(seed)
        cfg = config or GameConfig()
        state = GameState(config=cfg)
        state.angular_velocity = random.uniform(0.025, 0.05)

        from kaggle_environments.envs.orbit_wars.orbit_wars import generate_planets, distance
        planets_list = generate_planets()

        num_groups = len(planets_list) // 4
        if num_groups > 0:
            home_group = random.randint(0, num_groups - 1)
            base = home_group * 4

            if num_agents == 4:
                q1 = planets_list[base]
                orb_r = distance((q1[2], q1[3]), (CENTER, CENTER))
                if orb_r + q1[4] < ROTATION_RADIUS_LIMIT:
                    for g in range(num_groups):
                        gb = g * 4
                        gp = planets_list[gb]
                        g_orb = distance((gp[2], gp[3]), (CENTER, CENTER))
                        if g_orb + gp[4] < ROTATION_RADIUS_LIMIT:
                            if abs((gp[2] - CENTER) - (gp[3] - CENTER)) < 0.01:
                                base = gb
                                break

            if num_agents == 2:
                planets_list[base][1] = 0
                planets_list[base][5] = 10
                planets_list[base + 3][1] = 1
                planets_list[base + 3][5] = 10
            elif num_agents == 4:
                for j in range(4):
                    planets_list[base + j][1] = j
                    planets_list[base + j][5] = 10

        _ingest_planets(state, planets_list)
        return cls(state, num_agents=num_agents)

    @classmethod
    def from_official_obs(
        cls,
        obs,
        num_agents: int = 2,
        config: Optional[GameConfig] = None,
        rng: Optional[Any] = None,
    ) -> "FastEngine":
        """Initialize FastEngine from a running kaggle env's obs.

        Args:
            rng: an object with a `randint(a, b)` method used for step-time
                randomness (comet ship sizing). Defaults to a fresh
                `random.Random()` that is decoupled from the global random
                module — important during MCTS rollouts to avoid
                consuming entropy from the judge's global RNG stream.
                Pass `random` (the module itself) when you WANT to share
                global state, e.g. in the parity validator where both
                engines must consume from the same stream.
        """
        cfg = config or GameConfig()
        state = GameState(config=cfg)
        state.angular_velocity = float(getattr(obs, "angular_velocity", 0.0))
        state.step = int(getattr(obs, "step", 0))
        state.next_fleet_id = int(getattr(obs, "next_fleet_id", 0))
        _ingest_planets(
            state,
            [list(p) for p in getattr(obs, "planets", [])],
            initial_planets=[list(p) for p in getattr(obs, "initial_planets", [])],
        )
        _ingest_fleets(state, [list(f) for f in getattr(obs, "fleets", [])])
        # Deep-copy comets to decouple from reference engine state
        state.comets = []
        for g in getattr(obs, "comets", []):
            state.comets.append({
                "planet_ids": list(g["planet_ids"]),
                "paths": [[list(pt) for pt in p] for p in g["paths"]],
                "path_index": int(g["path_index"]),
            })
        state.comet_planet_ids = list(getattr(obs, "comet_planet_ids", []))
        return cls(state, num_agents=num_agents, rng=rng)

    # ---- Read-only API ----

    def observation(self, player: int) -> Dict[str, Any]:
        return {
            "player": player,
            "step": self.state.step,
            "angular_velocity": self.state.angular_velocity,
            "planets": self.state.to_official_planets(),
            "initial_planets": self.state.to_official_initial_planets(),
            "fleets": self.state.to_official_fleets(),
            "next_fleet_id": self.state.next_fleet_id,
            "comets": [dict(g) for g in self.state.comets],
            "comet_planet_ids": list(self.state.comet_planet_ids),
        }

    def scores(self) -> List[int]:
        scores = [0] * self.num_agents
        for i in range(self.state.num_planets()):
            o = int(self.state.p_owner[i])
            if 0 <= o < self.num_agents:
                scores[o] += int(self.state.p_ships[i])
        for i in range(self.state.num_fleets()):
            o = int(self.state.f_owner[i])
            if 0 <= o < self.num_agents:
                scores[o] += int(self.state.f_ships[i])
        return scores

    # ---- Main step ----

    def step(self, actions: Sequence[Optional[Sequence]]) -> None:
        """Run one turn. `actions[i]` is agent i's move list or None."""
        if self.done:
            return

        st = self.state
        # IMPORTANT: do NOT increment step at the start. The reference
        # engine's interpreter reads `step` PRE-increment (the Kaggle harness
        # advances `step` AFTER the interpreter returns). We post-increment at
        # the end of this method so that subsequent turns read the right
        # step value. from_official_obs() captures obs.step which is the
        # post-previous-call value; that equals the step we'll use here.

        # 1. Remove expired comets (those whose path_index is past end)
        self._purge_expired_comets()

        # 2. Spawn new comets at designated steps
        self._maybe_spawn_comets()

        # 3. Process player actions (fleet launches)
        for player_id, action in enumerate(actions):
            self._process_moves(player_id, action)

        # 4. Production on owned planets
        owned = st.p_owner != -1
        if owned.any():
            st.p_ships[owned] += st.p_production[owned]

        # Combat events: planet_id -> list of (owner, ships) snapshots.
        # We snapshot at collision time so fleet array indexing doesn't matter
        # after subsequent movement/sweep/removal.
        combat: Dict[int, List[Tuple[int, int]]] = {int(pid): [] for pid in st.p_id}

        # Fleets caught by collisions (as indices into the current fleet arrays,
        # at the time of collision). We maintain an alive-mask so later sweep
        # passes can ignore already-destroyed fleets; at the end of step() we
        # compact the arrays.
        alive_mask = np.ones(st.num_fleets(), dtype=bool)

        # 5. Fleet movement + collision
        self._move_fleets_and_collide(alive_mask, combat)

        # 6. Planet rotation + sweep
        self._rotate_planets_and_sweep(alive_mask, combat)

        # 7. Comet movement + sweep
        self._move_comets_and_sweep(alive_mask, combat)

        # 8. Remove expired-during-movement comets
        self._purge_expired_comets()

        # 9. Compact dead fleets
        self._compact_fleets(alive_mask)

        # 10. Combat resolution (using snapshots)
        self._resolve_combat(combat)

        # 11. Terminal check (uses PRE-increment step value, matching the
        #     reference's `step >= episodeSteps - 2` check).
        self._check_terminal()

        # 12. Advance step (mirrors Kaggle harness post-call increment).
        st.step += 1

    # ---- Internal steps ----

    def _purge_expired_comets(self) -> None:
        """Remove comets whose path_index is past the end of their path."""
        st = self.state
        expired: List[int] = []
        for group in st.comets:
            idx = group["path_index"]
            for i, pid in enumerate(group["planet_ids"]):
                if idx >= len(group["paths"][i]):
                    expired.append(pid)

        if not expired:
            return
        expired_set = set(expired)
        self._remove_planets_by_pid(expired_set)
        st.comet_planet_ids = [pid for pid in st.comet_planet_ids if pid not in expired_set]
        for group in st.comets:
            group["planet_ids"] = [pid for pid in group["planet_ids"] if pid not in expired_set]
        st.comets = [g for g in st.comets if g["planet_ids"]]

    def _maybe_spawn_comets(self) -> None:
        st = self.state
        step = st.step
        if (step + 1) not in COMET_SPAWN_STEPS:
            return
        from kaggle_environments.envs.orbit_wars.orbit_wars import generate_comet_paths
        # CRITICAL: `generate_comet_paths` internally calls `random.uniform`
        # (orbit_wars.py:233,234,242) to draw ellipse eccentricity, semi-major
        # axis, and orientation — up to ~900 calls per spawn via the 300-try
        # retry loop. Those calls go to the GLOBAL `random` module regardless
        # of what rng we pass around. During MCTS rollouts that cross a spawn
        # step (every rollout past turn 50/150/250/350/450), this consumption
        # perturbs the Kaggle judge's own global random stream — which is what
        # the judge's engine uses for the REAL comet spawn at that step. Net
        # effect: rollout bookkeeping changes the game trajectory in ways the
        # agent can't see. Empirically on seed=123 this flipped outcome from
        # heur-P1 winning to MCTS-P1 losing despite MCTS returning the SAME
        # wire action as heur on every turn (see tools/diag_mcts_divergence_
        # in_env_run.py + tools/diag_who_touches_global_random.py).
        #
        # Fix: snapshot + restore global `random` state around the call —
        # ONLY in isolation mode. When `self._rng is random` (the module
        # itself — parity validator only), we intentionally DO consume
        # global state to match official behavior for parity checks.
        _isolate = self._rng is not random
        if _isolate:
            _saved_global_state = random.getstate()
        try:
            paths = generate_comet_paths(
                st.to_official_initial_planets(),
                st.angular_velocity,
                step + 1,
                st.comet_planet_ids,
                st.config.comet_speed,
            )
        finally:
            if _isolate:
                random.setstate(_saved_global_state)
        if not paths:
            return
        next_id = int(st.p_id.max()) + 1 if st.num_planets() > 0 else 0
        # NOTE: we deliberately use the INSTANCE RNG here (not the global
        # `random` module) so MCTS rollouts don't consume entropy from the
        # Kaggle judge's global stream. See `__init__` for the full story.
        comet_ships = min(
            self._rng.randint(1, 99),
            self._rng.randint(1, 99),
            self._rng.randint(1, 99),
            self._rng.randint(1, 99),
        )
        group: Dict[str, Any] = {"planet_ids": [], "paths": paths, "path_index": -1}
        new_planets: List[List[Any]] = []
        for i in range(len(paths)):
            pid = next_id + i
            group["planet_ids"].append(pid)
            st.comet_planet_ids.append(pid)
            new_planets.append([pid, -1, -99.0, -99.0, COMET_RADIUS, comet_ships, COMET_PRODUCTION])
        st.comets.append(group)
        _append_planets(st, new_planets)

    def _process_moves(self, player_id: int, action: Optional[Sequence]) -> None:
        st = self.state
        if not action or not isinstance(action, list):
            return
        for move in action:
            if not isinstance(move, (list, tuple)) or len(move) != 3:
                continue
            from_id, angle, ships = move
            try:
                from_id_i = int(from_id)
                angle_f = float(angle)
                ships_i = int(ships)
            except (TypeError, ValueError):
                continue
            pi = st.planet_index(from_id_i)
            if pi < 0:
                continue
            if int(st.p_owner[pi]) != player_id:
                continue
            if ships_i <= 0 or int(st.p_ships[pi]) < ships_i:
                continue

            st.p_ships[pi] -= ships_i
            px = float(st.p_x[pi])
            py = float(st.p_y[pi])
            pr = float(st.p_radius[pi])
            start_x = px + math.cos(angle_f) * (pr + 0.1)
            start_y = py + math.sin(angle_f) * (pr + 0.1)
            _append_fleet(st, st.next_fleet_id, player_id, start_x, start_y,
                          angle_f, from_id_i, ships_i)
            st.next_fleet_id += 1

    def _move_fleets_and_collide(
        self,
        alive_mask: np.ndarray,
        combat: Dict[int, List[Tuple[int, int]]],
    ) -> None:
        st = self.state
        F = st.num_fleets()
        if F == 0:
            return

        # Fleets just launched this turn are also in these arrays (appended by
        # _process_moves). alive_mask was sized before launches — extend it.
        if alive_mask.shape[0] < F:
            extra = np.ones(F - alive_mask.shape[0], dtype=bool)
            alive_mask_full = np.concatenate([alive_mask, extra])
            # Mutate the caller's view by copying back — callers reassign below.
            # We can't reassign the passed-in array; instead return via
            # aliasing: write back into the buffer by changing everything the
            # caller reads. Simplest: do this in step() before calling.
            # To avoid confusion, we document in step() that alive_mask is
            # created AFTER launches. Let's just operate on `alive_mask_full`
            # locally and accept that launches added this turn are all alive.
            alive_mask = alive_mask_full

        max_speed = st.config.ship_speed
        speeds = _fleet_speed_batched(st.f_ships, max_speed)
        old_x = st.f_x.copy()
        old_y = st.f_y.copy()
        new_x = old_x + np.cos(st.f_angle) * speeds
        new_y = old_y + np.sin(st.f_angle) * speeds

        # Update positions in-place (reference: mutates fleet entries before
        # running collision checks).
        st.f_x = new_x
        st.f_y = new_y

        oob = (new_x < 0.0) | (new_x > BOARD_SIZE) | (new_y < 0.0) | (new_y > BOARD_SIZE)

        sun_d = _seg_dist_single_point_many_segs(
            CENTER, CENTER, old_x, old_y, new_x, new_y,
        )
        sun_hit = sun_d < SUN_RADIUS

        P = st.num_planets()
        planet_hit = np.zeros(F, dtype=bool)
        planet_hit_pid = np.full(F, -1, dtype=np.int64)

        if P > 0:
            d = _pt_seg_dist_pairs(
                st.p_x, st.p_y, old_x, old_y, new_x, new_y,
            )  # shape (P, F)
            hits = d < st.p_radius[:, None]
            any_hit = hits.any(axis=0)
            first_hit_p = np.argmax(hits, axis=0)
            # A fleet is flagged as planet-hit only if it's not already OOB or
            # sun-killed (reference uses `continue` to skip planet check).
            planet_hit = any_hit & ~oob & ~sun_hit
            planet_hit_pid = np.where(planet_hit, st.p_id[first_hit_p], -1)

        # Record combat events and update alive_mask in precedence order.
        # Iterating Python-side is fine — F per turn is small.
        for fi in range(F):
            if not alive_mask[fi]:
                continue
            if oob[fi] or sun_hit[fi]:
                alive_mask[fi] = False
            elif planet_hit[fi]:
                pid = int(planet_hit_pid[fi])
                combat[pid].append((int(st.f_owner[fi]), int(st.f_ships[fi])))
                alive_mask[fi] = False

        # Propagate updated alive_mask back to the shared buffer by slicing.
        # The caller passed a view; since we may have extended it, mutate
        # in-place by copying back (step() recreates alive_mask after launches
        # to avoid this — see step() implementation note).
        # Nothing to do if we didn't extend.

    def _rotate_planets_and_sweep(
        self,
        alive_mask: np.ndarray,
        combat: Dict[int, List[Tuple[int, int]]],
    ) -> None:
        st = self.state
        P = st.num_planets()
        if P == 0:
            return

        comet_pids = st._comet_pid_set()
        omega = st.angular_velocity
        step = st.step

        dx = st.p_init_x - CENTER
        dy = st.p_init_y - CENTER
        r = np.hypot(dx, dy)
        init_angle = np.arctan2(dy, dx)
        current_angle = init_angle + omega * step

        is_rotating = ((r + st.p_radius) < ROTATION_RADIUS_LIMIT)
        if comet_pids:
            comet_mask = np.array([int(pid) in comet_pids for pid in st.p_id], dtype=bool)
            is_rotating &= ~comet_mask

        old_px = st.p_x.copy()
        old_py = st.p_y.copy()
        new_px = np.where(is_rotating, CENTER + r * np.cos(current_angle), st.p_x)
        new_py = np.where(is_rotating, CENTER + r * np.sin(current_angle), st.p_y)

        st.p_x = new_px
        st.p_y = new_py

        # Sweep for planets that actually moved
        for pi in range(P):
            if old_px[pi] == new_px[pi] and old_py[pi] == new_py[pi]:
                continue
            pid = int(st.p_id[pi])
            if not alive_mask.any():
                continue
            alive_idx = np.where(alive_mask)[0]
            d = _seg_dist_many_points_single_seg(
                st.f_x[alive_idx], st.f_y[alive_idx],
                old_px[pi], old_py[pi], new_px[pi], new_py[pi],
            )
            caught = d < st.p_radius[pi]
            for hit_local, ai in zip(caught, alive_idx):
                if hit_local:
                    combat[pid].append((int(st.f_owner[ai]), int(st.f_ships[ai])))
                    alive_mask[ai] = False

    def _move_comets_and_sweep(
        self,
        alive_mask: np.ndarray,
        combat: Dict[int, List[Tuple[int, int]]],
    ) -> None:
        st = self.state

        for group in st.comets:
            group["path_index"] += 1
            idx = group["path_index"]
            for i, pid in enumerate(group["planet_ids"]):
                pi = st.planet_index(pid)
                if pi < 0:
                    continue
                p_path = group["paths"][i]
                if idx >= len(p_path):
                    # Expired; do not move, do not sweep. Purge happens later.
                    continue
                old_x = float(st.p_x[pi])
                old_y = float(st.p_y[pi])
                st.p_x[pi] = p_path[idx][0]
                st.p_y[pi] = p_path[idx][1]
                # Skip sweep on first placement (off-board sentinel -99)
                if old_x < 0:
                    continue
                new_x = float(st.p_x[pi])
                new_y = float(st.p_y[pi])
                if old_x == new_x and old_y == new_y:
                    continue
                if not alive_mask.any():
                    continue
                alive_idx = np.where(alive_mask)[0]
                d = _seg_dist_many_points_single_seg(
                    st.f_x[alive_idx], st.f_y[alive_idx],
                    old_x, old_y, new_x, new_y,
                )
                radius = float(st.p_radius[pi])
                caught = d < radius
                for hit_local, ai in zip(caught, alive_idx):
                    if hit_local:
                        combat[pid].append((int(st.f_owner[ai]), int(st.f_ships[ai])))
                        alive_mask[ai] = False

    def _compact_fleets(self, alive_mask: np.ndarray) -> None:
        st = self.state
        F = st.num_fleets()
        if alive_mask.shape[0] != F:
            # Defensive: if sizes diverged (shouldn't), only keep known slots.
            alive_mask = alive_mask[:F]
        if alive_mask.all():
            return
        st.f_id = st.f_id[alive_mask]
        st.f_owner = st.f_owner[alive_mask]
        st.f_x = st.f_x[alive_mask]
        st.f_y = st.f_y[alive_mask]
        st.f_angle = st.f_angle[alive_mask]
        st.f_from_pid = st.f_from_pid[alive_mask]
        st.f_ships = st.f_ships[alive_mask]

    def _remove_planets_by_pid(self, pid_set: set) -> None:
        st = self.state
        if not pid_set or st.num_planets() == 0:
            return
        keep = np.ones(st.num_planets(), dtype=bool)
        for pid in pid_set:
            keep &= st.p_id != pid
        if keep.all():
            return
        st.p_id = st.p_id[keep]
        st.p_owner = st.p_owner[keep]
        st.p_x = st.p_x[keep]
        st.p_y = st.p_y[keep]
        st.p_radius = st.p_radius[keep]
        st.p_ships = st.p_ships[keep]
        st.p_production = st.p_production[keep]
        st.p_init_x = st.p_init_x[keep]
        st.p_init_y = st.p_init_y[keep]

    def _resolve_combat(self, combat: Dict[int, List[Tuple[int, int]]]) -> None:
        """Identical semantics to reference combat resolution."""
        st = self.state
        for pid, events in combat.items():
            if not events:
                continue
            pi = st.planet_index(pid)
            if pi < 0:
                continue

            # Sum ships per owner
            player_ships: Dict[int, int] = {}
            for owner, ships in events:
                player_ships[owner] = player_ships.get(owner, 0) + ships

            if not player_ships:
                continue

            sorted_players = sorted(player_ships.items(), key=lambda item: item[1], reverse=True)
            top_player, top_ships = sorted_players[0]

            if len(sorted_players) > 1:
                second_ships = sorted_players[1][1]
                survivor_ships = top_ships - second_ships
                if top_ships == second_ships:
                    survivor_ships = 0
                survivor_owner = top_player if survivor_ships > 0 else -1
            else:
                survivor_owner = top_player
                survivor_ships = top_ships

            if survivor_ships > 0:
                planet_owner = int(st.p_owner[pi])
                planet_ships = int(st.p_ships[pi])
                if planet_owner == survivor_owner:
                    st.p_ships[pi] = planet_ships + survivor_ships
                else:
                    new_ships = planet_ships - survivor_ships
                    if new_ships < 0:
                        st.p_owner[pi] = survivor_owner
                        st.p_ships[pi] = -new_ships
                    else:
                        st.p_ships[pi] = new_ships

    def _check_terminal(self) -> None:
        st = self.state
        if st.step >= st.config.episode_steps - 2:
            self.done = True

        alive = set()
        for i in range(st.num_planets()):
            o = int(st.p_owner[i])
            if o != -1:
                alive.add(o)
        for i in range(st.num_fleets()):
            alive.add(int(st.f_owner[i]))
        if len(alive) <= 1:
            self.done = True

        if self.done:
            scores = self.scores()
            max_score = max(scores) if scores else 0
            for i in range(self.num_agents):
                if scores[i] == max_score and max_score > 0:
                    self.rewards[i] = 1
                else:
                    self.rewards[i] = -1



# --- inlined: orbitwars/mcts/bokr_widen.py ---

"""BOKR-style kernel regression over UCB values for continuous-angle sub-actions.

Inspired by Ji et al. 2025 (Bayesian Optimized Kernel Regression for
continuous-action MCTS; validated on orbital planning tasks). The idea:

  * Classical progressive widening treats each newly-added angle as a
    fresh arm and tracks per-angle visit/value statistics independently.
    With a 1-second budget we expand ~O(10-50) rollouts per planet —
    not enough to separate signal from noise on 20 candidate angles.
  * Kernel regression shares value across nearby angles via a Gaussian
    kernel `K(θ, θ') = exp(-((θ-θ')/h)^2)`. The estimate at candidate θ
    becomes a weighted average of ALL observations, not just those that
    landed on θ exactly. Small angle perturbations then accumulate
    evidence together — much higher sample efficiency.
  * An exploration bonus on the "effective visit count"
    `n_eff(θ) = sum_i K(θ, θ_i)` gives the UCB term. Angles far from
    prior observations have low n_eff → high bonus → explored next.

Why this fits Orbit Wars specifically:

  * The heuristic emits one analytic intercept angle per target; nearby
    angles (±5-10°) correspond to ships that pass the target on one side
    or the other — materially different trajectories for orbiting
    targets. Pure argmax from heuristic misses this continuous structure.
  * Angles wrap modulo 2π. The kernel here operates on the circular
    angular difference so θ=0 and θ=2π-ε are treated as neighbors.
  * We deliberately keep this a root-level refiner: given a base angle
    from the heuristic, it proposes a fine grid around it and picks
    which grid point MCTS should rollout next. No tree surgery required.

Scope of v1 (this module):

  * Standalone `BOKRKernelSelector` class.
  * Per-planet / per-target lifetime: construct with the analytic
    intercept angle, accumulate (angle, value) observations via
    ``update``, query ``select`` for the next angle to rollout, and
    ``best_angle`` for the final pick.
  * No neural value prior; no GP hyperparameter tuning; no shared
    kernel bandwidth across planets. All can be added later.

Non-goals for v1:

  * Wiring into ``generate_per_planet_moves`` — that requires a dynamic
    candidate set mid-search and is a heavier refactor we'll land after
    this module ships and soaks.
  * Full Bayesian posterior over the value surface. BOKR's original
    formulation uses a GP; we use kernel regression + UCB because the
    inverse-kernel-matrix solve is too expensive under a 1-second budget.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---- Kernel + helpers ---------------------------------------------------

def _angular_diff(a: float, b: float) -> float:
    """Minimum circular difference in radians: always in [0, pi].

    Angles wrap modulo 2pi, so the raw distance `|a-b|` overstates the
    true proximity (e.g. 0 and 2pi - 0.01 are actually 0.01 apart, not
    nearly 2pi apart). Wraps to the smaller of the two arc-lengths.
    """
    d = abs(a - b) % (2.0 * math.pi)
    return d if d <= math.pi else (2.0 * math.pi - d)


def _gaussian_kernel(a: float, b: float, h: float) -> float:
    """Gaussian kernel on circular angular distance, bandwidth `h`.

    `h` controls how much value flows between nearby angles. Small `h`
    (h << grid_spacing) → each angle is nearly independent; large `h`
    → over-smoothing, all angles look identical. Tuning sweet spot is
    `h ~ 0.5 * grid_spacing`.
    """
    d = _angular_diff(a, b) / max(h, 1e-9)
    return math.exp(-d * d)


# ---- Selector -----------------------------------------------------------

@dataclass
class BOKRKernelSelector:
    """Kernel-UCB selector over a fine angle grid around a base angle.

    Usage (per-target at a root decision):

        sel = BOKRKernelSelector(base_angle=analytic_intercept)
        for _ in range(sim_budget):
            theta = sel.select()                          # pick next angle
            value = rollout_at_angle(theta)               # MCTS rollout
            sel.update(theta, value)                      # record result
        final_angle = sel.best_angle()                    # argmax of kernel mean

    Attributes:
        base_angle: center of the grid — typically the heuristic's
            analytic intercept angle for a given target.
        angle_range: radians ± around ``base_angle`` covered by the grid.
            Default 0.2 rad (~11 deg) — wide enough to find a pass-either-
            side improvement, narrow enough that the kernel still shares
            meaningful evidence.
        n_grid: how many grid points inside the range (inclusive of
            both endpoints; ``n_grid`` must be ≥ 1 and is clamped to odd
            so the base angle is always a grid point).
        kernel_h: Gaussian-kernel bandwidth. Default = 0.5 * grid spacing.
        c_ucb: UCB exploration constant. 1.4 mirrors the non-root PUCT
            setting in gumbel_search; pick lower under very noisy
            rollouts (c=0.7) and higher when the value surface is smooth.
        rng_seed: optional; only used to break ties in ``select``.
    """

    base_angle: float
    angle_range: float = 0.2
    n_grid: int = 9
    kernel_h: Optional[float] = None
    c_ucb: float = 1.4
    rng_seed: Optional[int] = None

    # --- Internals ------------------------------------------------------
    # (angle, value) list of observed rollout outcomes.
    _observations: List[Tuple[float, float]] = field(default_factory=list)
    _grid: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.angle_range < 0.0:
            raise ValueError("angle_range must be non-negative")
        if self.n_grid < 1:
            raise ValueError("n_grid must be >= 1")
        # Force odd grid size so base_angle is always a grid point.
        if self.n_grid % 2 == 0:
            self.n_grid += 1
        self._grid = self._build_grid()
        if self.kernel_h is None:
            # Sane default: half the grid spacing (matches the "nearest
            # 1-2 grid points dominate" regime that generally works).
            if self.n_grid >= 2:
                spacing = (2.0 * self.angle_range) / (self.n_grid - 1)
                self.kernel_h = 0.5 * spacing
            else:
                self.kernel_h = 0.1

    def _build_grid(self) -> List[float]:
        """Equally-spaced grid spanning [base - range, base + range]."""
        if self.n_grid == 1:
            return [float(self.base_angle)]
        step = (2.0 * self.angle_range) / (self.n_grid - 1)
        grid = []
        for i in range(self.n_grid):
            theta = self.base_angle - self.angle_range + i * step
            # Wrap into [-pi, pi] for the external contract; kernel is
            # already wrap-aware so this is cosmetic.
            wrapped = ((theta + math.pi) % (2.0 * math.pi)) - math.pi
            grid.append(wrapped)
        return grid

    # --- Public contract ------------------------------------------------

    def candidate_angles(self) -> List[float]:
        """The grid of angles this selector searches over."""
        return list(self._grid)

    def update(self, angle: float, value: float) -> None:
        """Record a rollout outcome at ``angle``. Any angle is accepted
        — callers usually pass grid points, but off-grid observations
        still contribute via the kernel."""
        self._observations.append((float(angle), float(value)))

    def kernel_mean(self, angle: float) -> Tuple[float, float]:
        """Kernel-weighted mean value at ``angle`` and its effective
        visit count. Returns ``(mean, n_eff)``; ``mean=0, n_eff=0`` when
        no observations exist (callers should treat that as "unvisited").
        """
        if not self._observations:
            return (0.0, 0.0)
        num = 0.0
        den = 0.0
        for theta_i, v_i in self._observations:
            w = _gaussian_kernel(angle, theta_i, self.kernel_h)
            num += w * v_i
            den += w
        if den <= 0.0:
            return (0.0, 0.0)
        return (num / den, den)

    def ucb_score(self, angle: float, n_total: int) -> float:
        """Kernel-UCB at ``angle``.

        Formula::

            ucb(theta) = kernel_mean(theta) + c * sqrt(log(n_total) / n_eff(theta))

        Unvisited (n_eff ≈ 0) angles return +inf so ``select`` picks
        them first — matches classical UCB1's "try each arm once" rule
        in the zero-data regime.
        """
        mean, n_eff = self.kernel_mean(angle)
        if n_eff <= 0.0 or n_total <= 0:
            return float("inf")
        # Defensive log: at n_total=1 log is 0 so bonus vanishes; use
        # log(max(n_total, 2)) as is standard in UCB1 implementations.
        bonus = self.c_ucb * math.sqrt(math.log(max(n_total, 2)) / n_eff)
        return mean + bonus

    def select(self) -> float:
        """Return the grid angle with the highest UCB score. Ties
        broken by grid order (deterministic given a seeded rng)."""
        n_total = len(self._observations)
        best_idx = 0
        best_score = -float("inf")
        for i, theta in enumerate(self._grid):
            score = self.ucb_score(theta, n_total)
            # `inf > inf` is False, so a later unvisited arm won't
            # preempt an earlier one — preserves stable order.
            if score > best_score:
                best_score = score
                best_idx = i
        return self._grid[best_idx]

    def best_angle(self) -> float:
        """Post-search pick: grid angle with the highest kernel-mean
        value (no UCB bonus — exploitation only). Falls back to
        ``base_angle`` when no observations have been recorded."""
        if not self._observations:
            return float(self.base_angle)
        best_theta = self._grid[0]
        best_mean = -float("inf")
        for theta in self._grid:
            mean, _ = self.kernel_mean(theta)
            if mean > best_mean:
                best_mean = mean
                best_theta = theta
        return float(best_theta)

    def n_observations(self) -> int:
        return len(self._observations)



# --- inlined: orbitwars/mcts/actions.py ---

"""MCTS action generator for HeuristicAgent-priored search.

For each owned planet we enumerate a small, structured set of candidate
moves (attack each reachable target at a few ship-size fractions, plus a
"hold" no-op), rank them via the heuristic's existing `_score_target`, and
emit the top-K with softmax-normalized priors.

Why this design (v1):
  * Kaggle RTS action spaces are naturally factored: each owned planet
    independently chooses its launch, and the joint is the product. We
    expose the factored shape directly so the Gumbel-AZ root can either
    sample per-planet independently (cheap, good-enough) or sample joint
    top-K over the product (more faithful, Week-3 upgrade).
  * Ship-size fractions `{0.25, 0.5, 1.0}` replace the heuristic's
    one-size-fits-all: MCTS can discover that a smaller probe is
    preferable against strong defenders, or that full-send is optimal
    against cheap neutrals.
  * "hold" is always included — skipping a planet's turn can be optimal
    (e.g. when incoming enemy fleets force a pure defense).

Deliberately skipped in v1:
  * Defensive intercept angles against inbound enemy fleets. The heuristic
    already credits defense via the arrival-table; MCTS sees the same state
    so defense emerges implicitly from rollouts. Intercept moves are on the
    W3 feature list.
  * Sun-tangent re-routes. Currently handled inside
    `heuristic.route_angle_avoiding_sun` — we inherit that behavior via
    `_score_target`. Explicit tangent move variants can be added if needed.

Test coverage (tests/test_mcts_actions.py):
  * Bounds: max_per_planet is respected; ships > available; prior sums to 1.
  * Hold-move is always present.
  * Against a noop-like opponent state, priors rank reachable enemy targets
    above unreachable ones.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple



# Move kinds — used by the search layer to prune / bias at non-root nodes.
KIND_ATTACK_ENEMY    = "attack_enemy"
KIND_ATTACK_NEUTRAL  = "attack_neutral"
KIND_ATTACK_COMET    = "attack_comet"
KIND_REINFORCE_ALLY  = "reinforce_ally"
KIND_HOLD            = "hold"


@dataclass(frozen=True)
class PlanetMove:
    """One candidate move from one owned planet.

    Immutable so callers can stash it in tree nodes without worrying about
    mutation. `prior` is softmax-normalized inside a planet's move list.
    """
    from_pid: int
    angle: float
    ships: int
    target_pid: int          # -1 for hold
    kind: str
    prior: float = 0.0       # populated by generate_per_planet_moves
    raw_score: float = 0.0   # pre-softmax heuristic score (diagnostic)

    @property
    def is_hold(self) -> bool:
        return self.kind == KIND_HOLD

    def to_move(self) -> List:
        """Kaggle-wire format: [from_pid, angle, ships]. HOLD returns []."""
        if self.is_hold:
            return []
        return [int(self.from_pid), float(self.angle), int(self.ships)]


@dataclass
class ActionConfig:
    """Knobs for the action generator.

    max_per_planet: cap on emitted moves per owned planet (incl. hold).
    ship_fractions: discrete send-sizes as fractions of available ships.
    softmax_temperature: higher → flatter prior (more exploration at root).
    min_launch_size: drop moves below this many ships — matches heuristic.

    Angle refinement (BOKR-style):
      angle_refinement_n_grid: per (target, ship-fraction) pair, instead
          of emitting ONE move at the heuristic's analytic intercept,
          emit ``angle_refinement_n_grid`` moves spread ± ``angle_refinement_range``
          radians around it. ``1`` = current behavior (single base angle
          per target). ``3`` = base ± offset (typical BOKR-mini); ``5`` =
          finer grid. Odd values keep the base angle always represented.
          Upper-bounded by the Gumbel root budget: Gumbel will halve
          across whatever top-k arrives, so more grid points just give
          the search more side-pass structure to discover. Keep
          ``max_per_planet`` in mind — grid × target × fraction explodes
          quickly without the top-K trim.
      angle_refinement_range: half-width of the angle grid in radians.
          ~0.1 rad ≈ 5.7° matches Kore 2022's empirical "pass on either
          side" sweet spot for orbital targets. Wider than ~0.2 rad
          starts aiming at nothing in particular.
    """
    max_per_planet: int = 8
    include_hold: bool = True
    ship_fractions: Tuple[float, ...] = (0.25, 0.5, 1.0)
    softmax_temperature: float = 1.0
    min_launch_size: int = 20
    hold_bonus_score: float = 0.0   # added to HOLD raw score before softmax
    angle_refinement_n_grid: int = 1
    angle_refinement_range: float = 0.1


def _softmax(xs: List[float], temperature: float) -> List[float]:
    """Numerically stable softmax. Returns a probability vector."""
    if not xs:
        return []
    t = max(1e-6, float(temperature))
    m = max(xs)
    exps = [math.exp((x - m) / t) for x in xs]
    z = sum(exps)
    if z <= 0:  # pragma: no cover
        return [1.0 / len(xs)] * len(xs)
    return [e / z for e in exps]


def _kind_for_target(po: ParsedObs, tp: List) -> str:
    pid, owner = tp[0], tp[1]
    if pid in po.comet_planet_ids:
        return KIND_ATTACK_COMET
    if owner == po.player:
        return KIND_REINFORCE_ALLY
    if owner == -1:
        return KIND_ATTACK_NEUTRAL
    return KIND_ATTACK_ENEMY


def generate_per_planet_moves(
    po: ParsedObs,
    table: ArrivalTable,
    weights: Optional[Dict[str, float]] = None,
    cfg: Optional[ActionConfig] = None,
) -> Dict[int, List[PlanetMove]]:
    """For each owned planet, return up to cfg.max_per_planet candidate moves.

    Empty list for any planet with no available ships / no reachable target.
    Always includes a HOLD move when cfg.include_hold is True.
    """
    weights = dict(HEURISTIC_WEIGHTS) if weights is None else dict(weights)
    cfg = cfg or ActionConfig()

    out: Dict[int, List[PlanetMove]] = {}
    for mp in po.my_planets:
        mpid = int(mp[0])
        available = int(mp[5])

        # Enumerate raw candidates first; softmax over them at the end.
        raw: List[Tuple[float, PlanetMove]] = []

        if cfg.include_hold:
            hold = PlanetMove(
                from_pid=mpid, angle=0.0, ships=0, target_pid=-1,
                kind=KIND_HOLD, prior=0.0, raw_score=cfg.hold_bonus_score,
            )
            raw.append((cfg.hold_bonus_score, hold))

        if available < cfg.min_launch_size:
            # Only HOLD is possible — emit it and move on.
            if raw:
                raw[0][1].__dict__  # no-op (frozen dataclass: can't mutate)
                moves = [PlanetMove(
                    from_pid=raw[0][1].from_pid, angle=0.0, ships=0,
                    target_pid=-1, kind=KIND_HOLD, prior=1.0,
                    raw_score=raw[0][0],
                )]
                out[mpid] = moves
            else:
                out[mpid] = []
            continue

        for tp in po.planets:
            tpid = int(tp[0])
            if tpid == mpid:
                continue
            ip = po.initial_planet_by_id.get(tpid, tp)
            kind = _kind_for_target(po, tp)

            for frac in cfg.ship_fractions:
                ships = max(cfg.min_launch_size, int(available * frac))
                if ships > available:
                    continue
                if ships < cfg.min_launch_size:
                    continue

                score, angle, _proj = _score_target(
                    mp, tp, ip, po, table, weights, ships,
                )
                if not math.isfinite(score):
                    continue

                # Emit angle variants around the heuristic's analytic
                # intercept. All variants share the base raw_score
                # because the score is ~angle-invariant at this scale —
                # side-pass discovery is MCTS's job during search.
                # n_grid=1 preserves the legacy single-angle behavior.
                if cfg.angle_refinement_n_grid > 1:
                    sel = BOKRKernelSelector(
                        base_angle=float(angle),
                        angle_range=float(cfg.angle_refinement_range),
                        n_grid=int(cfg.angle_refinement_n_grid),
                    )
                    variant_angles = sel.candidate_angles()
                else:
                    variant_angles = [float(angle)]
                for var_angle in variant_angles:
                    move = PlanetMove(
                        from_pid=mpid, angle=float(var_angle),
                        ships=int(ships), target_pid=tpid, kind=kind,
                        prior=0.0, raw_score=score,
                    )
                    raw.append((score, move))

        # Rank descending by raw_score, keep top-K.
        raw.sort(key=lambda t: t[0], reverse=True)
        raw = raw[: cfg.max_per_planet]

        # Softmax priors.
        scores = [s for (s, _) in raw]
        priors = _softmax(scores, cfg.softmax_temperature)
        out[mpid] = [
            PlanetMove(
                from_pid=m.from_pid, angle=m.angle, ships=m.ships,
                target_pid=m.target_pid, kind=m.kind,
                prior=p, raw_score=m.raw_score,
            )
            for (_, m), p in zip(raw, priors)
        ]

    return out


# ---- Joint-action helpers (used by the root sampler in gumbel_search.py) ----

@dataclass(frozen=True)
class JointAction:
    """One joint per-turn action: a tuple of PlanetMove (one per owned planet).

    The order is stable by planet ID for deterministic hashing inside the
    tree.
    """
    moves: Tuple[PlanetMove, ...]

    def to_wire(self) -> List[List]:
        """Kaggle-wire format. Drops HOLD moves."""
        return [m.to_move() for m in self.moves if not m.is_hold]

    def joint_prior(self) -> float:
        """Product of per-planet priors (independent sampling assumption)."""
        p = 1.0
        for m in self.moves:
            p *= max(m.prior, 1e-9)
        return p


def sample_joint(
    per_planet: Dict[int, List[PlanetMove]],
    rng,  # random.Random — typed loosely to avoid stdlib import cycles
) -> JointAction:
    """Independently sample one PlanetMove from each planet's prior dist."""
    picks: List[PlanetMove] = []
    for pid in sorted(per_planet.keys()):
        moves = per_planet[pid]
        if not moves:
            continue
        weights = [max(m.prior, 0.0) for m in moves]
        total = sum(weights)
        if total <= 0:
            picks.append(moves[0])
            continue
        r = rng.uniform(0.0, total)
        acc = 0.0
        for m, w in zip(moves, weights):
            acc += w
            if r <= acc:
                picks.append(m)
                break
        else:
            picks.append(moves[-1])
    return JointAction(tuple(picks))



# --- inlined: orbitwars/mcts/gumbel_search.py ---

"""Gumbel top-k + Sequential Halving MCTS for Orbit Wars.

v1 scope: **root-only** search. Each candidate joint action is scored by
short heuristic rollouts in a cloned FastEngine. Sequential Halving
(Danihelka et al., ICLR 2022) concentrates the sim budget on the
promising candidates without the overhead of a full tree.

Why this shape matters:
  * At a 1 s CPU budget we expect O(10-100) rollouts per turn — not
    enough to build a meaningful tree, but plenty to rank ~8 candidate
    joint actions via policy improvement at the root. This is exactly
    the regime the Gumbel paper addresses.
  * Heuristic rollouts give a reliable value estimate; the heuristic is
    already close to competent, so value noise is low relative to naive
    MCTS default-policy rollouts.
  * Sequential Halving is *simple-regret-optimal* under fixed budget and
    noisy values — the right objective for root action selection (we
    care about picking the best action, not estimating all Q's well).

Deliberately out of v1:
  * Non-root PUCT tree — needed only once rollouts > ~200 sims/turn.
  * BOKR kernel over continuous angles — our action generator already
    picks an analytic angle per target, so the continuous dimension is
    collapsed at the root. Re-introduce if MCTS wants to search around
    that analytic angle.
  * Decoupled UCT at simultaneous-move nodes — meaningless for a
    root-only search. Arrives alongside non-root expansion in W3.

Integration:
  * `GumbelRootSearch.search(obs, my_player)` → `SearchResult` with the
    chosen `JointAction` and per-candidate Q/visit diagnostics.
  * The hot loop in `_rollout_value` clones the engine state per sim so
    the true state isn't mutated. FastEngine mutates numpy arrays in
    place; `copy.deepcopy(state)` gives us an independent copy cheaply
    (~tens of μs for typical state sizes).
  * A hard wall-clock deadline aborts mid-round and returns whatever has
    been staged. Timeouts forfeit matches — we never cross that line.
"""

import copy
import math
import random
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple



# ---------------------------- Config --------------------------------------

@dataclass
class GumbelConfig:
    """Knobs for Gumbel Sequential Halving.

    num_candidates:  how many joint actions to propose at the root. 8-24
                     is the sweet spot: fewer → SH halving collapses too
                     fast; more → sim budget spreads thin.
    total_sims:      rollout budget for the whole search.
    rollout_depth:   plies simulated per rollout. Short (4-8) keeps
                     per-rollout cost bounded; heuristic value at the
                     horizon does most of the lifting.
    hard_deadline_ms: abort and return best-so-far past this wall time.
                     Kept conservative — we'd rather submit a weaker
                     action than forfeit the match.
    anchor_improvement_margin: minimum Q gap (winner - anchor) required
                     before we override the anchor candidate. With short
                     rollouts and few sims, per-candidate Q has noise SE
                     of ~0.2 — so we need the winner to beat the anchor
                     by *at least* this margin to trust the result.
                     Effectively: MCTS never plays a move that search
                     isn't confidently better than the heuristic's pick.

                     Empirical note (seed=42, 500 turns, vs HeuristicAgent):
                       margin=0.15: MCTS LOSES 692-1525 (noise overrides anchor)
                       margin=0.30: MCTS barely wins 1146-1057
                       margin=0.50: MCTS wins 1356-874 (default)
                       margin=10.0 (forced anchor): MCTS wins 1675-698
                     The search is currently net-negative at low sim
                     budgets — until we widen rollouts/sims/priors, the
                     0.5 floor is the sweet spot.
    """
    # Tuned defaults (W2, post-RNG-isolation + multi-seed verification):
    #
    # Empirical multi-seed sweep (MCTS vs Heuristic, both seats, seeds
    # [42, 123, 7]) with margin=0.5, sims=32, depth=15 showed 2W/4L —
    # wall-clock variance causes some turns to hit HARD_DEADLINE_MS and
    # fall back to the heuristic, while other turns use search output.
    # Those branching decisions cascade into materially different games,
    # and at low sim budgets search-output < heuristic-output more often
    # than it's better.
    #
    # Until we have a proper neural prior (W4-5) or enough sims to drop
    # rollout variance (not currently feasible under 1s CPU), the safe
    # default is margin=2.0 — search effectively always defers to the
    # heuristic anchor. This locks in the Path A floor. Search still
    # runs and its statistics are exposed in SearchResult for diagnostics
    # and future tuning, but the returned wire action is the heuristic's
    # unless a candidate beats it by an unusually clear margin.
    num_candidates: int = 4
    total_sims: int = 32
    rollout_depth: int = 15
    hard_deadline_ms: float = 300.0
    anchor_improvement_margin: float = 2.0
    # Rollout policy — "heuristic" uses HeuristicAgent (slow but
    # strategic; ~18 ms/call, fits <1 full rollout at the default
    # deadline), "fast" uses FastRolloutAgent (~0.1-0.5 ms/call, ~30-50×
    # faster; rollouts drop from ~560 ms to ~20-30 ms, unlocking real
    # policy improvement at the same budget). Default is kept at
    # "heuristic" to preserve the shipped mcts_v1 bot's behavior
    # byte-for-byte; switch via config for A/B and future defaults.
    rollout_policy: str = "heuristic"
    # Simultaneous-move root decoupling (Path B / W3). When True, the
    # root treats my + opp action selection as a 2D decoupled bandit
    # (see orbitwars.mcts.sim_move.decoupled_ucb_root). The opp
    # candidate pool is drawn from the posterior-biased heuristic — so
    # when the Bayesian posterior has concentrated, MCTS marginalizes
    # over the top archetypes' responses instead of assuming a single
    # deterministic opp heuristic. Default False — the core improvement
    # only shows up once the posterior has evidence to concentrate on,
    # and the 2D bandit doubles arity at fixed total_sims so it's a
    # no-op-to-loss on turn-0-heavy matches. Flag it on once paired
    # with (b) posterior caching.
    use_decoupled_sim_move: bool = False
    # Variant of the decoupled-root bandit to use when
    # ``use_decoupled_sim_move`` is True. Options:
    #   "ucb"   — decoupled_ucb_root (default, shipped in v7/v8 as the
    #             principled sim-move fix; UCB exploration bonus + mean-Q
    #             argmax over warm-started rollouts).
    #   "exp3"  — decoupled_exp3_root (flag-gated W4 A/B test per plan
    #             §W4: "Regret-matching A/B test at sim-move nodes; ship
    #             if beats decoupled-UCT by ≥5pp"). Exp3 is minimax-
    #             optimal for adversarial bandits — the theoretically
    #             correct choice when the opp is non-stationary, as it is
    #             on the ladder where archetypes vary by seat. Same
    #             anchor-protection contract as ucb via ``protected_my_idx``.
    # Both variants fall through to sequential_halving when there are
    # <2 opp candidates (the posterior-sampled pool couldn't supply
    # enough distinct opps).
    sim_move_variant: str = "ucb"
    # Exp3 learning rate — only used when sim_move_variant="exp3".
    # 0.3 is safe for [-1, +1] rewards and budgets in the 16-128 range
    # (matches sim_move.decoupled_exp3_root default); tune if A/B wants.
    exp3_eta: float = 0.3
    # Number of opp candidate actions to sample when decoupling is on.
    # Typical: 2-3. Larger K = better marginalization, worse per-cell
    # noise under fixed total_sims. 2 is the minimum where decoupling
    # is even meaningful (1 opp candidate degenerates to the baseline).
    num_opp_candidates: int = 2
    # Per-rollout wall-clock cap, in milliseconds. ``_rollout_value``
    # enforces ``min(hard_stop_at, rollout_start + per_rollout_budget)``
    # as its inner deadline, so no single rollout can blow past the
    # whole search budget. Measured fat tail on step-35-ish states:
    # natural rollout cost has p50 ~0.1 ms (many rollouts end at
    # eng.done) but max ~685 ms in 200 samples (see
    # tools/diag_rollout_deadline). Without the cap, a single unlucky
    # rollout eats the entire ``hard_deadline_ms`` window and the
    # overall act() can overshoot 1 s \u2014 a Kaggle-forfeit risk.
    # 150 ms is ~2\u00d7 the natural median for the heavy-state regime and
    # leaves room for n_sim \u2265 2 under the 300 ms default deadline.
    per_rollout_budget_ms: float = 150.0


# ---------------------------- Gumbel top-k --------------------------------

def gumbel_topk(
    priors: List[float], k: int, rng: random.Random,
) -> List[Tuple[int, float]]:
    """Sample up to k indices without replacement via the Gumbel trick.

    For each prior p_i draw g_i ~ Gumbel(0) and score = log(p_i) + g_i.
    Top-k by score is exactly a sample-without-replacement from the
    categorical distribution `pi ∝ p_i`. This is the root-level
    proposal mechanism the Gumbel-AZ paper uses (Danihelka eq. 1).

    Returns `[(index, score), ...]` sorted by descending score. Priors
    ≤ 0 are treated as ineligible (log(0) is -inf, never sampled).
    """
    eps = 1e-20
    scored: List[Tuple[int, float]] = []
    for i, p in enumerate(priors):
        if p <= 0.0:
            continue
        u = rng.random()
        g = -math.log(-math.log(max(u, eps)) + eps)
        scored.append((i, math.log(p) + g))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:k]


# ---------------------------- Joint enumeration ---------------------------

def enumerate_joints(
    per_planet: Dict[int, List[PlanetMove]],
    n_samples: int,
    rng: random.Random,
) -> List[JointAction]:
    """Sample n_samples distinct joint actions from independent planet priors.

    De-dup by wire-format signature so we don't waste rollouts on
    identical candidates. On tiny search spaces (few owned planets with
    few moves each) we may return fewer than n_samples — that's fine,
    SH handles variable widths.
    """
    seen: set = set()
    out: List[JointAction] = []
    budget = max(n_samples * 6, 16)
    for _ in range(budget):
        if len(out) >= n_samples:
            break
        j = sample_joint(per_planet, rng)
        key = tuple(tuple(m) for m in j.to_wire())
        if key in seen:
            continue
        seen.add(key)
        out.append(j)
    return out


# ---------------------------- Rollout -------------------------------------

def _obs_to_namespace(obs: Any) -> Any:
    """Convert dict obs → SimpleNamespace so FastEngine.from_official_obs works.

    FastEngine reads obs via `getattr(...)` which returns defaults for
    dicts. Kaggle passes dicts in ladder play; our tests use dicts too.
    Cheap one-shot shim.
    """
    if isinstance(obs, dict):
        return SimpleNamespace(**obs)
    return obs


def _score_from_engine(
    eng: FastEngine, my_player: int, num_agents: int,
) -> float:
    """Map an end-of-rollout game state to a scalar value in [-1, +1].

    Terminal games use the reward (winner → +1, others → -1). Non-
    terminal returns `(my_ships - best_opp_ships) / total_ships`,
    capturing lead without being fooled by ship-inflation vs a weak
    opponent. Clipped to [-1, +1].
    """
    if eng.done:
        r = eng.rewards[my_player]
        return float(r) if isinstance(r, (int, float)) else 0.0
    scores = eng.scores()
    my_s = float(scores[my_player])
    opp_best = float(max(
        (scores[i] for i in range(num_agents) if i != my_player), default=0.0
    ))
    total = my_s + opp_best + 1.0
    v = (my_s - opp_best) / total
    return max(-1.0, min(1.0, v))


def _rollout_value(
    base_state: GameState,
    my_player: int,
    my_action: List[List],
    opp_agent_factory: Callable[[], Any],
    my_future_factory: Callable[[], Any],
    depth: int,
    num_agents: int = 2,
    rng: Optional[random.Random] = None,
    deadline_fn: Optional[Callable[[], bool]] = None,
    opp_turn0_action: Optional[List[List]] = None,
    hard_stop_at: Optional[float] = None,
    per_rollout_budget_ms: Optional[float] = None,
) -> float:
    """Simulate `depth` plies from a cloned state; return scalar value.

    Turn 0 uses the candidate root action for `my_player`. The opp
    turn-0 action is either ``opp_turn0_action`` (if supplied — e.g. a
    pre-computed archetype pick in decoupled sim-move search) or the
    result of ``opp_agent_factory().act()`` (the default heuristic).
    Subsequent turns use fresh heuristic instances on both sides.

    Fresh instances because HeuristicAgent carries per-match state
    (`_state.last_launch_turn`) that shouldn't leak across rollouts.

    The `rng` is forwarded into the rollout engine for comet-ship
    sizing so rollouts are reproducible given the search seed. If
    None, each FastEngine seeds its own RNG from os.urandom — which
    makes the search nondeterministic across runs.

    ``deadline_fn`` (optional) is polled *between plies*. When it
    returns True we abort the rollout and score the engine state as-
    is. This bounds a single rollout's wall cost to roughly "one ply
    above the deadline" — critical for the 1-s Kaggle turn ceiling
    because a late-game HeuristicAgent ply can take 30-100 ms and
    unchecked rollouts have been observed to blow past 900 ms.

    ``hard_stop_at`` (optional, absolute ``time.perf_counter()``
    seconds) propagates the outer search deadline into each inner
    ``agent.act()`` call via ``Deadline(hard_stop_at=...)``. Without
    this, an in-flight ``HeuristicAgent.act`` on a dense mid-game
    state (profile: 400-700 ms) can overshoot the search deadline by
    hundreds of ms. With it, heuristic agents short-circuit inside
    ``_plan_moves`` as soon as the outer deadline fires, bounding a
    single ply's overshoot to the time needed to detect + return.

    ``per_rollout_budget_ms`` (optional) imposes an *additional*,
    per-rollout deadline on top of ``hard_stop_at``. The inner
    effective deadline is ``min(hard_stop_at, now + per_rollout_budget)``.
    This guards against the fat-tail case observed in diag_rollout_deadline:
    while the bulk of rollouts finish in <200 ms at depth=15, one in
    every ~200 naturally runs 600-700 ms (state where the heuristic
    walks every reachable target on every ply). Without this bound, a
    single unlucky rollout early in a search can consume the whole
    ``hard_stop_at`` window, leaving later rollouts with zero budget
    AND blowing past the outer MCTS deadline. The per-rollout cap is
    what keeps p99.something bounded, not just p95.
    """
    # Compute the effective inner deadline. Every subsequent
    # ``Deadline(hard_stop_at=...)`` and deadline check uses this
    # tighter value — the outer search deadline (hard_stop_at) still
    # wins when it's closer, but per_rollout_budget_ms caps the worst-
    # case single rollout.
    effective_stop: Optional[float] = hard_stop_at
    if per_rollout_budget_ms is not None:
        rollout_cap = time.perf_counter() + per_rollout_budget_ms / 1000.0
        if effective_stop is None or rollout_cap < effective_stop:
            effective_stop = rollout_cap

    # Build an inner deadline_fn that respects the per-rollout cap
    # even if the outer caller only passed a global deadline_fn.
    inner_deadline_fn: Optional[Callable[[], bool]]
    if effective_stop is not None:
        _stop = effective_stop  # capture

        def inner_deadline_fn() -> bool:  # noqa: E306
            return time.perf_counter() >= _stop
    else:
        inner_deadline_fn = deadline_fn

    eng = FastEngine(
        copy.deepcopy(base_state),
        num_agents=num_agents,
        rng=rng,
    )

    # Late deadline check: sequential_halving's pre-rollout gate catches
    # "deadline fired before this rollout starts", but we can still
    # have fired by the time deepcopy + FastEngine init complete — AND
    # the single `opp.act()` call below on dense mid-game states runs
    # 100-300 ms, which is the observed source of the remaining tail
    # (audit pass 3: max 1190 ms vs 900 ms ceiling). Short-circuit here
    # so the in-flight rollout costs ~deepcopy (~1 ms) instead of a full
    # turn-0. This caps the observed overshoot from ~300 ms to ~5 ms.
    if inner_deadline_fn is not None and inner_deadline_fn():
        return _score_from_engine(eng, my_player, num_agents)

    # Turn 0: my root action + opp's turn-0 response.
    # If the caller pre-computed opp's turn-0 (the decoupled sim-move
    # path passes one opp candidate per rollout), skip the heuristic
    # call entirely — saves 100-300 ms per rollout on dense states.
    actions: List[Optional[List]] = [None] * num_agents
    actions[my_player] = my_action
    for i in range(num_agents):
        if i == my_player:
            continue
        if opp_turn0_action is not None:
            actions[i] = opp_turn0_action
        else:
            opp = opp_agent_factory()
            actions[i] = opp.act(
                eng.observation(i), Deadline(hard_stop_at=effective_stop),
            )
    eng.step(actions)

    # Turns 1..depth-1: heuristic on both sides. Abort between plies
    # if the deadline has fired — the cost of an extra ply is unbounded
    # (HeuristicAgent's fleet-arrival table scales with fleet count)
    # so we pay the check on every ply, not just every rollout.
    for _ in range(max(0, depth - 1)):
        if eng.done:
            break
        if inner_deadline_fn is not None and inner_deadline_fn():
            break
        actions = [None] * num_agents
        for i in range(num_agents):
            factory = my_future_factory if i == my_player else opp_agent_factory
            agent = factory()
            actions[i] = agent.act(
                eng.observation(i), Deadline(hard_stop_at=effective_stop),
            )
        eng.step(actions)

    return _score_from_engine(eng, my_player, num_agents)


# ---------------------------- Sequential Halving --------------------------

@dataclass
class SearchResult:
    best_joint: JointAction
    n_rollouts: int
    duration_ms: float
    q_values: List[float] = field(default_factory=list)
    visits: List[int] = field(default_factory=list)
    aborted: bool = False

    @property
    def n_candidates(self) -> int:
        return len(self.q_values)


def sequential_halving(
    candidates: List[JointAction],
    rollout_fn: Callable[[JointAction], float],
    cfg: GumbelConfig,
    start_time: Optional[float] = None,
    protected_idx: Optional[int] = None,
) -> SearchResult:
    """Sequential Halving: iteratively rollout the active set and halve it.

    Rounds ≈ ceil(log2(k)); each round gives all active candidates the
    same per-round sim allocation. At round end, the bottom half (by
    mean Q) is pruned. Ends with one candidate; the highest mean Q
    across all visited candidates is returned. Aborts mid-round if the
    wall-clock deadline is reached.

    protected_idx (if given) is kept in `active` across ALL rounds —
    used for an anchor/heuristic candidate we want to guarantee low-
    variance Q estimates for. It still competes on mean-Q for the final
    pick; we just don't let SH prune it under rollout noise.
    """
    t0 = start_time if start_time is not None else time.perf_counter()
    k = len(candidates)
    if k == 0:
        raise ValueError("sequential_halving: no candidates")

    q_sum = [0.0] * k
    visits = [0] * k
    deadline = t0 + cfg.hard_deadline_ms / 1000.0

    if k == 1:
        # One candidate — still do one rollout for a diagnostic Q value,
        # but only if we have any budget at all.
        if time.perf_counter() < deadline and cfg.total_sims > 0:
            v = rollout_fn(candidates[0])
            q_sum[0] = v
            visits[0] = 1
        return SearchResult(
            best_joint=candidates[0],
            n_rollouts=visits[0],
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            q_values=[q_sum[0]],
            visits=list(visits),
            aborted=False,
        )

    active = list(range(k))
    n_rounds = max(1, math.ceil(math.log2(k)))
    sims_per_round = max(len(active), cfg.total_sims // n_rounds)

    total_rollouts = 0
    aborted = False

    for _ in range(n_rounds):
        if len(active) <= 1:
            break
        sims_each = max(1, sims_per_round // len(active))
        for idx in active:
            for _ in range(sims_each):
                if time.perf_counter() > deadline:
                    aborted = True
                    break
                v = rollout_fn(candidates[idx])
                q_sum[idx] += v
                visits[idx] += 1
                total_rollouts += 1
            if aborted:
                break
        if aborted:
            break
        # Halve — keep the better half by mean Q (ties broken by index).
        # protected_idx, if given, is always sorted to the top so it
        # survives pruning for another round of sims.
        def _sort_key(i: int) -> Tuple[int, float, int]:
            is_protected = 1 if (protected_idx is not None and i == protected_idx) else 0
            mean_q = q_sum[i] / max(1, visits[i])
            return (is_protected, mean_q, -i)

        active.sort(key=_sort_key, reverse=True)
        keep = max(1, len(active) // 2)
        active = active[:keep]

    # Final choice: highest mean Q across ALL visited candidates. A
    # pruned-early candidate may still hold the best running mean.
    def _mean_q(i: int) -> float:
        return q_sum[i] / visits[i] if visits[i] > 0 else -math.inf

    best_i = max(range(k), key=_mean_q)
    q_avg = [_mean_q(i) for i in range(k)]

    return SearchResult(
        best_joint=candidates[best_i],
        n_rollouts=total_rollouts,
        duration_ms=(time.perf_counter() - t0) * 1000.0,
        q_values=q_avg,
        visits=list(visits),
        aborted=aborted,
    )


# ---------------------------- Anchor joint --------------------------------

def _build_anchor_joint(
    anchor_wire: Optional[List[List]],
    per_planet: Dict[int, List[PlanetMove]],
) -> Optional[JointAction]:
    """Convert a wire-format action (heuristic's pick) into a JointAction.

    Returns None if `anchor_wire` is None or per_planet is empty. Builds
    one PlanetMove per owned planet in the same stable order as
    `sample_joint` (sorted by pid), so Gumbel samples and anchors share
    a comparable key space.

    The target_pid/kind fields on an anchor's non-HOLD entries are set
    conservatively (KIND_ATTACK_ENEMY, target_pid=-1). They only affect
    diagnostics; wire output depends on kind != KIND_HOLD and on
    (from_pid, angle, ships), all of which are faithful.
    """
    if not per_planet:
        return None
    if anchor_wire is None:
        return None
    wire_by_pid: Dict[int, Any] = {}
    for m in anchor_wire:
        if isinstance(m, (list, tuple)) and len(m) >= 3:
            try:
                wire_by_pid[int(m[0])] = m
            except Exception:
                continue
    moves: List[PlanetMove] = []
    for pid in sorted(per_planet.keys()):
        w = wire_by_pid.get(pid)
        if w is None:
            moves.append(PlanetMove(
                from_pid=pid, angle=0.0, ships=0, target_pid=-1,
                kind=KIND_HOLD, prior=1.0,
            ))
        else:
            try:
                angle = float(w[1])
                ships = int(w[2])
            except Exception:
                moves.append(PlanetMove(
                    from_pid=pid, angle=0.0, ships=0, target_pid=-1,
                    kind=KIND_HOLD, prior=1.0,
                ))
                continue
            moves.append(PlanetMove(
                from_pid=pid, angle=angle, ships=ships, target_pid=-1,
                kind=KIND_ATTACK_ENEMY, prior=1.0,
            ))
    return JointAction(moves=tuple(moves))


# ---------------------------- Top-level search ----------------------------

@dataclass
class GumbelRootSearch:
    """Glue: obs + action generator + rollout + SH.

    Construct once per agent; call `search(obs, my_player)` each turn.
    Deterministic when `rng_seed` is set.

    Opponent-model override:
      ``opp_policy_override`` — if set, called to build the opponent's
      rollout agent each rollout-step instead of the default
      ``HeuristicAgent(weights=self.weights)``. MCTSAgent sets this from
      the Bayesian posterior's most-likely archetype when the posterior
      has concentrated, so MCTS searches under the correct opponent
      model rather than "assume opp is a default heuristic".
    """
    weights: Dict[str, float] = field(default_factory=lambda: dict(HEURISTIC_WEIGHTS))
    action_cfg: ActionConfig = field(default_factory=ActionConfig)
    gumbel_cfg: GumbelConfig = field(default_factory=GumbelConfig)
    rng_seed: Optional[int] = None
    opp_policy_override: Optional[Callable[[], Any]] = None
    # Decoupled sim-move root (Path B / W3). When set, called each turn
    # with (obs, opp_player) to produce a list of candidate opp wire
    # actions. If the list has >=2 distinct actions and
    # ``gumbel_cfg.use_decoupled_sim_move`` is True, search runs the
    # decoupled UCB bandit from sim_move.py instead of sequential_halving.
    # Typically populated by MCTSAgent from the Bayesian posterior's
    # top-K archetypes when the posterior has concentrated.
    opp_candidate_builder: Optional[Callable[[Any, int], List[List[List]]]] = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.rng_seed)

    def _opp_factory(self) -> Any:
        # Priority 1: Bayesian posterior override (Path D). When the
        # posterior has concentrated on a specific archetype, MCTSAgent
        # sets this so rollouts play against that archetype's heuristic.
        # Keep this path even under rollout_policy="fast" — exploitation
        # signal beats raw rollout speed once the posterior has fired.
        if self.opp_policy_override is not None:
            return self.opp_policy_override()
        # Priority 2: fast rollout policy. Cheap nearest-target push —
        # 30-50× faster than the full heuristic. See fast_rollout.py.
        if self.gumbel_cfg.rollout_policy == "fast":
            return FastRolloutAgent(
                min_launch_size=int(self.weights.get("min_launch_size", 20)),
                send_fraction=float(self.weights.get("max_launch_fraction", 0.8)),
            )
        # Default: full HeuristicAgent (shipped mcts_v1 behavior).
        return HeuristicAgent(weights=self.weights)

    def _my_future_factory(self) -> Any:
        # Symmetric fast path: my-future rollout plies also swap in the
        # cheap agent when rollout_policy="fast". Candidate turn-0
        # action is unaffected (that's already built upstream).
        if self.gumbel_cfg.rollout_policy == "fast":
            return FastRolloutAgent(
                min_launch_size=int(self.weights.get("min_launch_size", 20)),
                send_fraction=float(self.weights.get("max_launch_fraction", 0.8)),
            )
        return HeuristicAgent(weights=self.weights)

    def search(
        self, obs: Any, my_player: int, num_agents: int = 2,
        start_time: Optional[float] = None,
        anchor_action: Optional[List[List]] = None,
        outer_hard_stop_at: Optional[float] = None,
        step_override: Optional[int] = None,
    ) -> Optional[SearchResult]:
        """Run search for one turn. Returns None if no legal moves exist.

        If `anchor_action` is given (the heuristic's wire pick), it is
        prepended to the candidate list as a protected candidate — SH
        never prunes it. This turns MCTSAgent into a guaranteed floor:
        if search can't beat the heuristic, we return the heuristic's
        action.

        ``outer_hard_stop_at`` (optional, absolute ``time.perf_counter()``
        seconds): an EXTERNAL ceiling from the caller (typically
        MCTSAgent's outer Deadline). The rollout and SH deadlines are
        internally capped at ``min(own_deadline, outer_hard_stop_at)``
        so search cannot run past the caller's turn budget even if
        safe_budget math upstream was loose. This is the
        belt-and-suspenders guard that converts audit outliers (e.g.
        985 ms on a 900 ms ceiling) into bounded 880 ms worst case.
        Without it, the search's `hard_deadline_ms` is relative-to-
        start and has no notion of "the outer turn-budget has already
        been eaten by a slow pre-search". This parameter closes that
        gap.
        """

        po = parse_obs(obs, step_override=step_override)
        table = ArrivalTable()
        try:
            # build_arrival_table updates state in place on an ArrivalTable
            # or returns one; use the functional form for safety.
            table = build_arrival_table(po)
        except Exception:
            # Empty table fallback — scores still evaluate, just no
            # arrival-aware sizing.
            pass

        per_planet = generate_per_planet_moves(
            po, table, weights=self.weights, cfg=self.action_cfg,
        )
        # No owned planets at all → nothing to decide. Signal upstream
        # with None so callers can short-circuit cleanly (vs. returning a
        # degenerate empty-wire SearchResult that reads like a real
        # "hold" choice).
        if not per_planet:
            return None

        # Build the anchor joint (heuristic's pick) if provided. We'll
        # insert it as candidate 0 and keep it protected from SH pruning
        # so it accrues visits in every round and gets a low-variance Q.
        anchor_joint = _build_anchor_joint(anchor_action, per_planet)
        anchor_key: Optional[tuple] = None
        if anchor_joint is not None:
            anchor_key = tuple(tuple(m) for m in anchor_joint.to_wire())

        # Sample Gumbel candidates. We leave one slot for the anchor so
        # the total effective candidate count stays ~num_candidates.
        sample_budget = self.gumbel_cfg.num_candidates - (1 if anchor_joint else 0)
        sample_budget = max(sample_budget, 1)
        sampled = enumerate_joints(per_planet, sample_budget, self._rng)

        # Compose the final candidate list: anchor first (if any), then
        # Gumbel samples that don't duplicate it.
        joints: List[JointAction] = []
        if anchor_joint is not None:
            joints.append(anchor_joint)
        for j in sampled:
            key = tuple(tuple(m) for m in j.to_wire())
            if key == anchor_key:
                continue
            joints.append(j)

        if not joints:
            return None
        if len(joints) == 1:
            return SearchResult(
                best_joint=joints[0], n_rollouts=0, duration_ms=0.0,
                q_values=[0.0], visits=[0], aborted=False,
            )

        # Build base state from obs once; rollouts deepcopy it per sim.
        eng = FastEngine.from_official_obs(
            _obs_to_namespace(obs), num_agents=num_agents,
        )
        base_state = eng.state

        # Per-rollout deadline: SH's own deadline ∩ caller's outer hard
        # stop. When the wall-clock overshoots, `_rollout_value` short-
        # circuits between plies and returns the mid-rollout engine
        # score. This caps a single rollout's over-deadline overshoot
        # to ~one ply instead of "all remaining plies at worst-case
        # heuristic cost".
        #
        # The ∩ with outer_hard_stop_at is the load-bearing audit fix:
        # without it, a slow pre-search that eats the turn budget still
        # hands the full hard_deadline_ms window to SH, and SH's
        # in-flight rollout can push total turn time past the outer
        # actTimeout. With it, SH naturally runs less when the budget
        # was already consumed upstream, and the overall turn time is
        # bounded by the outer ceiling regardless of pre-search cost.
        t0_rollout = start_time if start_time is not None else time.perf_counter()
        rollout_deadline_sec = t0_rollout + self.gumbel_cfg.hard_deadline_ms / 1000.0
        if outer_hard_stop_at is not None and outer_hard_stop_at < rollout_deadline_sec:
            rollout_deadline_sec = outer_hard_stop_at
        def _rollout_deadline_fired() -> bool:
            return time.perf_counter() > rollout_deadline_sec

        def rollout_fn(joint: JointAction) -> float:
            return _rollout_value(
                base_state=base_state,
                my_player=my_player,
                my_action=joint.to_wire(),
                opp_agent_factory=self._opp_factory,
                my_future_factory=self._my_future_factory,
                depth=self.gumbel_cfg.rollout_depth,
                num_agents=num_agents,
                rng=self._rng,  # deterministic rollouts given search seed
                deadline_fn=_rollout_deadline_fired,
                hard_stop_at=rollout_deadline_sec,
                per_rollout_budget_ms=self.gumbel_cfg.per_rollout_budget_ms,
            )

        protected_idx = 0 if anchor_joint is not None else None

        # --- Decoupled sim-move branch -----------------------------------
        # When the posterior has concentrated enough that MCTSAgent
        # populates `opp_candidate_builder`, and the decoupled flag is on,
        # run the 2D UCB bandit over (my_joint, opp_wire) instead of
        # sequential_halving. The bandit marginalizes over the opp's
        # posterior-weighted strategies — honest scoring under sim-move
        # uncertainty. Only fires when there are >=2 distinct opp
        # candidates (1 candidate degenerates to the baseline).
        opp_wires: List[List[List]] = []
        if (
            self.gumbel_cfg.use_decoupled_sim_move
            and self.opp_candidate_builder is not None
            and num_agents == 2
        ):
            try:
                # 2-player only for v1: opp is the other seat.
                opp_player = 1 - my_player
                opp_wires = list(self.opp_candidate_builder(obs, opp_player) or [])
                # Deduplicate by wire signature so we don't waste rollouts
                # on identical opp responses.
                seen_opp: set = set()
                deduped: List[List[List]] = []
                for w in opp_wires:
                    try:
                        key = tuple(tuple(m) for m in w)
                    except Exception:
                        continue
                    if key in seen_opp:
                        continue
                    seen_opp.add(key)
                    deduped.append(w)
                opp_wires = deduped
            except Exception:
                # Any builder failure → fall through to baseline SH.
                opp_wires = []

        if len(opp_wires) >= 2:

            def decoupled_rollout_fn(my_joint: JointAction, opp_wire: List[List]) -> float:
                return _rollout_value(
                    base_state=base_state,
                    my_player=my_player,
                    my_action=my_joint.to_wire(),
                    opp_agent_factory=self._opp_factory,
                    my_future_factory=self._my_future_factory,
                    depth=self.gumbel_cfg.rollout_depth,
                    num_agents=num_agents,
                    rng=self._rng,
                    deadline_fn=_rollout_deadline_fired,
                    opp_turn0_action=opp_wire,
                    hard_stop_at=rollout_deadline_sec,
                    per_rollout_budget_ms=self.gumbel_cfg.per_rollout_budget_ms,
                )

            # Same tightening as the SH branch: cap the bandit's own
            # deadline at the outer ceiling so the decoupled root stops
            # dispatching rollouts in sync with the rollout-level
            # short-circuit.
            dec_hard_ms = self.gumbel_cfg.hard_deadline_ms
            if outer_hard_stop_at is not None:
                tight_ms = max(1.0, (rollout_deadline_sec - t0_rollout) * 1000.0)
                dec_hard_ms = min(dec_hard_ms, tight_ms)
            # Dispatch UCB vs Exp3. Unknown variant names fall back to
            # UCB with a warning — better than crashing mid-game, and
            # the config is the only place a typo can sneak in.
            variant = getattr(self.gumbel_cfg, "sim_move_variant", "ucb")
            if variant == "exp3":
                dres = decoupled_exp3_root(
                    my_candidates=joints,
                    opp_candidates=opp_wires,
                    rollout_fn=decoupled_rollout_fn,
                    total_sims=self.gumbel_cfg.total_sims,
                    hard_deadline_ms=dec_hard_ms,
                    eta=getattr(self.gumbel_cfg, "exp3_eta", 0.3),
                    start_time=start_time,
                    protected_my_idx=protected_idx,
                    rng=self._rng,
                )
            else:
                # "ucb" (default) and any unknown variant name.
                if variant != "ucb":
                    # Silent fallback is a footgun — log on once per
                    # config object so callers notice typos.
                    if not getattr(self.gumbel_cfg, "_variant_warned", False):
                        print(
                            f"[gumbel_search] unknown sim_move_variant "
                            f"{variant!r}; falling back to 'ucb'",
                            flush=True,
                        )
                        try:
                            setattr(self.gumbel_cfg, "_variant_warned", True)
                        except Exception:
                            pass
                dres = decoupled_ucb_root(
                    my_candidates=joints,
                    opp_candidates=opp_wires,
                    rollout_fn=decoupled_rollout_fn,
                    total_sims=self.gumbel_cfg.total_sims,
                    hard_deadline_ms=dec_hard_ms,
                    start_time=start_time,
                    protected_my_idx=protected_idx,
                )
            # Map DecoupledSearchResult → SearchResult so the anchor-guard
            # below operates without branching (it indexes q_values[0] as
            # the anchor's marginal Q, which is exactly my_q_values[0]).
            result = SearchResult(
                best_joint=dres.best_my_joint,
                n_rollouts=dres.n_rollouts,
                duration_ms=dres.duration_ms,
                q_values=list(dres.my_q_values),
                visits=list(dres.my_visits),
                aborted=dres.aborted,
            )
        else:
            # Tighten SH's own deadline to match rollout_deadline_sec. When
            # outer_hard_stop_at is closer than self.gumbel_cfg.hard_deadline_ms,
            # SH must stop dispatching rollouts at that same wall time, not
            # the config's looser value. Rebuild a temporary config so SH's
            # internal `t0 + cfg.hard_deadline_ms/1000` == rollout_deadline_sec.
            sh_hard_ms = self.gumbel_cfg.hard_deadline_ms
            if outer_hard_stop_at is not None:
                tight_ms = max(1.0, (rollout_deadline_sec - t0_rollout) * 1000.0)
                sh_hard_ms = min(sh_hard_ms, tight_ms)
            sh_cfg = GumbelConfig(
                num_candidates=self.gumbel_cfg.num_candidates,
                total_sims=self.gumbel_cfg.total_sims,
                rollout_depth=self.gumbel_cfg.rollout_depth,
                hard_deadline_ms=sh_hard_ms,
                anchor_improvement_margin=self.gumbel_cfg.anchor_improvement_margin,
                rollout_policy=self.gumbel_cfg.rollout_policy,
                use_decoupled_sim_move=self.gumbel_cfg.use_decoupled_sim_move,
                num_opp_candidates=self.gumbel_cfg.num_opp_candidates,
                per_rollout_budget_ms=self.gumbel_cfg.per_rollout_budget_ms,
            )
            result = sequential_halving(
                joints, rollout_fn, sh_cfg,
                start_time=start_time, protected_idx=protected_idx,
            )

        # Anchor guard: if we included an anchor, only override it when
        # the SH winner beats it by a confident margin. Rollout noise
        # with n≈4-8 sims per candidate gives SE ~0.2 on mean Q — any
        # gap below `anchor_improvement_margin` is below the noise floor
        # and we'd be trading a known-good heuristic move for a noise
        # draw. This is the load-bearing "heuristic-or-better" guarantee.
        if (
            anchor_joint is not None
            and result.best_joint is not anchor_joint
            and result.q_values
        ):
            anchor_q = result.q_values[0]  # anchor is at idx 0
            winner_q = max(result.q_values)
            if winner_q - anchor_q < self.gumbel_cfg.anchor_improvement_margin:
                # Not confident enough — return the anchor.
                result = SearchResult(
                    best_joint=anchor_joint,
                    n_rollouts=result.n_rollouts,
                    duration_ms=result.duration_ms,
                    q_values=list(result.q_values),
                    visits=list(result.visits),
                    aborted=result.aborted,
                )

        return result



# --- inlined: orbitwars/opponent/archetypes.py ---

"""Frozen archetype bots (Path D).

A small catalogue of stylistically-distinct heuristic configurations, each
implemented as a ``HeuristicAgent`` with a tailored override on top of the
default ``HEURISTIC_WEIGHTS``. Their job is twofold:

  1. **Opponent model prior** (Path D): the Bayesian posterior tracks
     P(opponent plays like archetype_k | actions observed so far). Each
     archetype is a frozen policy that scores opponent moves via its
     deterministic heuristic — the posterior-weighted mixture feeds MCTS
     opponent rollouts.

  2. **Training opponents** (Path C): PFSP needs a permanent pool of
     scripted baselines mixed into the self-play schedule (microRTS 2023
     recipe). These archetypes give us diversity without training them.

Design constraints:

  * Each archetype must be a *plausible* strategy a human or bot author
    would write. If we pad the portfolio with adversarial or broken
    configurations the posterior becomes a noise classifier.
  * Archetypes should be separable — an observed action sequence should
    have different likelihoods under different archetypes. Identical
    behavior under different names is wasted posterior dimension.
  * Weights should be *far enough* from defaults to be stylistically
    distinct, but not so degenerate that the archetype self-destructs;
    every archetype must at minimum beat a no-op bot.

Non-goals:

  * We are NOT trying to build the strongest possible heuristic variants
    here — TuRBO/EvoTune do that. Archetypes are stylistic caricatures.
  * We are NOT modeling learned opponents (AlphaZero-style bots). Those
    don't fit a 7-dimensional Dirichlet well anyway. If a learned bot
    appears on the ladder, the posterior will spread mass over whichever
    archetypes its behavior most resembles turn-by-turn, and that's fine
    — the exploitation headroom is still meaningful.

Public surface:

  ARCHETYPE_WEIGHTS : Dict[str, Dict[str, float]]
      Per-archetype weight-override dicts applied on top of HEURISTIC_WEIGHTS.
  ARCHETYPE_NAMES : List[str]
      Canonical order (used as the index of the Dirichlet posterior).
  ArchetypeAgent(name)
      HeuristicAgent subclass that reports ``name`` for logging.
  make_archetype(name) -> ArchetypeAgent
  all_archetypes() -> List[ArchetypeAgent]
"""

from typing import Dict, List



# ---- The portfolio ------------------------------------------------------

# Each dict is a partial override — unspecified keys fall back to
# HEURISTIC_WEIGHTS. This keeps diffs small and makes intent readable.
# Values were picked by eyeballing the reference heuristic's behavior,
# not tuned; archetypes are caricatures by design.

ARCHETYPE_WEIGHTS: Dict[str, Dict[str, float]] = {
    # Early pressure; small reserves; enemy-first targeting. Punishes
    # opponents that turtle / slow-expand in the opening.
    "rusher": {
        "agg_early_game": 1.8,
        "early_game_cutoff_turn": 180.0,
        "mult_enemy": 2.6,
        "mult_neutral": 0.9,
        "max_launch_fraction": 0.95,
        "min_launch_size": 10.0,
        "w_travel_cost": 0.15,
        "keep_reserve_ships": 0.0,
        "expand_cooldown_turns": 0.0,
    },

    # Big reserves, prefers close low-cost targets, slow to engage. Wants
    # to out-produce the opponent and win on turn 500.
    "turtler": {
        "agg_early_game": 0.5,
        "max_launch_fraction": 0.45,
        "keep_reserve_ships": 60.0,
        "mult_enemy": 1.1,
        "mult_neutral": 1.2,
        "w_distance_cost": 0.12,
        "w_travel_cost": 0.45,
        "min_launch_size": 30.0,
        "expand_cooldown_turns": 4.0,
    },

    # Optimizes raw production capture; patient; large deliberate
    # launches. Resembles the Kore 2022 economy-first archetype.
    "economy": {
        "w_production": 10.0,
        "w_distance_cost": 0.03,
        "w_travel_cost": 0.15,
        "mult_neutral": 1.5,
        "mult_enemy": 1.2,
        "min_launch_size": 30.0,
        "max_launch_fraction": 0.7,
        "keep_reserve_ships": 20.0,
    },

    # Cheap, frequent small attacks — goal is to force defensive
    # reactions, not to capture. Low min_launch_size + low
    # max_launch_fraction means each strike is small and replaceable.
    "harasser": {
        "min_launch_size": 5.0,
        "max_launch_fraction": 0.3,
        "mult_enemy": 2.6,
        "ships_safety_margin": 0.0,
        "expand_cooldown_turns": 0.0,
        "w_travel_cost": 0.1,
        "agg_early_game": 1.4,
    },

    # Heavily biases comet capture; willing to pre-position. Weak
    # against rushers but punishes slow opponents hard at the comet
    # spawn steps (50, 150, 250, 350, 450).
    "comet_camper": {
        "mult_comet": 3.5,
        "comet_max_time_mismatch": 3.0,
        "w_travel_cost": 0.12,
        "w_distance_cost": 0.02,
        "min_launch_size": 15.0,
    },

    # Reactive: large defensive reserves, exploits moments of enemy
    # overcommitment. Emphasizes enemy targets once contact is made.
    "opportunist": {
        "mult_enemy": 2.1,
        "mult_neutral": 1.0,
        "w_production": 7.0,
        "keep_reserve_ships": 30.0,
        "ships_safety_margin": 3.0,
        "max_launch_fraction": 0.7,
        "agg_early_game": 0.9,
    },

    # Pure defensive — rarely commits; lets opponent overextend. If
    # the ladder has this shape, an attacker-style bot with good
    # intercept math shreds it.
    "defender": {
        "max_launch_fraction": 0.4,
        "keep_reserve_ships": 110.0,
        "mult_enemy": 0.8,
        "mult_neutral": 0.9,
        "agg_early_game": 0.4,
        "min_launch_size": 35.0,
        "w_distance_cost": 0.15,
    },
}

ARCHETYPE_NAMES: List[str] = list(ARCHETYPE_WEIGHTS.keys())


# ---- Agent wrapper ------------------------------------------------------

class ArchetypeAgent(HeuristicAgent):
    """HeuristicAgent with a distinct ``name`` for tournament logging.

    Using a subclass (vs dynamically generating classes per archetype)
    keeps pickle/introspection sane and lets tournament harness code
    check ``isinstance(agent, HeuristicAgent)`` to know it shares the
    Path A contract.
    """

    def __init__(self, archetype_name: str):
        if archetype_name not in ARCHETYPE_WEIGHTS:
            raise KeyError(
                f"unknown archetype {archetype_name!r}; "
                f"known = {ARCHETYPE_NAMES}"
            )
        super().__init__(weights=ARCHETYPE_WEIGHTS[archetype_name])
        # Shadow the class-level ``name = "heuristic"`` so tournament
        # logs and Elo tracking distinguish archetypes from each other.
        self.name = archetype_name
        self._archetype = archetype_name

    @property
    def archetype(self) -> str:
        return self._archetype


def make_archetype(name: str) -> ArchetypeAgent:
    """Factory; errors loudly on unknown names."""
    return ArchetypeAgent(name)


def make_fast_archetype(name: str):
    """Fast-rollout-flavor factory for an archetype.

    Returns a ``FastRolloutAgent`` tuned so its nearest-target launch
    cadence and enemy/neutral preference match the archetype's weights.
    ~30-50x cheaper per ``act()`` call than ``make_archetype`` — use
    inside MCTS rollouts when the posterior has concentrated and we
    want flavor-matched opponent plies without the 18ms/call heuristic
    cost.

    Uses ``FastRolloutAgent.from_weights`` to handle the actual
    knob-mapping; this wrapper just does the name lookup.
    """
    if name not in ARCHETYPE_WEIGHTS:
        raise KeyError(
            f"unknown archetype {name!r}; known = {ARCHETYPE_NAMES}"
        )
    # Merge archetype overrides on top of HEURISTIC_WEIGHTS so knobs
    # the archetype didn't explicitly override still see sensible
    # base values (e.g., rusher doesn't specify keep_reserve_ships, so
    # it picks up the HEURISTIC_WEIGHTS default).
    merged = dict(HEURISTIC_WEIGHTS)
    merged.update(ARCHETYPE_WEIGHTS[name])
    return FastRolloutAgent.from_weights(merged)


def all_archetypes() -> List[ArchetypeAgent]:
    """Return one fresh agent instance per archetype, canonical order."""
    return [ArchetypeAgent(n) for n in ARCHETYPE_NAMES]


# ---- Sanity: archetype weights stay inside realistic ranges ------------

def _assert_weight_keys_are_real() -> None:
    """Catch typos — a weight override whose key isn't in HEURISTIC_WEIGHTS
    is silently ignored by HeuristicAgent.__init__, which would make the
    archetype secretly identical to the default."""
    known = set(HEURISTIC_WEIGHTS)
    for arch, overrides in ARCHETYPE_WEIGHTS.items():
        unknown = set(overrides) - known
        if unknown:
            raise AssertionError(
                f"archetype {arch!r} has overrides for unknown weight "
                f"keys: {sorted(unknown)}"
            )


_assert_weight_keys_are_real()



# --- inlined: orbitwars/opponent/bayes.py ---

"""Online Bayesian opponent modeling over archetype portfolio (Path D).

Given the archetype catalogue in ``archetypes.py`` we treat opponent
behavior as a *mixture* over archetypes and maintain a running posterior

    P(archetype = k | observed actions up to turn t)

from which we derive two things:

  (a) A posterior-weighted opponent action distribution used by MCTS
      opponent rollouts (instead of "assume heuristic").
  (b) A bias on our own root prior toward actions that *exploit* the
      most-likely archetype (if the posterior concentrates).

Why Bayesian updating and not a classifier?

  * Classifiers need a training set — we have none at submission time.
    The prior/likelihood combo gives us a *principled* online update
    that works from turn 1 with uniform prior.
  * The posterior's *uncertainty* is the information MCTS needs. A
    classifier returns a point estimate; an opponent who genuinely
    mixes strategies shows up as a flat posterior, and MCTS needs
    that signal to avoid mis-exploiting.

Cost budget:

  Per turn, we evaluate K archetypes (7) on the opponent's obs, each
  costing one ``HeuristicAgent.act()``. Heuristic acts are sub-2 ms.
  7 × 2 ms ≈ 14 ms/turn, well inside the ~5 ms target we'd prefer;
  in practice Python overhead dominates and we see ~10-20 ms. Still
  fits under the MCTS search budget.

Implementation choices:

  * **Log-space updates** — K archetypes × 500 turns × product of
    likelihoods will underflow naive float64 very quickly.
  * **Dirichlet-equivalent interpretation**: we maintain an unnormalized
    log-weight vector ``log_alpha`` and exponentiate on query. This is
    equivalent to a Dirichlet posterior on the mixture weights where
    we treat each turn's observation as drawing one category. The
    temperature knob lets us soften per-turn likelihoods (a real
    opponent is noisier than a pure archetype).
  * **Launch-decision-only likelihood** — for v1 we ignore angle and
    ship-count and match only on "did the opponent launch from planet
    X this turn". Angles are continuous (many approximate matches are
    meaningful) and sizes are dependent on the current ship stockpile
    which varies across archetypes; extending the likelihood to those
    dimensions is a clean follow-up but not needed to separate
    rusher-vs-turtler-vs-harasser.
  * **Per-planet Bernoulli** — each owned planet contributes independent
    evidence. An archetype that correctly predicts launch-vs-hold on
    most planets accumulates posterior mass.

Public surface:

  ArchetypePosterior(archetypes, alpha0=1.0, temperature=2.0, eps=0.1)
      .observe(obs, opp_player)     # call after opp's action is visible
      .distribution() -> np.ndarray # posterior over archetypes
      .most_likely() -> str         # name of highest-posterior archetype
      .reset()                      # new match

Integration sketch:

  post = ArchetypePosterior(all_archetypes())
  for turn in game:
      obs = ...
      if turn > 0:                  # need at least one opp action
          post.observe(obs, opp_player)
      dist = post.distribution()
      # pass into MCTS opponent-rollout mixing
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np



# ---- Helpers -----------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _fabricate_opp_obs(obs: Any, opp_player: int) -> Dict[str, Any]:
    """Orbit Wars is fully observable — same state, different player tag.

    We copy only the fields ``parse_obs`` reads, since feeding the
    archetype an obs that's missing keys it expects would raise.
    """
    return {
        "player": opp_player,
        "step": obs_get(obs, "step", 0),
        "angular_velocity": obs_get(obs, "angular_velocity", 0.0),
        "planets": list(obs_get(obs, "planets", [])),
        "initial_planets": list(obs_get(obs, "initial_planets", [])),
        "fleets": list(obs_get(obs, "fleets", [])),
        "next_fleet_id": obs_get(obs, "next_fleet_id", 0),
        "comet_planet_ids": list(obs_get(obs, "comet_planet_ids", [])),
    }


# ---- Posterior ---------------------------------------------------------

@dataclass
class ArchetypePosterior:
    """Online posterior over archetypes given observed opponent actions.

    Args:
        archetypes: the frozen bots whose log-likelihoods we evaluate.
        alpha0: uniform Dirichlet-prior concentration. Use >1 for a
            stronger "no archetype yet" prior.
        temperature: divides the per-turn log-likelihood before
            accumulation. T=1 is raw Bayes; T>1 softens (noisier
            opponent); T<1 sharpens. We default T=2.0 — real
            opponents rarely match an archetype perfectly.
        eps: per-planet Bernoulli noise floor. An archetype that
            predicts "no launch" but sees launch still contributes
            log(eps) rather than -inf.
    """
    archetypes: List[ArchetypeAgent] = field(default_factory=all_archetypes)
    alpha0: float = 1.0
    temperature: float = 2.0
    eps: float = 0.1
    # Early-exit: once the top archetype's posterior probability reaches
    # this threshold, stop running the K-archetype act() likelihood loop
    # on subsequent turns. Saves ~15 ms/turn (the dominant per-turn cost)
    # once the opponent has been identified. Set to 1.0 to disable.
    # Fleet-id bookkeeping still runs (needed if someone resets us later
    # with a fresh match), and ``turns_observed`` still increments so
    # downstream gates keep working.
    freeze_threshold: float = 0.99

    def __post_init__(self) -> None:
        self.K = len(self.archetypes)
        self.names = [a.name for a in self.archetypes]
        # Log-unnormalized posterior starts at log(alpha0).
        self.log_alpha = np.full(self.K, np.log(self.alpha0), dtype=np.float64)
        # Track previously-seen fleet ids so we can identify new launches.
        self._prev_fleet_ids: Set[int] = set()
        self._last_obs: Optional[Dict[str, Any]] = None
        self._turns_observed: int = 0
        # Frozen once the posterior concentrates past freeze_threshold.
        # While frozen, observe() skips the expensive K-archetype loop.
        self._frozen: bool = False

    # ---- Public ----

    def reset(self) -> None:
        self.log_alpha[:] = np.log(self.alpha0)
        self._prev_fleet_ids.clear()
        self._last_obs = None
        self._turns_observed = 0
        self._frozen = False

    def is_frozen(self) -> bool:
        """True once the posterior concentration crossed ``freeze_threshold``.

        Exposed for smokes/telemetry — lets a test verify the early-exit
        path fired after N turns of strong evidence.
        """
        return self._frozen

    def observe(self, obs: Any, opp_player: int) -> None:
        """Incorporate the opponent's action revealed by ``obs``.

        Must be called in turn order (step increases by 1 each call).
        On the very first call we only snapshot the state; we need the
        previous turn's obs to identify *newly-launched* fleets.
        """
        if self._last_obs is None:
            self._last_obs = obs
            self._prev_fleet_ids = {
                int(f[0]) for f in obs_get(obs, "fleets", [])
            }
            return

        # Early-exit: frozen posterior skips the K-archetype likelihood
        # loop (the ~15 ms/turn hot spot). We keep the fleet-id snapshot
        # current and tick turns_observed so downstream consumers don't
        # see stale telemetry. log_alpha is left untouched — distribution()
        # continues to return the frozen posterior.
        if self._frozen:
            self._prev_fleet_ids = {
                int(f[0]) for f in obs_get(obs, "fleets", [])
            }
            self._last_obs = obs
            self._turns_observed += 1
            return

        # Run the likelihood update path. Tick turns_observed and check
        # for freeze transition regardless of whether the update
        # short-circuits (opp eliminated etc.) — a pre-seeded log_alpha
        # that's already over-threshold should freeze on its first real
        # observe() call.
        self._update_log_alpha(obs, opp_player)
        self._turns_observed += 1
        self._maybe_freeze()

    def _update_log_alpha(self, obs: Any, opp_player: int) -> None:
        """Incorporate one turn of opp evidence into ``log_alpha``.

        Split out from ``observe`` so the freeze check fires at a single
        well-defined point regardless of which control-flow path the
        update took.
        """
        # Identify fleets launched by opp this turn.
        opp_launches = self._opp_launches_this_turn(obs, opp_player)

        # Snapshot current fleet ids for the next turn's diff.
        self._prev_fleet_ids = {
            int(f[0]) for f in obs_get(obs, "fleets", [])
        }

        # Evidence is over *opp-owned planets that exist* on the
        # previous turn's obs — launches come from there. We evaluate
        # each archetype on the previous turn's state (what opp "saw"
        # when deciding), not the current state (which reflects their
        # action + our action + world updates).
        prev_obs = self._last_obs
        self._last_obs = obs

        opp_planet_ids = {
            int(pl[0]) for pl in obs_get(prev_obs, "planets", [])
            if int(pl[1]) == opp_player
        }
        if not opp_planet_ids:
            # Nothing to condition on — opp has been eliminated.
            return

        for k, arch in enumerate(self.archetypes):
            predicted = self._predicted_launches(arch, prev_obs, opp_player)
            log_lik = self._log_likelihood(
                observed_launches=opp_launches,
                predicted_launches=predicted,
                planet_ids=opp_planet_ids,
            )
            self.log_alpha[k] += log_lik / self.temperature

    def _maybe_freeze(self) -> None:
        """Flip ``_frozen`` on when concentration crosses the threshold.

        Called at the end of observe() (non-bootstrap, non-frozen path).
        ``freeze_threshold=1.0`` opts out — the check becomes unreachable.
        """
        if self.freeze_threshold < 1.0:
            if float(_softmax(self.log_alpha).max()) >= self.freeze_threshold:
                self._frozen = True

    def distribution(self) -> np.ndarray:
        """Posterior over archetypes as a probability vector."""
        return _softmax(self.log_alpha)

    def most_likely(self) -> str:
        return self.names[int(np.argmax(self.log_alpha))]

    def turns_observed(self) -> int:
        return self._turns_observed

    # ---- Internals ----

    def _opp_launches_this_turn(
        self, obs: Any, opp_player: int,
    ) -> Set[int]:
        """Set of planet ids the opponent launched from this turn.

        Uses fleet-id diffing against the previous turn's snapshot. A
        fleet is "new" if its id wasn't in the prior obs.
        """
        launches: Set[int] = set()
        for f in obs_get(obs, "fleets", []):
            fid = int(f[0])
            if fid in self._prev_fleet_ids:
                continue
            owner = int(f[1])
            from_pid = int(f[5])
            if owner == opp_player:
                launches.add(from_pid)
        return launches

    def _predicted_launches(
        self, archetype: ArchetypeAgent, obs: Any, opp_player: int,
    ) -> Set[int]:
        """What set of planets would `archetype` launch from, playing
        for `opp_player` on this obs?"""
        opp_obs = _fabricate_opp_obs(obs, opp_player)
        dl = Deadline()
        action = archetype.act(opp_obs, dl)
        launches: Set[int] = set()
        for mv in action or []:
            if len(mv) >= 1:
                launches.add(int(mv[0]))
        return launches

    def _log_likelihood(
        self,
        observed_launches: Set[int],
        predicted_launches: Set[int],
        planet_ids: Set[int],
    ) -> float:
        """Per-planet Bernoulli log-likelihood.

        For each planet the opponent owned:
          If archetype predicts launch and obs shows launch  → log(1-eps)
          If archetype predicts launch and obs shows hold    → log(eps)
          If archetype predicts hold and obs shows hold      → log(1-eps)
          If archetype predicts hold and obs shows launch    → log(eps)

        We only evaluate on planets the opp actually owns (planet_ids) —
        planets they lost this turn don't carry an action decision.
        """
        if not planet_ids:
            return 0.0
        log_hit = np.log(1.0 - self.eps)
        log_miss = np.log(self.eps)
        total = 0.0
        for pid in planet_ids:
            obs_launch = pid in observed_launches
            pred_launch = pid in predicted_launches
            total += log_hit if (obs_launch == pred_launch) else log_miss
        return total



# --- inlined: orbitwars/bots/mcts_bot.py ---

"""Path B bot: Gumbel top-k + Sequential Halving over heuristic rollouts.

Integration of `orbitwars.mcts.gumbel_search` behind the `Agent` contract.
On each turn we:
  1. Enumerate per-planet candidate moves via the heuristic's scorer.
  2. Sample K joint actions via the Gumbel top-k trick.
  3. Allocate a rollout budget with Sequential Halving.
  4. Return the highest-mean-Q joint's wire format.

Safety:
  * We stage a heuristic action by EARLY_FALLBACK_MS so a search blow-up
    never results in a no-op turn.
  * Any exception inside search falls back to the staged heuristic move.
  * Rollouts respect an internal hard deadline well below actTimeout.
"""

import time
from typing import Any, Dict, Optional



class MCTSAgent(Agent):
    """Gumbel Sequential Halving with heuristic-priored rollouts.

    The agent keeps a single `HeuristicAgent` around as the safe
    fallback. Searches are stateless per call (the GumbelRootSearch
    owns only its RNG).

    Opponent modeling (Path D):
      If ``use_opponent_model`` is True (default), the agent observes
      the opponent's actions each turn and maintains an online
      ArchetypePosterior. The posterior is exposed as
      ``self.opp_posterior`` for diagnostics. A follow-up change will
      route the posterior into MCTS rollouts so search biases toward
      moves that exploit the most-likely archetype — v1 just collects
      the evidence so the data is there when we light up the integration.
    """

    name = "mcts"

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        action_cfg: Optional[ActionConfig] = None,
        gumbel_cfg: Optional[GumbelConfig] = None,
        rng_seed: Optional[int] = None,
        use_opponent_model: bool = True,
    ):
        self.weights = dict(HEURISTIC_WEIGHTS) if weights is None else dict(weights)
        # BOKR-style angle refinement is available (set
        # ``angle_refinement_n_grid > 1`` in your ActionConfig) but
        # DEFAULTED OFF. Smoke testing showed refinement pushes the turn-time
        # tail past Kaggle's 1-second actTimeout (seed=42, default
        # deadline 300ms: max=1156ms, 2 turns over 900ms — forfeit
        # risk). The BOKR module is wired into generate_per_planet_moves
        # so callers can opt in for specific experiments, but the
        # shipped MCTSAgent uses the single-angle behavior to preserve
        # the v3 tail profile (max 882 ms, 0 over 900 ms).
        self.action_cfg = action_cfg or ActionConfig()
        self.gumbel_cfg = gumbel_cfg or GumbelConfig()
        # Arm the decoupled sim-move branch by default. The branch is a
        # no-op unless MCTSAgent also populates ``opp_candidate_builder``
        # with >=2 wires (see _maybe_route_posterior_to_search), so this
        # is backward-compat: behavior only changes once the posterior
        # has concentrated enough to propose a multi-archetype mixture.
        self.gumbel_cfg.use_decoupled_sim_move = True
        self._fallback = HeuristicAgent(weights=self.weights)
        self._search = GumbelRootSearch(
            weights=self.weights,
            action_cfg=self.action_cfg,
            gumbel_cfg=self.gumbel_cfg,
            rng_seed=rng_seed,
        )
        self._use_opponent_model = use_opponent_model
        # Posterior is created lazily on turn 0 so per-match state
        # resets come free with the existing turn-0 reset path below.
        self.opp_posterior: Optional[ArchetypePosterior] = None

        # Posterior telemetry — cheap counters so smokes can reason about
        # WHY a run did or didn't see a use-model delta (vs. a null result
        # with no insight into whether the override ever fired). Fields:
        #   turns_observed   — turns the posterior saw an update this match
        #   override_fires   — turns `opp_policy_override` was set to an archetype
        #   override_clears  — turns we explicitly dropped the override (gate failed)
        #   last_top_name    — most recent argmax archetype (for sanity in logs)
        #   last_top_prob    — most recent max of dist() (0.0 if no posterior yet)
        # Reset on turn 0 along with the other per-match state below.
        self.telemetry: Dict[str, Any] = {
            "turns_observed": 0,
            "override_fires": 0,
            "override_clears": 0,
            "builder_fires": 0,
            "builder_clears": 0,
            "last_top_name": None,
            "last_top_prob": 0.0,
        }

    # Posterior → search override tuning. Conservative: require ~15
    # turns of evidence AND a top-archetype probability at least 2.5x
    # the uniform 1/K baseline. Below that, the posterior is noise.
    _POSTERIOR_MIN_TURNS: int = 15
    _POSTERIOR_MIN_TOP_PROB: float = 0.35
    # Decoupled sim-move branch gate. When the *2nd* archetype also
    # has meaningful mass (>= 0.2 ~= ~1.5x uniform), marginalize over
    # both via decoupled UCB. With second-top below this threshold, a
    # single-archetype SH is strictly stronger (no rollouts wasted on
    # a phantom branch), so we keep the builder = None.
    _POSTERIOR_DECOUPLED_MIN_SECOND_PROB: float = 0.20

    # ---- Overage-bank lift (plan §W3) ---------------------------------
    #
    # The Kaggle simulator overruns actTimeout by drawing from the
    # remainingOverageTime bank. On opening turns the map hasn't
    # diverged much yet and deeper search pays off; on late turns most
    # outcomes are decided and going long just burns the bank. We lift
    # the deadline only when BOTH conditions hold:
    #   (1) turn index is in the opening window (default 10),
    #   (2) the bank is generously padded beyond the emergency reserve.
    # Outside that window we return 0 so the standard 850 ms deadline
    # applies. The reserve is kept in the bank for late-game turn-time
    # spikes — if we burn the bank dry we forfeit the match on the
    # next slow turn.
    #
    # These constants live at the class level so a specific MCTSAgent
    # subclass (or an experiment harness) can tighten/loosen them in
    # isolation without editing the base.py default.
    _OVERAGE_OPENING_TURNS: int = 10        # turns that may be lifted
    _OVERAGE_RESERVE_SEC: float = 2.0       # floor we never draw below
    _OVERAGE_MIN_BANK_SEC: float = 5.0      # refuse to lift below this bank
    _OVERAGE_MAX_BOOST_MS: float = 2000.0   # per-turn ceiling on the lift
    # ``(bank - reserve)`` is amortized across the opening window; no
    # single turn gets more than ``_OVERAGE_MAX_BOOST_MS``.

    def deadline_boost_ms(self, obs: Any, step: int) -> float:
        """Read the overage bank and decide how much to lift this turn.

        Design — see class-level OVERAGE_* constants for the thresholds.
        Returns 0 whenever the lift is unsafe (outside opening window,
        bank below the reserve, missing field). Any exception in here
        is caught by the wrapper and converted to 0 so a malformed obs
        never forfeits a match.
        """
        if step >= self._OVERAGE_OPENING_TURNS:
            return 0.0
        bank = float(obs_get(obs, "remainingOverageTime", 0.0))
        if bank < self._OVERAGE_MIN_BANK_SEC:
            # Bank too tight — leave it alone for the late-game safety net.
            return 0.0
        # Amortize the *usable* bank (above the reserve) across the
        # remaining opening turns. This keeps us honest when the map is
        # still shared between the agents — we don't blow the entire
        # bank on turn 0 and starve ourselves on turn 9.
        remaining_opening_turns = max(1, self._OVERAGE_OPENING_TURNS - step)
        usable_bank_ms = max(0.0, bank - self._OVERAGE_RESERVE_SEC) * 1000.0
        per_turn_ms = usable_bank_ms / float(remaining_opening_turns)
        return min(self._OVERAGE_MAX_BOOST_MS, per_turn_ms)

    def _maybe_route_posterior_to_search(self) -> None:
        """If the posterior has concentrated, set the search's opponent
        rollout policy to the matching archetype. Otherwise clear any
        prior override."""
        post = self.opp_posterior
        if post is None:
            return
        # Always refresh telemetry when posterior exists, even below the
        # turns gate — telemetry answers "did the smoke run long enough?"
        # which only makes sense if we see turns_observed climb.
        self.telemetry["turns_observed"] = post.turns_observed()
        if post.turns_observed() < self._POSTERIOR_MIN_TURNS:
            return
        dist = post.distribution()
        top_prob = float(dist.max())
        self.telemetry["last_top_prob"] = top_prob
        self.telemetry["last_top_name"] = post.most_likely()
        if top_prob < self._POSTERIOR_MIN_TOP_PROB:
            # Not concentrated → no override (opp rolls under default heuristic).
            if self._search.opp_policy_override is not None:
                self._search.opp_policy_override = None
                self.telemetry["override_clears"] += 1
            # Also make sure the decoupled builder is cleared so the
            # search branch doesn't fire under noise.
            if self._search.opp_candidate_builder is not None:
                self._search.opp_candidate_builder = None
            return
        top_name = post.most_likely()
        # Late-bind the name so every call produces a fresh archetype
        # (HeuristicAgent has per-match state that rollouts must not share).
        # When rollout_policy=="fast", swap in the flavor-matched fast
        # rollout agent — ~30x cheaper per ply, same stylistic bias.
        if self.gumbel_cfg.rollout_policy == "fast":
            self._search.opp_policy_override = (
                lambda n=top_name: make_fast_archetype(n)
            )
        else:
            self._search.opp_policy_override = (
                lambda n=top_name: make_archetype(n)
            )
        self.telemetry["override_fires"] += 1

        # Decoupled UCB branch: fires only when the *second* archetype
        # also has real mass. Marginalizing over a phantom 2nd branch
        # wastes rollouts, so below the threshold we leave the builder
        # = None and the search falls back to plain Sequential Halving.
        sorted_probs = sorted(dist, reverse=True)
        if (
            len(sorted_probs) >= 2
            and sorted_probs[1] >= self._POSTERIOR_DECOUPLED_MIN_SECOND_PROB
        ):
            self._search.opp_candidate_builder = self._build_opp_candidates
            self.telemetry["builder_fires"] = (
                self.telemetry.get("builder_fires", 0) + 1
            )
        else:
            if self._search.opp_candidate_builder is not None:
                self._search.opp_candidate_builder = None
                self.telemetry["builder_clears"] = (
                    self.telemetry.get("builder_clears", 0) + 1
                )

    def _build_opp_candidates(self, obs: Any, opp_player: int):
        """Compute opp's wire action under each of the top-K archetypes.

        Called by ``GumbelRootSearch`` when the decoupled sim-move branch
        is armed. Returns a list of wire actions — one per archetype —
        that the bandit marginalizes over.

        Fails closed: any exception returns ``[]``, which makes the
        search fall back to plain Sequential Halving (the pre-decoupled
        shipped behavior). This is the contract the search relies on.
        """
        try:
            post = self.opp_posterior
            if post is None:
                return []
            k = max(1, int(self.gumbel_cfg.num_opp_candidates))
            dist = post.distribution()
            # Rank archetypes by posterior mass, descending. Keep only
            # those with non-negligible mass (>= second-prob threshold
            # / 2) so a near-uniform posterior doesn't pad the list
            # with noise candidates.
            floor = 0.5 * self._POSTERIOR_DECOUPLED_MIN_SECOND_PROB
            ranked = sorted(
                [(i, float(p)) for i, p in enumerate(dist)],
                key=lambda ip: -ip[1],
            )
            names = [post.names[i] for i, p in ranked[:k] if p >= floor]
            if len(names) < 2:
                return []

            # Build opp's observation once via a temporary FastEngine
            # (perspective-swap). Cheap — a dict shim + a FastEngine
            # construction, comparable to what search already does
            # per-rollout.

            eng = FastEngine.from_official_obs(
                _obs_to_namespace(obs), num_agents=2,
            )
            opp_obs = eng.observation(opp_player)

            wires = []
            # Fresh Deadline per archetype — generous, since this is
            # called from inside the outer turn budget and the archetype
            # .act()s are cheap heuristic passes (<5 ms each).
            for name in names:
                dl = Deadline()
                try:
                    agent = make_archetype(name)
                    wire = agent.act(opp_obs, dl)
                except Exception:
                    continue
                if isinstance(wire, list):
                    wires.append(wire)
            return wires
        except Exception:
            return []

    def act(self, obs: Any, deadline: Deadline) -> Action:
        # Always stage no_op first so any premature return is legal.
        deadline.stage(no_op())

        # ── Match-start detection MUST precede self._fallback.act() ──
        # Seat 0: obs.step==0 signals a new game.
        # Seat 1: obs.step is None (Kaggle engine quirk); we use
        # next_fleet_id regression (or first-call) as the match-start
        # signal.
        #
        # Detecting BEFORE calling fallback.act is load-bearing: the
        # reset below replaces self._fallback with a fresh HeuristicAgent.
        # If we called self._fallback.act first and then replaced it, the
        # first call's _turn_counter increment (0→1) would be discarded
        # by the replacement, leaving the new fallback's counter at None.
        # On turn 2 its counter then advances None→1 instead of 1→2, so
        # for the remainder of the match fallback._turn_counter is
        # ALWAYS one turn behind a freshly-created HeuristicAgent reading
        # the same observations. MCTS threads that stale counter to
        # search as step_override, so both the anchor heuristic_move AND
        # the search's synthetic obs.step drift off-by-one — which
        # silently breaks anchor-lock at seat 1 (confirmed 3/30 turns
        # diverge by tools/diag_mcts_vs_heur_actions_seat1.py). Seat 0
        # is unaffected because obs.step is authoritative there and
        # HeuristicAgent ignores _turn_counter when raw_step is set.
        raw_step = obs_get(obs, "step", None)
        curr_nfid = int(obs_get(obs, "next_fleet_id", 0))
        if raw_step is not None:
            fresh_game = (int(raw_step) == 0)
        else:
            prev_nfid = getattr(self, "_prev_next_fleet_id", None)
            fresh_game = prev_nfid is None or prev_nfid > curr_nfid
        self._prev_next_fleet_id = curr_nfid
        if fresh_game:
            # Fresh heuristic both for fallback and for the search's
            # internal rollouts.
            self._fallback = HeuristicAgent(weights=self.weights)
            self._search = GumbelRootSearch(
                weights=self.weights,
                action_cfg=self.action_cfg,
                gumbel_cfg=self.gumbel_cfg,
                rng_seed=None,  # fresh RNG; deterministic only if seeded at ctor.
            )
            # Per-match opponent posterior — archetypes are stateful
            # (HeuristicAgent holds _LaunchState), so we reset between games.
            if self._use_opponent_model:
                self.opp_posterior = ArchetypePosterior()
            # Also clear any stale override from the previous match — the
            # new opponent is an unknown, back to default heuristic rollouts.
            self._search.opp_policy_override = None
            self._search.opp_candidate_builder = None
            # Reset per-match telemetry so smokes running back-to-back
            # matches don't see stale counts leaking across games.
            self.telemetry = {
                "turns_observed": 0,
                "override_fires": 0,
                "override_clears": 0,
                "builder_fires": 0,
                "builder_clears": 0,
                "last_top_name": None,
                "last_top_prob": 0.0,
            }

        # Stage the heuristic action as our floor. If search wins, we
        # overwrite; if it doesn't, we return this. The fallback here is
        # guaranteed to be the one we'll keep for this match (fresh-game
        # replacement already happened above), so its _turn_counter
        # stays in lockstep with an outside shadow HeuristicAgent.
        try:
            heuristic_move = self._fallback.act(obs, deadline)
            deadline.stage(heuristic_move)
        except Exception:
            heuristic_move = no_op()

        my_player = int(obs_get(obs, "player", 0))

        # Opponent-model observation. Cheap (<20 ms on a dense mid-game
        # obs) and wrapped in try/except so a defect in the posterior
        # never escapes to the search path. v1 is 2-player only: opp is
        # the other seat.
        #
        # Exploitation: once the posterior has concentrated (>=15 turns
        # observed AND top archetype probability > 0.35, i.e. ~2.5x the
        # uniform 1/7 floor), we route the top archetype's HeuristicAgent
        # as the opponent's rollout policy instead of the generic
        # HeuristicAgent(self.weights). This makes MCTS search under the
        # *actual* inferred opponent model rather than "assume default
        # heuristic". Threshold and grace period are conservative — a
        # wrong override is worse than no override, since search then
        # optimizes against a phantom opponent.
        if self._use_opponent_model and self.opp_posterior is not None:
            try:
                opp_player = 1 - my_player  # 2-player assumption
                self.opp_posterior.observe(obs, opp_player=opp_player)
                self._maybe_route_posterior_to_search()
            except Exception:
                # Posterior is informational-only in v1; a bad update
                # must never break the turn.
                pass

        # Respect the outer agent-level deadline too: if we've already
        # burned most of actTimeout staging the fallback, skip search.
        remaining = deadline.remaining_ms(HARD_DEADLINE_MS)
        if remaining < 50.0:
            return heuristic_move

        # Tighten the search-internal deadline to whatever the outer
        # Deadline gives us, minus:
        #   * _ROLLOUT_OVERSHOOT_BUDGET_MS (260): after sequential_halving's
        #     hard deadline fires, the in-flight rollout can still run its
        #     turn-0 opp-heuristic call + step before the per-ply check in
        #     _rollout_value short-circuits the rest. On dense mid-game
        #     states that overshoot hits ~200-270 ms. Observed (audit pass
        #     2): max turn 1172 ms vs 900 ms outer ceiling → 272 ms
        #     overshoot. Reserve 260 ms so worst case lands under 900 ms.
        #   * 40 ms: post-search wrap-up (action encoding, staging).
        # Without this reservation, a slow pre-search (heuristic.act on a
        # fleet-heavy state + posterior.observe) burns most of the outer
        # budget and the search's internal 300 ms deadline can push total
        # elapsed past 900 ms. The audit measures EXACTLY this number.
        _ROLLOUT_OVERSHOOT_BUDGET_MS = 260.0
        _WRAPUP_BUDGET_MS = 40.0
        # The cap normally comes straight from the Gumbel config; on
        # turns where ``Agent.deadline_boost_ms`` has lifted the outer
        # deadline from the overage bank, lift the cap by the same
        # amount so search can actually consume the extra budget.
        # Without this, ``remaining`` grows but the cap below still
        # clamps us to the 300 ms default and the boost is wasted.
        effective_cap_ms = (
            self.gumbel_cfg.hard_deadline_ms + deadline.extra_budget_ms
        )
        safe_budget = min(
            effective_cap_ms,
            remaining - _ROLLOUT_OVERSHOOT_BUDGET_MS - _WRAPUP_BUDGET_MS,
        )
        if safe_budget <= 10.0:
            return heuristic_move

        # Rebuild a one-shot config with the tightened deadline. All other
        # fields (including anchor_improvement_margin!) must be preserved
        # so the safety floor still protects us under the tight budget.
        tight_cfg = GumbelConfig(
            num_candidates=self.gumbel_cfg.num_candidates,
            total_sims=self.gumbel_cfg.total_sims,
            rollout_depth=self.gumbel_cfg.rollout_depth,
            hard_deadline_ms=safe_budget,
            anchor_improvement_margin=self.gumbel_cfg.anchor_improvement_margin,
        )

        # Compute the caller-side outer hard stop: the latest wall-clock
        # instant at which search must return. We reserve
        # _OUTER_CEILING_MARGIN_MS between this stop and HARD_DEADLINE_MS
        # so that an in-flight rollout short-circuiting "one inner
        # iteration after the deadline fires" still lands under the
        # outer actTimeout.
        #
        # _OUTER_CEILING_MARGIN_MS budget:
        #   ~100 ms  — worst-case single-inner-iteration cost in
        #              HeuristicAgent._plan_moves on a dense late-game
        #              state (comments on that loop cite ~100-300 ms for
        #              the full outer iteration; one inner-iteration
        #              slice is the overshoot from a fired deadline).
        #   ~20  ms  — action encoding + deadline.stage + any
        #              in-wrapper gc.collect the harness includes in
        #              the turn-time measurement.
        #   -------
        #    120 ms  — conservative ceiling; tighten once we have
        #              audit data confirming the real pathological
        #              ply cost is lower than 100 ms.
        _OUTER_CEILING_MARGIN_MS = 120.0
        outer_hard_stop_at = (
            time.perf_counter()
            + max(0.0, remaining - _OUTER_CEILING_MARGIN_MS) / 1000.0
        )

        # Wrap the entire swap+search+restore so ANY failure (including
        # attribute access on a broken search object) degrades to the
        # heuristic. Agents in ladder play must never bubble.
        saved_cfg = None
        try:
            saved_cfg = self._search.gumbel_cfg
            self._search.gumbel_cfg = tight_cfg
            t0 = time.perf_counter()
            # Pass the heuristic's move in as the anchor candidate:
            # search will only overwrite it with something evaluated to
            # be better, so the MCTS agent is guaranteed heuristic-or-
            # better in expectation.
            # Thread step from the fallback's turn counter. self._fallback.act
            # was called above and updated its monotonic _turn_counter;
            # we reuse it so search sees the same step even on seat 1
            # (where obs.step is None).
            step_override = getattr(self._fallback, "_turn_counter", None)
            result = self._search.search(
                obs, my_player, start_time=t0,
                anchor_action=heuristic_move,
                outer_hard_stop_at=outer_hard_stop_at,
                step_override=step_override,
            )
        except Exception:
            return heuristic_move
        finally:
            if saved_cfg is not None:
                try:
                    self._search.gumbel_cfg = saved_cfg
                except Exception:
                    pass

        if result is None:
            return heuristic_move

        action = result.best_joint.to_wire()
        deadline.stage(action)
        return action


def build(**overrides) -> MCTSAgent:
    """Factory for packaging / tournament registration."""
    return MCTSAgent(**overrides)




# --- tuned weights override ---

# Applied by tools/bundle.py --weights-json at build time.

HEURISTIC_WEIGHTS.update({
    'agg_early_game': 1.4049309324473143,
    'comet_max_time_mismatch': 3.9275696966797113,
    'early_game_cutoff_turn': 99.72046632319689,
    'expand_bias': 1.414374615997076,
    'expand_cooldown_turns': 3.9825727604329586,
    'keep_reserve_ships': 5.717945722863078,
    'max_launch_fraction': 0.6944328492507339,
    'min_launch_size': 29.121542111970484,
    'mult_comet': 1.7020629160106184,
    'mult_enemy': 2.263682206347585,
    'mult_neutral': 1.36488540424034,
    'mult_reinforce_ally': 0.2916624844074249,
    'ships_safety_margin': 3.350880299694836,
    'sun_avoidance_epsilon': 0.07407572213560343,
    'w_distance_cost': 0.09539055917412043,
    'w_production': 19.440082334447652,
    'w_ships_cost': 0.13802833948284388,
    'w_travel_cost': 0.13828593213111162,
})


# --- GumbelConfig / MCTSAgent overrides ---

# Applied by tools/bundle.py at build time.

_bundle_cfg = GumbelConfig()

_bundle_cfg.rollout_policy = 'fast'

_bundle_cfg.anchor_improvement_margin = 0.5


# --- agent entry point ---

agent = MCTSAgent(gumbel_cfg=_bundle_cfg, rng_seed=0, use_opponent_model=False).as_kaggle_agent()
