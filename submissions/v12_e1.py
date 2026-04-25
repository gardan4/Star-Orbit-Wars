# Auto-generated Orbit Wars submission. Do not edit by hand.
# Built by tools/bundle.py on 2026-04-25 13:20:44.
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
    # Path C neural prior bridge. When set, called after
    # ``generate_per_planet_moves`` to overwrite the heuristic prior on
    # each PlanetMove with a NN-derived prior. Signature:
    #   ``(obs, my_player, moves_by_planet, available_by_planet)
    #     -> Dict[planet_id, List[PlanetMove]]``
    # The returned dict has the same keys as the input but with new
    # PlanetMove objects (PlanetMove is frozen) carrying the new prior.
    # Built via ``orbitwars.nn.nn_prior.make_nn_prior_fn``.
    move_prior_fn: Optional[Callable[[Any, int, Dict[int, List[PlanetMove]], Dict[int, int]], Dict[int, List[PlanetMove]]]] = None

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

        # Path C: optionally rewrite priors with the NN bridge. This runs
        # ONCE at the root — Gumbel sampling at the root reads these new
        # priors directly. Inner-node statistics still come from rollouts
        # so a low-quality NN cannot poison the search beyond its prior
        # weight. Errors here fall back to the heuristic priors that the
        # generator already produced (defensive: the NN path is optional
        # and we never want it to forfeit a turn).
        if self.move_prior_fn is not None:
            try:
                # Each PlanetMove shares its source planet's ship count;
                # extract once per planet from the parsed obs. Comet/orbit
                # planets that we own and that show up in per_planet must
                # be in po.planet_by_id.
                available_by_planet: Dict[int, int] = {}
                for pid in per_planet.keys():
                    pdata = po.planet_by_id.get(int(pid))
                    if pdata is not None and len(pdata) > 5:
                        available_by_planet[int(pid)] = int(pdata[5])
                    else:
                        available_by_planet[int(pid)] = 0
                per_planet = self.move_prior_fn(
                    obs, my_player, per_planet, available_by_planet,
                )
            except Exception:
                # Keep heuristic priors as the fallback. The search will
                # behave exactly like a no-NN-prior MCTSAgent.
                pass

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



# --- inlined: orbitwars/nn/conv_policy.py ---

"""Centralized per-entity-grid conv policy for Orbit Wars (W4 candidate A).

**Status**: SKELETON. Architecture decisions frozen; forward pass body is
stubbed. This is the *primary* candidate for the W4 architecture bake-off
per the plan — Set-Transformer (see ``set_transformer.py``) is candidate
B. Pick winner by 1M-step training result, not a priori.

**Why this architecture?** Lux S1/S3 winning submissions used centralized
per-entity-grid conv policies over a dense spatial tensor. The recipe:

  1. Render the game state onto a ``(C, H, W)`` grid where each channel
     is one entity feature (ships_owned_0, ships_owned_1, production,
     is_comet, sun_distance, ...). H=W=50 or 100 gives a spatial
     resolution that captures intercept geometry without blowing up
     the activation volume.
  2. Run a small conv backbone (4-8 residual blocks, 64-128 channels).
  3. Emit two heads:
       * **Policy head**: per-planet action distribution, decoded by
         indexing the output grid at each owned planet's (x, y) slot.
       * **Value head**: scalar game value via global average pool.

**Why it beats a flat MLP / set-transformer on RTS:**
  * Spatial locality is the dominant structure (nearby planets interact,
    far planets don't). Conv's inductive bias matches the game.
  * Translation equivariance on the mirror-symmetric board is free data
    augmentation: a conv filter trained in one quadrant generalizes
    automatically to the other three.
  * O(H*W) compute vs O(N^2) attention — cheaper at N=40 planets and
    scales better if we move to 100+ entity boards in later iterations.

**Parameter budget:**
  * Target: <2M params total after distillation (W5 deliverable —
    Bayesian Policy Distillation to <2M student).
  * W4 teacher: 5-20M params is fine; student is what ships.

**Training pipeline (not in this file):**
  * Feature encoding: ``orbitwars.features.obs_encode`` (stub today).
  * PPO loop: ``orbitwars.train.ppo_jax`` (not yet created).
  * PFSP opponent pool: ``orbitwars.training.pfsp_pool`` (not yet created).
  * Distillation: ``orbitwars.nn.distill`` (not yet created).

**Dependencies:** ``torch`` 2.11.0+cpu is installed as of W3 tail. CPU-only
for now; swap to the CUDA build on the local RTX 3070 when PPO training
starts. The torch modules below are live — obs_encode.py is shipped,
action decode (ACTION_LOOKUP below + planet_to_grid_coords) is in place,
and forward-pass smoke tests pin shape + dtype (see tests/test_nn_forward.py).
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ConvPolicyCfg:
    """Hyperparameters for the conv policy.

    Load-bearing values picked from Lux S3 winner analysis:
      * `grid_h=grid_w=50`: half the 100x100 board. Each cell covers a
        2x2 unit area — fine enough to localize planets (radius 1-3) and
        fleets, coarse enough that the activation volume stays manageable
        (50x50 @ 64 channels = 160KB per layer).
      * `n_channels=12`: see ``feature_channels()`` for the breakdown.
      * `backbone_channels=64` / `n_blocks=6`: ~1M params at H=W=50.
        Distills cleanly to <2M final; fits within the W4 GPU budget
        (1-2 days training on one T4).
      * `n_action_channels=8`: per-planet action distribution — 4 angle
        buckets x 2 ship-fraction buckets. Continuous angle gets
        re-introduced via BOKR-style sampling around the arg-max angle
        at inference time (see ``mcts/bokr_widen.py``).
    """

    grid_h: int = 50
    grid_w: int = 50
    n_channels: int = 12              # input feature channels
    backbone_channels: int = 64
    n_blocks: int = 6
    n_action_channels: int = 8        # per-cell action distribution
    value_hidden: int = 128


def feature_channels() -> Tuple[str, ...]:
    """Documented list of the ``n_channels`` input features.

    The feature tensor shape is ``(batch, C, H, W)`` where ``C = len(...)``.
    Order is load-bearing — the encoder in ``features/obs_encode.py``
    and the decoder heads must agree. Adding a channel requires a
    retrain; prefer slotting unused fields in at the END rather than
    reordering.
    """
    return (
        "ship_count_p0",           # 0. my-side ship density (sqrt-scaled)
        "ship_count_p1",           # 1. enemy ship density (sqrt-scaled)
        "production_p0",           # 2. owned-planet production rate, mine
        "production_p1",           # 3. owned-planet production rate, enemy
        "production_neutral",      # 4. neutral planet production
        "planet_radius",           # 5. planet radius at cell (or 0)
        "is_orbiting",             # 6. 1 if rotating planet occupies cell
        "is_comet",                # 7. 1 if comet-bearing planet
        "sun_distance",            # 8. pre-computed distance to (50,50)
        "fleet_angle_cos",         # 9. cos(angle) of any fleet at cell
        "fleet_angle_sin",         # 10. sin(angle)
        "turn_phase",              # 11. step / 500 (broadcast scalar)
    )


# ---------------------------------------------------------------------------
# Torch module — live. GroupNorm rather than BatchNorm2d so batch=1 MCTS
# leaf evaluation does not leak running-mean drift across games. The
# GroupNorm group count defaults to min(8, C); at C=64 that is 8 groups
# of 8 channels each — standard choice from Wu & He (2018).
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    """Standard 3x3 conv residual block, pre-activation variant.

    Uses GroupNorm (not BatchNorm2d) so inference at batch=1 — which is
    the MCTS leaf-eval regime — is statistically identical to training.
    BatchNorm2d running-stats drift across PFSP checkpoint boundaries in
    subtle ways; GroupNorm sidesteps it entirely.
    """

    def __init__(self, c: int, num_groups: int = 8):
        super().__init__()
        groups = min(num_groups, c)
        self.gn1 = nn.GroupNorm(groups, c)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.gn2 = nn.GroupNorm(groups, c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.relu(self.gn1(x)))
        y = self.conv2(F.relu(self.gn2(y)))
        return x + y


class ConvPolicy(nn.Module):
    """Centralized per-entity-grid conv policy + value network.

    Input: ``(B, cfg.n_channels, cfg.grid_h, cfg.grid_w)`` feature grid
    from ``orbitwars.features.obs_encode.encode_grid``.

    Outputs:
      * ``policy_logits``: ``(B, cfg.n_action_channels, H, W)`` — read
        the logits at each owned-planet grid cell via
        ``planet_to_grid_coords`` then softmax + decode with
        ``ACTION_LOOKUP``.
      * ``value``: ``(B, 1)`` scalar in ``[-1, 1]``.
    """

    def __init__(self, cfg: ConvPolicyCfg):
        super().__init__()
        self.cfg = cfg
        self.stem = nn.Conv2d(
            cfg.n_channels, cfg.backbone_channels, 3, padding=1
        )
        self.blocks = nn.ModuleList(
            [ResBlock(cfg.backbone_channels) for _ in range(cfg.n_blocks)]
        )
        self.policy_head = nn.Conv2d(
            cfg.backbone_channels, cfg.n_action_channels, 1
        )
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cfg.backbone_channels, cfg.value_hidden),
            nn.ReLU(),
            nn.Linear(cfg.value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
          x: ``(B, cfg.n_channels, cfg.grid_h, cfg.grid_w)`` input grid.

        Returns:
          policy_logits, value — see class docstring for shapes.
        """
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        policy = self.policy_head(h)
        value = self.value_head(h)
        return policy, value


def param_count_estimate(cfg: ConvPolicyCfg) -> int:
    """Rough parameter count sanity-check.

    Dominant terms:
      * Stem: C_in * C * 9
      * Each ResBlock: 2 * (C * C * 9)
      * Policy head: C * n_action_channels
      * Value head: C * value_hidden + value_hidden

    Returns the integer estimate. Useful for gate-checks like "student
    must be <2M params" without actually constructing the torch module.
    """
    c = cfg.backbone_channels
    c_in = cfg.n_channels
    stem = c_in * c * 9
    blocks = cfg.n_blocks * 2 * c * c * 9
    policy = c * cfg.n_action_channels
    value = c * cfg.value_hidden + cfg.value_hidden
    # Add ~10% for biases + batchnorm scale/shift.
    total = int((stem + blocks + policy + value) * 1.10)
    return total


# ---------------------------------------------------------------------------
# Decode helpers — convert policy grid -> per-planet action distribution.
#
# The NN emits a (C_action, H, W) grid. At inference time we read it at
# the (grid_x, grid_y) of each owned planet: `probs[C_action] = softmax(logits)`.
# Then map the C_action index to (angle_bucket, ship_fraction) via
# ``ACTION_LOOKUP`` below. BOKR-style continuous-angle refinement happens
# AFTER decode (mcts/bokr_widen.py expands the top-k angles into a fine
# grid and runs MCTS over them).
# ---------------------------------------------------------------------------

# n_action_channels = 4 angle buckets x 2 ship fractions = 8 channels.
# angle_bucket_0..3 = {0, pi/2, pi, 3pi/2} with BOKR expanding to the
# continuous angle at inference. ship_frac_0 = 50%, ship_frac_1 = 100%.
# Quarter-angle resolution is coarse on purpose: intercepts land within
# +/- ~90 deg of the target direction, so each of the 4 buckets maps
# cleanly to "toward quadrant X". BOKR refines.
ACTION_LOOKUP = (
    # (angle_bucket, ship_frac)  description
    (0, 0.5),  # 0: East, 50%
    (0, 1.0),  # 1: East, 100%
    (1, 0.5),  # 2: North, 50%
    (1, 1.0),  # 3: North, 100%
    (2, 0.5),  # 4: West, 50%
    (2, 1.0),  # 5: West, 100%
    (3, 0.5),  # 6: South, 50%
    (3, 1.0),  # 7: South, 100%
)


def planet_to_grid_coords(
    x: float, y: float, cfg: ConvPolicyCfg
) -> Tuple[int, int]:
    """Map continuous planet position -> (grid_y, grid_x) cell index.

    Board is ``[0, 100] x [0, 100]``; grid is ``[0, grid_h] x [0, grid_w]``.
    Standard conv convention uses (row, col) i.e. (y, x) order.

    Clamps to ``[0, grid_h-1]`` / ``[0, grid_w-1]`` to handle edge cases
    where a planet sits exactly on the boundary.
    """
    gy = int(y * cfg.grid_h / 100.0)
    gx = int(x * cfg.grid_w / 100.0)
    gy = max(0, min(cfg.grid_h - 1, gy))
    gx = max(0, min(cfg.grid_w - 1, gx))
    return gy, gx



# --- inlined: orbitwars/nn/nn_prior.py ---

"""NN prior bridge: ConvPolicy logits -> per-PlanetMove prior.

This module is the inference-time bridge between a trained
``ConvPolicy`` checkpoint (output of ``tools/bc_warmstart.py``) and the
MCTS root candidate enumeration in ``orbitwars.mcts.actions``. Given an
obs and a list of ``PlanetMove`` candidates, we:

  1. Encode the obs to a (1, C, H, W) tensor via ``encode_grid``.
  2. Forward through the loaded ConvPolicy → policy_logits (1, 8, H, W).
  3. For each candidate, look up the logit at
     ``policy_logits[:, channel, gy, gx]`` where (gy, gx) is the source
     planet's grid cell and ``channel`` is the bucket nearest to the
     candidate's (angle, ship_fraction).
  4. Softmax per planet → returns a NEW list of PlanetMove with
     ``prior`` populated from the NN.

Why this is a separate module (not a method on MCTSAgent):
  * Pure function, easy to unit-test against a fake checkpoint.
  * Lets us A/B "heuristic prior vs. NN prior" without touching MCTS
    internals — we just swap which prior fn the agent calls.
  * Decouples from torch import path — agents that don't use a NN don't
    pay torch's import cost on the Kaggle hot path.

NOT here (deliberately):
  * Value head usage. ``bc_warmstart.py`` only trains the policy head;
    ``model.value_head`` is randomly initialized and would feed garbage
    into rollouts. A future ``nn_value_bootstrap.py`` will land once
    we have a value-trained checkpoint (PPO, MCTS-distill, or joint BC).
  * Action-channel learning. The mapping (move.angle, move.ships) ->
    channel index is fixed by ``ACTION_LOOKUP``. If we change the
    ConvPolicy action factorization we'd need a new bridge.

Smoke-test path:
  ``tools/validate_bc_checkpoint.py`` loads the checkpoint and reports
  schema integrity. This module's tests build a fake ConvPolicy with
  hand-picked weights so the prior assignment is deterministic — no
  real checkpoint required to run them.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np



# Number of discrete (angle_bucket, ship_frac) actions in ConvPolicy
# output channels. Pinned to ACTION_LOOKUP length — must agree with the
# trained model.
N_ACTION_CHANNELS = len(ACTION_LOOKUP)
# Number of angle buckets implied by ACTION_LOOKUP. Used to map a
# continuous candidate angle -> a bucket index.
N_ANGLE_BUCKETS = max(b for b, _ in ACTION_LOOKUP) + 1


# ---------------------------------------------------------------------------
# Bucketing helpers
# ---------------------------------------------------------------------------


def angle_to_bucket(angle: float) -> int:
    """Map a continuous angle (radians) to one of N_ANGLE_BUCKETS.

    ACTION_LOOKUP shipped with 4 buckets at {0, pi/2, pi, 3pi/2} radians
    (East, North, West, South). We bucket by nearest center on the
    circle, with the wrap-around at +pi handled correctly.
    """
    # Normalize to [0, 2*pi).
    a = angle % (2.0 * math.pi)
    # Each bucket covers a 2*pi / N range; bucket center is at
    # bucket_idx * (2*pi / N).
    step = 2.0 * math.pi / N_ANGLE_BUCKETS
    # Shift so that bucket 0 is centered at 0 (range [-step/2, +step/2)).
    shifted = (a + step / 2.0) % (2.0 * math.pi)
    return int(shifted // step)


def ship_fraction_to_bucket(used: int, available: int) -> float:
    """Map a candidate's ships/available ratio to ACTION_LOOKUP's nearest
    discrete fraction. ACTION_LOOKUP currently uses {0.5, 1.0}; a 0.25
    candidate snaps to 0.5 (the closer of the two).
    """
    if available <= 0:
        return 1.0
    frac = max(0.0, min(1.0, float(used) / float(available)))
    # ACTION_LOOKUP has fractions sorted ascending — pick nearest.
    fracs_seen: List[float] = []
    for _b, f in ACTION_LOOKUP:
        if f not in fracs_seen:
            fracs_seen.append(f)
    fracs_seen.sort()
    return min(fracs_seen, key=lambda f: abs(f - frac))


def candidate_to_channel(move: PlanetMove, available: int) -> int:
    """Find the ACTION_LOOKUP channel index whose (angle_bucket, ship_frac)
    is closest to a PlanetMove's continuous (angle, ships).

    HOLD moves don't have a natural channel — caller should treat them
    separately (typically a small fixed prior under the NN).
    """
    if move.is_hold:
        # Caller must handle holds explicitly; this is a sentinel that
        # signals "no NN channel applies here". Returning -1 lets callers
        # fall back to a uniform-or-mean prior.
        return -1
    bucket = angle_to_bucket(move.angle)
    frac = ship_fraction_to_bucket(int(move.ships), int(available))
    # Linear scan — N_ACTION_CHANNELS is 8, not worth a lookup table.
    for ch, (b, f) in enumerate(ACTION_LOOKUP):
        if b == bucket and abs(f - frac) < 1e-6:
            return ch
    # Fallback: nearest channel by (bucket, frac) distance.
    best_ch = 0
    best_d = float("inf")
    for ch, (b, f) in enumerate(ACTION_LOOKUP):
        d = abs(b - bucket) * 1.0 + abs(f - frac) * 0.5
        if d < best_d:
            best_d = d
            best_ch = ch
    return best_ch


# ---------------------------------------------------------------------------
# Loading + inference
# ---------------------------------------------------------------------------


def load_conv_policy(
    checkpoint_path: Path | str, device: Optional[str] = None,
) -> Tuple[ConvPolicy, ConvPolicyCfg]:
    """Load a ConvPolicy checkpoint produced by tools/bc_warmstart.py.

    Returns (model, cfg). Model is in eval() mode and on the requested
    device (default: cpu — the Kaggle ladder is CPU-only so we want the
    inference-time path to mirror submission semantics by default).

    Raises FileNotFoundError if the checkpoint is missing.
    """
    import torch

    p = Path(checkpoint_path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")

    ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    # Two flavors: full checkpoint (`model_state` + `cfg`) saved at end of
    # training, or partial checkpoint (`model_state_dict`, `_partial=True`)
    # eagerly written each time val-acc improves. The latter does not carry
    # cfg, so we fall back to ConvPolicyCfg defaults — the BC warm-start
    # script always trains with defaults unless --backbone-channels /
    # --n-blocks are passed, in which case the eager path is unsafe and the
    # caller must pass the full checkpoint.
    if "model_state" in ckpt and "cfg" in ckpt:
        cfg = ConvPolicyCfg(**ckpt["cfg"])
        model = ConvPolicy(cfg)
        model.load_state_dict(ckpt["model_state"])
    elif "model_state_dict" in ckpt:
        cfg = ConvPolicyCfg()
        model = ConvPolicy(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise ValueError(
            f"checkpoint {p} has unrecognized keys {sorted(ckpt.keys())}; "
            "expected 'model_state'+'cfg' (full) or 'model_state_dict' (partial)."
        )
    model.eval()
    if device is not None and device != "cpu":
        model = model.to(device)
    return model, cfg


def nn_priors_for_planet(
    obs: Any,
    player_id: int,
    moves: Sequence[PlanetMove],
    available_ships: int,
    model: ConvPolicy,
    cfg: ConvPolicyCfg,
    *,
    hold_neutral_prob: float = 0.05,
    temperature: float = 1.0,
) -> List[float]:
    """Compute NN priors for one planet's candidate moves.

    Args:
      obs: Kaggle obs for ``player_id``'s view.
      player_id: seat id (0..3).
      moves: candidates from ``generate_per_planet_moves`` for ONE planet.
      available_ships: ships at the source planet (for fraction mapping).
      model: loaded ConvPolicy, eval mode, weights frozen.
      cfg: matching ConvPolicyCfg.
      hold_neutral_prob: per-planet prior mass reserved for HOLD moves
        before renormalization. The NN policy head doesn't model "do
        nothing" explicitly, so we floor it. Small (0.05 default) to
        keep the NN's signal dominant when it has a strong opinion.
      temperature: softmax temperature on the NN's per-channel logits.
        1.0 = pristine softmax. >1 = flatter (more exploration).

    Returns:
      ``len(moves)``-list of priors that sum to 1.0. Returns a uniform
      distribution if ``moves`` is empty (defensive — caller would have
      filtered).
    """
    import torch

    n = len(moves)
    if n == 0:
        return []

    # All PlanetMoves in `moves` are by contract from the same planet.
    src_pid = int(moves[0].from_pid)
    # Find planet (x, y) — scan obs.planets.
    planets = _obs_get_list(obs, "planets")
    pos: Optional[Tuple[float, float]] = None
    for pl in planets:
        if int(pl[0]) == src_pid:
            pos = (float(pl[2]), float(pl[3]))
            break
    if pos is None:
        # Lost the planet? Fall back to uniform.
        return [1.0 / n] * n

    # Encode + forward.
    grid = encode_grid(obs, player_id, cfg)
    x = torch.from_numpy(grid).unsqueeze(0)  # (1, C, H, W)
    with torch.no_grad():
        logits, _value = model(x)  # logits: (1, 8, H, W)

    gy, gx = planet_to_grid_coords(pos[0], pos[1], cfg)
    cell_logits = logits[0, :, gy, gx].cpu().numpy()  # (8,)

    # Per-move logit lookup. HOLD gets a fixed log-prior derived from
    # ``hold_neutral_prob`` so the softmax balance is configurable.
    raw: List[float] = []
    has_hold = any(m.is_hold for m in moves)
    # Pre-compute the HOLD log-prior in a way that's consistent with
    # softmax: we want the FINAL HOLD prior to be roughly
    # ``hold_neutral_prob`` of the per-planet mass, regardless of how
    # peaked the NN cells are. Easy approximation: set HOLD's log to the
    # mean of the channel logits (so it doesn't dominate or vanish) plus
    # ``log(hold_neutral_prob / (1 - hold_neutral_prob))`` as an offset.
    if has_hold:
        mean_log = float(np.mean(cell_logits))
        offset = math.log(
            max(1e-6, hold_neutral_prob)
            / max(1e-6, 1.0 - hold_neutral_prob)
        )
        hold_log = mean_log + offset
    else:
        hold_log = 0.0  # unused

    for m in moves:
        if m.is_hold:
            raw.append(hold_log)
        else:
            ch = candidate_to_channel(m, available_ships)
            if ch < 0:
                # Defensive — channel mapping failed; treat as HOLD.
                raw.append(hold_log)
            else:
                raw.append(float(cell_logits[ch]))

    # Softmax with temperature.
    t = max(1e-6, float(temperature))
    m_max = max(raw)
    exps = [math.exp((r - m_max) / t) for r in raw]
    z = sum(exps)
    if z <= 0:
        return [1.0 / n] * n
    return [e / z for e in exps]


def make_nn_prior_fn(
    model: ConvPolicy,
    cfg: ConvPolicyCfg,
    *,
    hold_neutral_prob: float = 0.05,
    temperature: float = 1.0,
):
    """Closure factory: returns a function that fills in NN priors over a
    ``Dict[planet_id, List[PlanetMove]]`` (the shape produced by
    ``generate_per_planet_moves``).

    Returned function signature:
      ``fn(obs, player_id, moves_by_planet, available_by_planet)
       -> Dict[planet_id, List[PlanetMove]]``

    where ``available_by_planet`` is ``{planet_id: ships_at_source}``.
    The returned dict has the SAME PlanetMove objects with their
    ``prior`` field overwritten — wraps in a fresh PlanetMove so the
    upstream heuristic prior is preserved if the caller wants both.
    """

    def fn(
        obs: Any,
        player_id: int,
        moves_by_planet: Dict[int, List[PlanetMove]],
        available_by_planet: Dict[int, int],
    ) -> Dict[int, List[PlanetMove]]:
        out: Dict[int, List[PlanetMove]] = {}
        for pid, moves in moves_by_planet.items():
            avail = int(available_by_planet.get(pid, 0))
            priors = nn_priors_for_planet(
                obs, player_id, moves, avail, model, cfg,
                hold_neutral_prob=hold_neutral_prob,
                temperature=temperature,
            )
            new_moves: List[PlanetMove] = []
            for m, p in zip(moves, priors):
                # PlanetMove is frozen — make a new one with NN prior.
                new_moves.append(
                    PlanetMove(
                        from_pid=m.from_pid,
                        angle=m.angle,
                        ships=m.ships,
                        target_pid=m.target_pid,
                        kind=m.kind,
                        prior=p,
                        raw_score=m.raw_score,
                    )
                )
            out[pid] = new_moves
        return out

    return fn


# ---------------------------------------------------------------------------
# Tiny obs-helper to avoid pulling the full obs_encode internals.
# ---------------------------------------------------------------------------


def _obs_get_list(obs: Any, key: str) -> List[Any]:
    """Read a list-typed obs field whether obs is a dict, AttrDict, or
    ParsedObs. Returns ``[]`` if the field is missing or None."""
    if isinstance(obs, dict):
        v = obs.get(key, None)
    else:
        v = getattr(obs, key, None)
    return list(v) if v is not None else []



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
        move_prior_fn: Optional[Any] = None,
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
            move_prior_fn=move_prior_fn,
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
    'agg_early_game': 0.5,
    'comet_max_time_mismatch': 5.0,
    'early_game_cutoff_turn': 104.60161165543384,
    'expand_bias': 0.7148756834104646,
    'expand_cooldown_turns': 3.6464331186834906,
    'keep_reserve_ships': 0.0,
    'max_launch_fraction': 0.9877046770973957,
    'min_launch_size': 30.0,
    'mult_comet': 5.0,
    'mult_enemy': 5.0,
    'mult_neutral': 2.002924634653468,
    'mult_reinforce_ally': 0.0,
    'ships_safety_margin': 0.9854161130378981,
    'sun_avoidance_epsilon': 0.005300802184648653,
    'w_distance_cost': 0.0,
    'w_production': 20.0,
    'w_ships_cost': 0.0,
    'w_travel_cost': 1.444902944661085,
})


# --- NN prior bootstrap (--nn-checkpoint) ---
import base64 as _bundle_b64
import io as _bundle_io
import torch as _bundle_torch
_BUNDLE_BC_CKPT_B64 = (
    'UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAfAEMAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRh'
    'LnBrbEZCPwBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlqAAn1xAChYEAAAAG1vZGVsX3N0YXRlX2RpY3RxAX1xAihYCwAAAHN0ZW0u'
    'd2VpZ2h0cQNjdG9yY2guX3V0aWxzCl9yZWJ1aWxkX3RlbnNvcl92MgpxBCgoWAcAAABzdG9yYWdl'
    'cQVjdG9yY2gKRmxvYXRTdG9yYWdlCnEGWAEAAAAwcQdYAwAAAGNwdXEITYANdHEJUUsAKEsgSwxL'
    'A0sDdHEKKEtsSwlLA0sBdHELiWNjb2xsZWN0aW9ucwpPcmRlcmVkRGljdApxDClScQ10cQ5ScQ9Y'
    'CQAAAHN0ZW0uYmlhc3EQaAQoKGgFaAZYAQAAADFxEWgISyB0cRJRSwBLIIVxE0sBhXEUiWgMKVJx'
    'FXRxFlJxF1gTAAAAYmxvY2tzLjAuZ24xLndlaWdodHEYaAQoKGgFaAZYAQAAADJxGWgISyB0cRpR'
    'SwBLIIVxG0sBhXEciWgMKVJxHXRxHlJxH1gRAAAAYmxvY2tzLjAuZ24xLmJpYXNxIGgEKChoBWgG'
    'WAEAAAAzcSFoCEsgdHEiUUsASyCFcSNLAYVxJIloDClScSV0cSZScSdYFQAAAGJsb2Nrcy4wLmNv'
    'bnYxLndlaWdodHEoaAQoKGgFaAZYAQAAADRxKWgITQAkdHEqUUsAKEsgSyBLA0sDdHErKE0gAUsJ'
    'SwNLAXRxLIloDClScS10cS5ScS9YEwAAAGJsb2Nrcy4wLmNvbnYxLmJpYXNxMGgEKChoBWgGWAEA'
    'AAA1cTFoCEsgdHEyUUsASyCFcTNLAYVxNIloDClScTV0cTZScTdYEwAAAGJsb2Nrcy4wLmduMi53'
    'ZWlnaHRxOGgEKChoBWgGWAEAAAA2cTloCEsgdHE6UUsASyCFcTtLAYVxPIloDClScT10cT5ScT9Y'
    'EQAAAGJsb2Nrcy4wLmduMi5iaWFzcUBoBCgoaAVoBlgBAAAAN3FBaAhLIHRxQlFLAEsghXFDSwGF'
    'cUSJaAwpUnFFdHFGUnFHWBUAAABibG9ja3MuMC5jb252Mi53ZWlnaHRxSGgEKChoBWgGWAEAAAA4'
    'cUloCE0AJHRxSlFLAChLIEsgSwNLA3RxSyhNIAFLCUsDSwF0cUyJaAwpUnFNdHFOUnFPWBMAAABi'
    'bG9ja3MuMC5jb252Mi5iaWFzcVBoBCgoaAVoBlgBAAAAOXFRaAhLIHRxUlFLAEsghXFTSwGFcVSJ'
    'aAwpUnFVdHFWUnFXWBMAAABibG9ja3MuMS5nbjEud2VpZ2h0cVhoBCgoaAVoBlgCAAAAMTBxWWgI'
    'SyB0cVpRSwBLIIVxW0sBhXFciWgMKVJxXXRxXlJxX1gRAAAAYmxvY2tzLjEuZ24xLmJpYXNxYGgE'
    'KChoBWgGWAIAAAAxMXFhaAhLIHRxYlFLAEsghXFjSwGFcWSJaAwpUnFldHFmUnFnWBUAAABibG9j'
    'a3MuMS5jb252MS53ZWlnaHRxaGgEKChoBWgGWAIAAAAxMnFpaAhNACR0cWpRSwAoSyBLIEsDSwN0'
    'cWsoTSABSwlLA0sBdHFsiWgMKVJxbXRxblJxb1gTAAAAYmxvY2tzLjEuY29udjEuYmlhc3FwaAQo'
    'KGgFaAZYAgAAADEzcXFoCEsgdHFyUUsASyCFcXNLAYVxdIloDClScXV0cXZScXdYEwAAAGJsb2Nr'
    'cy4xLmduMi53ZWlnaHRxeGgEKChoBWgGWAIAAAAxNHF5aAhLIHRxelFLAEsghXF7SwGFcXyJaAwp'
    'UnF9dHF+UnF/WBEAAABibG9ja3MuMS5nbjIuYmlhc3GAaAQoKGgFaAZYAgAAADE1cYFoCEsgdHGC'
    'UUsASyCFcYNLAYVxhIloDClScYV0cYZScYdYFQAAAGJsb2Nrcy4xLmNvbnYyLndlaWdodHGIaAQo'
    'KGgFaAZYAgAAADE2cYloCE0AJHRxilFLAChLIEsgSwNLA3RxiyhNIAFLCUsDSwF0cYyJaAwpUnGN'
    'dHGOUnGPWBMAAABibG9ja3MuMS5jb252Mi5iaWFzcZBoBCgoaAVoBlgCAAAAMTdxkWgISyB0cZJR'
    'SwBLIIVxk0sBhXGUiWgMKVJxlXRxllJxl1gTAAAAYmxvY2tzLjIuZ24xLndlaWdodHGYaAQoKGgF'
    'aAZYAgAAADE4cZloCEsgdHGaUUsASyCFcZtLAYVxnIloDClScZ10cZ5ScZ9YEQAAAGJsb2Nrcy4y'
    'LmduMS5iaWFzcaBoBCgoaAVoBlgCAAAAMTlxoWgISyB0caJRSwBLIIVxo0sBhXGkiWgMKVJxpXRx'
    'plJxp1gVAAAAYmxvY2tzLjIuY29udjEud2VpZ2h0cahoBCgoaAVoBlgCAAAAMjBxqWgITQAkdHGq'
    'UUsAKEsgSyBLA0sDdHGrKE0gAUsJSwNLAXRxrIloDClSca10ca5Sca9YEwAAAGJsb2Nrcy4yLmNv'
    'bnYxLmJpYXNxsGgEKChoBWgGWAIAAAAyMXGxaAhLIHRxslFLAEsghXGzSwGFcbSJaAwpUnG1dHG2'
    'UnG3WBMAAABibG9ja3MuMi5nbjIud2VpZ2h0cbhoBCgoaAVoBlgCAAAAMjJxuWgISyB0cbpRSwBL'
    'IIVxu0sBhXG8iWgMKVJxvXRxvlJxv1gRAAAAYmxvY2tzLjIuZ24yLmJpYXNxwGgEKChoBWgGWAIA'
    'AAAyM3HBaAhLIHRxwlFLAEsghXHDSwGFccSJaAwpUnHFdHHGUnHHWBUAAABibG9ja3MuMi5jb252'
    'Mi53ZWlnaHRxyGgEKChoBWgGWAIAAAAyNHHJaAhNACR0ccpRSwAoSyBLIEsDSwN0ccsoTSABSwlL'
    'A0sBdHHMiWgMKVJxzXRxzlJxz1gTAAAAYmxvY2tzLjIuY29udjIuYmlhc3HQaAQoKGgFaAZYAgAA'
    'ADI1cdFoCEsgdHHSUUsASyCFcdNLAYVx1IloDClScdV0cdZScddYEgAAAHBvbGljeV9oZWFkLndl'
    'aWdodHHYaAQoKGgFaAZYAgAAADI2cdloCE0AAXRx2lFLAChLCEsgSwFLAXRx2yhLIEsBSwFLAXRx'
    '3IloDClScd10cd5Scd9YEAAAAHBvbGljeV9oZWFkLmJpYXNx4GgEKChoBWgGWAIAAAAyN3HhaAhL'
    'CHRx4lFLAEsIhXHjSwGFceSJaAwpUnHldHHmUnHnWBMAAAB2YWx1ZV9oZWFkLjIud2VpZ2h0ceho'
    'BCgoaAVoBlgCAAAAMjhx6WgITQAQdHHqUUsAS4BLIIZx60sgSwGGceyJaAwpUnHtdHHuUnHvWBEA'
    'AAB2YWx1ZV9oZWFkLjIuYmlhc3HwaAQoKGgFaAZYAgAAADI5cfFoCEuAdHHyUUsAS4CFcfNLAYVx'
    '9IloDClScfV0cfZScfdYEwAAAHZhbHVlX2hlYWQuNC53ZWlnaHRx+GgEKChoBWgGWAIAAAAzMHH5'
    'aAhLgHRx+lFLAEsBS4CGcftLgEsBhnH8iWgMKVJx/XRx/lJx/1gRAAAAdmFsdWVfaGVhZC40LmJp'
    'YXNyAAEAAGgEKChoBWgGWAIAAAAzMXIBAQAAaAhLAXRyAgEAAFFLAEsBhXIDAQAASwGFcgQBAACJ'
    'aAwpUnIFAQAAdHIGAQAAUnIHAQAAdVgMAAAAYmVzdF92YWxfYWNjcggBAABHP9m99GKEWFhYBQAA'
    'AGVwb2NocgkBAABLAVgIAAAAX3BhcnRpYWxyCgEAAIhYAwAAAGNmZ3ILAQAAfXIMAQAAKFgGAAAA'
    'Z3JpZF9ocg0BAABLMlgGAAAAZ3JpZF93cg4BAABLMlgKAAAAbl9jaGFubmVsc3IPAQAASwxYEQAA'
    'AGJhY2tib25lX2NoYW5uZWxzchABAABLIFgIAAAAbl9ibG9ja3NyEQEAAEsDWBEAAABuX2FjdGlv'
    'bl9jaGFubmVsc3ISAQAASwhYDAAAAHZhbHVlX2hpZGRlbnITAQAAS4B1WAsAAABtb2RlbF9zdGF0'
    'ZXIUAQAAaAJ1LlBLBwjqIPTzkAsAAJALAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAACYAHABi'
    'Y193YXJtc3RhcnRfc21hbGxfY3B1Ly5mb3JtYXRfdmVyc2lvbkZCGABaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWloxUEsHCLfv3IMBAAAAAQAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAKQAoAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvLnN0b3JhZ2VfYWxpZ25tZW50RkIkAFpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWjY0UEsHCD93cekCAAAAAgAAAFBLAwQAAAgIAAAAAAAAAAAA'
    'AAAAAAAAAAAAIAAwAGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvYnl0ZW9yZGVyRkIsAFpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpabGl0dGxlUEsHCIU94xkGAAAABgAA'
    'AFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHQAvAGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0'
    'YS8wRkIrAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlrMfPq7MCUp'
    'PUlE5b2N30y96Od3vUGXaz3A+kC8RtWEPQkIlzyhjOY8liLrvBWcu7zg/qO9dSCfvUTUGb2PHaQ7'
    'siJRPQakcz3ToZK9wJQ5vRI0FD2dV6I9TBihvHlTij0vUou8TuIbu59asT04VLO9VbN6vctrrbx2'
    'e/i8UOSYPQwxc704sEy9TJ2Dvck7tL2MZ3W9Vmm5PYwnRz0Hr2o9bufoulMaYL3jXXk84vW8vQVb'
    'kr0FBl+9voRsPca8cj1j5y29tvptu1CVaD1mE8E9Gw3VPEOvQjztZFM9LCd6vfOKyDzvEZa97T6S'
    'vUs0W72qjio9T7ftPLUXbr0HpaU8iSsuPd+fobz/MY+4f6WkPFTAgz3IQ7o9AK93vexdCL02XBA9'
    'RBScPYL9sz2s7649y93IPPOdtb2uzwA8humPvWhBu70oj4k9FBCZPVM8sL0MCu87P0VLvJJFWLzk'
    'sR69fgMwPSWPTb0NOtM8ckZDPeG5hD3DayA93jO4vYayTL3ffQg9WgdHPODoer025Ua9S9THPUHU'
    'kz3FeBC9mB2OvGMUrb0/zJQ7yRCHveYNir2emxm8mBUhPft2QL17d7s9FD2uvYBE8D2GU5s9kunw'
    'vUxjjLtoTYK57LAfPK95jDyQLiE9ZJxEvPZRV72LFC09/UDvvZk2Xr2BzuS8Tx0BvSd3Ir0KRnK9'
    'SoWevJLZYDweh4C9nw8fOy9Ehj2PfrE7SJ7BOxsRFjz7Mgg9bsZWPe0x0j2wS4k9ne6xPRVuPLvs'
    '/rS9ntpPvXdcmT2BSs88fGdvvVCap70cssO9DrPAvYP8vrwaCkU9GyeOPcwLxL263qg9buSdvQS6'
    '5bwv6Be9LqhFvNrhubzqk7S9miCJvSw7pLzKSlI8gD0cvcsm3Dxkapm9ec4aO0rdmj1/viW9D8HE'
    'PToxCT0JMrO9RmuFPaWNALxxn0a9u3axPbm9mb2GJ/G6Ql2guzkRej3slTE922I2PZkavTxoG9q8'
    'gxRIvJ5fqr0mTCc9brKmPamVmT21jna9bzygOyiIe70/xPi8ZHlbvSi6Vz0d81+9JOrmu0PPrrts'
    'Ulk90culvZl0hL1Sz3e90bpkPegsFL0E+qY98UIUPWYwgDyg5HK789yrvH9kSDwET728FdIjulYS'
    'bDwyXsy9RZ42vWWd5T23HI296cI8vPZ5BD5neTk8l5O4PKydqr1Wrj09Cd2avZqUu7xLQji9uX8+'
    'vM3XCbz0V4o97+LvO4hpkz1teFs9MTSTvX/MKT1O+IS9CC8mvZPJ9zzJlfo8MKSIPXeBXLzUxvw6'
    'jIR3PanUub0DlpU9kxyhvfZ1p7qgF0y8XapUvUdf4TwhD6o9ffX1vCXxd71YI8a8UkPTve3tWz3d'
    'ok+9ufSNvdzqhb3tNk4918BJPenolD28nyA95E0AvbvrwLyOV/M8jciYPXoJMD2FQD+9D6t2vQja'
    'vb1uy8U8aWQ9vfTmkb2HMqg9CQ2IvUlWlTsNOhk9gLf5u6rChb0KIJy9yqyvPc19wjz+RK89SnsJ'
    'vYjy4zwgi+K8YFeCvKPzcL0AuhY9OlTgPG5q7zpA8l+8GNN5vRdSoT2vpLO9Bfc4PfAuALwocZC9'
    'oH2LPKN5Ub2a7GE8bRc7PDp7ob24zuI7nauTPEM99jw/PAi9L+6wvXFqyLw8NgA9w0O0vOzyMz1d'
    'QmU9Tw1rvUV0g7siSf88iIDXuDydJDtCUPY7HAqivTVQ0jtWS5c9bNflPUSYhT3LRR49HtuIPGYN'
    'Vr3oypi93CPWvdSMDj6q1SE9cGiFvbINDT26G4o98MW0PaZflT0BdJQ9VLjfvM0C7Lztzpc9XC5e'
    'PVKGYr1Bua097tstPQlVPD02bGO9nc/OPOdLlLwpop09MVCavLfZbj2KLl08CoIrPev8Dr26AlI9'
    '+naOvQGQZD0iSaC9qGXOvcHyqL3OSaU9LBFDPRcN1z0Ylb49J7zEvbKufb0aU6y8ewuBvalBoj22'
    'UpW9PkpXvRMVyr2riSK95jCcPTFxwT3uYXQ9pieSPT2ywD2q4tG82ictPUB6nj0MvTc92wS9PQ3F'
    'Sz1CMlU9j/IeveJUgj3obEI8dwOyvHtVRb00Lw69ckeZvUk9fz1vDIM99HhJvFi+Mb2ulKA9S/lY'
    'Pe8lzjq6c/e8Ye2wPdSYVj23Loe9O5G7PcUpsz1Z5dG8zXVFPUFILr056fG75GuhPdbEajyM84S9'
    'uoJGvVdSAr1k1p27nDyoO+NlmD2nx/07ZdxavW+Nd7obP1c9yHhiPcJ3pr1C1Ce8YsOBvdzRDL2d'
    'uig9sF54vaAoLj3ze5c99BGGPLiBlr0zc4m8hbG+u7FNsb1BN0o9luHZvWeVw71sVdC9SEsAvVfW'
    'dD1L4L6932ydPRFbSL14rio9vQP0O/V7u71J1pg95fJHPUqe1D3GVzs9TTh8vWlApz09CXk7ZQVN'
    'PWjRpr3ujem8D3LMve+OO73Wa8o8BiqjvdXP6Dx7rBc9hfBrPD4dkL2thZG9W30BvbSCOj0eBdg9'
    '7LAePcE5vrxa49E8kKm3vQ2em71vryM9Y2NvvU81YD15+4+9qG49vTJTnb2VzUM9VGikuyEfkD1N'
    'o2Q93UdKPU6TgjzVGg+9cdOAvNvoiDzuW1A9sheEvVEPnz2oJ9q87PC+vT79Wb0CmK897PjGPWpr'
    '3L2jL6I9iE1hPfz7Wj2kMJw9YX1ovaOnFz0jrh09bRPUvJ0vEz1IDO28zpOEPXGY+byKXx69wBML'
    'PV9hsTxb5Fu9HdXeOuiibD1Lzrm8A6KhPDUx7rwm7zA9NXaNvbsgK73Fu+K803zDPWhplryItZs7'
    '0A8Lvdlltz2C5sA92C+UvQXhnr3YNZW9R3jFuZB5xrzMqH+9CCELvZ40mzx91669EmxiPVmb1ryF'
    'S4a9qa6yPKFJ0T3pxp+9/U7BvPnIib2QSp09nm+zvbNqyL3P18U9EL9pvWDBtzzQS0y7NE5UvMU8'
    '1Lq8H3I9rebuOvl4tr2Ke3S9mvAKvaXzPj33EYS9d8a2PSDUe7zwSBu9bOeSPZjEKT322KU9GwxE'
    'PTHnFL34Pme9witHvAE9vD2zPtq8uOOZvX/aIj0hT849wxfLvKqzp70sq5O9ohsLPZdpf7x68lG9'
    '5NeBPA4aIT3T25c9RAOTPegqn7w5VLK9F3+xvcOfyT3M+TA96B4gPceArL0B87Q9KhevPbQGoj3y'
    'y569JtcEvVi1Z73MHMQ98LogvYSgVb3iXW09oCX+vDpZhL3TgQK8erJHO80Wk732n6U9ZJJGverU'
    'JD0E+qg9pek7PVOWgjsN2ps957OFPL1kDb2SQoM9n/FfO4Vq0b3yq1e9KmDEvSJQZT22nE29kcPW'
    'vR+P9rypBHA90I7oPETkEj0YrCc8o5vOPDC9/7xP+kK9l6mqvQ94qb3pRT89XXNbvQrggjwgsBA9'
    'XRAhPAQOeL1fwcY9g5RXvVZ5pj28W9a9/8Nxvc1tGD2zHae9Q4+xPcjSkL0M+MG9sRcQvVqDgry5'
    '7d0941UsvXzoQT3LsYu8jsWvvHf/iD3goZe9CtHWPP0dxj0E1TS8BGuRPbDbtbyK5Jc9HtC5vYVC'
    'Zbw4fGe8Uj7+PAoMED0CuMw9p9FwO/qJoDxi6n890n5nvEWCqz2OqYu9DG+nObS7jjvEnhQ9HNiU'
    'PWO0w7xTooa9NENYPa5pFT0a+zk9G7kkvacKfT1fPsI9f43gO3ntkr3BboI98/WIvfsDsTySmlA8'
    'lK01vaq+Bj3GEKC9w9zQPOf/Or11Yxe8StOcPJM8qD22cU29U9OBPb6zbrwY/0y9TOaWvS1kvb1O'
    '6749ClaPvF71qTu8Sog8cFWUPWijtD2Q/p09aX6PvQcPkLv5dHy9LVY8vVz83DyJi649tUWYvVAA'
    'dz2XKSW9gLqxvG05cz0y31+850wGPHX7lr0/xKs9/b6yvGTCmL0CBhg9Q9A9vVwdWb2kI4M88tSk'
    'PYbagby6wiA9MZK9PUdFjD0fOTW9v9rlu1aX47zfmZS9A9wIO5x8lj00nKw9hDwcPP0PsDzj+wW9'
    'fYaVPZCSXL3WFsS9BP0ZPepeXT1DZ769vE25PfHsUj3iPZG9NmadvLbqrz1l9LG9V6UbvbiBAT05'
    'R7q9xSEzPSExSD13YGM9Oa5svVw8tT3EaV+9NYezPdobxb1Xyby9NQBnPXH+Kzyxg+W7/dRdvUl3'
    'Xb085hK9CADKvRZW4DoBmdC6d3H4uzYI0jxtCdM9/OJAPUHFpbwcV9a8uL9gvZ95aL0MZzq8JOcX'
    'vEBpID167jm9DhquPbMulr1jxIo7JmmhPS1Duz1n8oG916WBPJj1Wzwc/lC9ZivtvOyB1DwLN4C9'
    'GmhlPH5rIj2cjDy90RWZPYArYTsvXiM9j7DQPWMrQrwlgmg9e3asvFQ2fLwdQ9C8QBNkPGijybwa'
    'nqA8YAiuOyKlbD3jJVC8Vk9+O15kEr3OmLa9svsqPRkGfjzbIxO9b0yzvPITDz1dHCi9fiyDPdsv'
    '/zs6ZGc8VhYUPSkQxbziEBQ9H6CHPVM2bjvijrc86FQbusA+lD3H/cy8K8ozvcJTIj28jbm9QPq/'
    'uzzBUb1hrmg9hDfGvR0zUz0sAbY9KK5PPNE4UbxtdMY9b38/vU3bGL2rS4C8i8n1vHUmGD1tZTE8'
    'BEoWva6dl735YyG8K0GSvJ7bMrvppp496d2iPbwVd70FjCq74EkNvWuOzb1buJu9geqxPSzZYj21'
    'X0k9rz1IPW5gm70jSE08q4y4PS4zOL0dm6o9ROu5PMacYD1lpiW9pfYbPez2rr1wrY87e7FmPZWQ'
    'OTzBZrk92rHNPZDwsz2L+Zs9rQ2ZPRDyC7z2NKG9jOs+vSvzzLxK9XC9Uf5DPZVNDj3GdEg8Lo2b'
    'PIHnrD2BBig98OmevQwBkr1c6Pg86kNvPBbAUT0bi8m9jmOgvbwCbj31kIc9QsqhvQqmKT0DBbG9'
    '0fatPR8kQz3ZlLw9gYq+vSvIur09PoU9bH6+PVR7hT2/QGo9RnGivQuqmT0n/8W9GqWEPJpQgL0y'
    'Juq8KcnwPO+0Ir0iMg+9C4jbvXuQ7L0HnPU8IwJ+vTyhs73MTV+92hnKvUqkyr0T2F89yk1NPeXv'
    'ATzFgWO9D0kzPdHQcTycnni9eyb/PKBTwT2Ll8e9Hg7JvUcgIzwVXYs9D1qAvQeOjDymsIA93wvO'
    'PJNahjxTbwY+QzarPcM0kr0x3PK8PRy5PT239j2Sgea9kcCFPbPje70ZsM+9We/BPL+EW720Qes6'
    'qYFkvc8pnr3V3ra8U+O9vEedRz03Azk8ZPSSPbzMVjye7gO9dVl/PY+IAzyGbaw8RJzJvNu8GTym'
    'R7i8lMo8u18kmb1HEii7luqkvRJgr7zpvHE9EKmMvWCXir0K/fu8rqW4vGYWuTz/6GQ8Hf1lPZZZ'
    'jr0PNwK9PsqLPTYItT2EIpq7DO5pvcyexjxxy1A93AaBvTEWWr3bljw9TtxROzjEq71kDWm9X1sY'
    'Pa9wH70GdoE9a/p+PVMOETx0SXW7Js/RvLeuQzueQZI8CyaGvTJyoLx/faS9k+FdPeMugr3/TKw9'
    'R6icvEUXT70e5UA9yIOSPDvDij15QAQ9hr+9vfbHgj1+hD+87z/5vCT9fz2jCR09A7e5vVruQzvR'
    'GlU9mqzIPFKenD0OoU+9ev5evWnqnj1/XmM9mLW+PXz0Yrzpmwi9wP04POWOh71KxqI92qWtPW1s'
    'Hz2IOT68GXA+Pb/zc70oG7s9ssDHvYbzur0kTtS8u1GcPUmLJDzC1xE9CEaPvVxNQT1JtFQ8zYdV'
    'PQvy0T0stz89UCChPB1b97mHyVC93l4IvX0ORz2B0gS9j7yGPU8A/LwGq6a8Z35xOwquq73gA4O7'
    'LcdmvBZ1DLzkLC479g2MPYkUjj0kJYq94QY2Pau8l70sEKW9QQoUPRBTZLxz2TW93jy+vfz2H73R'
    'Glk9mDG4veWAvr33NhM9GXSVvXu6qr3WBRq9sVPJup3dvL3wSGq9Ig7POzGMuL18oRW8l5uAvcbW'
    'jr1cTW89Dd2lvaW8lr3Yw4o9+PWIPAy7tj3wbs68hI5Uvczssbu9uXM8PB6GPQ1/tb1etx68SAsT'
    'PS7+Vb3Ydkq9OhorPZ9uxb0e/yk9876QvYALgz1Aeqe9PjxWPacBpb2485S9fczfO9VGlz0blW+9'
    'fa1KvboBrD0q22W70Q4mPQsZtb38zUa93xk+va+2zj3KKnA8CBqwvTSDiDyGy5O8WJBRvWccV70b'
    'vga9/JiQPYspk706JXK9sfXOvPZ7DDtvvAE9EbvIO5YKkLyflV69mqd8vJwyFb1/oKk9eOK5vYSu'
    'bb1mupW92g9vvPqZoD2XzK09DAO8vR6E/bvOej09miJ+PSc94z1QEZS80LxEvfcHsT24U8S8H8Ec'
    'PY5Yebx4LSY9XLmIPb66lz2lWMC9Cqmgvcj8db1+I4W9B4m3PYIlPLwKHoK9Mb0yvREHIL17Mxo9'
    'rE6WvRZ5pr3k8cC9D1xEPdJ9hz27Zna9TRRbPLVciz39aHM8nXX5vKiDPbyBZ5+8U/xdPWPAjL2h'
    'mX09sb9rPbS+h72aDas94xnMPRKQc71YYQK9eBFIPTnKjL3tyyQ9tXyXPKyIkz3Giqu9fweSvQRx'
    'Kz0mEcM84PShvca6R707VbC9PuapvRE/Q71gHIw9gcRxvMsmXj3Uk6Q9KMVVPbudmb0qElY9JTXP'
    'PMwBhj3Wic29xM6JPZJQmT1JKb69O8+LPNAHrz0gQ429w83xPAT+Sj2vtvg8sK7PPfDVpD2MPI09'
    'Rg+hvK28mj2RWZm9eH1YvWD1lLz29bO8gNzFPQAZPz3CiWo9cnt9PQIAlL1n0ZQ6WPeROwMliDyq'
    'g0Y9oAYWvbwcr7wffag9WU3DvJfFg718A4c9lkSHPdSS67wucJy8vrWQvfwqo71HT289MVdQvLL8'
    'kD2mvK291/z9O6XPbjsNbWA9z13uPREA0z04KRw9n0E9vdxep70/vMS966ZfPH8+rz1ep5S8w2xx'
    'vMwmrr1tZhE9E3RePbCu1r0RtZe994OMPe7EmTy0TJM7P7OTPIkh07s0Y369JUy8vRIKej2uwRi8'
    '3PAGvez5u7wqeDe9wgMMvbwArL320o89uAETPVLLSb1S7AC9aTxtPTZvKL3sThw92YUdPQNQWL3U'
    'Gfk8eCKBtyoykD244SG9I7HXvMxnoT0L8n+9yTkLPE+kzj3twDC9E9S2PRp/gj3sx6s9xKowvN3G'
    'oT2GBRO9Wqn0vPEYIz3gZB48orGQPQrmkT2aznC9QlBhPXmpGD3hO3I9H6VzPZSHBL03OYy9uKKS'
    'PTZQi72V/Vc9kGS4vfVarryMmZI8wf6fPF6/fjxPURw9kfXUu/YJTz3DC7k9Xdl9vcIORzuRrw29'
    '7WHUPVR86z2dQfm7KEkFvb+Jbr1E/3g9svgovF3fwz2V5Is90SeaPRPunj0VTX69n/ntPHjaHr3h'
    'gUw9O6LVu88kLzwkPni9sbKivUHsLz3wNSo8EYoyPGw3mD23lqK9EATTPIc8qL1LMgk8N70qvQ9c'
    'Tj0JKaY9jFKQPbCpmLs5KkI96qPLvXpcbL1Kr5g7lGk9vA73lL30URu9qMlLvWoHxL29LOI7uJaC'
    'PeHCHD1DlmS99yetvVSRuL2KkYS5De1LPSSxE71Bqoa91spPvRK51jzUXLG9jhWuvZ6n9j3LlTa9'
    'FxoGvZoyGb1BG6a9Sa5xvCwI3Dx6JeS8h/h1vNd3qbsvF7a9qVpyPVjImz1I04E99fQRvcEUVT0A'
    '5cm9OwGZvGJO1rwWMLu8BPPjPP4cJb3bhk68Ab8NvZFcs71Fjuy8dP6KvWDhVz0ysm09F0+ZPb2U'
    'AL38Q2A73q94u84Kyz1OwZY9HEMNvdygsb0J0pm9JsJyvaXNnj31cQQ9Jw+sPcGrlj1flM26c1Ff'
    'vb8moj35ZaU91nRMPahwHb1r4MQ9Dxn2PPaDlT2XnWg9bAiAPRhbgb3gnRy9Fu5kPcneP7yzQ5e9'
    'ttmcPXo5HDwysHM9R1aAvUbyMT2i1Du94RdDvYNE2ryPHSq9IXZKPbrrqrxFMZA9bAy1POmOZbzH'
    'k8K9Vyq3vUQxU7xq9+O9LA0dvVC1GjzD2768hsSOvWMnpj3ArE898SGOvP27IL3P0JU9wAa7PdTj'
    'sL24QZy8q2JWvEB3Jb0/SKc9yCeSvEwioj08gzm5IrQDvFYlUr2XygK9b3D7PBeQPD1grNq5UVHF'
    'PXcuhL1rET89nOq5PYhQVLyazMA9FsFAPGEekL0qsEC9XG8GPSU/FL150pY8VvnWPe5/Fj2N+bO7'
    '0ZzFPCoSR72g6qS9rTyTvUcZq73fuK88GYTEPOLZDb0qU669sItWPU6FT70exWq8P6uovTVdZr01'
    'Wxw9oKepvce8Jr0AnZ49g95IvTMhGb1HVw29v1wTPXZ7dr2/I8c7ftYnPQ4Y+jtuH408ln6jvWX1'
    'ED3+sxk9MNdKvL+FqL1/Dy09Wc2LPRZHcbxjnVU9UZMRPW+7EL3syVi8+3HRPTzHkbxdVaG9ri4A'
    'vLRARD0RlBC99KlMvVu1dD2iBTw91zwrvCyIyjsK6gy8JW+MPD/EsDrnOW+9txIIPCMjNjsmzoY7'
    'WlIcvcWCszwExKS9B5HaPO0FSD19EVy8Ld+dvRIOgLuBqAC9m2GsvdDmrL2k7/U9JKF+PSOmjj1f'
    'Sqg9aTU6Pb6etj38rEA9JGP3vBreCr2numo9G5Gxu7J6kr1/P6K9lXwqPWUQoD0Ao4s9KROLPN1r'
    'jD2uYYm93LeLvSGgs72byJk9nKUyutHKnrzEzYs8lBESveT6o71IlA49ZE0ZPJlTj72EEIO9j3Vk'
    'PVVQxT14daE8D2UuvTJiQT1nYvM8gOGdvelAdb2W8rM8BoWOPZaJtLxkNCY9bFt0vXpLxj0x6Lo9'
    'aDiHPP5UPT12sw89RyhovYa5vTxv+mk8wojJvHxPvz3NeeS8oyDBvQAlwbyp4KO9e/yTPXfTr7yo'
    '/Yu9uopvvDQoPr3LaVa9UkIKPfGIm72EcGg9kvRzPdSTajwhczQ9swBpvbRArD25z+A9S8C7PMcc'
    'pj2VwrS9RV5AvDRqHj3qd6+9C5hRPZB+87xz6gy9zn20vd3VLT3bnZa9OAM0vNVjPT0jFZc9QjtK'
    'PKDlIL1xSiy94hU4PZLmCzwcw2g9fwHjPDdEhz1sQpK9C7GjPQKtJb1PrKO8k0g5O3Y3bLxBEvA8'
    'GL69PecHSr1dfy49m2WBvbKBLDwY3Yw9dqN7PV+n8D3NuFo9f1osuwzyOz0ekla9rCq6PdYtKD1/'
    'Hai9eTqGPUTviDpJm3a8jdUEvczK27yFW/Y5Bj1RvXhHyL22DmG9986LvCJtkT1pi4k9xwmIvfx1'
    'VbstGxS9TSlmvVl567wU3iS9r35HPUG/R737UkU7BKF9vZvW6LwIgbg9UjB0vUQWirzd9N49WgUj'
    'PfLyYb3q54C9oZsYvWpeC71x0hg9gC/pvEz8ij0pUk2938DmvJv8CT2prZk9L0yoPRyAjzwJiSq8'
    'JjBPvc4QLz0OhYy9ufq0vJAIo73dGrq9n1QtPdNzfTyPRYY9OlqSPZAqNz3UXa08w468vEcObD3x'
    'CFw9Jei1PHG9UDxI9Tc8HCH7uyUMr73endc8fASOve97OD3HoFa9Y+YYvZDEJb2j1JY9Q3WMvaiD'
    'bj2AVBk9kFKUPYWYBz2lh1g9Pbm8PS7PGzwBKYQ8sgXTvc42T72cny88weCxu+59Yr1y97y91HwX'
    'verjAD3m6MQ9kxaVvfr9cT0fEaG9rOGgvYQMRL2PtpE59aEmPaupd724D8491GnrPXLBcbxLotw9'
    'O58bvajeE70rGMm7lmaNO1HiS70iJZu9+2EGPWn7oTyOdq4934E1vfV8rTxQ1GK9D1A+PT1Y9LqF'
    'oJc9ZO5BPaoimT1FB0o94w4IOuPenj16v3O99nS0PZz8xT3Hgdc9hWV1vZRXar28sXy6Cmq0PFPt'
    '0zxZLcs8wvibPb5167z8Yge9vkb+PNozCz0U3G+9Km2RPIojnr2b8eS8R3kgvWIzvj1weqq9f7iG'
    'vZaNyr2+aMs8pbyevdSPwzwTwZ29v5iGvTUoP70rNXc8MUyQPTQWuj0FKku8oPDLvf8VHb3lEBW9'
    '+dHFvIQGDj2SlUE93XmyPO+trzyOTge9Lw0AvJyehjxH+dC8QIqAvM2ipLvz6bC98MddPX3EVrzG'
    'w6G9F3yZPS1WY71YwfM8kmW5PX85AzwQ1T07J6DQPAJskz1rkBi9k9F9vST0yjrqLU09DX+Zu7mA'
    'gD0bkk28TaoOPGjekrxRhx+7N2e0vZOtpL33QR48Yq9hPeXvjD39h0A7g/8ePRScIj1+2qy95MOv'
    'vV0fmr35C306p2h7vcqb+7wJNRU8k2M/PWa2jTtjqP49Qs/vPQhBS73m1Ss8b25CPTpU1D0MAMm8'
    'o0QuvbILTr2x/Yc9zI37vPxotrxGfm49y7WwPR59ersT+Tq9x3GHPfIQJ72R2Qm9+TFyvVz1cb2v'
    'MY69t64cvVMDwL1DwEy8SlUwPVm4qL1DsRK7KhJivQnApz1nT7w9WfomOoAzZr3jcKO9saJzveBX'
    'JD3NvvE8JMNJPVajXrynI3s9v2u3vfjNAT3FnO89A1LYvTD8cr0KJ5A9/Jr3u0b3Pr1fKI49qlbN'
    'PMg4ibu2NZa8yGaCPRweIj2JNpY8MXqyPIzTHz1jhY09Y3XFvL8oNbzd1Vc9G8kLvf5xqD1kxYG9'
    'F6pSPf+1h72QkHA9uXO1PZ0asjyLPSs8LU5cPQpBnb0Zt3q9kDabva2NhL1sLXQ8AoABvdWkijul'
    'nYU7FrCoPbbsqLyZ1ey8LBCLPRCPobwZjMw9V5jvvJnRWjyq7K28ZOGDPczzYz2bmSm9cR50PfHm'
    'ijxP50286ULTO7bBgT1dccM9L99APQqpoDwkUn49PPQqvbh6Sb1IKcW8tRhzvTUDQD2F7jU9xGJI'
    'vbpI27yTbQy9eajZPSRCnT0Ni7+9pqotPb4svTzaSw49ZBJpvJN55jyBuZc8pHSxPX441727Rxe9'
    'l3+MvcPYbb1mWJo83I9fPdJul70AJGs9z3qcPWd9bb3K0J08zKFUvVsSnjxuTTu9+LcjvXzAY71u'
    'W7W9eZ+kPYdykbx9G4C8aARFvf4sgz1VwHi9FgQ1vYUzrb2y8a29pUTJPYI9pL0pAuC8JhWsvNjb'
    'Lz2nC9E8+2yYvT2OsT0GGc+9FYsYPOBEnbzjXJm8R2ODPU2SoT1/WkQ8fgr3vK2jlj0V53s9weKm'
    'PKXtEb0H4Co8FKepPWmzajsKS7u8jdqUvaR9RL2llZo9082UPAj9V7yHo089L/uDvbbplrz+lN+8'
    'C+9lPWpLpz2Ez2M9FEuhPZAhvb3nj+a8DmxlPL8Rqr33Ois8CbWIvWV9Lz2J/FM9G6DJvQywhD3n'
    'kMY8gnKfvf0OrD1kDbK9d1eVvZx1Pb3tB4U7zRY/u9EgFr1G6569oBKEvMCjbT2ItZQ9wfMqPcRN'
    '3DtC+7k8gpykvYhFAz26I8y9UFJDvZmmCr0o/Ns9VqPiu3TBjb0dEes7Q6XbvV+KOrzWYMA8eY5n'
    'vMTl7LwmWow9coyyveArJ71wN7U9ln2RPECEdbvFKUw92IidPTGDWrzYtgW9AEpqvJ5aCj2iP3e8'
    'chRdPQ4xp72dbF29zAWLPUHNWDxguyo9XG+kPNzByrzziQm9uz22PEfDaLxN8yy9oykhPUTNDj0b'
    'lIC8Dzw/vVXzbLxBsnC91ng/PbXnYL2F0qo9Vj4fPQRhm72YgaE93S7AvdnY2rs1wYC9DXqwvd4W'
    'oTu+K1O91ROMPX0Xlr1Oraw9Gk1xPWVApj25UAm9Qyl2vfpUnb16gps9JQs1vU8/6LsT1Lu7LRXP'
    'PIjlnL3JwaI9CoqSvZ6epD1oSxq9PbUSveqkxbyjPgw9BfJlvDhoyL0sFoU9Fci6PURnIz2I97s9'
    'XVbwPJUFxLu8E629yqW3vQYjc70e+kS8ZLY+PXATPb35vcI7JIFVPJ/Se70GEmg8UfTTPYfNoD38'
    'y5c9Sxtmu7MYlTxczVg9OQCgvG5qwrwv9qo9zorqO8D2gL2/GaY90kWGPL/7VT0ybKq9p56YPOtR'
    'dT2bDgO8XcuOvWPxqj1xRJI9oBmdusgULj0RZpE7dHCXPWnqVrzIV5G9t68hPdIsND2DXvM8aZYL'
    'vMh0MLwH21W7t2W6PBN+kb1YJbi9/bURPTI2Pr2Qzkm8Ck7WPOaadDze9Fg9NEuQPEqljTwJ4nW8'
    'd8TXO1VLXD3FQJg9SOrGvbIkF71Cm7O91DEyvUfSlz1Nxam9ZnuevZgkZD3o20S9ivGLPfjhBT1e'
    'nrY8tS2iPVoboL1cX4w9ba4CPf9dPr2T4a48BQHRvLVaG73nE568MIoPvXCHoD0505I9C8H4u2Bl'
    'ob0PqPW7In5cPGi2fj29FrS6MUQqPWlSlDwNW4a9hJZPvYxMp727bXk731EzPR/GMrz/gcY9kVlC'
    'u6JEib30xWM8wOXJvOBU2ztBcr89U1qvOmtgqz2nQJw7C3WSvK69gz0SVku9e2iSvBIyizwkyHU9'
    'cMXEvcJvWDwT+Y+8tadXPfjlLD2F1z49aYGIPeVnrLw4Zue7IxucPV4Zs71eEnS9o6kSO6bZBb3u'
    'ZYE8/dvDPd8k8blm1pS9oYILvZTryD1glDe8meoCvceXWz14MII6Y7KmvRlOsT1NFpk9YIibPTbd'
    'NT2ic0K9XcpvvW+J/DxZHb68sFp2PPbaxL1ZwU+9gYU5PdY5mbve53e9MGy9vRfFcD0/MJ+8aRTB'
    'vc4Fsz3LRUE9IT5TvVZckL35vLk9nt0QO7SgXr3EC3Q9juCovJi4Obu4SkS9alKtvaoPlL2djKe7'
    'mF+RvUxUqz3hVbI7FRXlvMzWEz2MMZ+9KHAvvXgL+TyieZK9nZEiPRUEqjxziKC9ABPAvZiR2DzP'
    'gT09KIIzvI2Asb15OGe9ZSdfPLZ5zDvptPI8CAyUPGj4p7pS2Ec90mbSPa+xtj33pK89arHnvN4J'
    '4ry1Axy8T8YQPSuXWD1xbhW8uM9GvdDRRb0Wws49+7rLvGx1Wjxyum69oLU1vS+loTzYdi89TXbn'
    'vOZS/bxFvp09YBktvbP3gb2ijom8SjHMPVgZIT112IA5fjnuOHclkT1kewK982SSPfP1ND0UsE08'
    'LtLcPGz8ib093589eVtjPWVhk73JUre9edbpvGRwUD138zy9ULu2PRZEQL0zbK29+G8xPVJJZL11'
    '70y8oAvqvEHfnLsK0sC9+Yf1vT+Xiz2hGAa9U0CovVwhqTuN1d+9h819PQSKgD19dt897EGmvZzR'
    '5LoRBWE9+4K3PWwIcr1VVic9NEtKPaWHiT10mKu9aQiQPVXu3rzx02O9Bm+RPAZupD0b4Hi9TxZZ'
    'PWHQMbyQ+jY94lJ+PQw0ajz99o69vV7ivL1TZjspYq29/bW4PEoyZ72jeOS79EIxPby/eL3nH1S9'
    'xCkyPW14n735vYw92tCBvTIH7zs3Syg8NzomPfHQO71Iw5Y8tZsjPZgApz3ADYy9XSU5vafB8DwB'
    'l0+8rUR5vRB0gr2Z15K9gcR3vbz3M700f2W9f6KivTvPRzznujg9h74pvYFXKTx8q1I7b/JQPTDx'
    'gL0yS6488UXJPQDXpj0KQd+9B7MHvF29Rzw26OS8b+aGvcRehL0GGok98HTbvDKUTrtXdg89ydqB'
    'vU+KLDzY+Ic9cSBSvWJlBD0N/k07H1MdPfgbaj3Ax5g8W4BEvP0BsL2dGo48EuIePa8CUr27sdy9'
    'u+l5PfQvHL1l84g8iKMJvYHxZz1jPrU94NtQPUHyHL3T5ns9XBIYPcKfkDpeRGO9TIwKvRgPvb0X'
    'INI9FbPlPSZ8Lz3aNIY9qotNPY1B4j1H+Gy9lTkUvRu+xr1mZo68TS8ivBXcjzzAGKa9riG0vNzc'
    'rb1Imb69dRDNO9qNt72cmrU8RZmvvffGfr3A+sm8OK3avdUjkz2HEL49vf9OvfU+Ij1Tc1M91VD6'
    'PLl3pryab9e9K7mgPaSlvr16gW+8d36tPEQMibw1noq8heWIve+soDyFmjY7SFPWPVJD3LxyBYo9'
    '+RlnOgZLCr1ZgNS8IY/6uzw4vjvtvqa8KBK6PDANlz0Ozds7hBj+PNdFgD1jY5Y8f2owPez/eT2h'
    'R6Q9DjO/vQJx+zxSvgO85drluxYX5Lzqg4M9r36yPb7nfL1Crnk9iiODPd/uCD27R2S8QMeYPakJ'
    'ODzGgtO8D+7PugBGEj2fWrA8a0uhvMZc1zwZDmY8pcSVvSedFj3L8xA9rSWqO0VZ1jsBbz49B32o'
    'vWKRDr1uCVy9RFKQPcISzL2fN2i7vRDXvUV7HLwLJsQ8twIEvUJOkD2HXEg8VeA4vXAeGT0bC2Y9'
    '94Qku2ymtT2ASU293iQyPc/Bhj0+PXs94MQ8Ozhs9zxh5M+9c+ZXvd4qGL0VxL27m33JPcvbpj1H'
    'O3c9gm04vRatiL3LbsG98PUZPbbOtr3MytG9KrpFvZBrsb1vqQe9iJSNveb1Qr25AbU92aGQvENX'
    'nD0As9668kJCvGOA8bwCLZG9x6aevSwIqz2QtGa9YtTCveY+yT0Qn445L5SavTBToL18VqS9R4eu'
    'vfkHwz0ZvaI9ExCePHR+r71IBx69vXe5vTvV9zynabK8M132vGVBgLziizo9MzlfPX4imzzifnS8'
    'w8UmvALTM7uSKnG9WXugPaxL1LxuSGo9UBVRvaXV3Tuq9RA9nDSvvTu4UT01PFy9jAonO3Zkrrw8'
    'C5A9w4HZPYTmDT0aJog7TIZOvd1ZJj13KC+92pD3PKkAAb35EpY9FKGdPV8FZz13bE47G7AtPTuz'
    'sb3i9pk96LgWvUDs6DwsFau8YzAPvTKKYL3llV49L4aCvTjw3bzkqzE9NoGFPN7klb3FrsW90cY/'
    'vTEprz2/kpm9Z5NHPbjiVj0NWwG98M0OvcRkfr2n7rc8Jbrbu+y5MTwRoEs9qVFDPfc4wbzYETU7'
    'T+7mPG+aXLuhMDc98/EtvWkpbz15ikA8LZDWvHhI1b3lR5q9a9imu741gLwynEK9uTRnPUbwBLww'
    'g8Y8SguPPVqYL71lUku9W4qOvCHEb7t0U0O9fQmivEtXFbx6aio9WLuuPWqQjz3EqQI8D+rUuxxw'
    '8TzsFQW7bbGMvO5+Bj32zLS95ITDvSjexbtPLw29xNkyvetVtr3inp49AVgjvC4qRL16wDK9WowF'
    'PZcomr2vrcI76/wUOuomEz0r2SQ96M9tvX+XDj2pmZI9ZqCGO5t5Rj0m/Si9tAisPGWO7jzPJza9'
    'uQaPPfBo170TKCm9X5RWvWgnX73gFHq9uNShvZ1OJTzD5Ba9+ACPvb4yAbwCVYS8nPEmPTn+j73+'
    'YrW8lnCYPCSoJj1nsBE8NbQNPbIxuj3hhwS9Oj+FPcA8rD3cWJa9K276PIm9TD3koOC6bERGPWxQ'
    'ITt0mKM9DfHOO64aZL1c+Jo9hRMSPS0Jnz2jrRM9fi+NPbj0lT05bEA8jDfSvIRrNDskFYS9NvSU'
    'O/Pe7byZLQg81IOavdV0dj0zfLG9PlgrPW1MhD093uy9LpqvPUp2rb0DzxK9kISzPZYn/rytqLU9'
    'huY3vQS0B72rg4+9OzrZvFK2oT1au3M9Ab/MPBH2ITtYc6m9ptScPXjYMLyY48m8hSqxvXUalz07'
    'AC09/YdgPTg3QjwvtIi9mDvyPKgGQjw7ZiA9/Ij8uybpR73m84o9grBqPTHH8LzI6uQ8Q+Hvu0ul'
    'wjySSHo9enFdvVqruz2y8ZQ9wR0UPE3RPb2E+AA9e7WKvRiwdT1RrxO9ZZxVPJmXyb1HajU9ui2A'
    'PcbMmz3Y+VY8dF5BPXubFz3Fm5g9ikg0Pc2cob14T248DxsdvR2EDb3UTZM9/VOvvDB2ujzkkYo9'
    '6T10PSw4kj2XUpC6Ca64vYncn70nagw9U+XGuzULM71FZ2s93hqlPemhWz1jqk28QskkPJPkY728'
    'HlU9Y+6svSIXmj0d0H29DdQPvfl4ob0kWXe9z2iDvA2xbb2b4Co90uFoPcZ20L1jT5q8UwCoPJRl'
    'tz0DRp28joRVPS9DKzvtRyC91ytaPZHTyr0fZJk9cHW7PdyU87zP8Ac9Owa9PR/a2Dxc+w09QE0J'
    'PbxZJL26gRg81ryXvbVe9j3imk69ivvbPJ/Jq7sUqYQ9tdKJvITItj3GRr28cYNVvQ2NQj0c+GK9'
    'QnyvvSUSID0fzk69XhcnvdDbxLyvJAs7NcsQvX1NAb0v0h29nGdkPYElmTzymju9RUigvd/unDnK'
    'wXA9fuzoOv+poj0GQ7A98JQFPRocSD0LJbQ9mjY2u14NSD0cPc49UGwAvZkU5zwq9Ju88J5ZvXC5'
    'kD1wEsM9vGMZvbBjfz2YCfw8ooIPPRNAaz3Wgpg9HyF8Pf49t73SUAa90ts1PQwWZL3ytbA8rSAw'
    'vZ2Rhjy6XRg93/inPDoAhj1bF1y8bzkbPXnJ7zzqd0m8fHBXvTGcxz08jYm8zNulPUHeoj3gT4o7'
    'r2mevbrRhTzbKDK9I87EPFV1CL3RQWc9yLTJvfJw1LxDRmk8jiuzOZ3ynT0f/fm8mui1vYjFhz2i'
    '0GU8sBB/PY3VKr1lax29pr7wvGe2Dj2NYOM8Ew92vfTHHLxmYjK8+dCoPW+X8Ly+CaM9QgM2vbe6'
    'Gb135Iy9JoSRPUl1fL3cMyc92sgavahiyD1mwag8fEKjvRvkyjx4G4A7pfLTvRx7ATzNQZg9D6fO'
    'POeG7DskbDq97MR5PH1H37wRo7U9uWSvPQ38OD0VVBy9LIm5vVDKgL1m3Mo9B+qpvd8/tb04+Ci8'
    'bOTsPGBewr1u5ca9Gw0eOeGbHr3ZHIE9SS/bPDY9sL07hJ+9TBOFPRS7YbxE8CW9fETWPHH3A73j'
    'BX09cJW1PPZeCr3cQpS9EutVPZpxdL22zIe8pb+EPA5lTL2ramY96ymUvUIrAz08ubA7qTYJPZgd'
    'vb1lRJu7SphEPd7dlb1DhLc8UhMTO3RzpT0IGbu8+ewIuRXtKb1mkgq9THgKPf5G1L0g5ZQ9vvOX'
    'PVTmGT01eG09dyLsvDuzdzzdgBq8ESVcvSk/XL1bDe68wwp0u6xKp705xJq9k/Q4vPnvIbwdp8+9'
    'otjBPI7JGz1PtqI91J22PRDBiL32Ka28LdoSvVoEMT2F1j89BnwgPVeQhT0CD3c9LUoyvbRCqj3I'
    'qKo95hfXPMpfS72d4wW9wYUjPQiARD2zm+s9siFnPdsmLz2tUWa77r2GPZXyAr1AICY9zI00vRiz'
    'cTsdK3M9uLBAPWFegrwJItq7de16vdS8oT07dHE9uKwWPRvPyL1niJK9UTQdPZ7MGLxhEYo9YhmU'
    'Pb3HEL0TO3e9ZvSovcfILD1hg149La+8vBei0r2wl3q9zFAPvNCFQj1g6/W8LPpzvU0ll71oKqM9'
    'fm7rPBMAlr025229dD21vZK6rTsG8iW9IaUEu2mbAryTo6i9fd6rvVAIlb13sQi9SdU8PGRyIL0e'
    'L2u8kvo1vKG5xT1Zad+81q3KPdjWi730yJ69nprePHg2Ur22alA9cu9JvUjVmT09+bU9pLgfOyGN'
    '1TwOSbw9WydRPbISk72DN8C7+UGwvRTEjD0Vg4W8JXqau3y7Wb35L0y9bCULPK2kqD3jQ169CXfN'
    'O6ZyqT2Gzc494xeXvMhjwD2XzSM9ES2tPW75/zt1KaG9cnWpvAcR9jwhlk09H/zCvc3Koj3lOsS9'
    'WnNKPSn/sj1vEsg7yTruvB4CRD0O7ya9rk+dvQJLi72I4DQ9T9CdPUSfGzxSQLe96/vDvS/NbLyY'
    'Ilo9L9aAveSx5zzr3V29ZYWbPXU44bx13JW8jOOqvd5Mrr1lyQE8JX2EPK/eVz39Qru9iFsiPfhk'
    'RLo5ubY95xqQvRz7Ez3iO4U99+FiPbuG77zHCb88ys0WvV/6Pzxo6mW9QSSyvYRhrr33iEe9nNhe'
    'vdtdTTy1/ce8VHZcvQJPeTyUQ+W8fYRrPTa/tb15Wwe9IAZjvXbwTTyptDs6WSoJvLIgqT0Dbhi9'
    'h/uYvEUDkjwUYMk9jW6CPYttETtR/ei87/oaPTX0ej3Lcho86YiAPYjKsj0LkEy9KizMvNQhmr1T'
    '8508QsCOPSqvWLyfeZk88+3sPIY5TLwqnak9TlqNPPdevr2vI6G9dIrUvWNcEjySq1o92sQ0vD+B'
    'mj0JMaW8V7ymPCLutj2nJjo9fDE6vMbb0j0iFVa9bjgGPTJjTz2Fv1K9U/OfvTAGDT08OCS9F3w3'
    'vUvb3TwiOoe9A6qxPSdqnbyGU7a8uuWrPSEY8jxFzlo8fGJ5vap7nT2biEy9wY6XPVlairxo5jY9'
    'EjmDPUTBpb1Lll69VoXRvTn/mz3/LKo9iKmzvNurir1QKIw8v6guPadr6TyDamS8EjNkvTEptD2x'
    'aWm9+cG0vYk9Cb2Tnsk82RMmPUavLD1QSwcIn7MKbAA2AAAANgAAUEsDBAAACAgAAAAAAAAAAAAA'
    'AAAAAAAAAAAdADUAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzFGQjEAWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWiXbaL2CCA69ky4rvdIJRr3O16e9'
    'AVrvPMbEBL2rxMy9blxbva7Qb70LLDW9rzV0vb69drxAu+08Q3idva/JErxIYau8qb9uvQk/Z7wf'
    'Jzu9MzLtO4/Xi73eNjY83m+zPVx9sD0kZy69DZ7kPNDHwL2KnaA9rZK2vT8Qrr03cbS9UEsHCJoW'
    'Q/eAAAAAgAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHQA1AGJjX3dhcm1zdGFydF9zbWFs'
    'bF9jcHUvZGF0YS8yRkIxAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlqdm38/PvqAPz0ggD9l2X8/HOV/P94ugD/Wjn4/W4d8P6fYfz/+3oA/+1J/P1Egfj90'
    'BoA/PmOAP6SvgD9iG38/h1yAP2bTfz8c+n8/skt/P2MigD/je4A/e5l+P8lMgD8s9YA/Fc5/P3wC'
    'fz/NbH8/04uAP8AHgD94AX8/jF6AP1BLBwgqlx0GgAAAAIAAAABQSwMEAAAICAAAAAAAAAAAAAAA'
    'AAAAAAAAAB0ANQBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvM0ZCMQBaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa+W8ju/PJoDvEqbS5aCRTO0XjQTuA'
    'z9S6y2HZu44NGbzhD8k7zFKMObI7u7odero7KhdRO9T8t7krcYU7yWuFOrq7Gjz3Z4W7dAw5O3Tx'
    'prvcPVQ7zimSuZhA07qkECu6cqttO0ICDTziSnM6tVZOOpgIOzrUZu87SAIYOqJdZTtQSwcI1e3H'
    '2IAAAACAAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAdADUAYmNfd2FybXN0YXJ0X3NtYWxs'
    'X2NwdS9kYXRhLzRGQjEAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWsl6b73SyWA7C+CQvEA4tDv9M/G7r6Opuxot0ry7xyu9zgk+vapxAT1HeOW8m5kfvLfU'
    'AT387JW8FF9evIMbJ710hkm97j33PC3sKrxfAhQ9jey1O6O7DLqcbW29TufdulK5j7zDyyy9s7ac'
    'vJWElzu6Nzu9JvMGPI/7Kj0dM1k8VrtoPZLa7jzzyD69TjiqvAPJpLz7zXA8hNw8u0EvKT1snoU7'
    'ejgWvU1S6ryt41a9uGaQvMbz7zzRx9g6WIxpvQVqT705Tbg8aXNTvR7hR71sfg6965NnPYRNa7wC'
    'Fti857MlvQ64CD2/oFO8SaBXPN4yYDxVFko84qAUPZhu9jrqFvw7BBCzuzV1Pr0kGmW82VxZPU55'
    'Nr0ePBu9o299vfwRxzrGTGm9F3bgvL2cVj3iK6o8bfkDPXQgCTwvmFe9Avv6vNuuNr3YWBq9RwLA'
    'PL0rRbz3CIA9foAnvXSsSL1rfXg9IrQ9vQytSD3mfo08zzJKvfEMRz3erZq8O70EvUbTW73rwm29'
    'G+dPPFUWUr3P5n48QZBnvDTupDzSc428/Pi1PJmK77tomCQ8FLbFvLJmJ73SiRU8dEgUvREyGL0X'
    'zmW9ix69vL+/A73WjUc96+wlPdKWfTy2czw9i5zpPHj2c72VaQm8b0ZRvJrR+ry1Y6I8RsmKvJQV'
    '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'
    'hYhQvOepQb3mC1+9krc2PUWZNr0Udl49fwdOvXyc0Lwivm27wxJXPUhKoDt3/TC9UiMkPf5RKTwi'
    'cCS9bxdnPVxMOT0w6N67iMpZvOAE+jxZyIE3BhdhvHUcnjw4cnM9KmBSvMM8GD2fAeK8BVZbvT3o'
    'rDvdD+K6xkmEPCgffb2JGE+9qmBOu6/8PjxqOni9gJVPvdTjKL06tAW9zXMtPSln+DwXWEg9A7E+'
    'Pa1WzDzfESm9HwtuvUO0Mr0tGmS9d2A1PQO/3rzFSBQ9A7CovGmZFL3E5iA8fc+fvPK2pTvQW1U8'
    'ev5yvUNgaL3yvV49ZyCIPNS2KLxeF+w8bc3kuyY0VzyglDC9SVUUvSqttjyMiwE7MdIxvElHXrxl'
    'uMU8lSw7PW6JHD0ym+S8ddTcvBMfhbzg4+u8NL+PvGM5DbySYhc9dF8yveIJUj2AsiE9vMo+PTEx'
    'Hj2uQ8o869s/vTUcFT1XgYi9T1FUPLc1gj0JdOA6mjiwO273K71oZ0S9CCc0O3bm3TzlbSW9JMiu'
    'uiWKQjp+myK9uXOvOx91Tb2Y/KA8t08qvT1RNz0YalE99fEJPd5BF73ZoBA9gAK2u1BLBwhSdJDo'
    'AJAAAACQAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB0ANQBiY193YXJtc3RhcnRfc21hbGxf'
    'Y3B1L2RhdGEvNUZCMQBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpabJpivdQKSz1Eu1Y9/qnIPKEtQ71hp4y8djIru5IRXD2ugUq9TikxvaP++Txotao8Nm2W'
    'vBOsQL3Usi28lucuPS31Vj28cAG7thALvc4NiTzwtSI9q54WvaWMcjx8rRA96CD2u+RhQb06owS9'
    '/qRovRBSyDq3iB09/O9XuwKpJT1QSwcIXkKYBoAAAACAAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAA'
    'AAAAAAAdADUAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzZGQjEAWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWtoXgD8IQn8/UKx+PySGfj9Jd38/EUeA'
    'P9vifz/ixn8/GXl+P2lkgD85hH8/90eAPx30fz+oLH8/5+J+P4gTfz9ZMIA/+eV/P71Cfz9xOoA/'
    'LIZ+P36DgT/+sYA/WHqAP1BDgD9lrH4/wed/P2k0gD/JzX4/7eh7P2rufj+VD4A/UEsHCJkR1lKA'
    'AAAAgAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHQA1AGJjX3dhcm1zdGFydF9zbWFsbF9j'
    'cHUvZGF0YS83RkIxAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlqCZgQ7Lli+OSfMmrvBnhK87VODOqxKYLtUBLA60YyQOu//47o+3g26EZY3u5wUX7k6DaK7'
    'Ed4vuzZSursvn2g7xfK+uU/Cx7p2dT87PHe2O4/JL7tRoyS7HRRRO3GxlDva0yS6DXFMu7IQI7vS'
    'bo67NwkTu+lj6LvsyPa7qEgJO1BLBwip9u0ugAAAAIAAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAA'
    'AAAAAB0ANQBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvOEZCMQBaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaOa9eveMaZj3R/ZW8gqglvYLSGD0n6n68'
    'd6kavBv82jy6aau7NTUovKxlxTyO9D49X5rVvOgUXjkaetm8QDdPOssH2jxJAyE9XAKyu10fEb1G'
    'kdI8AkUwvW1ic72ozOa8pwEgva+6P71WV6g83kF0PBTZ5zzlrIg8s748PMhEOD2LIhE9GqAmvYTK'
    'cT03+bI8GuVTPZ6YLj2/o6q8qZW4uyEvmjyf0m29SxXEvOtKW70YRVE87KNgO9bWLj1dEVW9YNck'
    'PSu6BT1rmhK6IMRovSJfhDtIQF28piIRvQ332LzyhYW8EnqDvBA7Tbk8ikC98SGePN7a8ryBR0G9'
    'dx6fO6ffF7vaQFC9w9OGPcMDQr0DLzc9lc2yO23yKr36kEK9RS3rvDWAWT3bUkE9S/LrPE5bbD3G'
    'afA8eH6qPJFfRz3Suuw8iK9+O3yPzTzfVpw7SyqtvPrp2zylaBQ9SO0SPWk2OD1LYX091XG2PGIY'
    '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'
    'AIAAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB4ANABiY193YXJtc3RhcnRfc21hbGxfY3B1'
    'L2RhdGEvMTBGQjAAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'Wlpa8/Z/P1CHfz+KtH8/iSCAP8O9fT/zKIA/+cJ/P12ifz9ZgH8/NbWAPzMagT/oJIA/NpZ+P8IH'
    'fz+oHoA/nE6APwYhgD/EEYA/C2WAP+wSfz87C4A/3FmAP2GRfz/Lvn0/UWJ/P5Mzfz9j034/CD+A'
    'PyxGgD9umH8/22GAP8U3gD9QSwcIn8eWmYAAAACAAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAA'
    'AAAeADQAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzExRkIwAFpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWjR/WztQkls7iTrhOZPTvzssH927Zf73unSR'
    '4rpSrCG7aqjWuoDnrDtxhj67/WUoOgv30buWHLW7x6LeO7q0qrpXqKy7ot5Zu1jtXzhwqrQ7jtVv'
    'uoznELkWZNC6cLnOOFjTgTsp/MG7GlV1Opt9EjmfLFM7rXUZO9g1qzuaNem6UEsHCNtkHq2AAAAA'
    'gAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHgA0AGJjX3dhcm1zdGFydF9zbWFsbF9jcHUv'
    'ZGF0YS8xMkZCMABaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlqUVRc95ZG3OyD2bb1PdB+8IaDoOx08SjxIz8q8DLFpPLX1ZTx0mvQ6bj0mvcRYPr3XVFG9SXcE'
    'vR8bIj1IxzI95tojvBEfLL01SWs8ldZpvb1+Nzxr+R49pf36vE91QL3Udiu99uvLvBn/CT14IlC9'
    'k1oWPBlTOj3iqUe9faQJvbFOSz3pu+y8KF6kvDSWOz0GWoq8b5UlPEr8dbzabkM99HAFPR8JPTzx'
    'xz09UAHUugxTO72MOem8pl1JPA4ZDTxErSQ9UAGxPLCfuDyPXNI8j1UevWzjDb2JqGW7Pxo+vctk'
    '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'
    'WsibZj3JB1+9Qko6vCqFF73hXYW78xQlPa29GL3E00o9AIcOvahFbj1lIjw9/vyMvDqmGbzo9m+8'
    '70YLvUnsrrtpaSu9LWcXvf7UdryjVQY9AEFpvc2YKj0N/8i83l/Yu1gGED14uyW9jg1TvWaOFTv5'
    'PBo9nrN6POjdEb1T4FE9UEsHCEyTLg2AAAAAgAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAA'
    'HgA0AGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8xNEZCMABaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWloff34/lu2BP1DMfz9NKoE/UZ6AP/tvgD/ZS34/'
    'XEt+Pxz1fT9TiYA/QbiAP4fJfj+E8oA/RnJ+P4RvgD8RH4A/bDx/P6c+gT8aJoE/OCuAP2Z7fz9R'
    'dIA/KseAPyl1gT8iW4E/sSiCPwo8gT9nsIA/PxyBPwUGgD8DPoE/nviBP1BLBwht3Ub0gAAAAIAA'
    'AABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB4ANABiY193YXJtc3RhcnRfc21hbGxfY3B1L2Rh'
    'dGEvMTVGQjAAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'fxsUvHoDPjv5iCw7TDBpOyFHVzqA8a877VvZu7Lti7sObKy6Un5dunTItrucaRi8/HiMOqwEd7uG'
    'jje7qjNRuvJOT7pGQAO6GtnzOxyph7tmEzQ73MbqO4Uesjt8MtE7PEXRO9/WtTvbiwc8SRxQu5D9'
    'zjsZciE7DlsKPHKjLjxQSwcI0RDjZYAAAACAAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAe'
    'ADQAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzE2RkIwAFpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWvJwRT3Bm0O7hE0tvWkvcD2GIlE9LXMDPb2yRj2t'
    'vSK9JuE2vWtFI71v8RW9iqr+PNsLYr1lmQM9PcMvvVlOJb2HGDW8oJ0svclQjD0F2B+9Us6BvCAb'
    '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'
    'Pc/lVD1nKLC6ES8BPVBLBwgB30r9AJAAAACQAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB4A'
    'NABiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMTdGQjAAWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaLJaRuzdKBD0Fnpw8ylMdvf83NjvkSAE9zasEPANK'
    'eL3CaoI9uuiFOV+Ocb3ZDhw9gB2Ku81XMD1I/Rk9EhWCvOsVaDwP1b68T0o9PbqRUj2esV09zjiV'
    'PMpIGz06QtM8FSCuvF4C0TwHvZY81U1MPRMf4ryxmwO9ew4XvT75N7xQSwcIJjDYk4AAAACAAAAA'
    'UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAeADQAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRh'
    'LzE4RkIwAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWksn'
    'gD+bsn8/OaOAP0KTgD+Wq38/N1WAPyg6gD+CSoA/vIaBP0bQez8YzIA/RKeBP9RjgT/GGn4/l8p/'
    'P5ECgD9xt4E/zY6AP1AQgD/8h38/xdB/P91ugT/SGIA/wR99PynLfz82434/YQl/P+hQfz88MIA/'
    '7AR+P2BsgD88R4A/UEsHCOX/0EiAAAAAgAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHgA0'
    'AGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8xOUZCMABaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlqWHV87luP5uibd4DtpxBc8TPgdOx4ysLoQR+E6lmq4'
    'O0QOCTw83RG8K3qUO+gvfzvgWbs7KVaDuwAAhTobcQe7R1suPNdttTsK+Lk5u4OXO3rGfTtul1w6'
    'l8aeu2y/Cbz1GAk7EnrEOysYKrnJWs67jcvmO1aCBbx3ZU26NSBGu1BLBwidP38vgAAAAIAAAABQ'
    'SwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB4ANABiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEv'
    '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'
    'PbP7DDxxg309oTlvu7DuGT0FizI9Os6LvH0/3bwodGY9YvcBPRsqNzwMoCC5ktQUPaRv8ry20ts5'
    'xlDLPKx+eLwmsFq8Kmc8vQ/a6Tw/wkI9LXVnva5VIj11nIE9ykY+Pfp6Az0H54c8Uh6HvF3LAr0T'
    'g++74ocpPast9zyb59i8sJDMvGuwsrxKOYm7p+fUvGojkTvluCS9UEsHCKFjLYoAkAAAAJAAAFBL'
    'AwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHgA0AGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8y'
    'MUZCMABaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpanUM9'
    'ZuYzvatFMj3P0/W7+98CvXyDibyWfem8waISOxU1LT2MI0q9VEE8vJihVr2TL1u9v150Pbkmnbur'
    'rec6Ovm/vMApQr1clCo91Ua6u2Ybkbk8MSi9RCbnvCWCP70/ZCK8dqKFPbhaQbwMwpu8aw5pPSEf'
    'xDyamgO96ef3PFBLBwiFD1uCgAAAAIAAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB4ANABi'
    'Y193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMjJGQjAAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpanS6CP77qgj/URYE/joeBPxuegD9yaIM/uD6CP8HPgD+E'
    '64E/vZV/P+3VgD8RTYI/pHuBP5Jugz/c24A/Df17P7ZJgD/jaYE/kHmBP965gT+KvYE/vVqAP6yz'
    'gT+AG4M/4OSCPw+rgD9OWoE/2omAP/IXgj/zsoE/cd+AP065gT9QSwcIwuXe3YAAAACAAAAAUEsD'
    'BAAACAgAAAAAAAAAAAAAAAAAAAAAAAAeADQAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzIz'
    'RkIwAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWhsUhDya'
    '0hs8M2tyOacM4jtWobY7BLKnPNXKdzyMOim6ZnfMO7FqIzpcgyQ8PhAfPOgYNzwqHaE8v8L1Ozdz'
    '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'
    'WD3nHOi8C0IfPc4cKb3zoLg8UswYPfLlKT1aqUG9KADWvPt/Ab3E0PM8uQd0PJ5BVTpYsMa8u6kr'
    'PDO0XDzTFgs9j6cWvQD/BT1iGF+9L2olPbl1BT32VgO9vu65vLqGFL0rXUG8w12ePPyRWT3yAe48'
    '7en1PCGqBrzV4FK9sV/YPBSF1zwPgE69lSaJPBpQQb0FDCu8cXIivQe9GDwOl8s810c4uyHx8Dxx'
    '5vm8mJAwPRbJfDxaMUW8NRiMPAdoMD0Q7aQ8ocJYPbcqjrwqdao8RB+EPBs0F72zGIO9UJNHvZe4'
    'CL3a/jO92PsaPcZDZTxn8Dk9raIqvDF42bxZ+uG8ytXTPFyRIT1Itxu9FXwlPcvJSj0Mwy08n2as'
    'vDY4mTw2UTS952EgPei/jzwroAo9SyIiPNitW7177Bw9cRw4vQsvpzgwOz49NtMtvXBeqrtZXFY9'
    'hkPJvENLnrwNhAm8dHeKvLb66jxfdNa8rShJvV5xLT2/Yhu8GnBbvbq3Mr0iV0O97pJqvO7NCT0r'
    'smU9ylsRvQliO73T4fe8OtF4vKOF1jzSrbI8GwBFPa/m9Dx6MVo9vyYuPaP3mrvO0FA9GKFTvAgg'
    'Ojcwz1I93o0avdH7k7zM+La86LHTPJqjmrxnmB89Puk9Pe+tYr259N688ZQ+vasXNb0ZMaw8nvxV'
    'PfAvxLpAQ8q8v0SGPfoW1bwP+1E9JM/IvCUrGz1dTfA8Ah++vNq0VT3VQi896gr7vC3BXz0ewVk9'
    'XlGBO5PczrpGbG89LN6oO7Zr77zEb8q87EcuvTU49rxRL0q9iGQgvSC4Nb0tybs815H8O2CUgb1L'
    'hly9cxSSvDZ7fzwmMCa6JONru1c+erzgfWC8j5r5O1YUeLwQAAY9elPJvGrWKj2Xs6Y8hEbNPDzJ'
    'QL0dris9CWwqPfVVCL3qdYk8SvkLPYQ2uTytkVw82y8FvdENx7w9vkA94jP9PO4HObz/a5K9NyaZ'
    'PJnqZT19Zgs9/0MmvYpevzyv9cU8EO9Cvaek07zWMXc8ZOI+va+x57wynCI9mtXovIg3wLwjIAI9'
    'uNoXPV7JVD06/ie9Lm1vvSuEYjwSGwY9DFjoPDG1b70bDkA9011kvLk4Zbpi6kO9Bs2APDiCAD1M'
    '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'
    'PAkOvjwFsrY8gWukPONDEbx3pkY9bp+dPHJorzw7Y8i6OfDcPCW0Rj39bMM8JxNXPRGfqDxm+Mm7'
    'fBEwPRLAjrsrmbO6r7UzPBzb1DwlNym9dojsPH+1hjzRNWa9lHb6POtKAr1FS6U8WthIPH8ZFD1v'
    'KJg8gsiMvCRtkrxV8D292STjPGhqVT3EWNQ72ZKEOq8+G7sx3Z28/IDOvEfRqLwdjaW8DmJ9PSHE'
    'ML1Ryiy81sKXvGGZh7p/lpG5DG8cvSUPLL22JEe9tJOquT0yDD3c9EQ9T4GbvObcJzzfzso8rxPB'
    'PHN1krt1dTW8ILETvdISN73zdgI9Mzw6OgpLM70fyzO9eHQ6veE1db2WR/k83PlYvXRtFr1z/Pw8'
    '0KAavczrOb1QSwcIrinAwgCQAAAAkAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAeADQAYmNf'
    'd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzI1RkIwAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWt3Xk7yuuYs8b6IDvYEs6jw5zpc8wby6PNSocLsF1iS9xZTq'
    'vN531Lw5soG9Q+/1vP7f9rusbKg8j6QuvctixTtT8dK87LLsvGgIvbzvzIg8CN8SvWUnHL2Ebbo7'
    'AxCxvAj+Pbu75Te9GGhKPOEgwzzFCOw8L2L7vCKg8bzJPN68UEsHCEpKE92AAAAAgAAAAFBLAwQA'
    'AAgIAAAAAAAAAAAAAAAAAAAAAAAAHgA0AGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8yNkZC'
    'MABaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWloCECc+7ZVo'
    'vVYElz1W8wK9MaLKvVqyij25wR++h9wfvfk+F773yBQ+zCjVPaKlCT5q3hg92IkavViNQrxQYRq+'
    'PCaPvfmzFL78hsw9vqcgvm6hJj6qko69f9trPaPqIT4ZBBa+IVoKO3rIHr40WwI8GNdoPS+EA77V'
    'UW88Ar94vXsywrykFfK9XGyLvOAMbb1EjwU+PF33vdeaFL4zkyw+Sgy7PUeFJj6SChQ+xAqvPJhN'
    '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'
    'NABiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMzFGQjAAWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaslhBPVBLBwhQFJpFBAAAAAQAAABQSwMEAAAICAAA'
    'AAAAAAAAAAAAAAAAAAAAAB4AMABiY193YXJtc3RhcnRfc21hbGxfY3B1L3ZlcnNpb25GQiwAWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlozClBLBwjRnmdVAgAAAAIA'
    'AABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAC0AIwBiY193YXJtc3RhcnRfc21hbGxfY3B1Ly5k'
    'YXRhL3NlcmlhbGl6YXRpb25faWRGQh8AWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWjEw'
    'MTk4NTcwNzU4NzA4MDQ0NzUxMDI0MzkwMDQ2NjQxNjU1NDYwMDlQSwcIZE9R0CgAAAAoAAAAUEsB'
    'AgAAAAAICAAAAAAAAOog9POQCwAAkAsAAB8AAAAAAAAAAAAAAAAAAAAAAGJjX3dhcm1zdGFydF9z'
    'bWFsbF9jcHUvZGF0YS5wa2xQSwECAAAAAAgIAAAAAAAAt+/cgwEAAAABAAAAJgAAAAAAAAAAAAAA'
    'AAAgDAAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS8uZm9ybWF0X3ZlcnNpb25QSwECAAAAAAgIAAAA'
    'AAAAP3dx6QIAAAACAAAAKQAAAAAAAAAAAAAAAACRDAAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS8u'
    'c3RvcmFnZV9hbGlnbm1lbnRQSwECAAAAAAgIAAAAAAAAhT3jGQYAAAAGAAAAIAAAAAAAAAAAAAAA'
    'AAASDQAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9ieXRlb3JkZXJQSwECAAAAAAgIAAAAAAAAn7MK'
    'bAA2AAAANgAAHQAAAAAAAAAAAAAAAACWDQAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzBQ'
    'SwECAAAAAAgIAAAAAAAAmhZD94AAAACAAAAAHQAAAAAAAAAAAAAAAAAQRAAAYmNfd2FybXN0YXJ0'
    'X3NtYWxsX2NwdS9kYXRhLzFQSwECAAAAAAgIAAAAAAAAKpcdBoAAAACAAAAAHQAAAAAAAAAAAAAA'
    'AAAQRQAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzJQSwECAAAAAAgIAAAAAAAA1e3H2IAA'
    'AACAAAAAHQAAAAAAAAAAAAAAAAAQRgAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzNQSwEC'
    'AAAAAAgIAAAAAAAAUnSQ6ACQAAAAkAAAHQAAAAAAAAAAAAAAAAAQRwAAYmNfd2FybXN0YXJ0X3Nt'
    'YWxsX2NwdS9kYXRhLzRQSwECAAAAAAgIAAAAAAAAXkKYBoAAAACAAAAAHQAAAAAAAAAAAAAAAACQ'
    '1wAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzVQSwECAAAAAAgIAAAAAAAAmRHWUoAAAACA'
    'AAAAHQAAAAAAAAAAAAAAAACQ2AAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzZQSwECAAAA'
    'AAgIAAAAAAAAqfbtLoAAAACAAAAAHQAAAAAAAAAAAAAAAACQ2QAAYmNfd2FybXN0YXJ0X3NtYWxs'
    'X2NwdS9kYXRhLzdQSwECAAAAAAgIAAAAAAAAG8SB+ACQAAAAkAAAHQAAAAAAAAAAAAAAAACQ2gAA'
    'YmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzhQSwECAAAAAAgIAAAAAAAAkMaWlYAAAACAAAAA'
    'HQAAAAAAAAAAAAAAAAAQawEAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzlQSwECAAAAAAgI'
    'AAAAAAAAn8eWmYAAAACAAAAAHgAAAAAAAAAAAAAAAAAQbAEAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzEwUEsBAgAAAAAICAAAAAAAANtkHq2AAAAAgAAAAB4AAAAAAAAAAAAAAAAAEG0BAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8xMVBLAQIAAAAACAgAAAAAAACovnZPAJAAAACQAAAe'
    'AAAAAAAAAAAAAAAAABBuAQBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMTJQSwECAAAAAAgI'
    'AAAAAAAATJMuDYAAAACAAAAAHgAAAAAAAAAAAAAAAACQ/gEAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzEzUEsBAgAAAAAICAAAAAAAAG3dRvSAAAAAgAAAAB4AAAAAAAAAAAAAAAAAkP8BAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8xNFBLAQIAAAAACAgAAAAAAADREONlgAAAAIAAAAAe'
    'AAAAAAAAAAAAAAAAAJAAAgBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMTVQSwECAAAAAAgI'
    'AAAAAAAAAd9K/QCQAAAAkAAAHgAAAAAAAAAAAAAAAACQAQIAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzE2UEsBAgAAAAAICAAAAAAAACYw2JOAAAAAgAAAAB4AAAAAAAAAAAAAAAAAEJICAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8xN1BLAQIAAAAACAgAAAAAAADl/9BIgAAAAIAAAAAe'
    'AAAAAAAAAAAAAAAAABCTAgBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMThQSwECAAAAAAgI'
    'AAAAAAAAnT9/L4AAAACAAAAAHgAAAAAAAAAAAAAAAAAQlAIAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzE5UEsBAgAAAAAICAAAAAAAAKFjLYoAkAAAAJAAAB4AAAAAAAAAAAAAAAAAEJUCAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8yMFBLAQIAAAAACAgAAAAAAACFD1uCgAAAAIAAAAAe'
    'AAAAAAAAAAAAAAAAAJAlAwBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMjFQSwECAAAAAAgI'
    'AAAAAAAAwuXe3YAAAACAAAAAHgAAAAAAAAAAAAAAAACQJgMAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzIyUEsBAgAAAAAICAAAAAAAAE3DdSuAAAAAgAAAAB4AAAAAAAAAAAAAAAAAkCcDAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8yM1BLAQIAAAAACAgAAAAAAACuKcDCAJAAAACQAAAe'
    'AAAAAAAAAAAAAAAAAJAoAwBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMjRQSwECAAAAAAgI'
    'AAAAAAAASkoT3YAAAACAAAAAHgAAAAAAAAAAAAAAAAAQuQMAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzI1UEsBAgAAAAAICAAAAAAAAFwuHEEABAAAAAQAAB4AAAAAAAAAAAAAAAAAELoDAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8yNlBLAQIAAAAACAgAAAAAAAAQRNgBIAAAACAAAAAe'
    'AAAAAAAAAAAAAAAAAJC+AwBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMjdQSwECAAAAAAgI'
    'AAAAAAAANOzqzABAAAAAQAAAHgAAAAAAAAAAAAAAAAAwvwMAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzI4UEsBAgAAAAAICAAAAAAAAEIzVxwAAgAAAAIAAB4AAAAAAAAAAAAAAAAAkP8DAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8yOVBLAQIAAAAACAgAAAAAAADJ5AN2AAIAAAACAAAe'
    'AAAAAAAAAAAAAAAAABACBABiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMzBQSwECAAAAAAgI'
    'AAAAAAAAUBSaRQQAAAAEAAAAHgAAAAAAAAAAAAAAAACQBAQAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzMxUEsBAgAAAAAICAAAAAAAANGeZ1UCAAAAAgAAAB4AAAAAAAAAAAAAAAAAFAUEAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvdmVyc2lvblBLAQIAAAAACAgAAAAAAABkT1HQKAAAACgAAAAt'
    'AAAAAAAAAAAAAAAAAJIFBABiY193YXJtc3RhcnRfc21hbGxfY3B1Ly5kYXRhL3NlcmlhbGl6YXRp'
    'b25faWRQSwYGLAAAAAAAAAAeAy0AAAAAAAAAAAAmAAAAAAAAACYAAAAAAAAAYwsAAAAAAAA4BgQA'
    'AAAAAFBLBgcAAAAAmxEEAAAAAAABAAAAUEsFBgAAAAAmACYAYwsAADgGBAAAAA=='
)
_bundle_ckpt_bytes = _bundle_b64.b64decode("".join(_BUNDLE_BC_CKPT_B64))
_bundle_ckpt = _bundle_torch.load(
    _bundle_io.BytesIO(_bundle_ckpt_bytes),
    map_location="cpu", weights_only=False,
)
if 'model_state' in _bundle_ckpt and 'cfg' in _bundle_ckpt:
    _bundle_cfg_nn = ConvPolicyCfg(**_bundle_ckpt['cfg'])
    _bundle_model = ConvPolicy(_bundle_cfg_nn)
    _bundle_model.load_state_dict(_bundle_ckpt['model_state'])
elif 'model_state_dict' in _bundle_ckpt:
    _bundle_cfg_nn = ConvPolicyCfg()
    _bundle_model = ConvPolicy(_bundle_cfg_nn)
    _bundle_model.load_state_dict(_bundle_ckpt['model_state_dict'])
else:
    raise RuntimeError('bundle: NN checkpoint has unrecognized keys')
_bundle_model.eval()
_bundle_move_prior_fn = make_nn_prior_fn(
    _bundle_model, _bundle_cfg_nn,
    hold_neutral_prob=0.05, temperature=1.0,
)
del _bundle_ckpt_bytes, _bundle_ckpt  # free ~2 MB before play starts


# --- GumbelConfig / MCTSAgent overrides ---

# Applied by tools/bundle.py at build time.

_bundle_cfg = GumbelConfig()

_bundle_cfg.sim_move_variant = 'exp3'

_bundle_cfg.exp3_eta = 0.3

_bundle_cfg.rollout_policy = 'fast'

_bundle_cfg.anchor_improvement_margin = 0.5


# --- agent entry point ---

agent = MCTSAgent(gumbel_cfg=_bundle_cfg, rng_seed=0, move_prior_fn=_bundle_move_prior_fn).as_kaggle_agent()
