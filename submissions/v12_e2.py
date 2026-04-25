# Auto-generated Orbit Wars submission. Do not edit by hand.
# Built by tools/bundle.py on 2026-04-25 13:25:17.
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
    'aAwpUnIFAQAAdHIGAQAAUnIHAQAAdVgMAAAAYmVzdF92YWxfYWNjcggBAABHP95v0+nJfCpYBQAA'
    'AGVwb2NocgkBAABLAlgIAAAAX3BhcnRpYWxyCgEAAIhYAwAAAGNmZ3ILAQAAfXIMAQAAKFgGAAAA'
    'Z3JpZF9ocg0BAABLMlgGAAAAZ3JpZF93cg4BAABLMlgKAAAAbl9jaGFubmVsc3IPAQAASwxYEQAA'
    'AGJhY2tib25lX2NoYW5uZWxzchABAABLIFgIAAAAbl9ibG9ja3NyEQEAAEsDWBEAAABuX2FjdGlv'
    'bl9jaGFubmVsc3ISAQAASwhYDAAAAHZhbHVlX2hpZGRlbnITAQAAS4B1WAsAAABtb2RlbF9zdGF0'
    'ZXIUAQAAaAJ1LlBLBwg8cFUFkAsAAJALAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAACYAHABi'
    'Y193YXJtc3RhcnRfc21hbGxfY3B1Ly5mb3JtYXRfdmVyc2lvbkZCGABaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWloxUEsHCLfv3IMBAAAAAQAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAKQAoAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvLnN0b3JhZ2VfYWxpZ25tZW50RkIkAFpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWjY0UEsHCD93cekCAAAAAgAAAFBLAwQAAAgIAAAAAAAAAAAA'
    'AAAAAAAAAAAAIAAwAGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvYnl0ZW9yZGVyRkIsAFpaWlpaWlpa'
    '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'
    'LGXQvOp3UDw3KuM9rI+RPRXp2zu1bLC8YdXAPHL6Uj3edpg7GU5tPeovwz3fZju9JGbMvOjter1T'
    'haM8OEaVPWy6Wbwh/JA890HUPDL0DbxJ96g9Cp6tPKTHtb0XGp+9mLHdvUtdCTys31Q9vwk7vFJf'
    'mT0BXge9X5E1PLArwD1r7yw9V0RAvImP7j1AMlq9ReoyPfqnWT2u2HW9OwORvfoeED0CSie9yjoa'
    'vZtQ2zzZx4e96F+qPcSpiryWUci8sSesPdfiBz2jW408DSFwvTUTnD0hh5e9kzyTPVuSzrynHig9'
    'U1iSPYxUwL3DXTu9Ulm5vVQipz1Fx7891TebvBhmd70WxJw8huUgPfY/GT1kdR68CXRPvfhNvj2+'
    'WFa93J2pvfLN5ryVyO48a8Q7PQHkQT1QSwcIDmgCCwA2AAAANgAAUEsDBAAACAgAAAAAAAAAAAAA'
    'AAAAAAAAAAAdADUAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzFGQjEAWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWizIdL0i3A29zbkTvfNgVL2GP6e9'
    'gYrtPO7mA73g+cy9G15YvSfZY73BfTa9Bd1+vaPNVrz9guM8DTWcvbzzQ7xWpuK8/GF2vQ+0G7zr'
    'JTW9gBcjPIgqkL0kWyg8U4SxPQwjrz3T6DW9bjHePDpeu73SZp09rYG0vYdRsr3CQ669UEsHCFHw'
    'UO2AAAAAgAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHQA1AGJjX3dhcm1zdGFydF9zbWFs'
    'bF9jcHUvZGF0YS8yRkIxAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlq6Jn4/Sx6BPyA9fz+9p38/EdV/P9fOfz+qjn4/q+F7PwOlfz8Q/YA/7Cl/P3lGfT8E'
    'tX8/qNiAP0jhgD/+Vn8/4WWAP0jQfz+lYIA/huZ+P0orgD99TYE/bzt+P1CMgD9ldYE/bRt/PyTO'
    'fj91O38/uvGAPzHKfz/9Fn4/v7uAP1BLBwgL5fyqgAAAAIAAAABQSwMEAAAICAAAAAAAAAAAAAAA'
    'AAAAAAAAAB0ANQBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvM0ZCMQBaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaiklnu9QROTuQxA47+Lq0OnkGgzux'
    '+IW7hQQHvKNKFbzrGuI76GfAOstte7tcoEg7QzSbO5J3Cjtz35Y7Oz87uk1xADyGc3W7rQfEOzoe'
    '+7pj25Q7fDiguwez/Lr2iui4lr1nO4iGBTy6E9s681R2O1udvjrK7QQ8y+OXOVA72DtQSwcIxtDZ'
    'h4AAAACAAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAdADUAYmNfd2FybXN0YXJ0X3NtYWxs'
    'X2NwdS9kYXRhLzRGQjEAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWnNscr1Ac5Y76eyLvCPmYDsn1ze8dykGvMEUBr1mKTK9CY43vbW6CD1ESsm8ezARvPb2'
    'CD1XL4y8iRhZvITeJr00wjy9ankFPVuR6LvS8hQ9z1UCPCoYArux5my96FpPu9iRtLzR00q9VXyp'
    'vIaHrDvQgTW9gebwO/YDOj3nI4A8cAZqPZjp4jxZeDu96JGZvPmtsLzKLDU8QL5OOdHPIj0PQY87'
    'Z9sUvZG1D70eLmm9ccecvKn06DzNrsE6AMxqvaIiUb3poMU8QPZSvcrPSL33dhK9WWRqPYWBbrxS'
    '9sq8OqMqvSIIBT0BrX68u15vPOA3IDyMURU8CrsOPZSqcjtL/wM8SvWduz3CP70t2lC8esFUPRZ+'
    'PL2Atx+9HReBvZPuczvK22a9HF67vHXQVz04eqE8v3ESPQJ37jvMhF69LXQHvSfmLr2xzyC9Bafc'
    'PB60Hrwv5oI9SWcQvV1yR72ZWH09lTtMvQwRTD3GuJg8uBY9vd4yUT0pnb689IgIvRVCXb3s1YC9'
    'IFRAPMBBO71+QYg8yYWJvCfWvzxThJ+8NUPNPGvX7rtmVQ08INrmvMZYI718RTA88bMHvZ44CL0X'
    'FWS9FRjAvLXY0bw1tFk942kiPX8mijztFDM9lijoPKBJeb0Pjye85bRjvBNl77yp/Z08EKqWvJ1E'
    'Yb0cEZq8yANhvSuvKz3wBAc9dz9LPT9XCb1Dc+w7FdRavL07WL36C/I80YkUvS9h3Lyuv9O8eStu'
    'vbrEGjvNooE7V+uNu/qxQz3Mp4U7BEKTvEil1rzWNBO9e0mavL+VJr28HE69Sq0AvIKhhrys2yy9'
    '2L0CPUlHSD1Y8yq92VjRPEtN7bxGMQu8htx0PWyULzw0fE+95dyFPYKQ1znYuum8Ppe4vP9vaj3F'
    'iGA7jY8IPRxhKD07Q6e8HucNPEwUIb1ZTlU9K3TGvCcJ/zx6EDw8Eg00PeZFNL1PWJG7KZMOPRpP'
    'QLyqygE8ZIwiPb1AWL16vHS7JAQPPcGskTyLRmA9teFOO7hopzuSsfe8ZEwfu6CjHr25Weg8Z16x'
    'u8tq8jtY5tk8GmyAPO4kvDu7LV27pponvVgXATwnPl28TFFqvZDPZz0HMqu7vZP0vNRVbL1GSGy9'
    '8ze8PH6nzLzXyta8lKYavQMyWz3k4gG8n3YwPfqPLT3wO0K9hd1JPU5nhrwQ3GA9MgYLPQOma71f'
    'ldk7vkKbPG/Qz7wFOCU9LEtWvSAmqLzPI0s9ER59PdmftTzhkki9oN+GvaoFR73GKOO8+qbuPBFb'
    '1DtnSnq8NPpiPfTsUD3rn9S83EmTvJURnLrgLEU8hzxsuw1huDwhZjc9k7e+ukzY+jyh8vw7frrK'
    'PI8R2jzMX3q8S8/yvLHK7TyXJiK98RMFPSAMGL0oLyY9WxfAPNQXhrzjCuE8saNxvC0tSjzHs2Q8'
    'LHr7OwDqpLpEj0o74OSWvDPa5TwsHHc88s/3PLNsHD3B3Qe9Vm/tvPaMTrrbVzs9Bv89vQyU27yJ'
    'ba28WtWFvbqeT71drjs9MIjXO9M3SL1dLjm89d64u3AUuLxxT2q9Bn7hPN7pJrzwyjg9IbU2vcac'
    'Er3E/GO8P1kFvUz6Wj30ej09JSJXPT0My7z2XU894noxPT0VwLzQO7c7SFLaPHSTKT2KGmK9g9la'
    'PbgtmbsqZrA8R6Y3vRKKTr3ntB699p4GPROmBr0wRSM9vMRUPAjCtTw9eIO8RhR6vNmF8bwXXAe9'
    'stpjPUZeHrwCfv68ccl6vf5t87qKznq9TmRlO2EpHD2mMhK9SY3ePE0lk7zvNzi9ewJ4vYXjCj0E'
    'vTi9y97zPBLqyzuJg8s846A9vaQTST23hHi9pKhBvbIMBD33feu7hOYIvUfACbz8ZVS9CZJ6vTdX'
    'jrxrfD89AIOAPIYcq7tZd5u8CC9KPUBeOzunhaY8GWnuu4ofOjynroA88O8pPXh0AT3oZt46CLin'
    'PGAfqLxd1wY9zCzbPCJEXT2zf7C7oS00vZsZWDwa/xQ9N5xGPaZVxTwZs6Y8stAbPTAKtLy7akM9'
    '5yn1PCigHj1Rpy893aBKvceGqTzZbFw9zgwqvQTosjpRcSm90sZKvG43Izy4B/I8q3dDvAs9Fr26'
    'xy890N0PPdfsQLw5D0U9sAd3PZOYi7wYiWU7Qq+qO3JkW7xS9DG80B3cvGs7+rzcF94863jOvFh0'
    'GL1J80A93PpLPGn5LT3mxkc9MksVvJy1ljy98Wg9s8V9vX4sp7yTu3s6+Jzyu/n1xjym6xQ9NzzC'
    'vNDsvzw1f2w9EjewvPEwPL3jk0292IgHPTTwhL0nVsy8RqYvPammDr117/A8hv7fvL7/fzyadMs7'
    'yheROunLFz21+0w91dwqvZQRyDxQdTQ8ouJeuidAH70jq3A8qpg7PRsvRb0M7129un5SvazRtDyF'
    '+Ds7AtmnvA6Gaj3E7ze8p0gevdfIPzyFIRe9fctdvHOqDz0Dklo8rgg0vVjNRT1faBA90QkWvSUO'
    '2jwH/8E8J/0vvR51UL1F+D+9AYWwPAAn6jvsqjc9sOzhvKS0RT0k+SE8c/o1vXgoeLuZCY88HOXW'
    'PNGbibxnHiG9utAevYfARD3aFB+96UwsPaWl17wt36y7U/1evYv5h7yLNBA97PlYvRbODz2yCVi9'
    '4aP5O30Ibb0xcVK9xrNDvbvfpbzV6jo7hhRBvapGojxlXvg8UEpSPe2udz0xgzA9sz9QvFs2nbtJ'
    'GEw9zhD3PK4dmLommi092aJfvTq3ND2z/s885KY3vSldOb3tMQs9URF4PTwRsjq9Zt68THrYPB4Y'
    'qTxhMbi83MAqPfphGz07G4Q8mokbPW+MXb1YaNK8DMcFPFKxszzIRw+8rvZJPJZDlDzi92O9hLq1'
    'PAK8qjvicRi9LSMCukW4O7z27uy81yBSvXtvXD0RcP+86uz7O0S2RT1spgy8cJkivZIjW73lshQ7'
    'xO+8vGfhCLxv89c7VvNKvFnFFj1zCy+8V9bUvJmsSL1PXO07jGsKvUmDMzwAKgq8kv/fPBAJ17xa'
    'G5282qJavJbaGL3gebI8hbbSvObXizwxPSM99gzJPE4xyrzJiN48MSkpvU1Qcb0q8Bq8qMgPvVCM'
    'Qb3nEbI7L45cPatp37ycTIc7wRAhvU3BBTyrYdA8xv0EvCXYMj0lZkG9uTGlu8ov9rzRzhU9fmoy'
    'PZMzez3TalA95NuFPK8ULL0ONyu9MqNFPMVGeb3fQiM9L7kWPBUfizzhk1O9ogkcPaKeMDuC0Pu8'
    'IigFPZUxsbtLkuq8tMYoOzCyPL1ODWM9//MkPahW4Tw1aAI9q2eSvb9n7LzFhoG92q/yvAQSSr1g'
    '1yg9mXf7PDQPIT3W1gW9kctFPSUmrLxRZJi895UCPZQ1Jj198Di91El1PZg0I7unmpw8DydWvUS7'
    'tTtfPya9Q/4evTVZ6bwXRAW8QZWXvAGSGr1gsxg9byJXPMrAab3LiH8728YePGXw3zotApo82Xsq'
    'u+tc0DzqBL28iwmyvE6DQj1xQB29ajK4PEVqdD0rwua818iDvA5JYD23W+66hqQ3POmeEDybPZQ8'
    '/LXOvCWSKTy/MEE9q2Czu82zVL0x7YA8rC2PvCYrUrxkHvE8JQO3OwxZGL3C/oi8L/Kbu1CFibzl'
    'VYe8FM0uvcrlJj2CQPk8M8uFPHdH5juPSuI7XQRPPV01+DtxN008iFD0vJEBojwjQF+9YgjBPLwG'
    'W732QQ29vJQBPbUs7rwqNDw92HbpvOSJRz2wGi48iHoEvUrPCD2PVGi9j4Y0vf6ktbxe4Io7H6yw'
    'uiA2Ob07q3U9GtkSO9rPCr2loIU7ck2nvH5IozxAmro8tPpwvAoLIzkCyQs9Q9VJvTXb7rynsuC7'
    'LsUyvTi8Ojh/tSq7XWA3vVtYDDyg7Vs9eZ+wPHg41rzLr1o9PkgzvdJknLzVRt08VV00Pdqpej3A'
    'DWg9EDDpvBNqbzwrL9+8WURUO03KST0bTQA9e3RXvWtdyLyCU2e9BmfTPPMB1bv2clC9dm51vGBM'
    'QjywpVU6+eNPPV74tbt4d+E8ZVxmvDChDzwzVCI8xFw9PZ89az1VZKO7EGJYvbdSer0pYra8iwMm'
    'PK73ED1WIim7w35VPCL4fzwGYU+96Z2BvZpKdjyH7Lc84D4zvetkk7qdHHC9pKucuk7S7DuilGQ7'
    'zJWbvDt/njzcppC8UklPPV0Dg72MAwu91V5GvV+7cL2SMTc9dChYvHrSPb1aFLw8FrcfvQgbxzo6'
    'Rdk8kO0nvIwEuTy0u+Q7bKJYvdwwXDwCtgA9/dMTO+LDCjw9SC478Uuqu4mudzy59ni9DjlBPfPD'
    'vjy3kuw6TzovPP8hLL3mq5e80Fo0uGNRbDzaxhC9j6CeO4DstrxPDA49ww1wvT8uozxNCAg9oWZO'
    'vcXjizyoaqs89y3LPFwo7DtwGf48Yp4NvZz4Vj1u5Wm7XfcmPTgNjjy91xO73tTcPNwgcD3GAUk9'
    'JPoqvUrKE7v7ikK9iZgYPdL7Kj3TXxu9VLiXuya1Rj04Zvc83uoSvYkXEj0etwa9XqU+vft6EDw3'
    'yO48DNcFPYZnMr1uTQ69fU0cvUow+Dxk3Vi9k14Mu4xL1Dz/2A89DMtEPB7EOL0Hawk8ITZZPY+C'
    '2jslVve8lH8qu3LihL3bzw+9vBCWvHWYIL0j+009OuM0O+c9DD3JK5S809FwvEzGFDyLC4W8V3TW'
    'vEqCbr28L00971tIPb5ciL2zvJ28lTsrvXtiy7y/8Tw9Ga0ePaUWh7yI+uY8BIKWPNHiCz0TumS9'
    'zvd9PGlEY70Wyz+9jovYvBtkN7zV4G292N65O8atsjuWBNm84qsFPMjGYD1yz268HGwkvSVzYr03'
    'Pwy9sIwCvfmmTL3b2ui8oFfkPJLYdTyv7lA9tp9uPSkhl7w4Ra68cOUOPCzHP739tec8KfkTvDku'
    'I73pBge9VmALvblogDxDUyA9dhlRPaQmwrzgiwq9cePVvMfZHL0ZFpW8U/xXvXllyDzYvmu9m0QU'
    'vQwz6jwYjCi9WMZLPOjr5TxMMl09xan6PG6FpLwgeQw9/jslPDlsfrz2o6Y77PObPKi/Oz01Tsw8'
    'Eq8EPccmNzs7Syu8lfXSPHG1GbtiEXG9ARssPBzPML0Cvuq8RUHYPD1fdjyNtg88RoIevQQSuzwq'
    'il09AxDEO+Q6lDtQG3u9koHrPPt1zDyWibo8j4jvPNFgjLxzUV06RAxxPfnoPT3i6i29QnFQPTQC'
    'jT1Yd4c8gJ0DvQsdgTxW9j28JPshPVK2GL2o3Zo8LOOiOjY8IDzWMzE9hkG0OjEdQL1Rywy9iOfy'
    'vAYCOTw6Bpo8k9iWvH345bxxjmg8bfBHvcoEJT0mtfW7ZKMjPdpsUD1jgqE7JXVzPMjC0bwglCs9'
    'PbbyPLNvB7sBF+A8YvTgu/U8OT2hvfY8ASA5Pc7ikL1vims9/0izPMFKyzwxTEs8bUsQPTnBVj3O'
    'yMy7F/9RvR3pWT1Eeua7UZ7ZvNfZKz3Z6xA95opvPMLbK7vdEAW8IENtPUJV+zz1kuw8SWMkPa46'
    'K73kiUQ7a9hZvRTSLTpVcJW9k++AvfH7lL3ixuQ8x2QaPCPVFbw4SL08eiq8vHkghrzFQhy9KiO3'
    'vD8377zcbBC8r0UTPWA2bL014IE8DLxNvSekzrzHXKA75ibvO+VuZD3xKWi75c2/vLc6+Ly6My87'
    'AxtLvS8qOL0KcbC8ItzjumamHb0Lave8GkokPfeEaD2/FY68tLpDvXOexbxjL/s8FeSuPGK9mrwR'
    '2Ui9gfuYvG3OlrwPCRE9OnXdvPuvEz2Q03g8p3ynuybJXr1S2W09AQaZvHazJT3tKQC9+wqBvQ1k'
    'RL24VNi8IlSZvPygRbwUsEI9nDdlvXbjEj2Y00w9is68OllEDb1XTLE7+f+SPKKZXTzQkoe8aFem'
    'vGia5rzN6oC9kfaVPDAIMb3tkAW9kOrhuhC1UT1xeVs9aiocvVIyMT1yA1Y9cvI2PTojCTzywTg9'
    'mC4CPRwJ7rzAaSG90bksuxeOxrywSyC8Ea05PUKvHD2clmo9b7AUPNcxm7ycChA902F+Ow0Zgbv0'
    'x6m8PDXEPG33UL2tGUI9hQxIPSKkpzzeYTw90y7tPGkr1bwhICK9vb8IPOAUID2TQKy7bAvVO+2L'
    'Sz2Q4bw8UXgXPfo4qTnAEBI8dZe0vEtTZLytN+K8QUQJPSTEZT1oX6M8gFG2Ok+VwLzDi329wXF3'
    'vKttOb3iHkO9dqMvvPNqTbv2FYg8L8I0vQeGVD36EHS9hJgbvShV4bzAe/E6pwqNPNntFrz/uQY8'
    'wSA2vCG1jLz/Pz26R/ETPIsJbr3lOhq9n0omPf/J9TwmXos7DrL7O9eFDTwSElg8Vs8xvc17pbx2'
    'K1m9ki9XvatECrxzZAS8kqY5PYF1ITzyHVO8P6TGu5qG2TxCiI68JQ5EvIRALrxbikK9oLSJvAI3'
    'c71HDwW9y0pHvWXjLz35nnC9KKdfPc8T0zx3wSi9xXA/Pf4mO7x8GUq9NExEvN3rujxJMWe9D9gw'
    'vCct1jzutL682GeTO8NLyryuYya97QszPZ6iO73LdK+6HWOCPD8RkLzK1QE91/NEvVxYXj1f1M08'
    'GHEQvEiKSb0kpBg7752EPPNb6rwHfdm7tHN3vd6icb1BIxW8nykvPRTdhbwwuis8WMYFvaay/bw/'
    'KAy88mUXvabHKT1rngY9qqguvXJ4uTxlEva8vBkDvdi40rumA8K8uqcvva7ZcTxg+fc82RhTvGwP'
    'OrvFVxK6qI3XvM3/1zyCpVg951YmvO52ADxroEg9XzA6Pcc7SL2q0xm96xiEvcG4Kz38ygQ9/Bye'
    'vHy0tTwCug08UmhkveBQMzwcXvQ8EoFau5DGS7rn5YQ8vzwxPfkBUD2UkBA7UcCxPIKYx7ynGws9'
    '2DP1vL/MQr2nAvw8ai5UPKtbsDxAF8Q8hObKug2krTy8zFI8DM0BvXyLsLp96iM8TjQrvO5YQr3k'
    '3x08R1ngPJuCDr1EZ2c9bYPgOylXpzy6Mc+8WNk8vZj6xLxe/b869aDmvCF5WD0gui69Yhkuvdt5'
    'Jbz5MF89KlHjvJLgdD2b5uc8ynAevKd9Bz1X5bs8Yg/qvFAL/LscDFM9DWcOPW9o+LzTV3W8qpwX'
    'PUhgojxpsSs7EbQqvSVzE71NO0K80fh7PKvtQLz444y96nORvE1gez0VdE+9el0tvZL4TT2InHg9'
    'iEtVvSfXWr0rAZi8BqA7PTOEIL3Gx5Y6dl4lPWKUqLwTT8W8lT4ovSEr8jrkvkk9Jr6pPCEfIz23'
    'lRu8R2ocvKSjPrzeEPc8lo8wvNVAYL1wg8a84iWbPIb8Xz2Asp+8C/ZZvZYUKr25Eie9lguZPKLp'
    '+DwChKg819RaPPITET1iXBU9iuMQPRc3Gj2KEqM8zALaPG13Ub3bQi69GpVlOo2w3zqEjG29L9Ft'
    'vWxdMr06eFY9V5wgPd1s+LyXEZU84y6KvFxRQj2cqla9FvJEvSZOWb3oJOU8K/0WvSFyEz3NK3I8'
    'vFQAvcggoLyTDQi9EPi2PMmPjbxhKsw87llsO5sJSL1EoFa9QhAqveyy2TyefAQ7rCGNursfEr3F'
    '4Fi9Q6avvIcPL73vHoC9nr54vZUJLb3q9IW9+OjKO2nFjTyJbqQ8sr6DvQy9Hz0FrDw835L1PNfL'
    'hLxPPqU8AvKAvYiXqTwrIDw9QP3UPI6QZb0+Wby8q0b3uwJKBT1paCS9r+EXPLLA8rtSYBA7Yuec'
    'PD0I+LxNqQW98Y9RPZZiXz0P+BG7gkJVPX/dTD3NGr28Z1A0u6jeOr1MXGE8gzYHPMwJGrzpllw9'
    'TV+YvN3Z67x4yiu9cWnsvH8Sez3VvN28hSzmuw6k7jz2FD29jSkivRXUGD16JNi8I/yfvDiLrruw'
    'iIu829ONPLUpfDzjMj+9XRxMPU2cXb30D888pSlkvG+P5rz5xl49Nvx+vXqQ5TqmKhM9M5RPPRBQ'
    'Xb1aHr86wNVJPVJMgb3msbc7ZDk7vYXuHDwn9369FlhXPDc3kLw3JrU81KGfutb0hryWmDm804BY'
    'vRT5Nb0ZzAs8nQs8vb7ROD2R1Di9UV7lvMX5orxuK1+9ZBqSOg4iDb3wK2i9hkNXvPEoMT3mluM8'
    'YIZAPe7+WLsXySi9xa8ZPQAhHb2v/3i9GiMRu66W4DzzecK8Z4fkPDLcuLxPJXI8je+fvEqPJbxa'
    'yQ+9q7CDPI7XebvFqlQ9ZfKSPJ47U705RU29qqj2vIMf5Dz3mfS8qT3qvETE2rr0gB29MaAAvQqt'
    'RD2Km7q8RZxxvNeZLT2nRyI8ZLcKPfjhcb0Vnsc8SYy5u187IL0Gc868xtlEvZsg2TwcRJa85hiB'
    'vbunZz2CoR89vJXBPOFJYz0LckM9OeR6PTDdWz3BjVE9au2KPLWPnLwAnZe7DLUYvW1ZwTy3Dru7'
    'y6JIOx5KO72uQey8pv08PQLyIjxlPAM9NOC5vPr7NLypJnM8rIDaOzeZdD1DwHu8QbpkPQqI4Ltk'
    'KIo8Fm8TveEOkDvSZXe9H65RvWpWJL1ZVpM8DI1VuycHhTwIm2m7VxZfPW8caj0TB608MdVmPUWy'
    'HrwBBry8AYCFPP52Jj3hbag41/sgvaSJzDxI5NC7I4vmvH0qGT1MdQw8sw3ZvL0zD73Y7jU9Nd7e'
    'PBLXTL2W+Ki8dhoqvLVeiDu+KWu9V454upRerDwl7Rw9dYg/PIueAL0VIIm9T/eTPLexej3ioIG8'
    '4tAnvTyorTwJSye9qrv2vD1WZTzEHD68vpsGPE77Rj221Ju8lHwrPYEWibyN/Tw9Mh3kPAcsTT3j'
    'lcQ853+WvHVT1TwBXVo9kOMKPNVjDj10PhC9UPsPPaqiQb11Oj09gU7jPAeYHjxvwV+9bMpyPSgN'
    'DD3YZ1s9bFoiPXPImzwjF0K85c/KvJ13qDxYrNO7VRcJvP+PlDuurdA8UxBlvY0Hpb2MIKy85b/K'
    'PBof0zwC42e9Pq/XvJ08FbusXBY8Kz08PVCigrt5BUG9WQXiOhPa7LxjPxy9S26mPJgKX7y+rPQ8'
    'NlpTPQKSNDyCEuk8vx/gvLg60Dxyygc7rZgTPeteaT0bYTK9gx1XPcGa7zxzvnC9gyNpPEZ5d701'
    'jc48QpkqPPGrY70vHdw8qu5Du/eQcT0/1jQ9j6UsPUO3KD3zxMI8bz11PbQJArwvuE089e5DuwB/'
    'Lrt+WgW9RBemvEpM4Lslemg8DqcPvebPhb3tqmc81PZjvSRsUrxouVe9aUVOPUqNSj2k8Ui9iv5e'
    'vbzgyryAxSw9C9/NvIkCXrzHy9i746NgPEcA4LxpX9g8UN0hvVQQczv2vUc9WDfjvJiQFD3s0+U8'
    'UJ7avGO9rzyxYxe9cgZ7PAosATzoR0g93rUNvd7K5byxGgO9HXt9vaxJhTw3R2G9pnC4vPzqMb0c'
    'CoK82sgTPCb4ZL3v+6i8IbElvK6HLr1HPlk8HmAZPM6dFL3VBVQ9pfc1PXcgY7xm0SU8WoubPAiC'
    'ErxJIC69GAhWPTK8AT1dfXC9bQUnvXU2UD1GzSM8z3eRvBMmDj2pIos80uxpPKqcLT2cEnm8B40r'
    'PdUWK72UsDe9wPKOO/iWKrxnXwE9EznNvB7mTDxX15284nMlPS/OHbrb8KG5Wy0hPeJMgj1s02Q9'
    'KPNmvWALPr3gRI68CACIvBJl6jzkf9U8iZqtuieHKzxrw6U7xdA0Pe7cBT0NarO8Jq8Xvbo1Ujyp'
    'OWG8qiAavWWqUL3a4A48GJQkux2h2bwBP4Y840FXPd/PIz0VzRk9hINTvbkWGj1rchU9ts38PNdc'
    'lbxV3Vi8Dgo3vZwCt7zIqWe8hp8VvMLsa70/ocM8N3qGvRcbX706OZU85TpRvdy7dz1uw2U9vTN6'
    'vBEMk7xRmlG9txMevJkUqrtAEAe9rQhCPL0QhzxsXuK8wzcCvI01kr2mRDy9ct6sugFcTT1+pBe9'
    'kMj1POL9YLtXDQa9HK65OhXPDT3KIzU8z5tZPdPewboUWBS9ToMgPf2g17ySLu48aKYxPQtuXDy7'
    'ZGW9liGHvdAvmTypJUC6fBbsPIh9W72/u2m7an31POPI3TxpOR49sxgSPYWyULwnyvi8DsjvvCXj'
    '2TzcMaG8Bonlu/wYYTx7phg91JcJPbsGvDx9cPg6E9AAPaPQxryJxeG8VcpEuwWoDr0gX1c9vbYN'
    'vMAckzzskaq7iclfPGKPxbwftos7FYkKvXJaCLylaVU9ezNovWAZBTwwgYc8zniJvUBG77u34RU9'
    'a9A8PRzygzz/9i09Jn+mO+gYoDyLV4c8C8IyvSXHZLz7XSC9FpgZPWKTmzxxgq68F8fbu8gT2DyA'
    'cEO961W3Oiwjuzy0vzU8CR0DvAcQED2EDLU8WrJjvRQ/OD0mRhS9/8s5PSU2VD0QDnc8PtHGvC3Z'
    'Cb1FT5G8AVVbvJRARbwmYEk9Gef2vAlVhDyVhli9SUUDPZMVPz2aUEq9HguGvHS2ejyQeHG9E1xH'
    'vcnVdTw7slw92siKvPtaNzsjb4C9TKxNvf9Qfz23hVY9TBpPu1j2V7zMEAm8pdMDPcMtLL1uwhK9'
    'QzBrvPmZMrwAIx48EbU1PTvdwTxaj0y9ITZ6PcL8OD3mm6I8HVvOPFQB4Tw4SFA9s9l8vRmWK73o'
    'hqC83hycvATXFb0IyVy9S+tZvW4bY7yesUE9Pv0fPaJEbjytAxi9PyHLvH0RRzoo3fK8bnOfut1a'
    'lTzX2JU9s7yovJzwfT3Gci074jY7vQ8bhz1V/0k9TXsmvdNUB72BHMU8BxcvveyBFz21rNu8+RWK'
    'vDqafLybFIA9b6rHvG/6TrxliwI9jeEUvdRIEju0z0q9ckNyvb/tirvMsd47po4xPZtkd72w5Y+8'
    'Tl2mPE1bnzz9Id48LGRlvYIyR7yCcXE8VpEyvVCr7Dxc4nu97lINPVLxpDwa2py8v6p3PP94Jj1L'
    'cyM9pPPePDQYDb3AkWM89Lo8PfcIcjye7g49FfkXvQjQHT0iTKu6MatSvdJrlrzvV0A9H1O9vFpW'
    'LrwEkui88+MFvZOaMzz4FSI9yS4YO73mHbtEbz88+VQ1vf7IAL0VXeQ7C1RLvdKQFz34h048xLB2'
    'OuUz9LvlWz49zJ9TPRXAxTxZk9I8v+60vGcksjzqUh+9/yL9PAnELr3Lj6K7TRAkPcy8ED0TVEo9'
    'yUG/PJYkqzz9/qA8ZgZWPasx0DxWz3o8xeSxvKjoCb3Eskm80/K/PGgUEL0kOVQ8+nRSPbZ6Qj3S'
    'J0W9BGGGPEOgET0lJmO9C90nPQ68Pr2Qcyy9jugGvUQXSr00AcW8j5N7O59PRzzH3GQ8sQsLPcoZ'
    'FL1dbGO9kXIfvRZoSL0wkwa9qRo7PAkVVb1wlyu9NWQDPaleJL2xCTs8nQsZve4+lTygqAI9iLdT'
    'vXp5sTz1OdO8mE01PTwPQj1SBjW97GfKPEHgOD2zFTY9x39YPfVNAT1hbai75C9KO1SE5bwQ0gW8'
    'PxgqO84bOL03LZ+7UccevCa4jLxDGtK8QgJ6vVhgfT0GJBY96idGPQe/ir0GH3m9wnAZPR4ZM7w/'
    'kI4714+QPC52FDsTegy8ocrRPAjWQT0CNWe8q6w2vQ7VKr1dOBw9Rei5O3JTTz2A5L28MZ5UvQML'
    'mjzmjPS8YbEGvWDxLT3AdYE7SArqu9NwcT1t30o9k2QqvVIYkr00RSS9ya9ZvU26Gb2g5Rs91kQ2'
    'vfqgSD1JNk48aiQhOwANDTz50zU8jXSRPCGkSr24ZhW8WyxbPZwKfT0vj5i7/0IKvRGyrjxCKWq9'
    '7nYsPZXUBz19moo8KQalu/+pZD1uX7e8wJwvvXBM/bzV8zA9ADHWvFL3Pr23rhM8LD4pvVMxcDuS'
    'bCq9DXmBPIi7TLxglio7yOapPEe8Ib1jn4G80I0gveyOST1yVra8VYXyPPXpNTxVY3K90aZhvQqW'
    'd7yx2ke9z2rKPLX5Qj3//o+7+JkNPAUoczu39mw8Vn9DvZlzv7ykcwi9oxqGO+ZANz2fEcq73vYr'
    'PdKFnTzC7NK8o4/lPP1CTDz2pwS9RPM8vCQS4bZhoOU8VNNcPTDPlLyoiAg9RZ7cPBC/aL3qnSI9'
    'ijRwujp4HT0zVU89nCt/vQwHWL2EU1e8Vbf5O0Xw0btnSFO9ulP7PKR8Ab2B39s8Cb0tPZChmryB'
    'qAs9pSHrO6sMX7wtsZI4YYSKPAy0HD03zOu8SFNfPdBEOb2rjky97PjIvHy5a73w4XQ8nKZxu0jd'
    'bzu2x0y9zuEAPQurFjyfWV28krCAuwzvwTyh2hG9rnYFvLbZ4bxaB0E9fTkBPUWoQb2Jch09I4LM'
    'u3W+VjtVVqk7oXo0vGG6fz2Svig8dY/IO61enbzsK4Y8Zh1avdKEoLz+XQs9Aq+SvcTXbjs1rgi9'
    '2d7JPHblDL1rv007d0mfvL71PT2BMrS7PZojvch+ibwnLUE9kAxJPF7INzw7L/48//hwPGNjS7sc'
    'Vw69vdNtvQM38TxttzA9ZMWFvAA757zOIOE7q1ZcvDFtMT3EJAG8XZfBu744cL2UiiM86EgxvJEe'
    'hDxgqfs8+lmxPCEfHrvG9Ju6XirrvBe/bj0xiUk92fWVu8Wtej3qP1c9/X0RPR1VDz3/icc8LD90'
    'vHczojvMki09rac0vHojZTxj/pO80/Q7PDf3AD18CgY9PXkCPf7n3Tzq4FK8KwlNPdBnCj2PBlk9'
    'H99JvTWZ0by4cWK9OYPGOxohOD1YYy69tOk0PIz3Ibxc3lO9ErHIuu//gLwOYqa8IMAMvVA9CT3Z'
    '++K7NxbPu+nSuby2vna8Eo6XvF5lL72Jqwo9LueKOxb6RL3XGM68nqKIvNJwVLzzCw48WK5gPXn4'
    'Hr0VCpI7Fd1RPVtuDL1xRji9vr2NPAe7grwm8oC7MeBCPRawZb3sdfe8u0sxPbagGr2q7yC8MDjb'
    'PJpSI72UFj69UdMtvX4MOL0cn3W9Bq2yuw2aTr2oqtG8HhN2O+rOYj3xtYQ9wtcmPRIXcjwLo+A8'
    'IpCuvI+qLL2hiMQ8AFqkPLVS5TyYUfE8j7+uvOzYPT0qZ+A8GAFyvKLpJz33jao8R5Cmu2lj4DpO'
    'Hh49skatu/WwlzxOmEU9QHe4PGvBij0Z8Aa9FT8PvPmfdbsOyuq7uxguvZRFYT3DZRi9UfQNPZyV'
    '47z1S6U7IoEYvegU4LxJYZA8eeEuPeepE7xOBWY8yBMhvcNXZjzswxO8m+YTvFsGxjx7zrG8+cMa'
    'vQZ3ibxIyeE8vUOBPTgCVz3KOjs9Ru0KvQZVKD1TSp08Ih4YPCz6Ar2vrwe9feBtPDEel7r1JcW8'
    '9SVivdihTD1ZGZM79bQPO8cV+bvymxE9vyGkvHtmgjtCg0A97mWfvMkzQb0rYg+9jD4FPWy3FT0a'
    'a/s83o6rvPkSBL3BNlS9NKgVPIMkebxg6GA91ZjHvOE6Jr0u7eU8eWT+vBDdrrvQ6WU9YFXIPCY5'
    'gb002fE7LuSiOwvedD0mukO9uMX6POwcgb2AW4S9UJcDvVScZ73d+Um8CLRHPeFfg7zy6zm9SP30'
    'ux75ML0rySc9DMM6PAkETD1zn9s7FHOePNHfjjyXBFw9cv4YvWgWG72ipuA8X3jYuyVaWD0/Lj88'
    'myvKO9LuED1zP+I8tRglPauIP71TsY88WTYGvYw6ozodweE8d8HRvDQXTj3bR0g9MBCUPGjOOb2k'
    '1NI7HMcwPT+3hzzhnkc9xLwRO+Ve27wuUCY9PcjrPM6YFj1Lm+68px5hvRnZKD1Ve4M8eEElPahM'
    'Rb1KreC8BhZPO9xVBL0oUkI9JywiPbI0Lj1bZga6ou2GvdXTuLwSDl08fzaDvJSk5TwtJTK8ucQ8'
    'PZ4xQT3zC0I9t1UuPAixab3mll+8kc74vGhjnjzf1zO8tXSuPEzE4DzPk+K8OhOIPE84Brwcphw8'
    'GvIMvVsg7LkIQCw9EBgYPdUXwLzu80E9cYDpvJtTez3fHdy81xI7PWTkyzxyPaa8CDwLPfYzXD3j'
    'FU29jJiHPF+GGj0BmWO7nQ4TPdBBIj2r0CK9qVoqPb7wib1MRvE8ShVMvfnL+7wzS7Y7v25Yvfvo'
    'VbykD8a8SEC5PFUM4rqNxGy9xReDvCvLEjw5AtE8thUuvSb0IL1Y3LE8qHIkvZmEh7z56Hm9GdGs'
    'vPoNTb0rUlm7ol0lPeCRzDxBxS69NiFiPQnJer3Q7Ly8cAfKPAWaezxJfwM9NPCUvFV0b71qG4W9'
    'LsmtOqKWU725Bj+9j1MWPTErebyc6mo88dBGvWfkPzx/5zc92P25usR60zyIvhG9FlFhvafFRT1b'
    '3TK9IVVVu7E7S73pf3U869IkvU7dvDwnHms5VB82vXvfLzzSPFO8JxeSvBG61rwZH8474qDpvIw9'
    '2DzHyym9S9iWPS7qvLz8LJg8psoQvbWTGL1MKsG7ZZkPvRYSZbzRcIy8iPR2ug3wNL32iVw9w2Il'
    'PZ5jury+jUw98AYYvDVwgb2PoK88hLrGvPSKKT0T17C8UqEivZ3/ZD26gk8801U6PEa6VzzV3U29'
    'Xwf7uxw/DLyRu129+yKwPKxkgr1pgxm9kJWtvDe9YL24jHq94bi4vNVB5rzKxxE9KGG4PMvyOT2J'
    '2CA8fgCiOY6XnLtodQe9UwUAvOVH7by6p/M8Fa3jvE0MEb0ig+M89R//vOL/pzyQkwC8ohcxPDlY'
    'Eb34Jiw9keBpvVMJc71dP0E955hJvWG1dj0MG/08QnT8PP394LumUM87tGatPGZU5bwjAVU9IJhI'
    'Pe/bMTyjqqU8AI5IvPFPt7wDmiq9/hNYvLVebb1l1Jq81BYlPVEqfz10loS97uhAvbw6BD2m2t+8'
    '5qSJO2J1yTzGFho9YzSgPEcQdDwuinS8x5yLPO7ewTzHJai9K3Vtvd8ODj0QfIu9L4ooPWuyHD3l'
    'Exs92BdgPMS+qjt1f5M81UA2PRqKCbtBoGG9/dIiPSz6NDxJJBY9l/4HvdLFFLvroke9BkfXOmna'
    'cT1qBTA8fM9VPXP5dD1zQBS9pTCPPUV6e7pg8i89rmCmPAL2X7xGSi09G4Umvc3+rrv7Aj+8n9lL'
    'O9j2c7wA5lU97A7Ru+5P5TwbRrE8H9JmPVoB3jzgWXS9Fi38u+gRcb3GkFe9kNU/vWFRgrylCUE9'
    'XjdAvf7ohDwPhxK9Xv7EPDcaxDzRJww9UawWPYcNQ70zqEm9rJ8zvYw5iDxHZ1Y9ekNDPfMr2jwb'
    'eYi873r0vCxuhT0eVU29jawOvY3f4Twvu1A94SmYO67RMLuNBri8LM3EO3K9R73jelE9xEFTPbh1'
    '1LwOVFA9EviyvL/gCL3yzRi8UK1UvID35TwSgTu8um6GvJRhCb1k5OA8/kpaPYVK4zwfmge5H1lE'
    'vOR2GD3NyIY9mKbYOzB5kToGdnS9F49HPAwyBr3lY6G8g84Dvf1ZRj23GAw9pDUNvU0XMrxfYbk8'
    'pUoMPXV347zHrNs8FzelPO7AGT2QteC8cOYPvdNUAT0NqfS7VZ7avJ7MOT0MaRe8OiYovbU4cbwB'
    'yeW8MYOOvOiAVr3T4To9dDhSPewpR70gaSI9vhiNPLGjEb1BZrq8bP8JPCPKqTvH+jg95Vk2PUeO'
    'rzuKPl49IunqvEcCoTrTjVS9gobOPC82mjwNFh29VlI6vfu3Yby5Jqm8+am3vERKQ72cIBu9BDRT'
    'PSXtCL3qjaQ8SwDQO2xRAbxykqo8fVssPQs9gTwG2sA7CtcwPRVm6LvM51g8EyVbPQBlHL2fBh49'
    '1pLuvPQtjbu7TSq98tALPfKob713k/m8MSARPC7SkLz2iog75jhePRoVZr3fOya9EJMkvesFdD1l'
    'q4E8i6syvb2xgr2OXkU93iFvPamdJb1NMSk8EF4ZvQVALT2hOAS9gzcjPcHBUT00rgc9fI19vN7Q'
    'DLysgvg8Tko6vStQIb0K4Xa9j91HPVqp9Lx/g2M9mgftu6Zs1rwVzug89gOzu31PML3EMVa7DaYE'
    'vcNTMr3aoBS9tvzPPOvQ7rtB58q8EhxVPAbXgLx1kSC9L/LQPJVbwLww4KO8xP06PQdhOT0Oblu9'
    'm350Pdw/ET1d4VK90zGNvU27O73oBMG8qo2wukN3ezxzOI+7WOQkPCVBVL0nDEs9FeU2PbMlAz27'
    'SS88lOX8vAm+MjskyS696uRcPS9pPL1S40k9ef9RPb71E71rZGO9G2QEPZ6UP70vRCi9bNo8vUCp'
    'FT1E1i29d5zBvBFVwLz8WkA9k97LPAR3Az0vuIK80A8yvaXisTwue6c8t4loPWFXTL0UZxO9oIQh'
    'PfnVmDxWyBg9M1hyvHlwQz2ziuM8p/KevNmlCL3gDzs8qThDuw6c9byhrgu9m7OyPHm94jxq+FA9'
    'pDi8vAWtA73ydTi9Iv0Pvb1zdT2MceO8J1rivE1yQj0e2gs9yr9dvVt/pLy31u87ZpwBu4qLO70U'
    '3k+9UsYGugEfEjzvw768ixhLvaxVgj1J3L48GkGpvHFjyDuHBPq8yhdevLQ6U72IihE95mxVveY7'
    'NL0d4Uk9crkXvcQTjrweg086ErVmvZEVZL0+IlC98oadu+mZmbx9hyk9xWgPPR4haD2B1VG963/Y'
    'vLCiMD2GG5k8lpcHvTB/LD0x8wK9q69vPfGRVr1D9aI88oEGPHTNXL0+4dg880XCvDTFUbxuG9C8'
    'LylPu+iagryokpo8p31zOgIdDD3y9/Y8qvEiusQ10zzJmdW8UARFvZ5TKL2jiii9ADAmvV+FJTzf'
    'O++6TD5YvY7hBj3sqQk9E5dcPUNBqLnMDIK7gcEWPafZW71U7os7MsAWPSdMHD0GXjM7b3Vfulf4'
    'yTwfNog82BOnu0rH5TxjcSY9jn8PvThSYD15AFu8QvHsO0wWMT2OYBS9FTPpPI6zsrwaw/28XOK7'
    'u8LZMD2YmvU8oMlNu8ltAT37bBs8ZR6APS6WOT08CtO8xRoHPUytVrw7v1G7MfLIu40jwLqusC07'
    'UMdhveNyOjvcZWA9QHAbPSSHOr2hqYS6hZgFPD1MFj0Ctg29uqSRPDH1R73V2Sc8K48IPRkLLT0c'
    'H4C8DWE5PWHHrTzz+DC9mC3GPMHCqTvrw2C937IxPWwMUj2Lw/w8NGJzvILvL72Relw9zPbyvEkW'
    'MrihlHu9MkHXPNn9HrymKnQ93sWmPDVth71GLzw9aic+vWe1Ab2dUd88ML74vK4gdzzWkkK6BqbA'
    'O2ROuLwsBCY9j0c+OkzNUTwHDdS7ITb0vK2uvLxMOvc8d7h0vccIr7xhBoG9xRo+vdexCDxVhW68'
    '1ZAnPHtLKD0xt2I8j0taPSn4RLwN7Hw9AMQFPK5yB7xRBw08kMdBvWjIIj0eCEs9kf+AvbpdJb0K'
    'Ajs9k3WKPBJvlLws8iK9p0ISvSHuDbyFm0E9lUxTvYlzEDweYO48k23LvF3mY7q6iiy8s2soPWY+'
    'PD2VE/66zn7zvCHGk7y1N2O8cIaZPWlMXT1E7rI81dRPvbBMmTxMCSS8zskLPRjHyjy0cjc9Z8ET'
    'PHTlbjyjRiE8j2cdPYIUIj2A4QU95CVJvI144rx+kFU9RR7nvE9A5ztYvQY8deOoPCJMPL1g5QG8'
    '+4oCvA+W3btDpE29E+YQPN/wMb3F1gi9jzyUPLUfKj0qIXS897A6vXLoCb2LE6W8pAIqvRX0czww'
    '9sI84wlCPTioPT16bqO8N3zlPLNHw7szgJs8gLDnvGaHdD09w6c7jagfPcfTSj3C/Z68fIlzPX1x'
    '9buauvg8Ad4XO5loZr3OoIC8Z+9aPL2aIj1lTza9BlcnvcYI9jzxwou8sVUoug2C/bweWGc8a9PR'
    'vKGZWzz23+M8iStaPFQiU70MlRA9sgRLPfyVQL147kQ9xIgevcjR1zzs+jq7rqUEvDOk1ryx6VI9'
    'NYikPJLkHLzv8AK880MZPRvpJz1aMhC9MCRPvRsSP71DXri8kjYUvb2KHL06iOe8fpuaOuxsojwL'
    'y5w8WF/vvHgvF72hdhi95vkTvUiDGruVpjK9k+8kvBW0bbwayDW9OMPivIYkyzyRBqG8YxgjvWTK'
    'VT2qB2q8TCYgPLN0ST3zS/Y8LIwpvZxQWz02W+S8g7qXPGx7Bj03x8I8h95UPabfg7zn73Y9YUJO'
    'PV9GU73A7pK86i+KPdfYXr2StW297g4SPZEaQDwXqgG6+nkoPXb8ujvEpk09Y48evQjJbr1aaQy8'
    'W0xiPUFeYL3XIrQ8GitCPP18gLzh4RK9YOKJPJv6kzx+UU+8SGg3PAtuyTwOoTq96GOTvQ4uPrzZ'
    'Kzk8TVcbvZC7qzwqxke9hzOJPb1nG7yPpMa8S7fJu5eDDb0bf2C97mWTPH/txLw+kgq9IzCnPC5S'
    'Lz11GPs7UDIXvTf+mDxCJR69FXaPvIojHLwanIu8oNNBPTw2sbtap1u9XrGdvKkjvjyog8I8y7qN'
    'vDpeMr0Gppa8bke2PFZ9XL28y4I80SZBvM2hyjzjTIA8ZqtFvHUclbuRA2Y8dmkCvWtzcrx2iKE8'
    'LwY5veORX7xYMwo8ydrlvOTaCD2Jfx89sSpFvBtVq7w5i+w8j/2Ru44+Lb255x89l7hEPfegUD0A'
    'd/O8bKRZPRY0DL3/CFm9qHgePbcPSL0Zx1k9srmBPf31a70bOi49XIdhvJFYc7w2Q0U8PauEPBd2'
    'GL3UtUI90d8yPU6OF73ZHMM8oK1EPLriFz0wZy+7CItePKWYpDxZhUw9KhUdPBxYUD3IO4U8esO5'
    'PGxMvDuWeCK8TpE+vQSFfj2+Ngu9CcwpPeczgTy2uS49eV00vSTogrt9wGi9mnYcvde1vDxIOw28'
    'zvg0PZYOMT2U7xe7WzOhvB2mFj2bbGE9iRH2OqOUMr3ox4Q8LNpQvOPuKL2pgbo8dCWFO87EPjpl'
    'rm49mhX3PIH1MT0/bz07pxqCvDm6dzzexm29iI2DPUCp2TsYLm+9rskKvZyC0jwXr2M8xFSjPK5O'
    'f72/9oK74mXCPFRemDwb4kM7GvdwvSIePj3NyJA8Z92iPIkyFD0kLBM9oEPuO0lHGj39Ojs9lgxw'
    'PKWqT72fSDw9waoFvcpnNz0D20W9+fViO2VayLyw1/u8xSGAvBlz/TyOqKQ88z8MvPqFTr1BoB+9'
    '3TIQPW8u8bwj+Vq953YTvXqiD7ytnu68njsyPf9pE7w+Bnk99/87va6nLL2ExIA8zCX0PMEZo7wD'
    'XW48ccnPvFLTLL1t3wc98QC8OyYcKb1iXmE5e9VIPXRvsjyCKIu8Q8zCu9E2M70z5tc8LBG0u8pp'
    'Br0fUug7WOMzPL2WhLusYZG7xeh6veANxLrdMpa8wrLwunNkBr2GnT08UEEsveC5krwdxg29L5AU'
    'vWH/ETwpkw68Y3vIvNGc1Ts6w+I8SK/yvPv9BD0p7Iw81zwxPWYErTuSKSc9P0hTveqmBjz+TGC9'
    'I8HdPDcn8bz2fV+8r/ZIPbrQbD1U5hy8/mz9vHHFEz20fpW8+80Cvcna2rw7lJM9aQ86vAlPbDx6'
    'Hvq88rTmPN+par0oNZW8gyQyvLEy2zy2e/65tkauO6++MT3sBJY8MX5ZPYJQ8bxCmg09VX0UvOK/'
    'HLwbPX08n7lIPcqkwzvoUC+6fsWmvLmtB72w7xA9/nCAvGFe1rwZRbg8v7hiOwXgvLuEFlU9CDgM'
    'PbHJKTwIy8+8sjXyvHbITT3FlMg87qJoveag2busEys85Y6nPG/aWD1k2Om60tZ4OnmDDj2jA2S8'
    'uiDFPEXUKr0XhKu8SgcxvRtz8Tv67WS9M55BPdvN6bxdXxO9Q1g2PKpOkbw1Vwe9bsFCPSy0Pz30'
    'l8k5xvUAPWgqGrtPhLy7CjeMPXF6abz3C+c8TnFfPX23FLzddmU9cxWMvMauKrylJk883H4oucWH'
    'NLwjWIe8sFGQPGlGGb3RdRm94d8gvWu3Rz346hQ8y4fqPAF/Ur1MpoG8Q1gkvWyRzbz7Wj4917Yz'
    'PSBEEb1iStO8JdFYvdD1vLw66+Q8CYwgvVYL/zyNYXC80CNuPUG2PLwPJtK6uGkDPdB5Cz2z5yE9'
    'KikfPbeld73P6P48lvH3vOAlTD2szzs9O5NvvJhwDD1I+GE8Snz/PEKqFz1GeAY8w/oxvLjOlDuR'
    'ENo7NVoUvckN+DzPBBe92BAdPSA4Mz2+tEg71ihMPXqd7rjCkAo9VmjePKNhF70YVgW99uyBvIPY'
    '3jwhG7m6j8INvPPm7jzVhlw9zkHRPBz0Mjy0wd48hsSpu7SaXr36T2a9gWJdvXmQUb0lCi89DgJu'
    'vCP34byL06Y8tj1eO2Pe7Ly9p2S932OoPNG50Tum+0a8znmuvDGu+rt3OFc9k9BGvc/EtbwE3xK8'
    'CmJhPeH3QDy4OSs99AcLuICYorwWMPg8BCvJvP9pET0e6WA96m1mPMMsED20rEe9lEc3PamcG73k'
    'xCE9OJ6qPN32qDtayoQ8QmlbvXxGgT3RFZ65Qi86PRdgNTyohbu8loEPPSH8x7tdFAW9CiWeO0IO'
    'Sr3EAQu9cAhSvetAEjzL3/S8I+HtOn0EKb35lmq9koW8u2ftJDxYki091pu/vAHYgj1IKca8uKbV'
    'PIBBbr3lYls9noLDvFcfEz0mXL871t3VvE+trLxNtOq8xX0APSm7Rj2mFXU9J7qevBQbRrytYLy8'
    '9gXXu7HuZzkVvja8PckfvXZDbTwr8Lk7BWo8vbEgmLw/3oQ75A83vY4GETwkfAE81Eb/uLo70bta'
    'WuG6yTQtPKwD9LwHEgu8ahkoPc1M4byBCi27CLFhvZfkNL17bzm8FVWivPeWhT2Nxqe73W5jvc7o'
    'XD2t5Qq9BB5dPaQh2LrxFug7a3hZvVpIbT2wIf25jAqbvJA/Gb1cD0g8tmCGvS6yHj1YHBA9MCxJ'
    'PU0FWD2vpxq9RZT5vB6BKT3orBI97VGIPR0ixryAjPU7zgaYuvQVMT10k7y8ost2PcKQ0rvq9Qk9'
    'dYRFPe5eX71LJJY8p3QavWaq2DxbSEY9hy+mvA7AIb0evtG89xRJPaQ0n7ug1x697N1bPSKMKD1H'
    'V4u8A30lvOA68zul+Wc97YYmPYCVyDs0MUi8kS/+PINR5LzDggM9yHaoPA+VjTst1AY9A56/u/dh'
    'W71+EUC9HHlgPZ1sgLws/0q9Y8txPVvpDb3WYEo9UihfPSSr/rrWz/w8cyBBPcoxLb3OyvS82fuH'
    'PEbywTzxYyS9cQYjvXjgaj1v4L+8DmpJvXB7Az2+JMO88D2iu9jxjbwuLG09WtbCPPJZcT0bj5S8'
    'KsBBPYKMTr18Ow892TX3OvhBHDxvXlU9koMCuaFF5ryA0RG9L6ApPJ5ToLzTKdA8Nc1jvbK3E73k'
    'IGE9LKfPvJTMer2dzS09FLrLvAKGsrom5ws9ZP90POH5WL3qqUI8UdqBPSBaJDym50090cxyPRgF'
    'yDxEQE+9ramDPSPjOD2B17Q8X4DPulC8Rr1E8r47ACDAvE8WgD2tCmG9VJw8PDFQfLvg+Ws5Ifi5'
    'u9QnpzycN1K9cTU2PDKFF70eRSs8A376u7zqO7svIQ09JJqGvFv4F72PqKi8ExRJPSkrOr0vhyc9'
    'X/owvDjO1Ls2T4Y8oyfmvMxIR73NmlM896bjPM2clTwlySq8MSW9Osqenzw+tT88uFYKvaBGHz2C'
    '2Gi8vfvivMXpq7zkH+U7tawFvYdiQ70fG9Q8/DMhPT/Pgr2TDiQ9SZosvfxGWjyKplA9NQsrvdN5'
    'BL3tzCc8QPM/vVYQ/LyxdXC9DeVou3teGj31eTI8voOSvBD9Rz3xlB09laQhvYOBzTwTrKe8YXVG'
    'PHAijDvtx9A806cIvfVPDr37/iO9dqo+vdtYAb0iU7c7ZNsjPKMI1zsojBw96zodPbGbKT1ayOU8'
    'ehYxvCkREzy10BG9BrW2u4asmDr0QmG9gakfPfsAiLw6Eh09H5TIPNwIEb2NY3U8xBTuvFhs5bwj'
    'TU88wdzKPBtkGr1DAAy9CmcYPWaR0LxunzG9QAZjvRGVBj0xghg9K9JSPWV8Pr2wPCE7gnThO3Vf'
    'lTvu1+E7quZ2vSW4dr2fT8e8aGBGOZzlxjozE1C8+qd8vRddET1+7C69Ea0bPBLJLr3vZiY9MwZK'
    'PV1iHb2UjTE9tobYvGcSgr2IqXK8+9JVvQJvGzwz7QU8ahUVvW6mt7xQghw86sQkve2bpDyufVG9'
    'uWXbPDR8BL3b5Fa9EB+uO0ZVqroQA0694VE5PWQD6LwZHvo8sBXIvDRzDT2jOBC9mNs+vY4pZr12'
    '5iu9O8U6vXVYljwcH0C9NuiwvCs0ubyRtsG8lcbzu2BP6jx5mO+8khFdvXD6Ej1n7Bi9WPQZPSJu'
    'o7thmSS8MkNPPadle7ooaYG91O2JvRKQhb1VJ069gSNlPIyykDzRR9484qVtvX8O4jw7jzg9hImE'
    'PLa837wE3XC9rOVKPZlptzt1IhA85XNSu38kDD2trYC9Q17IPOGAOb3Eyk89EdROvS7XjbwtQJE8'
    'OSh7PLB7gDxLFWK7i20fPBbRMb12Gvy7tH83PMAZPT3wgVe9gIMBvYazM72NChG9K0EmvcDYSbwp'
    'do48+kyRvKaLZTyUfIO86Bl1vQrIsLqpy/C7GoBivVTJiL2UmVA7dTPpvKvQMzxycku8yp6dvIob'
    'AL1xckk9Phl7vVZ1zLwJf8u8W1LzvEbg4TxhxlI9sAk8Pa4EEL0Z6wK9quiIPIUx0jzYn2q9YO4H'
    'vV9rB72KtTa8UETjvJCvWL3nMp28AO6vvNmkA7veYvC8D9wIvfJEQT2wsMC8bTzlPMdVRr0e5gm9'
    'WoiJPI7+eDsRiFG9je0fPWd937wp4QK8jo+bvOUTCT04dKk7YtOoPCLqzbyeSkU9hV8YPJE72jzu'
    'mWG83Y9dvQAHOD1oOly9fxdtvfN6BDwOaha8KRQwveVNQz0iCc07c9O4vPu0+DzbQVQ861VavfbQ'
    'iLzf4W29w7srvWN5kzvfSik8DwjuvIjhmbxwTEy9EEofvTyeN7sKvoM7mFkEPae4Mj3rClw9Ct4u'
    'PMdKQ73FQ8684ZxnvU6OK7xG6+Q8qzRFumyytDwXk009t+/lPJNXo7w9IzE9q82uPCbCQb2CsWA5'
    'nwWuPMtt1Tz1QKA9Zk8XvV1kDjvA6V49XQ8avfVIBL2HeiA9QwHgPI3IMz26X1S8Ug4FPQg/UL3P'
    'DvW8T47VO5I5B73mAoQ7WsW3vKcMjj1W90W85ch1PD9QYTsCcl89IHWQuyFCtjxlvzO9q3QqvVzd'
    'Lj3UzTc961zVvP2EFj1hhWY8Jf0XvZIQ1TyFBVg97UGWvEy4Ij2+y8i8eo1wvSycWL2rM+i8Ty5E'
    'Pa7+tzwzhDU9QmnUPEZ1vrvpiim92gpcvYcElLypcLi8HHNbPaZNGrzGt/m7s7mRPNDqVD2dE5o6'
    'YdEFPfcLSb3ZDHi9uKMBvF6Xqzy5/xk9KLO7PClNBj33BxW9ohm7O/t+Nz1JPhm9jgcYPRtnR733'
    'WRY9hycDPUWDQT1xLCe8lNk3OQ8gB73joF09zSH0ukBFarxTew08yXD5PNxUvrzOPiy9afVovAvL'
    '8zuByYQ75M9fvOxslzsXP3K7QP0nPS8AD707P3k9XTODu/tcET16FM08KNtHvcxdLD2ecJa8knUU'
    'vf3FxTzOtnI8e7acvOKa5Lxscis9Q+9zvdcQ2TtSDJw7hnJHvRhC5TzfEy69C1tFO5YWVLzX8vM8'
    'f+iKvNAJcL3SUwm73d0zPeuJM73Q4kA9rum/PG/GB7qM+ua6mS9bPTt3vro2KoE7pgKDO6w8rbsP'
    '0Wa9lmAivf2QOz2TBwo96m5KvJWxVz0HPv+8j9DVPMJGFr3Z6So9023cPCwlvTw6QBY8lqEdPWKo'
    'DT0Y5gQ95dBbvC/TErvRB4u8Yo7QvD32mDuCmU49UZqgPMgi3bzPyQI9P19EvNQbWz12qHi98SxA'
    'u4WUar3b7bc8Jv4FPS6UKD320uq8bBENOv+YQTx46p28KvptPZhFLT1gsOo8EskAvWw6Pz2yUde8'
    'Cb4/vYCykLwO9HG7hbshPX75oTylE5M85aWovCzuKz1iJGk9OVldva8BTj2NJiw8eNeCPM7fxTwo'
    'WDw9OLU7OH/i8rumZo68YKAMvQOYLDx9oPq8ukRqPSOsR72XiUg9OvxivVq0HT0X8Ys7CrWBvSU2'
    'ljz5hxS9+kg4vZkIErwBubu8m3+2OxctIj2LTGi9PusxPWZrGr2VGzy8x/8LPaakzbxkP/O8WPyL'
    'vJfXtzzLn4s7R/BTPVlMnzz7CUm9NqBDvQfGgzwoUxg6EjqwvN8XvTz1jRI9bJMSvUTD9rtV1dI8'
    'hbuePEqDErzIHL4843asPDLBdL1yaHm9HNMKujx0TTyrHyC9aGkhvQ1Ler0bniA8JxY1PW2pLz34'
    '6k+9ikJFPeIPtrznT0a9aeMnPVoIUz2nCMi7dl53vC4dgTyHU8g8qxUAPUC9zLwNf7074bKoPPml'
    'gzyWGZE8SueQPBRcoDybCCg983jkvPhBWj06OV68Q8M+vXjYFD2i0Su9cAV+vX0prbxphXY9C/JE'
    'vZZHxjm9+jG9qYyaPM7BCz2TDUi8bcA9vQtYW711oRC9/7/6vP4yy7wSQgQ9Ad5MPA1eRr2gg9w7'
    'xt5gvbOiLD3Etqc7JUdIPTYtGj064yi9qEwZPf2VPr2MtGO9nj/6PF5n1bwoxi+9xDsrvbl8CL1g'
    'ZyS9wbu9O7Z/L731opS8yTIQPSxqTr3Wb68841wLPUQmyrquo5K7KwKCvd7RUD3/ZGY9WMLbu2dR'
    'FT34MIK9LVVWver6SD2A+3a9SjuAvLW53LwD74S9gBhUPBJFBL133YE8GpjevHnPZTz2hjc8wQSI'
    'vT6d1Dz/bOI8nuJXvefp1zupzei8/6gVPa1EGL2dihI99+o5vYMB17yo3wE9LECUPDybe7zRRCc9'
    'F6wJPLeg9zxmTeq8PEJvPLhsxjwg+Ag9InaDPXttCj1tX3U9q86CvHQBOb0SobE81f/LOsTRcLzl'
    '30+92kIJvHc4HroRLg09cTjmPOaEoDsJhSU9k95LvO3SSD19qFm9S9iuO+pqrbzvLUY9DOT8u9PP'
    'Db3cQ627KjYHO3GCJD38bxE9yeJ3vQ6e4LxEBR299z8Wu5I35DxWf2c8r0RnPbqh4zs5L1U9xG2M'
    'vK7aBD1E5Hs8KXH7untqo7o+Zxg9ti0SPD7P4rwPhRu87c5wu6Z4KTzCNTE9zhLPvO/RnjweX2W9'
    'bdLOPFttcb12CJu8X3BMu1gPpru8+SA9gRgLvM8Lnzx24BI9wTC2PBvdib1B5we9lIk4vJmguTs/'
    'ywQ9NQAsPVl9lzwAgO+8Pvwzvfo7Lz3T3Za7y+htvaTtcLyr25083jaCvDnRPD0rDfq7Tbcnu4HO'
    'Oj3FWHK9ASfBOsAnfr3htQ69rQhKvcg8Hz1WW++7JisUPROyF7qw3p88sfAVvaYHA71znJq88OIy'
    'vRIfJDx51RQ9g9eQvEqPK73t9wO9K65zvedHLr3XN7u8dGIUPDLUFb36g2W9+augPI4wQD3NOqE7'
    '1nW6O2Co47x6+r48wVlcPHQ3U7wVAH49GCIkPUJcID0AkyE9CRzyvMqAXj3Ko807BvxgvcK85DyO'
    'Vka9OefIvAEZsjx0vFk91t8cPHwTCL0ZoiS9eNIFPZZgrLzHqJY8k3M4PbKrPL1Iikq9P3RtvKFc'
    'x7xmkQS9K+pwPQRvIb1zaRY9sm4cPY9zTz2bqG+9w4hPPZhG0LyM57e8tlFIPCJfxDxf5Ng8Hd8K'
    'vZ1OnTwYN4o6isKwPFvCzTuQ6Y68zkvlPEPkXj01aQ89/B6svG9sOjyeCRs8oKPvvI+KOT3oAwQ9'
    'IxzoPIUfbT1wJVW4BT9MPRt5X7u020Y9u5oXvCtAY70Umfs8g8TGPOzThb0vxMo8sqiIvAxHxzzZ'
    'w309tMRMPTnHxDs9ebM8OqJjvfxd9bvJMAE8zuSiPGYXkTyJ8jA9TKT1OzeYjTxwx1e9S7gRvSN1'
    'Cz3E17c8bTCLPIC+Fz3eIFG9fnzQPK7JM7wTGC67MJSsuxXhiD3iBFU9kjEXPQMExrwK/708Di2i'
    'PHy7CT2qiQK7JuCDvWyKAj0iO0W9GZplPLmXED3vOM48XJRNPcXoTz1NHFC9N1KsO2gsErxZTsq8'
    '0XgIvQtZez2AMBC91f2/vAY4Lb3Pvem8TWwfPCLCiz0bg+I8iXmYvFEeID3ob0W8Mb6TvBYbsruS'
    'b6g8ZYVfu12wqjx+Obk8JeILvYSiVz2vKz89wFtZuz6sID08bDS8kKVIvPlM8jy8WiK944p+vF/8'
    'Ib1rbE+8ExrPvDa30Lwb52W8VWI5PcsL3LpvkHO9NwgVvToqMrwK94E8miobPV+NLD0F5yW9/J5M'
    'Pci+MT3Gy1W9V5RJvSxqQr13eym73KGUvKJkgb12KMW8T9kHPRThybukEVi9aBM0vZHB3zx+bU08'
    'ozCivJASV72uQ+Q8xv7UvF8Dbr2GAlk9/MHSPMt+lryHlLE7rAJrPc2iXL05n2q8swDFvKAib70r'
    'LTo9qZxVveUi0rxVTb886cNAPRmLhrxxaik9mDSuuy8xdb1jWGG9S6kHvYi6KjwZL0Q9heiovJ3V'
    'rzpy3h89u0DSvLfT+zyaxay7v1udvETAKr3CrGo9QfMkvS/BuTo3K5I8vG0yPDITv7sgkG68CQAm'
    'vTNVTjyuYJK8gpRLPAMIwDw/l5o8ESjtvJGlML2mno+86hiSPHJ7aDzjuog9IkNEPXpso7wWb0U9'
    'BQv2vF1iILzpzCs9kCkFvUIelbyookK9xZK+PMMDDj1HBkg9/ZEGvc50OL1p1e88+VUvveoQxrwL'
    'glm92D9FPfN1I7z6Ow27b3XzOy1zJzx2BO68H32DPUEydr2mcXe8DzUIvUZ/Q71AKf88xq4yPa7M'
    '2DxkAt08X/QNPRZc8Tz2iYG81/wEvWvNkzxdVwI9rroTPSuLr7uLzHI9jD4pvXlInDw1S089vOiP'
    'Ooae3zzfeGg944hGvRDeBj00j4Y8CUPzPOS88TxYkaE8RPsHPET9/TxEZHi9c5eSveyU7zztozs9'
    'wfXtvM3YLz2GHgS7/0EDPW2EgbuJqeM86n3IvHEuKL18TuA84aTcvHbCET32uMc7HWKgvLt6XD0c'
    'hV49DwPgvFjd8DyW15M81VQcvYpbszwNMky9ppejO69cHr1LvEk9zh/RPOFEbz1UUpY8FIanvKx9'
    'ZTzNAyE9eOegPJM3g7wX1nw9pxk6vW2qfL3MdP07hWdePbRVkTgyVYe80YhXvfVAAz3tIXe9euzY'
    'vP0PqTxoFzW9jYg3vTkn8rwhoAc9mYnJvHeslbyP0C68hGLoPCkeWD0HgZo7KSaNu6vhX7wSr/c8'
    'dAQ4PaeyVDzm4w89yd+SvOmMy7k+0ke9yLDMPHoYA71GVGy8s+VRvfGDp7v/lqI8/34TPT0tcbz8'
    '/pE7ZUpsPVGwA7lzxxE9rVNfPY81XD34Ms486f4hPdA/Dj3SDIu8L7EzPft1L7yhhui8QOcmPc0z'
    'tLzNGuq8QacjPIaF+zvvesw7y4wOOjaCrDx5eGk91sB+vAtaAz1EKQM8l4kjPforTj2TzXG7R2o2'
    'vLSbxrwt1WO8O2PevDg9YrxGlRY9IfmYPBg53bww2A89eu/XO+2IxLsSeto8rKcNvdLlTbx7ri89'
    '9ysiPdZ/hzz1y4C84wRUO1FaRz3qaBI8OZsXPd6MJz2L2LY8hi4JPfQdsrzzfEG9DGJ4vWyQUD3u'
    'Uwe9PUxevbBPnzvOHYW84fpxPK62yLwbuko9r8xbvTzyb73GpV89I5XgvAbzuDxndsI8LpI6PBe4'
    '5LwHM148LnAVvV/MOD3W6xY89lFUPeE6QT0BbbQ8f7GsPOufFj30WSC9JXU+PQLpNLyTtg09klwq'
    'u9OL1Tyjkic9+/bsO8kMfrw3KbM8kI5vvc8Wirsyd1O8sXstvZ7ilbsaWTM8raGGvbK0NT04A/48'
    'TvdrPb+947w5MR+9+nwaPTAweT1VkQe9S8sUvaykbbqPUE49oUTGu5I4X73Cmuu8gnIavT50Gz35'
    'HDi9URMpPTIYjDw6OkG8H7SAvMpu5zyA6PK82ROqvCRePz26H628O7wIPCBFLT1BVEi9nSgdPLWS'
    'JrtSv+O8kcFyPWfGAz3sdAu7lUcJvLBODj11Soy8BwoPPQH8OL15NlK9etHuPAOIJz3IGja9bZYb'
    'vN5ZvTw2ZSQ92aAtPRTEIT3a7c28dp9dPJLwST1VGDe9AZUXPfx6gjxtLi08ineCPbEu07wJAYC8'
    '/WUuvah2PD3JVki9t4FhPe0wQb1SlwQ9X0ngvJXVZT3T8uq8MihlvYU4m7w8cDC9R81xPUx2hjzn'
    '7AC96igwvYCLrbwozvA73+cFvdHi8jzX8Pq8wJc3PDaMHD3KfDu9XtY2vW1Djr1SSV66qHRDPSkS'
    'CL11P0K8czoUPQCmGD1Fl908SxBqPKbnuzzMnYM77OluPSbAqTwj4Lo8VhyQO1aPNz0R2nS9+HGl'
    'vHFigb2O5h88YRucvIKOS723Y1G96jw5PBx1QT1to2U9aP4TPRhyVr2Hfz09uL9RPXtehrslonm9'
    'KAMqvc/jVb1Aps87RLNRPdWgIz2DkK68FmL0vMoZYTwfmpK8FJm4uzXLhDx1D4q88uyPPLBSzLt3'
    'fFY9VYc3vbgZRD3ibB09crRQvTB7kzuC0867mIsDPVKjHz1mouC8/gfavKeBR70Rw0U9T3w/vd+N'
    '67zVN4s93JP8PAiDEj3RDUQ9bcVcvYGMn7vPPYO895eFPe0Yz7z4SD+6UF8oPTbN8zz4jTG9XulV'
    'vcuR27yXv208iK9mPaAFwzyzfQY9K8VWPH1BdbyQz5W7BkU7PXm2EDyb/Yg8J5UGPY7wIz0uPGs8'
    '75HEuz8xoLxEKwa9+fz5vHADaT3jJ1U902LXvOpNPz2MvCq81uZsvClO7zznjF+9Z/41vQopbLyL'
    '+sC8Sl3OPETUHrvAW2G9S6MUvXMcST1AHbc8Ql2EvUfFW73y5SS9KyFpPSTzAj0ydZE8HMUFPVJQ'
    'cDzLsig9IasqvXAHPD3HBew8XyqMPOcyIz3Txhu8jZjSvMpVEbyZACe9puUau9Fw9zyCj1s8WKpl'
    'ujNd1TxWu0+9EIkbvbN5ab1A7Wg9jbFYveBayzxelzc9l5VBvBqfbbxJskE9kY5HvbNChr3hr3i8'
    'TUVFvJ5kBr0lhaa7QCc6PeM59zwBxZk8UPIfPc1/DL1kzDK7AxejPNy5JTxR48a82fpZPeORVL3Q'
    'O+e8LXFvvS5c+jvcAV+9O3M6PUUdHL0orJw8vHh+PTDVfj1mLHg88pDZPFoQVDwgJsE8loMivE3H'
    'QL2jCTw9ETBFvZD6ZT3sxvu5jyTEPAGgOjs58cM8PkMNPcw+dr2s6Rc90eKDvMWuSL1ZqiO9Evq8'
    'u8FvXz3df3C9BTQ8PGO5ND2zt1K9rr5XPK1mCb3V8B084c7ovIJ+NT0uAIS9+8jBPJ5wLryOTVU8'
    'GxjBO/f4gr3vmeS81DjyO8iPfTsWFAW9eYn9PKg5zrzBHvu8e3fVPB59VbuwFbI8rdgYPQf06LvD'
    'XT08u9I9Pb1izDsGZFs9C+LbPGUCOT2bUjE91u0PPXsbdjwLxOY8CVHqPFB+/zvDtDY8bdigvGPi'
    'Prxl0gA9/M6Tu5WohrwWeGQ8TfaMu/cVHjxK+x+89+Wfu/3THbyIXDk9iKNJvc2vN70KAxi8Zc7b'
    'O10UQD07c4o8He7Su7A7W73QVgC9i4uoPGZqibwlvBG9Nio7PKmHMz1DJzM88EkyPctWUT0Xt1u7'
    'yHx1PdW/jTy8Da08Fc/9O8RDXL1wR9o86astPWoghTyZ6YI7uC9kvG5ESr1qsbw7Yjt1PCCaBD0y'
    '6W89JFshvXoJYT0ogFe8lViivIV93rzijim92W1ePAx9WrwXSDI8lG96vawHvbxwq8U8eIrRvE+R'
    'Xz2sxEq9hEc7vfNYOj035109BY/aPCQOHj3m0Re9WGtcvJdHRbzRo++8pRtEPKlYET0+XCW9UsQL'
    'uzChezxqFXM9Ubo3vekvQz2SDgy8o9UePPjnGL1VWUO9r+X+vJSsm7udSBA85FdvPVGwCLw96hs9'
    'pfB1vfpqLj1USF+9O2RwvSDnBj155Du95EMfvZjPFr2SZLu8kXXWPMLF5TzQiTS9CiqLOlpjP71I'
    'TlO9ZbVfvFGg/jsFm/Y8TOagPGQpizxKjEQ9nag/vPXLlrrmqFI9Q5aWO+SSJz3Wk968dVcNPQ4v'
    'MT2hjUw9hGQBvajCZ72NRy48ZfY4vO+5WD0h2Ly8m9rXvID68jorgUC95iQtvXIYOj0t7Kk7RmdL'
    'PMVBCbsT3QA9MbbXvG0OhjyZjGw8q3ijPPkaFb0L6nm9iHHovFOST739HF69oae0vOL497uuWgU8'
    'w3oHvV5xEL0eUJO8VJ4IPYv0Gj2IVVk9M4nPPFpeHj1n2cQ6FGwfvbzusDwktkg9H0uxvK1vzDv5'
    'VFa9Tk1APc96br1bu728omaBu6Jh97zNJVQ6/66+vL83Yz30hQ69JhtsO3/1dL2fyKQ7t7AxPY+c'
    'Bb2RyHu82eM3vHHvWDzQ7B88fLlUPav0ZLzBS7w8W05CvT9DEr121S69Tf9+POGNLTyw+mY9xbuR'
    'PN0ymrutDsG8LtedO+xn67yBBiQ8ONJOvFE8F72KKNI7LauWPLelCbzD54Q7Rnl0vdFBRrxxK4K9'
    'v2FRPakp5zzBlUa9t4RFvetlTL2jUCY9OqBrPL4jRr06blm7l6M1PemdW7xGolG9UFExOirahD00'
    'fH09g1BJvJA84bxiH1C78JkGPWGkAD3eul+9eODOPOgSGT3LtVi9cXAnvZ1/BD3yntQ8NhdwvW1P'
    'Dj3wOyE9v8/vPDpyBT0PVLY8l5wVPdzyJj3OXgS8IksPPPcQmLw2NBa8QjfqPNPOrbyvdmO8Fp89'
    'PeaHqDyZNF49rs1KvCtvk7xWhjo9mDL/vJXmxzwfVSg9CrkpvTco77zNvTU9y6VNuyJnSr1QtQq9'
    'kkmBvYsVo7vq09I8fymBvBbheLz+fRu6JdO5u+FuI7wYaYq8a5I2PLvMV7uIHlm81l67vDmqfzzr'
    '2by8DmSuu8VCs7wn++Y8i8CmPGesNjwCEZO8k4I/ve8JrLwRmla9aAwxuyZZRT07a2G8ceRDPQD3'
    'Or3bVCE9CxGmPHB6Fr1UfIy9lA3mvKIGojxBJ1I9qjFmvfyFGb0rYPM75EPlvMzrbb2VRv68MDpj'
    'PDRXer2wJLG8IArHPFEkoTwNmwC97D00PfmYlTqSuuU8fgncu7htybxr1Dw9WNVgvVq7bzyLiNq8'
    'P9eUvLtjPj2mxmu9yBOvOtIwu7y+0C49OavfvOLVobz++wS9MWEWPbKA5TwLcPK74GdUPET5Sz3M'
    'vxu9oCBWujZXRD2AIHe8B1gcvSSAET0yJCk93m34PALQ7zxYn4u8LEBYvQ5jgb1vu7s8YLzNvFYK'
    'Uj0khto7KdNHPQteWDu093q8h/bGvHkBKb2fpYe9St/SvAMjQL2aro69YG/iPGic9LyzA0O8Cpaq'
    'vJvJOr3oXNW8xsAnvVELKzyZA049WfoIvOnFrDwYQuw8VAYtPcO0yDtR8ai8X5Vzu80Oh7v8mYI9'
    'mrmSPGDpIT3t7tA8bLU0PWRC8TuZ7N67g08gvILK/TwdewI9rJ5NPUeGD70HE1A8S1LEvKbIEr1R'
    'DCG9ifu4vPXMcD1zbBY9MToTvSxJ3buKIQY90z8iPQ3CKD0rnQG9AxLgvA8aeDxSFOQ75cf4vAbR'
    'Wr3p4bI6uVQrPVZfAL2WBki80EWKPAVLTL3WPQy9il+Ju66xBr39WRE9xjOPvNsKNTwIXem8TSxs'
    'u2mrRz2cPSq9kbrmvDIzm7wrXAI8abOYvM1XGDzhpge9PueQPBzhZL0DYbi6kBn1PGSYCDwcIJq8'
    '62UsvFApHjweChW9dzQovYdYezpmGkO971zBPKbR6bxdTSk9kSc0PUVFNz1K4wu9zW4mPOvNUT1t'
    '00y8bdGaPE05Jj27AYe7xDvivCdHXzw8aOk7k6dLvU+N4bwxeFy99Gxavc8GDj3nAww9c78/vW0b'
    'hb1mml29I0H8PMxfLT0HSLC80QEBvb65S7060Ye7r2H2POKIxjxP6/+8cQ8CvbEzXDy/XeY8k8sb'
    'PRpmhDtPETA9oERLPXjfCLyNi0y8ApsgPV7HBjovG149Eo8dPcEp2jv4BYK9hXr6PMMuE70mHaK8'
    'O5B7PAuKnzww9bO8xbVaOgraUL1LVfW8bR+LvIvgP7wiBtm8O935PG0/n7tM/j69CAxPPGyqs7xU'
    'PFU9FQOcu8fEEz3AZxo9rzd2PThXwDwV2zI96gq0O9J69Twg5AC9bJ1GvdGKN710c2a9rTEfvAkW'
    'P73YPWU8e55nvZlKQz0oYW08YDJGva70Uz3hnvk8IQ4CvN8ISD2rDla9zjZ/PeVjDDztmVC8Vtan'
    'PJadtbyqQ048W7wZPU7ibTyXlBY9k0MdPU1NFb0tpYm771ElPYo3fLyldVI87z/HPKitgb2abHi7'
    'h26HPcjEULyu45O9QfB7vY7DTjwAOde8rXYVvWEPRj3qjh691LAsvbfBPz1dFT49hDyPPOrwdD1b'
    'cdM7HzkHPQDwDb24Pia937xgPKcvGb0aZCi7r5iRvGgobDwiFdC8GpJFvKRHM71Cd7u8f372PEMf'
    'Mj2msE49kNtgPEWCgLzDkTo9iN5BO3mb0rtnJ2y5LVw1vUuVobyJO4q9NXYEvN9der1CNc8731Fk'
    'vFjxEjzMyCe9mYcIvOHtLrv/lSK9ixYTPYKTg70CKX09Q221u1hMCL25h2G9j5xGvUsPPryaUig9'
    'J947PHWyPL0ryHM9xjhXPTs9mjwQOmg90m6cPLmGjrsnwTA9yALwvOnbzbu6dfS8DKXsu/7cfL3t'
    'qsc8vknnvCT6qDzl92Q9eIwHPTiSF7yy+GY9DNNtvQs8LT3HxU29s7tfPXmS5zpwaQC9vFmbvBr5'
    '0rxve9o8Q+pyPS/MIT2I6Vc9cMCYPcg0N72Tbw49yHYEu63/Kj26OSI7iTyQO7A+Tz3eqJe8K+4Z'
    'vTlMT70jtdg8RYxrOwciM727VRU8cduou1XDzryZUUI97o43PUpSgD2XHAe7wRWmvEPRK70ElRo8'
    'aBYdvTYaHz1yd1O9BpYXvamQXDzBGcw7IXbxu5R8ebzdJge9osOPvDFoEzxr+nm9JxiROvhvKrzG'
    'vSM86JgtPVMRQr3gd8g83UXSPIGX0TykzDe8DKUcPbxqJL14R5S8W2BZuslTVLxqTnG9syEaPYWS'
    'pDzekFS97j1QuzTNPr07m6q8EnwKOrzUJz2aIio98mGHPCPXXDxanIG9dNwEvSDLdT3w3I27b5xB'
    'PLp9s7yKqpQ8Mu41PQtMcbxE0Jk8EaR4PY0zGD3NfIc81OeDPS9ADD2ZC0M8frBKvWghTrzc3na8'
    '3CDtPNFXgLxifka9V29pPcGAw7w61Ey9CxHivI+Fijzh2oK7yxF1vWQ6mjylpPE8YGFrPdFwcb1j'
    '94K8gWetvBvzabxrlZ26wxlLPdlZ2DtwC4I9TCNoPNhkAr0oqOu7sqAiPf4DED0yE6u8T3c4O/WZ'
    '97x1cz69P7eHvBorBz3kzOk6OfvKPLoEWb0POiu8S7g1vRI2u7uA2DQ9Mq9fvHqW7LxXYmC9ofxO'
    'vH8VwLwaB2y94Wb7PNR4qjoY2NC7xz05PRNMZ73ABUA8eOM1PYQ5HDwP3SE9f+aovEgELb3CK389'
    '+QBZvXLcjry+0Gc93tCgvJAUQDwJ7Xu8plo9PDL3u7z+dkw7A1KgPQ7gfL3JqQU9enCIvG5glL1Q'
    'peQ8kDZvvQPkIz2Xf7u8LaSkvAPioL1XOOg8szyCvb3f9DzYYc4819UMvSlaJz3mZz292bYiPOq0'
    'kLzlRjE9hOZkvVosw7tdc2i81mXZvLKWGDyb/tu8NuRQPExfLTzc+rO8YUuAPDYfST3zpBs9KwOo'
    'PFDJnbxq2Du9URIRPbUxUzxXCCA6i3jiPEEzpbxSMk+9iFYuvUnTR72PySU9u38EPbSEsLzVUi68'
    'XY2MvOurpLy4jj29idt3OzjGJbxayxq9nQVuPOkDtbybiXE8Vg7rOWBqizyXkUs9aj1wvWbY47yq'
    'BVk9T+eXO/nRC7yEnTU9QPtaPUY6wzziYy291tFpPXCBUb06nXO9sGhIPBCtnbxK0Wu9qz02PfYc'
    'UDw5PRu9CI6QPFUVNz1/7au7blsUPfMkpTyYUBY8OBIgvUChUL1UI4W71yzWvMv8TL3ChGe9QtIP'
    'vf5gSDrbibq841VBvaDhIbqSZrK8ZaEbvD8/Gj2Fssw80wjzPBEOhz2CuNo8uhwTvRtuXLzs1E88'
    'PZi6vLrJQz1vlsM8zxhYvWaugjz2RF06TRKhPK3sprxEOIY9BwRcvBWUDj016+G8F/BHPLnOjbzz'
    'OQO9lUdFvd9nFj2iaHI7G1+1PITkBL3HuiM81bNAvQROdzwoeHQ9DJV6vQieSjwJY2O9suOsum83'
    '7LvdJM88Tk8hvY8eED05DAI9N60pvSW6FL0vCPw8PA49vQkGGz2M8hG9HtY4vWlSuDyA1928vTIW'
    'vWMzxjxHCX09HzkWPaX0aT3gTes8l0elvBjzwbtq8j09FHM+vdLIUT0SruM8Wr+pPC9mtjyvfj29'
    '+KQ0PIYjg7vIXWA7fgUcvSwO/DxS97y8nleevMJcTb3Esks9hIBNvQr3KD0H19881Sk1O05/JL0S'
    'jau8aqPvu8oHJL07Aji8Xq8qvG4gg7xHqTQ9nldhPf2HzrwDbw49bxSBOzXDQj2v6li9IOCsvLJM'
    'WLwj+F89/5ZTPZU+ubpwLym9sBMRvZPT/juk1Xo83YDxvN5Rpzwfa9g6DNhHPRApGryCPfK8vapv'
    'PHBxPLtxSgY9Q7VRPVgNHr2AdUY9FFbZPP20YT3j+Bg92pXbu1H/jTwi3mY7LhQ9vaOclzt1Tgo9'
    '3PYAPGiIfTyE2zM8qg7HO6tbLr1jIP67OF0rvFaQsLx/oiG9p72du/dkQ7rq9pe8NUNkvOuoe7x7'
    'bou77PgxvcgDRz2fQAG9jbsBPT9cQr1BU3y6gxaDu+KCLL1VGPi5Ss0PvcxAjTzs5Co9s1ZdvFUc'
    'TDy7KSy93chrPKB9fL3wpV09mX8dvA6PpTwzeig92E5EvQJMx7wmF2Y8/EVWvXVNrjxD3wG86Epa'
    'vZMMr7zxTF+9ty4VO/HHMT2Awnw9ntNtvSzZyjysRjm9HhG0PJCcGj2reVu9ywErvY03RL23vYm7'
    'r0S8PHxaJL1K5ZO7t/PAvDeAQj15qm29XU0Svct0PbqqYUi9UbcqPSPjWb1HHqC8B4kYvZBdRDxq'
    '+Q+9+Hj2PPXRT73iGKy8JgjDvIm4Yr1FGWS9yfmhPK3/oD3mYAW894sXvVvOiD0YpYU9LE8ivf7n'
    '+7zPtS09SQbDPOfJYb1WQ7y7pIclvautHL1xZ/e8lR+JPWXWobwbyFw9Spc6vSY5Wjw2k1w9MyFL'
    'vIQQaLmqit28bskXvfP4DD08uBK985EdvZ/odz2XMRc9OzA4vfq7qDws+j07Cs5LPQWwhr3vBHi9'
    'Uc/xPPlDmbpDoKm8K04aPe5Dszybujm9C6AAPa3CYDzDi5y8OxVTvUurTzwNTww99Ac1vc3uYb24'
    '5zi8Nj88vcjOkrw4o1U9RsE1PKr2JD1kKBs8BDVsPQQxFT29ocY8poQ2PcF38Lx4WxM9QMfruyaP'
    'QD2YKT2908ZBvT2rLT2QLgy9vMkRvHmJLz046xs7+IUavRGFWb1ohRi9cxw+vc/h8zyMCu87spUA'
    'vHH4hzyPeOA81OCLOw3rJj1wJgK8JvLYPFSyxzz25dw7hKIsPaV0Uz1VSCE9UNwrvAJRAr2eKHE9'
    'BGaWvJiPD7w4syy9QoquvDJMaLyXalI9N3jWu1WYv7xXWd08W2zTu4+5HTyM2RK90ARzvOsVAT0v'
    'PXM82m9Nvd3RM73uqTg8DlyvvN2FHrz4QfS8AvQevepvCb3rWu88midrvSZVKTy1mlu9Lp9dvIBX'
    'Br2+tAQ9cc1mvdppE72TiYK9ndzlvMENtTz95j09WpM1Pam93LzvHhi9LHK0u0+PCj0yiBc9NSKm'
    'PJkE2zsDi2i9NtQtvRYQUD2IGWy9onZbvXP3Or3ZMZ+8FLjLPJyoqjxt3gE99zH/PIdKSjx5no68'
    'LHsQvWO+qLwYlJM8uuhUPQvFxbyU5J68s7JDu07iBb3xzbC8Ko3jPNq3Z71WBz689XRYvLedDjw4'
    'sLY6Cm2FvCMGUbxHAaW894vbPHKMfDwP1y28qtUzPX4LCDynhfo8sHHSvBD4Jb1g5YC99crgvNXK'
    'Fz0+bxQ9uZbsPOsbQTxBzx+9JwUQvPrqwjqSuEG98s9EvRtbNL0UhXw83o62u9ASJ7yukUA9LuIX'
    'vctzvryfs9m7giUgPfX65juIqiS9VnMlPUqfaL2iLfA8nGJWPDOGAz12xVE9rKfOPGrjgr0kEKI8'
    'xG+FPcso87y/YTg9MshcPbIPuLxHZka7bbwfO7B+bj0uceY87/VAPZZwSb1NL768C9M9vZUsrzzL'
    'uq+8sDGTPBONGz2zv0K7N95WPMO+Hr04UVe7p34CPMQ3PT1njs08QoOsPCnn3DzsTRs9lDOvO1Qs'
    'aD3k3hq9+pC8PG3oqrwHeyU9ukCQvfwAAT0hmWi9hGGYvHqs6LyT3ti8frHCvDq5Kr051ja9psZb'
    'vJygZj3CUE28RmkNvVojJbxL0la9aXc0u+wuUD3ibGO9jF/RPA8YaD2hVTQ9yIjnvKGbkjw5ByE9'
    'TI1FvSiEVD30jJg7z0JTvcVXt7zOUXq8WU39vMmnPb0Dc5G8Yl87u+sxrDzh3IG9//90vCQJWjzg'
    'HzA9sQv6PKm70jyXxaY8Jia1uwlNHr2NFGm8A/XoPDGWNLzcUVo9mKGhvFKgXj37ChM9t88APfv+'
    'cD3Gs8W82zYUPXwLdr395vC8VktHu5AZTj1jHHa8vldMvVz/ab26N7u60ZCTO28WszyS3rG8c1Qa'
    'PN4dVjyEQZ+8/ivfPPLq4byMZBO9+YapPKlI77zVlEq8gsVKvHgEBT3ST3s9Rzw8PUpQsjxYmx29'
    'IN7dvM7HFb12K4C8h6ADvVex7jz7TYq9kcUrPKCRAz2LR1G9jfEFu1PkwDyFteG8Bo+ovEhsAj1n'
    'lz69QgN4Pat9MD1NboU6o2ydPF06AL0431g89gAbvaR9Xrx4S0Q9LvecPMpSPT1LBhw9/cZgPRgv'
    'lzzTri29JKNCvRqSGD33aUs9ujuBPdjuCb0lf3Y9rdVmvOFcJTxv7yg9VlkDvTsodL2BFhI9d6hQ'
    'vSkFtTx8Eg09q0YhPXgILrv/oX688bs2PDghq7syL1u9zR03PIDPCL3ERj09ZPS3vOmsmbzSHka9'
    'M6CWPV5n/zymW3I9SKsHPRTAJL1kr0c9tEyEugKwPj1yYgs8Ec08PTiTzbzhtt08uVrLOxnqUL1C'
    'HhI9/Zw9vWxNUL0pvh896sNRvT11nzz3iko9b6kGvDilZL3m0zq91cteO1uOuLy5/lU9s1pyPTyU'
    'GL0H6za8xBMlvS/ylDvrOzQ9oF98vOeMwTq44AQ2IeoevQZyWj1gYWM8MRmJPY/WaL0NJsq7W+3Q'
    'u9GA3DwvXpm8ZP4hvUTHYD04ZjY9xytwPWgAKT2DyyY9R/ESvS3hibtfRta82r8RPE4EXTxLdE08'
    'f1XVvFF9F7s5f6y8D8uiu5R/Qr1/Yxu927f0vBZiBb2rNrW67UZRvYMeRb1A7AM9nRxCvYe9KD2+'
    'fSA7tKLAOn6MGT1BJVm9j1/OPGUyuDxIIjY91F4zvcCfJjxn9Zm8oLM8vXrdKz0FsNu7qt49PSi7'
    'X7yO58y85rouPXW19rrAHwc9+EXLPIyONj1EQju9ppKJPNgDFT3/sRK9oP9Ove5xgT3tyTe96GEV'
    'vQO2P73v5sA7eN80PEilWL3ZmgA9n+CuPPo/HjyxYIS9njYVvdE1JL2x7dW8SqtpvZK6Gz1jZFC8'
    'Y0HyvGScOj0MXPM7WbBDPGAAJr2eP5e8wM1NvSZZCb0JaXM9eIgUvYNVFj0G+DE9VekcPcDcWT3l'
    'Img9WiL7PM1aqju9WGW8E8FdvTRaBD2UKy+8+MleOwqhQLyL5iM9kuB2u/4DYj0LQiU9pD+fPGrb'
    'Brws5OA8xc4BvdZFIT1c+k49cG03PLpdGLy9Fmw8gKdHPRqS4js0NOu8F5RlOSbhfb3r3Jy6V6de'
    'vTb/sLpm6wo8CiVWPU0yorxhqQu6FUgiPN9uoLzJ6168rhlnvMY5y7w/7jk8HZe0PP2EaTx4V2Q9'
    'wJiVu59bvTycUjS9poYuPUlkUj3JhtA8RMkcvYX+lLsSylq9ropQvSSEPD3/juY8ClcuPYPmJD2a'
    '2wS8HLcKPXxP8rx9NDW97oUSvH49Jr1LXmI9sC9evWXEYj2r2Ua8mWgUO+YHVD2AsdK8ogrIvBpd'
    'xTxZ68a7WKFePCC1JD31I6o773djPSp94rwForA8QOS5OoJES717zE+8cqk3PDl8Cb1CmSo9BzAg'
    'vZP0PD2hlIc8wgzhvLWOWD1wm5K8cDgRvP9MKLwGqnm94k4lPYvpprwAVAA9vHsPPXNlET2MLLs8'
    'lv6DPPXzPr36opW72QcMvQjtMT1lNpU88ZcOvLIJTL2jNnm98LFSvSngMD2tWfk8UiqIvUa1Zb07'
    'Tyg9cnCMPPIBMb3OoHG8R+DpPM3ZBz0Ah4G5tn8RvY2ud7zOQUS8AYUKvU2dHL1woyg9u5urOxza'
    '0jyj8hW9L6Y1PHoNST1aTRY9QmslvDp96LymW2Q97TR9vBtaELwX4W87mGkpvP1C2bsq1r28BrUP'
    'vREJcLudxrI8FN6MvYeqQTyIagG7M7RUvdd5Gz1S+iA9PYIoPDmJubxQwIo8uIUSPU3+f7lnKa+8'
    'Iy2sO+N2Zb05ixs9ze90u78kvLyKk9Q7SxLhvP3ldru34yO91mg+PJMfBT2t4bC8yj4CPdtWOT2B'
    '9my9NwlVvUX/ajuJFQC9ZK3rPCtwRT1UeOe8e5oePHy20DrnQ0M9970UvfcplTxFE2+9X7xPvZqL'
    'Lz1A4fQ8VV96ve21Mj29CUe9LMlTPec3KzwQmvO8aT9pvDG5jbwyCjW6d6jfvH5EEj1czhi9OzUb'
    'u+ADQr0mEea8jvadOyH19TxOR3q9J5WePD0C6DwC1xa9Jia0uWwe27zLvXu9YFzFvG75Ij26X7G6'
    'B/MTvX/lSz1slu+7aCRIvSc0wjzYzza9G0BEPULZCTzrVpU8RkN9vaVxqDvVOYq82qy/vPUDIzre'
    'gpC9WBhxvb1OYr3giv68eH5uPNPPujw0njQ85ZfpvOkeQzw15SM8w9MfvfwbVb3FRw28Xbu5vO/t'
    'Sz3TlrQ8MVwRPN0oLz3/BzO9riDcvH1dUr3HDpW8W/IxvSutP7thyQC9Z4QsPKC+bL0n+xC9joAC'
    'PeaRTj3RSFM8VLiuvC19Yr3kxmc9zHprO1CKHTxuaGA82CUTvFOVoDwaMpy803ikuzUT5jzfgqu8'
    '9qfNPHwD+jvrAYi9701jvDq82bsyXyW9zqcYveywJT25uA89wvxWPW51HT100rs79XMMvXDywTys'
    'F+m8R5tFPXS/Pj3fwzk92LYvPUD/m7xv/a+8Q1OhvP9bOL1nHos9OW/gvLTGgzufb1y8/5SoPHtr'
    'Lb07kyw7E8NHPQygx7zZnoE84x+fPN18OT2NQoO9qTgfvY7ZQ73e3US9N78ZvZv5ijxd7yc9vFhy'
    'O10HPLvDwQu8/ytfvdpJaT0A08886i83vXpzYTzvGDA9y4YQPVTOeT1+VGY9S0wmPYGbBrxiIC48'
    '0k1/PHSiJb3Mk+U8WXjHvBcBgT1YtiY9A9lQPVhvZD1wZGc9o+RJvSQ4Ob0NtES9kD5mPTeb6ryQ'
    'ExM9BJqfPOt4Lz1Zjw+988WVO6J4V70E82a97L99vWtjvTwq7sm8AEGsOl8kyDzdS/g8qaR+ucNJ'
    '5rzk3Aw9rW6GPdl+Jr3MuMW82N9YvZcPlLxdqQQ8TeNIvfuzJb1munw84hkwPQ5YuLwSvtM8cQph'
    'vd0ECr1IgB89Z5yVvBmVzTzCggG9WO4FPY7oGbxRw5u6XannPFAKojvKljs9ETv1u+jeqbyixSo9'
    'jYeDu1LXR71zWFa9b8DyPMHY6TyJ89S8F8gfPRZ9vTwjVow85cBmO9iq0TylESi8b05uvYZyUbxj'
    '8gQ88pwBvWrLbjyqqE09rnIaPONS4zxszMK7tmEuvGZ4wrwTNUQ94m1kvbn/xjx3VbE8UN1SPa2I'
    '47sQqvm8EjFdParpIb2IuJy6l68jvZNJHj2YW/08rEnFvHWOHbyb/Am8wM5UvYt5bj3LGck8lNwK'
    'upBTSD11GcA8faybPIoXdzzHKsC8qq1WvWGHdDy5N667OCraPCw0Mr1+Iyq7eco4PemZDzz/XzS9'
    'eR5xPL6Cgj3HBiS8bfiKOrt9MD3CUis9wJKOPKmWkTyqflE99cZiPVqfNr3y5Tc9aLE5PWrjBT2y'
    'b5o8kzOGPNRZNj17HiU7wgxRPW4/Kry/nQk9su5HPZCkozzdWEM9ZOlPPUv6tjy5qJu8BRAQvYz4'
    '9jzJPxA946o8vRkoKjsAIqo76iwGvAjPAz3AN2k9urc8vOUz2zyfpzQ9oluuugTkBT1afmc9p/m1'
    'O2dtUr29tii9LcNOPReZLb30xSc9LoQjPHgFaL02nVQ9R3Y7PcAas7xEZGS8RYYcvZF6EL1dX5o8'
    'U9QguwkwDL3xdAe8/uCqPH+AEzzZLy88HAcdva8MiLys1qE8L08xPcKDMTyJBTy9kPDYPMGBLr0B'
    '+nu9I/NDPbCTEz2IvA+8DEgBPb0W3bynVHE8QDmFvT+h7Lw6V3G870S/OyQVtjyZY/I6oD0Vvci2'
    'TT1q8zY9Y1vNvK0TCzyqtIE9da5AvEEnY72rzWI9NmYFPYPiBD0Dfz47vfxbvEtpXLxb3OY8XNaC'
    'vL2XEL2L1FW8VVyCPPZseL1a2BG8alPPPEyqJL07EDo9geMXPOhbHz3eyRY9NgiPu5W4rzzh+yG9'
    'SeVcPZgm5Lw5l2O8YZnbvKCRPr3HYum8azD2PMx+orz/MFc91dK/vHUueb3Y8w+9fDgSvH4sN73r'
    'LRu9Yt22PDSYpzwn/Ci6bYMyvftiWL2T6qq7gHlcvRLPbjxK55W7/S0APMpbfjuPJR896UXDvEOB'
    'Wr3UiqC85KniPIE1Pb3dNsM7TQBbPbgnDT2gwa08YhefvLvnFTyPcoW7DPAgPQrG6jzMCxE8WzEc'
    'PAC0qDwIWV08kAkqvHW2O73ctMI7PamgO+hKFT2Cr9E8q61CvV53Fbx4mHu94SMsPcc8BjyUFiq9'
    'ULUaPYRcGz0lzzq75E+wPBgOhDs+fui7sFn2PH4nhbs+5q68DI3qPCtBZbwqp/I8mONnPVaj6Lw5'
    'YI48g6mCPE94gDxfQ9O7jHamOxS5lDwzdyq9OuPVPAAyhL2pZru7VDXUPJGQ1LxvdrS8QMFgPVUN'
    'dzyfEM+7gwaxOzmFCr10y6k7gVlsPVJP+Dzawmk9AegcPfgOB7vbV8w8BDwAPW/aO7uraHs8aUp8'
    'vVEDbDxMOKY7UZBhvXH1izyH3ZQ8e0+fu/hwML0IBXU9j35GvQkVLD3LMQQ8BP2CvNTLFL1HT289'
    '8TdfPc6/Kb0bM6O7Gc5NvVw8/byxDHK7Z55QPRORHjm4UHc9zoQ0vdrTK7xV3Lk6Opr6vPsNybuT'
    'A2K7naYrPb/RgLxHSIe8+FjzPDTvj7wEhPs8YtMaPe1dEL1gInk7bRWavMVOLz2hLkW8w/VuvNfK'
    'HD32I2I8w7AcPa//q7xaLju9zJ5YPJ2jUj24+Tw9ORHGvGZfUT0ilA29x0FRPa1hOLs5Wwe9ZqMX'
    'PGeqOz3MeCE9/E/PvByhDjzUoC48XtRTvESWdLrx5Fk95mOOvKykID3TU1i9eCTPvIHgPzwKy9W8'
    'Ju8uvdeBVbwyffK8dSyrPL391Dwt7AG9PMAEPaWiFD0toGU9dFQPvRchTb2LNUO73rHAPMSenjwq'
    'zGO9ur8wPU5FYD2wlmo6MyoUPDTsYTsb0Ei9P+yFvaFS+zwGHhE9qFqEN2RcQb12WDs5NsINvX+o'
    'Ez1GRXg9wKNwPY24RD1AIrw8TTScPfdEurxXDf27prApvbygmjydLRM8w1+5PIFqFz0rjSg92oiI'
    'vauGXLx+fji9aIhFPbVY+7oWV0O9XtYEPdzqCb2XJm07MVUiPUtjDD2JdII8Vo58ugrru7vV3667'
    'ni2Dvd99orwYzy89zjwePcLGT71jKiE8lgFKPX7vJL0ISea7CpAFvT3cXTyrRse8fFpzPW2nBj2X'
    'gEa8sl6MvQiP77znSyS8BYMcPI4gr7z2iTW95JkDvXx8Sr1fw9A7i0BlvQePg7z4I4C9/lGyvM79'
    'xLsGxBu9JP2vvORaTTy1uNS8v1OiO/zTyjzzxL26T5lOvZldhLxGfX47LQ+hPMOuEL1C9gs9ZSyN'
    'PI/xjjuMc8w4Up8rvKoPoTxM5Qo9vESqvPtqxjxe2zc9K3rFPBBTVDbojpY8QYrxOgf9IT1wS2a8'
    'a6iSPKUsjzvAfH+96metPI8kPb1iQqs8y9rwuxMMtTwJjzy9FNNXPeG9NT3GRJo8NzXXPC6u4LzG'
    'CDs94Y+qPMB4Lr1+eYK7TKGpPOMPqDx2yl088fQ2PYwEkbsfEgq7u4LXuzh2GD3TU0C9AMA4PR9j'
    'g7xCovk82rwevUqjf7wQnyC7b4cDvHKWAr2pA1e9HoxQvfrtPT1MGNw8gMWxvGHUMj3JEBe82SNA'
    'PB/R7byHjS+9OF9KPbJbZz3Cm3i9X8A3PHqgCjxtzS29xhDWvLOPXL2hZE88U+icPSLEvTzUEmC7'
    '1luJPDCsMb3yPAG9mm6KPEFiTD235EQ6QgV9vPxJKDyvH1+9QAoTPf5tFL2GYzS8GuNHvRBZkTy+'
    'Wse8GwRcOz8kG73qkNo80XkYvdoRYb0RyX69+FsjPToR5Tyqkg29rLmNvAZfMz345X88KUxPPWWP'
    'KrzA6RO8FeUAPfffkbxNM4i94RopvZ3AJD1Ej8g8YbVTvE6Cdr36xUs965mFvKcFpbyAkVQ9B7Qi'
    'PbismDwZ5588vgbSvAtTbjs2cAa9D+mNPKL70Dy9tvc80X7VvAidkLsF/EA7rLUWvc9TMzyxbBo9'
    '5hEKO4/M5ryXElE9ZvsBva8ExTyr+Ke6BGK8POABzDwiI2W95gtsPJmfj7vGv0e9Oun4PF5EkjoG'
    'yb68kwKauzbsCz0jkGO8qZYavYwVmrw2GpS8yk7FPB4CVrzmhPg8QxpivZQPdbs9rEo9SsuMvLSj'
    'OT17lao81u1svS0l17x8Mk89coDgu4PHdT0WIFe9FfLxvBzMTT1xCms9i9YwvWzTYzzQuWW9ZxEC'
    'vfnI8Tx2pNK8ReV2OySWjLxoIHe9YtI0vYIAtTzkgvA8G3Hku/BmPT3TRUO9ImptvEaXYL1L1Fg9'
    'bHMIPCohuDrhq+K8LV0hvUtJXz2vbuQ8kJyzuzzIvzpBUa08DemtvANfcj3fpp466ASIu1sKDj0U'
    'X1e931k/PbGMC712Ni699+XHuyRV5rxWs0498MbQO1VvLb3x6DK92uzpvNkagr3xII+7DT5NO1c9'
    'Yr17Gai7V3oqvC0yGLy7m2W9VDISPdH/mbxgSG66qOMhPORAfDw66L88rM89vGS7jrx7/oi8+88y'
    'vcwoBDx6Xei8SJViPVs0aD09JFE8ynCdvILdjrwe2iw9YPOEvGnYDT3WIBq9fQ0MuzRUiDyqiSQ9'
    'mZHou9KuXbvCEF68bscFvULBaT2rYuK8oAiZvF3cJD3QPGI9De1hvPsUxbz/7xe9+wvCPOxTIrx2'
    'tYy8M7RVve/7vbzYyZ48XUWRvURkgTzvTDi9Qz1JPRhSO70AM1Q9LHBBvDWgCT2nc0K7VdbPvALs'
    'U7wpOrw8Xi0BPauyQz1T9eo6o0qbPDBGKz3bBLK8nb2NvO631jyBEfy7GwmLPHzP2jyJXNu7g2Np'
    'vJm1ZD3dnXI8TkJwvRlJYr16OpG8FGLhvBouMD22vRo9d70rPfTEYj3ycVo9iL1nPbxUPjqd+lU9'
    'V1euPGs41DwqPKK8ij+1PLkEbr2OsoQ7QcAjvRJ9yrrQiTE8ky8NvN1KYT2I81m9IyVqvXMzQjur'
    'eyu91DBKvQaoIb1jwUc9Rx4avYOjZ72O/zI9cx8tvWG4p7sEAP68qXmcPH6zTz3/6gS9FXkZvTyh'
    '3TwkYgi9nUHkPA0BKz2dc0c9guMDPK+4Zz17nja8NP8RPWDCuLzKLL+8EXe3PAFUDD0qHOI88xuF'
    'PDMJ6zr70JM8cWQSPSYSJL0aC3S8UGAsPZVdWbwwGz89WQnKPKyDs7yQbC895OqvvOf1Bj01LSk9'
    '1iksPSN4Mj3GRlw815ZMvV88Tz2nqig86IvSvPsgJrxN2x29GHnRPLnaUL2fMQc9mxZfvbhn37z9'
    '+yw9VMTLuymiQD2N1Eo9slIRvO2SGLuklWK9hYOHvOo0SjzqB6q8nmFZPeqqxTwPmAY9Xa05PeO9'
    'wjtdfv6704Z3vJpZZz3feRQ9AWYwvZuFgTznSL68ntxEPYbTIL3JnGS9Zr9RPAfmLb0zIx09ZHaV'
    'PWwVLb3yTAO8sT4Wu2N5Ur1NjzO99SFgPZFlIT1TMN68txcYPLt+qrsh54w6Uo8IPOSDEjxEtfu8'
    'f7FgPQr0lbzFwVM9MaPnvDqjibwEOgY97apYvW2CkDy16PY81G4KPKdTKr0u5L+8E+voPOjUhL16'
    'MTe9TRInPScDfbzGvmC94nb2vAPxtTsFJr+8lXcKvUfpB71qfF89EqAwPYr/KL3CuAM92VVZPcSX'
    '+rzYoqs810eFvVX6Cz2X1Aa9B4suvcppLL0wZ2y958ZSPebKVL06RUS9hSuCPC79CD3zEQm9jsn8'
    'PGgZnbs9SDu9yuFtPE624jwB2nC9RCBlvcqkKbztWD+9d4gmvSaRXzx1Jw+8F5/FPOgN9Ly7yu28'
    'hPqCvQa7+zzBTGy914/KOz/7TT1//p48g9zgPJDraz2E/Bg6nGgAvUn5SzvS2ia9WnsmvTO8+DwN'
    'Nhs9KTG7PFNWR70OwC29WFlUPWNK+TsvjsS814yBPFr5Cj2MgjA9nnw8PVkvo7yhD0c951sgvTRd'
    '0zw66li9M1eHuxC7Oz0IEl49MS6OPN+11ztMUuE89sttuopTOz3zt6k8fYZpPUoMFrwv9Cs9BSYO'
    'PU8BDT0Q6yS9kwaAPK4t67x44GC9cAo7PWgUhb1tAwU9MjfAvJcXg7xwTzq9nc1fvGiJezxPUBW9'
    'IVZrPXvWhTxRRzs9jOEnvcbXAr1wQcm6Ol7cvBFVQ72aq+q8zlVEvQGWaj279f082N41veuUs7w2'
    'smw9qHwbPculAzxVbZe8PPMuPeGJ0TtdvFa8wYUbPFG4cD2yD049jnA1vS0TuDsost67n5kFPaOx'
    'rrwKCQa9Uk4rPRav+DxY1Os72oMxPWITEb0YZDM93UNgvZcR8jylmxA9lLAZPeTF5TyzqTo9ok5T'
    'PSv3QL2UDFw9IIi3vCMQnTzV30e98y9qvUPVTT2hkmq9dmXdPFEoID0uLCA900n6vCwoiTyYERc9'
    '43RNvdqm0LrOIGa9WIaAvRAz4byueQ+8rRJQPT25S7qmUMe8ckdYPOOjEz03fnw9fHpMPD7WHz3z'
    'TAe8B6+mPUjaZb1mEhQ9PqC2vESOA73zIxa9gcK+PGpzOj36Q+48d380PbaM/7zAR0y9HtGau2YA'
    'trz5C7U8sf+wPLOYQLy5FU+9jZXIPIWN2byzQO87L86EvKGIUj2Q4AE99ai6vOSXnjsJ3ai8pGz1'
    'u6UXBT0n4vI6q4HluwV8LD2MXiQ8dC4RPRapJ71VwD89UiwVPeX3G70Cqk+9Hey9ujLLV715cZI6'
    'cXkXPanZRr2ze4C9a8D2PGMd8LzuxA08nVUBvAMlED1e3kM9J5gRPQ5wUr2rTQu95oxmPRUNJT0C'
    'vRu9F7VQvY+qKT3ncZi8HGM6PaQDDL3felQ9f7L1PJI0mbxSJAm87FRoPRBr1jzrjwi9OmLBPDtY'
    'Xr0n/1K8hXPLvM7ReL30sSk9ioQePfN/7zwqa+G7FHGbvAicBb0pXTA9Ek5XPc+kVT2oAC68Nvxt'
    'POEgCbzBJ8M8K6IvO7BIQ72MoDA9oBEcvbj7D7xt7y29h7+AvDpigz1G+SM9CBxOPUJvYr0qsic9'
    'kLQTvSyCNbzJJU49Dwe6O5s91Dw08qk8FNIkvS2wxrwvP/26y337O6HpArx4LCm97AUnPeoYObqz'
    'hp48t2/KPDkIQL1ze088+YN7vWEqmDw0Kjs9Sxa9Ow74yDwO5U48Og5xvGTaV7kjJQm7jARKPVfY'
    'ET3Kvkg9uXP8vBJPDL1HmzW9CUEdPQwG4rwqNAk9tnD2u5KPXr1Ekp28sInGO+5ioDxQ7Xa9dsVQ'
    'vfbtBT1sTyG8YfRMvPlAP71km+o8wRJIPbI/r7xdhEU9CPNivRqgQ71B1bU84zAGvdZtvTxBFwW9'
    'ZPBIvSgBJTyqq+w8kyDrvEnELLyv3CE8DNRhPfIYWDxx/4K7z/TxvPBwNbzwbr88igZRPRHp+Lxn'
    'ECY9+kYwvc4FjzxOFVm95Oh1PVPIOz29NBg9Vo/qvLnvwDzxAsa7vej4PCMKwjxyMCQ9AyoMuSb9'
    '5Tw1yf+8C9rRPInjrTjgUWs7X98fvRAiJD2X10O9PEUGve3QWj02Mms9/acOPWY577xhn049uRpq'
    'vOr9Qrx9bko9vohxOv9S6rsPAes8uglyPYx+Mz1wSje93FfUOwOrWzsbiDU8gsmIPDy6kTx5Mgq9'
    'jRDQu88+UDxGNSu9jaXnPE4sxLwahMW7T5I4vIhoBT2ITyC9EXSkvGo0KT3tjQM97P5aPf6Sizzp'
    'PaQ734IBvYHjUD0wk2Y93xI/PWgkuDyCiQu9xJIDvQ5mabzOWdE8pQPDOvRnPDvwNMM8C48SPbT8'
    'GToLWci8eEBqPYbmCL3il1E9NDRmvUOiPb10PxC97FkCPT3itDyGdGk7Q0ZZPGPRMLmjvhi9iEeb'
    'vLiZljwYIL+86CeLPZJ2lTmVyEe7YQekvOAHAD2UKV491YNaPTEJTr3oBxy6OQwuvGToZ7wEwGa9'
    'D0KAPVcMFj2viCC9FtEhPUqXorxIzD091CWSuxB/zjyQ1Ly832vZvDxEi704fEq9TcdlPUNtXz2+'
    'Uog9aOtePapQKb20tQC9t6ezPFp1GL13ppG7J0cbPPf7qbyAdvS7tFrTOvr9o7xmgOc7DVpFPPJD'
    'ab33FHg9d3tuPStWVL3Ip3K7LY9OuntWGjvSg1e8B1G6OzOQdjwmjLE8JyD4PPIDfTyVOBU979er'
    'O1fSO7yI9v489PldPb4eijog72G96Xu+PEVHL70RJ2Q8RGZOPS++H72EolO8h4RcPZveMjz02VM9'
    'PAXavEk5/Lt4KjS9LILGvCA9IzyoxS89+fpPPWAnl7wWzvc8KesqvRi6T73Papw7k9F6PQKMJz04'
    'oTG9WkPvu+nvST3gfgo9bQUIPDTPAr3ECjM999dmPd7oJT03GgQ9bsGWvL37R71zYDg9egk/PNXz'
    'mjutT9A8D/uTvGqxHL3xo9488pr2u+xS87ySlaa875UlvHjSW71Xq129Kwd/vKSoHb1o5N88Cq5x'
    'vb8tFr1jgym9i8+2vH2yh71YkoO8yXMRvcazAz2rxFK9SohnPWAxcj0vv3k8iBpZvHHbFL2LH+O8'
    'nT8bvbpnOTzQG1o94A73vI5hnLxMrhA8hvuFvXVBp7xATE29x1lgvdG/qrz0VRw93O13PCZh1jsK'
    'Y0E9qYP3PCX9tjsNrJQ7RadWvJXpsTxyek48Gh6WPOCnML3WKI27tZm9PAUzrjzKa2w86I8CvZlJ'
    '7zzeL3m8SxQXvXq3LL3TMKQ6GOtevYgULz1P60A8LWmVvLs7Lz2Jwuq7Yw8MOxOeBz04efO4AZFv'
    'u9N4WD1WNz690rKqPH4zsTu64Ta8gzV7POzj/zu4fCK9x4RNvWevDb0K21M9i3QmPTHZzbtfI+U8'
    'n1EEPTpmNT2+l5w8Yn4sPYcJnryoLWo8HcgivTczUr0CZGm8/EZMvbo6aDyIoRI9uWwhPOQWg73Y'
    'FN48171QPUHsAjwTgmC9I8t+vH8tcD30b/e8yG1lvTp+PD344YA9IOv3upwATT0K/Wi9W2EJPdXp'
    'Yb0o6r28/IoqPVOgCr0t3Tu91eCHO2vMXj12/Nq8aGl4vQZhBT2OcDY9letSvW9I77wNzbE8PzEB'
    'O08EOb0uAie8KwgWPFk0ALty1DQ9XACrPPuO+TtVf1A9SqF6vRMzKTwb+qm7STUbvbI+XrydD9G8'
    '+Fh/vZmdxzib+Xs9pnopvQPW47uq+nK8y3cUvfZk/jzEH3C9LtVmPQAIJL2sFCK8G/o9va5/Ib1r'
    'ajA91ZcFvCNfoTwCkcu87JFnutdl1Ly4J009CoOLvT3pnLr2LUY9s+EvvQ3+VbzBv4Y7uhRoPS2y'
    'ejtkWEq9tydlveNV8jsznjE9WJxovc3RQb3vCkS9r7tnvcnbNzxpGds8LppvvdF+oLzmYdk7H/xJ'
    'PBViojz2fF68E9a7vGPsG73di4m9HVptuy3WejzauBI86j84PXJIgD1BA4k8NMYfvWmwrjzpa9o8'
    'X2tovD5VTr2WrmC9Y+ouPaf/Qr2Ermw9sd9RvewJ5Lwsmy47h/BTPT8+sTsCRjC9fmsnPduizzsm'
    'uiu9OFdvPbboMz2yRQO8xWjNvAc72Twrq3A5RgygvFdngDyG43A99KsyvPGvCD2njfu8rgVqvXky'
    'nDvPd1a7/QBDPMWEhL2zlE69qiauu/Z9TDx4JX69x5hXvQtPNb0tRgS94BUoPaSI8DzobEY9bqMz'
    'PbFt1Dz6Ki+9C31uvRPcNL22MGC92ZsqPcF27LwFLxg9hnLyvEFYH72AnAg83BWbvGzIhjuc7Qo8'
    'Nd6Dvaqldb0W6lY91k14PHtCiLxiutE8Y3MhvDfGKjzTOja93S8dvYKEwDyr0lY7fmhQvMWoj7yk'
    'ScA8JXY0PRSbMT1GWdm8iMbkvI6rg7wKPfK8RVKqvHWTI7w0sRk9Igs5vfj5VD0/ChY9NXs2PXX5'
    'DD3J9L089jBDvQfnCT1qmom91jg2POl1ej3evU87UuA1O35qPL2NtE297zK2O3v58DyIZ0G9rTLu'
    'u7LrDrsNBSu9rfNcu3w2S73Uumo8z0FCvc3TKj3AU189QmoDPVFvBL3qluo8EtZ/u1BLBwickCQF'
    'AJAAAACQAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB0ANQBiY193YXJtc3RhcnRfc21hbGxf'
    'Y3B1L2RhdGEvNUZCMQBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpa5xJevaWuPz3jG2A9UizOPNWlRr2eIpS8NA1Mun96Wz33xlK9j5UqvSDY9DwUxbc8aMaS'
    'vIjGQL17+Ei8n3U2PZMsWj2Bdym7f+EKvSnxiDwGxyI9tTgbvR35cjysfRQ9Oonyu3vyQ72wsgO9'
    '9/JnvfOuIzv2Ihk9j1FDuy+CHz1QSwcIIfMX9YAAAACAAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAA'
    'AAAAAAAdADUAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzZGQjEAWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWvvvfz/6AX8/YDZ+P3mkfT8KD38/q1OA'
    'P13/fz8AN38/scV9P4CHgT9BXH8/r9B/Pxbofz/b5n0/uyF/P9Y8fj+4R4A/smJ/P+YAfz/8Q4A/'
    'wAV+P7/cgT9z/IA/s9yAP8xJgD96P4A/mup/P+1vgD+/CH4/n957Pw1Cfj/L+X8/UEsHCILiEAmA'
    'AAAAgAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHQA1AGJjX3dhcm1zdGFydF9zbWFsbF9j'
    'cHUvZGF0YS83RkIxAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWloTTFw7/xohu3q5g7vIOUG8pXEVO6TxrLvFTEY75RDfOr+T8bq/HSo7XKAFu63Q3TozGwy8'
    'xH/muw3Lgru06dY6JdDUumGlnru9JGU7bMDhO4bd9zmP5+i6j02jO8y1HDxH0Ik4kB6Lu2IJLbt4'
    '9M+73Bx8u7DdJLwocja8NO7oOlBLBwh5Lk2+gAAAAIAAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAA'
    'AAAAAB0ANQBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvOEZCMQBaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpafHpqvYxIYT14Kr+8/9MmvWPgGz31ga28'
    'xthavEgixjximJi7CHw+vLDsrDxdTzI94s7LvJ/xMjrlyOa8WLhMO90x1DwIwRk97aSMuzaRD71a'
    'leo8Mo09vZ9Sbr3qUtu8J6ckvcBARL3OpLU8LbyMPCJl9DxSTb48gCsyPHvSNz3/jxU9mYAtvUxh'
    'eT303rI8Zl9LPZc4Jz3R34y8KUbgu+lNljxEu3a9L2fAvGkQY71O7G08vHuZO19kKj0oX1i9qdIq'
    'PYWVCj3PDDE5ieBlvcOSpDu7F2u8kmQCvVZ09bzi8qG8rzZ2vHJpB7pKj1i925SfPN7W57zBbUa9'
    'iGvfO/aHqboGCE29PlWGPXyIRb18UDI98V2yOzbBJb2KlDm9qlzuvIkEXD1OHjo9pyf3PFURaT0Z'
    'dOQ8aRrLPJUVQT1o1es81fVIul3HsTzpTXI709qovFom2Tx0Pxo9WXoQPQtvST1jN4g97e7pPMV6'
    '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'
    '8KbyvI3OSz2uv289eitIvW5lRj1a7Dw99ih0vRZYBTjHEAu9q9s1vXtI9rygnwm9eEtsvJ1cHz2p'
    'zP08VNXgvCFgMb29dak7zi3LvA1Ux7yB5GA8QM/BvPYo6zzqkZ88fXUjvT11Xrs5yzs975pVPTQm'
    'hz2NADo8FCsAPNnmiT00eQG93e1ovVMzFT3GnCO9ImvDuxjV+TzDheW8H7tVPGfnjry8dUk8z8MJ'
    'PelQoLtXAry8Pn08PZfyOLzvzFa8VwFQPeEbgjyN2Ao96ImtO6FhW73mB8e8jAa6Owx5XT3SK3O9'
    'uRU6veEtbbwQiU68ibOTvWYEKT3tBAA9BbvJvC7Rr7xW8P48zC2+vIntEj0Tw349IXKWvMw3/LxL'
    '3j29ReyYvLH/gTvCJV294MA5Pe72fL1Seag7xismPddW2Tsz7he93HFvPOIpKb2vxQm95i8lPfCH'
    'vDt7YQG88+4IPfl2QT3UneU86XQSvYqcKT2jqTQ9vQpgvIJwUj3ajA08MI9lvcbj6zshtYG9QZrL'
    'PG4WVT2twyq9pr/ePCYnaz0+SSq9+iEivCERND3+cUQ8aNKrPEIgprx6nEg9M581PM8zB72JmhS9'
    'b+lUvQahHL1mhQI8f/roPEddm7sCEFS90j3nvOZbEj0mTIW6h8m7vGIjPz0qXFg9F2PaPLM+4LyE'
    'IXa8vtoEPWDEPz0yC/g80p0vvUTLXb3zWaA8ma5JvAyEBrufenM9qqA2PK8ocr08Unk6B7ACvRUy'
    'lDzBuvC7fEncPJdMP73a+Fg8UEsHCJl60oQAkAAAAJAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAA'
    'AAAAHQA1AGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS85RkIxAFpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWloRhec8+TVRPc1dWr2W+5m878hPvG8xXb1T'
    'JTC9clNHvZ1qFDteT1C82kXIvKH5E72DZma9cMBLvJgSIL2PV0M9d2sDvdRzK73UaCm98NkzvTU5'
    'Zz3Hv8I8sjbePGq5mDvTeIq7mmHKPPvPU72q1wu9AiCsvK/Rw7zsp548Oh/MvFBLBwicRV8ggAAA'
    'AIAAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB4ANABiY193YXJtc3RhcnRfc21hbGxfY3B1'
    'L2RhdGEvMTBGQjAAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaQfl/P8MkgD8Wln8/1gmAP//JfD8MYIA/I1t/P+5sfz98vn8/dJCAP+w7gT8Kbn8/DV5/P89T'
    'gD/een8/RcqAPxkLgD//VYA/OEWAP2xffj/3zn8/t1yAP6/vfz+ypn0/q3R/Pxodfz++Sn4/iUuA'
    'PxfJfz9MbX8/4nWAPzMngD9QSwcI1CGveYAAAACAAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAA'
    'AAAeADQAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzExRkIwAFpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWtQ6oztJPsk76m3vOkVYMDwAdSK8I2Yeu5jD'
    'kjkyEhi7doLNOqIDTjsk1Is65XgYO5oooruScZi71KjcO5kWd7s2lJc6DfhDuxpAWzuE/xM8HsWW'
    'upBcFzvO6tQ5lxeJOtMOyDvSGqq7rhOnOmTe/TnNISE7Z/QXOwO0ojtGGpq7UEsHCMAtr3eAAAAA'
    'gAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHgA0AGJjX3dhcm1zdGFydF9zbWFsbF9jcHUv'
    'ZGF0YS8xMkZCMABaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'Wlr60BU92pYoO8Rqa73qfL27ZDmjOlVRPTzfvZ28cJd8PNfr+zsjs6u7ptkzveUKQL3UPES95bn6'
    'vB9BGj3fqDM9rAsMvA+aHb14dnU8eRJpvYoiRDzvQyM9fmjsvIlXQr29YA+9HjvjvPLKAj2a6E+9'
    'xscDPNDxOD2JzDu9eEzyvN15VD3LrMq8dVOgvG9mRj1+uYi8cLcEPGSmf7xL4Fs9ygvpPJJrejyt'
    'mTo9DDnTuqauM70vrvW8rKkJPP7NfDxveCQ9ZymSPNZEtTzmVcA8MRIZvXcPFb2x6Gy7Tv89vUCR'
    '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'
    'WnFpZz1FZWW9CxccvMz+Fb000Ra7760wPSdTI70YpEA9/+YLvcu/XT0rSkc9Dm+FvEWJI7wDmIK8'
    '12QLvVNMPrsCdi29sPUNvRhcgbw+9gQ9AxlqvWJmKT3sm8O8RdTXu1rTDz3tiR+9D6BPvXo0Czoi'
    'LRM9wNF0PBiXE70/4Vw9UEsHCOERGxqAAAAAgAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAA'
    'HgA0AGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8xNEZCMABaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWloFnX8/DpyCP6j5fz80SoE/a7qAP0LqgD9/4X4/'
    'NJZ9P7ikfD+pW4A/jKeCP8gufT9lKIE/zQ9+P+N5gD/pH4A/XiuAP4/lgT9ZWYI/BPN/P111fj8x'
    'AoA/oU+BP6lggj9ri4E/rOGCPxqQgT9NQoE/30eBPxtXgD8+gIE/IUiCP1BLBwgLnrT4gAAAAIAA'
    'AABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB4ANABiY193YXJtc3RhcnRfc21hbGxfY3B1L2Rh'
    'dGEvMTVGQjAAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'FdqOOv2opTsou7E7cLECPLGqljrdseY7/tAevDz+1Lu0+FS75cquu2fIxrtLt3S8auDOOk8xU7sT'
    'RmK6UcxDO8QZMTn2vPM7ygNWPDS4bbuwZFk7O/XNOzRr2zsT4Sg8bzwtPAK+czyv5mo8Pj31Oslq'
    'KjzNo6c7yHZJPEfGjzxQSwcI31NX4oAAAACAAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAe'
    'ADQAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzE2RkIwAFpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWqLLOz1ZzgK89b4OvbWRXD0uniU9MejrPJKlOT0L'
    'yhy9PuQ9vWHVOb0fvh69RhfHPI8Ocr3B3Qg93PQqvXrWP72hryO8mvEpvSL9jj2uVS29DbeGvCGN'
    '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'
    'PQHhWz3wdCA7rKjaPFBLBwjGC+D4AJAAAACQAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB4A'
    'NABiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMTdGQjAAWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpav7unu9dCCT2f9ac8nscfvchLaDt6rAA9KHEZPCwh'
    'fb1GdIA9NHVquuadcb2HiyE9YsmguwY3Mz0ezBE9WyyUvHv9Ujw/17W8B+U1Pbj8WT2QbWI96Y2Z'
    'PKq5ET2dltE8zXuuvNT91zyZPKA8Y7E8PeAe6rwOZgm9sAANve9gDLxQSwcIoXCN+IAAAACAAAAA'
    'UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAeADQAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRh'
    'LzE4RkIwAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWv7Y'
    'fj+zMH8/o7mAP/bFgD80mn8/F7CAP5n+gD+nhYA/6AaBP9YFez99a4A/162CP0wfgT9ItH4/dnOA'
    'P3yAfz/WDoI/bjiAP+jtfz83IYA/n81/PyMegj+vGn8/OgR9P8V6fz/mS34/8M5/P2JWfz9bkoA/'
    'czB+P8ytgD+6LIA/UEsHCHGMx2yAAAAAgAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHgA0'
    'AGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8xOUZCMABaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlrqPsI6Flx1u3DCHTzrxx08HSDNO5+y7Dqfy307ThEI'
    'PIG8oDuteU68Ba6EO7ViLzzpdLA7M1HcukzHgjtiDVy7wTUePFvtqTs64ke7gMoEPG4NgjttkSQ7'
    'gYcIvMrZOLwXrAw7TrjnOwVioDpNmBy8OWcrPPv8CryJAi87pfshOlBLBwjfd9kagAAAAIAAAABQ'
    'SwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB4ANABiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEv'
    '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'
    'PdpjFzwBTYk98GrZusKXHz0nxU093oudvMu/yLxE33g9OzzkPIdkczxXM4g7A3cWPR0/BL35nTW7'
    'zAHJPDX9frxtN2e8OUA3vcX16jzvAEY9r7tuvVuNLT2Y+4Q9OHpIPVlzBT2602A8ajGAvANqt7zu'
    'ihS8zH8sPe8TAT30B+28u7nVvOSJ1byy3Su7TWmCvBn/rztd8Ae9UEsHCHgyZf0AkAAAAJAAAFBL'
    'AwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHgA0AGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8y'
    'MUZCMABaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlrVdkw9'
    'dd02vWvyNT1vFii8d9wCvb9cjLyeDeW85A0JO4bPMz0nsFO9yaEjvJiNWb2ak2W9LpR3PUBCoLq/'
    'rj469X7AvHu2O73jZDM9P+wOvCB47brzSSq9UTPuvEo0Nb0lyyG88XiDPRMpKLyTFqG81kpuPXit'
    'xTz0uQa9YLDtPFBLBwhAry5zgAAAAIAAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB4ANABi'
    'Y193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMjJGQjAAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa4PKCPxgohD/roII/LCiCP/oqgT+J5oQ/MCiDP/5kgT9O'
    'NoM/H1OAP0ULgT/AaoM/czeCP3HVhD8ME4I/5h1/PyM7gT+34YI/cMODP7Gdgz9jBoQ/pvKAP98E'
    'gz8p2YQ/0DGEP2M5gj8BVII/BBmDP/Y/gz/6QII/SkqDP8GMgj9QSwcIa+s+A4AAAACAAAAAUEsD'
    'BAAACAgAAAAAAAAAAAAAAAAAAAAAAAAeADQAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzIz'
    'RkIwAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWmQrwDx5'
    'T1I8GQejOymXIjz+C0Y8cGzbPKg8vzyWx5Y7DD4/PK27rjsG6oU88qNjPNU2hTwErNA8q0eMPH+e'
    '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'
    'PGMubDzQ2608GL+KPJQogrxD+k09gUCoPOaCojxlJSM6qwDpPK+3OD3EU/o8U9VyPUAXvDw31Ry7'
    'qiozPY6psLt1bgA7NTI8PM3ZkDz+zhG9B1MHPdnakjzT5Vu9fQj6PBlWAr1VYK48NJ9PPKpaFj0D'
    'HfE8mgt8vAqIuLzBdiC9MpoFPbDoVT19eS08rDXgO4hR2buCkKe8AkDBvNhRmLzaqqq8VnuKPUG/'
    'Ib3B/VO85HHDvFxjgzgmv4E7BpIGvX+gFr0YYju9yaNYOyIxCD0A3UU9kmWrvJIeUDxdYaA8+QR8'
    'PNh8U7oMB4O8UIc4vVWwLL0cQyo8ChGnu6qrSb1wwx69c8kovf7chL2i+wE9tzxpvRcMHr2Mc908'
    'lNUdveiISL1QSwcIUJFfjgCQAAAAkAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAeADQAYmNf'
    'd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzI1RkIwAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWognibw0hp48PigGvRNS6DxTAYg8bhO/PACOjLpsWym9oefe'
    'vAu00ryBO4S9MTgAvSGa57uLWpw8jslCvYqRtDuw4NG8IlHmvEmmxLxDT4o8Z0URvfdDJb3pRLk7'
    'tBqlvKhRRLvvjz29QTc4PEeJxTw32No8O1zuvFMR2LzlBM68UEsHCDnA30aAAAAAgAAAAFBLAwQA'
    'AAgIAAAAAAAAAAAAAAAAAAAAAAAAHgA0AGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8yNkZC'
    'MABaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpkayg+wX9q'
    'vVUHlT2PbD29YKjWvYPxiT1XFiO+Q0IJvXDjGr5NChc+l3TZPcTQET5xLyA9xiMevfWIGLxANx6+'
    'rM2JvfCeFb7q7ss99MggvpVHJz6V6ZG9yEd0PV8YJT7viBe+2h1Auqv8Ib4wiT88AdZxPfHvAL4l'
    'pkU8MNVrvXHGwbzYbv29vn88vHbSVb0y7wo+Oav/vbJnFr7o3TA+TJq+PXXYKD4rABs+OCO9PJ98'
    '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'
    'NABiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMzFGQjAAWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaslhBPVBLBwhQFJpFBAAAAAQAAABQSwMEAAAICAAA'
    'AAAAAAAAAAAAAAAAAAAAAB4AMABiY193YXJtc3RhcnRfc21hbGxfY3B1L3ZlcnNpb25GQiwAWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlozClBLBwjRnmdVAgAAAAIA'
    'AABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAC0AIwBiY193YXJtc3RhcnRfc21hbGxfY3B1Ly5k'
    'YXRhL3NlcmlhbGl6YXRpb25faWRGQh8AWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWjEw'
    'MTk4NTcwNzU4NzA4MDQ0NzUxMTYwNDUxOTQ0MTAyODEyNjYxMDhQSwcIFj1IwigAAAAoAAAAUEsB'
    'AgAAAAAICAAAAAAAADxwVQWQCwAAkAsAAB8AAAAAAAAAAAAAAAAAAAAAAGJjX3dhcm1zdGFydF9z'
    'bWFsbF9jcHUvZGF0YS5wa2xQSwECAAAAAAgIAAAAAAAAt+/cgwEAAAABAAAAJgAAAAAAAAAAAAAA'
    'AAAgDAAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS8uZm9ybWF0X3ZlcnNpb25QSwECAAAAAAgIAAAA'
    'AAAAP3dx6QIAAAACAAAAKQAAAAAAAAAAAAAAAACRDAAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS8u'
    'c3RvcmFnZV9hbGlnbm1lbnRQSwECAAAAAAgIAAAAAAAAhT3jGQYAAAAGAAAAIAAAAAAAAAAAAAAA'
    'AAASDQAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9ieXRlb3JkZXJQSwECAAAAAAgIAAAAAAAADmgC'
    'CwA2AAAANgAAHQAAAAAAAAAAAAAAAACWDQAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzBQ'
    'SwECAAAAAAgIAAAAAAAAUfBQ7YAAAACAAAAAHQAAAAAAAAAAAAAAAAAQRAAAYmNfd2FybXN0YXJ0'
    'X3NtYWxsX2NwdS9kYXRhLzFQSwECAAAAAAgIAAAAAAAAC+X8qoAAAACAAAAAHQAAAAAAAAAAAAAA'
    'AAAQRQAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzJQSwECAAAAAAgIAAAAAAAAxtDZh4AA'
    'AACAAAAAHQAAAAAAAAAAAAAAAAAQRgAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzNQSwEC'
    'AAAAAAgIAAAAAAAAnJAkBQCQAAAAkAAAHQAAAAAAAAAAAAAAAAAQRwAAYmNfd2FybXN0YXJ0X3Nt'
    'YWxsX2NwdS9kYXRhLzRQSwECAAAAAAgIAAAAAAAAIfMX9YAAAACAAAAAHQAAAAAAAAAAAAAAAACQ'
    '1wAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzVQSwECAAAAAAgIAAAAAAAAguIQCYAAAACA'
    'AAAAHQAAAAAAAAAAAAAAAACQ2AAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzZQSwECAAAA'
    'AAgIAAAAAAAAeS5NvoAAAACAAAAAHQAAAAAAAAAAAAAAAACQ2QAAYmNfd2FybXN0YXJ0X3NtYWxs'
    'X2NwdS9kYXRhLzdQSwECAAAAAAgIAAAAAAAAmXrShACQAAAAkAAAHQAAAAAAAAAAAAAAAACQ2gAA'
    'YmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzhQSwECAAAAAAgIAAAAAAAAnEVfIIAAAACAAAAA'
    'HQAAAAAAAAAAAAAAAAAQawEAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzlQSwECAAAAAAgI'
    'AAAAAAAA1CGveYAAAACAAAAAHgAAAAAAAAAAAAAAAAAQbAEAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzEwUEsBAgAAAAAICAAAAAAAAMAtr3eAAAAAgAAAAB4AAAAAAAAAAAAAAAAAEG0BAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8xMVBLAQIAAAAACAgAAAAAAAC49JltAJAAAACQAAAe'
    'AAAAAAAAAAAAAAAAABBuAQBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMTJQSwECAAAAAAgI'
    'AAAAAAAA4REbGoAAAACAAAAAHgAAAAAAAAAAAAAAAACQ/gEAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzEzUEsBAgAAAAAICAAAAAAAAAuetPiAAAAAgAAAAB4AAAAAAAAAAAAAAAAAkP8BAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8xNFBLAQIAAAAACAgAAAAAAADfU1figAAAAIAAAAAe'
    'AAAAAAAAAAAAAAAAAJAAAgBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMTVQSwECAAAAAAgI'
    'AAAAAAAAxgvg+ACQAAAAkAAAHgAAAAAAAAAAAAAAAACQAQIAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzE2UEsBAgAAAAAICAAAAAAAAKFwjfiAAAAAgAAAAB4AAAAAAAAAAAAAAAAAEJICAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8xN1BLAQIAAAAACAgAAAAAAABxjMdsgAAAAIAAAAAe'
    'AAAAAAAAAAAAAAAAABCTAgBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMThQSwECAAAAAAgI'
    'AAAAAAAA33fZGoAAAACAAAAAHgAAAAAAAAAAAAAAAAAQlAIAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzE5UEsBAgAAAAAICAAAAAAAAHgyZf0AkAAAAJAAAB4AAAAAAAAAAAAAAAAAEJUCAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8yMFBLAQIAAAAACAgAAAAAAABAry5zgAAAAIAAAAAe'
    'AAAAAAAAAAAAAAAAAJAlAwBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMjFQSwECAAAAAAgI'
    'AAAAAAAAa+s+A4AAAACAAAAAHgAAAAAAAAAAAAAAAACQJgMAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzIyUEsBAgAAAAAICAAAAAAAAJbTiAyAAAAAgAAAAB4AAAAAAAAAAAAAAAAAkCcDAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8yM1BLAQIAAAAACAgAAAAAAABQkV+OAJAAAACQAAAe'
    'AAAAAAAAAAAAAAAAAJAoAwBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMjRQSwECAAAAAAgI'
    'AAAAAAAAOcDfRoAAAACAAAAAHgAAAAAAAAAAAAAAAAAQuQMAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzI1UEsBAgAAAAAICAAAAAAAAK4oXeUABAAAAAQAAB4AAAAAAAAAAAAAAAAAELoDAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8yNlBLAQIAAAAACAgAAAAAAABEv8TYIAAAACAAAAAe'
    'AAAAAAAAAAAAAAAAAJC+AwBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMjdQSwECAAAAAAgI'
    'AAAAAAAANOzqzABAAAAAQAAAHgAAAAAAAAAAAAAAAAAwvwMAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzI4UEsBAgAAAAAICAAAAAAAAEIzVxwAAgAAAAIAAB4AAAAAAAAAAAAAAAAAkP8DAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8yOVBLAQIAAAAACAgAAAAAAADJ5AN2AAIAAAACAAAe'
    'AAAAAAAAAAAAAAAAABACBABiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMzBQSwECAAAAAAgI'
    'AAAAAAAAUBSaRQQAAAAEAAAAHgAAAAAAAAAAAAAAAACQBAQAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzMxUEsBAgAAAAAICAAAAAAAANGeZ1UCAAAAAgAAAB4AAAAAAAAAAAAAAAAAFAUEAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvdmVyc2lvblBLAQIAAAAACAgAAAAAAAAWPUjCKAAAACgAAAAt'
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
