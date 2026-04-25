# Auto-generated Orbit Wars submission. Do not edit by hand.
# Built by tools/bundle.py on 2026-04-25 13:43:19.
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
    'aAwpUnIFAQAAdHIGAQAAUnIHAQAAdVgMAAAAYmVzdF92YWxfYWNjcggBAABHP9/V8rBujllYBQAA'
    'AGVwb2NocgkBAABLBVgIAAAAX3BhcnRpYWxyCgEAAIhYAwAAAGNmZ3ILAQAAfXIMAQAAKFgGAAAA'
    'Z3JpZF9ocg0BAABLMlgGAAAAZ3JpZF93cg4BAABLMlgKAAAAbl9jaGFubmVsc3IPAQAASwxYEQAA'
    'AGJhY2tib25lX2NoYW5uZWxzchABAABLIFgIAAAAbl9ibG9ja3NyEQEAAEsDWBEAAABuX2FjdGlv'
    'bl9jaGFubmVsc3ISAQAASwhYDAAAAHZhbHVlX2hpZGRlbnITAQAAS4B1WAsAAABtb2RlbF9zdGF0'
    'ZXIUAQAAaAJ1LlBLBwik/dAIkAsAAJALAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAACYAHABi'
    'Y193YXJtc3RhcnRfc21hbGxfY3B1Ly5mb3JtYXRfdmVyc2lvbkZCGABaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWloxUEsHCLfv3IMBAAAAAQAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAKQAoAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvLnN0b3JhZ2VfYWxpZ25tZW50RkIkAFpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWjY0UEsHCD93cekCAAAAAgAAAFBLAwQAAAgIAAAAAAAAAAAA'
    'AAAAAAAAAAAAIAAwAGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvYnl0ZW9yZGVyRkIsAFpaWlpaWlpa'
    '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'
    'H1ULvdb1gjw1S9I9ihiMPVQqKzw2CXG8TS7rOwwKKT2ubsA7Xl1nPQWv1D2NhWe9cISNvB4ZE71S'
    'cJg8w7OSPVN9i7yef2Y8VUnCPDU8LDttFJM9PDdBPC6Rq73oi5y9SWr5vfowCzzv0j49d7cOvDtv'
    'lz0LXn69WnqrunTCzj2DHBA93tOzvIJ3Bj4Vbku9gVBqPUlcXD3DypK9/XeavZcDFj3f8Sy9ZFkW'
    'vTQtrDzKSJG91vuiPcRGFbwhOBm9MJiNPYvmHT0t8xQ9BRBkvTohpj1kDaS9IeKnPaROJr1u5Zo8'
    'rfyUPXpRzr0kzR29dMSmvbX6pD3Oqd89jpExvDYZeL3WWLs7gvAaPSC+AD3cHAi8CkFAvWCBxT2A'
    'AUu9YDSfvRLJvrw08f08Ls9MPVQUUj1QSwcIvV8azgA2AAAANgAAUEsDBAAACAgAAAAAAAAAAAAA'
    'AAAAAAAAAAAdADUAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzFGQjEAWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWiJVdr1Q7Ra94GcIvTWOUr1NBKm9'
    '1UfrPHkA87yV0c+9lrFgvYxNU70w7Da92xOAvZTJObzdIMY8ygGavTtrSryKgAe9G95/veoBw7ts'
    'nye90+ZMPCAZjL0ZtxE87h2sPSwZsz3ET1W9ZdLjPNDKtL3F2ps90mWuvQceur0LT629UEsHCAAD'
    'pQiAAAAAgAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHQA1AGJjX3dhcm1zdGFydF9zbWFs'
    'bF9jcHUvZGF0YS8yRkIxAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpA6Xw/MY+BPxK/fz++kX4/Bcx+P+OUfj/Ui34/Lit7P19Dfz96JoE/FQN/PwADfj8F'
    '938/EFOBPyBSgT+CSX8/XpSAP4LQfz+FOIA/wHN/P05SgD9Vz4E/hE19P6bwgD9qWoE/R49+P3C5'
    'fj9ozX4/g/WAP8ILgD+UbH0/BT2BP1BLBwgAgCRRgAAAAIAAAABQSwMEAAAICAAAAAAAAAAAAAAA'
    'AAAAAAAAAB0ANQBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvM0ZCMQBaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaNYH/u4trJrsKYOg5u5uGuowvhjse'
    'ZwS8/70ZvESGGLyQmeM7Id6TO3g4ArwmXsY70l4CPNzGnjvy4f87LPYrO1HM9jvYR2a7KwgUPBmI'
    'jTpqsjU73kVDvPYpD7yZx0i7HFRkOwaF7TsdoYU7vW8DPAzUzzp4fx481Z56uxCPFTxQSwcITM5+'
    'aoAAAACAAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAdADUAYmNfd2FybXN0YXJ0X3NtYWxs'
    'X2NwdS9kYXRhLzRGQjEAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWjINgL2A7u+42fqQvE+ydTpTE5O8zTpVvFPSIb3GtVK9oEBJvVqgET2RksG8kZoTvDHE'
    'FD19bIO8Jr0TvLgdLL10wB29FPIJPdidWbsUnQ89TuIaPEC0C7v0GYC9Mh7cu/TsmryjPWi9Dozt'
    'vLeCujtOUDq9jKmrO4puQj3vepE8Dl1kPW0k7Dz8rkG9JbZ3vKg107yzwio7fxHEO5DJFj3ZG6C4'
    '8CUZvb4OMb2RgIW95ELUvLin5Dz7FA47CdhtvbEHWr0dtNM8mWBTvYksQb3VbBG9BfRiPYp9erxF'
    'n8m8CJ4wvYTo9zwja6i8nRtsPLU/njty1hw8R0wIPfDmXTsH6R08A1M9vIA3SL2NVIO8+15PPZDM'
    'Pb2Ivi29NouGvVHYgTvxf2q9o3uMvMQHXT26BYs8KVsLPY+NFDz+S2e90AYfvSPjK70Usya9olTY'
    'PLX+Cbw/mYI9nCoGvb8ETr2dzIY9qhJyvQUERT2H2Zc8c31IvanShD0aYce8GUYVvXqCW71Tu5e9'
    'GAO2OyX5Rr3xAjk8fMCRvMlz2Tz1Fby8o8S6PJx977tAOY47G/kEvejyKr221D883OUYvWyf77zW'
    '11y9GG7HvNIx9rwipXM9FioxPUCThDyrnx495yHnPJAMjr0ZSDG8//IivC1b2LzlVCE8sWOEvCDf'
    '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'
    'XIVsvKNxU72PtWy9a9AvPReVV71nhHc921NEvZfF8bwyfCc870FsPbuFsTubVhi95i8sPX25Zzxk'
    'yxq95w1lPePNMT1J2go8G5oPvQVg4TwA3dC6V1LLvM/cnjxbcmk9cgdWvMtyHT2vPwS99p6GvewT'
    'kztD05e7rAPtO6+vjL3fikO95PjkuyMwHjwAM4C9ioBVvQw8Rr0aEw+96PQxPa4W3Tz6aFM97F81'
    'PcH17DwHFCG9vKt0vcrzI73Hpl29wBsyPda1CL2oxQ492xcPvQShGb20Xwg8ft/NvNLZlzv3iyY8'
    'xB+Svetegb1IBUo9EnNoPGGll7xKV948cMZNvMz5BDzhQy694WUlvZZqvDx6OAw7/mF3vAkZlbyh'
    '/cM8jbA4Pf/KQz0oQLG8Y8rbvCoPm7x5X9i8X7SzvEFWCLz9pSg9/+YvvXhnWT31Chg9aaM6PUuj'
    'ED0UfcA8Q5NXveEJ6TysNZm97EglPO5OXj2GYzO7VeHAujRfJL2KgFq9KJX2OkekET0etlu92fe8'
    'vF82TTkICjO987FhvBCNVL1//mk78whDvU0YIz0flWI9BxwIPcSw/bz1l5k8T1B0u1BLBwhpG206'
    'AJAAAACQAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB0ANQBiY193YXJtc3RhcnRfc21hbGxf'
    'Y3B1L2RhdGEvNUZCMQBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpao1VavSYWOz0fCV89MpDVPATVPb3ydaK8xMA/ugw8YT2sQFa9kP8lvcbf7DzKcbo8fxaY'
    'vDGOOr0g6U68QZUyPatDWj0zux+7KgELvbZLhzyeeiA9RdwcvRddfTwuZxU9FPjFu2uFTL1CuQG9'
    '1OFsvRDhITsKVRc9UbFGu2PnIj1QSwcIQamJ/4AAAACAAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAA'
    'AAAAAAAdADUAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzZGQjEAWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWjt1gD+xen4/gOV9PxcbfT9I6X0/tdN/'
    'Py4agD8pdX4/yeF8P1bDgT95xX4//Mh+Pyl/fz8Wxnw/Fb1+P9GBfT8pGYA/OuZ+P0j2fj9GhoE/'
    'BJ99PyWogT8olYE/ywqBP7I+gD/3N4I/brJ+P0E2gT8Kjnw/Vk97PzOMfD/DXoA/UEsHCIRccMyA'
    'AAAAgAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHQA1AGJjX3dhcm1zdGFydF9zbWFsbF9j'
    'cHUvZGF0YS83RkIxAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlom0pw7dNhHu10Qj7sVCle8CHyQO92bIbwvCRC6dGWFOoeet7qvD2g6xKTuuioj07mZbGK8'
    'H0g9vB6kwrvD7Fi7cgicu0TGErx70Bk7Mf2SuJburDpAEgm7WtSUO6OvEjw9HGS6dVoHvCS3m7tu'
    'OyK8FlgevKjQdryPK5i8+4vOO1BLBwjZrJregAAAAIAAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAA'
    'AAAAAB0ANQBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvOEZCMQBaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlparV+DvbMyVj31vMO8GP8/vbMBMD03GNK8'
    'J+6JvAALqDx/t427TwpevG5wjTxnszQ9XSO8vPyxnDoNR8i8uLTIO2Oe9TwN7hU9aZfVu5p8Cr2I'
    '6AE93e5DvfntZb0W48G844wevYhtU72+o9I8RTGtPAog1zy+5Ms8diobPGoFOz0toiA9dFEqvYml'
    'gj3G9eE8l+hYPaQ3JT36cZC8C7veu+2ggDwZ2Xi9q2C1vMWBcb2pIU88AKwJPJbxMD3HM1+9B4s8'
    'Pc+cED2eE0C7NnVZveckoTsyaYW8XWEXveuo/7y1ZNq8a3SZvLb7Zjqs4m293g2bPNZtCL3e1Em9'
    's0sOPA6A77rEv0q9HbOMPUv+Q72OPSY9qf0BPJIMG72eDhu9juTfvMPHWj2+Cj49GjgCPcq0XT0y'
    'vNE861bsPNh6Qz0+P/g8KGPKu84yujyR+HM72jmEvAb2/jxechk9tpoXPQzIVT1+t489W1zUPF6D'
    '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'
    'AIAAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB4ANABiY193YXJtc3RhcnRfc21hbGxfY3B1'
    'L2RhdGEvMTBGQjAAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpalASAP+j+fz/xan8/i5iAP7FUfT+8IoA/osl+P252fz93ZoE/VYWAP83ZgT9vV34/cHuAP+Gw'
    'gD/sH38/3FyCP+rmfj9tl4A/LT6BP0dEfj/nQ4A/tpt/P84DgD+Cjn4/dQmAP2d/fj/PcX0/1iGA'
    'PxA5fz9TuX4/ZEOBP8qafz9QSwcIps7M3oAAAACAAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAA'
    'AAAeADQAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzExRkIwAFpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWnDicTu8jkE7GunbOkHUhDzzrkm8Zh1Su/6m'
    'zTq4mIS7POYCPGkHYzsIGbM7QjD1OtcKhLtHu1W7OQniO6pxT7yS46U7kjTkuoGzrDslm1U8TZPr'
    'u1lAvTtZwqa6Sxw/OmRtCTwBySK84ThtO+S8lbrgEoY4ejM2O9+KGTw3byu8UEsHCCDrOmOAAAAA'
    'gAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHgA0AGJjX3dhcm1zdGFydF9zbWFsbF9jcHUv'
    'ZGF0YS8xMkZCMABaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlomQjA9/pRwOlXWh70jCFy7+Hs5ux2j1Dvs1KW8jdFxPEAk4br23J68UZBGvZCJQ72e8029ntns'
    'vPr3Jj3jgyc9zNqeuwYcAb3R5Vw8BhRivUqpkTzZax89zOu/vBSqQ71uMN+8JOkHvfHKqzwmX0+9'
    'KoofPNI+Qz3Yp0K9FzDlvKXjYT3FoWq8rWDPvLA3VT0p1n+8O/fdOzc5SbyQT3c9pGbZPFtIhDxx'
    'B0Y9LS8MO9a3Qr2ESiy9H87gOr6Ovjwyrw09Dc2YPH/oqDz70m48zEcavbOKHr1ZX6S6T/ASvUrs'
    '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'
    'WpL8ZT3YhWa9QKIjvPJaEr1t0Jg5z7gtPUiBLr28HkE9rcIMvbLCXT3KN0g9CECEvFPELrzOLJm8'
    '+X8BvaJECrtueTW9rmYLvc+TbLyOmAQ9cXlgvaKSKj0etc+8K8wCvPOAAz1G5Ce9qms0vemFSzpf'
    'Qw89G651PKT0Eb3jgV89UEsHCOkhudGAAAAAgAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAA'
    'HgA0AGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8xNEZCMABaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWloirIE/A/+CP+uPgD/9qII/2fuAP2H3gT/7l4A/'
    '55B/P4Xgez9D44A/qdCDP8rtfD8ULII/H2d/P+3yfz/Pm4A/dKeAPy4Zgz9BXIM/fE6AP0A2fj9o'
    'WYA/NVqBP05/gz+vPYI/sMqDP9tsgz+9zoE/eV2BP5/RgD9iqoE/9ziCP1BLBwiBhn8OgAAAAIAA'
    'AABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB4ANABiY193YXJtc3RhcnRfc21hbGxfY3B1L2Rh'
    'dGEvMTVGQjAAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'ntwzPMAGJjzSVgw8QAWOPNqjnLr/oic8hRKBvCNuo7qVIg+7oF6LuwXdzLuLMKO82g4FOwZc8Drr'
    '55U7FCXTO0EkXrqqrno8FmCiPNcomrvPvUY7wXfmO29PgDuVq7Q7JD9RPCfawzyuUdk8iaGrO8Wc'
    'kTyKdho8DVOTPD8otzxQSwcIKaMIPYAAAACAAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAe'
    'ADQAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzE2RkIwAFpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWs1lNj06mP+7fenavOSSTz3Spsc8yjfIPNhDHT2y'
    'eQ29TZ5rvZziRr0Q9Q+9eYCPPALpZr2spws9oCoEvYWrT700WNe79sQ5vTzCiT2UzzK9gA1VvICq'
    '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'
    'PcthXz3uIv87Yly1PFBLBwjOatzQAJAAAACQAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB4A'
    'NABiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMTdGQjAAWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaBImxu4fDED3wFY48YqsfvVyVUjt9w/88GSYcPC98'
    'e72QDIE9nZCCuxSYcr05Py09XJ5su4ypMT3KXAo9usuXvOJcNTyjIbC8ej41PY89Xj2GVHs9EpZ9'
    'PDBhBz0hGtQ8YLifvMsB6zz5SLM8srYoPVJl6rxQJxC9oIAMvTkYo7tQSwcIfqvGD4AAAACAAAAA'
    'UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAeADQAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRh'
    'LzE4RkIwAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWo5e'
    'fj/q/H0/EBuBP2psgT8EaIA/bgaBP0prgT/W7IA/F62BP3Bfej++RoA/3dyDP/MhgT8Jcn8/7j2B'
    'P2JwgD+U1YE/JiR/P0uygD8uBIA/z8WAP5qKgj862Xw/d4p+P5nqfT+EF34/VlmBP28Jfz+TZoA/'
    'yiZ+P/2ggD+xh4A/UEsHCAnBxcCAAAAAgAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHgA0'
    'AGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8xOUZCMABaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlotA2C7PX/yu7kyrDqBRN078FYVO6YAOzvW+Sc7ZXjZ'
    'O37LBTvtlZO8q9yiOm9hozu8gJU7tN7mOL00aTts3Yi7y5bIOyJhZzv1KIk6nlQQPNcX2jtiDJ24'
    'V2FbvGG4W7yNble5JnGdOy5FFzkdiG+8sABiPP32JLy/sa471Cu3O1BLBwh/bitygAAAAIAAAABQ'
    'SwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB4ANABiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEv'
    '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'
    'PWYmOTx2pJA9SonNuqx/CD21qlk99ba3vM1UybwXGn89AdbePIx3iDw0WaU7P00PPUdTD73FKqu7'
    'ZhTHPDith7y2UW281s5EvVew2TwLgjg9pEV6vcxDHj06qpI9frVDPX8nDz3rMCg8JV+AvCQTb7ws'
    'LUK8kSw3PUEj8Dzquu28uKvpvBWa7rxdKCu7eAo5vPDQETzYPuq8UEsHCFOETiQAkAAAAJAAAFBL'
    'AwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHgA0AGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8y'
    'MUZCMABaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlrQ9FI9'
    'x3M9vapdPz20UUa8BWoDvRsZmrzmPtS83rbuOg0gMT3bqFS9BakxvJroVL0gw2m93JxuPUUuSrq5'
    'Uaw7fR+zvB7/N71DtEo9JKt1vKz+kLsE8B+9tU7wvO1/Mb0ocBy8ftp/PWhEBry526W88MZ5PSay'
    'zDwu2gy98fHVPFBLBwjXivZPgAAAAIAAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB4ANABi'
    'Y193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMjJGQjAAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaZziEP+ikhT9IuYU/1L+CPzrWgj/ZZ4Y/h/ODP+mKgj+D'
    'TYU/tjSCP5oEgj+Pt4Q/Xl2DP7sYhz+Aa4Q/e4WDPzlvgj/Ze4Q/yI2IP08shT8nYoc/Eg6DPxpk'
    'hT/XFIc/aUiFP2DnhD9TiYM/JemFPwAlhT9B3oM/BZiFPxktgz9QSwcIrjqEOoAAAACAAAAAUEsD'
    'BAAACAgAAAAAAAAAAAAAAAAAAAAAAAAeADQAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzIz'
    'RkIwAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWqK4AD1x'
    '1jY8g/wzPBp0PjxEO5g8tfDxPGHz+Tx+5xY8OM19POE2VjwNjsI8lM+HPNu0qDzjzgY9eP/8PGWe'
    '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'
    'PXDOgDtOuKQ8odWOPFzJtLw4zVM9KR2oPPaLpjzEHY87xXrxPMmEJj0hcgE9REhhPcLyZTzMBc47'
    'DM86Paf3SrpVQjM8y6wAPJQwA7xzSAq9FUkfPXZVsDytAFC9qNEMPQRFzryIbds8xfVcPJAJFz3x'
    'oQc9BBafvGSbDr2AzhG9AaUHPWIkUj1beYk8E+WSPEAc5roTtsO8XViyvBZPk7wkq4u88SyUPeMV'
    'Bb3blEO8/ILmvG/EgTsky+87NUn+vBf2Dr24jjC9za8RPGir9Dweh049hIyrvLt0sTyv7p48o80g'
    'PFjmQ7tWoh68lpJFvV5xPr1+8KG6PW1ovJVrbr3fXB+92WAgvY2vl70m9v08z8lzvSxGEb3UAJ88'
    'xwUmvY7mWL1QSwcI2jebCwCQAAAAkAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAeADQAYmNf'
    'd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzI1RkIwAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWtn6d7xgY7E8+7sNvcBOzjw6G1k86rm9PHaOSrmYViS9xpTg'
    'vDBi2ryxx4S9QNDzvJGPwbtnJZU888pOvcFzkzsWf8+8XonovLDCzbyWd4w83iYRvW3rK73tEa07'
    'J6STvFDc6rr7BT+9cYQyPFbD1Dz/nNA8ze7XvAUIzrwJbLS8UEsHCJCVT3SAAAAAgAAAAFBLAwQA'
    'AAgIAAAAAAAAAAAAAAAAAAAAAAAAHgA0AGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8yNkZC'
    'MABaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlo0syw+Xl1k'
    'vffpjz1+mHy9QFjmvZ1alD3u4ia+YZ71vBv+Jb5nRxo+GR7bPZaiHj7+kic9tAc2vd2jNrw/0iK+'
    '2FqEvRevFb6aYco9a70lvgxPKT48PJe9mIl+PbzdKT5n2xu+1wCfuyg+Kr55YIc8PJxzPWLvAL4d'
    '7yg8++BPvcykr7zOAg2+aya0u2P1O72XdxQ+lpsCvvMcF75MmjY+TbbKPfs5LT4fbRw+re6lPHSp'
    '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'
    'NABiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMzFGQjAAWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaslhBPVBLBwhQFJpFBAAAAAQAAABQSwMEAAAICAAA'
    'AAAAAAAAAAAAAAAAAAAAAB4AMABiY193YXJtc3RhcnRfc21hbGxfY3B1L3ZlcnNpb25GQiwAWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlozClBLBwjRnmdVAgAAAAIA'
    'AABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAC0AIwBiY193YXJtc3RhcnRfc21hbGxfY3B1Ly5k'
    'YXRhL3NlcmlhbGl6YXRpb25faWRGQh8AWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWjEw'
    'MTk4NTcwNzU4NzA4MDQ0NzUxMDc5NTkzOTIzODQzOTk0ODA1NzVQSwcI7oXSNCgAAAAoAAAAUEsB'
    'AgAAAAAICAAAAAAAAKT90AiQCwAAkAsAAB8AAAAAAAAAAAAAAAAAAAAAAGJjX3dhcm1zdGFydF9z'
    'bWFsbF9jcHUvZGF0YS5wa2xQSwECAAAAAAgIAAAAAAAAt+/cgwEAAAABAAAAJgAAAAAAAAAAAAAA'
    'AAAgDAAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS8uZm9ybWF0X3ZlcnNpb25QSwECAAAAAAgIAAAA'
    'AAAAP3dx6QIAAAACAAAAKQAAAAAAAAAAAAAAAACRDAAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS8u'
    'c3RvcmFnZV9hbGlnbm1lbnRQSwECAAAAAAgIAAAAAAAAhT3jGQYAAAAGAAAAIAAAAAAAAAAAAAAA'
    'AAASDQAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9ieXRlb3JkZXJQSwECAAAAAAgIAAAAAAAAvV8a'
    'zgA2AAAANgAAHQAAAAAAAAAAAAAAAACWDQAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzBQ'
    'SwECAAAAAAgIAAAAAAAAAAOlCIAAAACAAAAAHQAAAAAAAAAAAAAAAAAQRAAAYmNfd2FybXN0YXJ0'
    'X3NtYWxsX2NwdS9kYXRhLzFQSwECAAAAAAgIAAAAAAAAAIAkUYAAAACAAAAAHQAAAAAAAAAAAAAA'
    'AAAQRQAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzJQSwECAAAAAAgIAAAAAAAATM5+aoAA'
    'AACAAAAAHQAAAAAAAAAAAAAAAAAQRgAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzNQSwEC'
    'AAAAAAgIAAAAAAAAaRttOgCQAAAAkAAAHQAAAAAAAAAAAAAAAAAQRwAAYmNfd2FybXN0YXJ0X3Nt'
    'YWxsX2NwdS9kYXRhLzRQSwECAAAAAAgIAAAAAAAAQamJ/4AAAACAAAAAHQAAAAAAAAAAAAAAAACQ'
    '1wAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzVQSwECAAAAAAgIAAAAAAAAhFxwzIAAAACA'
    'AAAAHQAAAAAAAAAAAAAAAACQ2AAAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzZQSwECAAAA'
    'AAgIAAAAAAAA2aya3oAAAACAAAAAHQAAAAAAAAAAAAAAAACQ2QAAYmNfd2FybXN0YXJ0X3NtYWxs'
    'X2NwdS9kYXRhLzdQSwECAAAAAAgIAAAAAAAAs1mVVwCQAAAAkAAAHQAAAAAAAAAAAAAAAACQ2gAA'
    'YmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzhQSwECAAAAAAgIAAAAAAAAHxVDWIAAAACAAAAA'
    'HQAAAAAAAAAAAAAAAAAQawEAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzlQSwECAAAAAAgI'
    'AAAAAAAAps7M3oAAAACAAAAAHgAAAAAAAAAAAAAAAAAQbAEAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzEwUEsBAgAAAAAICAAAAAAAACDrOmOAAAAAgAAAAB4AAAAAAAAAAAAAAAAAEG0BAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8xMVBLAQIAAAAACAgAAAAAAABKIBY2AJAAAACQAAAe'
    'AAAAAAAAAAAAAAAAABBuAQBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMTJQSwECAAAAAAgI'
    'AAAAAAAA6SG50YAAAACAAAAAHgAAAAAAAAAAAAAAAACQ/gEAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzEzUEsBAgAAAAAICAAAAAAAAIGGfw6AAAAAgAAAAB4AAAAAAAAAAAAAAAAAkP8BAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8xNFBLAQIAAAAACAgAAAAAAAApowg9gAAAAIAAAAAe'
    'AAAAAAAAAAAAAAAAAJAAAgBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMTVQSwECAAAAAAgI'
    'AAAAAAAAzmrc0ACQAAAAkAAAHgAAAAAAAAAAAAAAAACQAQIAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzE2UEsBAgAAAAAICAAAAAAAAH6rxg+AAAAAgAAAAB4AAAAAAAAAAAAAAAAAEJICAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8xN1BLAQIAAAAACAgAAAAAAAAJwcXAgAAAAIAAAAAe'
    'AAAAAAAAAAAAAAAAABCTAgBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMThQSwECAAAAAAgI'
    'AAAAAAAAf24rcoAAAACAAAAAHgAAAAAAAAAAAAAAAAAQlAIAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzE5UEsBAgAAAAAICAAAAAAAAFOETiQAkAAAAJAAAB4AAAAAAAAAAAAAAAAAEJUCAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8yMFBLAQIAAAAACAgAAAAAAADXivZPgAAAAIAAAAAe'
    'AAAAAAAAAAAAAAAAAJAlAwBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMjFQSwECAAAAAAgI'
    'AAAAAAAArjqEOoAAAACAAAAAHgAAAAAAAAAAAAAAAACQJgMAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzIyUEsBAgAAAAAICAAAAAAAAAXwq4CAAAAAgAAAAB4AAAAAAAAAAAAAAAAAkCcDAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8yM1BLAQIAAAAACAgAAAAAAADaN5sLAJAAAACQAAAe'
    'AAAAAAAAAAAAAAAAAJAoAwBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMjRQSwECAAAAAAgI'
    'AAAAAAAAkJVPdIAAAACAAAAAHgAAAAAAAAAAAAAAAAAQuQMAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzI1UEsBAgAAAAAICAAAAAAAAJYGGfwABAAAAAQAAB4AAAAAAAAAAAAAAAAAELoDAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8yNlBLAQIAAAAACAgAAAAAAAB4cMJsIAAAACAAAAAe'
    'AAAAAAAAAAAAAAAAAJC+AwBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMjdQSwECAAAAAAgI'
    'AAAAAAAANOzqzABAAAAAQAAAHgAAAAAAAAAAAAAAAAAwvwMAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzI4UEsBAgAAAAAICAAAAAAAAEIzVxwAAgAAAAIAAB4AAAAAAAAAAAAAAAAAkP8DAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8yOVBLAQIAAAAACAgAAAAAAADJ5AN2AAIAAAACAAAe'
    'AAAAAAAAAAAAAAAAABACBABiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMzBQSwECAAAAAAgI'
    'AAAAAAAAUBSaRQQAAAAEAAAAHgAAAAAAAAAAAAAAAACQBAQAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzMxUEsBAgAAAAAICAAAAAAAANGeZ1UCAAAAAgAAAB4AAAAAAAAAAAAAAAAAFAUEAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvdmVyc2lvblBLAQIAAAAACAgAAAAAAADuhdI0KAAAACgAAAAt'
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
