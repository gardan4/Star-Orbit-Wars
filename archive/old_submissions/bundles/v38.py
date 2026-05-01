# Auto-generated Orbit Wars submission. Do not edit by hand.
# Built by tools/bundle.py on 2026-04-29 21:46:40.
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

# Memoize the default-max_speed branch. The full game only ever uses the
# default; memoizing avoids ~600k repeated log()+pow()+min() calls per
# 30-rollout MCTS turn (saves ~5% wall on the heuristic act() hot path).
_FLEET_SPEED_CACHE: dict = {}


def fleet_speed(ships: int, max_speed: float = DEFAULT_MAX_SPEED) -> float:
    """Engine's fleet speed formula. ships >= 1."""
    if max_speed == DEFAULT_MAX_SPEED:
        cached = _FLEET_SPEED_CACHE.get(ships)
        if cached is not None:
            return cached
    if ships <= 0:
        v = 0.0
    elif ships == 1:
        v = 1.0
    else:
        s = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5
        v = s if s < max_speed else max_speed
    if max_speed == DEFAULT_MAX_SPEED:
        _FLEET_SPEED_CACHE[ships] = v
    return v


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
    # Per-turn invariants. ``is_orbiting_planet`` and
    # ``initial_orbit_params`` depend only on the (planet, initial_planet)
    # pair and never change within a single act() call, but the heuristic
    # + arrival-table builder collectively call them ~1M times per
    # 30-rollout MCTS turn. Populating once in ``parse_obs`` eliminates
    # the recomputation hot-path. Keyed by planet pid.
    is_orbiting_by_pid: Dict[int, bool] = field(default_factory=dict)
    orbit_params_by_pid: Dict[int, Tuple[float, float]] = field(default_factory=dict)


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

    # Cache per-turn orbit invariants for every planet so downstream
    # callers (build_arrival_table, _travel_turns, _intercept_position)
    # can skip recomputing.
    for pl in p.planets:
        pid = pl[0]
        ip = p.initial_planet_by_id.get(pid, pl)
        idx = float(ip[2]) - 50.0
        idy = float(ip[3]) - 50.0
        orb_r = math.hypot(idx, idy)
        init_angle = math.atan2(idy, idx)
        p.orbit_params_by_pid[pid] = (orb_r, init_angle)
        # is_orbiting rule: orb_r + planet_radius < ROTATION_RADIUS_LIMIT (50)
        p.is_orbiting_by_pid[pid] = (orb_r + float(pl[4])) < 50.0

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
            is_orb = po.is_orbiting_by_pid.get(pid, False)
            if is_orb:
                ir, ia = po.orbit_params_by_pid[pid]
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
    # Read orbit metadata from the per-turn cache when available.
    orb_cache = getattr(po, "is_orbiting_by_pid", None) if po is not None else None
    if orb_cache is not None and tpid in orb_cache:
        is_orb = orb_cache[tpid]
        ir_ia = po.orbit_params_by_pid.get(tpid) if is_orb else None
    else:
        is_orb = is_orbiting_planet(target_pl, initial_pl)
        ir_ia = initial_orbit_params(initial_pl) if is_orb else None
    if is_orb:
        ir, ia = ir_ia  # type: ignore[misc]
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
    orb_cache = getattr(po, "is_orbiting_by_pid", None) if po is not None else None
    if orb_cache is not None and tpid in orb_cache:
        is_orb = orb_cache[tpid]
        ir_ia = po.orbit_params_by_pid.get(tpid) if is_orb else None
    else:
        is_orb = is_orbiting_planet(target_pl, initial_pl)
        ir_ia = initial_orbit_params(initial_pl) if is_orb else None
    if is_orb:
        ir, ia = ir_ia  # type: ignore[misc]
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
    # Macro-action library at root (Plan §6.4). When True, ``GumbelRootSearch``
    # injects up to 4 pre-expanded "obvious" joint actions (HOLD-all,
    # mass-attack-nearest, reinforce-weakest, retreat-to-largest) as
    # additional candidates alongside the heuristic anchor and the
    # Gumbel samples. Insurance against a bad NN prior; documented +Elo
    # trick from microRTS literature. Macros are NOT protected from SH
    # pruning — only the heuristic anchor stays protected. Default off
    # so the v12 baseline path is bit-identical.
    use_macros: bool = False
    # Mixed leaf evaluator (variance reduction). When ``rollout_policy``
    # is ``"nn_value"`` and ``value_mix_alpha`` is in (0, 1), the leaf
    # value is the convex combination
    #     V_leaf = α · V_NN  +  (1 − α) · V_heuristic_rollout
    # where V_NN is the value-head's 1-ply-ahead estimate and
    # V_heuristic_rollout is a depth-``rollout_depth`` heuristic-vs-
    # heuristic rollout from the same post-action state. NN provides a
    # long-horizon prior (correlates with eventual outcomes); the
    # heuristic rollout provides a short-horizon accurate signal.
    # Combining them is a control-variate-style variance reduction.
    #
    # ``α = 1.0`` (default) recovers the existing pure-NN-value path.
    # ``α = 0.0`` is equivalent to plain heuristic rollouts.
    # ``α = 0.5`` is a sensible starting point for the unblock; tune
    # via H2H. Cost is ~2× the leaf-eval wall when α ∈ (0, 1) since both
    # paths run on every leaf, so the rollout-budget projection halves.
    # See docs/STATUS.md ("Possible structural fixes — Use NN as a
    # MIXTURE-WITH-HEURISTIC value: untested.") for the original
    # motivation.
    value_mix_alpha: float = 1.0


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


def _value_fn_eval(
    base_state: GameState,
    my_player: int,
    my_action: List[List],
    opp_agent_factory: Callable[[], Any],
    value_fn: Callable[[Any, int], float],
    num_agents: int = 2,
    rng: Optional[random.Random] = None,
    opp_turn0_action: Optional[List[List]] = None,
    hard_stop_at: Optional[float] = None,
) -> float:
    """Apply 1 ply (my_action + opp's turn-0) then query value head.

    The "AlphaZero-style" leaf evaluation: instead of rolling out
    ``depth`` plies with the heuristic and scoring the terminal state,
    apply only the candidate's first ply and ask the NN value head
    "what is this resulting state worth from my_player's perspective?"

    Why 1 ply instead of 0: applying the joint action is necessary to
    actually evaluate the *candidate*. With 0 plies, every candidate
    would query the value head on the same pre-action state and get
    the same value — Q-aggregation would collapse to anchor wins on
    tie-breaking. With 1 ply, the candidates produce different
    post-action states, so the value head can distinguish "good
    move" from "bad move" if it's been trained to.

    Why not 2+ plies: that's just a partial rollout. The whole point
    of value-head Q is to use the NN's own state-value estimate,
    avoiding the rollout's heuristic bias. Adding more plies dilutes
    the NN signal with heuristic continuation. (Future: configurable
    extra plies as a post-NN bootstrap, like MuZero. Not needed for v1.)

    Returns scalar in [-1, 1] from my_player's perspective. If
    value_fn raises or returns non-finite, returns the heuristic
    score of the post-step state — graceful fallback so a defective
    value_fn never forfeits a turn.
    """
    eng = FastEngine(
        copy.deepcopy(base_state),
        num_agents=num_agents,
        rng=rng,
    )

    # Turn 0: my root action + opp's turn-0 response. Same setup as
    # _rollout_value, just without the depth>=2 continuation.
    actions: List[Optional[List]] = [None] * num_agents
    actions[my_player] = my_action
    for i in range(num_agents):
        if i == my_player:
            continue
        if opp_turn0_action is not None:
            actions[i] = opp_turn0_action
        else:
            opp = opp_agent_factory()
            try:
                actions[i] = opp.act(
                    eng.observation(i), Deadline(hard_stop_at=hard_stop_at),
                )
            except Exception:
                actions[i] = []
    eng.step(actions)

    # Game ended on turn 0? Use terminal score directly — value_fn
    # would just be predicting the known outcome.
    if eng.done:
        return _score_from_engine(eng, my_player, num_agents)

    # Query the value head on the post-step state from my_player's view.
    try:
        post_obs = eng.observation(my_player)
        v = float(value_fn(post_obs, my_player))
        if v != v or v == float("inf") or v == float("-inf"):  # NaN/inf guard
            return _score_from_engine(eng, my_player, num_agents)
        # Clip to the same [-1, 1] convention _score_from_engine uses
        # so anchor-margin and Q-comparisons stay scale-consistent.
        return max(-1.0, min(1.0, v))
    except Exception:
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
    # All joint candidates Sequential Halving evaluated this turn,
    # parallel-indexed with q_values and visits. Carried for external
    # tooling (tools/collect_mcts_demos.py) that wants the full visit
    # distribution per planet, not just the winner. The act() hot path
    # does not consume this — pure observability.
    candidates: List[JointAction] = field(default_factory=list)

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
            candidates=list(candidates),
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
        candidates=list(candidates),
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
    # Phase 1 of NN value-head Q (see ``docs/NN_VALUE_HEAD_DESIGN.md``).
    # When set AND ``gumbel_cfg.rollout_policy == "nn_value"``, leaf
    # evaluation applies the candidate's joint action for one ply and
    # queries this function on the resulting state — instead of
    # running a depth=15 heuristic rollout. Lets the NN drive Q
    # estimates instead of the heuristic. Built via
    # ``orbitwars.nn.nn_value.make_nn_value_fn``. Signature:
    #   ``(obs, my_player) -> float in [-1, 1]``
    value_fn: Optional[Callable[[Any, int], float]] = None

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

        # Optional macro candidates (Plan §6.4). Built once per turn,
        # appended after the heuristic anchor and BEFORE Gumbel samples.
        # Macros are not protected from SH pruning — they have to "earn"
        # their visits through positive rollouts. The macro module also
        # de-dupes against itself; we de-dupe against the anchor here.
        macro_joints: List[JointAction] = []
        macro_keys: set = set()
        if self.gumbel_cfg.use_macros:
            try:
                for mj in build_macro_anchors(po, per_planet):
                    mk = tuple(tuple(m) for m in mj.to_wire())
                    if mk == anchor_key or mk in macro_keys:
                        continue
                    macro_keys.add(mk)
                    macro_joints.append(mj)
            except Exception:
                # Defensive: a buggy macro must NEVER forfeit a turn.
                macro_joints = []
                macro_keys = set()

        # Sample Gumbel candidates. We leave slots for the anchor +
        # macros so the total effective candidate count stays
        # ~num_candidates.
        reserved = (1 if anchor_joint else 0) + len(macro_joints)
        sample_budget = self.gumbel_cfg.num_candidates - reserved
        sample_budget = max(sample_budget, 1)
        sampled = enumerate_joints(per_planet, sample_budget, self._rng)

        # Compose the final candidate list: anchor first (if any), then
        # macros, then Gumbel samples that don't duplicate either.
        joints: List[JointAction] = []
        if anchor_joint is not None:
            joints.append(anchor_joint)
        joints.extend(macro_joints)
        for j in sampled:
            key = tuple(tuple(m) for m in j.to_wire())
            if key == anchor_key or key in macro_keys:
                continue
            joints.append(j)

        if not joints:
            return None
        if len(joints) == 1:
            return SearchResult(
                best_joint=joints[0], n_rollouts=0, duration_ms=0.0,
                q_values=[0.0], visits=[0], aborted=False,
                candidates=list(joints),
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

        # Choose leaf evaluator based on rollout_policy.
        # "nn_value" (with value_fn supplied) skips rollouts entirely
        # and queries the NN value head on the 1-ply-ahead state.
        # See _value_fn_eval and docs/NN_VALUE_HEAD_DESIGN.md.
        use_nn_value = (
            self.gumbel_cfg.rollout_policy == "nn_value"
            and self.value_fn is not None
        )
        if self.gumbel_cfg.rollout_policy == "nn_value" and self.value_fn is None:
            # Configured for nn_value but no value_fn supplied — fall
            # back to heuristic rollouts with a one-time warning. The
            # search must NEVER forfeit a turn just because the NN
            # plumbing is incomplete.
            import warnings
            warnings.warn(
                "rollout_policy='nn_value' but no value_fn supplied; "
                "falling back to heuristic rollouts.",
                stacklevel=2,
            )

        # Mixed-eval blend factor (only meaningful under nn_value).
        mix_alpha = float(self.gumbel_cfg.value_mix_alpha)
        # Clamp to a defensible range; bundle.py also validates upstream.
        if mix_alpha < 0.0:
            mix_alpha = 0.0
        elif mix_alpha > 1.0:
            mix_alpha = 1.0
        use_mix = use_nn_value and mix_alpha < 1.0

        def rollout_fn(joint: JointAction) -> float:
            if use_nn_value:
                v_nn = _value_fn_eval(
                    base_state=base_state,
                    my_player=my_player,
                    my_action=joint.to_wire(),
                    opp_agent_factory=self._opp_factory,
                    value_fn=self.value_fn,
                    num_agents=num_agents,
                    rng=self._rng,
                    hard_stop_at=rollout_deadline_sec,
                )
                if not use_mix:
                    return v_nn
                # Blend with a heuristic rollout for variance reduction.
                v_heur = _rollout_value(
                    base_state=base_state,
                    my_player=my_player,
                    my_action=joint.to_wire(),
                    opp_agent_factory=self._opp_factory,
                    my_future_factory=self._my_future_factory,
                    depth=self.gumbel_cfg.rollout_depth,
                    num_agents=num_agents,
                    rng=self._rng,
                    deadline_fn=_rollout_deadline_fired,
                    hard_stop_at=rollout_deadline_sec,
                    per_rollout_budget_ms=self.gumbel_cfg.per_rollout_budget_ms,
                )
                return mix_alpha * v_nn + (1.0 - mix_alpha) * v_heur
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
                if use_nn_value:
                    return _value_fn_eval(
                        base_state=base_state,
                        my_player=my_player,
                        my_action=my_joint.to_wire(),
                        opp_agent_factory=self._opp_factory,
                        value_fn=self.value_fn,
                        num_agents=num_agents,
                        rng=self._rng,
                        opp_turn0_action=opp_wire,
                        hard_stop_at=rollout_deadline_sec,
                    )
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

def _softmax_np(x: np.ndarray) -> np.ndarray:
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
            if float(_softmax_np(self.log_alpha).max()) >= self.freeze_threshold:
                self._frozen = True

    def distribution(self) -> np.ndarray:
        """Posterior over archetypes as a probability vector."""
        return _softmax_np(self.log_alpha)

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



# --- inlined: orbitwars/features/obs_encode.py ---

"""Observation -> feature tensor encoders for both W4 arch candidates.

Two encoders sharing a single source obs, pinned to the schemas in
``orbitwars.nn.conv_policy`` and ``orbitwars.nn.set_transformer``:

  * ``encode_grid(obs, player_id) -> np.ndarray (C=12, H=50, W=50)``
    matches ``conv_policy.feature_channels()``. Each channel is a
    dense rasterization over the 100x100 board downsampled to 50x50.

  * ``encode_entities(obs, player_id) -> (features, mask)`` matches
    ``set_transformer.entity_feature_schema()``. ``features`` is
    ``(n_max_entities, 19)`` with padding rows zeroed; ``mask`` is
    ``(n_max_entities,)`` bool — True where the row is valid.

Both encoders operate on a Kaggle obs dict (or a ``ParsedObs`` — we
tolerate either). They are pure numpy, no torch dependency, so they run
at MCTS-rollout speed. The training path wraps them with a torch tensor
conversion.

**Perspective-normalization**: both encoders take ``player_id`` and
encode "me vs. everyone else" regardless of the raw seat id. Under the
4-fold symmetry the board is identical up to rotation, so the encoder
alone is enough — no extra rotation is applied here. Data augmentation
(flip/rotate) happens at the training-loop level, not inside this
module, so inference and training share one canonical encoder.

**Sqrt scaling**: ship counts go through ``sqrt(x) / sqrt(1000)``. The
game's fleet speed formula is ``1 + 5 * (log(ships)/log(1000))^1.5``,
so fleet dynamics are already log-scaled. Sqrt is a gentler compression
that keeps small fleets distinguishable (1 vs 10 ships matters early
game) while capping large fleets (1000 vs 5000 is the same for policy
purposes — both just move fast).

**What's NOT here** (out of scope for the skeleton):
  * 4-fold symmetry augmentation helpers (rotate/mirror). Training-side.
  * Batched encoding (``encode_batch``). Trivial wrapper once this
    single-obs path is validated.
  * Fourier positional encoding of pos_x/y for the set-transformer —
    applied INSIDE the model (``set_transformer._fourier_encode``), not
    here, so the encoder stays model-agnostic.
"""

from typing import Any, Tuple

import numpy as np



# ---------------------------------------------------------------------------
# Small helpers — tolerate either a kaggle obs (dict/AttrDict) or ParsedObs.
# ---------------------------------------------------------------------------


def _obs_get(obs: Any, key: str, default: Any = None) -> Any:
    """Dict-or-attr accessor matching ``heuristic.obs_get`` semantics.

    Kaggle passes an AttrDict whose values can also be accessed via
    attribute; ``ParsedObs`` is a regular dataclass. One helper fits
    both so callers don't need to branch.
    """
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


_BOARD_SIZE = 100.0
_SUN_POS = (50.0, 50.0)
_MAX_STEPS = 500
_SHIP_SCALE = float(np.sqrt(1000.0))


def _sqrt_scale_ships(ships: float) -> float:
    """``sqrt(ships) / sqrt(1000)``; saturates gracefully beyond 1000."""
    return float(np.sqrt(max(0.0, ships))) / _SHIP_SCALE


def _planet_to_grid(x: float, y: float, cfg: ConvPolicyCfg) -> Tuple[int, int]:
    """Continuous (x, y) in [0, 100] -> (gy, gx) cell index, clamped."""
    gy = int(y * cfg.grid_h / _BOARD_SIZE)
    gx = int(x * cfg.grid_w / _BOARD_SIZE)
    gy = max(0, min(cfg.grid_h - 1, gy))
    gx = max(0, min(cfg.grid_w - 1, gx))
    return gy, gx


# ---------------------------------------------------------------------------
# Grid encoder (conv_policy candidate A).
# ---------------------------------------------------------------------------


def encode_grid(
    obs: Any,
    player_id: int,
    cfg: ConvPolicyCfg | None = None,
) -> np.ndarray:
    """Encode obs as a ``(C, H, W)`` conv input tensor.

    Channel order is locked to ``conv_policy.feature_channels()``. Each
    planet and fleet rasterizes to its single (gy, gx) cell — we do NOT
    splat into a radius, because grid resolution (H=50 over a 100x100
    board, 2x2 units per cell) already matches planet radius 1-3, and
    the conv itself learns the effective radius from neighboring cells.

    Args:
      obs: Kaggle obs or ``ParsedObs`` for ``player_id``.
      player_id: seat id (0, 1, 2, 3) — perspective for me/enemy channels.
      cfg: optional override of conv cfg (grid size + channel count).

    Returns:
      float32 ``(n_channels, grid_h, grid_w)`` numpy array.

    Contract:
      * Output dtype is ``np.float32`` (torch-friendly; 4x smaller than
        float64 and negligible precision loss at our scale).
      * Channel dimension comes FIRST (PyTorch convention), not last.
      * No NaN / inf produced — callers can trust the tensor is safe to
        pass directly to a torch conv without masking.
    """
    if cfg is None:
        cfg = ConvPolicyCfg()
    names = feature_channels()
    C = len(names)
    assert C == cfg.n_channels, f"channel count drift: {C} != {cfg.n_channels}"

    grid = np.zeros((C, cfg.grid_h, cfg.grid_w), dtype=np.float32)

    # Channel indices.
    CH = {name: i for i, name in enumerate(names)}

    # --- Planets ---
    comet_pids = set(_obs_get(obs, "comet_planet_ids", []) or [])
    planets = _obs_get(obs, "planets", []) or []
    for pl in planets:
        # Planet row: [id, owner, x, y, radius, ships, production]
        pid = int(pl[0])
        owner = int(pl[1])
        x = float(pl[2])
        y = float(pl[3])
        radius = float(pl[4])
        ships = float(pl[5])
        production = float(pl[6])
        gy, gx = _planet_to_grid(x, y, cfg)

        if owner == player_id:
            grid[CH["ship_count_p0"], gy, gx] += _sqrt_scale_ships(ships)
            grid[CH["production_p0"], gy, gx] = max(
                grid[CH["production_p0"], gy, gx], production
            )
        elif owner == -1:
            grid[CH["production_neutral"], gy, gx] = max(
                grid[CH["production_neutral"], gy, gx], production
            )
        else:
            grid[CH["ship_count_p1"], gy, gx] += _sqrt_scale_ships(ships)
            grid[CH["production_p1"], gy, gx] = max(
                grid[CH["production_p1"], gy, gx], production
            )

        grid[CH["planet_radius"], gy, gx] = max(
            grid[CH["planet_radius"], gy, gx], radius / 5.0
        )
        # Orbiting = initial (orbital_r + radius) < 50. Approximated here
        # via distance-to-sun < 50 - radius, which is the condition at
        # spawn; it holds for the life of the game since orbits are
        # circular. Exact threshold may drift by one cell for planets
        # sitting right on the boundary — rare; the conv learns around it.
        dist_sun = float(np.hypot(x - _SUN_POS[0], y - _SUN_POS[1]))
        if dist_sun + radius < 50.0:
            grid[CH["is_orbiting"], gy, gx] = 1.0

        if pid in comet_pids:
            grid[CH["is_comet"], gy, gx] = 1.0

        # sun_distance: normalize by board half-diagonal (~71) so it
        # fits in [0, 1] with headroom.
        grid[CH["sun_distance"], gy, gx] = dist_sun / 71.0

    # --- Fleets ---
    fleets = _obs_get(obs, "fleets", []) or []
    for fl in fleets:
        # Fleet row: [id, owner, x, y, angle, from_planet_id, ships]
        owner = int(fl[1])
        x = float(fl[2])
        y = float(fl[3])
        angle = float(fl[4])
        ships = float(fl[6])
        gy, gx = _planet_to_grid(x, y, cfg)

        if owner == player_id:
            grid[CH["ship_count_p0"], gy, gx] += _sqrt_scale_ships(ships)
        else:
            grid[CH["ship_count_p1"], gy, gx] += _sqrt_scale_ships(ships)

        # Angle components: sum into cell (multiple fleets in one cell
        # get a vector sum — conv learns to decode).
        grid[CH["fleet_angle_cos"], gy, gx] += float(np.cos(angle))
        grid[CH["fleet_angle_sin"], gy, gx] += float(np.sin(angle))

    # --- Broadcast scalar: turn_phase ---
    step = int(_obs_get(obs, "step", 0) or 0)
    grid[CH["turn_phase"], :, :] = step / _MAX_STEPS

    return grid


# ---------------------------------------------------------------------------
# Entity encoder (set_transformer candidate B).
# ---------------------------------------------------------------------------


def encode_entities(
    obs: Any,
    player_id: int,
    cfg: SetTransformerCfg | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode obs as a ``(n_max_entities, F)`` per-entity feature tensor + mask.

    Entity order: planets first, then fleets. Padding rows (is_valid=0)
    fill up to ``cfg.n_max_entities``. Order within each type is stable
    across turns (by id), so the set-transformer's attention is the
    only mechanism that establishes entity relationships — position in
    the tensor itself carries no information beyond identity.

    Args:
      obs: Kaggle obs or ``ParsedObs``.
      player_id: seat id — perspective for owner_me / owner_enemy.
      cfg: optional override.

    Returns:
      ``(features, mask)`` where features is float32
      ``(n_max_entities, F=19)`` and mask is bool ``(n_max_entities,)``.

    Raises:
      No explicit raise; if the obs has more entities than fit, the
      extras are DROPPED. A diag warning would be appropriate in W4
      training code; this skeleton stays silent.
    """
    if cfg is None:
        cfg = SetTransformerCfg()

    schema = entity_feature_schema()
    F = len(schema)
    N = cfg.n_max_entities
    offsets = feature_offsets()

    features = np.zeros((N, F), dtype=np.float32)
    mask = np.zeros((N, ), dtype=bool)

    # Global broadcast scalars computed once, applied to each valid row.
    step = int(_obs_get(obs, "step", 0) or 0)
    turn_phase = step / _MAX_STEPS

    my_ships_total = 0.0
    enemy_ships_total = 0.0

    comet_pids = set(_obs_get(obs, "comet_planet_ids", []) or [])
    planets = _obs_get(obs, "planets", []) or []
    fleets = _obs_get(obs, "fleets", []) or []
    initial_by_id = {
        int(ip[0]): ip for ip in (_obs_get(obs, "initial_planets", []) or [])
    }

    angular_velocity = float(_obs_get(obs, "angular_velocity", 0.0) or 0.0)

    # Pre-scan for ship totals (for score_diff broadcast).
    for pl in planets:
        owner = int(pl[1])
        ships = float(pl[5])
        if owner == player_id:
            my_ships_total += ships
        elif owner != -1:
            enemy_ships_total += ships
    for fl in fleets:
        owner = int(fl[1])
        ships = float(fl[6])
        if owner == player_id:
            my_ships_total += ships
        else:
            enemy_ships_total += ships
    score_diff = (my_ships_total - enemy_ships_total) / 1000.0

    cursor = 0

    # --- Planets ---
    for pl in planets:
        if cursor >= N:
            break
        pid = int(pl[0])
        owner = int(pl[1])
        x = float(pl[2])
        y = float(pl[3])
        radius = float(pl[4])
        ships = float(pl[5])
        production = float(pl[6])

        row = features[cursor]
        row[offsets["is_valid"]] = 1.0
        row[offsets["type_planet"]] = 1.0
        if owner == player_id:
            row[offsets["owner_me"]] = 1.0
        elif owner == -1:
            row[offsets["owner_neutral"]] = 1.0
        else:
            row[offsets["owner_enemy"]] = 1.0
        row[offsets["pos_x"]] = x
        row[offsets["pos_y"]] = y

        # is_orbiting: same check as the grid encoder for consistency.
        dist_sun = float(np.hypot(x - _SUN_POS[0], y - _SUN_POS[1]))
        if dist_sun + radius < 50.0:
            row[offsets["is_orbiting"]] = 1.0
            # Only orbiting planets have a nonzero angular velocity;
            # non-orbiters are fixed.
            row[offsets["orbital_angular_vel"]] = angular_velocity

        row[offsets["ships"]] = _sqrt_scale_ships(ships)
        row[offsets["production"]] = production
        row[offsets["radius"]] = radius / 5.0
        row[offsets["sun_distance"]] = dist_sun / 71.0

        # Globals.
        row[offsets["turn_phase"]] = turn_phase
        row[offsets["score_diff"]] = score_diff

        if pid in comet_pids:
            # Tag comet flag via type_comet; planet flag still on (comets
            # are planet-backed in the engine). Models can disambiguate.
            row[offsets["type_comet"]] = 1.0

        mask[cursor] = True
        cursor += 1

    # --- Fleets ---
    for fl in fleets:
        if cursor >= N:
            break
        owner = int(fl[1])
        x = float(fl[2])
        y = float(fl[3])
        angle = float(fl[4])
        ships = float(fl[6])

        row = features[cursor]
        row[offsets["is_valid"]] = 1.0
        row[offsets["type_fleet"]] = 1.0
        if owner == player_id:
            row[offsets["owner_me"]] = 1.0
        else:
            row[offsets["owner_enemy"]] = 1.0
        row[offsets["pos_x"]] = x
        row[offsets["pos_y"]] = y

        # Fleet speed depends on ship count (game formula). We store
        # raw vx, vy derived from (angle, speed) so the model sees
        # velocity directly without having to re-derive it.
        # Speed formula matches orbit_wars.py.
        speed = 1.0 + 5.0 * (np.log(max(ships, 1.0)) / np.log(1000.0)) ** 1.5
        row[offsets["velocity_x"]] = float(np.cos(angle) * speed)
        row[offsets["velocity_y"]] = float(np.sin(angle) * speed)

        row[offsets["ships"]] = _sqrt_scale_ships(ships)
        # production / radius / is_orbiting / angular_vel stay 0 for fleets.
        row[offsets["sun_distance"]] = float(
            np.hypot(x - _SUN_POS[0], y - _SUN_POS[1])
        ) / 71.0

        row[offsets["turn_phase"]] = turn_phase
        row[offsets["score_diff"]] = score_diff

        mask[cursor] = True
        cursor += 1

    return features, mask


# ---------------------------------------------------------------------------
# Convenience helper used by both encoders + callers that want a single
# "owned planet" list for decoding the policy output.
# ---------------------------------------------------------------------------


def owned_planet_positions(
    obs: Any, player_id: int
) -> list[Tuple[int, float, float]]:
    """Return ``[(planet_id, x, y), ...]`` for planets owned by ``player_id``.

    Both decoders (conv grid indexing by planet_to_grid; set-transformer
    row indexing by (type_planet & owner_me)) need to know *which*
    planets we can launch from. This helper keeps the decode logic
    aligned with the encode logic — if we change how "owned planet" is
    determined (e.g. if we introduce allied play in a future mode),
    we change it once here.

    The list preserves the engine's planet order (by id), so training
    labels (policy target = visit distributions at each owned planet)
    and inference output align.
    """
    out: list[Tuple[int, float, float]] = []
    planets = _obs_get(obs, "planets", []) or []
    for pl in planets:
        if int(pl[1]) == player_id:
            out.append((int(pl[0]), float(pl[2]), float(pl[3])))
    return out



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



# --- inlined: orbitwars/nn/nn_value.py ---

"""NN value-head bridge: ConvPolicy.value -> scalar leaf evaluation.

Mirrors ``nn_prior.py`` for the value head. Where ``move_prior_fn``
re-weights PUCT exploration via NN policy logits, ``value_fn``
replaces the ``_rollout_value`` heuristic-rollout with a single NN
forward pass that returns the value head's estimate of state-value
for ``my_player``.

Why this exists (see ``docs/NN_VALUE_HEAD_DESIGN.md``):

* ``move_prior_fn`` alone CANNOT change the wire action under
  anchor-locked play with heuristic rollouts. The Q values come from
  rollouts using HeuristicAgent on both sides → all candidates'
  Q ≈ "how the heuristic would play this from here," and the
  heuristic anchor wins on Q nearly every time.
* ``value_fn`` lets the NN drive the Q estimates directly. With a
  trained value head, Q reflects the NN's strategy assessment, not
  the heuristic's. This is the "value head as Q estimator" path
  (AlphaZero leaf evaluation, MuZero terminal value).

Status (2026-04-26): the BC v1 small checkpoint's value head was
NEVER TRAINED (``bc_warmstart.py`` does ``logits, _value = model(x)``
and discards _value). Plugging this module's ``make_nn_value_fn``
into MCTS today would feed garbage to the search — the value head
outputs ~uniform [-0.07, 0] noise for any input. Phase 2 of the
design doc covers training a useful value head.

This module is the Phase 1 deliverable: the *pathway* from value
head to MCTS leaf eval. Smoke-testable with constant or random
value_fn to verify wire actions diverge from a heuristic-rollout
baseline.

Convention: value is a scalar in [-1, 1] from ``my_player``'s
perspective. +1 = win, -1 = loss. Matches the ``_rollout_value``
sign convention so the two pathways are interchangeable in the
search.
"""

from typing import Any, Callable, Optional

import numpy as np
import torch



# Public type: caller supplies (obs, my_player) and gets a scalar.
ValueFn = Callable[[Any, int], float]


def make_nn_value_fn(
    model: ConvPolicy,
    cfg: ConvPolicyCfg,
    *,
    clip: float = 1.0,
) -> ValueFn:
    """Build a value_fn closure over a loaded ConvPolicy.

    Args:
      model: a ``ConvPolicy`` checkpoint with both heads. The policy
        head's logits are ignored here; only the value head's scalar
        output is used.
      cfg: the matching ``ConvPolicyCfg`` used for grid dimensions.
      clip: max absolute value to return. The value head is in
        principle in [-1, 1] but training instability or distribution
        shift can produce out-of-range values; clipping keeps the
        downstream MCTS Q-aggregation well-behaved.

    Returns:
      A callable ``fn(obs, my_player: int) -> float`` that returns
      the NN's estimate of state-value for ``my_player``. Errors
      (encoding failures, NaN outputs) return 0.0 — neutral leaf
      value, search will still anchor-lock on the heuristic so a
      defective value_fn cannot forfeit a turn.
    """
    model.eval()  # idempotent; ensure dropout/BN are in eval mode

    def fn(obs: Any, my_player: int) -> float:
        try:
            grid = encode_grid(obs, my_player, cfg)
            x = torch.from_numpy(grid).unsqueeze(0)  # (1, C, H, W)
            with torch.no_grad():
                _logits, value = model(x)  # value: (1, 1) by ConvPolicy contract
            v = float(value.squeeze().item())
            if not np.isfinite(v):
                return 0.0
            # Clamp to [-clip, clip].
            if v > clip:
                return clip
            if v < -clip:
                return -clip
            return v
        except Exception:
            return 0.0

    return fn


def make_constant_value_fn(value: float = 0.0) -> ValueFn:
    """Diagnostic: return the same scalar for every state.

    Used by smoke tests to verify the value_fn pathway is wired up
    correctly. With ``value=0.0``, all candidates' Q estimates collapse
    to ~0; anchor-lock should make the heuristic anchor win on tie-
    breaking. This produces wire actions identical to a no-NN
    heuristic-rollout-only run if the pathway is wired correctly.
    """
    def fn(obs: Any, my_player: int) -> float:
        return float(value)
    return fn


def make_random_value_fn(seed: int = 0) -> ValueFn:
    """Diagnostic: per-call random value in [-1, 1].

    Used to confirm the value_fn actually steers Q estimates: wire
    actions under this fn MUST differ from a constant-value run.
    """
    import random as _r
    rng = _r.Random(seed)
    def fn(obs: Any, my_player: int) -> float:
        return rng.uniform(-1.0, 1.0)
    return fn



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
        value_fn: Optional[Any] = None,
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
        # 2026-04-28 DIAGNOSTIC: under Phantom 4.0 bug, this line silently
        # had no effect at search time (tight_cfg dropped the field). With
        # the fix propagating it, decoupled actually fires on Kaggle and
        # something there errors. Disabling unconditional True until we
        # can isolate the decoupled-path Kaggle issue. (v22-v27 ran with
        # decoupled effectively False at search time; was working fine.)
        # NOTE: caller can still set use_decoupled_sim_move=True via
        # the gumbel_cfg arg if desired.
        self._fallback = HeuristicAgent(weights=self.weights)
        self._search = GumbelRootSearch(
            weights=self.weights,
            action_cfg=self.action_cfg,
            gumbel_cfg=self.gumbel_cfg,
            rng_seed=rng_seed,
            move_prior_fn=move_prior_fn,
            value_fn=value_fn,
        )
        self._use_opponent_model = use_opponent_model
        # Posterior is created lazily on turn 0 so per-match state
        # resets come free with the existing turn-0 reset path below.
        self.opp_posterior: Optional[ArchetypePosterior] = None
        # External-observability handle: the most recent SearchResult
        # produced by act() (or None if act() returned a fallback). Used
        # by tools/collect_mcts_demos.py to read per-candidate visits
        # for AlphaZero-style distillation BC. Pure observability — the
        # act() hot path does not consume this.
        self.last_search_result: Optional[Any] = None

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
            # PHANTOM 5.0 FIX (2026-04-28): the previous rebuild on
            # fresh_game CONSTRUCTED a new GumbelRootSearch without
            # threading ``move_prior_fn`` or ``value_fn`` from the old
            # one. Both fields default to None on the dataclass, so the
            # NN prior + NN value head were silently DISABLED at the
            # start of every match — even when the agent was constructed
            # with both. This is the second Phantom-class bug: an
            # internal rebuild quietly drops configured behavior. The
            # impact mirrored Phantom 4.0 — every nn_value bundle
            # actually ran heuristic rollouts under the
            # ``rollout_policy='nn_value' but no value_fn supplied''
            # fallback, with no warning visible to the bundle author
            # because warnings dedupe by source location and the same
            # warn line fires once per process.
            self._search = GumbelRootSearch(
                weights=self.weights,
                action_cfg=self.action_cfg,
                gumbel_cfg=self.gumbel_cfg,
                rng_seed=None,  # fresh RNG; deterministic only if seeded at ctor.
                move_prior_fn=self._search.move_prior_fn,
                value_fn=self._search.value_fn,
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
            # Clear stale search result from the previous match.
            self.last_search_result = None

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

        # Rebuild a one-shot config with the tightened deadline. ALL other
        # fields must be preserved so the safety floor still protects us
        # under the tight budget AND so that bundle-time config overrides
        # (rollout_policy, sim_move_variant, exp3_eta, use_macros,
        # use_decoupled_sim_move, etc.) actually reach the search.
        #
        # PHANTOM 4.0 FIX (2026-04-27): the previous version of this
        # construction omitted rollout_policy, sim_move_variant, exp3_eta,
        # use_decoupled_sim_move, num_opp_candidates, per_rollout_budget_ms,
        # and use_macros — silently reverting them to defaults. Every
        # `--rollout-policy nn_value` / `--sim-move-variant exp3` / etc.
        # bundle since the introduction of these flags has been running
        # with the GumbelConfig defaults instead. Confirmed via
        # diagnostic: nn_value bundle never invoked _value_fn_eval; rollout
        # cost matched HeuristicAgent rollouts, not NN value forward.
        # This bug explains the universal "+51.8 Elo H2H phantom" — all
        # bundles produced byte-identical wire actions because their
        # config overrides were being dropped at search time.
        tight_cfg = GumbelConfig(
            num_candidates=self.gumbel_cfg.num_candidates,
            total_sims=self.gumbel_cfg.total_sims,
            rollout_depth=self.gumbel_cfg.rollout_depth,
            hard_deadline_ms=safe_budget,
            anchor_improvement_margin=self.gumbel_cfg.anchor_improvement_margin,
            rollout_policy=self.gumbel_cfg.rollout_policy,
            use_decoupled_sim_move=self.gumbel_cfg.use_decoupled_sim_move,
            sim_move_variant=self.gumbel_cfg.sim_move_variant,
            exp3_eta=self.gumbel_cfg.exp3_eta,
            num_opp_candidates=self.gumbel_cfg.num_opp_candidates,
            per_rollout_budget_ms=self.gumbel_cfg.per_rollout_budget_ms,
            use_macros=self.gumbel_cfg.use_macros,
            value_mix_alpha=self.gumbel_cfg.value_mix_alpha,
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
            self.last_search_result = None
            return heuristic_move

        # Stash the SearchResult so external tooling (e.g.
        # `tools/collect_mcts_demos.py`) can read the per-candidate
        # visit counts without re-running search. The attribute is
        # NOT used by the act() hot path itself — pure observability.
        self.last_search_result = result
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


# --- NN prior bootstrap (--nn-checkpoint (base64 inline)) ---
import base64 as _bundle_b64
import io as _bundle_io
import torch as _bundle_torch
_BUNDLE_BC_CKPT_B64 = (
    'UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAPABMAYXpfdjM3L2RhdGEucGtsRkIPAFpaWlpaWlpa'
    'WlpaWlpaWoACfXEAKFgLAAAAbW9kZWxfc3RhdGVxAX1xAihYCwAAAHN0ZW0ud2VpZ2h0cQNjdG9y'
    'Y2guX3V0aWxzCl9yZWJ1aWxkX3RlbnNvcl92MgpxBCgoWAcAAABzdG9yYWdlcQVjdG9yY2gKRmxv'
    'YXRTdG9yYWdlCnEGWAEAAAAwcQdYAwAAAGNwdXEITYANdHEJUUsAKEsgSwxLA0sDdHEKKEtsSwlL'
    'A0sBdHELiWNjb2xsZWN0aW9ucwpPcmRlcmVkRGljdApxDClScQ10cQ5ScQ9YCQAAAHN0ZW0uYmlh'
    'c3EQaAQoKGgFaAZYAQAAADFxEWgISyB0cRJRSwBLIIVxE0sBhXEUiWgMKVJxFXRxFlJxF1gTAAAA'
    'YmxvY2tzLjAuZ24xLndlaWdodHEYaAQoKGgFaAZYAQAAADJxGWgISyB0cRpRSwBLIIVxG0sBhXEc'
    'iWgMKVJxHXRxHlJxH1gRAAAAYmxvY2tzLjAuZ24xLmJpYXNxIGgEKChoBWgGWAEAAAAzcSFoCEsg'
    'dHEiUUsASyCFcSNLAYVxJIloDClScSV0cSZScSdYFQAAAGJsb2Nrcy4wLmNvbnYxLndlaWdodHEo'
    'aAQoKGgFaAZYAQAAADRxKWgITQAkdHEqUUsAKEsgSyBLA0sDdHErKE0gAUsJSwNLAXRxLIloDClS'
    'cS10cS5ScS9YEwAAAGJsb2Nrcy4wLmNvbnYxLmJpYXNxMGgEKChoBWgGWAEAAAA1cTFoCEsgdHEy'
    'UUsASyCFcTNLAYVxNIloDClScTV0cTZScTdYEwAAAGJsb2Nrcy4wLmduMi53ZWlnaHRxOGgEKCho'
    'BWgGWAEAAAA2cTloCEsgdHE6UUsASyCFcTtLAYVxPIloDClScT10cT5ScT9YEQAAAGJsb2Nrcy4w'
    'LmduMi5iaWFzcUBoBCgoaAVoBlgBAAAAN3FBaAhLIHRxQlFLAEsghXFDSwGFcUSJaAwpUnFFdHFG'
    'UnFHWBUAAABibG9ja3MuMC5jb252Mi53ZWlnaHRxSGgEKChoBWgGWAEAAAA4cUloCE0AJHRxSlFL'
    'AChLIEsgSwNLA3RxSyhNIAFLCUsDSwF0cUyJaAwpUnFNdHFOUnFPWBMAAABibG9ja3MuMC5jb252'
    'Mi5iaWFzcVBoBCgoaAVoBlgBAAAAOXFRaAhLIHRxUlFLAEsghXFTSwGFcVSJaAwpUnFVdHFWUnFX'
    'WBMAAABibG9ja3MuMS5nbjEud2VpZ2h0cVhoBCgoaAVoBlgCAAAAMTBxWWgISyB0cVpRSwBLIIVx'
    'W0sBhXFciWgMKVJxXXRxXlJxX1gRAAAAYmxvY2tzLjEuZ24xLmJpYXNxYGgEKChoBWgGWAIAAAAx'
    'MXFhaAhLIHRxYlFLAEsghXFjSwGFcWSJaAwpUnFldHFmUnFnWBUAAABibG9ja3MuMS5jb252MS53'
    'ZWlnaHRxaGgEKChoBWgGWAIAAAAxMnFpaAhNACR0cWpRSwAoSyBLIEsDSwN0cWsoTSABSwlLA0sB'
    'dHFsiWgMKVJxbXRxblJxb1gTAAAAYmxvY2tzLjEuY29udjEuYmlhc3FwaAQoKGgFaAZYAgAAADEz'
    'cXFoCEsgdHFyUUsASyCFcXNLAYVxdIloDClScXV0cXZScXdYEwAAAGJsb2Nrcy4xLmduMi53ZWln'
    'aHRxeGgEKChoBWgGWAIAAAAxNHF5aAhLIHRxelFLAEsghXF7SwGFcXyJaAwpUnF9dHF+UnF/WBEA'
    'AABibG9ja3MuMS5nbjIuYmlhc3GAaAQoKGgFaAZYAgAAADE1cYFoCEsgdHGCUUsASyCFcYNLAYVx'
    'hIloDClScYV0cYZScYdYFQAAAGJsb2Nrcy4xLmNvbnYyLndlaWdodHGIaAQoKGgFaAZYAgAAADE2'
    'cYloCE0AJHRxilFLAChLIEsgSwNLA3RxiyhNIAFLCUsDSwF0cYyJaAwpUnGNdHGOUnGPWBMAAABi'
    'bG9ja3MuMS5jb252Mi5iaWFzcZBoBCgoaAVoBlgCAAAAMTdxkWgISyB0cZJRSwBLIIVxk0sBhXGU'
    'iWgMKVJxlXRxllJxl1gTAAAAYmxvY2tzLjIuZ24xLndlaWdodHGYaAQoKGgFaAZYAgAAADE4cZlo'
    'CEsgdHGaUUsASyCFcZtLAYVxnIloDClScZ10cZ5ScZ9YEQAAAGJsb2Nrcy4yLmduMS5iaWFzcaBo'
    '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'
    'iUc/3Mscj/1vlkc/3IE5U8ywnkc/3Cq+VF8tvkc/28Gn4S6Qqkc/24qR+Hqv6Ec/22aGhqsHbGVY'
    'BgAAAHZhbF9jZXIaAQAAXXIbAQAAKEdAANDdsNq/Z0dAAKkoZ+bqDUdAAJ8Ln9vgZEdAAJrbXOFF'
    'HEdAAJiSjyVAmUdAAJcq7Ea95kdAAJZdV4CQfkdAAJXMRFvQVkdAAJWIHIj32UdAAJUoet+Dk0dA'
    'AJUuHtL4mkdAAJTuTYPTWUdAAJTXwRTyVEdAAJTdVzhRR0dAAJTY0GJWC2VYBwAAAHZhbF9tc2Vy'
    'HAEAAF1yHQEAAChHP+S6Rf09leNHP+FbrE/19zhHP+C9dQnMkwFHP99bHig7M81HP9/rla+xu4RH'
    'P965gL/T2V5HP958rt2x+7xHP91u2cCgjHtHP90Do2kX9PZHP9zPJogXFDJHP9yyIizPNU9HP9wJ'
    'HUEY9ddHP9wVcOU8lQJHP9wzKTiVgtJHP9usHqRr3mNlWAwAAAB3YWxsX3NlY29uZHNyHgEAAF1y'
    'HwEAAChHQBU+aP7IAABHQCQa3mjXAABHQC2cVtXQAABHQDOH/MaQAABHQDg2UJpMgABHQDzdseug'
    'AABHQEDDXV2pQABHQEL/6sKBQABHQEU3d8iOgABHQEdrVc5owABHQEmgsiL9gABHQEwA2tJJQABH'
    'QE5VBmV1AABHQFBbWRzR4ABHQFGJLClTwABldVgNAAAAdHJhaW5pbmdfYXJnc3IgAQAAfXIhAQAA'
    'KFgFAAAAZGVtb3NyIgEAAGNwYXRobGliLl9sb2NhbApXaW5kb3dzUGF0aApyIwEAAFg7AAAALi4v'
    'Li4vLi4vcnVucy9jbG9zZWRfbG9vcF9pdGVyMV9wb3N0Zml4L2RlbW9zX2l0ZXIxX2JpZy5ucHpy'
    'JAEAAIVyJQEAAFJyJgEAAFgNAAAAYmNfY2hlY2twb2ludHInAQAAaiMBAABYJwAAAC4uLy4uLy4u'
    'L3J1bnMvYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS5wdHIoAQAAhXIpAQAAUnIqAQAAWAMAAABvdXRy'
    'KwEAAGojAQAAWA4AAABydW5zL2F6X3YzNy5wdHIsAQAAhXItAQAAUnIuAQAAWAYAAABlcG9jaHNy'
    'LwEAAEsPWAoAAABiYXRjaF9zaXplcjABAABNAAFYAgAAAGxycjEBAABHP1BiTdLxqfxYCAAAAHZh'
    'bF9mcmFjcjIBAABHP7mZmZmZmZpYBgAAAGRldmljZXIzAQAAWAQAAABjdWRhcjQBAABYBAAAAHNl'
    'ZWRyNQEAAEsAWAgAAABsYW1iZGFfcHI2AQAARz/wAAAAAAAAWAgAAABsYW1iZGFfdnI3AQAARz/w'
    'AAAAAAAAWAkAAABsYW1iZGFfbDJyOAEAAEc/Gjbi6xxDLVgKAAAAcG9saWN5X3RhdXI5AQAARz/4'
    'AAAAAAAAWAoAAABwb2xpY3lfZXBzcjoBAABHP+AAAAAAAABYHQAAAHVuZnJlZXplX2JhY2tib25l'
    'X2FmdGVyX2Vwb2NocjsBAABK/////3VYFAAAAHNvdXJjZV9iY19jaGVja3BvaW50cjwBAABYJwAA'
    'AC4uXC4uXC4uXHJ1bnNcYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS5wdHI9AQAAWAwAAABzb3VyY2Vf'
    'ZGVtb3NyPgEAAFg7AAAALi5cLi5cLi5ccnVuc1xjbG9zZWRfbG9vcF9pdGVyMV9wb3N0Zml4XGRl'
    'bW9zX2l0ZXIxX2JpZy5ucHpyPwEAAHUuUEsHCOXiI5xTEgAAUxIAAFBLAwQAAAgIAAAAAAAAAAAA'
    'AAAAAAAAAAAAEAAvAGF6X3YzNy9ieXRlb3JkZXJGQisAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWmxpdHRsZVBLBwiFPeMZBgAAAAYAAABQSwMEAAAICAAAAAAAAAAA'
    'AAAAAAAAAAAAAA0APwBhel92MzcvZGF0YS8wRkI7AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaPJ96vOTtzzzQsxC+tYZfvSf1mL1uJ5M9'
    'di/1u511mT3pR9A8x3/OPE+/xrzuv4U8YRJVvankk72gfFG9Jx/SPDdbdz1D5ok90CS1vUXYab0j'
    '2ws9mjKLPeu5v7zb+4Y9gBuAvDwMhLwFZKc92nCsvWSPfb2IOKi7VBuLvBulrj2vM2y9nYRUvefu'
    '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'
)
_bundle_ckpt_bytes = _bundle_b64.b64decode("".join(_BUNDLE_BC_CKPT_B64))
_bundle_ckpt = _bundle_torch.load(
    _bundle_io.BytesIO(_bundle_ckpt_bytes),
    map_location="cpu", weights_only=False,
)
# Decode any quantized weights back to fp32 so the fp32
# ConvPolicy module accepts the state_dict cleanly. fp16 halves
# bundle size; int8_per_tensor_symmetric quarters it. Inference
# precision is fp32 either way.
def _bundle_upcast(sd, scales=None):
    out = {}
    for k, v in sd.items():
        if v.dtype == torch.int8 and scales is not None and k in scales:
            out[k] = v.float() * float(scales[k])
        elif hasattr(v, 'is_floating_point') and v.is_floating_point():
            out[k] = v.float()
        else:
            out[k] = v
    return out
_bundle_scales = _bundle_ckpt.get('_quant_scales')
if 'model_state' in _bundle_ckpt and 'cfg' in _bundle_ckpt:
    _bundle_cfg_nn = ConvPolicyCfg(**_bundle_ckpt['cfg'])
    _bundle_model = ConvPolicy(_bundle_cfg_nn)
    _bundle_model.load_state_dict(_bundle_upcast(_bundle_ckpt['model_state'], _bundle_scales))
elif 'model_state_dict' in _bundle_ckpt:
    _bundle_cfg_nn = ConvPolicyCfg()
    _bundle_model = ConvPolicy(_bundle_cfg_nn)
    _bundle_model.load_state_dict(_bundle_upcast(_bundle_ckpt['model_state_dict']))
else:
    raise RuntimeError('bundle: NN checkpoint has unrecognized keys')
_bundle_model.eval()
_bundle_move_prior_fn = make_nn_prior_fn(
    _bundle_model, _bundle_cfg_nn,
    hold_neutral_prob=0.05, temperature=1.0,
)
# Build value_fn from the same model. The value head is only
# used when GumbelConfig.rollout_policy='nn_value'; building
# the closure unconditionally costs ~0 bytes (just a closure)
# and lets the same bundle support both rollout modes.
# (make_nn_value_fn is inlined from nn.nn_value above.)
_bundle_value_fn = make_nn_value_fn(_bundle_model, _bundle_cfg_nn)
del _bundle_ckpt  # free RAM after model is built


# --- GumbelConfig / MCTSAgent overrides ---

# Applied by tools/bundle.py at build time.

_bundle_cfg = GumbelConfig()

_bundle_cfg.sim_move_variant = 'exp3'

_bundle_cfg.exp3_eta = 0.3

_bundle_cfg.rollout_policy = 'nn_value'

_bundle_cfg.anchor_improvement_margin = 0.5

_bundle_cfg.total_sims = 64

_bundle_cfg.num_candidates = 4

_bundle_cfg.value_mix_alpha = 0.5

_bundle_cfg.hard_deadline_ms = 850.0


# --- agent entry point ---

agent = MCTSAgent(gumbel_cfg=_bundle_cfg, rng_seed=0, move_prior_fn=_bundle_move_prior_fn, value_fn=_bundle_value_fn).as_kaggle_agent()
