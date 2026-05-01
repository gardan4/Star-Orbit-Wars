# Auto-generated Orbit Wars submission. Do not edit by hand.
# Built by tools/bundle.py on 2026-04-30 16:52:10.
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
    # NN-greedy rollout policy factory. When set AND
    # ``gumbel_cfg.rollout_policy == "nn"``, ``_opp_factory`` and
    # ``_my_future_factory`` return fresh ``NNRolloutAgent`` instances
    # so MCTS rollouts play NN-vs-NN instead of heuristic-vs-heuristic.
    # Q estimates then reflect NN strategy, not heuristic strategy —
    # the structural unlock for NN-on-wire (see
    # docs/NN_DRIVEN_ROLLOUTS_SPEC.md). Signature:
    #   ``() -> Agent``  (zero-arg factory; creates a stateless agent)
    nn_rollout_factory: Optional[Callable[[], Any]] = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.rng_seed)

    def _opp_factory(self) -> Any:
        # Priority 1: Bayesian posterior override (Path D). When the
        # posterior has concentrated on a specific archetype, MCTSAgent
        # sets this so rollouts play against that archetype's heuristic.
        # Keep this path even under rollout_policy="fast"/"nn" —
        # exploitation signal beats raw rollout speed once the posterior
        # has fired.
        if self.opp_policy_override is not None:
            return self.opp_policy_override()
        # Priority 2: NN-greedy rollout policy. Argmax over NN logits
        # per planet — Q estimates reflect NN strategy. See
        # docs/NN_DRIVEN_ROLLOUTS_SPEC.md.
        if (
            self.gumbel_cfg.rollout_policy == "nn"
            and self.nn_rollout_factory is not None
        ):
            return self.nn_rollout_factory()
        # Priority 3: fast rollout policy. Cheap nearest-target push —
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
        # cheap agent when rollout_policy="fast" / "nn". Candidate turn-0
        # action is unaffected (that's already built upstream).
        if (
            self.gumbel_cfg.rollout_policy == "nn"
            and self.nn_rollout_factory is not None
        ):
            return self.nn_rollout_factory()
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



# --- inlined: orbitwars/bots/nn_rollout.py ---

"""NN-greedy rollout policy.

The structural reason every NN-as-leaf experiment (v33-v36) lost to
v32b's heuristic rollouts: MCTS rollouts use ``HeuristicAgent`` on
both sides, so Q estimates measure "how the heuristic plays from
here" — exactly what the heuristic anchor already represents. Search
has no information that disagrees with the anchor; the override rate
stays at 9.2%.

This file provides ``NNRolloutAgent`` — a rollout policy that uses
the NN's policy logits directly to pick moves. When MCTS rollouts use
this agent on both sides, Q estimates measure "how the NN plays from
here". That's genuinely different from the heuristic anchor, so search
gets meaningful disagreement signal.

Cost per ``act()``: ~1-2 ms — between fast_rollout (~0.02 ms) and full
heuristic (~4-5 ms). At 850 ms search budget, ~30-50 sims/turn vs
heuristic's 12-16. The quality is hopefully better than fast_rollout
(which lost -190 Elo in the v35a A/B) because it draws on a trained
policy head rather than nearest-target geometry.

Invariants (mirrors fast_rollout.py):
  * Only my planets launch.
  * ``ships <= planet.ships`` always.
  * Angle is finite.
  * Falls back to no-op if NN forward fails (defensive — must never
    forfeit a turn).

This is consumed by ``GumbelRootSearch`` when
``GumbelConfig.rollout_policy == "nn"``. The root anchor is still
provided by ``HeuristicAgent``; only rollout plies swap in this agent.
"""

import math
from typing import Any, Optional

import numpy as np



# Pre-compute angle bucket centers (in radians, [0, 2π)) for the 4
# canonical directions used by ACTION_LOOKUP. East=0, North=π/2,
# West=π, South=3π/2.
_BUCKET_ANGLES = (
    0.0,
    0.5 * math.pi,
    math.pi,
    1.5 * math.pi,
)


class NNRolloutAgent(Agent):
    """NN-greedy per-planet rollout policy.

    Reads the trained ConvPolicy's logits at each owned-planet's grid
    cell and selects the argmax-channel action per planet. Channels
    correspond to (angle_bucket, ship_fraction) per ACTION_LOOKUP.

    Min-launch + send-fraction constraints mirror fast_rollout so the
    actions remain valid under the engine's combat math regardless of
    NN output quality.

    Attributes:
        model: a loaded ``ConvPolicy``. ``model.eval()`` is enforced
            once at construction.
        cfg: matching ``ConvPolicyCfg``.
        min_launch_size: don't launch a fleet smaller than this many
            ships. Matches HeuristicAgent's default to avoid dribbles.
    """

    name = "nn_rollout"

    def __init__(
        self,
        model: ConvPolicy,
        cfg: ConvPolicyCfg,
        min_launch_size: int = 20,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.min_launch_size = int(min_launch_size)
        # Idempotent — but cheap enough to call every time.
        self.model.eval()

    def act(self, obs: Any, deadline: Deadline) -> Action:
        # Always stage a safe fallback first; if anything below blows
        # up we still return a valid action.
        deadline.stage(no_op())

        player = obs_get(obs, "player", 0)
        planets = obs_get(obs, "planets", [])
        if not planets:
            return no_op()

        # Forward pass once per act(). Defensive: any failure in the
        # NN path falls back to no-op (still a valid action, never
        # forfeits a turn).
        try:
            import torch
            grid = encode_grid(obs, player, self.cfg)
            x = torch.from_numpy(grid).unsqueeze(0)  # (1, C, H, W)
            with torch.no_grad():
                logits, _value = self.model(x)
            # logits: (1, 8, H, W). Drop batch dim.
            logits_np = logits[0].cpu().numpy()  # (8, H, W)
        except Exception:
            return no_op()

        moves: Action = []
        min_size = self.min_launch_size
        # Single pass over my planets — no defensive scoring, no arrival
        # table, no sun-tangent. Per-planet argmax over 8 channels.
        for p in planets:
            if p[1] != player:
                continue
            available = int(p[5])
            if available < min_size:
                continue
            mp_x = float(p[2])
            mp_y = float(p[3])
            gy, gx = planet_to_grid_coords(mp_x, mp_y, self.cfg)
            # logits shape (8, H, W); pick argmax over the 8 channels
            # at this cell.
            cell = logits_np[:, gy, gx]
            best_ch = int(np.argmax(cell))
            angle_bucket, ship_frac = ACTION_LOOKUP[best_ch]
            angle = _BUCKET_ANGLES[angle_bucket]
            ships = int(available * float(ship_frac))
            if ships < min_size:
                ships = min_size
            if ships > available:
                ships = available
            moves.append([int(p[0]), float(angle), int(ships)])

        deadline.stage(moves)
        return moves



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
        nn_rollout_factory: Optional[Any] = None,
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
            nn_rollout_factory=nn_rollout_factory,
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
                nn_rollout_factory=self._search.nn_rollout_factory,
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
    'UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAfAEMAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRh'
    'LnBrbEZCPwBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlqAAn1xAChYCwAAAG1vZGVsX3N0YXRlcQF9cQIoWAsAAABzdGVtLndlaWdo'
    'dHEDY3RvcmNoLl91dGlscwpfcmVidWlsZF90ZW5zb3JfdjIKcQQoKFgHAAAAc3RvcmFnZXEFY3Rv'
    'cmNoCkZsb2F0U3RvcmFnZQpxBlgBAAAAMHEHWAMAAABjcHVxCE2ADXRxCVFLAChLIEsMSwNLA3Rx'
    'CihLbEsJSwNLAXRxC4ljY29sbGVjdGlvbnMKT3JkZXJlZERpY3QKcQwpUnENdHEOUnEPWAkAAABz'
    'dGVtLmJpYXNxEGgEKChoBWgGWAEAAAAxcRFoCEsgdHESUUsASyCFcRNLAYVxFIloDClScRV0cRZS'
    'cRdYEwAAAGJsb2Nrcy4wLmduMS53ZWlnaHRxGGgEKChoBWgGWAEAAAAycRloCEsgdHEaUUsASyCF'
    'cRtLAYVxHIloDClScR10cR5ScR9YEQAAAGJsb2Nrcy4wLmduMS5iaWFzcSBoBCgoaAVoBlgBAAAA'
    'M3EhaAhLIHRxIlFLAEsghXEjSwGFcSSJaAwpUnEldHEmUnEnWBUAAABibG9ja3MuMC5jb252MS53'
    'ZWlnaHRxKGgEKChoBWgGWAEAAAA0cSloCE0AJHRxKlFLAChLIEsgSwNLA3RxKyhNIAFLCUsDSwF0'
    'cSyJaAwpUnEtdHEuUnEvWBMAAABibG9ja3MuMC5jb252MS5iaWFzcTBoBCgoaAVoBlgBAAAANXEx'
    'aAhLIHRxMlFLAEsghXEzSwGFcTSJaAwpUnE1dHE2UnE3WBMAAABibG9ja3MuMC5nbjIud2VpZ2h0'
    'cThoBCgoaAVoBlgBAAAANnE5aAhLIHRxOlFLAEsghXE7SwGFcTyJaAwpUnE9dHE+UnE/WBEAAABi'
    'bG9ja3MuMC5nbjIuYmlhc3FAaAQoKGgFaAZYAQAAADdxQWgISyB0cUJRSwBLIIVxQ0sBhXFEiWgM'
    'KVJxRXRxRlJxR1gVAAAAYmxvY2tzLjAuY29udjIud2VpZ2h0cUhoBCgoaAVoBlgBAAAAOHFJaAhN'
    'ACR0cUpRSwAoSyBLIEsDSwN0cUsoTSABSwlLA0sBdHFMiWgMKVJxTXRxTlJxT1gTAAAAYmxvY2tz'
    'LjAuY29udjIuYmlhc3FQaAQoKGgFaAZYAQAAADlxUWgISyB0cVJRSwBLIIVxU0sBhXFUiWgMKVJx'
    'VXRxVlJxV1gTAAAAYmxvY2tzLjEuZ24xLndlaWdodHFYaAQoKGgFaAZYAgAAADEwcVloCEsgdHFa'
    'UUsASyCFcVtLAYVxXIloDClScV10cV5ScV9YEQAAAGJsb2Nrcy4xLmduMS5iaWFzcWBoBCgoaAVo'
    'BlgCAAAAMTFxYWgISyB0cWJRSwBLIIVxY0sBhXFkiWgMKVJxZXRxZlJxZ1gVAAAAYmxvY2tzLjEu'
    'Y29udjEud2VpZ2h0cWhoBCgoaAVoBlgCAAAAMTJxaWgITQAkdHFqUUsAKEsgSyBLA0sDdHFrKE0g'
    'AUsJSwNLAXRxbIloDClScW10cW5ScW9YEwAAAGJsb2Nrcy4xLmNvbnYxLmJpYXNxcGgEKChoBWgG'
    'WAIAAAAxM3FxaAhLIHRxclFLAEsghXFzSwGFcXSJaAwpUnF1dHF2UnF3WBMAAABibG9ja3MuMS5n'
    'bjIud2VpZ2h0cXhoBCgoaAVoBlgCAAAAMTRxeWgISyB0cXpRSwBLIIVxe0sBhXF8iWgMKVJxfXRx'
    'flJxf1gRAAAAYmxvY2tzLjEuZ24yLmJpYXNxgGgEKChoBWgGWAIAAAAxNXGBaAhLIHRxglFLAEsg'
    'hXGDSwGFcYSJaAwpUnGFdHGGUnGHWBUAAABibG9ja3MuMS5jb252Mi53ZWlnaHRxiGgEKChoBWgG'
    'WAIAAAAxNnGJaAhNACR0cYpRSwAoSyBLIEsDSwN0cYsoTSABSwlLA0sBdHGMiWgMKVJxjXRxjlJx'
    'j1gTAAAAYmxvY2tzLjEuY29udjIuYmlhc3GQaAQoKGgFaAZYAgAAADE3cZFoCEsgdHGSUUsASyCF'
    'cZNLAYVxlIloDClScZV0cZZScZdYEwAAAGJsb2Nrcy4yLmduMS53ZWlnaHRxmGgEKChoBWgGWAIA'
    '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'
    '0iH/aEc/39XysG6OWUc/4HFE4rUt/0c/4FYjhhVHxUc/4F+iZoBx80c/4ImvthHjmUc/4HFE4rUt'
    '/2VYAgAAAGxychsBAABdchwBAAAoRz8zqSowVTJhRz8zLznnz7TsRz8xzVi5E7iWRz8vTFUQ6rJE'
    'Rz8p30bNbejgRz8j238XN1QpRz8br27CAX7mRz8Q1VI7B+weRz8AcTLxHNycRz7liKXs8+ewZVgM'
    'AAAAYmVzdF92YWxfYWNjch0BAABHP+CJr7YR45l1WAkAAABkZW1vX2hhc2hyHgEAAFgQAAAANDcz'
    'ZDRiZDg3ZTI4MWEwMnIfAQAAWAcAAABuX2RlbW9zciABAABN6etYDQAAAHRvcmNoX3ZlcnNpb25y'
    'IQEAAGN0b3JjaC50b3JjaF92ZXJzaW9uClRvcmNoVmVyc2lvbgpyIgEAAFgLAAAAMi42LjArY3Ux'
    'MjRyIwEAAIVyJAEAAIFyJQEAAFgOAAAAY3VkYV9hdmFpbGFibGVyJgEAAIhYBAAAAHNlZWRyJwEA'
    'AEsAWAcAAABocGFyYW1zcigBAAB9cikBAAAoWAYAAABlcG9jaHNyKgEAAEsKWAoAAABiYXRjaF9z'
    'aXplcisBAABNAAFqGwEAAEc/M6kqMFUyYVgMAAAAd2VpZ2h0X2RlY2F5ciwBAABHPxo24uscQy1Y'
    'CAAAAHZhbF9mcmFjci0BAABHP7mZmZmZmZp1dS5QSwcIdPp6L/oOAAD6DgAAUEsDBAAACAgAAAAA'
    'AAAAAAAAAAAAAAAAAAAgADgAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9ieXRlb3JkZXJGQjQAWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWmxpdHRsZVBL'
    'BwiFPeMZBgAAAAYAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB0ALwBiY193YXJtc3RhcnRf'
    'c21hbGxfY3B1L2RhdGEvMEZCKwBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaPJ96vOTtzzzQsxC+tYZfvSf1mL1uJ5M9di/1u511mT3pR9A8x3/OPE+/xrzuv4U8YRJV'
    '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'
    'vVMhuj0BZjy9G6d2PG56jD0gtsq9C1YhveBNpL3FTaE9pJXsPZqQDLy1H3K9HDGvO2QhKz11XNQ8'
    '3BsTvJUzP710IcY9e8dKveEynb0VXLa87m36PI7bTj1LTFQ9UEsHCMAW3GAANgAAADYAAFBLAwQA'
    'AAgIAAAAAAAAAAAAAAAAAAAAAAAAHQA1AGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8xRkIx'
    'AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlrzGne9jTkZ'
    'veOCB71C8U+9tgyrvTDv5jyIKuO85v/Pve1mZL1Lx0y9E2w4vbRjf73pWDq8ojzFPEZom72dLD68'
    'g6sMvX1Re71h47S7hHArvXSBQjwQb4i9/d0FPCvoqz0v2rI9T2JdvZFi7zyyvrS9wBidPdPzrr20'
    'nru9+BCtvVBLBwgi+RYvgAAAAIAAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB4ANABiY193'
    'YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMTBGQjAAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpa2EOAP3emfz+HTn8/tU6BPx/Rfj/2iH8/O+9+P8i0fz+J6YE/'
    'VpOAP6wMgj+xp34/lViAP2SQgD/FWX8/LrmDP1Cufz/HY4A/MPaBP5ZQfj8TdYA/fgt/PzDYfz8r'
    'z34/I1yAPxJ9fj8F0nw/tNV/P9Pifj/lcH4/LaiBPyX1fz9QSwcIcwMFWIAAAACAAAAAUEsDBAAA'
    'CAgAAAAAAAAAAAAAAAAAAAAAAAAeADQAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzExRkIw'
    'AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWj0SPTuYA9c6'
    'iKbrOtqykjztv0m8VFdRu8ZuRDstb5O71/bjO++JcTtcvL47i5JDOmGzhrsvsI679tPbO74Egrww'
    'IIU7AooZuzKSszoy+0w83Z8avPZwtjtfhXS70gSuOT61ATxwCiK8JqvyOm2zXbuc80e6hY1kO5bF'
    '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'
    'vPfKQL1ggEO9yqtPu9yDJT2GLD499hGOvMusP71Q9HO8Ufp9PYuO+bx0pIG87oUlPfdHZbxjsCu9'
    'NVhrvW2zybyy5ls8igTzu+rMejy0a828ANshOo6oBb0tUp+8Vr8GvDoBzDw9jUm9XFsiPdEvJjsg'
    'Fy08YAHLvEy1ozyrEBs9MJvXPFeYzTyiJoe7aAiavEoA3zuSx0y9bTCUO1tXybteRoM5maSgPW9O'
    'RT2+dCo99z8avS9pKj06uTA9jh2nPIa+Z7wEMK48Dzv2O2tQaD2g8gg93KkdPR70IbwgDiw8GdUV'
    'O6cRoDxQSwcIXjV0zACQAAAAkAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAeADQAYmNfd2Fy'
    'bXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzEzRkIwAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWmvxYj26Eme9Z6knvIc+Dr0g3/Y6k50oPfWUMr1HHkM9i1QPvdBZ'
    'YD3JkUg9LmyGvL2+NbwHD5+82qb5vCM1FrsVBjm9fUkKvYpzZrzkvQU91jlavfZ5Kj3te9O8xCgV'
    'vB9A9jyPvSa9wRYsva3R5DrQeAs9+BWAPJWHD70mI149UEsHCGr7/oWAAAAAgAAAAFBLAwQAAAgI'
    'AAAAAAAAAAAAAAAAAAAAAAAAHgA0AGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8xNEZCMABa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlrxyoI/2N2CP9EM'
    'gT/iyoM/+W+BPx8qgj8kYoE/fgyAP/J4fD9r64A/ETOEP7Fdfz/djYI/uyWAP+DIfz/jN4E/tymB'
    'PwxFgz85goM/Pb6APyr/fT+svIA/j2uBP6cVhD8AkYI/xqiEP5hPhD8HcoI/JF+BPxEKgT81/YE/'
    'gj2CP1BLBwhz/WIogAAAAIAAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB4ANABiY193YXJt'
    'c3RhcnRfc21hbGxfY3B1L2RhdGEvMTVGQjAAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    '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'
    '/ps7PV8eXTyF4Ag7hhs+Pe1ZJD0so0+9tVaBvZKUNL2zJSC5gfnqOpq9Cb2jUxc9vOthPL0UULuE'
    'PDW8+W4BPeKtjLzIeiC9LhJbvOi7rD3RVgi8WNu/vFeqpz12JFI8cv/nvFUHnD1T6tw8CTkQPV1E'
    'zbwwh6c8ncAPvfW9nTxGbjk8ddeuPI8DBj1Wjv082MbXPIlMM70hoBy9PwNrvcQupbwkGyi9nR02'
    'vPjbHzvB4Sm9V6NWPfVbPbt+p7Y8mQtnvflC1zzeJh+9A0dcu04lET3CnCU9xuSIvevA4LrvrTG9'
    'l68kPCQtrrx/0lg9shg3vWreJjq8Kyc7pjAZvZLZUTvkO2G9qtREvWZzRL3cG20892nAvGn87rtI'
    'ZNU8FePiPODEzDr/aBw8j8trPQ2DJL2FKU89KYzFvAqIwb2i1bW751/6vNe1Kb0IBha9tosuvCe1'
    'Szwjxta7iqC5vHBlxT1e3TU92qMRvetxer0XAZi8NdeHPP8PQD3LcQI9vRnYPPn9EDzQ7Su9tNHQ'
    'PD2WSDw1BOY8G2AKvecLU735ibw8PIVHvLtQfLy6u4Y9Q0OGPAuZhT1OOD89Yh0gvRCuXz0+4Gg9'
    'puhJvZtlCjxWhhc9l/91PN+PubxYLTm98QlGvXT67zxV6Vy9Rl6XvLRVCr05QCy98vsPPZRkjL1J'
    'z7K6m0SnPGwqKr2zusy8/2z8vGeXOz1I+p08ucB+PHZkrLyQQSW9QN8gPWzM2jzy6WM9hWheOyCl'
    'OzyquJ47DRgKvcJcybyf5pg9fEMOPPuVVbucVSy9Ln6TPbXdp7w1AwQ8n9vzujJXhD3bi2A9U9sE'
    'PcTusrxRXwq95TipvPed+DxQOpA8MxqCvWa3+Lz+ude8WezBu2i2WT3IEBy5wq5EPVtrOLwz3CM9'
    'jr9vPaGMXj2NOKi89hkaPeTagDxt3168WTsdvJHpGj1uS6I8SepCPd7RXT01uWQ8pnQEPcPurjxo'
    'zoE9o+WAvKiSAT3UDAq8k30OvVTjpjxfu/O8yvK/vHo7LjwtO3+82OMxPWbnn7zkdDG9rZfKO1IT'
    '7byHlzM8XvU+vTThQr1aXfk8ake+PPW7sjwiywa9ZApnPU1vSDxGOq48aXc+vYyA8ztjMbq8TDFQ'
    'vfbUPj3I5vs6A4RDvcpig7pcRai8c8HmvfjMAjpRfHe8BEcRPbqklLwlHwa9rwN9vPE9urv1FoC9'
    '+05/vHiMcr3r+S29rQJzvMzROT3kbR29AuksvYCTUr0Nki29Xz8kvJRYXb1AP448QPN7vVUQb71w'
    '1mO9ws3TvAQLFz0XWYg9QVPnu5v+MD1iQnW9BHAyvadH8DvxeHW8f7Qwvbk2jjwN9Us874qCPf6v'
    'Ej11j6M8ZV8AvW9qnDqH+Ay9igcsOzC/fzrWBMC8AoQOvXMp7jx/Egc9fauMO9bbr7ySNUA8YA4f'
    'PTiwkDzZzFw8/Iy2O3OacTwqtQg8/iyDvDCCDL1WlM48d/sGPaRqYL3V9a88izr/vOPtRz1eX9E5'
    '1F8tvHZHr7wfGos89u/GusAGabxTpDw9ty/gvIm5CL0u52o8iBI/PCQTMr25o4w83B8gvc7Zsbyu'
    'iCe9JKSqvPnOuzsD4sO8aPURvRoTfryPguw8tF9HPLC0yL0f36O9VIgwPcU2iDzYZAQ90wSovBcm'
    'Wr0fizS9akZgPLnIR71thQu9lt1evUAVwjyARkQ6KM1CvR7ntLthkuC8n9eqvO1+9bzpRle8gNOn'
    'vLSxwrzpb568LNRbvXkewzwS4OW8k6cTPfoUGD3n+vW85Kb4PGkKOzuumla8MkPoPHFlFj0HCI+7'
    'pvliPR4OEr0aWTE9MmA8PTomOj0BwGy7UKiBPY2odj2Vcui8RRmIPcsKdr3ouqM9eS8evVl2TLoT'
    'fTO91KREPYjz1rzv7M682DAwvBvSXb24xgq8qGlpvKUEVb3so9y8xFsgvYnVaj139vU85zenulow'
    '1Tyqd7E8H1QVPNWqDL1mTS+9wQ4NvWhvNT0eVt07O8RYvb0qGj099iA9Z2tVugK4Uz3W9zm9k7sG'
    'vWaW9zymF1U9sRYBPJSxa70Xkeu81K/ivEazS71Y/pq8vtGzvOHzgDxp+kC8/y9BPb36SL0kZuA7'
    'IJZLPectM7xAMWq9WQWLO4AqBT2ypjG9mXkiPXShIrwO7ia9VanSO3lMHr0cnUc9dlo7vTtdBb34'
    'N8M76nkSPEiMD70TcAW94so8vdZowzy65wo9FtX3vHgZ27vJvFW9HzwfPXbL47xjtZG99wlbPKwS'
    '3jx12Ns7MxplvZxd47yskd08X22FubQfl7vMC2c9axykPQy0nzybRp09NiV5PVIO+bwvB7m7PE2q'
    'PXfkGr04t2U98LA1vbomTz0F2iQ97YE0vUbpEr182QG9pZIsPUjlOrxH1os8hZ2FvKuOi73BuFA9'
    'ITqpvPQkKLzDCEY7GCg9PRgnLDtptCa8r9GIPWByGj1ReI28+PcePSpuAr325za9orgdPSPdvLxj'
    'V0a8lvFtvYbiCL1H7jm9eDpHO9GoZr36bO88KzJ1PFwXGjzvLTS9j8FnvVSdm7wqFkQ94rJKPW6y'
    'ljw5x+q8ctQYveTipTtb6Ie9j0JMO6JFHj0yRlO6K6ETPYdnUL1GZ2K8XPqGPJ6eL7tSiBC8hXKJ'
    'PZjNab0+XhA8Jr86PI55Pz1QLFW9C2tHPYnagT0GCuS8j4pHvQUYsjyylAo9HEM2PSNlkDxBSse8'
    'DEhNPX5etTyMcRq9Acc3vfskVr1Un3Y8rqA7vM8rlDoFs8k89jpLPJ8GBz2pwU+9Eu4cOyNWOzt2'
    'uA280J4+PQyXyzzVS3W8iys9vSGEbT2CbA+9/jNpPXdWHLzcrFa8fIc9O0RhAz0GRAu9YyM0PTTk'
    'fzzuEkw9L3NUPP89L7tR3dG8ti19vCODmbviM/s8zCVQPcB+2zqStY09B7IpvYaGmDtMObA8nqAI'
    'vdv8kzzXe6o8wYigvAYW07ysuo09aSngvCRJaj0Xf00913RfO+RGuT09GSe72boXvS34Cj2NiuO8'
    'bx87PRyZGz0SPLA6VW5pvH5M6TtvooS77b+/Ovt6Sb1e/Km8m+WsPBv+WTtOHZo9NBzWvIqjyTxG'
    'iSS8ToWJvAS7dr0ImJS888ezPJ1RKD1mdxi8Y1gMPbsgwTwhOs+7NCT8O1CudDo9aik8bhVlO1VE'
    'MT0r1HQ7PYsBvVUohT0DcGc9swTHPUevmbssWzY9fZsMvb7UFT0zpYi9QcobvW32sby+H0U9feTj'
    'PDibgjwvnwE9b9qLPahVvzuOzzO9drNOPZgGdr1NLK88Nkd1vf5+Z70/uFK9dRQCPZTZYr0OqVQ9'
    'UoZSvZzKPb1xAV89HuDeux1PCbw/0Ci8HjKPvamAxTwIAHO8TGXtPL17iz0dutk74ASSPc0EwTs5'
    'tkA8jBs2vaValjvgLYy8m80fvZkkTTzMwQO9LB1qvWXAFL3gWlC8nqdMvUfIbb1vDtY6eeLWvKxw'
    'rTswOT69HychvDepHL1puqW8Z48APTPioLwufIY9fCtOvTdZ/rvvQQA9sYbYO5sgjbztpnO95lae'
    'vNu/iD3UDVE8E8tMvXVelD1PpYG8WNLgO2wE7ryccNm6iinJvFE/Kb3/asu7qJjnPMF7C70Wj948'
    'IpEoPW+dEb1Scjq9HKc1vR91cTyLVIe7hxODO7tdH72b0B49wdxBvaydXTz0RX68jVxKvdch2jzR'
    'bpk8+xM9PTR33ry7biY9v//vOybEp7zB7VE9++faPHl91rzk8c68Osv4OiUiAz1MUZQ9whoHPZfR'
    'ZT3sU7K8AgGYPdcoDb3F5NO8i/RGPKIK2TxnOQ29jWDZvEoROz0dVDQ9APJhvDrJgbun4h+9a+lE'
    'O/EubrxEqkK9R4WRPPI2rT1KZ9I79luFPNy73zy/pDQ8PRV1PB+XXD0Ed309BT3yPG35NLwoS408'
    'uYSNPT9ObT2rPO282i5OPGLn0Dz+biS9isspvNw7ST0RD/Y8KIZ7vZVXXT2IPAm9wy18vXHFpD2E'
    'jHW5QacVPVjvAr2Q8ja9kLYavVk95by5OQo778C0vGuCQT0gjGK9PXtpPGj0Ar3bVoe7vehrvZfW'
    'Y7yN7wS8drwBPdDMzDueu7A8wslKPU/Ivjt9d788ZTK6vMZLi70tBGg8Ytv3vCkmKb2s6j89A0kv'
    'vT7TlT3Aa5a7Br7VPEmfwbxsLvS8d6syvGyUxTuNoCq74an+PGxLS7zCybC8lQsBvUFkOLwxB2o8'
    '6iPAu+CsOr0zo3W8+x8Yu7rpGb0Njoq95hGju8DsML10kTK8RPBlvD8c5jzwKQA9bBgKPaBOOb0i'
    '6Za8XHpTPdEEdb0pM7U890ccvSHbKT36Qec8N9nzOhN6bjws6ge9QiCPPXVNqLw0y308PM7YvGnd'
    'kLy0KYe9ZfmMPSvCW71yeV89kWPlPIZveD006Rm9/pkwPF4ebL3Qgaw86RQwvZoxlbu7mFI8e/aL'
    'Pc8XhrxVw1I9+j9ePEVAtDtWNTi9lm+JPUgcDjw9/wo9/ijgvDa1gbxQEt48kqG8PEKzGj0s/lU9'
    'uQKGvUzFnLwpyZU7HKIhPN0NuDxuSZg789qDPZTZKj3ufS08PjkzvTAFbryueWm82vcOvegxFz1e'
    'EkY9mSYQPHfUcrxw+1E9Fx5PvWpklz0cKLO8nv1SvWVCfzyESaC8FNVCPZ56hz1oTBe9hG2APcws'
    '/DoQHE69bZX9u+Uv3Dpa83I8LW7avNlvJb2VQTY9flXyPMLRoL2EySY9VTiDveZwVLxUAmo8PVcw'
    'vX7MkDzBJWq8bUIZPUMElr0J7Cm84u6HvQuSLT2TXkA9cZuUPTUphr1pd9U8+AZVuixGg7yfZEs8'
    'QkvJPKsoVD0zmCI95jRYvS3EZbx8OMi7JTiWPPK/ZLy5Qzi9bKpIvYFYJj0uMRG9UY0OPc92kj1+'
    '82O8DJzxvFp2ar33Y3m7jP9gvVW/0TwfgAG9+BmXPHAzXrx6Tui8ndtovV4bFj0WC768GwKPPBjo'
    'S72gEbO8YiD/POI91LzfE6S8UFcUPfkOUL05bVK9lYgUvQhmU705rXg8N4RZPNd/1bzB3zI9hsYL'
    'vcqhBz1nDSM94FQZva0Wc7xi41o9v6eqPE4Ji7xOr2M9i8JIPemeNrzsmj08jDucvJEyKbzEWmu8'
    'od03PNG+krxf1tO8ZclAvIEGybs9S8e8zgD/vAFFe71dCyc95wLcvEjQrDo+nTM9FygOPePvcT1u'
    'xPE8OEoTvWmbUL0uNK08tlLpvGkP6zwYGgU95RZmvQnGl7z9TQq9GZEJuz33LjzKDbM85E5Lvcqk'
    'nrw04VW9xxpSPT1DKz2zuA69dRMsPc1RUz24F7+8XMw+vcpuAj0yJes8i8AevHJFhz2XsXM965Sb'
    'PWw92LxIPyy8BkWgPa2Aurzvu/c8dshqPY0yLj26aeu8eQW1u8BNJ70gal49HxwyvYpS7rucEJk9'
    'boqLvZnmXD1tAzw86NZwvIzK67wWfpG8JBpUPZmg1LphU7u8iN/5vCEprDzR/vs6u9oavbbAbjwz'
    '7RQ9014IvdfQ8TzBeoC8KXFrvciRer0SwJK8KwyJPB4IerziV2o96+NmveOSmb0ZvOm8gpPRvHm8'
    'IT0hhTS9BW2MO373Nz1girm7eDsDPRObeD36UG29kYFPvWIJz7s5dK4899zGO5OWebqxgKG8L6Cs'
    'PJpWDzzd7FG9oY7QO7G9iT1Ne0i8ltgKPYmumz1tQPo8xqrdPL2zED2M6Gc8JoFrPYr2nj3h6vU8'
    '9NqePTbrrDyjWT09065XPYbeE7255XI7Dw7KPMfMfT0CVVe78KugPSyFSbyQbY68Au1EPb0LvzwC'
    'lya9na73POjoEj00/Iq8LVyCPcGZID3UfRw9PgtRPNK8uDzVaig8pQ85PSARCj0CMWM9gtLfu69g'
    'zbz6O4y7Zj4uvEqm+jy0d+k8EUpfvSOEUr2qgbQ718c3PHXaY7lwZY+97wYFPXFjtLt/9wS9Flb5'
    'ulLRuzwTq/U8FRcUvYIXNr1O5a27b/9BPeusjLzX2Ry8cg0ZvdDakTxQdqK7cMkCPdcq6DxIZFg8'
    '46gAvZr8Qb0Ha1C9n7VzO3nGgzyc+Q072Kxwu5DOYj360De8Qi0UvCaejjycbR+9zInqPBejO7tA'
    'FWa8AtYMvAV0P73O+ha9rj4mvaKM/zxjLOU8o+Q7vduSTbxQcj09ZVUhO1eP/LtKBNU8MjJOvbd/'
    'C7yHuoS8oT5aPThllD3Y55m7sV+ZvPa2WLo+kp28DhBpvRSTFD0yBR69k9iYPJ02BT3DmRI9vhBz'
    'PU0X7Lvo7cc8h3EgvCj7Ub0kp0S8VS/2PMKIAj1IJSi9gniQvJ7BWDy1bjQ9ADRTvZCPbryTYzC9'
    'yUcLPdGL7rz25y686aZGvSWX4zwZ2z29QI3SvLc3UD0aVkE9u8EdvGbhGT1d9IU8giKxPBNQu7uq'
    'O/A82eE+Pf7qfL3WlWO97+RPvQFF1rx7COe8NUXUO+ISVz2JnjE9tv0HPK7iGj3NCus8ZwSEvccG'
    'sDz1TQY9Q60SvIJBO71N4jW99ciVu+QS2DxSW0s8XxiBvXCBF713Q349YUrFPCACn7zc2IK9/eEm'
    'Pdj9C7uC89I8YkoYPbdRkzzsPae87Ol2PbM5ab3o3nC9stEzPR8R/ryxo5o8hBgEvUYfkTziCIw9'
    'G2L+PJ66PzuU0w486L3kvIXdQD3eEG+9M3otPaHXULw5cko88XSvPHLhQz0PlyI9YgHTPDxowLxN'
    'oCY9M1UZPYC+Gz1d1gU9cf8wOzeQUzzVxxg9kvMQvVMevrxAOY282ys1PVN6TD1NFDg9SfsRPPeJ'
    'R73cyas66G/HvN8XGz2R6e07zD1bvdVALDvgE6s8RwzBvDgINj3kVC09U5oRvRGuvToEO+c8rdyy'
    'vaMnlDrXoim9nFKPvYL85Lqqocw8btsVvVgsTj24ALQ640c2PbOkfzxjsS29tRU0vZjgmbuQ1fo8'
    'p+c2vaZZej1T+B69XL1gvc9udTxVGQ09YwinPAYdKb1wXKu8rIapO+W1yTwHEGK8PHdyvShRuby8'
    'sqS8+i7tux1gCzxrET298MJBvboTkjzWyXg9Ii+UvEY+gT1eVRk9l6+NPWlQujyMkKK8gFUavE9w'
    'mbzISf28QGLbu0DhND20VCG7yAVFvZqS7byPnUa9IIFjPcNEFb19vbA8Qwl2POfLAbwEWai6z3GJ'
    'PNJzbTzcMGO947KMPLwjcj2zrey8v+uEO9YTmjzx8Aa9pMMIuf00WT1cS0i8ByjkvJnhcT28wz47'
    'FCBhvcDGgb0TEPM8zHeZPX3ejbs6u4K8vI+UPYoyiL0SYT497/Xhu5bF3bzWiGO7sqFZPdXYE722'
    'cX28trxVPFzqhrqbwwY9MghPPYX2Dj1WbwC7YaxuPRqu9DvCk5G6CCJZPe7lKz0i/FG8MWgDvY5j'
    'izyGouI85JMBPbKkaT2ePgU9Ki2OvGRchTx3vT28S+bLu3A39zwhBSA84Jg+PXRQaz1i8zw8tv5D'
    'vR/NV72oOUQ9nJOZvD93mDu/KBs9soalPf8TUzwkTzg8IkynPLa9ND0k+yI9OYdWPTBV47z48v08'
    '578ZvVMkQL1mVic9wmYBPSGBzzoykC498AsIvSJeIT1pCPC8gCtBvYPW9jy751q9jFipPMHLqryK'
    'Nco75xE/PRnNZbtEQW+8hrHQPFh5PjyNtX077sDsPO3IOD1LDYo9DLw0vTA8LD2srKw7PXFOvDgT'
    'hby1OB07U3NcOx8sGL2fuoE8j3oKu99wdLx5qWI8csWovd8wszwQtVq8kKwmvWmSgrxGOm68/C40'
    'PecJnrwk7z49qEYbPexdbb2n94s8bqQaPR38hbzxtFS9FrCxPKXQBL1u2d+8L5jlPH/K1bseQiW9'
    'WEpUPC4DFDyviDw9sZmXPKaAYTurnYG8d+pgPNrak7syZD284kNSvRuMcDxSUM+6/M6CPQmzHbx7'
    'xKA8Xl5IvOQGBr3zTRa7COA7O52dMz3v8H898dumPRSeCr3cnV49cqRSO9G1GD3RwB89aNMWPAOF'
    'hTu53JM756MYvVlBYru8rBU9IMI+va4ByDzal4s8Xq8PvXMcSbveKyW9KdRcPRQCar1hWZU8aDkW'
    'PU8HgLzeciK9eLYRvfIe+Dyw2UA7EkDFvPQaxDurC7e7cbWMPRFlP7xzRVa9TWmRPO7xEbime/Y8'
    '9ouHPdSZpLvDl8a8N3ZDvHaDhL3zMzE9Ycw0vdB+J70SjKg8sQn+uoOdLz3krR69QKYrvNXRDb1W'
    'wHc9RMryPAWaGjyRQP48lBdVvbeUoTxRmhC9wwBqvS1gWDzQhaY5TQ3NPPflAz1D9h28qb4RPTYF'
    'Rrwcx/W8DlMVPADf2zyyYD49m90ePYQY+zqOaL+8UwOZu0apSr247dI85UTFvNbz9LvOnlo9vkja'
    'Osazob1f51U8nVZGvRx6iDtilp08pqzGvAkNKL2HEUm9yh3uOuf+wrv96jg8vbsoPVfXT7zuHT29'
    '4vUivUfe8zyu2Vi8tz7oOg5QQDz3WZE8UItrvCdgED38Ks68ya9OPY/gXDwX1IW6U0QLvQl8mj1P'
    '/jW9OBRFPR71Lr2x7Oq8YkvLvFKAzTrCau+85plDPcFyeD22cNY7lPj7PO08Bz0z7hU90KYvveh8'
    'Xb3cPXC6OPYHvRq0srzdmYk9Ldw6vWF9Qr271Pe85+kZu6ZsVb2pqB49cPiBPdVkCL3XINi7N7XS'
    'PMSjGzy1kAm9NBvnvG9xC71qUy09EeVDvVxWMzwOj7m8xl7OPHJI9Dvcv+s6bzHWvEDsLr1oH/e8'
    'ofYlvG9Ekj3f3Gu8nWqRPdf+97xzlok9MysOPTklez1ZVHC7xoufvL3HkLxwejW9ydS9vJ0rWT2K'
    '5Rk88nmSPYRJKb2fkKe8Dp0ZvO84JbycuYm8k+KiO5S4izyInUE9osT5vG1fqryj1w496AfjPMXL'
    'mbx8qkY8UQHIPN4+D73kqRG84aa5vJEaeb2lnAu8QSG0vF2rWb1gZvc8dlaRvGjgeLwYqLg8T3Pl'
    'PMMNULzB5N48e/YBvN/CpryRyyA9P0EbPTJBab2K83a8d/ATPRLNFD1W3as8VGDuPDwZK71yK2s9'
    'TIFaPXkbOb1GHOy7QWiAPKW3/Lp9b9e7C4wvOxQ0YT3iszA9NLUnvQiRYD3JlKu6YDdvvBZUZ7yQ'
    'FCq9HcguPTD6XL1WFfE8/J+GOR1nHTxfha684g/0vBYNCL2WHf68NLK2vFApQjy7FQq9mCzovKeI'
    'Mb0lhIQ9OT4pPdh1lTt3Hbe6xG4EvfhaYT2HFlA9tzRPPV0ZMTyKZDE9UnaLPA+3vzt1mT+87YoS'
    'vXCx9LwGGLi9Uwt8PLlDzTsCOw89pxYPPcWYUj1PcAe9rYp+PFnpkjyQr8k8aDH0O1uGJ73FSIo9'
    'KjlhPUlDOz3zqOQ86gprPRwncbw7NvK7UoqLPQsVBD204de8fjktvfR3cr2DDSM9T9i4PNSTDj10'
    'Gzy9r2k8vDf1y7x9tug7avKGvDvS0jxYnk08PooTPC6rAzw7JFO8F53vvK28OjyEswo9+o+mPJgI'
    'Nj1GlxS9hhMwvZPCTD2FLvk7B4BwPHJB9LtLCbo8vWxgPC4ErryPyCW9LF8vPabiBr2acAG9bMoA'
    'vXpvjDx6/qq8UaguPbCMajwOEdM8nrtYPbfGi71wPOG8MD5evYCEoLxpmwM9WvN0PZmC07p0PzA7'
    'Xts6PdekU73pPLO8cAglPZ/RRL14QPQ6LsVYueftv7yzYiq9VXoiPdS/pjwpJVM9DjGHvNZ5NT3R'
    'Zxw9Lp6wu46e6jy1cto8RD9KvebVWbuELI+82nHlPOKPBz0JbAS9C+Seu7ee0DwD5To9C98JvXEl'
    'Ebz9kHC87pCtO6qmgjwLSb+7/Y6Au9z8Ir1SL2I9fopbPbIsB7w/ANi6cXotPDYDDruyqDW85YAe'
    'vJz8sL3/ivA8xKkSPRxtgTt7yDS9Mw8aPVdvPD25GOo8eVgdPK+0dLycPQw9Oh0DPZjjbz14cf87'
    '72AHuqt5jD0klV08nLakvJEz6zzDbh28cj1JvbI3jTwkYWy8rBpmPQDPJz17azK90gOBPSyDEr0r'
    'Zjo9QRKGvI96zjo2VBq9jhzXPPImbD0xBkC9dR+dO5Ji57yWsfC8y65LvVRrTD1zHFO6DCKkvH8u'
    'Wj2BqA69ZaGLPU0ETzxYHkK91QI4PNstFL3BkgC90Q4UPV1FILtL1b68B9rhvJLXTz2dpe48yBhb'
    'vQogMbzOqQS9RLN9PKykyLupodU7jTd1vI9U2Ly5WVU9/EZLPbQQlL1yUzi92IKuu8GnoLwVT1Y5'
    'rHQMvX6RMryFnEU9rS4IPefxB7zJIbU7ZoCYvCXPAz1xBtG8O5tpvd8TzjzZsAu98PI+vUiyEb2Z'
    'op08T+XKPPxeXL186428q0kHPWmCZr27ljI99FaEvbWmU71i1488BNC1vDUmn71k1UG9UCFlPKC4'
    'aDwOOyW9y3MSPVCSU72viMO8f+/SPHzoCT2VmQw9eTWhPSZnVT1/Wvk8I/1wPfixZL2EudI8xIN0'
    'Pbvuezui2Pi8mOybPERPfDzRdVG91rxVvVlERj1C6Yk9JiSWOzF9Zz0gmoa9IpTwvKJ7lT1gz088'
    '8XoSPf74jjuA0229Bbc/PW0/9bzEqRs9gWdDPa6MirsLBZ88a7DHvEmuFb1tPaK9bqk9vebCnzvI'
    'pyG9023LvDC90Dxq/hO9hMV7PFYMIz08CMK8v0gSvcxbCLx1Jtu80YcZPX1EE7zM9Xq84dtbvR4x'
    '5TscUOk8FdUrPQmmDz1vxNg8YVhJPaCmjbx48S89bHGDPUpVQjykH8A8LuAbPOLq3zzPSag9X7dr'
    'PJyer7yYs4I9nXA7vT1mz7lOtMo6ZiTluwne57zsWsk8AkAMvfCPWL1Gxlc9SapJvCjPc7y3oDy9'
    'Gk5MvQaIybz7oxu9JXUGPWqMwDwjXvK7KO4LPSzZHL0EURA7Zn7SPIRD5byGUL67OG0rvdCXYb2B'
    '3aG7qLP5PBoym70h/Ha9GazgvBg+hjzJDPM86I8nu3Ilm7zOP3C9/fhdvFFUq7qFaVU8rIOUO9NG'
    'iL0xRyi9XH/RPO83WbzYQdu7/PuePGC6KL2owgY7zMsHPb5aKT1Fhy29ByacPPhKaD3RKnQ9qVZ7'
    'vHUVaTwv8B29A7T8PLZkTj3/PAm9Q2maPNZzn7v2J6g8hoXyPIOiOb1T4n27L02QvTyHVj3oZg49'
    'DxVwPeIAJb3riIk7HeIKPC1On73ByjS9/iKevAplJb287Ic9xTZePa0/Zr3qWl+95d5XvX7D4bwl'
    '5Ca9CaAovRwjRDya7/S8vja0vGGyEL0g7G89lxfZPAnOG7wYxjg9LELKuyQVsbyoBrm77MTFvL9z'
    '37w31TA95ONHPJKoBj0VZja9o/RtvWEHAL0fcBq7gNgkPDr0Jz1KDFu9FFEcvbJ7zjxbF628Wd4g'
    'vXypp7yyZL49MPduPOzFjzzNG1O9VrfLO1V1lzyqS169Hvd5Oja9A73eloK9RmHSPIzRdj1f79y7'
    'av1+PPlRMD0ztGE9bBbnO+MO8zzfzSK9PkYYvTYfJ7w7OZS8f2rmOpauGr3V7Pk8ehwnvSAoFryS'
    'WS49CK40PUXYPb1eJ5I8fenwvFdNZrzvBpW8D0scPf8CrjyAED49yYaHPCBjkL07BLA8nOGvvDab'
    'hLvt8es68iPUPGa7gzzP9US9CrwDPcfxvrwsMbI8mDZBvWwGB72qkQ08Zt9qPe2cRr2vdLU8asVI'
    'PG7PAb3JWJQ9RT/BvKYaG7yYLYY8AlAjPdy9BbtV84I9eLtovNOjdr19cyi9MZ4wuxD3PL3mZTc9'
    'Mm4SvRAjjrvohIO8IF1FPVtlnzyycXK7Vdv1u6cHHr0KXLe8didXPaHa8rwX9HM98R3avK7NOTyy'
    'TcG8S/imPSj2vbtgNtI8wj6BPa/XkLwBi0e9vPclPfucIrugwjC8DwmCvL0WebnFQL88w30EvRgi'
    'T7xk9dM83t8Qve8TIj0Azls9oCUfu7JlPD0/Phu9oBPdO7719bwc8kU9bdVCPfwfZb1f21+9lUaV'
    'PJLNJj0wmj69ENnAvCOqOL1vOVA8DVF0vXT2Dj0KeeO89J2WPRMeCL2AuUy90dsZvCn1fT0ae6s8'
    'AYleO73QH71vg7a7jaqcPSi46jtkhw09NYAGPVu4rTtHBU+9huVbPTjGBjwypis8B3Z4veQKdTzM'
    'Ora8I3pdvbQXHT1JVha93b3svEiKdL12ZyU9ki0dPatGMD1uzDI9cxoavVgtnr0HDAE9ZIAwveF0'
    'uLyPt6y7VRT+vJobWr3E3DK9B455PJo9bD3fR6C6BLNKvZBS1bzjRTS7CySjPCYn6rxZVGE91vDD'
    'vPIQpr2ycE69H6/0vDahhDtz4my8XmYbvSS71bz+a0Y9OIquPAJk8Tyv2KA8SnU2vX8EDD1G9KY8'
    'HuM9vUEExLt3cFk9dYDMvEtPgj2E0is9u+HIvOGjUT2m5ys9MImYPG6Br7wciV29WJkNPX4fIL19'
    'IE47Lj1kvdN8kjzEt748aPYsPZxFBj1Uo469AfxUPW35UDxN0P+8ae+DPerwh73/rAs9fIcKPUvm'
    'Z70sjjO8t6gIvYgmer1udXI7Y5QUvbSwDL2mUho9sI7Ouu4omjzNdxa8j8TVvHEsIz1kW6g89FCK'
    'vavewTxgnIk8iEesvHzThDwRtGs9fWDiPIXRKzxwSSU9UramvHqzVLyVf4E9KU4VvV/LS71+Y1+9'
    'IdEKveeYljygfm+9q6tZPbT/pDtgiiW9we7uvL4Sdb3HwRY9eKJwPflrxjxyUby7qx+guM9kB701'
    'SzY9OQO2PJoKAj3ZkMU845VYPRIxR73B7IO8cuICvbhrwbxuORw8X05bPcyaFT1xywe8KDkzvWBf'
    'ILyIiws9vP8mPd0ZW71IVtU8JjcCvRJ4Ib2fvoG9AABTvEWH7LyBtWS92wJFvTY1wzwxzy69bLgF'
    'vLWtsr03Lyq7aylRPEbt4DvdVws9kkHVO6f4jTyfuC6863UYPFCag7xFkhg9kNWfPAsdFj0XafU7'
    'b3tkPQq6Gz1kcdA8nvOkPO020zyP8AS9bye/PAHfiDuUWxi9tYaLPLDyPjzRe4k9l8OEPMK1CL1/'
    'TGA9dx8tvOMtpzwvd5a8VEGXu+dOLLwIA1g9N7rnvLZal73j1Dm9k15UPEkt/DyXkW+8jXedO2C4'
    'KbyOrQu9or98vSKZnbwEJSg9PpcpPFvhgz1jkQ29PAo/vVdUTj12Tho8xuHiuiMlIr0wTEs99Gc8'
    'PQiWwDuQMs88PhMmPW7sBDsAMDe9mel7vAlPOj3KPhC9jzxrvTUed73GVyc8YY4DPXXTMT19gf+8'
    'oDOtPGCVD70Jc0c921GkO9DBFb2rOCe96sQRvT5ePb1605Q8MMIvvBKWETxE6FC99qk4vRtoKbyn'
    'NDM9PlFDPCxdQz3YIjA9mF4ePYnPXj3FvTa9ZPg2PWvsib0S5Eu9++PDPLieKjxtPTK9Ik9RPT2U'
    '4jxiCL688aWHPaaFWz2Aani9Efe1vELTSL28lUY9szYePE2v2rvdTFM8BegqvW/NCb0DHt48Iquh'
    'vKW/0jtOnTE90egiPayZtbsiFGo9CM84vADYAb0X4bo8W8IIvU+FZr37DPa7+5tNveNnFDzgOqk7'
    'mjccPIMjTz0prlw9Yq/kPGSu2zwBIUQ9CQe2vIYku7xrNrs8Sty1PE/1Dz3KF4w96aFPvRDvR72o'
    'dlA9pBXkvNf4Ez3aEh49Z4yVPWCLjLzHEBE9A9XUPI7MAb2x9ii9b7oQvXnBSjw/xgK9STY3vHV/'
    'RryI6VU9cPFYva4KGr1/az+9bXguvY3kDb323Qw962krvNNnI7z2e+67M1T/vAo+jT0yC4y8+TpI'
    'vToHU7td/Eg8qN5xvWloAz1kI5K8rqllvTpeaL0fFV49WawZvU7Z/rz9zns84kDrPGIkHD3vWw09'
    'dWyLvWqMB7ocBAG7kuI1Pcm4Oz0h3gW9NolvvXqeWz2vWjm8ZGIxvRgDRz3f+kg93fKJPHud6jv+'
    'JBg9+v43vcuN6TtL12m8eDRDvFo2SDz2OAo9IP2bvNiIojxL23i8OXwevUQnX71QOnO7vBkBvZuw'
    '9jyySpA8lhvzuzRMqryTuk697BIZPbID7zxXKEM52r4ivRAIOD2QwBW9h3NivYlSszvL1UY9AhCc'
    'vDGp0TzdfxE9N7XBPJ7XBz31lvU7KSW2PPTtPj1gynE9gkr1vDG9ELzS+FQ9+PNuPW6UpbzknUy9'
    'BuDHvBfYQr0YV9a82tvQPF9ShTyAIYe9mSXjO03oCDvMTYC9CF8ZPDY5ST1uAzg9zIEYvfTurzxt'
    'VAu9wkXYPEKeijw8Kdo8phlxPWYDPj2yyzQ9pB8SPWdTvLzR0Ec93alDvXKeAj1HNti8KwFwPd6J'
    '8TuPWFc9CpmQPEd7I71SQCu5a7QdvL68ljz3O2Q94fUtO9UlDT17Aaq8cwQYPcbDT726yJK7UYuB'
    'vIaTJj3GEtI8ip4tvWfjsryTqQ29sSmYvAIiSLwdFnS92TIrvSDTe7wPdAy9OAG7PDfOizzkWUY9'
    '1sXGvIA1VrzNYI08LxOBvLaQQz1ohZs6DiNqvZuaiTwTjk89zXZOPVYqTb1dMGK9AxyYPOZ/Pjw3'
    'Asc7/T4wvWp5Cbz4DFa9Qa7avGIzmDxdXd48GDveu6gnubxKikW8bQcNPUVm47yhnK+8mMxWO+2S'
    't7yTTio9oRwiPUeN8DzVRdy8eUV3OybeHD1gW828kom6u4qjQD0gOdo800EpPQk0zTvbuOq8m4mw'
    'vArz3DyK/Mg8PR2tPCLapTyZW4A9a3Q/vcgJ0zy17Bi9uhfFOxA1fT0eDP+85kqwPGFdCDpiCvg8'
    '/+mKu5iOI71t2Zs8MBk0vTZyr7xbO8u8QiIMvNv6YDzBVQK9ASZvvdai8ry5wX08hwEgvSghp7x8'
    'wzC8L5chvUnwor25UOU8LygIvMVnGD0BWi49PEoEvD/iijxTBvY8yT+aOtbuGL2CZCo9vXZBPGJc'
    '+7z7h0G8IwSWPeq0j70ukEw9E+AFPc8pQ730jSs8xvLavInIFz3fOO+7EOUXvWZXFTyJlwK9zHuG'
    'Pb51W72loY+8cipVveY0K70PTQi84Ag+PSSe7DzShbS8HjJ2vUY9aLxe8Bs9+xoaPT8qOT2hYqq8'
    'q7kFPLW0g7t2wIC92n0RvZMGdLucFwK9MjEevetv2TuiMea8P5gAPBc5Kr3M/S28WffrPIL/lzs9'
    'QEi8AiF7PC9vmbwWUAG9kNSAvQNWF73r4CG9iWn3vHvBybv5O0g9fW23PAvHFbpJyba6mKunvDTM'
    '/zzLyug7XxLpPPlVkbv3M+s8RaorPbiET71zqXy9QPU9vDvf1Tyf1UO9XoaGvXYCRb0ueso8+phc'
    'PVtQTrzlOXa8eHkCPQof3Lx3UNA8r4lPvbtkQ709iSW9Q00Duyo3zrp/VT49ppm3vPbTfT2Nwkw9'
    'yCmKvAYvej1fZX48dajIPJOfUz36uK05RVPuvH4GX72iaZ+9FAWivFiXPTyOa5o8uDibPAoUp7w0'
    '4FM9cGDvu1jkvruwxsY797YUPECOHr1tSqY8E3ULvbDsaD2dXNM6gGxsPeqbMb10OGi9NyErvXGj'
    'E70kf+665hIVu1lyLD2j6JC7JdaCPTDhEb0gv4S9R+WWvTVNDb0J8jC9vY+SvQmHqrv93XI9dvC+'
    'vJxXm7vCgWC9AucaPU45Ir0Jx4K8iO67u614Mr2KFk69iVcRvZ546TwTSYS8EvNOvDdXTTxdSMg8'
    'lPsfvONMyby1bIe9R/8GvAwplr16ZNK97qG7vRlBPb3+Rci7Q7KFvZKPFLyUeQ282xtVvbia/rxb'
    'utM6Pd+UvD2Qyby5S6m66Wc5vc/xwjt06748/G9LPbx7izwc/1u9dy5CPdWoFD0kcJ68Ylo/PeNH'
    'lT0TnRq8c830u5MHTjwSKA29rtJkPPNsC70+Tim9XeE7vUf6ezwBuiK94y0WvEoIJj0NUwk9XM0t'
    'PRNunL0jOga9D/VXPT9PYrsLyBe8X3lxPBAyZr39QBS89kCQvf7IT73b54q7eWS5vFSAOr3VO7U8'
    'DalDPbnuBDwp7g69vvsYvWTbwDySpFY8p97kvC5CFr2AZEY9TWTPvMs6TrvbJAc9YmhEPWBtfDzP'
    '4Ck91p0JvVlX+zvx+wW9XnNbO9qEFb0GLZq9mbQQPVAwDb1+0qO7hLCTvf8qhz2VqhY95fKmu6Xn'
    'Lz2hTRM9h6CJPO7tSr1fT6g8/hkOPSGpKj2ir3o8Iz0QPVnfsLvt5p28EvsnvDEYWb3tiXG9RdzZ'
    'vHALcj2LpHq9v/ZZPcpGAb2m1hE9iSR7PHjRh73xhja76R9Ave0+kzzpQKK8B3cNPRa6ODwJh147'
    'hIEsvXrcVLy1dEW9hYw9PcJps7xtJ309K9dtvNIwUbuODCA9Y8juvF2kgTxoLjQ9BRbTvA5kDLxu'
    'DdE8b9YWPaXtML27kRE9SpxfvQwSj7zoOc48vHLNPGqkbTs8mbi8bryoPGiVOzv6gCY8zNRAPSnW'
    '97zXE029wgN7PUJCDL1xZjI9XVQvPXqz2zzMhOm8ujsBvLntcTwMuQa91lzRvG7yyDxshLC7Jou2'
    'u/uKprsJKMK8RPnevFZftjxLfQm75tnpu4bNhLwyi2k9vBYnPO/D1rxoatW8rUcGO2ox8LxigVW7'
    'gUdkPIu0mDxn6ra7dDaDvONWpz2e5Zw7aiunvFd8VD2b5q07qA2hvPL1Mjw354C9GPRSPTKP5rxj'
    '5Qa8kKe5vC/KFD2RLb28xrRbvCXwobtqeQC9Vl88vf7R5LxdJM485iuGvbfVyDx0ySu8ZDebPcWf'
    'N7xXBwW9U4aDPWrNLb0GjBC9d9K0vLmWYb2siaC7E8mpPHYACr0fETE53FkAPdvUMjxT0G28G3dS'
    'PQ5JKj3oUSa8eF4hPFwTNL0HT+s8Lr6JvA7aFTx99QI9AfP6vPOAXrzlkXK9MADkvBlOWD0h+jS9'
    'NX4HPEKJqb0dcMs8QEshvYT6LT3JMFc9n1IHPX66sTyq6g09MN1luQce6rwp3w+9Ip8lPVWP5LvG'
    'Cxe9epaovCKmFbzcL4C9L/y6PDp/ML1gCiq9TK/fPJ8WhT2Hsz67fWg5vfzmPT2hMno8QnTvvO6o'
    'NzzCgkQ8BUUNPWhEGb1UTIk8DNE1PQaMiLwjJYQ8DErFvL4xhLzUnhe9pF6pPHPwsjwU7iQ9MxYZ'
    'PTasjLwJONa8onw9PZrYJr2ZyoS85aMhvYZyZDzy77y8yU+1vJVLsDzqS029TFARPaLvF70tH1I9'
    'i6UMvThXKjqK5Y48/qi/PEnPjbxTT6w7n0rZvCuiNb0gxqA8y+5qO9T7Sb0QENU8DkV8vH4fUz32'
    'B8y8Uatmujchgr0VrU+9e918vVcxxbzKNka9IwYqPbKEU70pCBi9pE45PSfuh7uS9xw9vccTvHQc'
    'KD3uTna9mYC/PBiyML059UM8Q4RmvHWeIb2c8mg9ixjqvGp62bur47+7ODsoPVD+vLw8L3s9sPYD'
    'vR1cZj19FBQ9JBI6vbbfrrxRaKa8ifO1vP+ISD1f/X+8NUqLvLLGVj0nHbC85xkevaCOYjvCf4C8'
    'eMsMvWOLMT3Uvmi9eEg1PeR91j3+lAO93FeDOWhwWD0az/G5N73PPNbhF7xNBWU9fSm6vHZORjvW'
    '2lE9gJRqvRwSQj3NVzM9m42CuB8R1Lz8Vq08fqwmPeY6Yj2U5y+9lbtyvUd1vLxtbLW8cCAoPWc9'
    'Hr2wAoA9XvxSvY3uUb0ei/06tj5OPeaXmTzujxG9KctZPRHvdT0VKEG8OxlTO8mNTbxRcIC8QS5M'
    'vd5XBD2MVLU8iShHOUMsnTvTjRW8H8dTvV0ESz2yQK48rcgKvE95Jz2os249xkITPcsHPL2fo4W8'
    '5G8BPPanaD0y9oa8yR+Qvd6ON71NHUK9SOC1vAuuajzoL9E8vVuSvPgMtzpNol4880L7PNL+Fz3S'
    'SYQ9FNpSvS1Ewbzna0e9qxn2PPVAG71ynii9N/IxvQ+Pbb3cQGM9ek2oOw4LOLya+QU9J4xFPGQE'
    'S7y2th69gEEIOwfFgD3+RbK7UWlEPT0R7LzQLUO8VgiMu+oHO72HnVo8sqEmvR/8Ar1XGJQ8Pf+/'
    'vJ40YTzsp4E8Krs9PfCUOL2GXRg7QNMwPcXCE7y1qxs93VIxvZtFJjxjtRQ9+bxKvVhvEz1/72I9'
    'aNZ9vf/vID1Ze3G7tZcOPPdpCr0MsZs8/e/svN8H27tFGqQ8xFYlPGUBd71uLiK9tkN8Pbdhzzwj'
    'yvQ8Tz7XPAazz7zoJI09NGNePengb70Rn4w8KtE+vdSmMr3PCN67uQIwu5+3Dj0C2mO8tYvsvIYV'
    'bb3adwE9md0/vRgVOL2RqES9ObCzvEEbAz3XNFM9U7EYu1IrFz2/i/C6gOxKvZNFAL15fh69VKg2'
    'OzG3cLvVn++8RsExPVLnnrvtutA7FRwtvVSiDDyFbRI9nfIjvRC1cjwDaRG9D2VbPXlIOj3snVs8'
    'kTWVvfSEiD0Ro7u7eSfqPNrrcT26ypS8GNGHvYxwTDtURK07bb1JPZYBW70u0QY8kcmxvEuhY71g'
    '0Ry91HQ/vXm+IL3ln1U9oPJ7PFqB07wGS/K8ldWSvGzixbwkslS7iE/YOzGhJLy+MMe7BhFRvFKE'
    'br09WEy9oJA0vYD2i7xks2Q9cpFwvLPqED1evUI935m2PEmdCL2asR69+lMKPdW44byMwcg8wxEa'
    'PREdn7sztUm770myPMoGBz2Jeee8i3ywOzSLPr1ZOA27513fPFOAXL1+Q4o8WTwEvf2g6Ly2hLa8'
    'XWA+PeVPPb2fupe8kwcZPdBeX7zrYr27OUl7PHPVfz1foVa9Ps7ouZvjCj1nATo91jxAPfKppzyH'
    'Ppq88b/RPLJy9Lx1nGM7DbkcvdfzG73y4DI9P0wYPZq65Lq1elc8bZQmvU0/SD3vLk686tMIPcCZ'
    'yzwMZ0S8FFIoPTrmOD2FvGQ9D+3qvOo4Dz0RQRo9KD9FvcAgyLtlLwo8Iwjfu34g7jw/YYO618eL'
    'vH6PHL293DO97Y9cPPt0Wj1qKxg8X/VBO6sGBbxvNJw8xj8BPYa5w7ynTaM7R8IGPRKxFDyb/1W9'
    'tA4pvZgm7rzmjwk9LgfMPMHd9DsMzpa6oLRfPXWhm7ydtk07dgY2vZzhQL1N3Xi8IyA5PSQCRr3D'
    'THq9iARKPdFlKj325ja9aUOevKWAhj1cYnq6dICyvIbOrLu4BGW9ob8BvZQFKD3OO208iiJOvQVV'
    'bLwAAh09DBQTPTv4Xb3yfvM8HYoWPUwBorweLw89J5uTvZXjDTwWCgW99ShKvarXH72eJ207JbKd'
    'vGgTCL2har87rYZAPc4dGD0AVYk87p04vbZxUT1vBo88pMMZvY66k7ybgwa9lQsAPb8/gLyxlti8'
    '1TgDPaI3NbyAHTy9qZMjPcQ11TyXQV49hhiovIdxO71hbUy8fh6cupnpPb1gV5K8Nv/+PKsxgbw3'
    'iTO9xcytvC1hVTuP1AK93+jDvLzEuDyyTl89+mlNvAfYEjx/BOu76IRBPTLnDD3Ze1E8OXJRPQhX'
    'PLsg4vk8HJeCPPwWTTxPaFC96qF/PVwJUTzpr/i8rUWWPSrOvzx07JA92GlOPUX1SjsKZTi8Jvsg'
    'PdAFK72jS0G8sIWkvH8+Fj1X+dO87KABPGnhQD04YqU7txfGPPuFTz2i9L+8UdDTPLY2G70vRV+9'
    'EF1JvNlN4bwYG5a90oqrvPbWGz37JEw9YflEPcUahr1FSEM5kGooPUASGL1Slgs7spE3vaR9Ob21'
    'FW+8ST6JPDPUhDzgib26pT+qO38aVr2pFGG9rO3JPPvUszumpbU8S/fBvCN2kTwqI9e8bvpdvd6h'
    '+zz9LwU9AI2VPFCEJDyTraC9GFaoPNRIOb2dx0G9oPa9O/deobxkcZ69GxN1vTNibzpYYR+989ha'
    'OwXyIz2mWXk9iKVMvfj9Fb0kPYq91uvuPP214DudKnY86wtTPffEVz3DbhQ9xDAtvbFGebz1s2Q9'
    'afKVvYp7njwioLq92/CivdvsG72IF948OysfvNWSRDyr3kC8mXd0vVkxiD0yqvA55DfSPKXYjrtP'
    'R9Y81mGVvHoT/jwJzjo9nC6CPAkEGL1RoZu85sycPYqGYjyF1lw8NSDIvE9KjL3Q8ca7AcJEPQ3b'
    'f73UmC29hYBzvRkIBb37Jvo7s7YSvEsxkDyV1mW85WyGPVloCz3pnFy9mE5DPcVSCz1j5Qy8a+Nw'
    'PUHSHD1zDHq9wAxFPWIzpTlwFCy9YzIIPVc1EzruSQG9j1ZhvIuEC7zzeAW6UOMpvaixAr2mzK85'
    '5TQ5vMM3sDzEfVq9Lx7QvFXkLb13gY48F1a3PO58Cjx7/xQ99e2/PZ8E57zjmTg9cWcxOjUWqLxo'
    'dpY8q22ZvMHMYT2m4Vw8yBybPXEvTbugofw8nip4OkCOMTyqnuQ6BKmSPbGqujvPuqI8CgUnPNng'
    '7Tx+prI8ME2evHeJgz2qPEy9fboXvQ22Gj3iiW680zjzOwQNHb3yA4i7hlASPMG9Mb1qckI9lAFC'
    'PEPjQr0bbDG9+JQgPbQlbbwJQAU9AcJmOx1PWb2TmXU8Kn1tvf+HibwHOnc8J3W3vKZjW7woDV+8'
    'IeaBPLhn/7wN3TO9+HEyvWotHj2jthG9e5vDvLbxGz2hE+O89OIBPMsQBLyfhfq80PLFPX5eTb2C'
    'hRk99SkCPZY/Sj1qO/O7uT6KvEvug7xxfxE9Imn+OYFaGD1uKQM8b8LMvOIShbw5rTk9fJI1u6q6'
    'iDzCsQM9H9mnPOk0IL1bLuC7IxoqPfuYRb2bC2Q9Zcr8u1P7z7wxAno7GCJqPS0sDL3qI7a8Zw0X'
    'PbD3QL2rvKU86mg7PQfYZT1eRG+9/E6qvVafPj1GdI28y5zvvPqkxjyeFCg9QWtGvfRV7TyPeFQ9'
    'cYVJPD27ZrzvJZs8+3A5vWX7Mr3KqkS8Ur5Uu91MPj0BmBC9dVXePOKfKLwejj48AGx2vUvcXL0a'
    'shi9o6bBvCdlcj3F9fE8BGoVPVdAAj2krPu8U857vR90zbxKVQW98PoAPSVE8DwYWkO9CGIavb1k'
    '+7zERS098EcHvVpNJT3HkLI8fOLIPCmeKz35Gk49Xb9IPYHtL706F168+isiOy6SPb2JCoE9eDTO'
    'vKmFEb0ziH89zKf0vFfamTyB9cQ8z4SVvDf7Mr2ItZU7p/fFPGLuaj0A8Ta9IqcKvKQvgb05qXw8'
    'j92PPG3f8jqGoGO9AWkyPRICGT32zCO93LYNPfM4+Dtu2xm9WSqRPVuSGzwj5SW9ZfwpPZUGG73l'
    '8zc8X9glPdBBQL3z2zs9Ixpgu5OkE73KqEO9h8JNvXXdzTxXxDk9asWRul78BT32nSO8SY95OxkU'
    'hrdZdTO9g2SNPKdE9roFVCc9Tw1ivTpNAT3w+/s8hQz9PCUgmjxeStG8DUFRPZQZOL2AySg9Yaod'
    'vA2gH7yyV548E5FHvbl5SD19mBK94vdYvNUR6DzrCEO9mJYAvbbQGT1Ao0w91hWTPKlkGLyUzxw7'
    'T+NOPX+Qcz0HuCO9KncIPQZ8vb1N2jK9JhfCPCjCP70pS426XDGWPbmkyD1nGo09abSYOrIQDL2b'
    'nXi8Da+dvOUYtLxVQ4m7zxDMu0S3X72F/C08KNhmPM8SCL0JlIe8GxmuvDS4Nz29UVS8fQNFO23l'
    '3jyxfU69+fpEvZDVJr2pma68Oss5POGylbykOfQ8bm27uw2n1Lq5H0E9gkm6PJESWz0YkWi8secB'
    'vctYzDy6ghu9GvuHvCWMyDwjnSy5pOSbOxSkGTyhujQ9pCV+O2huYL0wrUa9JXjNOyEoY7zDAEi9'
    'u5QLPMBR5bv3Rq68+ClPvcVmCj0KM4i89EIAvVZySD0S8mM9CuOcvPTdWry6PRW9FgVpvCeeGT1s'
    '1N48XmFhvS0PEj1PA/a8K2IaPEfNJT2dDX491YtFvLhDZD0Wf8C858JevYp5kbzqDuE77vJvPV/K'
    'RLt6OTM8nLAfvRprH71zPwW7K6ooPYNwDb18Zn89RU1RuoL0sDuds7067xLVOkd3uzscDxm9ayeO'
    'PNW6KDyIDJu8jFqcPJhoCTwE6Fc9HPpLPYEtPj1UXDY9S/RdvS+J4jy9KF49a4EVPRY9sjxhY2u8'
    'LRUFvX55Xj2Pnl05HBmqvPCmHT1NXsO8NfLCPOvrW73TUSy8K/ZQvW8/0TlEhwQ9s3WLO4d0Z704'
    'nRa95t6DvY5ElLxfXkI9HOFfPTLbQz3MQbO8RbtNu1cCIzz7mkM8u0FnPZNOfL0Axgc9uC1xvYjz'
    'gD3Jeiq9ybcJvXtnXT3oz1Y86k0AvcyY4jyH1IC8TA/VvOvORr3XKSa9SgloPQ7hqLxiKto8yS8F'
    'vf2DCb1kgRE9MsVUvDLJDD0qDwW7DtBjvRReAzwgEja9Wr5Mu0cJGrv+oSo9edE5PJMcGT0nGHC8'
    'tgMqvRGdJD1jkNa8dgtyvNxezbxgBaS7r0qMvAZCmzvF9Uw8ncM9PB4M3LtRrTm9b4ObPOBrSL2E'
    'oE69X8kuO3Nofj0iBRW9qJqcvLKrLzyK6dY7RpctPFkwB7152Fy9WcKBvNOCVb3VQsu8ppiAvKsw'
    'GzwoQT09xdMMvUAR5jp5DR09u4fuPOzYcL0X/Ii8AcsBvJu9A73e7ey8goSJva1GOD20Wzo9Sb0F'
    'PZ4mfbz86is90xwrvSmBHbwELEi8yZOtOVP+N73pf5a8qpyRPY+AQr2lczi8sv4xPWIvjr0sM0E8'
    'eLIvPUoRLL3Z50K9uxHlvIYkED0eTmc9CY2evMBzBL0zp1I9oMyfvJboo7xXkck8OS/JvCdCVj2Q'
    'pDK9kiV3vYpTPD0yM1c9SMd8vAhkJT0jNoq7F2s+O4v7urzLVmY8buiEPcPTIT3+gaA9YIaQPY1W'
    'XD1cJYA8MsT/vBELAL1UeWc9SGD8vMdmlLx/54Y9O9wnPRV0JDzl5OQ8rxdFPQenrDwrZ8Q7sbw2'
    'PKvM+ry9yQC9R1jNPFSSYrwp7ZE6SUhDvFRBRj1VVrc8+EaUPFhey7y8dzk7N4sjPQiLgLwWbni8'
    'ld1+PfMk1Dsdis+8eDJrvGQ4Nb0+U0g961iKPKG5jzzhOmU9o8YjvKynLT1DpOs8+fI4PQdTirw5'
    'mus8FksOPV5+hrwQOjm8PDK7PDxupTtbxVG7npQrvQEeyroMGAi9+bViPa4pQT2ddwA9UJyoPK2/'
    '8DwUK0g9aa7PvLeLvTwJ0Yo9JpxCPBlLA71hcOM8lJKNvA4QM7yfbmE8r4IMPW1exDwYDfY8xpK9'
    'vN1qcr1Myaa8zMwpPZN1QD3lUoG9nUqgvJzUgb3Tfxe9yK9YvVqim7yer3C9saCRPfqZNb0aN9c8'
    'FMOBPaURFjpULww889FgvWW3zzzVyVE9dPXAvGRKfDxqUMs85DlePdA92jy1WEU9OP2cPOaCKD2p'
    'thS9xIaiPUaPNT0o7ha90zfJu/xFyTzXVAE8TfAAPHpUaz1/Gk09bLCGO/pNuTwwH6A8ydlUvTL8'
    'qL3cSdu7CVNZvdjeNz3/9rE6JB2ROzN+rDxgbUw9ZzrDuzWrN7xO8we93u9tvXfQUr08swI7FfhV'
    'PLHpPD2jIiI9nlcaPLtYNz2lmxi96TY2PcoB1DxunDS8Z8sYvXa+Mj2jP4G8n6vGPCRlJr2teTY9'
    'zXWQvey8PD1+Vgk8dv5ivMnOAD0I0Io86nDQPKEl/rvCbWm7sYllvH8Uhz2XNLu8Hb8FvTV3+bwk'
    '/js9zrhfvfiEH7v7a3g9vKSjPTbxiL1KvJ27lImWveHo97yGe828McEfvb+dQbtYjSi9tt4NPYFy'
    'Xr0n7ae8tQx4O1nnSb3faEM9iopgPT5fSL0zpzM9VtWCvdsgojw/0/a8BDptPCQN9bzqODg9pJmf'
    'PGXPWLuvKAO9xmFGPcOdhLxdsHS8Vw9EPRlgfboGFqY8JbgcvRUOXTw84sy8bdNeuqp2gD3gn7A9'
    'E0nSPIL1hz3knok9IfGoPOQZpDw2Ae+89spQPPJo6Tw4JaS6z0sBPbulHb3rNzW71FAivZB/Vj1V'
    '1t28s+UAvbZU+brhvSY9c1R4PZn1abySo8+89kGcvNm/lLsMy0q8x5M9vXhDELyfFti8hEcIvd5y'
    'tTxZ98y8ifkhvXBbPj0yZO68PQ0wPU54y7tS04E9HWVuO7WSq7yrQa88u1guPedAbjzpris9E7WY'
    'ODaCmD2iqeo8bGO4vOrXzDxJnzQ9Nr5gvBpJgDyYheY74OmCvCeoB7vu8TI9rIFBvPbVUjs3NDm9'
    '51yFPSi45LwjP8W8nRNQvZzsNrv5hsy8WGFpPNucRT0u25M8UoEbPPvt9bqfR488HWnpPBD7uLxE'
    'jGa99x9WPaLZJD1D6dm8H5uwvEYQnzqiYFc6iQwVvdK2crxAh4+8EEQoPR5JqTxV80o9iQ8PPe2C'
    'OrynDgQ9nfSavIWtLD0kToW7yNtfPZFP5ruaf787FRuUu48eBb1butu8cba+vF2i67xs/z49wUs2'
    'vVraJr2BMZ27Je8DPTlryrzysK68agGMvR7YGr1kx+M7QrF5O/jAiDzvqpa8rKZJu3GKNr24KZQ8'
    'rA0XvUr3Mb3X5Ma8Iw4VPfBUEL3JdDu9GKwNPa4TPzweUjq9Gh1bPbZfd7z96nu99ny4vOwuJrzU'
    'zhy8JrALvRg6GT0trIK9YI3yPLOlcb03CN48gxNHPeROaDxVr1c8y2G0PHQ5zrvEXFU95icFvdos'
    'Pr0TU0y8OcwXvQUHdzuJGpU6+RFXvWSP+bxFAVW8eEjjPBJlBzv2vhO9jpuAPOIv77xiF4Q8Rr16'
    'vAKayjyIlck8qXmqOF/2GL05iyU9V3zvO2kQQzyI7xI9vpYFvLlBWbw7BAw9QSdwvR6iKz2EcyE6'
    '/TeHPdxmibtGZ/W88l3MPGaTd737WFG98MOHvGpQCL3YXSE7d3eVvJ9Imr1iQSo97x8gPDuJtzz6'
    '4Gm9/yA8vB8lI7sejNM8EIIvPLv+eD28bhQ9jOtKvcRagDxdMUa9gav5OAi2Bj0dODq9YYVCvBZw'
    '3TzWEAy7EldpveIfKL2AAS+9agSGvZGU2jv7KJy7E+YcvZHUW70A8hE81Jf5O3CvxryX7SY9G3gH'
    'PRwJFLwMaPo8YOIaPfqXSj25cDM9j9EVPfGFOT1iLRg8wvwEPRVTOT1QNq48ESYavf97fj0gPEm9'
    'iYFAvYsKZbxLM7Y5IYUYvRf3LTxEzIe6c6kMOxUOQD1Bvy+9azQHPYRtET3XJPa8KjEGvTa4zbxP'
    'cWU7C5nvvMAj57wuhNk8hW1rvTCpOj213XA8f8nFuyW6jL05jNg8GGX0vAYSL70wpbS8Q29JPVT5'
    'Zz29paY79AVgPXWf2TvIpSe9yP6JvPsgZj2uyoO9Y+1SPFv/g706VqG9gxOKPHx/OD2JFn89ovOo'
    'vPAfi7wD2kK9NIDHu6WfTz3IZ7U7Py2KPfqtWT3AyBY4cAUBvDpDST2SAYY8l+EqvOpIyLxIcgM9'
    '+aulvCeN67yqkIw9dPWqPDNRtrymHCs8E6uTvO+RCD1GMRe9oXOyPB9Xeb2O88c7LK04PN8b9bxp'
    'uvU8fjo+vc3vZb1aK+w8PKmfvNPyLT3o7/W81FsqPdgacDsgFfk8GsbMvLFVZT0Bh2A9Sc0FPcl0'
    '6zyMaVe9K8PQPBeKHL0e/Ws8BY6RvGjR8zw3oeu8LFzuO7hyQj2JAJg8Z3IZPb78MjwwwRe9YxDH'
    'Olm3ADzWuAW75HcsPVLFZjz4yAK9/EC+PJe787zFZ0e9CvgQPdBsP70EqG86iJJWO4NGZzzMiV+8'
    'uvnGOyruhr0hVBi9yH4svWF827wOhMG8SgtQuyHQBT20nYw9JZKVu5V1gLsuLok9nNSCPVJ3hzyF'
    'sD28VpqKPamDMbxSDFS95QdKvOlIvzsRgkM96V0DPQL4sju1quI86LQHPUkXRL2L38M7sQTVvL59'
    'hDxz1/Q8DvwOvQymIruYO1u5qKW8vCW4H7xn1iQ8s0/7PEWcXz0HWFe9Y2hWPdUEJL2Z7oG8RR1B'
    'vNdm1rsbAl47XSIwvMDpKrl2TTe9HL5VPUEIiTsYYeq8kZyCPdjWSL2u6gk9StGUvCE8Oj2SBI+8'
    '6TOCvJcHcb34D/8845KnvMT6Kz3GAf68PBOhvCAzkDy41ZA7nwmjvF3nMj1fQBo9mh9fvf2cdbw5'
    'KnS9d4s8PcGbHj1n4g89OfmAvflL7TylQ4u9K3kPPaGPNj0/9iq9EKlZvCXNYjtm9RI9+IiivNhz'
    'qL3uCK27eF/huwjrGr1Woh49hMGOO2KG4LzCLkI9P7VqvSAnFT3UD6e7B9VvPRiMRb3tReO8AjtR'
    'PUmZQD30Pw6987stPcBwfL3lOjC8MfgMPSrTAD0LEYQ9di2eOvIy5rp6vX+8JjIzvAEgBz0EkDM9'
    '41/9PA0qNLyACF463NRTvVEoS71EoW09TFqjO5jyMb1tBOe8ZyahvHKye7wjlQG9Pa6KPYz0Yr1F'
    'ZTA9zhU4Ox/9UL0l6988v8V5PLK5Zb0mCzI8UZppvbFYKj2Oxqg8C8xHuty/Lz2bULU871tlOzZE'
    'przlYz89Z7xpvWCUGL31rY+84fAvvPhWPT17pz69TBwQvVKcYb18zd488CF+OxuKBr2oX6C8dWgm'
    'vZFVc72c20m9RtsKvQmHdr1j5vA8yCsavXnwP71cCDu9TSPaPJCjRr1xToK80CjrPH8SHLyKJba8'
    'Kn2kvJNq97yvrQs9hO7ju2LIEz19gyi9UxyBPYZYMD3Mt0A8/xj9u83GazyoAbC8cq1tPdoqab2/'
    '7wE94kt3vSh4z7wR/bO9OaTJvCHHuL1Er5C9yKkyvXFhjr097ai9bqWdvUvbAT3fcdE8tws/PZLA'
    'rzyN1BS9AP4cvdgmHjo+vmO9wl2UugrHX72/mg29dvMvulceVL1DMwU9bNRZPa3QCj3Mn4e7HRkf'
    'vX0bNbznP9y8wywAvX3nGzyaTyG9fshsPZq26zxMWT49TA80vU4WdD2pf4I9eBJaPUOLYzuWBB49'
    'EL4qvap9WL3/NgE9m/ukO+B6f73Sf/68Q2mPPKczkb3wKLo8cXX7PMVj7zzSTe28AV6fPC6KQbzK'
    'lqK6gdQQPXWXTjlGSq88Cg8JPVGz4DsfY0e9WSkzvIkqyLyKGEe8zlwrvN1nl708IGq8tu22vYCs'
    'pbwGbjA7PIitvGkodrtxg5K8szjJvAWwgr29lWG97aPBPArNpbwbdYK9521KPTC7HD1sUG48wP4I'
    'uyldET3iF289egsSPXmk6zxMm1W9PPuCvYM79rxq+bQ7+57IPJydGz1Gqxy91OXnvL7kcr1jIFm9'
    'esMuPWElQr1YUiq9RgKtu97QfL3Arkc7ZBuOPB0gHj2Ho1i9uBOCvELsyLugGGC96povPQKcJr1A'
    'FGc9ssVEPAG/J70UNXw8jSTRvJVRET3Zsg+9Wgc/PLinWL2SEy09ypqivV5epzxTYgA8FbI2vVtO'
    '9TyWVg89iNA1PX9+WrwxCgK9ptkoPciecr2WyxG9gaISvTHPdTzl2ee5lpbAvCw82TujxVk9NGgD'
    'vfqABr100PY7mxvovI6DGr3tTew8l6gAPUF1rbwc4Hs9tPdwvbyO9Tre4Dg7anPQvFI59TyIbW29'
    'YoHrvKhw8jzXEyG9hysDPWG6b73OfVA9bVbkPPrTND0Axhe9Y85ivVgSXT3apT+9Y0uSvCaR3Lza'
    'rjq9P3BGPSoyN72QO7M88Qm0vJpbBz1FvIm9/SQCvU3cXr0J4Sy942AJvXSAdjtVQgo9Za+6PPDC'
    'cbujiX08iIiqO0R2l717DSI9Ph82PKkdWT2hChg8bn3RPHL0Jz1xXoe8AQIsvTnQVLykBqm87RSA'
    'vF92Cj0NtgU9rp4MvfSJVryeF5c8buSbPEZ3wbvnk029emp0vSrsojyzOw+9HvWIvfYoC70p1Jw8'
    'QWoovQpzAzxwAAC9RaNnPWpAc73Zdxk96JOQPKUxJTxL21C97umBPe4BmD132Vw7+vYmPcOQHzuT'
    'PXq80yXfuS7siDyJi3M9uosSvZUbqzzBByA9NOggvB5z/7tzuEw9AlcRvQGSW7zTwPK8bbMUPYE9'
    'Lr3/dUy9CrzYu0pdZTxFcWs9oMg6u9s8jbp8foI6cHg5PQvlPL3u4TC9rkE9PVLezTzAxoS8p0rw'
    'utLGYzsoj2u83yRRPWacCryzcxs9KsJPPVIgZ7yTAGk98FN2vPZwgb08RlU9jWCLvIn4MLwMIl09'
    'PND5u1nsZL1sv8G8kSBGPJW9kryYFJs8SpMYvQHk9bxLasK8ZsvgvFBKDb0gAiO85CbHPH4qWTxj'
    'ZvC89IEjPV5Hz7yV9DW9yQoVvWnAgr3A0688UBexvOEXjL2Ebuc7lA/qvB6tG71zK/e8LqQgvK9v'
    'Hb3vezM9RneOvKjR3zzRkIO8dQGXveDyMz0/a4W8IRtMPefk3Dyx5Dq74PVSvNvXoryZxZS7V15G'
    'PTAp6LzURIk9l6+Ruky9WD0bSzI8UWk+O8Zouju/i8w8j4N6PZdKij1l6MS8fPPkvBoYrTyK/ga9'
    'g8WiPKEY+DzfeUA9CerJPIJQRT2jCi29wNJzPU3zvzwthhq6+T3NPFit9DwiOSO9anuePJAeC73M'
    'Ghu9VvIFPbG7NT3JiTk8bJtmPMA6rjweHlG9ooU1PVpfvTzJkg09bdp6PaBXXb3dDsO8Gy7OPIqx'
    'a73gqCO9FwJ/vVDSPDwC0yA98NMpPbYGDj3qMNq8o/lXPbkVhL1GTpo65j/tPOJYnj294Ks8mfwi'
    'PaCa6jx+p0S9TWV6vbJVKT1LY1c9JGo1vXYrGD0/fEg9jCgevIhvLL0z8Po8pwdhvPV1J73QrWu9'
    'qHpwO7wtKT05Lyu9QplRvbrADj3ujmE9uarmvAkg5rjI81u9YNPNu1D6Pz2esnc82pH1PKDn5Tww'
    'ViU9DzcXPf6pizxuzVK9NfUbPYI+Oz2/EDq8IH+cPJ6ywrwT0Fw8wjWYvcD6Db3AzDg7cMQaPDQB'
    '5Dxjae68qytavcLxYj0JE7E8sVV1OjinIr1uqI48Glj3PLTxIT3KIKg8ViZzveeOFj190lU9+Qgz'
    'vD+9rTwvczG9AVs4PRXChzxfgme9J4HwvH5/UD3ncKo8D9yRPCjZXjypKL88Q3ebvWyFLrt4R4C9'
    '08k3vVNnOb147j68Q3dnPbANyDvGelI9rypSPfUhUDxUVF89NlAtu+6v6bzaNf08pll6vE4lrjxt'
    'BYg8e/I1vJYkMTxfkgg9nSAPvNbaLT1B+gw74NRkPb49NzzDuFG9X2mJvJLEZb35iRU98ce1PaNp'
    'Czws3gW9KH6WPelcn7ooQjk8/3iHPD8tVj1gtSA9NjEbvcwJcT1vMA+9hjfqvJvbeT0X89q8IEGH'
    'PLfb0LyTWPE7RAj1u4GuOT3APzu9etUmPZipHr2uQ8C88YW0PEozFbzi5Cm7O4MKvWXb0rqI3U29'
    'YhHEuxc/Qr1bF5i8mjY4PeUWSb3YBPw8oVSqPNCSZz1nJ4E91DhKvf84DD2A7le9EeyWOapri7xl'
    'nUO9bkEyOuZylrxu44c8TK0Fu+haU7wYfMC8ipR3ORNHDDzZ3D284OVmveMgiD2Pfti8WnzLumU2'
    '+Ly8QQk5M6FxvQC8ZTzjEvG8AOTTPMZT+zzk2pe8fc5svYGmbb1tSoY7Q+pKvXspizw5x/S8bGqs'
    'u4svhroMndK8u/q3PErlUTycVaC5wk51PHE+AL0tixc8AYg5vRVoLrw3+T48AzsXvRB7y7vhIKe8'
    'XJFgvep4ULu0QzK9jwKUvEaROr348GC818tbvRNQezxb9Ry9jXsIPXqUgr1eLCQ9fbL2uzlAATzt'
    'dUM9flKZOQlOBDvIjI69gO2AvL9YMjz9nLS8gDNMPWsza7wjkHW9KgJlvOLZk7yKuKO8oxiyPCzX'
    'hz183kc9q6xdvbzebD2yNFS8ZOlbOqKCJj2QiRS9nRK9vLWSvbw/GCq84/W1vDTSa739FfK8nKa0'
    'PEfBF72nxcM8jZ83PXAlYr3PHVU9Ki8HvefF2jtkHi092lhpPfA83LvWVak8jv9MPGbJgrzZX5A8'
    '8KEAPTYCRT2gLg+92chbPdn5zjxUkqi8axCCvIujiDyUzfc8OpzEvMKlTjthVXa892Y9PfqoP700'
    'zzs9Q6/2vB/HyLsGEm89jiLzvBsBTb0AKeM8ZCnkPBZOEj30dOO7CqPmOygpS71lCSa7VExGveE3'
    'ST0f19i6lwAau+ZCxjx37Si78dWtO4NUD71pWUW9wvp9POeZobz4L3844fUHOrrC2jq8mHW9T9do'
    'vUWyVL2yEws8kaGAPeTq8Lw3qtY7SnQnvDXW1Lz/bHI6quBJvEaXqrwQ+so801gHPFffjT0Vb3c9'
    'QuszvKfJLr3ny4g9HnafvFWbgLygP/M7wizCvCefGz2Xh+K7099/PRbeUjywawy99M2sO8IS3jxM'
    'LxK9QlMMvbGJK728Zuc8YTAPPc4H7jtdln29OtmUvXAKkbz2kTy9z8QoPR5YLDt07TE932NNvWQp'
    'kbyTgN685eiGPNX12LzsTtI8jlK7PBX3Y73kPp+8Awunu4vhQD2LS1m9X19OvXR/Oz2H84m9w2+Y'
    'vPsvED0DQAY9U5OPPPpBAz2DCLq8EKbrvNnzhzwlPQW9GpgMPYTLvTwPk0k9sb+xvGQXAb2gCoA8'
    'MeuGvTRMWT3pgge5fwtmPOCAIL25hBQ9v9NjvUO+RL31psG8oQ6GPMvueLxgtu48kuFlO8xdgrxF'
    '+Hc8jSqZvFS6ZbzKBDw9PBCfPWPIsTySlhA7pqAavWu3cD1Bv4a8LbeLPf4cKT0gRis8ELRAvGx9'
    'Jj2CSdg73lrvvJ+EJT0t3Ba956jHPJsYOj0xt7Q8rkcDPd3FvLzQ2Wa8rWkuPS/QJz37XNC8L0Fl'
    'vLSR0zyV95e7UFMnuywtnL1tKuY6cp8kPT3SpTzBTB09FfxevWNQlb1wwEK9TyOHvIPOhb30mDs9'
    'KCQiuTMe+jzK8i+7P5lFPZqrA71W/bG8Op7bPLA97ztKDY+8nQksvbHW7zxuRAO9NQPoPIqQpr1c'
    'Z1g8zplbPXSSgT1Qf3O9yqBGvV1LvDwaox28Qh8CPFJAULwoMQW9edqRO63sC73Q1SI8ebsvvdFK'
    'Cz1kdoW8Hv6TvMYejj06PBk9vM+JPIMPkT3rJws96Ne/PMAUC7u27386YNCevC+QzLwKDhQ96EGB'
    'vVHsgL1rWZ+8S+wKPdOC9DxGdo884j6Xu6LKMb1ThFu8G/QsPe4Frjx9GRU7fU9tvL9iUz1vmPO7'
    'K9Fzux5poTyUFy29gTd+PG0Z2jz/1SY9GwjSvERC8LwP23M9vlWBPbFLRL2OR2s9vO5lvc2oGT3M'
    '6G+8GHUGvdJVkD1ZpPY7bMs5vWWerLxWku48/jGePDXBA73Gmx49KZelvNwRkj0H0vg8yGR5PZ5E'
    'UbtoRTI9hIQFPcwBcT1U6EQ8Q23jvFwwrzwciKc6AYAOvO7kj7u82I08nIVOvJeBLjv89oU8rm2i'
    'PNzIkz2Pmc688IlcuwewHL1kMIs9/YijvC5zjrwI5FY9aHRXvX/4Tj2mmWC7KoiEPSOELr0TzAK9'
    'DNZXPZ5W/byiz7G8T3+FPO0fuLyhmn89ABtuPDedAzsUuIs8iw6EvVkPMDzIlhE8nCDUvI01yzth'
    'tvc889hxvXQ2OD1VPRe91ONLveMMLDsnCjy4yDV8vEZEwjyyVYK99XYuPVnFMD0jTSE9rS5VPcmQ'
    'grz7MGo8uQNmvc1fdD0WHBC9M1Y2vcE2ab3IAFi8FkxXPQvhZTsIhdA8a+c6vIM5EDxVz0o9U0rM'
    'vHaLzDxNss+7bdv7vJyYzrwcFXm7Pa6sPNw1vbzKrUe9bQmQO5WyK7u9od6851BjvY/FSr3WUgC9'
    'AT9VPfeUs7yZ5oC8NHovvdW/JT1fdCy9K0S2PFBgbrxjgES9LmL0PA0WUjvTNs+8YWuFPNgYIDyK'
    'zh29V0SVO0ipnjxpNIW75MS+PIZ2Az0t+R+8ifhvu9gQ/zwGXoK8odeLPbMQHr2NQSA8uHpOPERQ'
    'ZT2kWE48LE5YPQOFLTwXLQk8zs2BvV7UT7s37Gs89Ja0PEiWiDzKJWy9vcg9PeJ8QTxxWtK86rst'
    'PRS/GD18XgU9JQsiPaCTGr0hz/o8Ap29vJl3k7uy0ME7xVxivZc4qj0ToQg8FaLZPG5fhb2Ii/U7'
    'TUhzO7ddJz0AbTu9g651PMPgxTzUpTO9xFU/PU7nZrwDMaU8hahxu53jRzyl84i6M5ifvEQy0Dqf'
    '4e0708GgvCP53bwrxRo9JV8ePcUTKr2/wII81U4HPYwmWDwkgT892yfdPD+4LD1WAn68caiEvBMu'
    'Ib0jXbg8AccVO6uljbwcNi09lTZ2PYIqVz3F7p87Xl1evRznkb0kIPS8+Mo7vMzPdrzvNMQ8DBsM'
    'PQvIAD0RqsC8NM2LvSM7PL2szry8914cPLbmCz3JDAg9Fzv0PPQwPLs3+CE97xoQOwsTq7tgMVe9'
    'GaYCvduTmzzy2a68yP5bPZ2CALswgZO6hDb9PNCgU712XtE8h+cIPWrVwLyo4249HFMAPL0FSb0m'
    'SAk8hRAavccChrwF0vo8l7p8PA4HPL2gRxI9BWLxut1gUbymHY68/F/fvKohNz1lxko8+KbtPK53'
    'Vb1rMkk9rPeDPY7ixTxjK7A89jenPGYrWzyU6Kk8JGDQPGUv0rx9tr27U0BqvP9MUjy/r/S87XaH'
    'vQw0ODxhhjU9oLIqPYl6Ljz5KDG9RVFvPR0TzTwAeCi9NH9XPQyxMr3w8I68r0nZO09QHbvPOME8'
    'lR5NvXDfBj2venU9EaYDvZUdLj1Rsri81QICPbkAVL3tuHo9/hvTu6wMg72Y+gk9SOzlO1t3zjxE'
    'cQG9E/GMvBrsorwZnXe7TDdiPFF8/zzcXp47LWxxvSgYBL0vFYQ8tsYjPe1rUzzvXTM7kYyHPDou'
    'gL36W+u8YeUhPRjkgDxTlNC7zmSAvXXHkzsrE4a8IeNevPwQ4jyA8DO9D7V7vGg5fD3Q/Lg8xf0P'
    'vQpvrDyYa567JtyrOurJM70wMOC8GCC5uy9KIj3iSwc8tvVovWeIWDyFYfc8ZqMjvVAWubxkEaa8'
    'HNBAvesEZL0YwVI9ne/9O5s5vTuDx1M9gBqEPFAi6rrKPJe9R9mrPLzbTr08nrk89vqvvK1u4jzN'
    'g+A8CO85POrDgjuUxFG8bMVWPSHleL3nC4a85tpRPLYhGr1wEhw9iYOVvLkQ1Lt7qAM9erIUPQHl'
    'TT1xy0W8qXIiPGNVBb05gpy9B+QAvbv+bb2uGNo7A/qwvOepeL1n9k061RSGveCsPjw1a/e7xhgv'
    'vNm40Tq/Oxi9HCkVvSfhY73FIyc7dB1kvO1BKL0etec8FefwPMOMVj3msVE9Qw6/O53UHr1Z5oW9'
    '8I3NOoLyIr2ZHEg8WgMavR8imztU0TG5/isuvdSQgLyaM7i8nbyGPPMxeb3m05a799owPf/p37w0'
    'v3e8+A2lvAy5kjoyHUu9LSI5vRSNOj2oz2u9hxatvHauPbz+rns6/LEyPIPLE73pToy8qgmAvU4p'
    '+byYFKu8dfpnvbdECDxUQUE9QKTfvMAYNj01/wq9wkOHPRkrJD0zIHo9NSPXvD2Bgj1yYpq79ZvA'
    'vIfy4Dzc0UG94aRDPHrf7Lv3t5W8FBRuPSHk07zZ+iq98cNpOxVxhTymEWO96esAPXnKYTuaLQe9'
    'P6yEvGs9Lb3ZDDg9nHfuvGJuZjwvUie9ukKiPC2BDz1FCK28qlb9PNDIFj3g/EI9/SEIPXSXp7zu'
    'Q0y9iyGqPF9tAbwLkoK86kCEPNlE9zwkiDO9iYWFu3f3LL2CoTY9TD0tvdm0Ibx5sek8+2CavYbE'
    'QDyQhm08E86/O5TYXzweV1W9AZ8vPC5Moj0NNbw8BNsfvREzSbnaNUq8cx43vbzWjLv3Fbw9jOMe'
    'vb79Gr10lKu8R+8BvWJ4gz12KWw8hg5+utZdHD0SUv473mcXOoj3c70I6AY9/Xudu5Emfz2XJcU8'
    'BB+6vCUAVj0rCcw8OIEUPRC0fL2hMfG86KX1PAo+NzzxND09PcQSPThUNDyYFks9CDw3PfA9Fz20'
    'aZg9lbEYPdtVND1csBW9jMAZPUL+ULvgDqa7OERSvRoVQL08/rw8SwVhPbQYG7w7T349+XlJPILd'
    'KT1z+qC9GPX3vHhsrrtutYc9JrlwPPxdAr3DaPG81ysDvNJeEb16SQw8DPAKvX2uZT1LR2A9eQrc'
    'PNRGrbwdurc7E+bHPJOSnDtQe9K7I7KxvHbXFrvbyma9PWYYvTOSDL2Tnkk8z+drvXJBwTyDw3E9'
    'ytcFvfu87zz5bBS9AUivupYSoTy3nYk8jymhvK3hyjsm8eq8W3AWPZTK+DpnDm+884VPvMyn1Dw8'
    'f928i+dZvBNjFbz1eUW9TIE3Pb/Qdb1SGA09YZgxvaaxqbsSa1q8+h+yu4EBB72dg2m9s5kSvZR6'
    'Tz2leju8dcHPvIO8fD3OGQo9oDgQPVkVDr0kgY+8jbm6uhW4Hbu+DpI8nCw9PVuZN7zuUO+8mxTi'
    'O2pPEz3f/4S94lduPKeE+LzAfzi9rmYUPa4f4rzIa8w7hWseu91WOb0Fq608587YPCQXRDzsHVa9'
    'aiMOvetqmDx6XKi8/E2jvEevhjwYh/+8hDFbvav2D73dr4O92J6Lu+rdMzxQB+08I9Lpu8vhszzn'
    'GSE9XoouPXojSz2pRjI9JP5qPML2iTxGrAO9vxFLvRKpw7yYKP48a8JcPZvv97zBLz49fuzXvLsx'
    'mLsf5A69YV4JPcsmRzwGa1s9KYyFPA4kCjzvfdK8ge70vOq50TyVE+w8voPaPAHs7bzR9G68qHI5'
    'vARtj73ar6A7Lf5auyfC5rzAs0S9I/UjvaSARbwt7Si9u5EvvCsTKz0Kqk09OKXrvMrbW73ci0g8'
    '5tHiOv6CRb1DPQ29zR8gvZe/hDuyESO8vFIcvcsz/DtkqH29OtC9PP/4yjtl6cg7PDrXu88SCT2F'
    'oUe9FF2AvMVjADqXgAs8Ppasu84H0rxu1YM8NTfUvJ9zpLvCyMe81wE8veCpMTwiLbU8cAnnPNK7'
    'y7ySsnS8ydsSveXugDyvRKs8RQeAvWlWTb2h+327zypSPT6mZzsWb8K8/1wwvXQYgjwQmOK8sVMh'
    'vTc86zyrFHy8btbVPIKJFr0SFEc9TH8+PTpTLT0Miiw7qxzTvFOGlDn4O7G7+ej1u1E3gLyIQ6y8'
    'kVEKvORElT3YokI94gZ4OgaSbL1G5pi8ISm1PG4fQDyHFRi8ZHKHPTwBIz1iG8Q8g9ruPKHrIr0C'
    'Qe+7rwPdvPH5Db0KhYc8a4a6vNzWbrwHgN+8Wp0cvZ4zPz26S1q924cavWvaV72jFRQ9Hl4uPBnQ'
    'Jb3aZMU8i+zBPLfthj3taca7DQcXPTlC7LxUCkY9ZATLvJIfS73QsEy9W3E9u0/H47z1sKA8GGpC'
    'vS85DTtRGk+99iEcO/AdiL0Gjzw9HkqFvf5/Ubzk4Eu9IaxTvV2eXzo9leo8kUvSOrLEBr1y0U28'
    'jkg1vcngBbuK6EU88rzVPBVi7Dx6ZR+9Hf+CPBFXSD0RORE9kHspPTFqNr1C+wo99s6yPCLyFr01'
    'MOW8SLqEvJRI1rwW8IO5Rpg2PBy56TyWzV09sinlunUdRz2f53e9W0hXPb2//Tz5TWE9Qnhlu6nO'
    '5Tvp1Z05oQo8PW6JPz3toSY9OupBvAWYOr3NlsA8wNOcvO5vwDuUHxe9nna/PMk3ar0WcHG852Hd'
    'PCePxzz6ZBU9r6AYPacZHr1mc3285OXYvHTeDL3J+xM7ZYFJvVuQOb185pQ8XjBjPUlPJL0OniC9'
    'BMpcvIdYdzyEOV29n0BcPQXrNL25U6m60KEfPU1zSLsb7UU9dPa9vMOKRz1GFeo8FdU0PcMnvrxe'
    '4wE83DXBPC52MTz+G0O9W+XYvGgqNz2eBx+99g1OvI68dDxqp0g9+ixmPRvdBT12/aI7EPt3vYb/'
    'bj2iDmc9oBjWvIh5TLssABc8TSuTvKuL8jokyRK7EPMevcMyBb1Ufz27gSn/vDthWjxOfaG8mQkp'
    'vbAdFjyLoBU9EZThPA07F71KY6G8kiY/Pb5ueb32bjs9bY8Dvb2nLz0XhCS878xjPVTz2Tw+EQ68'
    'P51BPJiLKr1NyjM9KngjPAOkOb3sYIe8yymkPC4jPrzvPSw9DWU7vbsjQD0ozyW9nc4DvU9cbrwb'
    'prq8Jh6DvARYrbrGQaW8Mom+u66c+byMCkw8ZDYRvCENLL2WB4O9UcUUveic27jlnEw9m3uPPIsY'
    'aTwJYpE87c5gvGKcEzy95+K8cqliPT5y+bxWChC9twARPbe1vLzkHTg9a3dGPcVY5TwE1cg8UsMg'
    'PCFDt7yiJzg9U8eguyKzo7ssyCc9hxaTPHJybz034Yu8utk5PRlBHD1Wt9q8ZcUHvSlSo7vCCDu9'
    'xq6SPDZVjr0qImY93ttQPdWy47weApQ8dkUBvQaTSryXc6U8mG8yPAN6/rurfls9BHMHvZ24DDyy'
    'H0u89vxYPSQ5c737sJ088I+Au4/6fD2MaCk9D4IVve7uFb3PYtW7jA2kvBdmCz0/2TE80osTPH5G'
    'o7yjTBO9NnwpPNzRlD3KKc46wGnwu/V3mb2OjlQ9xCphvEwLVz1SZhC9qFwIPcvggr2N7Gm8Ex57'
    'PQ8iAzwPyFw87iTdvPSMJ711Qrs8MYhVPYV8orxqlwI8uLt3PXAOBb3STzc9lk4wvVkD4Lz3rwo9'
    'tscQvBnBIb2xQVg86XxXvdCDV72IijI93uGGPaa2K73ttUK9h2JnvFS1GrkimS89oo5GvTKG4bx0'
    'PF48lFJMvANKir3TVOo8bnloPXnKMT2B+rM7yxpQPLziTb3Lepc8mtBtPbbc+TxPOIC9KfCGPJsO'
    'dT3fHyw9QWU4vDozDbyLMhI9ilePPJlUMb0kw307RdjlvIX6R73I5oc8nfQGvTl1nL3lkwk9FlUG'
    'PEkcSD2xTkw8t0USvYeqiLoODK+5ELcju5VgKLxzIWG9iYP8PA9NVb2wmlu8vyI2vZcKkr0w81U9'
    'jWYrvRIoQDxL3JE9zBa1PBr1FDzY0um8PwCJO69vJjwucmm9NuVxPKLuCL3tP4I8as5WPZNJOD3+'
    'n0K8TW2IPKVVxzwKmxE9inaou2JN8TzNweq8GwyMO9JTPTwD0OM8thsKPcEwKL0lvRm9M1aZPFVi'
    'Ij1/fhu9iQJ4vTuZSzxjFJm7kpE4vJozRb1Ldam6LU9bvf8aory8RDm8eD42PM7n+zzY8O087o1x'
    'PGw4Ij0Wvne9l61uPSptnr30mNC8+ii6OltvlDxFbas8/xQmukk1sTt3Na68PCz0vAVBizyrZ808'
    'luMLPI9eG7xUK0E9PjmCvOgVRj2PUjW9y5ZWPcFROr0xFBm7P4ADPZPyIT1/IqW8+vl7vS/jez2o'
    'nwO9jFwDvXG6XT36Tyu9ZSI/PKrqP7337Hu9a0usvAlFDrx1kKU8vPgDPcu+JD0756W8/ly4PBWa'
    'DD0/YT89JQC/vKJ8gb25qza9RO/jPKGKDT03KZw8l9MoOjNVnryJxOi7YvVMvY2lOrwnN4a9Y94o'
    'vV60C70SLy09RV4cPXPgEj390hM9CPrvvHgGWzwzbFw9L7YvveL8IrsFGUW9K5a1OcWE9TwcOZu9'
    'CTYVvZtEGz1dEiC9b/N5PUxqwTwf0pu7+snbu0y7Krxsyj69nha0vHTUTLyryt681fbIPO2VaT21'
    '8WU80/V2PG7AjL0BA9Q7wuxZPee4A709WRW8gjB4PSr6STrNXoA9vjdYvGoIib3O0E+7ccIwvfnl'
    'A731iqc7VUpXvcNMtDyjSEc9p1uNPRj9Tr3G7VG99PHWvFHAIrtUjw+9heF0PdahQj0BJjI91MF1'
    'OoOpd73u4wY9lAOAvHh1/TxAuE48E3+pOwwUZL0t0Og8CB5zOxDkVj2rmzo9HaW5vP5nZr3XAE69'
    'ckw0vMPznrv5O3E8mywMPWgTNL3ilCa9erSKvRkhIz1G18A77jwAPUs9G72zzt284mtSPfk4lTv2'
    'u0W9cz1FvJA1QL1sD2U9k0s4vYnRUD3cryQ9XctcvNCdZLyBuxO9pd5UPUA2QT2RoSK8C4yKPJO8'
    'KD2tYpy8GPQCvAbQWD0Js2o8nlJfPH9tEj2jlFW8LBLkvIu8YLxdhiy9rwExPVhD0DwY/Mq75qpF'
    'PbSiuTwlkWY9SUeKvfKHjb1iZ4G8VtDDPDzi2Lx92848NOYavIimIz1qWwk9+QEXPYYl0zzV4Iy8'
    'CekSvd9qADxblyE8N2QvPaAE7DvIJ2C82cogvGtvdT07qSK9gZJMPKiLNj1LUdy8l4MsPF0VTDw3'
    'j6I89TJVPbzcVL2dzWM7HGLGPOTAPL2fYU49BK0iu3RXwrzPnHa8PB8XvXxhqrz2nSS60Dx9PH65'
    '2LveDIa5G2+NvQ/YKLuZY8y8S5X1PAjPoLtLByw9/RgFPVCCkrznc8o84DC1PIZoubxBbGi9Fhia'
    'vEvBRbwiaGE8Zta+PGN5ZL1tEFg9m/6fvKlWu7xTaTY9uag+vBIMPDtvnL68OeyAPS7TI71xOsS8'
    'uH+bvc9pMTuPeb084KkWvaAy7TwSu/Y8dtGOvf+cBj1Td1C9RzmBvMsqOr1jfPU8Pu0KvTA6Ubx6'
    'imu8HkM6vQ3nGDzXOOq8wA1ovW0Na73ZNpk8j8WRO1/v7LxiMQA9vqCgvSxSQj09LTY9u/4UPQQU'
    'Wzt5xyC96h83PUIDab2PJYq9Waw+PFAjJb3m7qU8G+1evamxhLzEWGK9afZQvEDRXzzFjia9meA8'
    'PPBL3buxoDA9kzlUvWvN7bxabju9LYWFuTSWKL176W49hdpLvAPWoLvP5ba7G50cPXO3DD3/epy8'
    'pW9iPRdUPT1mI968FpjsvOKiCz1YtXg9I61QvVakSb0g42w9AaSCPd2ubzxN2Ou5YQDGPNckSL0o'
    'RPc8hKFLPDpHD72xH6A8DNMxPWT6rzyiTyk9uUw5vUQ52bzxzO08wWgSvOiHLr0095o8IqeIvVAi'
    '6brC2gK9ncPIPONygbuEcrK70xwGPY2Alb0iBDs7sGV0vbH5BLz9z4k8LEZtPfm4L70NBco83FuR'
    'vJJfwzyxIFm9OcoyvV8nez086nS9oqiQvZt34zzglek8yLgDPVb3O72QBJW7hYY6vL5WLr1N9mO9'
    'exaPvFeBs7wyPGW93lT6vG/xyLsmLnU9deyrvAWUZryuXrG8efCFvI6rFb1JQa67fuYSPNxHuTtK'
    'JRe9i6sdu2Glcjw+GU09Yzo7PQNH0bqur0k8sSBJvaPOe722BTU9YCRdPdnterwm5FQ99nGEvIP1'
    'izyBXVo8Fo0ePYMsmLzxy5Q8MF+1O74ifrwScjW9tNwPPXgXSD06pDO8U2xOvajv3DzwUoc9e1BI'
    'PRAEa70D/0E943lOPARzyjy28zg9zPlQPC0b9rvSmsa8NRr6vEl7ZD0MMCM8UkqxPLC0o7xK/9c8'
    'j9LBO/o8ZjxiNBg8EjdBvW/qn7wdm4A9FMzQPGaPED23/xG9MPDvvFtNtLyMEAO9yZhuPY3Nozxz'
    'fAm9mjdxPOhRTDxPc+s8nVvHPGkPsjxQvmi9AzASvZ+tuDyCo0s9It/mPGF+x7z+I0q7Y7J6vQA2'
    'OD2EaUq98rTOudQqqzyKxB09mYXSuzeNCb3lcnI9OSVCvehgCr1QJxO7+fgtvcpnETxFXg+9fy4g'
    'vZT4Jb33Ojq8JThAvaVCvjpu6Lq70X7Yu3CmAL3pFHC9XfTiPHYofD38Vd48oIEcOwcT+Txt8A+8'
    'FLaevAGQhbywofQ8Wh8vOxekRz2wmyA8BtjTvKeZ9byZPka9Hij3vP5n5ztyzGE9b5QYPZZ/pLy3'
    'UBA8ywLYvAHpHr2Cl+S8/aY9vWnLZDqJfyI911AEPRQDNLw9VFq7Op12vFF/Rz3Locc86Qk2Od46'
    'P73/I2K9R660PDpZRD11BAG99e/tu0YFXz1+pzK91STQPN/qBLwZ7Yg8+KbNvK0JPbyBX1s9ECne'
    'O4Ld/jwDGok8SsUQPZz5YDtJRpI7JKkkvetIIT2v9Im9ZZrJvHRBMz1i21a8shfzuxhNAzwVCDM9'
    'gOpbPD/WcDtqkUI8u4AMPYtsrrxhpNy8cw8mPRAcxD2wkAI9PhaFPQbe9Lw7QyY9PuRxPJ5hRz1e'
    'BVe9lsT5OzTa6ryhq2C70/E0PC5jsDzcybi8GHPkPMoGND1gT0Q9x0+RPe6gWD3oPMa8TjhZPciF'
    'njztYO68nJMjvMO8v7xZvja9vFU6vFtjvTyZpnq83CIEPV/7DTzoclC91oXtPHZu8LuLoDE8vnuJ'
    'vbtJGz1Ojc88riyuuohFFj3l/TK8ZS0iPYooDj1JVTS9+VlyPXQ6DLq3tGM9CalXPH0odr2tII68'
    'vAUsPKi6gbv28Zi89nl8vUG7jjyMKj09HK9PPWI5Bj3eSQO963PKvFWXPr1965C6hwhYvZ1gpTuD'
    'eh6924qFutqXAL2D37u6rfZTPMKJjjzjnT496gIJPPFmbL13s+E8CFPJPOv8kr3Qdqg7y/TgO0WZ'
    'eL2oHSq9xytNvWBj4rzpgiY96UIOPe2tG7y5zIe9OCCIPPRRAz2Gk9c8Tk+yPFHkyzxxZ4q85YhE'
    'vSBblD0UnCK9xwdBvJF5Lr3UkF87xMwevKIGwbu9ESu9UyR7PUikdTwUR149nfvjPD13LD3DMO88'
    'oq2ovO6yKbxXcaY8WvgxPTWdZz0N8dq8r4PuvO3L5bzagic9eHSOPE6fHL1fjo661ldzu7bsQLuf'
    'iQm7+K+bvYW6Zz0jI9G8kuRyvVaMqrw0MAS9Sc4HvbDwnL2v3Cq8u+LxvGF/sryjel+9F0xNPWRH'
    'a71Rkxw99raEPcoerrvkKg686+46PIWx9buJRzU9ljd3PKkiezvt6mg8JdwLPX8nuDw0axq9ZYIX'
    'PSnfR71ua+68zuc4vEDEybp+XnU7TOOMPZc/V71CpYG9SlsjvSbcLz22CP47wyeEO1fk7ry3uv28'
    '4ClgO2DMC7wFwAI9FYpwPTtbNj0r2Mg55zffPJGGLD2vckU9S73wvAv7lD2+mxG9u6SqPEIvsbxD'
    '3SK93N0tPVc1RL0T0rc8zuzOvLgfCz20n/u8/Zo8PevQ6TwpeR08DDeQPCjyujtXEo673cj7PNTa'
    'BTynArA8kIXMvF2yTz2IaQg8olkvPbtlkjze7fK7gL0JPQ1LCTzzwjQ90D+evKcynzvbbCI9GlFK'
    'vUN0wLsJbze9SlRCPEysIT2AYAC9N2j5Op/xC73C3hi8zAguPJ1SjrwPdLw8khIvvTwFIT16wyA9'
    'a3FbPHrNc73Nq5S7RJcDPfVy3rzS+Cm9HCr0vO8NgjzNtve8q8J/vbJ3tbzHRx+70V72POqrArzr'
    'amA82n5svdVujbyrKlk8yDmQPcl1er1tYh09PBMwvbxIeb1VqgY8KDHwvLxNxrx61Wg9QI4rvRXu'
    'RD0kFD69JrmPvNG7kbu4giu9VNInPRbsRj1RiNa8NoRJPaBSAT0Gsze9qWPmPJvS/jsnYX68LLtm'
    'O0t2oDyPqN07zSiXvREs5rsSivy8oI8CuysnwD1F9hO91R6fvc+lWj0ce/O8+tK9PEptArx1U1w9'
    'Ve83vVJ+DT0VJVw9CyxSvc74E7zR4Ha8gzbePJ74QjjuyEG9/2KmO8i+A7zgxVQ9jXh6vYK4m7zp'
    'fHm9loGdPJlhkTwgvIE8NmMUvTcPhbwfbAG9GTsjvUtWHzxR2We8zBaXPYP3lDz53g+9RL7vOtM1'
    'Qb2WdBy94pP6OyQSNj2YB+o8gpoMvTA7YD2kS0a9451tPdyWhr18+Sm9SChwvKh4Br1n4CM9tMRE'
    'PXa5JzwKD5y9VLwGPbhfq7xxji29cwLmOzdJcr0VJ0o8MvcnPe68uzy6Dwc9YFNcPGnnSj1dxxo8'
    '3fLRPFPlEb1RuhM9qvzuPLgrPjnRqig9IuTgOZJblb1C25G9YsTXu7Ev7jxsSXE875vSvG98ML13'
    '0zi9XsOoO3EjJTwn94K9wNECPKoBZjz4tjK9aW52PdJbbb3RNUo9UiocvUAuLj1Kkka95tI2vHoo'
    'e7y6MYS8F46APWyA5rwbsiu9acw2PQR2IT1NWe680UAyvStzurxtQy48kY8TvaXwQ7yUMcq8paQ+'
    'vY9jBr3AJQU8h2WXOx1p2TuI1Ae9wbqVvMHnXjxKfb68WnCiPFk85zym+8O8qZmHPIn+NT36uns8'
    'hL5HO8leezup/Yi9OUdZPQXDjzym8f28QYLlvJVEvbqu7J487nlpPJN1hzzWBxs9Tu4UvFoBND0Q'
    'Fsg8jh6AvcxpPzwFggo9JqTvPIWV4Dy9j5c9og1HvBKRFT07UKI8T5hCPBUrJT0yRRK6fyhlvWcm'
    'YT1Ld5E8gNgXPUGRCT3yjA+9LEOOvHkolrztJu08guMlvaKEfzy/IiM717RePAn2Kj2ws+C7ozC3'
    'PDUdcr3zvT69JKJHO1/mrjwuzSI88/i9O7Yauzw2SRK9GlMuvQz4nTxjxCW9wXfVO6CW4DzbBBa7'
    'Y+h1u44bHLzCJVM9Rz+4OwDuprz2g+A8pieKuwO5lzxVNys9OTg7vXq3LT0M0PO6MFMavdFCX735'
    'xB69RnjxvHiEhbyqJ+I86hmcPJqPHr2716S8brHwO9UrbjrDPTA8NzNDvbruYb1po9I8BKdXPcky'
    'BT1v2we89EthvBfzZL2/JEo9HmUxPQvDQjzE+BS6Sg/fPG1K2LwGmWW9D0HFPBlZXz34iNW8O55f'
    'vQjOET1Dppq84fy2vLaKtbw5r9U8YpEePYkf8bzWHLO8o2KfuelGD72w1js787tPPWHgqLycVJU8'
    'ZBLWPNvGJL3U6Ke8GN7UPG+lRz2BJEO9n8ccvXoTFDyPpzS9Yl04vCAxMD3o9Ye83OmEuugPZD3m'
    'UFQ84VS5PHUV0ryhdwY9oTw/PA8OBj1Q+kw9tpAwPSIpFb0iRky9EC13PX5lyzxL2pc8fZwHvBjr'
    'fb2bUGO9lGgXvYUXo7wPnMU8sUpDvQqwiD32ayQ9ClNfPZrZDDzvEuw8pFrzO1iSFbyfBBa8qhUR'
    'vbVuQj1Vdgk99DpCPQZcWjvFISY9OIU+vYvXyLy6HCQ6UO62O2nJbz3fsLq83/zZu7igQz0Te2W8'
    'xu+KvfWCeb3DriC8eCQwPdfxnL2m8U07uxtfvXoiNT2Tqyo6LrkEu+3HzDx05zY9dvSeOXfjab3Y'
    'GlQ96SKFPFbZmrwQe1c9acABvWwdGz1UZzo9lXDpPM8c/DtbuX48OfGWvBgcl7yUh0s99eyMvEYS'
    'aLsgzua8ItEEupzjQD2Iwpo8XzslPZuu+rx4duY8Cir8PLaHuLz5zaG8xjR8vd7nTz2yK5Y73DmJ'
    'O0xB1bwwO3K7df8MveLWrTxkg3G9u0LpvMN3i73aorg8ShkovTmowbwdlyA7kfDBPMYMy7xFLnc8'
    'OMG8vFhVajzHPmI9nuYGOrUpVb3Ktz09ZWCBPYY7PT0u4x+8eLk3O+CrrD3hyQ09XshOPN+9wTyE'
    'G708I97NvLyX5ju2zzE9qUs4Ped8iLwWKUg9/CYEPNKMmjy5zgM8li3rvHT+Rj0+8X+9OCdpvRcZ'
    'ZzwmPLw8DnuAvGd9wDzKE3i8O1sGPSrG6rz0nRw9hJeQvXd22bp1OR09tgi8PPuaILwdaXc9tZKj'
    'uy/CR73azfg8LELWvG8TSL2QE4k9zPUFuy6Deb2tGI09LJ7LPITEKr3/Vpo8cRunvFqjFb2iVYK9'
    '0vuNPPw9H72qkkY9zGc6u7D4tjyk29i8riS9PKvijD3ZqmO6gDw+vbzRcT3VbHC92OTnPJ0Lcbxt'
    'Dp+7Sx7cPBWeAD05SBe9x0miu9PyjjxZlZG7lU8mPUzHHL2NGew8ZOtLulCwGTvprHw932RkPZ0A'
    'Jb38L6M8k2K9vBva1DxFRQ+9ZgtmPdxqED01B5A8uA2LPRjmAD0HJyS9zGWrvCwUQr0mw4q72ln6'
    'PCSKPD0uSxS9HJIHvWIV77sCaVO9ENkHvVJOxjs8fJI6sH1qPSq/TDxiG6M8sho2vSBceTy01fw8'
    'vAtnPWigBL2GgpY8xpGCvecbgr32W3a9gyeJvZerhDwgrZo8YyR1vSizIr1PP6e7x1a2u2n68rxv'
    'ROw8qe+7PMfInboredc8okW8PObPCr0WDC49tasZPQfhPDzFZRG9a8rlvNdw67ygIGY9ZOIkPW9F'
    'hz30pRm7P2bSPG9eU73w/aC7FyNZvGbJbb2asYm8q2FXO1plUTuaQT89X7movPaVIb3Lamm9Sqe+'
    'PFORGzzGjxC8kmyJvTggkr0uZj89BZKFvIbHUz1DYgW8FBdfPbv8Tb2AGGS9nIWBvVQKBTu0YHG9'
    'l0q5vdmzUD3U/T29zRj+PKS9/rxber48vfdrPGhJNzx/AFq9z8GAPVDsTT2eUfK8eUv/OimtWj1X'
    'U/88/MO4u8ZTMb1uUhg8jetuPBSywzvw0xY9hvF0vc10ML1q6Y08KyfjPLuKiDwJCj69FPrfPC1j'
    'QTvI1DW9M1GjunjMT73EmtU8dHTMO3pUIbxJ8Eg9oJuMPNXC6jzu+Xc9LOdYPH+8cD392He8SDoV'
    'vTPQOb2zG3s9o5SXu5W9Z73vLrE8ApAfPUD1Tz33Kxc8UZQ6vXpTuLyrHf28skxxPDoHKr0pqKs8'
    'oZgGunW2b7tFSV28UtsUPTyaYD1nGTg9VVUdu4nYyz1s13M9OVRyPSfOuTsZrRE9SvQNvTxxDL2O'
    '6pS8t0cUvbphNjwFYF49w49JvUj6bbyc64M7IrDyu4LflLw/iTg97lQIvRuhTj0jEA290syPvXq8'
    '27weRx29QU/0vMtceb1ZXZO7+jJjuYSLqbyC2MC7u25cPdzh97zf3Pc8iWMTvRKFWb0hYMU72TBF'
    'vRGvYr1TCus8JnwPPQKhiTpCbTs9fbmrvM3XWj1ScBo9pncwPdu5drwzXaM7d8jpu/B96LwMipw7'
    'OrKNvbRjtrxS0Po8DKE/PIKuSr2nqiO9ElxuPJ97OD3ct0o9S6s8va7kd732BjW9IIkqO3FeQr0H'
    'RvS8z8esvEdjBryDMAy8BZNKvZ+P4Lx7dZY8M7MbPfvXcbxVESY910fjOy08bjzWgQ+8/LHgO2T1'
    'nDsFbxU7K0mvO5u3CT0hkwm8TYxkO7htCTygyu+879XJuuqA4Tzm+9w7in1DvdUIUb0RDSk8ES1E'
    'PAwW8jxuMhq94eB2PTQG57wz1BU9KGY5vb4hP7wgmUI7sXcmvMYpFz1sk4+8pMhhPN+ekzzgP4S7'
    'StiDPHU5lbyEcOO8+eNmvUzmpjzMjjW9pBulvUhHFz0BMzo9viKFum/DcTykvCe93S6jO8aOkT0l'
    'WnY9VOIKvWieWr3DWeS8ZSmIvFLKRL3HJSa8ca7DPH9Khz1RIv07UyQHvGb2GL3Q/oC9dQD2PH3x'
    '2jyYuxI9PSzZPMm2nD09H607ZW/EvIy4Dz2XAXq9NjM6vHwDkbxTsSO92BoVO58ejrvfcdm6u8AE'
    'PeBHdrsSoc08rSotPDnrCTt/wYY9TDIMPafsE70pD0m8iJCWvINJzTtjVtY89hRkvJ0MMLwoM1y9'
    '8yBNPRUm1ryFoyM931v8PHVIsjwcaCO8wheWvMng8jz0OUc6eKnpvNIRJL146UU9imMrvawZZz3Z'
    'ICi87oMJPVZYYL1mGJS8KA6bvC4YPbxCsDy9PzBIO7BZAD2wdHW7HnOfu9oROT3SEmO9+hTEPIOG'
    'uDwDI4o8HtC7vMUIhD1eKDu9Kw9jPYvPorzOQUK9HcK5O7hdmb1hZGk9Bz0+PAI+27xrlWE9xiIx'
    'vEAxWTztBDQ9r7EfPYuxpTwsFxM9hTqnPOdL27wfc6S6kmVVvRU7S70KoQm9UlhdvcgwHb1zKVq9'
    'C5fyPGKGh7zcWoy9S6wkPMxF17xLKxK9G0APvVDYCbyU2G08fhg8vehSwLuFGb668ZxBvVsAKb0h'
    'Fkq5OdJHvQWUIz3KDB+84k6jOlxm4DynnLo8ws5bPXmsnjzp6ZI9RVbHPEzEtDtKHFG9TyaXvQ+k'
    'Vbq0MA+9gYUYvQcfar0ucLY8DcGPvAzdjTy1AiW9S2IzvaFpyzsu+wW9ntvyvOF4Bb258hq9lnJC'
    'PPlzRDxyE788YwrzvMQaVLxhnpI9fmw9vBhLozsBP0695VGtvLx9jr0XRyO9v7SKPSQXCD0/Z3I9'
    'nKfcPNAwMj2Qr9C8t0wAvbjfKr34c3i9XCtkPUn1Fzsj6yC8tRPIOzF817w/IjE9TEqAvZWF27xv'
    'ySk6t7QEPckpTrzxNAc9kdzzvJ6C2zxOT0Q9J2VNPXcRmDyZuHw9WGoePf3xTb1NEim9Pu3vu4Vf'
    'Hb3AlA89e8GJvYNmCD03h088F7epPBnR27wgoM+88TKePPlhgbywTXQ8MusLPVYdS7xs1cq8rUym'
    'vQ7IwjzwU8A7VOLyvOoQxLwZyoW66FwYPSq/Ob0idDc9H9nUPMqtwbyDHkW9zOcsvVginjxmoUY9'
    'K1gCvEjVlrxYXTW941eqvQ3b+zsVa+W7uKA1PTi5YD35NFC9XG6HvJ1ZEb2nc129OxXsu/fgnLxJ'
    'pzY96FKIvdldxLzvycO7qEAevY4IWD3R53k7QTUkvYT7YTx1CFY8e2rtPAD487vrCj49sr/NPEaw'
    'Ib1IyQO9JY1pveLZAD1/mqk8BssdPCmqEjxjUxY9sGr+vC3LXT1wE289zJGWvTa3yDz+0us81Rc5'
    'uUgrXD2Hs0U9Pf4svc/O8jwFc0S8bUHovFBiGL18DLW80COLO3TTwbt13aE8mUHLPHFsP71QQko9'
    'pSCQPRaM8zyON967fXJ8vQhqS71JUqQ6ji7cvLBgEj34zLO7SqF7vb5FN73efhi8pLGQvZ30F70r'
    'yBK93NkdOdHH0bzk0ye9PwKzOwQ0O72sGHm8z/4EPXFzUzyNkgc9uEgvPe9Wdj2kBVg9Y5e0PLOQ'
    'Dj00QcO6uhdQPaRoRzzMDzE9GAPhPJxMEbvswea7f8dkvVnr5LvLBz+9ajQHPH6j5jy2xC26D7ZA'
    'PZukCD2kYFm90+2LvDKLFjzUt6u6AzaLPRWCe7x7uEC8Rv+huh0qxzzFb3U9aXF/PFAeDbnZA+Y8'
    'jHD7PDrshj3pnzA97BYUvfP+dL1U8zS9u+lYPZUILL16dqG85H47vaoYdz3Vy7g7ymgOvajY8bxH'
    'ZQu7UZ3qPNuBj7wF8S49fZYkPUwdYD3V8wc8N9KVPFBLBwjSjB2CAJAAAACQAABQSwMEAAAICAAA'
    'AAAAAAAAAAAAAAAAAAAAAB4ANABiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMTdGQjAAWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpahoDIu5ktFz1hE4g8'
    'XJEhvcp+Lzv6/vY8ZCAqPB+Beb1tSIA9L3Z1u2hTdb3aRS49Pd5fuwzbMz0sawY9b+CSvGFGIzxE'
    'eLS8tsVAPcpIXj2gL3892JFwPNNeCj2rrc085eujvCDa6jy9CLk8ZV0sPQID6bxyLhm9HLgIvXLk'
    'ebtQSwcIl75Z14AAAACAAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAeADQAYmNfd2FybXN0'
    'YXJ0X3NtYWxsX2NwdS9kYXRhLzE4RkIwAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWuhAfj/BeX0/cxqBPw3YgT/FT4A/oeKAPzdNgT/JxYA//iCCP2ooej/T'
    'DIA/D3yEP+FegT/VaX8/C5KBP8+TgT/LhYE/Vfd+P1DvgT9R5X8/iCiBP97pgj9rznw/qKV/P9HW'
    'fT+aQn4/KjCCP6N2fj9ddoA/EvZ9Pw4PgT8UlYA/UEsHCMp5oRmAAAAAgAAAAFBLAwQAAAgIAAAA'
    'AAAAAAAAAAAAAAAAAAAAHgA0AGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8xOUZCMABaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlp/3PK7uDQfvLEfb7sA'
    '4RU7ojEhu6L4Prt+90s4ZLksO7nwBbs1XJ68La6kuo3TG7q1Qyw7wCMCOsr18zk8GMm70MmFO/2S'
    '2DoH6rc6vU/mO6Lonzvrhw+74qJ7vAL+k7y662W7bjGuOhnmzrpdP4C8/z5tPKQeS7zv+8k7b+K1'
    'O1BLBwiFvmB+gAAAAIAAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB0ANQBiY193YXJtc3Rh'
    'cnRfc21hbGxfY3B1L2RhdGEvMkZCMQBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaB658P5KegT/U4X8/RWx+P8q7fj8nFn4/UJN+P94vez+PQX8/nimBP6Qc'
    'fz+sEn4/pfh/P+SLgT8WgIE/gjh/P6WmgD/ksn8/0w2AP+7lfz8cW4A/xguCP55RfT/M8YA/GjWB'
    'P4iHfj/R234/q7B+P2ehgD9GIYA/Qit+Py6SgT9QSwcIgvZBr4AAAACAAAAAUEsDBAAACAgAAAAA'
    'AAAAAAAAAAAAAAAAAAAeADQAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzIwRkIwAFpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWsZ/Sj0KDQ89TIkZvP+a'
    'szsTAsY7j5ZBvT+hUjyAxE09duffPHyGbztq9Ow8y4EVPTNVSj0Vx4C9FX1/PXgZ9ryIdz69s9BZ'
    'vQbXCL1buQO9zn1dvMtGQr3eV+Y7iewJvNxxkLzGQi63sVk6vQU8kTxAUYI9v7xaPMZ3Zr2vBBQ8'
    '5IfqPB5G+Dzk3Y89k19yvSEINb3D5Qm8oHuXveEdbjwT/Bo9ActEvN1sD72viQ6967ShO4K7lzwV'
    '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'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa8NZQPXTLQL1eEUM9g6Iv'
    'vM95+7xTFai8gObMvGbx2zoAxS89K/hVvQWIQ7zJjlC9EkZivf+VZj0ztrW68lXaOzuArbyt/jm9'
    'hD1PPc8xg7yRXIe7SWAavRTi87yiQjS92X0LvBa0eD1G7fS76SmmvKglfj3Xg788GCYOvXGW1jxQ'
    'SwcIu6KvD4AAAACAAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAeADQAYmNfd2FybXN0YXJ0'
    'X3NtYWxsX2NwdS9kYXRhLzIyRkIwAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWtRAhT9Tb4Y/a3SHP6Jtgz+rHYQ/vTeHP3h7hD9QPYM/V4aGP/EihD9t6II/'
    'I4qFPzGwhD/7VIg/TN6FP7RGhT+eg4M/rY2FP67Yij9N/4U/R3yJP8iohD8S4YY/MISIP/f7hT/E'
    'z4Y/tGOEP6crhz8ec4Y/0wqFPxmwhj/A3oM/UEsHCAkad2KAAAAAgAAAAFBLAwQAAAgIAAAAAAAA'
    'AAAAAAAAAAAAAAAAHgA0AGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8yM0ZCMABaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlocqAs9q+UjPAO+ZzyuCEY8'
    'XR+iPI2s8zz4HP08eIENPEqsjTz8LIs8JxTbPA7MizwOZbs8Kd0OPRozDD0V25q6goeOPByDqDxW'
    '0qk8GjNaPJA+zzw88d07+gzCPP9Znjz8Nu88B3+APM4RlDw9ljQ8HaWoPATXeTxJBZQ8QQ5ZPFBL'
    'BwiRcyVigAAAAIAAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB4ANABiY193YXJtc3RhcnRf'
    'c21hbGxfY3B1L2RhdGEvMjRGQjAAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaeNcZPS7Oej2GzE89a3w8PXv+CT1WLAC9J7o3vQXTvbwVnV29GTITvDKpRT3X'
    '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'
    'Nr3XiVG9+7pBu6qBj7zidna9MdIfvZq0I70Rwp69QiX3PMnCgb06rQm9nqxpPMBAKr36UFm9UEsH'
    'CJjxORsAkAAAAJAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHgA0AGJjX3dhcm1zdGFydF9z'
    'bWFsbF9jcHUvZGF0YS8yNUZCMABaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpVp3G8Vf62PO0iD71cv8Y8IxZKPNkhvTzA5yK6VZQivY9x57xGjt+8DKSEvb3s'
    '7LzhOKy71/eXPNljT72ApIc7prDSvGOu57zTa9G8YtmKPPT1Er3r8Sy9hRqxO97djry/ZKa6gP08'
    'vY5SLzxBsdc8oKzNPNLE0rw0zM68N/CrvFBLBwiyCDR6gAAAAIAAAABQSwMEAAAICAAAAAAAAAAA'
    'AAAAAAAAAAAAAB4ANABiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMjZGQjAAWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaJKUwPkRhaL324pM9UY6MvaT4'
    '8L1eb5s9Z0srvhjV6by0VC6+v9UdPgPV3j1nrSU+0b0vPTytN70c7BK8rn4nvsY1hb1AfRe+V+DL'
    'PRiDKL4TrS0+td2WvROCgj30vi4+I74gvg0gmrtqOi++2XiYPGf6cj05TAO+nxnrO7Y5Sb1vFLK8'
    'DuMXvgvSWrvF3Ui9RwAZPpgPBr4lRhm+F987PqDy0T26ZTE+SLYfPtHEpjxbFwC9dHBYvdQZGz7T'
    'NzS9sg6YPRNNIL51iSs+VCeXPZZPUz4bWtk9F4ibPRBF7Lt4JsS9ZUwIvrFMFT4yiss8nLBHPkFM'
    'Q71JZgG+DB0zvmQ7+bysrY495T89vAsOWz1vK427AC8AvscJyTtuNBA+cxCyPazuIz1rKzU+wJoA'
    'vtAVnz0nJpG7M0JovrDoNL6fuF6+X3YfvVKOAz5Ftyg+/2YWPl56WL7WG0k8/K5xPbJFpr3q7pw9'
    '5EtcPApnKD07Qzu+8suDPIuLuTylFAm9e4MyPnpSsjyc5KQ9ZpL6PaIEzT3KNlW97WYZPuZon7yC'
    'GQ0+X4LWvanSpr1kkza+0mkqPv9dxjsNXSm8X5usPcFoib1gSVS+PJnMO+1OYjtMRz8+XWtdPcQ8'
    'XT0byro9KoEGPngw9z1r5ks9YPQivkfvN763agG+LJyUPcelzb08cre8OKvYOwr5qb2whvQ9KPK2'
    'PbAsFD4RN6e9kaUJPeOxKL6LIkC+sJGsPRK7pj2fFxC+eBxkvd/ivD1/zWg8RFlhvTVfvj0cc6a8'
    'u+76vb9g+b2MNTO+9y1ZPT37M77JBQA+Ft0hPiy/ijxUY0W7OIfBPcLCIb6IYI49+vOFvb/TAT2D'
    'jti8hAwaPYvn3D0yfT4+jHjjvJATKL7SkSW+S5pJvdW2iD0r7fy9mZYFvqhmGj7vRYK9EswWuYk0'
    'yLztfh6+qk1oPr6c3TzvoQi+FLiEvSFptT04CBs+W3yhPKfVgb2zuJO99dyTPVcvU749cNK9rvN3'
    'vGDHiz1fCBo9VXXgPbb4fz05uTi+26sgvh0uQL2dISc9lgYTPq7ahT1DGZA9PlSovSucvL3QYHA9'
    'XZP0u1zTI75kHUW+TnwiPJ6QCj7OZvY8vp2QvSfXHb3ao+A7g5KovYXt3z2vu3g8LS08Pmo1Pr5b'
    'Su09SbtNPce1LL1IAN49J7ItPkBIKz6wUbo9892kvfkABrxdZSk+BaYCPsLzIj7gUTQ+dJ09vkcp'
    'Fj6lCRQ+yg61vXvnBb423Ta+wBo3vomnv73xIS495WIcPtC94zqf5Q09BvcIvsJG7T2U7oG9PpqZ'
    'PQymib0w+4q9DmBqvmCGALrnZau9ji7bPcA7Ir6O1kk+62AavlBLBwhDjfZvAAQAAAAEAABQSwME'
    'AAAICAAAAAAAAAAAAAAAAAAAAAAAAB4ANABiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMjdG'
    'QjAAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaQPmbPa32'
    'Jj4a+di9JC++vaIGgD1Kzw69OsYRvoGskz1QSwcIOSJ+7SAAAAAgAAAAUEsDBAAACAgAAAAAAAAA'
    'AAAAAAAAAAAAAAAeABQAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzI4RkIQAFpaWlpaWlpa'
    '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'
    'L2RhdGEvM0ZCMQBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaY/kUvC0nLLuArRe6T9unur/0ODt7MhK8zPAWvDnZIbyjWO47b/rQO6CdHLz2exE88MwKPEY8'
    'xDtSgbg72RaOOy0M2jvvIO+6Gv8SPCSZbbql2tI6DuhdvGVgM7wJ4HO7YnRvO/FwDTzXqeA7esE4'
    'PN1GxToxbCA8612Qu4+UGzxQSwcIZPvlHoAAAACAAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAA'
    'AAAeADQAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzMwRkIwAFpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWrlOoj3A9ty7dxWiPbXnmz2wb5e7ECP2u9HX'
    'jj3g3+Y7CtgqPUoZJL1IzI+9HmAXPcthkz2HuJG9uYKOPbzlwDwAqEO4olFYPWgBEz1+zHQ9huZw'
    'Pdz057xayhw9EuVKPaRZpL3YxM+8OHrrPKhLNLyg2Xc8Yf5bvVXzsb3VQ169eDOmvZcSqz1mXUm9'
    'bIyWvSiH8jw4QU68MM+1vKYzdb0Drqg9SNR6vFQ3ZL0PuVO9kvkUPVfUhj3ilDE9GtQwPWWMiz2w'
    'xuK7U4OAveynvbzAYo66j/2nvbXRkD0QBga8uR6UvSTDzTxW7Nq8tbKbvbCCPTz4hyk8gMPsugA7'
    'B7quIUG99GumPPz3uLyOqRQ9Jf03vYAcbrvexIq9+GMJPBr1GD0goui7xhwtPSmqmr04uDY8TEjy'
    'PE2FlT2pPZq9u2pKvXs5lr0I7Dm9PIQMvVxilrx4G5S8yxKsPbTzcb1SlGM967yUPcCrxzwA39O7'
    'MeaQPShk/zxPT6q9aOuNvWkrpb0ES1O9jLTRvH6emL3bEIs9ENbfO6CJVzvxdJk9Jj59PbCT07yc'
    'axa9NcmpPQ7ZbT3Atae6CMC8vMCt0jvE4v+8IGWKPFM0s71Ukpq8qp0bPQW0qL1URgI9EN9cvWDF'
    '1Ty55SK9d02bPZDIn7uvATC9YTOpPSAidb1aOhu9UEsHCMnkA3YAAgAAAAIAAFBLAwQAAAgIAAAA'
    'AAAAAAAAAAAAAAAAAAAAHgA0AGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8zMUZCMABaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlqyWEE9UEsHCFAUmkUE'
    'AAAABAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHQAxAGJjX3dhcm1zdGFydF9zbWFsbF9j'
    'cHUvZGF0YS80RkItAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WjeFgr2oV5i6FR2YvACzvbO45Z28euBxvLF/Mr3aXF29Pl9Qvd8oFz0fb8e8vqQavCzAGD3es4O8'
    'oA+wu3JELL2OhxC9beEJPf6wSrvGsxg94BUSPAkpTrtxboC9m0Xiu14zjLwVfXi9zqIJvYzB5jvQ'
    'ET29wObWO9JGRT39X5c8Za9rPYel5zymNUa9LqBpvG4P4rzjHPI4SIrTOwPDFD3BKsi5SCUZvXWF'
    'SL1/4ou9/sHxvKz+8TzQ/AM7yadsvZ2IWb3lYNo8Uy5Vvf6HPL357Aq9A7xjPS/3abzmHs+880Uu'
    'vVGX9zxBRqe8CGKGPKwoZzuQ9Rg8nckFPVRcXDsYOSI8kFtuvFKxSL1tZpK8EF5JPemRRb1V5jC9'
    'mzSHvX8dxzur2Wq9E7mGvO58Xz1hn4g8ngcLPSvcAzzlKmu9CL0cvWelLr0z0yO9TTPHPCzRCLzA'
    '3IA91XICvbznU73xiog9rJB8vWidQz2YGp08bs9Ivdlkjj0nct28a8wVvYRNXr0RFJu9+7iVOzR1'
    'Vb0j0EY8C+OUvJ+v5zyfSsK8G3HVPPC67LsgGFk7IfcCvbSRML13q0E8ymUqvcsX87wxAFm9ii3H'
    'vI8z/byXUnM9JRxCPYFigDy28hc9jTn+PH1Llr0AbSq8gO0bvIBS1LyjQ7M7CFZcvGsGiL0zxRi9'
    'SHGevQ6IOD06PBQ9nvIxPcqDrrxfxAI7AlunvE4Vb73SlGE8CMDqvDnw0rwHb9+8VU1ovcqKnDny'
    'EmQ69K7Su+R9VD1JawQ8MXxLvJUOtLxSDxu9+gb0u+/CFL0N5D69BzlBvDc6uLzVcE29IPPwPAqH'
    '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'
    '/ZLqOygLkr3t7D+9r1D4u+C/BjzNZIK9SC5cvQdaTr3IEhi9WdgyPZPPzTy+S1c9EVQyPRyQ4jyi'
    'Nx298ZJ3vVxNH71FHma9ZZs0PdDrBL3g0go9N/cEvZJZHr29Dhg8l3fmvNC8BzxjBzA8gmOOvb7C'
    'g70gAkY9kHRjPF32jLyEatI8zCIDvG8n2Ts0dTG9CHgfvYSZvjzx4x46/nmTvCR8krz46NE8gXw7'
    'PaVIQT0tnKS8Nc3XvI+Tr7zVpOe8Fp66vCOm/bvTbyI9nh4rvScgVj23pho9tOg2Pbc6FT2cpcc8'
    'SnhdvVYP1TyMzaK9vw1BPFm8Uj3NzO27z2HFuuCKI72q5l69bo5ouoodJD3aeFu9UNP7vPVdxjti'
    '/za9u+KJvEQoVr1RQsy6CR8wva0tJD0cI2E9BBoHPWF3AL2h0oY86A6Mu1BLBwjDQzC0AJAAAACQ'
    'AABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB0ANQBiY193YXJtc3RhcnRfc21hbGxfY3B1L2Rh'
    'dGEvNUZCMQBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'bUtVvUK8OT055l09IrnVPJ/LPL1Uu6a8Noi5OP26Xz0l4VW9GyYpvfn+7zwfFb48O92hvAKKOL0z'
    'zUq8v3gzPaN+Wj0ZsDG7pAMMvYLejDwj0R49RcAdvU1ngTzVeBY9wTXQu30fRb27TQK9Y6BxvVUg'
    'KDsHjRc9GlpIu2cRIj1QSwcITG8xeoAAAACAAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAd'
    'ADUAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzZGQjEAWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWjsGgT+hOn4/Akx9P1POfD9rB34/uJx/P8cOgD9i'
    'qX4/BQp9P0C/gT+SmH4/OSR+P0Aqfz9lO3w/sFF/P8JCfT/tHIA/8ad+Px8Ifz/eK4I/YXN9P6N7'
    'gT8K3IE/v++AP2gXgD9EC4M/sfB9P4JIgT+4Enw/3Sd7PzLaez8X7IA/UEsHCAlr2kCAAAAAgAAA'
    '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'
    '3LNGPAOWjD2DYq68MoRLvcrqJz3K7ze9EhsDvDle5zzttPG85q9xPHV6iLzeJDs83RbtPHZuEbyx'
    'TNm8i4JQPXiuXbxECpC8achFPd0s/DpHXZs899sAPFScSr1mE+i8g20tvMqMTT3fVYa9OY1BvTXk'
    'Vbzoi0e8iw2hvSI7Kz05SOw8/Y4rvVqh1bzr2Pk8IDYAvVVDBj3veYA9A0ImvEPowLx7WRi9mYHm'
    'vKIzNTvz/V69jPBCPfPjhb2uAKI74sYrPTmjHTymJuS8pwAjPF+lJb24qAq9g1zgPAKutrtfliS8'
    'H+OTPFyVWj24Nok8GogvvViPxDyDlUA9KoBYvNkyUD346EI8gzllvQdzCDxm5l+9kUvRPIsSUD2R'
    'Uyi9l+3xPDMBWT1xDB29NKLwO0NQVj10s388bvm+PMrTMry6xXs9F//OOw1wBL0g0yW9H29pvTnK'
    'F70C6Do8A4EQPVF6erkU1T+9UqnPvElIQT2j+CE8XTe/vHNJTT1/QUw9SC/7PFHZDL2ZIaS88wP/'
    'PDweOT1amtc8M+gyvc0aW71QOqg8zsdPvAxU+rofRnA9bnEUPC7igL2JssY73cwEvZF5xDxOFge8'
    'XFOZPMCeUr1CZ408UEsHCC5UzRIAkAAAAJAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHQA1'
    'AGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS85RkIxAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlr5lNg86eNNPe9OX73ztmS8T059vNr4WL2dsym9tsZJ'
    'vap4TzsrlGO8e3C6vIFBGL2X8GW9F21SvJPrG70FDzg9U+oGvZw1Mb2bOy69dmIfvX0fXj3lxdo8'
    'KYrSPCYmpDsLVz27mpu8PNedSr0wdw+9wsS0vLGdv7y+qLc8l3vQvFBLBwgChdDAgAAAAIAAAABQ'
    'SwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB4ANABiY193YXJtc3RhcnRfc21hbGxfY3B1L3ZlcnNp'
    'b25GQjAAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaMwpQ'
    'SwcI0Z5nVQIAAAACAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAtACMAYmNfd2FybXN0YXJ0'
    'X3NtYWxsX2NwdS8uZGF0YS9zZXJpYWxpemF0aW9uX2lkRkIfAFpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWloxMDMyNDkzMjE4Mjk0MjkxOTU4MzE1NzYyODcyNTA2NDI5Mjk4NDkyUEsHCEu7'
    'NZQoAAAAKAAAAFBLAQIAAAAACAgAAAAAAAB0+nov+g4AAPoOAAAfAAAAAAAAAAAAAAAAAAAAAABi'
    'Y193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEucGtsUEsBAgAAAAAICAAAAAAAAIU94xkGAAAABgAA'
    'ACAAAAAAAAAAAAAAAAAAig8AAGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvYnl0ZW9yZGVyUEsBAgAA'
    'AAAICAAAAAAAAMAW3GAANgAAADYAAB0AAAAAAAAAAAAAAAAAFhAAAGJjX3dhcm1zdGFydF9zbWFs'
    'bF9jcHUvZGF0YS8wUEsBAgAAAAAICAAAAAAAACL5Fi+AAAAAgAAAAB0AAAAAAAAAAAAAAAAAkEYA'
    'AGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8xUEsBAgAAAAAICAAAAAAAAHMDBViAAAAAgAAA'
    'AB4AAAAAAAAAAAAAAAAAkEcAAGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8xMFBLAQIAAAAA'
    'CAgAAAAAAACFjr2AgAAAAIAAAAAeAAAAAAAAAAAAAAAAAJBIAABiY193YXJtc3RhcnRfc21hbGxf'
    'Y3B1L2RhdGEvMTFQSwECAAAAAAgIAAAAAAAAXjV0zACQAAAAkAAAHgAAAAAAAAAAAAAAAACQSQAA'
    'YmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzEyUEsBAgAAAAAICAAAAAAAAGr7/oWAAAAAgAAA'
    'AB4AAAAAAAAAAAAAAAAAENoAAGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8xM1BLAQIAAAAA'
    'CAgAAAAAAABz/WIogAAAAIAAAAAeAAAAAAAAAAAAAAAAABDbAABiY193YXJtc3RhcnRfc21hbGxf'
    'Y3B1L2RhdGEvMTRQSwECAAAAAAgIAAAAAAAASlPomIAAAACAAAAAHgAAAAAAAAAAAAAAAAAQ3AAA'
    'YmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzE1UEsBAgAAAAAICAAAAAAAANKMHYIAkAAAAJAA'
    'AB4AAAAAAAAAAAAAAAAAEN0AAGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8xNlBLAQIAAAAA'
    'CAgAAAAAAACXvlnXgAAAAIAAAAAeAAAAAAAAAAAAAAAAAJBtAQBiY193YXJtc3RhcnRfc21hbGxf'
    'Y3B1L2RhdGEvMTdQSwECAAAAAAgIAAAAAAAAynmhGYAAAACAAAAAHgAAAAAAAAAAAAAAAACQbgEA'
    'YmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzE4UEsBAgAAAAAICAAAAAAAAIW+YH6AAAAAgAAA'
    'AB4AAAAAAAAAAAAAAAAAkG8BAGJjX3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8xOVBLAQIAAAAA'
    'CAgAAAAAAACC9kGvgAAAAIAAAAAdAAAAAAAAAAAAAAAAAJBwAQBiY193YXJtc3RhcnRfc21hbGxf'
    'Y3B1L2RhdGEvMlBLAQIAAAAACAgAAAAAAADC2A+pAJAAAACQAAAeAAAAAAAAAAAAAAAAAJBxAQBi'
    'Y193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMjBQSwECAAAAAAgIAAAAAAAAu6KvD4AAAACAAAAA'
    'HgAAAAAAAAAAAAAAAAAQAgIAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzIxUEsBAgAAAAAI'
    'CAAAAAAAAAkad2KAAAAAgAAAAB4AAAAAAAAAAAAAAAAAEAMCAGJjX3dhcm1zdGFydF9zbWFsbF9j'
    'cHUvZGF0YS8yMlBLAQIAAAAACAgAAAAAAACRcyVigAAAAIAAAAAeAAAAAAAAAAAAAAAAABAEAgBi'
    'Y193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMjNQSwECAAAAAAgIAAAAAAAAmPE5GwCQAAAAkAAA'
    'HgAAAAAAAAAAAAAAAAAQBQIAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzI0UEsBAgAAAAAI'
    'CAAAAAAAALIINHqAAAAAgAAAAB4AAAAAAAAAAAAAAAAAkJUCAGJjX3dhcm1zdGFydF9zbWFsbF9j'
    'cHUvZGF0YS8yNVBLAQIAAAAACAgAAAAAAABDjfZvAAQAAAAEAAAeAAAAAAAAAAAAAAAAAJCWAgBi'
    'Y193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMjZQSwECAAAAAAgIAAAAAAAAOSJ+7SAAAAAgAAAA'
    'HgAAAAAAAAAAAAAAAAAQmwIAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzI3UEsBAgAAAAAI'
    'CAAAAAAAADTs6swAQAAAAEAAAB4AAAAAAAAAAAAAAAAAsJsCAGJjX3dhcm1zdGFydF9zbWFsbF9j'
    'cHUvZGF0YS8yOFBLAQIAAAAACAgAAAAAAABCM1ccAAIAAAACAAAeAAAAAAAAAAAAAAAAABDcAgBi'
    'Y193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvMjlQSwECAAAAAAgIAAAAAAAAZPvlHoAAAACAAAAA'
    'HQAAAAAAAAAAAAAAAACQ3gIAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS9kYXRhLzNQSwECAAAAAAgI'
    'AAAAAAAAyeQDdgACAAAAAgAAHgAAAAAAAAAAAAAAAACQ3wIAYmNfd2FybXN0YXJ0X3NtYWxsX2Nw'
    'dS9kYXRhLzMwUEsBAgAAAAAICAAAAAAAAFAUmkUEAAAABAAAAB4AAAAAAAAAAAAAAAAAEOICAGJj'
    'X3dhcm1zdGFydF9zbWFsbF9jcHUvZGF0YS8zMVBLAQIAAAAACAgAAAAAAADDQzC0AJAAAACQAAAd'
    'AAAAAAAAAAAAAAAAAJTiAgBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvNFBLAQIAAAAACAgA'
    'AAAAAABMbzF6gAAAAIAAAAAdAAAAAAAAAAAAAAAAABBzAwBiY193YXJtc3RhcnRfc21hbGxfY3B1'
    'L2RhdGEvNVBLAQIAAAAACAgAAAAAAAAJa9pAgAAAAIAAAAAdAAAAAAAAAAAAAAAAABB0AwBiY193'
    'YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvNlBLAQIAAAAACAgAAAAAAAAd67nzgAAAAIAAAAAdAAAA'
    'AAAAAAAAAAAAABB1AwBiY193YXJtc3RhcnRfc21hbGxfY3B1L2RhdGEvN1BLAQIAAAAACAgAAAAA'
    'AAAuVM0SAJAAAACQAAAdAAAAAAAAAAAAAAAAABB2AwBiY193YXJtc3RhcnRfc21hbGxfY3B1L2Rh'
    'dGEvOFBLAQIAAAAACAgAAAAAAAAChdDAgAAAAIAAAAAdAAAAAAAAAAAAAAAAAJAGBABiY193YXJt'
    'c3RhcnRfc21hbGxfY3B1L2RhdGEvOVBLAQIAAAAACAgAAAAAAADRnmdVAgAAAAIAAAAeAAAAAAAA'
    'AAAAAAAAAJAHBABiY193YXJtc3RhcnRfc21hbGxfY3B1L3ZlcnNpb25QSwECAAAAAAgIAAAAAAAA'
    'S7s1lCgAAAAoAAAALQAAAAAAAAAAAAAAAAASCAQAYmNfd2FybXN0YXJ0X3NtYWxsX2NwdS8uZGF0'
    'YS9zZXJpYWxpemF0aW9uX2lkUEsGBiwAAAAAAAAAHgMtAAAAAAAAAAAAJAAAAAAAAAAkAAAAAAAA'
    'ALgKAAAAAAAAuAgEAAAAAABQSwYHAAAAAHATBAAAAAAAAQAAAFBLBQYAAAAAJAAkALgKAAC4CAQA'
    'AAA='
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
# Build NN-rollout factory. Used only when
# GumbelConfig.rollout_policy='nn'; constructing the closure
# unconditionally lets the same bundle support nn rollouts
# via a single config flip without re-bundling.
def _bundle_nn_rollout_factory():
    return NNRolloutAgent(_bundle_model, _bundle_cfg_nn)
del _bundle_ckpt  # free RAM after model is built


# --- GumbelConfig / MCTSAgent overrides ---

# Applied by tools/bundle.py at build time.

_bundle_cfg = GumbelConfig()

_bundle_cfg.sim_move_variant = 'exp3'

_bundle_cfg.exp3_eta = 0.3

_bundle_cfg.total_sims = 128

_bundle_cfg.num_candidates = 4

_bundle_cfg.hard_deadline_ms = 850.0


# --- agent entry point ---

agent = MCTSAgent(gumbel_cfg=_bundle_cfg, rng_seed=0, move_prior_fn=_bundle_move_prior_fn, value_fn=_bundle_value_fn, nn_rollout_factory=_bundle_nn_rollout_factory).as_kaggle_agent()
