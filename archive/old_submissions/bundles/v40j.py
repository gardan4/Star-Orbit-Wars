# Auto-generated Orbit Wars submission. Do not edit by hand.
# Built by tools/bundle.py on 2026-04-30 17:02:53.
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
    'UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAVAA0AYXpfdjM3X2NsZWFuL2RhdGEucGtsRkIJAFpa'
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
    'BCgoaAVoBlgCAAAAMTlxoWgISyB0caJRSwBLIIVxo0sBhXGkiWgMKVJxpXRxplJxp1gVAAAAYmxv'
    'Y2tzLjIuY29udjEud2VpZ2h0cahoBCgoaAVoBlgCAAAAMjBxqWgITQAkdHGqUUsAKEsgSyBLA0sD'
    'dHGrKE0gAUsJSwNLAXRxrIloDClSca10ca5Sca9YEwAAAGJsb2Nrcy4yLmNvbnYxLmJpYXNxsGgE'
    'KChoBWgGWAIAAAAyMXGxaAhLIHRxslFLAEsghXGzSwGFcbSJaAwpUnG1dHG2UnG3WBMAAABibG9j'
    'a3MuMi5nbjIud2VpZ2h0cbhoBCgoaAVoBlgCAAAAMjJxuWgISyB0cbpRSwBLIIVxu0sBhXG8iWgM'
    'KVJxvXRxvlJxv1gRAAAAYmxvY2tzLjIuZ24yLmJpYXNxwGgEKChoBWgGWAIAAAAyM3HBaAhLIHRx'
    'wlFLAEsghXHDSwGFccSJaAwpUnHFdHHGUnHHWBUAAABibG9ja3MuMi5jb252Mi53ZWlnaHRxyGgE'
    'KChoBWgGWAIAAAAyNHHJaAhNACR0ccpRSwAoSyBLIEsDSwN0ccsoTSABSwlLA0sBdHHMiWgMKVJx'
    'zXRxzlJxz1gTAAAAYmxvY2tzLjIuY29udjIuYmlhc3HQaAQoKGgFaAZYAgAAADI1cdFoCEsgdHHS'
    'UUsASyCFcdNLAYVx1IloDClScdV0cdZScddYEgAAAHBvbGljeV9oZWFkLndlaWdodHHYaAQoKGgF'
    'aAZYAgAAADI2cdloCE0AAXRx2lFLAChLCEsgSwFLAXRx2yhLIEsBSwFLAXRx3IloDClScd10cd5S'
    'cd9YEAAAAHBvbGljeV9oZWFkLmJpYXNx4GgEKChoBWgGWAIAAAAyN3HhaAhLCHRx4lFLAEsIhXHj'
    'SwGFceSJaAwpUnHldHHmUnHnWBMAAAB2YWx1ZV9oZWFkLjIud2VpZ2h0cehoBCgoaAVoBlgCAAAA'
    'Mjhx6WgITQAQdHHqUUsAS4BLIIZx60sgSwGGceyJaAwpUnHtdHHuUnHvWBEAAAB2YWx1ZV9oZWFk'
    'LjIuYmlhc3HwaAQoKGgFaAZYAgAAADI5cfFoCEuAdHHyUUsAS4CFcfNLAYVx9IloDClScfV0cfZS'
    'cfdYEwAAAHZhbHVlX2hlYWQuNC53ZWlnaHRx+GgEKChoBWgGWAIAAAAzMHH5aAhLgHRx+lFLAEsB'
    'S4CGcftLgEsBhnH8iWgMKVJx/XRx/lJx/1gRAAAAdmFsdWVfaGVhZC40LmJpYXNyAAEAAGgEKCho'
    'BWgGWAIAAAAzMXIBAQAAaAhLAXRyAgEAAFFLAEsBhXIDAQAASwGFcgQBAACJaAwpUnIFAQAAdHIG'
    'AQAAUnIHAQAAdVgDAAAAY2ZncggBAAB9cgkBAAAoWAYAAABncmlkX2hyCgEAAEsyWAYAAABncmlk'
    'X3dyCwEAAEsyWAoAAABuX2NoYW5uZWxzcgwBAABLDFgRAAAAYmFja2JvbmVfY2hhbm5lbHNyDQEA'
    'AEsgWAgAAABuX2Jsb2Nrc3IOAQAASwNYEQAAAG5fYWN0aW9uX2NoYW5uZWxzcg8BAABLCFgMAAAA'
    'dmFsdWVfaGlkZGVuchABAABLgHV1LlBLBwhKLWXsMQsAADELAABQSwMEAAAICAAAAAAAAAAAAAAA'
    'AAAAAAAAABYACwBhel92MzdfY2xlYW4vYnl0ZW9yZGVyRkIHAFpaWlpaWlpsaXR0bGVQSwcIhT3j'
    'GQYAAAAGAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAATADkAYXpfdjM3X2NsZWFuL2RhdGEv'
    'MEZCNQBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'Wjyferzk7c880LMQvrWGX70n9Zi9bieTPXYv9buddZk96UfQPMd/zjxPv8a87r+FPGESVb2p5JO9'
    'oHxRvScf0jw3W3c9Q+aJPdAktb1F2Gm9I9sLPZoyiz3rub+82/uGPYAbgLw8DIS8BWSnPdpwrL1k'
    'j329iDiou1Qbi7wbpa49rzNsvZ2EVL3n7oG9FkC4vei7g707SNE9IDKEPb/Bcj1sBWA8239AvcSC'
    '9Luf/ay9Znxjvee8fr3Rvkw9e2Z2PS2nTr1oliQ6alBmPXXCwD0DuGM8+LoBPDatDD30bWe9LrLx'
    'PKNLmL0KsWm9CdVgvbSjLj1a13g8ZCyFvQl47Lrm9zY9B3ASO9VUBLxLoW08lWs7PZ2y0T05Yku9'
    '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'
    'QvFPvbYMq70w7+Y8iCrjvOb/z73tZmS9S8dMvRNsOL20Y3+96Vg6vKI8xTxGaJu9nSw+vIOrDL19'
    'UXu9YeO0u4RwK710gUI8EG+Ivf3dBTwr6Ks9L9qyPU9iXb2RYu88sr60vcAYnT3T8669tJ67vfgQ'
    'rb1QSwcIIvkWL4AAAACAAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAUAD4AYXpfdjM3X2Ns'
    'ZWFuL2RhdGEvMTBGQjoAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWthDgD93pn8/h05/P7VOgT8f0X4/9oh/Pzvvfj/ItH8/iemBP1aTgD+s'
    'DII/sad+P5VYgD9kkIA/xVl/Py65gz9Qrn8/x2OAPzD2gT+WUH4/E3WAP34Lfz8w2H8/K89+PyNc'
    'gD8SfX4/BdJ8P7TVfz/T4n4/5XB+Py2ogT8l9X8/UEsHCHMDBViAAAAAgAAAAFBLAwQAAAgIAAAA'
    'AAAAAAAAAAAAAAAAAAAAFAA+AGF6X3YzN19jbGVhbi9kYXRhLzExRkI6AFpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlo9Ej07mAPXOoim6zra'
    'spI87b9JvFRXUbvGbkQ7LW+Tu9f24zvviXE7XLy+O4uSQzphs4a7L7COu/bT2zu+BIK8MCCFOwKK'
    'GbsykrM6MvtMPN2fGrz2cLY7X4V0u9IErjk+tQE8cAoivCar8jpts127nPNHuoWNZDuWxSo8N7ox'
    'vFBLBwiFjr2AgAAAAIAAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAABQAPgBhel92MzdfY2xl'
    'YW4vZGF0YS8xMkZCOgBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpadgQ/PQX2YLpZc5C9OX+Yu7iSabue21A7cF+4vH7PbDx2bMS6dI6svKM8'
    'TL11hkS9XOJZvW8G6ryv+C89n10qPUzxnLvsFta8Jjh3PD8ZXL1BsaM80DAdPR/gk7ww9UC9IgbS'
    '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'
    'UEsHCF41dMwAkAAAAJAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAFAA+AGF6X3YzN19jbGVh'
    'bi9kYXRhLzEzRkI6AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpr8WI9uhJnvWepJ7yHPg69IN/2OpOdKD31lDK9Rx5DPYtUD73QWWA9yZFI'
    'PS5shry9vjW8Bw+fvNqm+bwjNRa7FQY5vX1JCr2Kc2a85L0FPdY5Wr32eSo97XvTvMQoFbwfQPY8'
    'j70mvcEWLL2t0eQ60HgLPfgVgDyVhw+9JiNePVBLBwhq+/6FgAAAAIAAAABQSwMEAAAICAAAAAAA'
    'AAAAAAAAAAAAAAAAABQAPgBhel92MzdfY2xlYW4vZGF0YS8xNEZCOgBaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa8cqCP9jdgj/RDIE/4sqD'
    'P/lvgT8fKoI/JGKBP34MgD/yeHw/a+uAPxEzhD+xXX8/3Y2CP7slgD/gyH8/4zeBP7cpgT8MRYM/'
    'OYKDPz2+gD8q/30/rLyAP49rgT+nFYQ/AJGCP8aohD+YT4Q/B3KCPyRfgT8RCoE/Nf2BP4I9gj9Q'
    'SwcIc/1iKIAAAACAAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAUAD4AYXpfdjM3X2NsZWFu'
    'L2RhdGEvMTVGQjoAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWobcQzyvHUE8dsYaPJONqjxjCCO7eUkhPFcYpbxfPYW6CSjbujDTkLuBydi7'
    'xALDvJovV7oMnRo7xzWiO55gjDuG9A67Ntt4PI3jqzxYQd67NPa8OqmNxjvOMLQ6uO8hOgAnPjzi'
    '7ts8SOnnPA/Qijsi+pg8UpUnPNtCnTxSsL48UEsHCEpT6JiAAAAAgAAAAFBLAwQAAAgIAAAAAAAA'
    'AAAAAAAAAAAAAAAAFAA+AGF6X3YzN19jbGVhbi9kYXRhLzE2RkI6AFpaWlpaWlpaWlpaWlpaWlpa'
    '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'
    'pGBZvdPti7wyixY81LerugM2iz0Vgnu8e7hAvEb/obodKsc8xW91PWlxfzxQHg252QPmPIxw+zw6'
    '7IY96Z8wPewWFL3z/nS9VPM0vbvpWD2VCCy9enahvOR+O72qGHc91cu4O8poDr2o2PG8R2ULu1Gd'
    '6jzbgY+8BfEuPX2WJD1MHWA91fMHPDfSlTxQSwcI0owdggCQAAAAkAAAUEsDBAAACAgAAAAAAAAA'
    'AAAAAAAAAAAAAAAUAD4AYXpfdjM3X2NsZWFuL2RhdGEvMTdGQjoAWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWoaAyLuZLRc9YROIPFyRIb3K'
    'fi87+v72PGQgKjwfgXm9bUiAPS92dbtoU3W92kUuPT3eX7sM2zM9LGsGPW/gkrxhRiM8RHi0vLbF'
    'QD3KSF49oC9/PdiRcDzTXgo9q63NPOXro7wg2uo8vQi5PGVdLD0CA+m8ci4ZvRy4CL1y5Hm7UEsH'
    'CJe+WdeAAAAAgAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAFAA+AGF6X3YzN19jbGVhbi9k'
    'YXRhLzE4RkI6AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlroQH4/wXl9P3MagT8N2IE/xU+AP6HigD83TYE/ycWAP/4ggj9qKHo/0wyAPw98'
    'hD/hXoE/1Wl/PwuSgT/Pk4E/y4WBP1X3fj9Q74E/UeV/P4gogT/e6YI/a858P6ilfz/R1n0/mkJ+'
    'Pyowgj+jdn4/XXaAPxL2fT8OD4E/FJWAP1BLBwjKeaEZgAAAAIAAAABQSwMEAAAICAAAAAAAAAAA'
    'AAAAAAAAAAAAABQAPgBhel92MzdfY2xlYW4vZGF0YS8xOUZCOgBaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaf9zyu7g0H7yxH2+7AOEVO6Ix'
    'Ibui+D67fvdLOGS5LDu58AW7NVyevC2upLqN0xu6tUMsO8AjAjrK9fM5PBjJu9DJhTv9ktg6B+q3'
    'Or1P5jui6J8764cPu+Kie7wC/pO8uutlu24xrjoZ5s66XT+AvP8+bTykHku87/vJO2/itTtQSwcI'
    'hb5gfoAAAACAAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAATAD8AYXpfdjM3X2NsZWFuL2Rh'
    'dGEvMkZCOwBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWgeufD+SnoE/1OF/P0Vsfj/Ku34/JxZ+P1CTfj/eL3s/j0F/P54pgT+kHH8/rBJ+'
    'P6X4fz/ki4E/FoCBP4I4fz+lpoA/5LJ/P9MNgD/u5X8/HFuAP8YLgj+eUX0/zPGAPxo1gT+Ih34/'
    '0dt+P6uwfj9noYA/RiGAP0Irfj8ukoE/UEsHCIL2Qa+AAAAAgAAAAFBLAwQAAAgIAAAAAAAAAAAA'
    'AAAAAAAAAAAAFAA+AGF6X3YzN19jbGVhbi9kYXRhLzIwRkI6AFpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlrGf0o9Cg0PPUyJGbz/mrM7EwLG'
    'O4+WQb0/oVI8gMRNPXbn3zx8hm87avTsPMuBFT0zVUo9FceAvRV9fz14Gfa8iHc+vbPQWb0G1wi9'
    'W7kDvc59XbzLRkK93lfmO4nsCbzccZC8xkIut7FZOr0FPJE8QFGCPb+8WjzGd2a9rwQUPOSH6jwe'
    'Rvg85N2PPZNfcr0hCDW9w+UJvKB7l73hHW48E/waPQHLRLzdbA+9r4kOveu0oTuCu5c8FZgjvTtn'
    'cz3N6oQ8aMRuPZuLaj2OjIS9J8BnvIs/Tz2ESM66DzfTvK+Bh70KbiK9xHlKPdvnNTzmnIY77err'
    'vBgjSj0SL1I9vZGFvfb7XD1YV7m8H2THvLC9NL2JwLS8BGeDPWdnnjwKzL47BdVMvfN78rzH8mg9'
    '2/3jPGJh6Tx6K0M9y5tpvX7bnDzUp9S7F/dFvEsQQjz6ADu8p1+FPGsg4bybCqA8fpn7vIrgXD1G'
    'BzI9w8EsO8K+/js7VHA9peUuvQivmzzP4PA8ht18PImKOT2Zhd68GdnZPMR2o7nbKis937obO9hg'
    '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'
    'OW8AvWdLKr1Bxz69Ury9O8KWHj3Y2BW9N+/Wu2FLJLuuTFk8hQVOPF7+ZLyEcYA8YesUPZ8tszxU'
    'Upe8XrYQPXvTDj3+z6y8aNXOPKtrUD1wTCq9i7bpvNYKb7yR8iA9e3cuvK77o7yIcTe7qgk7vcwF'
    'jb24kt08XSolvQGyHb1otzG9B4VsvFc0g716zYS73xfavKgZ+jph6Zm8+N5bPWl8VLx0sXI5WP3u'
    'PGrXZTxRZzY9edyyOwB2BrySrws9H/ZevSAvDbz31187gYYUvV2MtzzRqya9+kYQOzHw/7sJ/QK7'
    'p9w2vY4wY73CN6S877IuvZ9bFzz9x948tSYbPbRMUj28xlO8p/tgPSr43rza6om87M8Mvf3P3ztX'
    'Pmi8zAI0vcps2jxcRic9eNkovbZppTuQQj+9m+fsujrQSj0jFVG9ze3XOyzM0zy0UEc8s3Y/PUmj'
    'kjziWKU8DHITvEt9tjw4CBs9YSFQPXwngbwd/hK9DpnSPFYHFD1kQEE9i/5WPNzIkT30qji7/QQA'
    'PZj2YD2TVaO8d3XEvB1TfT3ifew8SIGHPCGHtjtu4Qw937QSvX9roruC1MY8HTyKvEGTTLw/hE29'
    'g3/KPCOYMD340oG9DzUbPUv9lT2zuzU9w24ZPcqbNDw4NTi8SbMGvA40h7zxZkM9LhXtPFlQ5LxR'
    'w+68cfbvvNluobsTPRy8+o0QPNDh5LxQSwcIwtgPqQCQAAAAkAAAUEsDBAAACAgAAAAAAAAAAAAA'
    'AAAAAAAAAAAUAD4AYXpfdjM3X2NsZWFuL2RhdGEvMjFGQjoAWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWvDWUD10y0C9XhFDPYOiL7zPefu8'
    'UxWovIDmzLxm8ds6AMUvPSv4Vb0FiEO8yY5QvRJGYr3/lWY9M7a1uvJV2js7gK28rf45vYQ9Tz3P'
    'MYO8kVyHu0lgGr0U4vO8okI0vdl9C7wWtHg9Ru30u+kppryoJX4914O/PBgmDr1xltY8UEsHCLui'
    'rw+AAAAAgAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAFAA+AGF6X3YzN19jbGVhbi9kYXRh'
    'LzIyRkI6AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlrUQIU/U2+GP2t0hz+ibYM/qx2EP703hz94e4Q/UD2DP1eGhj/xIoQ/beiCPyOKhT8x'
    'sIQ/+1SIP0zehT+0RoU/noODP62NhT+u2Io/Tf+FP0d8iT/IqIQ/EuGGPzCEiD/3+4U/xM+GP7Rj'
    'hD+nK4c/HnOGP9MKhT8ZsIY/wN6DP1BLBwgJGndigAAAAIAAAABQSwMEAAAICAAAAAAAAAAAAAAA'
    'AAAAAAAAABQAPgBhel92MzdfY2xlYW4vZGF0YS8yM0ZCOgBaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaHKgLPavlIzwDvmc8rghGPF0fojyN'
    'rPM8+Bz9PHiBDTxKrI08/CyLPCcU2zwOzIs8DmW7PCndDj0aMww9FduauoKHjjwcg6g8VtKpPBoz'
    'WjyQPs88PPHdO/oMwjz/WZ48/DbvPAd/gDzOEZQ8PZY0PB2lqDwE13k8SQWUPEEOWTxQSwcIkXMl'
    'YoAAAACAAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAUAD4AYXpfdjM3X2NsZWFuL2RhdGEv'
    'MjRGQjoAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWnjXGT0uzno9hsxPPWt8PD17/gk9ViwAvSe6N70F0728FZ1dvRkyE7wyqUU912q+vE7+'
    'WDzIFFy9JMTzvH0oRLyiMtG7QHsQPX9bSz1PUGM9A47DPRycar18C9y8yQ2RPP/YITwiHP68+cqU'
    'vHHgvjzKa5i8FIQbvf3ntzwJFCC9xZN5PGtfVDtWHYq8HZ5APZXiTD12fZM9VFkZvXNkAT2vyS29'
    'zqEYPbSuBz11oaQ4rs1hvV6brjrW9Ok7gD0FPU7zxbzOjW897i82PYjOJT02MYo9M7PdPKd3Ab1f'
    'jwW9t38YPW4YwbyX7Js8Gx20vFaYRD1leUm943H4PMe7ELvCpo689UCqPKQ1DLwsq3s6c8tBvV+B'
    'bLzTso+9ZpDNPG8dcTys7v48Y5lzPdkvE7pnwEQ8NzYEvep60T2meUS9xAEZvLyRyTykXJG9He+s'
    'vflPqrwXz4U8/LYMvdhdtTwx51q9lfM2vB+iKz1Wtai8PjXjPPIiRr08WYg7NJ9iPM7XAT1jVWM9'
    'waCMvQCemTxHROs8BnDgvD4Q4TubWYA9//MPvbBGPD0Qf4y8xcqAPSqwcjyN4Ek9v9pSvTChND08'
    'XcK7LJHGvNpyxDyqn2S9wpf2vDJXGL2IHSw9mMBuO6SOiL2DTyi8k8X2vPhZdTudC5c9GF0AvWHG'
    'Uz1SMSg9QWZuPC9WVz2GEco8PMtdvRYTQ7zorWs9lRlZvfEL5Lzz2Cy92q2bPAt3Bj25OFw9Z6JA'
    'vfR35Tt3P9E9MUuqPR7JOzvQkUU9WY5CvYl2Cjy3uDW9gWGWvWIk9rtNEKk8wPJdPb56tL35EcU6'
    'LnCwuRO1lL0sL1A8w5CNvJqTFr3gnkK980qKvaseLb2tSQy9CMu4vZMAMj1t3XC8q3TKvI63ybyZ'
    '5+45zZ06PeiUZb0GFbI8CPEKPSC0Wr2HfRq9/FN0vHEsDT3ZUPw8Uk4SvWp7jTvVIlS9TEB7vRLV'
    'Az0W/4w8/4lIuor2yrytRIg8YxtivTe8WrxuSVg9ViPevMMF+zuj6ZG8mgDtPDTfGLwnBFc9NR1K'
    'PZAawLwWwh490+cPvd+a/zzTQjO97dBoPHkd9zwRf5w9CN2ePYInTz3+uMI9JASBPNfqlz23CVE9'
    'wuHAPPNQozygb8A6c/kpvSjnSb00spo7FlxavTMEhL0VrTA8uCxcPYsgoLsbFd+7B5QlvU0/5byp'
    '56a8oPFTPTPnUrx/Tdm5/gt+PLrScT00m4W8eLCvPMCrhzwMMXs7vLMCvAjPAL0EqkM91LvOvPqp'
    'LDo+EGY9TRQwvaqFkjvdFTm9emS7vWBlFL2p0kO9s/BcvdwzAL202mG8R8mGvQTGnjx9QsO9jcrc'
    'PFMlEz1VDco86af7u8uWqTxjpiI94r9yO697XT1hqHC9bb6wPKggOT2h/M+8CXoBuzkaNL1TKhO8'
    'jJl2vAA62Dyuub888oSOPKrMnz3keKs95oMdPWfRjTtDDQG9yeIuvV1kW7tn0pm8uJu3vN+UiD2I'
    'p588b+KUvK5rOjucjxG8n986PW9nurxhW+Y8Bg5uvevAZ70rZUo7zhhpPZpU9TvYvX+9Y6rfu9vn'
    'M72zg4s9XOWgve3Tbb164WU8DbowPDcOSDsIASG8LPDgPJ4cxLyMLEm8enuFvKDpITzS67i8htqU'
    'vLdhg7qHSKA8IA8BPPhy2DwjfN27GC3JPLFsSLwes7W802EtvRTTNLwtOmG9JydePZPh9bxr1l08'
    'ymxdvIDfi7yivIq9bF2jPIwUfDyRa7q8Kc9JOi5yIDsPpVE8mZcIvdgppryJi9U8qBZ8PAM2mL1m'
    'w/a8x76KvZhjnrqmGoy8iic/vDn4P700Leg7M3xlPBUGCTzrtY691ssQvQ+gzruw6aA8/Prwu2D+'
    'Br2qbkG6PVIhvAnbEz1PLH49gTK5vE7urj02Mbw8AXn0vEyh8rv1mCk7t2SPPdr32DtcIQs9uyQZ'
    'PabDYD2U2dE8rMVqubRKLD39ODa9geJdPShG2zzLIwm8N6d+vcgFIj1GbQ49NkhdvTbhPb0uk3G9'
    '8b6Mu9YbZTwdXXA7V151O3CmlDx0/ky9h9UwvRUPBr3HAys8jo53PIy66zwqeX87Baz6PKesXzzQ'
    'fUG9sUNmvEbwSr3UCke98gw6O/xGvroqosa8i2qVveglobvXe6u8ERGhu8oBpjy2mNw7gM2TPW7o'
    'oj3Cu+u82KqoO21TsbwJjd28nw4PvR5W/jypgKA8kZ6ivNkZi7zKuKG7/shMPfmtpjzDEYI92cQ4'
    'vR/1PzsBBTC9z+QIvNPirLyC2iy9Li/gvR5OeLzrwCI9C8sNvfjohLj8F8o8yJZaPdpRXbypSJK8'
    'n+87vD2ulj0nml895VfKPHVhET3V8FG9skVaPA12Ar14V6k9PROnPQZg9rwDgI09ahyWPQQdorxn'
    'pJ+6ArbEO/8Q5jwe8tK8CCJrPeD9dbyEajc9gT/JPFoeTDyLF8i8ZiTIPMXk7bx+BG+9fBQOvWXX'
    'lr1tmJa9BqjNvbagNDwdqIq9k7eGvTok8TwSrFA8E7oNveXvir1e6Qy9mEOdvaK2sjzl9h496hVl'
    'O9NghjzRNyw7is6EvZwHmb1tjkg8Fi3IvVbFlbxTcWg9MmEQO7eARj3POVM9b0qCve4ZYDztbL+8'
    'PwHBvJxyZL0YRq89bDjKOQDlHb3Xe5w9z7DUvWOQhr1QMGg6Q4qNPBp0rDsFmck89jLcvOSawTxa'
    'WRg9e9wSva0YcrxbOAe9rurUPLvlhbw4raC8qKUAvUra8DzNI8268SY9PC+Xgjy/rh49o50UvXFc'
    'Jr0hFsk8jJYzPWI/MT3y0dI7QXUpPWFOlzzuKKm9ARiUveZroDxq0IG9i+K2vGB6rbxR8229MUQz'
    'PLcjmTw1Mgo9E1ztPH2RU739gic9F1Zcuo+vQz3HApA92CNiPQx8PD3wApg7PKgZPVFMmbqzbwY9'
    '+bysOw+V37zzBds8Hagivbc+7jyGqnm9rKKCPKPYG7x04IM6XSJBvTJTdDy7gqO89imRPMw49LyB'
    '/mg6N+Amvf/UU70Vrjs97yn6O78rAr3ZYV09VnMivR+JfD2lK+e8PwdVvVgoJj1YKnc9N0QLPWoB'
    'Vzy4Hxw9SqqKvTBPgj2Ty6W8cAqSvMPvNjxbVvm8HaBfPRx2CD3k9i493ZoFvZ8Ep7wC8rc82pQX'
    'vTw6qrxFw8K8cdSOOeNaK73YXAs9F6uTOhM+Z7wH2yy8gJnrPJJbRb2Zs449zZi4PCxrWbxUO+g7'
    'YUJ1PTIhLz2fz3y9gRXLvJuZLj1Lneu8pcVYvZhmrbz9mjU8pZOKPPyDET2wEba8m8yKvSrayDxH'
    'Lwg9EA2zPGPshrxseEs8YOivu4++WT39Exa9R/msvalvFD2bhgW9VPkbPVJRbLx1e428kIsovWum'
    'Xjsj/zC9WjENvVUfHL1R04C8A8BXPfHkv7yJCEC8wONmvXqfPj268kY9He5wvO3W8TxKOxG9HBze'
    'PDopiD352HA9ZZ0EO+7bhDzPt6i9HqUTvUI5dDz7bAs9H4SBO6Pr/7wssjO9YlVIveGbmz3hbDw9'
    's+InPWxRJb152V09SRbjvCH3Nr3PZig9dAr5vMs/mT350ls9z88OPbskOD3ZFlW9YbtfPUFieT35'
    '27c8gtHFvGJlhD093Ei9JpHnPFoCuzxps1o8oSbXu7adPbxSzYC9G+4fvQ9Ydz1t9fS6TB8xPIKV'
    'YT3SYwc9k90iu3l9A7wF9rs99ZkzPW16mDyx+zY98YYBvY/COjyVTQE9iDVTvU158DsLUFO96+ik'
    'uqQJvroVHf+8FVBiPYtrwbyEaIC8+0hDPdcTHLsgMya82j41PXwE3bzSoUW9suYMPZxu37k7kxw9'
    '7IC4vBz3D7zp1o67W0slPRUDDj3xDLk8sqagPCV/czzljr88iyudvEaVQT2VTHK9KXQUPXF++jx3'
    'CUa98pc1OBXl0jspFv08vC9cvRM6fD3rcZC895AkPdOhtD1aXxQ7UZdgvJovm7zqVo+9GeitugwI'
    '4bvN4fG8kwySPCTrHz3IUY+8unQFPUU2nDx8zt48OuzOPBYWNT0CCbY8W7rrPIEadzy/TKW8gREn'
    'vYiZk70sqGC9Vx2APU1R5Ly5eJA9NjJzvYlmZT1hJgM9oq54PCzAbbyu8AG9+kEBPWUilby4NpE8'
    'jLmbOxE9NT0lEoK8n1PmuVF7lLvoo0y8FXVWPIA+lTwKFaO8Fc/evE1HRr3zwYG94cSHPenF/TzA'
    '3ka9750wPae3mTojLoq6QjN0PY5MYD2nHRQ8fnQ4PRmRuDy3xH+9ktSqPLtX2ryoTVs9pLFEvAhp'
    'pjwN92w98HZtPaRVNr2IpQC6FuwbPbbwAb0FIqQ6ct6SvLrunjx8Qw49jWLEPOYTBT2GtUC8Jj9i'
    'vfIhlT2GoXI9xSuvPCgdMr1VMT+9tRoOvbMX9rvjCZq9iujlO6hDnDwhntw8AsJQvXV0HT23Xas6'
    'nYy0PP690jpGbgY7bDk9PcobBD1udZU8Wu2HPViebz3ScNo8aXP2PKnGRjkNDsm7G6giPDIglD3P'
    'uhs9tpcyvGqMxbuRnzk9E+ByPZOtqTwO3ME73Km9vHKuQ7wPDCC7JoUMPPMHBr2F8L89zjX0PFTh'
    '8zwfnEy9Fht1OYtvXDyT4rg8chxCPExlVjwwA9O8IGa9vKGUTb2XS6U9Fp2rvEx3dT0OUYQ9dNA8'
    'PEF4qrzIclE8/yXlPBEVxDztjlG8NY6IPe/nnbtaK948EqNruKJKHD1mf2a9ZDyPvTiIkrsKiWk9'
    '62WoPN6U5jy0ZlM9nWFXO9dB6rxbSDm9cbp/u05RXTx5bbw8MtZLvCvgdLv9/PU8yuVYOv+cljyk'
    'wce8DVqAvLV1uryySka8UGevvYfsajwxFZS9uF/OvaT5Qz2TuQ47lmqovMGk7rxcGRy9SVoRPeKP'
    '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'
    'GL169Zk8kkRIPViM/rt69lW8aNEXvVrU0jxy84M96rZAvApmIL1bb+88QSrFvbhgFr0povC8/prp'
    'vB4lKDy0tTY9b0YDvYT5RT1jEMy84/ScPUpNUT0bJpi7Qy1qvRv+kryb46q9tPUYvRP+lLyzjHo9'
    'wApGvTcdfD09YyW9LYMIvTJs2bsXYzk9/3eiPCa8IrzFyMS8MGHLPCX8Ej0jfWQ8iZCgvPL4sTws'
    'FKe9dQThvKdNgbzPnLw7wHfFvG0iID2aNH697sucPdedvry58IY9zbkrOz9Hqjw2F1W9beXeOtYd'
    'e70rhSm9Pi/dvN5xiT2V4xq8TOcwPenCTL19+Ty9+h1cPHCCZz1dSzI9t2guPN12XL0nNiQ9Aag2'
    'PeHXQDxsOSS9QKHaPPiNmr1GLJK7nnA/vasU2DtDsq27FbNbPQ7lNb2I0bo9eQZHvb1hXD1JtKQ8'
    'WPr6vOP6j71/9fG82hiSvXPLl7xD9Zi9TcaTPe1Ne71QSwcIFyo8BwAEAAAABAAAUEsDBAAACAgA'
    'AAAAAAAAAAAAAAAAAAAAAAAUAD4AYXpfdjM3X2NsZWFuL2RhdGEvMjdGQjoAWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWhOQn703/Zc9YM7B'
    'vd3TAD48hnO8qwuVPTaKjr122QU9UEsHCObB/KcgAAAAIAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAA'
    'AAAAAAAAFAAeAGF6X3YzN19jbGVhbi9kYXRhLzI4RkIaAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpavgdKvRaTkb7uQ4W+9VZuvfII2L10uBc+eVlVPv/8zb1/w4C+bWSbvKRqnz26iNu9NDd/veDR'
    'YD6oUhm+6yg/PumMz72JlwW93gsxvmpmfT6sMpy+0v+WvZxV0z3VEi69xYIOvm+HXj43jxG8qx6A'
    'vrrNaD09Y8o9XdQcvgN+Br5/I/C6pLvKviFRX76v8gI+C6sPvm2ttD7ebYc+WPUZvmz5i76R1oS+'
    '+xBTvjXqSb0WheK91woKPr9GM76OGfA+TyEFvl04YL798ga+4P+6PidQEr4dKbU9wEUtPtIq2729'
    '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'
    'AGF6X3YzN19jbGVhbi9kYXRhLzMwRkI6AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlrn/QU/tc/JPukiBD/NcsA9sS0rss803DqLGvI96wth'
    'u5LhuzxclMm9j6auvVwsPT5Dtfo9Fbo4vSuUVz4lDsQ7YW2rIBBwyT48GHA+gG7BvN8oMz97Pwe7'
    'LH8aPu02fD23HKe+RTnUvVC9BL5PXEQ6iGCPvAPRnr4cZ469OTa1vvSJg75sXQebddfVvlpOHLjh'
    'lgm6d48XvNk/VjKqL7y+UlgjPz6i1r6GrBe+Og4OPGAjIT4m2zQ+Z61KtxAx4z7tOiM7PrcKugCA'
    '/7v8hWi+M6g3PhuXJzxscu49/3PLvtvZxb6PcN6+9avfO5biH74uSdu+oegPPmtkojViueu6pV7H'
    'uhRu6z2KfY48CbQGPzFHWb6V5MU7Vf0avloNfRFEK1y8KlfjPkQ+lj67Lb+9KpgHPi9xSbwd2/Q5'
    'QNRvvVKGEr2ipIaomJywuqJ3mrxU8fu87WmQO8SEQz4aQgs9oDiVPpqsmj7p3OC7XZl3PvtQ9j5H'
    'NMw8vouXvqYM/r7DLoK9ZoIhPzOOL5D6wu8qzr4TP226576HLgI+UooAgMd/JD4jXpO94cNDvqsY'
    'jz7flLA8y/y/PorDCjwrrF6+4gyBvvd7ID9bJtW8jRhoPJCiYDs9qj+9WGBMPgH1FLhzDds+yCFo'
    'O1FIuTzASYI+8zatvUOKMD4+5zG9S03DvlBLBwjKSzDWAAIAAAACAABQSwMEAAAICAAAAAAAAAAA'
    'AAAAAAAAAAAAABQAPgBhel92MzdfY2xlYW4vZGF0YS8zMUZCOgBaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa+S11PVBLBwggkp/gBAAAAAQA'
    'AABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAABMAOwBhel92MzdfY2xlYW4vZGF0YS80RkI3AFpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlo3hYK9'
    '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'
    'VHvcuuqWP72Su1a9MquDu1tzW72K2Xk8Q3cavAgF1DsPSf86BNYmPe56rrximV+9dPqdvNUK5zyY'
    'PyC9PIHtO5y8aj0NxAk9+g2ePFvTOrxeSRs8GDh9uyOoIj2MYvM8D8UKPKogWDxjkr08VxxUPF+l'
    'G7yXFFm9kAumO3CSozshowM9mDa/PNY4Sr0XMvS791mJvU6c+Dw8FR48NiokvYRqDD1a+iQ9V3XI'
    'OqNjwTw36pY7LwcxvGa7Bz0RqoK4hiijvOyM9zwh3mG89GDcPDv+YT3eB9K8dMlKPM4xuDyHoKQ8'
    'XgsQu2NHTDzTqaI8zrczvQaO5jwzXY+9igMXvFPnpjyay4S8YPqmvJHtdT0poMs7wLLdu69mkztZ'
    '7QW9K2CdO2q2gT0Jcio9C3t8PWQPKz3fFok7oOkbPeZVAD0tz047hZ6GPII7QL177KA83C4qPBKe'
    'v7ySPLc86hvCPJ5iBbyRFC29NoSGPdZMSL0IijU94TI1PBnCebzpux+9xEBzPcZOYz0iaie9Wwqb'
    'u73peb3L3A29O6e+u2F2dT1cQZG7j+GEPaupM72n1MW7XJDAuszC8bxaXY27ak1Tuzz+LT2ouGW8'
    'T5aFvNMD7jz/YYG8T/fpPFh/BD2zfAa9O3SuO/d3hLzXpRw9PpZtvCDaxbvTeh09x7R2PMH7FT2x'
    'jbS8riAzvSHIZjxjVE89nNQ/PaTZrrwq7Wg9JsXdvESrYj0K7ta5sTwKvbiK3zvQJS89LpklPYzq'
    '7ryY0OU7kp2+OlSLN7xJiOi6abtTPUNJjrxI5RY9LttUvVZZqLz3bkw8ENenvLpgM709mXi87R3v'
    'vLHTkzxqsdE8UtPgvHenIj3E1dQ8kPZ8PXa6Br3wOVq9QHYYvGx2wjxLmpE88WpovTUCKT0v5mM9'
    'O3uSOqwnKzx1hHQ7/DOCveYmUL2hNgo9cDwxPZ+8rDurGzC9KXudunwMBb06Ky89DpOLPZliaD3+'
    'ykE9wwgBPalDmj2FzK68WPwAvE09LL1Tt6Y8LwSQu+S+zTwRmRk9R8cSPd4dkL3QjaS7ZiE/vd+I'
    'Rz1EmhS7CDJRvSD2Cj1abAK9DKUcPJ6DTT3ktjE9Mh9kPGNPAz39NJQ8Z2vlurhmVb3+Eay8ZfYr'
    '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'
    'D9U8jM2ivb8NQTxZvFI9zcztu89hxbrgiiO9quZevW6OaLqKHSQ92nhbvVDT+7z1XcY7Yv82vbvi'
    'ibxEKFa9UULMugkfML2tLSQ9HCNhPQQaBz1hdwC9odKGPOgOjLtQSwcIw0MwtACQAAAAkAAAUEsD'
    'BAAACAgAAAAAAAAAAAAAAAAAAAAAAAATAD8AYXpfdjM3X2NsZWFuL2RhdGEvNUZCOwBaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWm1LVb1C'
    'vDk9OeZdPSK51Tyfyzy9VLumvDaIuTj9ul89JeFVvRsmKb35/u88HxW+PDvdobwCiji9M81KvL94'
    'Mz2jflo9GbAxu6QDDL2C3ow8I9EePUXAHb1NZ4E81XgWPcE10Lt9H0W9u00CvWOgcb1VICg7B40X'
    'PRpaSLtnESI9UEsHCExvMXqAAAAAgAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAEwA/AGF6'
    'X3YzN19jbGVhbi9kYXRhLzZGQjsAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlo7BoE/oTp+PwJMfT9Tznw/awd+P7icfz/HDoA/Yql+PwUK'
    'fT9Av4E/kph+Pzkkfj9AKn8/ZTt8P7BRfz/CQn0/7RyAP/Gnfj8fCH8/3iuCP2FzfT+je4E/CtyB'
    'P7/vgD9oF4A/RAuDP7HwfT+CSIE/uBJ8P90nez8y2ns/F+yAP1BLBwgJa9pAgAAAAIAAAABQSwME'
    'AAAICAAAAAAAAAAAAAAAAAAAAAAAABMAPwBhel92MzdfY2xlYW4vZGF0YS83RkI7AFpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaIHmdOyuP'
    'ibvNbMK7nm1rvN4ZujsFRCy8psLouG14DDmEl/O6m3h4unSUUrs00Zi6tSqGvOjLXLyOJo+7GRWb'
    'uy+GuLsFqTW8l6t8OuREgLuzphM5Bihdu9FOWDvKEfk7jpcku7QE7LuApd67/w5HvJ5UNbwhaIO8'
    'VsqnvASR+DtQSwcIHeu584AAAACAAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAATAD8AYXpf'
    'djM3X2NsZWFuL2RhdGEvOEZCOwBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWv3hgr0SEE89+NrQvAcsRb08YTI9VwTmvPhehrwVYpw8puE5'
    'u2c/brw7pJY83KA/PcNovrxX+cw6XsTMvPy/xTs/bvI8/awWPcQZ4LsFSga98JkEPQB8Qr19LWO9'
    'qaSmvLeIGL1uM1G9ntTmPNmVrzyrVcQ842PLPJgrNjyWzT899zsjPZ5mKb16JYI9PH/zPNbWXz2Q'
    'izA9QKJgvJakyLsozYM8F3VxvWVtu7zFbGm997p7PNLuJDyPmjM9KF9avWfeSj2c/A89UmN+u+ZL'
    'Vb2sT1k7wip8vIJiI731Wwu9lHX7vNJGobwDgdU6Tel5vaYmkzwC6wu9JuJLvSx8LDyg92a7bMhJ'
    'vRCwkD3XIkG95U0qPXxmDDyqywu9GdQKvSdf0bzkyVk90tc/PX1o/zzS/109B77IPNVd6DzspEk9'
    'Q8EDPQxksrs7nKw8ebKbO5SLY7z6vgY9HRkPPSB+Fz3j4FQ9V2KPPSZ8yzzx8H68UGzUPDMe1jwP'
    '5B+9qh0YvXXfUb1+ljG93NQuPVPW7DxRDwW7rA4Tvb1rbjwXtDE83poFO7lRLL1JFHc9XFYfPHXZ'
    'cr0RmNU8oimMvNJbO738OiS9sqQ2vMzf1LwGIyu9jiQWPeFJJbzEVyW9ZdhUOxmFTL1goSc9pMFI'
    'vcAO5byB55q8PHI/vcTSI71D0BG96WAnvUc0Bz0w1DY9kim3vMMZNz1KMd27dzZvPUR4srwVzLw8'
    'LPcbPPeT4byonGU7lRiwPJRt6jwvioI7gByfPIrlMLv7Muc8Hy9XvXsR/7tUhpm7ZeIYPceudr24'
    'a8U8uXKGPXEz7zzV+Ro9BFfbvH+iWThBrE49JBoevT7acLzHoJu8y4JJPfAaIbq0oz+9pLMBPcim'
    'aD14nv+71QAzPT0VTr0RlK27eFB8OxCb4DzIIV89c9rrPK9XQj0EuQa9esm4O5YWLz2o6CS9N/AO'
    'vfSXhb1eW+W8/e7rvGOzKD3/Jwy9gplXPOPjVrwSzRo9LAe6O4vpfL1OptK8LEoqPOxfAzzpen89'
    'Pw7hPNNODj1VZz+9T2GRPCYtoT3AfJY8qj8bvXO9Rb2Ti1S9IZY8PVhYGj3D4lI9pSGvPNntZL3T'
    'd4I8XIcvPZEd+LwHwV29/uOvvKxm97zG2x49vHMFPUSzaj1WfCo9tMSDO9BZH73T7bQ7wXchvIns'
    '67wEfye9zm5DPZMLnL13zUW8kOzaO3vghbx8CCI4LyNUvZ7zDzwdW4s9vOUyPebDFbwhXi49lvk6'
    'PY3lUz0YH/O8rcC0PJfhkjy9tYU8QMAaPeSbKL39ANC8nq6rPOxyqzz65QO9j2MfPdVZh7xiBmE9'
    '9MYIvc73sbwbf8E8asMFvdUr5DtefVu8s/b1POouaL0IkSC9R7dovTR+PDxBBAq80xTnu1+AQr0r'
    'gRU9CdjkvF8qUL3kLn26p8tVveoTv7x0Mbc7ZOn6PA2BOLy10VA9vsw2vIZLzjypyiO9eQg1vOX6'
    'Mj3pY808ZC9FuztS77yMy8i86qFkvRslbjwcrVg9CISrvBJ3Ur0RS5Q9PartvJ0w0Ly887a8riVq'
    'PcEDfTx09HW9t48sPRZMxDysXrw8hUxoPcnaiDw1xAC9biiPvJOeGz3ltfe8FZEQPRF1GL1kCws9'
    'IqJkvDhlLz1j8+W8QxMUPFFKLz1Ta5S7vIEjPM8z0Ly0sri81myGPPVxTD2Wmr68MgjKvG8iM73u'
    'cTc8+Z1VvDr7TzxQxeq8r4Q6vOp6PD1uIRK9Ca4NvTPmeDtyHDu9S8gOvLdHbL1obDa9/w6Ju3Tz'
    'PDxeJha8e9SAva2jTL2U8rm7EUqxPAjpiryBDAU9FpTLOwtaTj0/YZc89dduOo8zorzR/M27i2hb'
    'Pbi0vzz3txG9c/wxPcsW0TwCJCC9LEyeu3RAVTyBFno895S3uvEQvrt8BwU9lZCGu4eW+rx4zhs9'
    'PofePEPIuLzbUlK9MxK5vLB9SL2nzoa8GO1DvU6jMb2L3fE8pQ+ju9k7NTz5Xge7rwjzPKAeTz1J'
    'jZe6kDckPIM1h73E02u9Au4rvTOmAz0Tzq48IKm0PFpejzxurq684LkKveb1ID0s+4u8tpgdvELd'
    'Nb3K7xg93CAVPe5Unrw5rlc9740PPfhStDzgtQm96d4fvSsDEzwHUsI8+tf7vLP6UT2n18w8ugUw'
    'O33vYr2aISm9coUMPTIVhTso0Ii8ffIKvWy5rbzAyXk9CigdPWgForrqhj09OwMEvTTUErxNk6k8'
    'DNTzvAZQMj03rr080riXPFeXKb1YUAo9a4JNvP6rgbwY11I9NDp8vLZreT0uBGC8z4IKvfBzRryU'
    'tGS9mb4mPVPYBD2eooc8fUI9vUPLWDy1nRo9aZl0PdyevrxBGg4937wEPetnKLzzEx08BcoZPXR+'
    'v7vxYqu8rVTPPGKBSruSSCS9ZYCIvBgctzxvTzM94isDvWjWRj3rsW09E7j2PM+4Gz2r3oi8+YIW'
    'vW+dk7wx7X091PzOOz/iZL0dsze9fJ9fvYzfDr2o7CO9RvTbvF7Y9TxAOFS7MvJjPcV/cLs8Rlu9'
    'qEOovNVvU7wJEDG9I7NRvemGJTvbTgS9vLYVPCuOJbzxiMe85miGPBaaX7x5YmC8lvhDPSb1Pz3W'
    'twa72umPPERbhT3hlTs9HfHBvFBEQjwy3Am8KWwSvSg6Gj1lrwQ9j6OBPCzxcz1UeK88LcMRPfqv'
    '3zyfbMs8xkP0PCP92Dw1QjA9ThK3PJOxrryLYS09viZ/PTmxjT3dX7Q8zs04vFf5kz2RRim7l1f0'
    'vFfTzrtUoSY9RCksvbo/obwg6Sc9z6c4vMHS4LyRhJO8ZvyqPALjNr2031o8OqeVu8r4izxa26I8'
    'MvWcPOnuT7uRsyk9yYg0Pd9xQr10Slu8q2MpPfCE5rpepxq9WLsBvNHoebytLx29gJiCO3MMTD1M'
    'Rj070TI/PRgUVz3u5mG85zI3vWk/JryXV5s8MLTIO/kPyrwDwwK99wWYO2oUJD01fAE7Apx1PTWk'
    'xrzMc1o961QHvdYWMT0zlhK89zGRPSQqUD1bA6y8Ubg0vOUE87whF5M8SW5WvUTarLv3sha9o5Ue'
    'PWOMhD13Un2732+/O21BX71WqYO9nX83vRiMZr3W5lC91uupPPPbNr3uu7g8mc3jPLpI+TtCb5A8'
    '839IPepMd70Wn2U9Wms2PejlFz2pZwO99iimPF9cObzKEjM8lJZAPfR4SLvmcKg80yU8vEEdGL31'
    'vTA921pFvc4KLjyg6ti7dvJvvLKtQL33e7m89qQdvYAYWDvj+Ys8eQlGvY4hwbxiT469n+oCPe/s'
    'V72lQpi7M0spvH1uG70thSI9SaLDPD5lxjySH/Q8QQUrvYmkJz28MIM9hr4eOxjCPrw+YW477qh2'
    'vZdyCLyBXqq7n4rnvPaMGT14vlm9zmKNvKpE9DzAsB88s6PfPFD43LsPMla9cRIgve2Q07w8dTS9'
    '9qopvd/rBb3y6S090M84PUOFbTzwaRu9mR0FvSbqjj2D6qQ83kuDPLuuJb1Bvwy9UFzUPKmNeLy9'
    'QWY8KiqevImtUr2fCSe92AhNvTAqOT0pCpU8IsFlPHytDz3KqbY888fCO0aqQT1uhIG5vQd5vcCW'
    'Uj2Nmre8b1JTva8lb7wUuSc82g4BPZr4jL2trD69V2DvvIAcD73DI4w9wwW3OjktRT13LeM8XgtE'
    'PeUW2bwCqaI8v07FPOCN2zwv1gu9WOapvAQ9DD3/95m7YchyvYzXZ72g/KC8v7nLPIPEirwwKSw9'
    'J+2kPFB0gL2WsAC8NCp/PVlt0zzLre68hI1avRmf0TwYCCE9B5H/u8p2YL3ReTI9presPGluUL1V'
    'AhE8rVhAvLSJdz3oNxI9dVVaPJj5w7ycsh+9g86LPGl0vjyfNWQ8SWN/PYhj/DxuJzm9O3QGvNL0'
    'Er3yPUS6mTpAOxtWH7wVW8Y8xeHWPHMcUTzzjM+8xQn1vMQV4TwuiHU7GFdEPdWDpbzthgy92iDw'
    'vOlJZD0mqU89PEuIvDwhIL2Woxs8Bo5nvTfAVD3dvQu9ekWMPKFiwDo03wg9hcIbvQ8mCj2Gt5a7'
    'WO5qPLohr7ywQYY9ibB/O9af5ry83Pq8nh5jvDFEjz18fjQ90s4RPAKORTxDrLo6w/e3O6TQHTz1'
    'BEq7DYbJuWdErbpYNMW8CeftPP7wHD1xCE89y6mjPBjIDzzmM1G8J2qPPbUGT70jbc28NDDLOkZE'
    '3zwRci094OBPvaqyjDxv5j29lMVEPdf3Ub3KDGE8ADJLOyN3Nz1TuDY9WFA5Pf2EBLuQoSQ8bC0d'
    'PZ6vXT0OcHa9OPlfPZ9wOT0+5jq8yhaXPA1rGr2V8sW8c7qsO9oEsLwCYnY8XGXAPHI/xzykZDe9'
    '/yebuiXDRjyZXFA9WotyuwchBr3toqG63ru3PChaNjtVdCa8o0+EPEz0RL0854G9+k+XPDbsAT1s'
    'Jia7HloqPJqZBj1wlRQ9lbEWPFCSiLuir2O9m1H/unEwv7wcRyS7RH1zPV4OqryIVlG9BFVhPWZj'
    '8DuCEC29ySFeO0nEg7sAccY8O4IvPbtGSb3zxCi9u15WvKnYNT1Y0GU9LoEAvSNfD71vCUw7aKRt'
    'vbOP8Dprmik9PWhMPbzJhj33wPy8xQE4vAL6Pz3hp/685pDtu9pjTD3dNOC76Yw6PXKA5bz30sc7'
    'z7RpPbGlgblnqtS8j8WYOzcaErxZPAo9PUAAPaG3V7y89D89ecYfPQzJGD0gHDQ96GSSvdaLQL14'
    'SS89fnv0PNxZNL1GJks9tdH/O5jZUr2910m8GAhHvbuduLxXsGK6A3jMvKPJkby/You82ZhPPXlW'
    'Hj2Tbja9DYW0PIs30rxg8yO8FzXxvF7KNL2EEwO93naEvbWj3rx1G568i3tfvXv6hDyx6Su9f4YQ'
    'vdR45jzl22I9kaXqPOFc+7whkhk9mYG7u6WtbzwRsQm9qT96PbfWCr1W2x49ZgA3PXhexTvyNwI9'
    'cGAPPStxCrwhaSW8xXVzuymDXbztKF69oVboPMM9fr1ECm08LWewvOhl5Dtd5cU6U8vuvHahlLxc'
    'i3q9QtglvUg6Oj3E5GI9nXx3vY/BybxNzjs8Vt2ovKfYFzv/tl+8EymKO1WWRL3tIWu9KM3ou6oL'
    'FD1+7iY9KII6PdUA9zqXtQs9iikAvYJ6Ij131B48QqZTvVlI2bwyakY9zPoKvYZwhL1Sh4A9meID'
    'PBXceDxQrey7dY9qPSqqDj0/kny9WQ2XPAF/VD3jPE29Qw9nPJyBnLzqSC093AcuPf/P/zzPexS8'
    'Op9aPUEyl7xaYg49opwDPV+NK70FUdG826SduweiKr2zMrG8bNMTPWBGdz2aKVG8nJVxvK4ZTb1p'
    'OEI8tQdsOzzblzy9D3u96vVIvPcGAbzUsfy8mqbvPEUxHL3nJzW9T5ZDPT/nKb3UlgA91cLivF98'
    'M70j+AC8FtoUvUiBjzzP8zQ9O2oWvUcZP72iLhq8c0sivKso2jzk+FM9MzksvZkF8DzPdAY9QIxC'
    'vaqIgj3CI4Q9+M0DvfYMXD0MUW89Xr/nO30IMDpq6WM9UOxsPUr3PD1v7jO90+4mvIfcEz3Jix+9'
    'frc6O1V1C71tTii9aRluPeAbgL35OGm9QFE8Pa+7czxTTwI9RlJaPRHHxztkzKy8+khQvB2a67uy'
    'ue+8QrBQvS4JLT2P2VM82UBsPJhkZbznBya9jDCQvVDBlTvhdD+99nHPvI49K72tS8g8AkzzvIvL'
    'L73pUUo83L+qvN96Fj1ag0k7LejLPP7AxjzZWge9xdJIPXuimzx7cEW8q8UpPOL7xD3H5sE8uUE9'
    'PRgRILy5czq7QdMXPYWLLLx0qUm8kIFGPVmxK72Oxvy8gDC4O/mH/rw/5QC9IihJvUzjPT0zMp+8'
    'ye76u+FAaT06B4K9JlcUu7E88rwguTC9mtUEO+JS6jvP4BE9DOtvPf+qpTw0PSE7PnARvR6mUbtp'
    'RGi9uC52vQCcKr0/my89XGOCPIebgj3OnVE9+fftPEPE0TwwGAI9QcLPPDTDI72HrDS95jtKPU3r'
    'Aj3/C4O8WJYQvZGKW70Ixz69DVSMvG2oDjzRpIU7+yicPNq3d7uHpq48QiYkPW/DsTzzFjU5NeTq'
    'PHvb5TwbCfc8Wf3PvI8Ugr3HTMK8/KwEPZjBBb2NBli9vViXvFNX/DwjQ0A9RmVbvSey07w92T49'
    'yqzSO/sfLD3x/0W9yqvrPJvGubzI1fG8fqPYvAOAIb1frJ47lMJZPb8TZr2NfcA8ELpBvZQQ0Tze'
    'u167b1gPPJ6UEL2n+6w8T0MDPd4yRLz7FrG8N7JRPezTMz1fMrm8DJhCPF7EM71QVWW9Z8AePM0U'
    'Q72KYfy8zF/+uSO6M71q4ic9V/jivDMFNL2XsKA7RhlVPHJmMryKj3a9tezpPG0/ubyRHCm92pFa'
    'PVrt+LyJJ6M8FOj3u84ckbzRLCu9p5JTPWsW+To7Xr47nnQZvZd41TxJwiy8ZXvXPKEmDD1Byjg8'
    'UAMVPT8Ho7wIGpk8ByXqPFoD5TwdXA69HHsLPd7oBj3rVna8OGWIvO3eAL2Xa9Y86V0IuwlvkTy7'
    'Yia81/GDPA5z9Dw1g9I8zxWgPdtPij3glfC8tN7yvDybNr1+UQu8OtwVPdAUAr2INIy8qCAMvb7S'
    'WDxMnks6ZrItvXrhGL0I5fu8LRplPcAgUb0wHHy9KUmSPOGHoTw4s8e704VIvcdHp7yiYCo9E3tF'
    'PTLAFzwEYIc94+r/PC+A4zrnRKq8jZ0oveW+jbySb1i9NNtoPWIX+zktQlq951kKvaJ+LbwosVS9'
    'Gl05O64XEb3BXKa8yYE6vfXwArzx3Qi9GPAAPTi4FD0t91I8N3w1PfmFqzw/Nty6w/HGOySGujw8'
    '0le9rrBlPEYKBbzQszi7ZKp4vABv7LzZrsk8uJ4pvbd3BD0Z5oi9IQrHPDOK2zvHb+e8DrBOvAHC'
    'prs0eBW98naYvE/rmrw+zEM9WTyeuUUfaT3htki8bmtVvSvvs7yzkQ09DNhpvYoqJT0noFe8xpkO'
    'PZ+QN7xk0WK9jwQ2vTjkajyAvTa9VGO4PLohQz3huKu8VHJYPXBXwDztbKQ5V1yAvUdtJL1Uciy9'
    'jsRZvCs8bj1rFnE8KVP8u7TLAz0yPhc9i5oevdYISTtb2Si8Gdb7PFc/uzoybY66ajNFvNaJcT0d'
    'yY48z1AWPZs6SzyiPec81V97vbaApzsDpwI9tgbFPCiQZz0XDaq8Ib9FvbGHUD0TUw89xsWAPTpj'
    '3Tycekm9m+IbPEtfeT1nfhG9OaAUPEzsxLlkJ0a9V6aUPC0zezx06Cs8CPlNvb7NRz0M+VS9nmu3'
    'PHCWiLyb0+S76voDvbt8x7wyVgm9v7aKvNTPAr2yeg48Zb1xPStz3ryCl0O9yV3SPNVKWj2s9Gq9'
    'rtA0PQdjTj1LaoU8u74IPWZrJ73PHi693hyyPWfRpDyDvjG970nwvIjyFjy1+kg9+mQNPRDrqztD'
    'ExI9bpRrPMTFDT38PQ29EdQTvaKJIz2WYyE8JamBvfSTm7wZVVU8zCVXvHgAN71lHT49chN1vfM1'
    '4jzBVrC8qW0NvcHWrzyBKYe8tyHWvCpnDbxBE1q8IZlWPTjHP71vW+S8TAUyvAXez7ydYtm8I0rF'
    'OxQv5Luqn+g8ZU0IPRtQAT3QrHK9RKtXPV90ibxbsIE8bbrjPIbKXLx9elG8UAstPbnpKj0T/ki9'
    'oQXTPE+K7zwk7HA8qhc/PBKE/zzhfE09Nnb/PCJ0/roEFi29xMyAvPo+Ar2H8RM92VNrvF8c0LsM'
    '3908/6Q4POisVL1nVDK9ALvNuW8XQ73ipmY7R9XAvOs4q7x2O1+9Ypqeuwch87yoBW49eKHoPLxO'
    'Dz28+uw8Oj2xPOZfTb0ByF09ocZbvBFunjxYpVs9w64UO4r5bb36diO7lN/hvABQhL3l+BI9X6qp'
    'vLdH1TwVFDm9oVR9PG9UHDsT+jK8QNc8PRWyR72mzz09XhyovM6Byjwjqc28XyyWvJN/GLwPRPq8'
    'b2o8veoXvTy+ij682yb4vB16gbxJp1A9Hp9dvaym7buX9FS8xpoaPQlkKT3SISa9t3p2vI10N72b'
    'lkq9Rw3hPGzDcDuFq4Y9M1KbPGNoz7yjY6e8hei9PDRFeTwyW+67IVKrOyH8hbz5Y808jhoQvZDA'
    'zDzLWba6aXQ5PfCPVb1simk9k9D6PJEfSr1LeiI7/dB1PdeMpjz/LjE9avsgvAhY1LwS6gs9Kfmt'
    'vHU2dj2tew09I+htPV9STj3us0K8dzMMPcjZMb08w4A9yhBNOmPqfbwaA808wHo+PdassDuRugQ9'
    'nQ+avNGfir17d1Y9LrMDPPI7wTxWiSa9gq0WPUTkNTwsdDG9UuGCvSwZwrruwiA8UnXKvGb1oryC'
    'KFa8xhp9vSMzwrwO1wS9bAB1u+TIXT3N1gg98Zj9PBxOWj1MihM7R6nFPOv1L70WG+c8/lAfPUqk'
    'sDy4qrG8oWyQOxwUZjzjBlE9BhJ+PKciYz0wLF285NqOPCc7lDxpHiu9W+qkPHnuVr07OWm9gnxF'
    'PZBQYr1PCza9p36tvHj5ST1UHnc9KdfuvHq2qLziSdC8PVD7vG2ENb3pGJu8vcj9u2PjYLx4pnW9'
    'VBuSvGU6ej3Ta5Y8B9a5PI3MDb2g7TG9Lqz9vPiOr7y09pg79GG8vFcpRT3kA0w9jZDAvBr7Ur0H'
    'Ed68x/PRPGy1Qb1s1QW9k2SAO2m6qrwicT89Zi/ou0e/xTxRvXM86JQ3vVr39Dwa41K9X3YYPbll'
    'h7xyilW8GuAhPZbd+7pHigi90jMjPeh39DxUymg7KUlrPbDUkDyD5yw9TJHeu++hYL020A49/qFY'
    'vUfy1zwDk1u8TR90vJcsa7yCw3q8hPYqvSxXI7zky4S8XzIKvUoGED2s8zI9fZbMO8e2qzxh7QG8'
    'PzItPc3Ia73fLye9mGgPPRgvDT0elQy5DlLovDMy+Twn23M99QSEvP0/Fz1IN1U9FKVLuCEOJ73q'
    'PV486kHUvJZ4mjxhuyy9NLCYPJSKfz2wsBK9KGlIPRUgnTzCJ0u8jYYKvYmQ0Dw9exq9UoiGPHRB'
    'Gr0kgPy8QEO2vGdPAz3HT2+92YVgvJwpADy5Uge9++ZBvJviUz27OCC9MLfFvHzCAzswdE694cMC'
    'PJBZmzzhKlw84mzCPMevJT3+5HQ6MKtMPfwMp7zdIl48x1c/vea4Ir0qi3s89x7eO02Tj7wVYRO9'
    'Wo0RvWpoFDw1ASk97EQEvdd6BT2GkUq9jTY0vXw5Zz3J2rw891flvFSAqrwAwwy8uVbfvJFkDT2J'
    '3fQ608R8PeOtJj3n/Ao90Dc9vaqr+Dwn+SO934NBu/dVGz2EeFm9TC/HvPGoAj2BAxm8Xz1PPWwq'
    'ND1b3wa9tfOOPLzwa71fxg09OmuIu95JYTxxsDm85wtMPcHdbDwMLjO9cUYAvbX1H70na4i8Xg7g'
    'vF6sqLuaTd68pbEIvC+xszyHmhy9xMa4vAPfBrvKj0Y9AnXxu4cllbyBfxM9weu4PHOB6jwu7qu8'
    '3h2kvGxsvzx5Hvg8E2vBPEfvkjsGMje8u8xJve/lHDzOGG88e5TrPAJ/DDvB6no9SGkQvJfK/Two'
    'ble9580nvKmrNb1X1so6eMxvvedw3bx8PwU9hTTfOruOIj2FgKU8bUizPCUcFr2JI3o8FeSavRGi'
    'V7zm4Tu9lkXxPJSKaDzJyQ+9B4WRPGCtSj0Frf+8/CgYvXIdV70dNWa8tXFNPSBy4br1nuu7AlWT'
    'vL9yr7zGLuu8RyNIPd19sLw+nsq8+ZWSvAiCWz3kBv+8xHB7PXSRMzzQnN28HjM1PRNaZj1bdJy8'
    'R8baPKRiCb0b29i8FgLlPMfIWT0MFkS9ZoInvVb2Hb0txZO8sY+6vEfWMT1ip5A8HGDbvFFbhbxY'
    'dgi9Bb5uOntbbDvIpTm9krmKPflACD1bUGm91nnTvB2QNz3Qg4K7KjwgvVlb67zk+0m9pod7PGEe'
    'sLzdL1899a0nPDicJL28rCg9kruavHLd1rzfyh+9h53uvG473zwk8jQ8RdSlvItLw7whIGm8mJkV'
    'vQMUKz1z7WC9XhLKujgG9bsmSiG8CyQFPK2Xd72XZhW9UxHbvFCufD0qpXe7hz2ivN9g1LxGyiC9'
    'LuB7OibuyrzhZh+90y2ivF7NTj3bEjq9A/Lmu/CaYj3Czgs9axp+PIT5MTwmMIq8GcJQvTwWFL2X'
    '/w69gxXsPNFR4zu+Qze97Q1RvaYvNjx4JAI9zg6Su+ncXjuJizM7EogOPL/THj1VHCk8LyEcvOAQ'
    'MT3LfwU9gMwnPdr/sTxkxgw9xfcUvIc6DDyRqkO8orZgvEhS8ztlmCC9XyKQPQT3OL2UjRS9/MPF'
    'u0PzFj3WcQG975TTu+A0Gz0kSya9PK56PbhhET064hM9c+ZdvFmmmTzMoSe96WGEvCvGcjyh1gu9'
    '2KNkvG7xQT1Su0k88gngvD2LBDzFiP+7ENRFvbggPbtujag6KxsLPba7XLxBZ069nPIhPXzb5Ts3'
    'l1a836XyPHveSj3a2Fs93cVEPRrg0LxyeFQ9fiYZPQb8Qr3IE9E8eEH6vK+qqTxi4z69F0QXPZ1v'
    'Dr1AhJI8hqW3uqNfDT0upa25yPs8PJItt7ylIes8ACO0PLKxCznzNbW8IvNHO4KYYL1u+FU7FuY4'
    'PTLtBD3+sRI8K4m6vIoRZzvTyGA9dewRPaxtIbt63rq7wi+cOp7Wc7wkdW69lLZ1vIKDB72uyjO9'
    'QuJaPfqGNL29miE8ovxSvJf6IL0ubOS7F4TZPPaP7bw7/EQ8CI92vMNutLyVGgW9KySZO7NZk7wJ'
    'rSM9kaWNvYfw7Dxr6zw9hCahOwcrDr3rX3K9NZecPJcw1DwRYEM9gR5IvbnU4Dz29uS82FxyvUmQ'
    'QTwemLu81kaiPJAvorsi1Dc9dGlCvW7vTDy7+988sKHRvKDZVb3dkJA8mWo3vXldB71+EQi9ygNd'
    'vRpJFT1Pc1q9RV8xPUWTe71RnxE9P+GpvA/Oobyu+gi9dBOJOviTkryf61S9L4bFPNe4Dj0nDBW9'
    'KqZ5PdQASDZyEDI9tu0JveBSUb1RqfC829JfvVOtVj3qkiM98n32ujHqSr29hTU99/8uPHaxFD0K'
    '7dM6g+eTPD1HyzhbW1k9nS8HvN754zy1BGi9jzUpvPYJNjzfZVU92IISPZlWST3NK7Y7f8cHvDPH'
    'I7r+Y028kPjSPEO/tDs2CO68HIDEPCpS4TtXNFy9yM14vAddK72OG/a6PCasPFjQX70DXyA9pcnZ'
    'vE0ASLwK1KK8Z1lZPRuk1Ly9WoG9O4kjvTUW7DzzFNC8FjmKPD0QBz3rw7+73D9sPUvuFT2syEu9'
    'lPk0vVtjeb0gos+8pfNWvAvt4bz4RFG9GU3BPO2edjvU/Eq8Qk9qPN/in7y0qsm7qezqvKiBDj0S'
    'K8k8zbHHu1/Aqz2AcdE8KWWavOQaBD31Cqm7Iq8oO5sIoTsoCEc9ZN0dvTrE4LxxixK92rkrvV+U'
    'Yz2Qrju8Zj/GPOUAID3bcSo9dHUMPB6bWz2ZdKG8KQkbPffzhz3OBD+9sXe/PNfgrDyQWHk8yflI'
    'PZ//Xj28IHe8XWQiPLUhO7xTbIC9UkJmu4CSqDx31Ye89lTOPOn3eL06FSK9ynZIvVLQ+zpbDg89'
    'MjIxPXwxhj0HSnw9VJwCvYoZSD0UDmy87WB1vQ+TgD1BQDS9n6QhvXPwX72SDCQ9lsUovc6/mzxY'
    'qQa9pIVoPL75xzyATVu923o9vCrYMr17UAk9c05KvJzVybzwlqe8quPqPPLYVb0J1zc8xuE9PFUH'
    'H7ya6H081U7kujMrzT0kf2m7t/LCvNfN37y1y9E88ONEPVdI9jxcTju94l09PVsZwjwtI9O8rzsG'
    'PbPMzTwfcES8Wa7cvG5WEb0hlNc8py9vPVybX7tDK+U8+az3PDINjDxpKwy99h9gPXU6/bzQbVS9'
    '1AA3vdBGcrywBBs9nToavSLqOTxjCiW93bopPWMu9bwDLR299OdUvK6vbbv4VhO8EkQwvQnSOL3+'
    'CIE7+BBOPW68GbxabA09uKjPvEkrZLzh8Ug8kkmouGdfWL1aOCS9kqcevYdg6js0qV69EHlKvJPG'
    '/Lz3PoG8dZe4vDQeZ73nwUg9l877u1CzEr1jbFs9U0Q4Peu1fjvKG2O8MO5DPYrVjDspPPM65O1o'
    'vOTthTyM8PU86GJPPfm+7ry0CTU8bb6ZPGVUcTv8jdo84MnpPAUjbr1qYoQ8bSZvu+Ka2TxF24+9'
    '38ZJPRB+zzyDoRG8qUxzPQF1JD1RmDi9kfKLvIuyYL32Pgu6TjszvGjEfr0Tfbq8zcdbOreIRz3v'
    'wSQ9vVFMPdnUI7yc4FK92uj7POkTHzwpTSQ8nxpwPVo2LL0QzW48ZcEJvZ6tlbwtKuS7jTfgPKtj'
    'Iz3TTU09/hVEPaLmOTveASc9ro/8vIsss7xJJRi8EnA0vTzclrzK65a8YbwBvN1LCzx2c/o8QbYd'
    'uywuc731Hbu7tZ3zOo5fSzyND/k8quAsva6LH7xO2ue8p8jRvClXWb1ZKlI928B+vXft1TxJECM8'
    'YIEjPFprTL10WQk9/T1dOza4XrwTgGc9SGAhvTz4xjoXsZW92qMJPYK7CL1yWB+87wsyvb5EBT2J'
    'EfW7ci5aPJpEgD1MhN08on49Pfhm0TzcNoM7rguIuxTaNb01Wju9M4SWPIL+ZDwGV0E8gt9NPZus'
    'fr0NiBA9fhIGPaPVQ72TGdg8MMm2On1zNr2reZw8k9cGvUv0Gr3Yz8i7ChfDPOsQBj1by3O9V37j'
    'PGqLhr0FnQc8vCsbu9Q5VLvgR2e9utP1PLkMY70B/TW8/zmTvc9QNz2URlQ8r7lWPMHGLj07QWQ8'
    'zhozPfAbWL3QbC288wWuvBCu5Dw5Imi92keFvLB+0DsfL1s9XSHovCWoIr3WzR69dP0qvQx1crzg'
    'fgM9lPG2vCepEL2pFNu7Cg2RvcqxHzpVNKC8CGlyPWPfEz0kya28Sc3pPBY8lrwkPXW8gqVaPZ/q'
    'Rj2IyR899gBivUPlEjy7ZwY9kL3JPOveFr2qR4k8vVHSvNqDADx23ZG8Gu7vO3I427xEKx09ixI9'
    'vFX6QbzV7NY8HNnDPMVsnTy4LMc88of2vCJ5zLykEsY86ZhmvQQTezxlrAE9EKrJO4bh0Dxf20k9'
    'Z0aevHw6JL2LQyS9MbABPdptArwPtgO8eWM5Oya7E71dRCU9kCZ0vUb98TxvMRg9FJfLvDzDjbvp'
    'CvO6CK6ou9WsCT30LYg8GJpzvT1Acz0RjoE9pb/ivDM26jyK+dQ8gvnkvAuBNzygq927Cr6RPXFl'
    '8rzW3Yu8eiOTPdkKbryDXEi9QSKgPbucFTz/Iy09OPHZPGMVw7zimuw8HRwqPQkUdLxhlQw9NeZP'
    'vc+AQzxSB1w97kdSvJaTZj1xEQm65Ay0vE20EbxVQCA9xg4QvAvcHL2adwk9TvhxPEqd+bwOKRY9'
    'GAkMO2u3cbzdxUC7bE0gPVXXf72bJEe8ZFbWvDkLo7xphfG7qq0wvb7paztvOAA8/pU6O6tcfL03'
    'B/o8oPe6vAZCOz3OFU89uEFqPS11QT0grTU9DkQNPD7NOTw/ztG7oxbaPKP4ar3bxTS81DJRPEkL'
    'LbzUoE+9CklnPANaZrxNae28AeUNvUc2sbxJ1lY9xD6aPI0hQDy9Vwy8nP6XuweuD72Mg209KlE+'
    've8AwDzRkDq9GywYvQ/nfb0SKc68+mxlvd8bEb01DFu8DMvEO8uJSbz5vUA9/8uwvO8JUb1KEYm9'
    'u3EqvQ1cOjz6wzg93mPuPJQPJ7wowTw98WnqvBE03TzTHAC96NDZO3s5jrzcADM9afXQvEluzDz3'
    '9wS8Z+4WvQzJvbyK/GK9yAIqPdMlfDyfJxm9E3Ikvd+J6jxrQQE9vTMWPZK5Dj2gj8i8Ne1aPfUG'
    'M712UYm83Tu2PFYAjjwcVs68CrYBvTW5izyWKK884mVQPBzqdjw4xMg8sBsuvXaRLb2xyC69jN82'
    'PdpoPrwkvU881bkDPeYX27wK5ay4Ong8PY6RxDywy0g9SzEqPXjeO72g32+9tA0NO/MLTL0r/rY8'
    'J+5NPaXoEz1hnU89ggslvWO4OD04kmO9j8IuvVq6HrupzhI9I2devZe2Or1yb0q9bMdiPIsLHj3u'
    '8Fs9ujZXPWqydz3TfzI9Cm5LvdRJbL2YQX885OEivY09lDxyoEY90MaHO3O/Ibz1v628i+jMvNhQ'
    'KT0c3TO7JV1XPWy+PDwkeKq7FYCJPVcsLrwAgb887HEyPbW2W729sOm8BrxmPastdrsMUaC8sJUl'
    'veNjS7ujFlG9ZcJuvPXtKj10MoY914tyPTjXXrx2xVS9FOoLPVUijbyOkz69r4EdPFMOorxNUjQ9'
    'gvVFOzHVhTyKkYI8mqagO7MOd70i8Vk7pmECPSuAKL0meWA9nL2MPEqDPTxv93o89IFovIlt1zyY'
    '/bM8/bnfPAddq7wauMU4WAPIPIS/STrCXAO9G76AvL8ysjwoWAe9QOlcPW979DubWzE9+cUoPIhO'
    'gDxGyxC8sYHTvISBdrxxRya9tpt8vZM147zrDOA8d48LPUgZGD0665a8DMKFO/i5gjxvhkQ94AOI'
    'Pekcc70o56m8lZJZu+AMNrxZmhw9uIVavYvHCr0DaQm9E9nuPCN1Qr2wUAE8Ni0+vPDUQDxFzfG7'
    'Txf+uzQDLT10XkM7GPpJPMJcQL0VSes7I160vOawirxpAaS8VHhOvE7SAD2qyxa8dAMwvdzhJ70y'
    '6mU7mMtuPB79Rb0zUAM8o2m5PM34gT3Rw7K780RfPYXMtLw8gnk950o/vPbD3Lwm3iO9wIC1vJtp'
    'PbqaZfg8VQiKPSjJ8jzrBYq8SjSJvITaHj0I1l69t102vBUHVLzFXHG9D/FAvOSGXbxMwge9cQpA'
    'vdk2c7zaF7a8GnTovDW+Pb20kae8N92CueXVAr1P/Ti9Lt1bvTGaZL0kIp08a+A8vLanlDwmabi8'
    'j34UPKD/ZzyKUuM7RhSKPbn/Jz3qo2g7QOxhPTtx4DvNtkK9H6G4PFIXTjxYRek8kCX6PGE3MDyA'
    'X/+80P0WPL1YQTy+9o88mYYMPbP1vrxS4lw8aTwxveWz4LxekkG7wBkcPS8xO7yg2z68LXOSPXOK'
    'abzfWTg8Ej5mPBfLSz0FTUw7C0yAvAqXhTwVTUC8YnD2PE3d2zxaTLC86aH8vNASyrw4wUU9RvNw'
    'vVxdvTqElds7ATNKPepOWb2/Uwc9/kMcPN88Ab17SF69LCYqPC5DQD0a5Tk9qSSTPEW/GD3yzuY8'
    'imH9O5E3sLxAIuM8X1RqPTZ4Or12Dqq8L4xovNJsvzv4ozs9LQW3uel92jxpigW9TRlwPXFIhrxN'
    'XSa9SE1gvY797LzmsQ29GMoiPeCKsjwh5y+91Kc+PakXeb1LBLg7JNnnvJr3tTzYH+28oavxOeQ1'
    'STy30jy8m0rBvACdbjsusmi6JVwhvYOOZr0BeAe9ZXoOPZ0LAj2+sYI9VWquvJne1rxKdAu94l6W'
    'vGB6NLzSCXW9Emcxuv8wwrxMrqs8gkjhO3ZesTxfRqq8QP4vvTNnLT2RPSW94MuDPZPILryhLzc9'
    'LNjnO+xWHL0bCZI95t0XPCC6Ej1QZMe8om7QPPb/Nz2u5p08FheFvT9F1ryMht27TmVTvfYrhLzs'
    'lBC7aGi2ufanSj2lDrm8lMtSvYcXZj0Zv+48tMEePAxuIjviiAa9fwa1uwIxRz2iVUG9MDcoPUYZ'
    'Vb2sx6A8yycsPYwFODzw3Dy734UdPUoBiTqjeDM8vpFvPRYKkzy/FxW9nLQNvUEJbTxdnWi9XqdU'
    'PVFCR720wT29uTlXPXQpdby2kiC97u5fPWhqk7xCSvk725b4u2E3vLt1oe88iClQPVj2LL2DnO47'
    'lUEVPauWqzy9VGm9s7xPPf5ZczwIdRC9vaADPft8Crz2PkG9XI0RPII0Hr1OvOW8Q68AvbkDvrxo'
    '//Q7ERQTvDCEODzzXco8g3YoPUg73LycyGw9Z+zQvF1sWb3IxVq9WhKWvKO5xrtzI309oKVIPRDr'
    '8zyBQ+o4jmcDPN6yN7zur3S96qyNPFpKOL3+dpS68V6Du5H9eD0rIUS9RT8DuqlzXT0um1i9CceJ'
    'PAIBWL2YG5C8QWf2vK+OOjwojQe9b5phPZHhXjwrCrC8ll8pvWkl17vWLf48rfkWvbM0x7xiJS+9'
    'Bet7vVG0NT3qbpa8Myg0PXb98DxGqns82O9KvURZmjwquWc95ssAPSImVbwIWFM9xVeMPKNSO73+'
    '0Jc6qB/hvK0UozxNqoA8iXBnPVVJgjorVUy7gL0JPeiFNr29jUs9bkX+PB8bkrtVLKy8/EipPCQ9'
    'LDy/yqW8/JJwOnKhOT3DNR895jarPXHnhb1DWmC83hYovd0zQL1jaOs85AdLPQtiw7yX75W8ss8w'
    'PXHrXT3l8bA8QK4KvWJFMbximS69wvhzvX5QBz02MuC78bL+vLZSzzwGlS49zHhYOwOwZDyZQEW7'
    'ujYOPUoGirnXAFa9wLJjPRADIj1YWyY8pzVHPeiaOD1tVl88kFROvEWlZj0geOg8bbw7PQHr9TlN'
    'zvg8SlxMPPA4/7zWh/M88KDWNw6CBDy1wWM97+DsvPnO8rxLVIw9tyyNPL92Cz3Xz2w8RSauvFvq'
    '1DpGwCm8GOGDPeffEzyUph+8hJAovWEdrTwNFnO8Ux8lPByndD3aqRo9lrMkO36mfT0q5jA9PFCh'
    'u7OumLyu1Hi95L48PZPzT7270tQ8D/INvRSZRj2Ru2S8qw77vJXa7rz/+Gu950eTvK1YNL1poDe8'
    'mnqIuxyGpzx45QS9wtA3vawWxLt7xbM8YKMkvWgXF707bR099HtBvONuAD3H9ZC9ZReNvC4mCb1U'
    'OOg8iCLUPCe2/zybg1w8qMgVvG6fKz0PaEc9sZq8vEZUVL14WHq8hlpsPYUENz0rBny8BJ0OvHwG'
    'zTgoS2O8RK+QvJmOJz07jxs8tl7VvJxs9zy8SZE7UXCevUkGTb012hW94XP3vGsqbbrvUhg9/dXP'
    'PD88Mj0L+7k8dNSOuxrMB70cG++60VM9Pfhb87zoiiS98Rm9vD3ZSj1fhIQ8idT1vHpTSzzoqyK8'
    'XIjtPPv6XTsX6ze9FRJTPZaXnbzS7Pw8LV68O2Jb7LvRxwe9CN0uPczIDT3KCxm9hEeGPRRwujzG'
    'WZu8izpBvNMgfr30ABe9MOoPu24KZb3U0Y291nsGPYf9Er3UEVc9nhKHvOy4f72Zfhy9diSNvMN5'
    'eL0yOgo9rZNdveXUI7wDvCU8QmTyOuXNWL0UpaW87hKEvcLs9DwjZx+9qqxdvFy5L72ZrjE9xEFn'
    'PONKL71+TEW4vzJzvcpZoLsG2QK94FtRvL/gKr236fc86RYKPHmjAz2CUrs8iX63PHXpZb082QS9'
    '23dNvYv//LoAdEc8MlA0PSRiEr1mzFI8OqojPfENxzyuJus8q0nDu8KCUL06NkA8rz/JPHfXCjyO'
    'bsu8Xj3OvGmCj7xX8cY8bS5GPTsQcr2QVpa6Rn3YPKgpsjx0ky09WPHqPJSX37ykv0g9J7SwPCdQ'
    'Ej37Tty8WzERO/9jRb0QDOg8NgbSO4hU6DxZN0y8ZoQUPTVFHT09hVw92gsJvai67zyst2U831sM'
    'PWkgCrwCFYS9S0QwPbs7iTxoiYK8d6EkuX2SRr21lbI84muOPerkDT3EWyu7VJA/vY79HL2jIzS9'
    'cQ3mvITZVD0ILAq9UNKRPG4YwbwOnDy8yv9tveeAEDxp6xQ9AWOAvVhSC70576m82MQCPRNi/Txx'
    'ibq8QYUePeTO6jz+fx69SaMEvf9DwzxaePA8xAo3PQNFkDxTLUY8paJhvWGRUzyWcXG8mTfAu/wo'
    'cjxFZWi8OGUevKNpQD2/pwE9DmadPHO3+zxDewO8FYTTO0XLG70/eKS7WGKBvfgpOj1PWXW8WQ/L'
    'PO4P2LxjHGC6eJEaPGEMIDyRdEi9vwwYvVLoij3VWpE7AfbYvFRcJz2oiS89pqG1PGO9v7yJKNI7'
    'nxh1vbP0Fj0aa/S8tT8wvd3PCLxhhCS9ZmIIvbnqFb3dJga8HO7ku9ECAT1T1zW9VI9sPaiU6jxd'
    'vju8+6o6Pe2F9TzxeSK9UkLgPN+9Lz14glQ7GzE6PctwWDx1lgw9qlEdvWqMZL3TYKY7wziwOK6O'
    'VL0qgEi9KwMsPXPuLb32Xww9/6p0Pdm1vLzK0Q+9qLnAvGB/OT1Ofhs9sr9MvRKjQr2B4h09Lz8z'
    'vSCEA7qm7yi877laPebHR70O8Ui8VvIhPSZHQT0R0Cu9Sm5DvQwvBj3DbpG8hgIqPeClgTxdHhs9'
    'BBD1vMNQdruR4iK9i/knvUTxrbzJJGg9/XgBPV49xTyZLh49wWJovW8jpjwkIUE9hRObvN4777z2'
    'uLm8RdkUPWE9C70JrSi9zlPguzoXDr3lpKK8MZz2PP/7Az1Pzq66MzJAvQH5HT1ZY7S7PcOFvC52'
    'ZL2hiQU7NlKYPBsINb3qpRW9ZUABPSHnFTyo/FO6S9pDPGpKBL0jYog8674+PINDtDyJ+9c8YYNt'
    'PONqjzzjydW80sBZPdIFozxyAwa8KgGvu48Kbb27Y7G8qFhkvACi6DwPcnc9KLusvM/sHT396pU6'
    'NH/1vALl3bxcxpq7peeYPEl7jLwKVq086zqMPbo+GTyfkAo9oy96PDzDMLxWBCA90cA0PW0xbL3a'
    'fvO8SP5KPcjNnTy1gDA9c+epPAgEZD2NfAW88DdKPEFKHL0GgjY8NzICPbqxmLvGGoc8gEFAvTw2'
    'LD1x6li91NCQuxIGEz0epbq8m/K7vB0QrbzRY+e87xfavAb/Yj113/k83Z9fPUrtbb1zcQ88Tp5+'
    'uuMK/LyVnxm9dBxnPdKtQ73tQvK8Lq8WvY4d4Tyesjo9ibHiPEiYZD1fkQ88RTIGPASZQT2k5yE9'
    'mZtfPft/Lj1YEQE9rmskPVV0zLyI5yC97bMkPOtX6byEsQ+9y8/jPFPeKz1QA0i9S2BFvWG75zwO'
    'ZU69fOKGvWgDzzx2MmC9K0ELPb81Br2YPO877t1WvVq2dz3Xf5y7TnaJPcmhtbvp2pU8rXClPPeD'
    '0LzAc4+8+MaOvBGjHj1IhoO8WLozPfePQD03KQ48Z1xRvQ+a67xw2F29CtRevQ0ISrw8A1I9ra8d'
    'PTeqFjzdxtC8P4DevF9F4LxnzYI8nOUOOy3nL736htk8fykqvQl6GbukBZm8Ep1xPdMAcb2Xlig9'
    'QJM9vGOeOb2tZGa8GJTgu2g1NLsIP0g8GPM2PaAW7DtXZxY9J/FHvZS2CL0SjE+9huG+PNnYXT14'
    'IgW98mf9vLGVors1rDG9ZVtDvXzTMj3PLqu89HA3vXBqJjvTbV69uqRFPX6etzuK8yW9yaVoPJ4j'
    '4jzniae8hL6NPPwwfr1n2VU9mb5wPYc9y7wkwd48DJtfOxz7wrziWGU9cn17u3Us2bwAByM89TAA'
    'vbAbHL0GJkW7w6mku/9xjr3HjAk67D+rvJ3EGz3hSm+8q3lfvPNojL2bjTe9M1yxPE4EjTzEec68'
    'WNI6PMDXXrwuzDO9C0yEPKK7u7wEMhI9uceIu3ofiTxoWKq7adcOPL3rjj1muwu9H8STPWqynzwD'
    'Eoa9M+kZvZceETobDy69T1OWPVIUJr3eGlE9qUMtPQTuJ70Bp5C8nFAKPamUE71FvjC941EYvc+F'
    'dbw+vRM9RIBTPGXtVjx2ptA8ClCAOvC4bTwjnT89XW8nvYd6qzx0yni6/+Elu+MvR73rbgE9Un0T'
    'PVJ83Tx8UwO9qX8JPO1dAz1g5Kw7V1asvd0ntDwLsx09/3fyu620AT12Ygo9ryhJvXL7lLyzlS49'
    'IjXLu3nVgr2mFnC9cDEaPYvftDpQCtk8eoEwvZ9h/Lsh11e9b/7hPCNjK73hiNG86TwavZ36KL0a'
    'z0W8UNqeO23yLT3zcJg9RcAovMOtljwKshC9aGyTvZTGSLxvXts8MYKxPOEaeLuJdSC9BhTOvHyp'
    'hzzXtWA96f8WPIJCKb2nKxA9R1lGPBeB4zztTwy9C0iIPHFHVTx5dAm9HagFPAsJIzyLs3U9YVhs'
    'vSvWJj39R5W7iAAPubgCizw0Xmm9T7wlve02Vb3MElA9GME9PUojv7xrlKk88zAiPSjMhr0K2dM7'
    'toYLORoBMr1Izd28/72jPGqwYjz9JKi8C08KPen/JjwQQB+94xEjvVgs/DwfWqC6dGvrPCwXlDyP'
    'tnW86uJ8PFmM6rwJXIU7uFi6O7bKzryyDGw91QxEvHeQU706xEs8O4pAvVEp/Lwbrbg7+KGgPF0y'
    'hz0EvR88Kac8PT/PLT3eG728QKYsvIcYn73tLMy8sk3pPAqdN7vkaDy89uC4vC06Cz2rTvI8x6e/'
    'vHuU+TwLXVK9oAviPKfO1zytfzm8ywhSPR25Xb3lWko8cncevGCECL3EohW98BtDve0fYjy5Q4A8'
    'QIEhuy10ybxt45q85HLGO3ARMD24mba7chi/vMKb3rtsZ1a9LA1jPPo2mLy/ZsA8hEEPuwfXcD3B'
    'Abq8uWJGvUZIfzwtqCc92KsJvbWQMr1LuSg9ZKY1PQcIKz1ruAc9Dl61PBYkPT2jpmI7RI5UvTq9'
    'Fr3ypLU83qDdu12lYz3my1k96VwmPe9IvryKS1C8gihOPRzfw7q/B0Q9NJvjPFvqgr2EWXc82zms'
    'PFgAHrymKDU9cscnvUi5Mj2e0h899RELPCY8KT3Nbxa8uE/uvNgki7yZwVM9rxspOJbLVzye+ye8'
    'c0r+PLR5szt6jki8RVgMvUJZUD1TUE09s56TPBJSSLyJOWI9GxVPPcosE73gEg69jJ6Hvb/SGL1N'
    '7F29mxICvWE0zLrma+q88Us/Pbn4gj3iZ009pfmPPA0RUT18vni9OQ6IPBIwQj24+k48PE5gPNe8'
    'JT1ABpc7NXvGvPPahT2bU2W977r5uruMyDuR2sS7G8gKvRrqI73y/E49S0MDPIVDujxNyCk9wx22'
    'PLyJVjvwRyM7G4POvGKyPLwBfDa9Mc8FPfwvoj01G7Q9P3ikvBZYzjw60RG9MqxdvUSFXT14dHo8'
    'ImkovavtEz0rkJs9tkfCOx1TWDsWN1882BnoPC2nFbu+ds47nT3XvN6fET3arEi9QiOOPDiQkbyJ'
    'rBe9uHfdPGNIkbyukcG8hG3svH7gXzuDra28XJUUvfSzXb2fBz28JQpcvPlu8jsw8lQ8xPy/uZiA'
    'GTt73QM8ETILPSsNorxQZtc8ZVYSuwzNCT1NS+E72idSPcZ9Cb2VG5Y8MbLBPCcDqzxqMLU8N/hE'
    'O41AOb28/ji9v4cXPBOyDD3Y4xW9Mfs0PAG0XrxaT648G5tMPIaVTDu5hfc8f2r/PPqmVjvkRhY9'
    'askWvR4BkDuSODe9typCvZNRYb16mK25CLakPItUTLxPrdq8vesnvfn5G725exK9pOz2vDP0CT2h'
    'hIY8VHGPPMwT/rzo4oO9IIbzvLCXh7wlZDO9BOPku+/8LL0QXd08CsNGvHhUND0KVEK9vcRCPNmV'
    'Q71sjYS878fovFRZY726fWO9oWkfve0RVbzjYIm8JvGLvWR13Tsd2ws8N7+SvALnOT2qy6C8ynsf'
    'PajJSbyy+ze8aVVxvEN6oLyZzG685dRhPFA7I70OgYY8kNBPvRBn4bu81IS8qqnlvBsvID0ZoF29'
    'N/JiPK4JR70qICE9GJI2PcU7gbzm9FW8uzTLO1P79zzA9DE8yszgu6CkDj0gzRU9DtdRu3nIUr2E'
    'f1Q9mq+yvD0BC72hnC69d/ksO+MombwijBc9b0YfPRxLgbyiH6+8R0cxPQqql7xqniE9u6QXO82i'
    'Jj3598k8pKE2vQJhLL0E0hA9FgfovD2b/DsfYuU6RB1CvcLeMTwxnDE9/RKkuYNwGb3z0nA9VolT'
    'PcqUhzyDp8y8ppXMPEk12DvPWzm9bPEcvVBES7zmckg9urV3PI6NcLyj9v48giMhvWPvArwDaH87'
    'OW5APazQWTorf0A854aZvQpEUb1Y6jk8vYT9ulXXIz3t/mg8a2VBPQuj3bvCbju8/cOCu0ik3rzF'
    'GzY78x5Nva/WRTxJUcy6y+aYPNgPiT0JFc486hRjPMWrI70fU4W8WwXvvAP/QL15kBA9LIZsPYXm'
    'MD16Ejc99FULvfBUwbziBK68hdL9vJ+FNL1A0ja9D/tyu+SzDr1rSS+93qySPdIfeb3team8pHkG'
    'PRCvwLzAxLk87IE7vfi9N7uEpcE8+R+qvBBjSb11Zfy7WIIlvOPqSD2Q+Wo7Z1ZePUbcJL1MyxS8'
    '3qibPCcF3TzvTVQ9dUOlPP6Tl7wK+uI7OMn+PPuPQj37bzc9PQ8DPLiXwjsW5RY9ho1PPFstK72Y'
    '/V89yPYZu9/0XD3qCA09/inju5NNar241ZK94yYtvbbnTjzkrci7AFLxOj3MR73o4B49eeeuvCq0'
    '4TyVonu9cCUvvCWoAT3rxoI8yTx8veJWLD3ddBy9XVVmve6qUL3DAkK9C0CsPHnFHT1xZpg9q/wn'
    'PNHI27uUjaE8asD9PC49qr1w6iY8t2iUPNySOT36hvE8VgTMvCTPMjwu5CI9BsxAPSCKKbwZTno9'
    '5elBvE1hJr3oJ6c7XgJKPZlc8bwA8gC6j/WxvOXHzrpzzlM9/Y3fvPqB4rwPlqk8WHzUPJoLiboN'
    'Ci+8OzKQO/QizruIqyw9Inq9O7vlKT0qnCI8DV91PELYErx6Twu9ryJAvD3bhTxhtxA8lZPXvCqw'
    '/bw8Ruk8kYC7vBnuKD0EV2o8c3kgOwPMwzo3UX+8m1fFPLpWJb0qocS8pdmVvOXyg7tUKzC8gTCp'
    'Of3/ErwIqxk9raICPW+hOD26U6S7rQo0PXMc1Lyo3cQ6RMgwPAKwhb0FVR46j87gPG7FHD1pvkk9'
    'PSUePTO1Wb0N0g098SzAvOWXmTyvg0a9w64xvPFew7zYxfm8sDpLvX+EfjsRFBm9gsMBvQ7/g72o'
    '0gQ9ABDbPAmwy7veRgE9mUT5O98UDr12yCW8xSCsPHp1mzxT7LQ70JKGvIZD9DykyzO94jwRPMGH'
    'rrykPNI8Mi9BvCq7E70giDi9zs5mO0ct/LpMFV49FlHKvEs2Ur38oFu9HPWRuzRUiD1onX29VSFm'
    'uzoUVD07+G496foBvTKNeL3h04Q901rhvLL0uDxzAWK9UT76PPgmIjxo7AM6ahelvEbOnjyverC8'
    '9sLTvBjcjrznGzw9WMijuwt/5LtPwUg8k8JhPYdvjL0+7gM9iY2mPNK1Az3Bv728uyA1vaOpuTrh'
    'IjI93cpCva8LQb2A2eu8iQ10PNI8aL26e8454dBaO1pJ87yPV0i9r/OZvKQwd7wRMJI8nX9KvQDW'
    'vjz/Mpu8P34ZPYFWs7wO0+o8PIGOu2pG0DzQ7UO9614HOxpTBb16EM880KxNveBIAz00VDO9HZ8A'
    'PfVvt7x51lC9OpRuu3P1Sr1Hxg89OdAOPUj6sDtKkhu9bascPSIwtztPuRg87BEGPdLwNL1DIoy8'
    'HE2sPGM4S7wGzaC8znBkvQIw/zzq5J28f4AAOx/VWjw/DT87vCs4vSKufL1JO5M7UWxjPPrIsTwc'
    'GeU8fDlrPeKEkzy82wM9ngJbPfID6rxQAgc9hBCivM89pjyjSS49QX9MvWOu0TybhpE8NF+DvBp6'
    'rrxbUN+8jyMWPDijzbsqyzY8VITGvDwDxDsYjxY9CvljvVqonjsxQjE9Q1a9PF4PlzxcQQ89zT2H'
    'vQIiKr3X4U69o7xxPbAtEz26DTM9EHdNPUJ6vrzZDjw92it6OncfejyYAiO9SAfYvJDnybzd4lY8'
    '5sjePAwA/Lw9dwa9PoMDPHj8Y70ZPh68cgloOz2ucrz8ysU7Y2AhPVnFUL0L9Zy8916GOnAbhTyQ'
    'WwQ8VetHvVB/ALyfhEG9Wvc0vfTqT7xtGDO9AslDPba3nLveaom6foYXPY+SKT376dU6pqlQO7Uy'
    'AL3HPsy7AhtXPQL4/Dzxbs28Ki4wPR7VgDy3BUC9jmDuvCJQsTs6BjO9TlsNvYvn2Lw7VpW9/ZIP'
    'vSoKpLrSED48rIDqPEbfMj3uOWg939kXve6/CD0QZz29+8rEPLX/yrxsHY09OC1kPcLSFL1cI+u8'
    'MSdZPbPd/zy8flE99hY2vXawx7zvTgi9rymHvClMszuE21E90c4SvbH+Tr10wQa81zDJPAYOBjzG'
    'MU28Iwh/ve2sa70XNHE9Tog2vHNZDjyBAS89lIjWvBcuPr0oAyq9P52qvE82BD2bOuc88gCZvFAo'
    'DjwA5uA7dWUbPbTtAL0V0FI9h7EiOwUugDwuW1a984C6vIroarwahk+8l/R+vCSyubxkds88nfQ7'
    'Pdz2Kb3lVNi7gM+vvDQvn7ymoqG8gcLcPCd+FzudiU29LpWwO5Tfsj3IM7G7NqdOPZR++7yvyHM9'
    'QIqrvOHLOT2Wixg7nOWiPLm/fb1VHv46CChJu7o2z7t3+CY9kqqRvNqpBL1cxh698MS3PJH1Hz1S'
    'GjE86HoVvbkgPj1vTuo8cTgQPcTfMTzwVFE994FMPSPA1LxmBTU9X3ClvH6QY71bMdY9dm9Ovbj9'
    'lzxeD5g9PKU4vTiDRT1pg5u6TK7fOyEjLjwsdRK9noCivLdTWL3BCS49gG0mvfJFGT0V5zo8i1+d'
    'vNa7SD3UiTQ92fbhPPNsDbwFfmk8+SM3vWa5NTtvR4E7Z+QyvZj8b7zExHc7+z0FPZe8yjvPFqG8'
    '9JiuvKXQZb2XZNW83DwHver7Tr2i36A8sYagvEvfwbvYTC098hgIvaxT2Txat0W9aGoovaWzQT0e'
    'f6e8czVtPTbKQj24RIA8v7iQvD2djDsH2588VmWaPAYWcjwa/2A9jmM3PBOsBj3R+5y8h57evDew'
    'aj0dYRa67eLWPBM5EL2TRUO8rJuAvdJJszvmR+66PjxpvFijnjzGrac8J6AcPfgYWr0h7Qu60MoY'
    'PQD9Hz3drju90G2YuqFtrDpOly+8W2QCPckZej0Nhy+9dK4OPR3kGLxSQ+s8UZn4PBoXkTxMOYS8'
    'yGZxPXA38LzyXyw9YOOmPSMNJ7wbUyw8x//KvPArsbxxLxe9vUgPPVyQEz1zOFA5OX7AOxunQb3q'
    'KJg8vyb9PNlZILzG8Ri9wfHePBu4I70UkTo9aSwNvb246zwVTRe9SPY3PHmeej0UuhY9MmEQvc3Q'
    'vLrynZo8FreSu+lirztgQz49cev9vEnJAr0yql69Wdfvu+itwrxJJGU71a4SvZBfA7394gW97k/o'
    'PM7DdT2Tvk+93zooPa/hCj2DHiq9Xg0rveXNDj2eRpu8+B/SPG2CGzuuale9yarnPOFMXTyRov68'
    'i0CfPEqNRLwAs9W8ma/2uxShSr3tHXO9KvoOvFUE1LwLBqM86AWDPaRiQruSTzu9H7NUvakkwjzo'
    'd7w721N1vc+SUr3IcUY8mGxfN6xb9Tz+AVM9rDrsvGPIsrz4ame9lZRFPUT4SLzdrR28jUN3OuLl'
    'y7yBree6IalXPUS54zuAjHw8mO3EvMSxOj3p0WM9LmUmvee5IL1naUw9TagqPXqaMju0US89Yj0J'
    'u7wJ5Dy0r1Q9qSBzPdZkIb09GAs9iTJGPHaEa7zaXyy9vUurPL9WPL1y6vw83mchPQY0H7xs7Ry9'
    'ZlSKPU0d6jxTX1e9hZKJvQLJ0jyoafY8S884PSwvJz10Dig99uGYvFX0Ibxr7D69tduius01yDzS'
    'Ufo8SeK3PGHyUTwBzQo7AV8nvQdYBb3Lxca71Iy+POCWEj2JLbS80bZLvN0xyLu5Fx49G/S7PPFR'
    'ZL1+Dby8ClBKu55XQbwxtly7e5wVvZpcT7t+Nsw8GjN9PRKq0bxh87m8mA+NuzgwwzvlXbc8Dvx0'
    'vC1bXz2EzAw9WbqBO+vLVjvvGzW9LPjoPM+jK73h9OU8vpvkvDbZOL3Eu8+8h1rhPFQCNL04Zy69'
    'QJ+cvD0cMz3Akwc9AQwGvP6ynLxsUOu8gI4uPPMUtTz+VQa9JBYaPQtmS73MiSu9O3MkvfXvpjzq'
    'pRQ7V9+1PA1JOr3B2g+9ExzuPGFQPT0u6jS9Hk43PMDkjLx/oJW892+TPOEZa70mU8Q8rj+kvMnW'
    'KD3EXwm8hDxGPc+4zryEy0K9hxhXPXFAAjyq1Ya89HcLPdQPWT0DQGE89CpmvU3mlbwRPkI9rKaX'
    'PKXQSj0HTBc7textvYabvLtilrA8YuXkPHvkSb2wZ9y8CmgyPYbgCz2Gf2Q9r70TPWrKELsi0yG8'
    'w8dMvThvjbwoF2+96j8qPRQ7Gz2GaAM9Ok6Du7lsWL0bYBU95etaPHU/ADz7Ygy9A48uvSS15DtH'
    'bXG7WJ1HPXNXdDxXD4G7H+/cPAkJLr2NOd08+AmMvDJojL0LJ6Q81xxVvQBAVD1TGjs92TqiO0ab'
    '37uWtX08mHQxPRUzEz3G4hI9zKVuOhTzX73A0eU8PqYoO7XSJzxrK0w9m8yavG6VMD0UBPG8i3En'
    've6vDj06PtA8aRXGvHS1jruS+y27wMxHPF/ANr3sdre9G2YoPKSQv7weLBm9U09cvYgwIj21U9U8'
    'i4Ukvdserz3RRF+8jeyCPYut0zw1/L28DmoIPaYNBT3tSiM9h9lqvS9ct7zMGy691R8/PbaJ1Dyf'
    'bAq9RVc/vA4HSj2rdUU9ThI3vNxiYb1oDNq8+8D9POkdmzxTDSU9MLVcPTUnS7xBzvE8+KsuvHAg'
    'OLzb3hu93dUoPLVL9zvQ5MU84h04Pcf9kL2aRAO8pFDtvKzAnjyhCMg8IloqPXqWLT0DDiE9OsY7'
    'vR/cB72vTkO9fs7ZvONv6jtWAUg9sHt/PGxxNz12zTc93VOxPPWC0jwpI+s8S3QiPHbPL731czQ9'
    'IIEuPQAdTj0v//C8fLmnurNASz3DOO87DadKu03RkryvAR09QxCTPD7mF7yE8yQ9RXgTvSMx+TzG'
    'VsG6QtW/vG24QjxH73u9fe8dPf2UYLzI7BC9pqrkOw+T+zw/zks9hxkHPalsubzGG0k8EW5APbD8'
    'DD2+Q5A9qZkTO6fdib3wsQ48k5COvLe9kLpl5LA8bdYePf906byjGxC9sE5lPeZ/hDxUsHI8IYU2'
    'PPQ/jbn8/VO9bX0sPcZuTj17usO88RNQvBhVbTyzOoG9sgDmO51ovzzckOO86WBovM8hIzwd1za7'
    '2XGAvGa80jxB+jw8T/4lPCS7QT3f+i09zqxuPVUMOj31poa6eMYcvS09WD3qZCE9crkxvQk1iz2n'
    '/yk9O/yyvEuSiTlS4Ya9pZMhu9q9/jzqnSC9zfy6vIMLaj1UFkQ9Qz4WPbf/vjzFeJU8JdUnu9Yp'
    'pT1byhE7bFntvCEr2Tw3dlg76fODPO0hnLyge4e8DbJzvTOsz7yaWv48qR2uPJDMdj0qUT49pRsG'
    'PV+yZ73Ty1Q9KDCSPOWTgLxFVRa9n7YEvR72sbzH8Lc87bvbvBrjKz16rEM8KnDguwKaGz3r5uU8'
    'XhQ3vFthDr3l7EK9FARRPZRMij2v2gK6MWeCPVzv1TxlsgG99JvXu6SlWr3/6Vg8bWXvPFPI7Tx1'
    '5Qa8BUFkOypiNb3QakG9ZoTavOh2vTyzbnq8T/SPvCk9Q72AU668jBEkvZyKM7ypP3s9N4MsvGed'
    'hblAEhO9j1TtvK9Elr3txkm9V9FUvKg+xLxaKSC9nbqsuhlcDj2PiDK97QwmPYplVr34HII7ufsq'
    'PJpeTb1ObRG9EOkYvCdtHb2BB4Y8KFpfvZJ7Nz2SUne8RW1zvWuKnLzNuBM9mk+RO29iHD1EkBY9'
    'tEOzvDzrQj3LM9+85A5HvUEVJz1VYSa9bjcUvZvtIb0Vgdg8ovLdO1sSnD0YHjK8P45WutLr9zxw'
    'yBw8i1ZZvKRoRb3dcfk8XvA9vKzesrw8H+G86ZUTPThkPr0K9Q09ONEiPYKFPbxCTv883p0zPEQI'
    'ijx9Bue8vIMSvJh9T7z3aKE8cNh9vY41tLt6oto826/ePDjKyLyQvVm8bZ1WPfmo9jvWoWs9ZGGZ'
    'u1el8Dx48am8D6gIO63FLDumuGq9P2iEPPMyOjy0t3G9Mz7huxtmuzz2UmM9DEw+PdAICr2jOOy8'
    'kjvJvIOwQ73Z3II8P/JsPdtFYD1BwRG9u6wkvQH3A70qZD48PXP5vNZjYb1qNzG9uuK2vJr8lrwc'
    '/ry6yiKEPbEPir122Yc84+ceveKQhD3Aa0O97y5evcdwe7u5sZM8JMlSvWgiYbxfq0q9vkn8PEJa'
    'xbxoYr289sRZvYURPjz1D2I9Q8+2PN2BkzwrTxQ9Ju+8u4+vWrsAUis9lerdvNpDbL1IdRK74GO3'
    'PEzckTxDhjw9x+FOPbzjAjxOTRY9dsRcvX7OhjxBXzw9HrhZPRZIsL3jInA9ZasyPESMXDxllGY7'
    'gU6YO2K/Br0NP0w9oLh/PN0/Ob3ZL1a9YSIIPUidDj3uhGw8ZBAtPffafDwOaWm7LGQRPZ8Bnbsi'
    'wOG7UfimvEFKNj3sJKs8i5WxPB5y6boxjHq8rm8wvT+lDrurl2i9Ib6xvEIOo7zfmTc9g7yHvapj'
    'KD3rChq9gCOLvZ99br0wTw29Cjh3PcM7fT1W3hw9BpgavZXDHL01piG7KME3PU42prs6SFi9eSnR'
    'vCJVyzuhzGQ9P6YmvV6LPT1iehc9FDRBvVIZIL1KKjO93JiXPCXkY70Nu4Y9Z2iIPdl2Hr10xao8'
    'E4ibPWTNEr0MRfU7dHrpO549wTzF1QE90JhpPfYOpTnRnYk8WjmfvLkDsTsKvmG9sW6nvAKHBb27'
    'xi27GvQIvUWWGjvbxZS7UDjpu9OIAL10diO9BUlnPVjiqTyyt1w9qyKVu8YoWD0/2SI8qgEEPNul'
    'i70Pj+E8kgYFPRIS0LzohBu9q2pyvYHkHj2tVoA7M3coPQyWYDokwho9lf4iOielGzyomFC94SAA'
    'vaEZOT28tja9zrVCPIBNxL0ca7q8YQlevG/kIz1qKxE920LTPPEEs7y/+jk9cIZfPYYdljtTAu+7'
    'lhy8vInzND3/MO88tK9uuwxzgb0BQy48iReAu2b0HjzLlDC9q9X/O4hUGD39h1+6xJgHvMBIz7wL'
    'Gss8GDiaPBYoSb2c/yg9ejgavSEQdzyodhI9q40AvTQiFz1owIe7LZvcvDeUbb2Jxh89qimqPCiz'
    'tbx5bQo7bdGsPNH067yiwT09fTBmvSNMrTs3fHg9HUJKPN4jdLxv+1O9mNsbPSkVMrxBW5G8avW4'
    'vHHnwboVWEU9EXWAvVrPQzshc029ZVBsve1Pk7wYAgc9+3OYuwk3ETxy8es8yzbeO7jcXTxdTdC8'
    'J3YkPDNpPj0KUpS9gS7IPH/69rtY6h49ZoEVvFCNvTzFjia9WIqzvGYIsLzXqhG94Yv6OyiaUD1U'
    'wxW9wKMaPR0Jqjl3hHW9p15QvECRHT1EXTq9EJ2Iu1QogD2iFhm8XrYhPbzZabxUlQw79TXMvAU5'
    'FT1RgTq940UjvS8nN71J9RE83z3BvEJ8Ij2VGM08B3QxPatvVTwQZWM9bwDcvHyODr3hGAU9Ppei'
    'POzmCb3w2be72Rm9PNAdjTx544k9hGvjvCrIbr24yy09V4mIu2gsb73nDAG74P5pvQ+GBT2HBkQ9'
    'iOZ9PKPMgj3PAee8gXB+u177rDqDvYC7lzVrPbGayzwxKgi8q4hUvTl+YjyiZUE9UpPVPIZ2ybxl'
    'IV08Rs6VveIHOT3fz0g78lukPTqVaT17DBE9XXRQvJ1q0rw0+zY7OUNqPaz4J7x3uSM9k06JutDx'
    'yDyePvO8hhfjumJvXz2haj+9qf3tvAJEULvX8Vy9UeIyva5sHbzR84a7sK4NPTTplDqLUiQ9YeEF'
    'PTc4Iz1p4Qy8Wl9oO65kVb2soNc8z7ObPGdZeztbwB+8U8E3PURib71ov9o89g2MPUm2YDzAiFs9'
    'ItkKvdeKMD2y2So9VwG1vMts9DxLE9W8omE/vUl+Xj24btk8IBIJPbrQYDyCGwg8niDqPBg+tTyG'
    'R/68CzoNPUT6MjzSNRo9Ds8pvQBDS72xotS8NQMzvZSnPD25AEI89wQnveeoJr384uC8osZAPSil'
    'K70G6UA93y/gPGdm/Lxc7IG9okOnPKiHDj1ncJC8iJdFvaJdDLxcNpS7HnNxvGH7QbpBN1g93CRu'
    'vZf/zTxwQAS7/zYxvVCtXD0Hey67iDRgPREeHb0MrN+7YzWBvDTuwDsbwEe99Ql9PH8U97zZDog9'
    'j2SxvBFQhTx9V1W9qmHPvPWhSTyXhV49eh8LPf5dhL1g/U+8xA8kPY3u9TzzGDg9+FB3PZ5jsLxf'
    'zom76RkLPGvWlb0Q9Tq8b19wPUaVvruzaaa8Ax7SvAB+E7xP7Aw8ZlQ5vSWoHj3G2EW9n8gXvSGw'
    'Gr04nR294HKnPGwrlb3QRX28DtEkvR3X47zadIA97eGjOzy55rwsA3q98h/LvJg+4TzSZku8kEhL'
    'PZyjBb2z8E+9moEnvc2R67wRsSU9Ff7SvLyUQb2i3wW7AvRqPK74Zb3axc+8XI7lvGp5Fr3E3CG7'
    '8w4nuyXYXr0JiTk9ZCjmO1W0MzxqNt47EcdCPLLpRT2oF1s83uWPu3GQCrv1xs+81umEvTxjbLzK'
    'jJi9I0SvPMe23rwWcB084NdoPIf9Ar1TI4O82KtKvdLglLxmAD09mQOXPC12Kz1wfwi9pZhjvQ5C'
    'k7zu7yC8PhwkPb3uAr2pZpA8UzEyva9ET7vU9D26R0v/PEs3MD0nKR69iE5QPWNAlzsXpEc9yNgX'
    'PQjqiL1zJ9S7dlXlvM5dBj0wqJK8cdMwPDgwQb3lYp68YgFevQcI/rpNpJa9jKCfvJvY5jtDVlC8'
    'F2N7Pb4Zk71jotE8Hi6avYmfnbxAEui8NJTjvLu4azxHNQq94+fcvJb1Mj2wAVM8CcGZPEQk57wF'
    'grw8Vo0Ave+HUTxfDlW9cKmIvajxBD3QvAk9rjUkPQ7/wby+Ihy8w24MvRtbWL0tQQ27ljA6vGuT'
    'ID2d8UY9sY6KPZPwg7wAJpY74ZUyvJHQbzy0DFK9wfWzvBEimLwLhAo8Vh6qPGstF7w6Axm9h8lE'
    'PXrudbxxpcq6TXvrvMYcmjzTWJY8D27nPE/6ODy8elG91LYoPVYX/zypZCg93NOiPFJjrTvaAy49'
    '+3/WvF/NY73kzMe6ktwfvU3eMryagoO9rb0QvQs8irsHm5g8LnF3PVrX3TwOq4U7iCccPd9jXjtc'
    'ShO96rFaOigpcjtKPba6aI4cu455D717RkY8paHsPBaJDz3oquW89IFxPKr0LD05P8u800q3vPRq'
    'FD3nBwe99+nxO2wcu7wsA9W816p7veP2NDw4e2w88iRjvU1ngryZzXu8EXoDPGF9gb1kDyS8jj71'
    'PIsU/7zfBke9kgf+PMl0eLxrdh29fPkLvY+M2Dr91q87Z6i0uyLIyztB4kQ80wUevTjAsDvvSDu9'
    'sZbYvMXC/zxI4ZK6b9IDvaj6z7zM9Ss9HEREveT8Br0yNlI9lDM+vSbyTb3Avj49dVM8vSHUNrvb'
    'zR89bPe+O2J7VL3HqTW97n4dvXWaizyPvIC8QteEvNpDDb2iLyK9DH0/vJV0MDw2su48SAAYPUvG'
    'Jbp7CE29+EIlPf/3jrwqja655X2TvFLf/LxDN8G8rgMmvCh9Vb1PwDa9PtrJvDUR6jzIC2U9nPHs'
    'PDaUIj2zhKc7HdZhvGcYILxy6Eu9VAkzPej+Dj0dwEg95lEGPTY40Ly3DX+93QhoPXjdOb3Pkj28'
    '7g+BvGuXmzuM5E89DYVVu81hHjwKd8i8n6uvvGbxYz04zMk77uQuvRj6bb2grCy9i0iuu4wX+jzw'
    'aou8yLGDvH1A1jwp3Ku8LIRkPJMmR73Lf5c7aeecvH0LsTvSkwQ9r4tkvWMTBLwXEwG9IelVPXEn'
    '0DzpJSU9dfKEPLf8/Lx7BWI866xHverYh7zFYva8v+U4vX3Agb0l+6s8/dMxPZBnBb3qhDm98lHF'
    'O4w4qjwlb2U9shgMPZ003ru4GE49ZHEfPc3Hmbs6OvY8D5NAvUb1Xzs97hI71/XDO/P8Tj16lTU9'
    'X/roPFT91jvAwqy8SwIVPUPKZLyI1Bi9EnTzu4FC2bx+tBo7rVsTPMSOzbs7oxE9ov4WPWrdzzzT'
    'Kya9awObvI7oobyar628cjVyvXspVz2CluU7AnJsPfYRTj21dlY8YOzPvFvNVD1M6QY9mfMiPSlu'
    'f7z0bRU8gwZMvcfOVz2pUyE9FESNu/OomTtlp9e8eRi4PCVFd7zYSDm9ujvQvCixBL1ZfIE9VAIC'
    'vVp/Ib0bhTI91WwnvWxUKj3r+ay8PqZ8vRqPeL1/W+e6F6PUuik2xjxM1C070XIXvT5Slj1RYkc7'
    'P6wtPK/xjbto6O27DqAUvdczN71kNRS9QfwHPfUq/jw3YNw8hJNPvYm34jgkt4K9bTOVvIZIZjtN'
    '8v88AaLBvO9//zypVxk90woxu6SJTT0qNuS8S8eCPHhc3rxlDIi5spYHPcUrSb3EZzQ9F27+PP5f'
    'mrzTdR+99E/zvMr6RDzAjJe9Z+/qvMRHb70d9w49r7wWPe28ZL212E+9N9AtvczwLD1eXVU9Qb+q'
    'PH1TKDxF1mu9FUyju+qcGz09QhG97e2MvAX6Jz16U1q8hCsJvcP2Cb2wYuG8Y6xavTpbdLx9JhO8'
    '8EgjPULhJr1DvT+8wPYkvdIQTz3kOr67kxlWvS5F+jzOjhk9YSzJPEezcD0OEqm8ZXxOPBHfSj3b'
    'UwI8nHQGvBhpHzw84Le80pGQPZr5iDz9XRA9qadTPHaPkjyhfUO7GT0KvVxJ2rpwDDQ8cufNvBRf'
    'eD0PnAk9jtZKvUWZQr0lM7I8zj06vVkbxDxcOxU9cEekPEWnfrtyY4K82dlhvQ/8zTxNk1Y9zFDR'
    'OwMqtbuQBiG9RFgNOveHZLza+Bq9m6zRO17wfrw1KLW8T/dJvX/qW7wBc5s8ZpxbvS1gUTrDSZC8'
    'XdmyPIauFT3d5sa7d4pavU6+ubxw2lG9Nj++Ozc1Sb01SmC9v3URPTNcEr34fRk8dzdBPae6Nr2y'
    'MFe97eRFvbYCOz1dk0s9IBiTO3zNM7282iC9hWWsvO1Rbr1e/4Y8vdQ+PVt2Nj1oskS9FtTmPBxx'
    'Wbz7Rkm9W2szPV4LpDtma7g5oPInu1tRTr3w5hw9/3bSvD+N3juTtCE93Yb3vI4+RT15vgA9ZxZ9'
    'vb0f/LyJ7mq9nEMzPWuBO7vWIjk8C7NRPfGRSrzJybU8wRmgO9opSzsv+RO9zCUcvV3mFL0y0Tc9'
    'E45SPe+GcD03+jw8o8yHO7ESLr0qIj+9iHAPvSdLLr2Mh4y8BBgivVbV97w/yjg8J6RZvWHA2ro0'
    'vIc7reBiPTH3/jzUux09vi9hvQJ+TDzMIA29L6cDPaI3UD1ckC69JTMJPS82Lr2k6wC988c1vcmS'
    'zDzH2S69A0WCvJwExzqfSeS8K8ckPeOenbrgqJo8/lDEPKm6Eb30sFk9HhUBPbhsFT37UGe9C0Ro'
    'PVmCX7245ic95eYUPWaIlrtIHrc8/rcgvdRnNT0B4Eg8WspWPMCtJrvnrEs9dHcuvZtK2jlEoIk8'
    '7NaBPBhuSL30kv66leAAPRlxBr1LTwq9m9BoPK+E27yfKow9Y7i5PNt5q7wZDIk7gX73vOEKfLwk'
    'IXU8+ueFPC1MGjzZW248SnuBvEYASb39sg89xtJMuo11Wb0TWDQ9MIwmPa0mp7xtLew6kyv5vMDx'
    '/jz57Ee9Fb0NPVeO5zxlY2w90tDyvNc9hbwvNtU7uru7PDPIDT0Q1fM6wpCGPa+QHjvlwiy97XQY'
    'vYzXSju8HB49NS6FvQ5pBb1E/B+9TtSwPNYmsbyUG8g8mSG3vDLL3zy4s8o6yGW5vOJEQz2CqX49'
    'kwbqvPh2ijzDIDq9oGlYvbfZJjxUqRo9EMJLOwXeCz2g4Qo9vwQIPSXtHzy4fH+9YBokvS4wcr1U'
    'Pkc7o8Fqu4IM8ztYCIy9Wm0QPBalFr14w1u90u5yPBdJOru2i1W9864vOznZtDyb+Fa9hve5OQZk'
    'YD1POkY9nzERvc0T2LyWcx29iu4xvRwRy7wcziQ92UcLPWdq+LvRdCc7eGQUvc3uuLsMtK88u1im'
    'PO4OM7xgZH29oBIGvRN5RL0op8M78DIzPWtPwTzeEfK6/VeavLFSOTsLdJc8aTKaOxgIe72Xpk09'
    'ceSwvPlYhj1zWsG6pjwMvD/29Dw49/C8P77Bu3nesbyZa/28DGpNPfCqET3Func8ItiAPREo8DvM'
    'zzs9UFSQvCosIz0R1hs8O4+KPN8nNT2uLSW94yLeu0oMxbwiPlU9H41pvBpCvDwmRx49MdwUPbkt'
    '6bzbvqS73SMqPagExrw7keq6G6YwvX+lsjvg8/o8gn79OX9OmjuQbC49RX+Gu54nGL2Ywyg9b5ok'
    'vVrNojyBcgs9v/+7O5LTdb3kyBM9qM1TPUMshLzpegA9Z1S5PODe0jtbIxM96bUdvVY7kzwWhh+9'
    'NfZIPW9xnTyGsm89As0+vc3kgD3q8h49/7qfPbm867zmeHe9NVe1vOBUVT1Yv2k9D3w4vF9SqTpA'
    'JQy8LSAqvM43tbkBR7o8CXBdvIIEVL0lVEK80pPLPIv2GL1seG09ZyCFPGzvqzw/IOQ8BZw8PbdK'
    'Vj2yCaq8TSapPAvP+jwzzVO99xYxPTwAPr0jgj29O9krvSbEJb3jUiW9YwrfPHO/eL2EytO8uO6H'
    'vRio/LwHdkC8pmWiPGGM5byumZg83W0TPacKWT1xeEW8eqLDvD5HGT3XlLu8CJaEvW/iobzgqoG8'
    'lHt4PDNCND2EUQo97giPvHVOTj11tOM8FWLqPEAfPr1aUpa8FCWhPMLJsbxSOAg9a85xPVPvwLyR'
    'aSE9Y2LuPLnNtzuWpfk7AMEEPXycMTwoG8e8Uo7hPOX1sryg6Cc94qXavHHgd71JzWE9m7F6PC38'
    'Sr26jhA9AdOLvYkX7LyUnO28DDDwOg+yjTznW9W8a6HhvOJDCr0lfz49C6O9PF0x/buvVwK7Eb3F'
    'PEPR/byzmra8Sx2OPWsb97w9sAa7Z98iu0c71Tvo9YG9SDXHPL5WZLsqEgW9/ZRAvDbRvTuT/F29'
    'izvdPGyy+TyqTQK93EnoPItGrbyEwRo9x5sevDhM4DybdSA95xdhPbPR87wCw6I86C8xPKeQPrzp'
    'gQc5mJjHPEacI70ZPt87oik0Oy1Hir3U6xg8KwJsvc4Rcr1aXzk9MdfVPPYlED0uwFo9KK0KvaAE'
    'R73uTsC773BiPN6suDrz9zW8+6RrPVDZIb2YIc88FrsJPZtuI715h8Y8pr6wPFaowrvuDkS95+MA'
    'Pd+uTj0rQlc8PBm0u6pzBbyvC0k8apPrvD0SHb0X2eS8d3tJPIpJwLw5egg9+tIRvWbVX73BbTY9'
    'auUAvTqbcrrLGE896EXfOhdJ7jzw2la8ilVzvTOHGr2rUu286L0tvbZIG73gSXQ89nqKPUu0ID3y'
    '7rc8QTYFvXz6T7zQfaE8SuhLPT42Ar0kxwU9N5wcPeqYirwQ6ma9HRzQPBqupz0PvBi9iAGBPYNS'
    'Lzs/30q9rCP8OvmHqTx+ZTo8pCjUvBoA67x/4v88BZLEO7uMYz2drIg8dDPCPJ2QWzxktqc8XSWB'
    'PI/iuLyx30U9QsrfvIsO2TxMo+i8HGYpPacBJLxkAwA9D9wHvVxzaTxTsEU9QuRtPD/eY7wSXeO8'
    '39U9PaQ3/DzpNxi811GTuoOkKj2dRQQ9z8VYPbKwj7ynoHu857UJvSgfKb03v8874HIDPExLPL1i'
    'yDq9e4nRu5P27byMZFi9VgMzPWC64LszMXU8kTAWPaqtHL37L5y8ifhBvZ/iYD1jsea8vqk3PZ0t'
    'ETv28b68OfitPBVGQb1pyiu9Bq60vG4fMD2UA3q9W+C/PH11Yz2L1k28RDfbvGE5V72c7hW8YTEj'
    'PdqGOL1ivHY9mvw8vRr+Hz0+PZ06j+TdvDFGQ71IWyK9mQspvcVnWr2WVJW8zJ1EPTNHUL0cVSY9'
    'tmKNPDAGIj1mTL+7qNe6vIOiOD2jAIE9h72xPLzrjDxWeUk9Inp7O70jP71Rmw49rfg2vS1GYz10'
    'mdw72ESGPE0WVjy4Qzk96kD3PBJnhTyKxwy9PnqLvHcrEj0zWQc9FocPvXBhLT2mB4W9reIgPeZ6'
    'C71860y94b42PZhH4bsOTs68p6g9PUVhAT35O4U7oGH3vLXjDb0Uj/68hRcAPfzRZL0FBOG8c4dp'
    'vXtXW7tks2U9/6mNO4pQYDxaEcM8XHg+vT+40jxGumk9nWgtvX+Icr0RguW8I4Q5vYcQGL1Arxg9'
    '33y4uxGetzwuZvg7YasevRl3n7x3jyg8+cBGPQKHKr3W7Ys8YVMDvc7RWr170vi6O5hePWxw6rrN'
    'mM48WvU9PHPqDr1k/Yw6LHjpPGRYujzCkkA8z6ACPIPXgb2EqmA9ZRpvvVTohTxKix29bnXJvNcA'
    'gDwp3AQ9p2LuPHu/Cr1pgR87CKP/ulHLNb2ci/a8rf8RPTehtros2Rm7I+qcvAKnjTxeLuQ8cSEF'
    'PVlhgzw2ntw7RZJTvSEqiT0ySJA8ZKCDPbnE6ryANR08BHG1u6mz/jz93Uc3ZdMBPVCsg72+s4o8'
    'EGBQvQ0Djb1QM4m9AGYsvezIIb3kJ9G6ZYphO/XKjjsO5hI9n2ldvauHCz0gEC+96hXYvGy+Ebxl'
    'ZQe917E0PBU6ELwHJBM9kBB8vKM3CL30eGS7GsYpPJ9WubfM8O26KjEjPR6IB721Zi+8JnBPvR75'
    'jb0PQWe9Ty9mvTT+dzzPrC290NlUvTmCXr1YAVO9I2uZPK0BtLz0FSY7MKWcPEjBZz23HiW9M9AH'
    'PRbUGj0aEP+7IY2nvPR7wDuz5iW8gL6oPKPYRb0U5uC7NEedO+rxQb3tSAm9UGUgPDWyZb3WPCU9'
    '3K0mPaEGBLuO7NW82q+bvNuDUbw3o1g9RHkjvXA7UL2kcSq9mCmHvIvwI73F1ZW8e4S9vNFOHT15'
    'uue8eDyeO9PfsLwveOa8gvcivfETrjzsqJw8wVhpvLWiMby3xYw8Eq/LuVZCPrwXt4U83vDUPI5R'
    'pzwxY+68KxIlPdRcwrxwWXY8lvvPOEcUyDyAEG480CWluBocUb2yAFK9m9eRvDta/7xj8na9rJQV'
    'vePggrwAFDa9zuHYPBt2DD0RL1W9xCshugEKwzsrBt87lY0SPRR9o7xfEHM95gwNPX1Rgb211dI7'
    'rQDKPEWWErs9b7Y8hOFjuj1WAj0h2U+9veVUPD8jQz2skR49rp92vc9W0jzwwCg8/5KOvMTn2zwL'
    'Avw8ZkbRPHq2IL1d+Fi9azkHvH0Vdz2zjwe96bE4PaZ82Tt5SW68wXsEvV0/hz1nRLY8xBI9vbrZ'
    '7jydNgO9UtFKvdhQAruq5k89HYwaPTIC+bx7Itg7fWdfPEF0yzyxazW9RLRhvTI/Eb3KrYK7IkkI'
    'vQsAEj2PQ1m9kmWyPJhqS7zeC1A9DbJevfPFFLp1thu9CaVovdq+xjwDdEI9YOO5vFdIh7xC6Aa9'
    '3fGCva/FHj1Uwu+6NGmWvMaHwDyTIRy8xjGnvGK09byrAnu8MsUCPU00hDzTAf486KkEvZKSYb2F'
    'qDO8YOssvYiBy7rJNcE8bETAPPeVNb3Q3mQ8r48nPcdwkLt70k099e7uPOEAAL3cUDc8PL1aPfgl'
    'XrzYf5U8cTq6O13m5zwvhV67bcosvbc+lzvQtPo8xyi9vOJfmDy/kpw4JmQnvZmUM73dqEC9AS3u'
    'PHiaHTu+QsQ8mbdBPCTrCL26mBm8z+n4O2zVAT3cIZq8Oq07vUAW9jzDZiy9xfXwO5vM2jxBBbc8'
    '9qbGPBgS+TwPnMO8UXw1u4AQ5zwGDxu9G0c2PcC5r7zyUvQ6rJ6FvGYYZ7wjPmo9+LN4u2aJ8jzh'
    'Cym9rd2pvAisgL2JYEG9vLALPeSMGj22Jl09oREHvG4M6Dweyxg9xd8MvXMuMz3jblO8v/xGPX0J'
    'vjxQJHk93AAMvEAGILvFg8m8neMMvdqpNr1iFdi7YTWBPfuWCzxdwp48DR1cvEVH+rtvPEM9n1x7'
    'vU0ooDtt7DK9fZjXvD3okT0YKh09sDXGvOz+Hj3unfs8pQ3YvJPo/TyOqH69q6WDvO/cxDuzj2i8'
    'VkUovdFqP70Slpe8CAAivR5YWL2mScg7MlmwO2AcOL3Kqx+9MtEVvb7BD71CvQU8wBUVvSZndz2j'
    'PZc8SxcZulFiAz3a8yW96W2VuxJuKr2Z2w298VTQPOQGKr1y6Tw9Zl+1PMaAWzzed665bqdYvcOf'
    'sDwR/YK7Cif5vL9uGL2Wpri8KkcuvW6TQr0s31m9OlazvLN/HD34fxI94ZAXvZNKXT0svAS9VQAg'
    'ujyDbj1fqny8HHw1vEqwfTvv7nC9FQW0O+FVFr0qGEA9KbDKvJ3zBj2oJN28YdEqvc0/mz3kgXs8'
    'U2ViPWUNWT1LbZ894jNFPWLWGT2enZs8iLgkva1vmjwRPjs8G8TlOdBytLvV3Qu8ttGJPSuMbDs4'
    '7y+9aJKUPL2aUz336B69xn1BvLvRE7q+FJK9snkUPcPrQL3k4368IdOAvMgN8zxbnJq9EYMqPNp+'
    'ED1ia6w8BF8hvdCdBzyEeQo8oYpePX0WxjwapC2765MTvRcNyzwWzdu8btYfPYxjMj02QFY9Q42b'
    'uyrpRr0OQz09nKtMvZu3CbyNlM+8zI/vPHrDcT2OhPG8skrCPBfwPz1szD69HLIEvR1pSr3Dyp+8'
    'nlUHvUkabD23miA8GQaOvIrfhb20FMs71NhHPPNsIb0DEZw78S8nvX0kab0og0m9zGiDvSrnVz2J'
    'GEE9tOcWPS4w1bxvt/C7TdHxvOq3VLyBO7Y86jpBPfxOazyGRhW6d/3juzMc5zy1jyy7w+pdvT+D'
    't7tgn4E9EpYxPRW9NLwjoxC9T0rZvD9qvbyGZL289nkKvTLQFT1n2Ea9GHAvvGC5DT3NGok8INaO'
    'u5J/Ej0bNQy98B+NvO+HOz0ZPwK89EkvvV8RKr1+ZTK9jx/WvJSYfjxkOea8TbcePOBTcr3xza+8'
    'Wh17PKHFeDx0HE09WoEXPBbBk72Afgg9lm9lvbrairwEbbA8b4QsPXU+Grxog4W7RDD5PGCohD0A'
    'BCU9WmLUvNjhyTzFv1W9a1/9vGcmpjxGf+48hvsYPb9xdLun4Mu8vZwvPTewMT0/Ynm8qN22PJKP'
    '3bqJ+ww9xOsnvU4ROr13VW+8/RgLPQZ9gTupJNQ8wyWHPIBCfD2n7zG9yrrTugUX0TwWD6M4qnwo'
    'PUNfJL0LISK8YAgbvQsybTqToW48WTfVvEdIMD0D62s5CPk9vTxqvLzXugc9B7uDPAC+ubwvNaQ8'
    'UDIyPXfXcD03fEu8f+AuveSBm7vAMSe91TxtvaF36LyjIR28FiNmPEi5Sj293Ke8KWrxvHW0grw3'
    'bKI8l/ZYPfKk/zwwvYi9jkHAu2vsWj1Nz+m84aBBPF4SKTzc17C8zf4avcuOozy3CXI90mfhPJUO'
    'LDwGyOO8QlEHPcjeaT33Sxy8TJYQvBqeW7w3IJM8kJsuvc5jJjz6UAq9DnFGPUXrY7199gK9q6N1'
    'vA0AW7xdgFm8WNM8vW6LebteIA69IL+CvQkegrzUk2I9NOmQPE11jD1V//88heqBuyQUf7xgJIK9'
    'L156PbYPOD22tyy8fpy8PAnwKD0k8M28BaQjvW9cB73+1jq9Bk6NPc6tfDwM9Gi7cR7KPJQb1bx3'
    'jT09TvZrvR7wAT1UInC9sx+xPJPXDjwBdWu9n4YuPS+AIj0K3FQ8fcNjPFV8Kb2I1JI8BqCjPD1R'
    'UT2QKi69/kXSPB06Vj1V59y8Q0O7vDC9Oz1DoZE8fHsJuwFSxbzhdMY7oNZyPCmxUD0UQpW7hqsD'
    'PT+DuzxuqZg8DaM4vZ/ahDw/AwY8EDELvX3CJLwnK328mimWu1ubtLvoe+U8hUfGPK2JVDyKeas8'
    'qcPkPKevKb12iU28I1GFvZWhZ7wp6YU8yg9iPOxLhzxc9LW8n2MNvQbRqLyv25Q8XT9NvTTjaD0J'
    '+eU6apaGO4vH5rykkyC90AIhPEFfgzsu3JA73D05PeA+Z72n9Dc9y0hevM8TNb2nks88GYJJvaVc'
    'Qj3HhhC9gEb8vMqIojxFP1I9++dGvSf92TwznnW9UgtBPZZVtLup4qA8WkogvUpCCTuIdEk9fzvr'
    'PFAK9jwvmSM9YPoTvJqEWr0hP5I7ME8GvSJtd73pQqY82iJUPXgEczwiCzW96oydO46DKj2r1ls8'
    'FzVgvbv3HrwpVCG96KgLPEqiv7ydt3S9bpaDPDvpjrxFlkS9xpN6PXW1k7zAFX29JX2xPOleRD0t'
    '7re8iZxHvSlFBLx1hlE9UlZLPaZHszyG0dG85lLPPGEQYTwn0jk76ZvRPAOlAr2gYdS8FKUIvSi4'
    'mrwhvz88D8hpPAFXUTz9zCA9kBYoPfV1xDzgzkW8SnMVvYkcKz1AEmM9ZVHWPPVqBrwadCq8ClYw'
    'veo4UT3x7UE9MxayPLtzI7zyB+g8keQUvVByJD3AkUm9d7tQPVOFDr1wRBe9bbvoOyrGrbzdX3Q8'
    'DcX1vJGUNLwl0ec71FzCvKTdfbxphrG7ywmDPPSYCz0SrYg8kiblPCH7GL1BtyO9AuX8vKu+Sr2s'
    'lEe9D2HdPEMsiryCTOq7OgX3PHgQJz3G+ZI8BiprvItsVbzydB091QOsvCLmzjx0WuE8Ld4WvTvC'
    'KDtAbfM86uM4vTd4ozzfVdY89jtGvfLWTT1nVmA9sYz7PPnmpDzObUU9RnhEPGWnmDug1l47jAVQ'
    'vVwFh7yvqAo9dq0bvFSDi704NBM87+6qPPYnlzx9BO080PLrPHzIgjzAwmM8bykXvbRoL7wHdEE9'
    'NmD3PF0ZqTu+FTI9hEwkvBpCKL3kSgC8h8JPPZrBGL1GKpE8K08rvB73fby/tUi9cQkDPW5McDzk'
    'chA9HchUPbSfg7201aE8xRygPMDGd7ze/4q9cqpIPdSH4TwYyCE9uMAmPKTrtLzlKrk8SfsWvdGN'
    'Cb2yTO28cRqivf/bIz1PZAM9IH5tPMfLlDyeZ1O9K14GvXUIa73hZyA86dulvBytJz3/Yis98cQW'
    'vZGkRDxoVjO9hZybPLPNkzsEAg08YBWEuqc2/jvSIz+9j20RvLQlwDxmYQO9lj6UPZhOrLtnxxK9'
    'rJndPH0AmrwdF8c7pGXqO1F+ybrEuyc9y4MIPWIkrjqQykC9J/6AvWG8Fb3xUXk6t6eIvSIoCz3p'
    '9XS9+r0MPbGeZD2uSuc8diXEPDV6njyw3zA9HCaePTRtcL0CeRG7Ia8svShVUT2L9so8HLYBvRHl'
    'jLxphvk7G8JwvHYcWb0rJgm9kagCPM+zebq+zWG7yQBVvbvd3rwV/Lo88UOtPE3KSr1HYDe9D9YN'
    'vWV6dz3DJTu9M88DPKQfhDyYvgg9yQVhPLtibL0maVS9Ds5YPAgiaL3ShTu9p1MDvZh3fr1gkho9'
    '7buSPbN1fbzcS6Y8TlFuOyiJqzxRvO48QvWBPLTxE70ZEZc8YSNEvD4i6TyttCe9r/ssvWUtuLxU'
    'lHQ7R1pbPSUYUD3kmbK8KP7au8KuU704tAe9eip4POZhl7zPo069rfVWPfh/NL0cY4i9dnkaPbFB'
    'NT3PR4k8oVsmvVnyFr1W7qk7Ke/JuxysEL3x7ho9otEPvbCph72sj6Y7X0beOnA4Pj3est48gF6j'
    'OTrrfrxI7/070L4CvVYeMD0jDow8T5TMOxmNNb0m9Vs9Tx8YvS9ORz1wJN07ltNMPYfIdLycdi+7'
    '0qHbvKvx2zwO50e978g9PW4KXD1cZs68iEQoPTrqGj2BBDY9aFJqvSbNcL3D02O9s+CAPGYw27xk'
    '/jA7/mfWPF9lCj100nS7TELmPLn4n7ub3DG7lYUoved89rxptw49MJqvvP66ej1e6Ro9uAKJO0l4'
    'szzpFTI9vg8lvcvsQL1zBxk64xeBvQ0WL70CxT27YPlFPUlEE73FT628u18FvSqEMz0r/TC9aQWJ'
    'OSfmNT3F+Ig9qDMFvIi1Fr1oHEa8YDtFPWoIvLvHbC49ewEnPYrO2TzQUQi9JNrqvI4337zT+oe9'
    'fHtEvZzfcTxwljm9cfSdvBUtizyjEBg9LcpZPdjZfL3owuu8v7QSvR8YWDw1rA69DgOUPLcmjjwt'
    'v+W8QEEVPaQdr7zXNjg9YwNJPXW3Hr3TWDg9GTYXPWdzKb1pxx+9/cTgvAhPWz3PjGw9itevPJ6J'
    'gT0vne28oWWWPFTpB73DZ7u8RgVuPMTFSr3+pZ48CPDxuz3ngr2xwzW9qFZavb+hPjz7nP48Atru'
    'vCxk7bwxhLq8jkYMPc0bTz0gnFy9lvQBPWl+Az2RG4c8nU93PbWgzrxBPiA9aXcPuwBLCD0zlxo9'
    'VgSAuwUqHj2bI5+87rUgPdXPq7wTj6y8K7/dvLG6ST2lS2W9dIY3PdLyHT0B4349C/+buwDdRbrN'
    'hCq9PQx2PWNUXbyd8Po8vMN9PLKLjbtF1Iy8nvqgvAxGFbwlL927jFGOO7cgkjytpiK7f+6MvA7h'
    'pjs+XG09Z+JXvW/rwzya30a9uk4lPVB4lzxK9oq93oaAPc4JST3DRCk9BFD5O9WKEL1yqhm8qCoJ'
    'PefP6DyGeAI9XFlQvHYPr7yZvNi8RNqgvYtIaj2qi0E9CNEiOphLErqYKji98r5OvZspPT2L/go8'
    'yRJQu/lRpLt+Roi8KW/dPO3KRr2X8xG9pxu6vNOZvjuMpGw9uPRyvPS6VL1TqwW9fEVzuBokRbzO'
    'AyS9uHwjvFbCNDwU4QO9TytZvLISRL2BcR+9iz0qPSSsIT0Glkc9UBO6O26Kb73Gdfs8Y/qfPKwL'
    'gL2uK9i8V3jcvK9VGDs3qdc8GLNLvEbS8ruzNDo9qh6JPFCnAL15ZMg88amFPM4nibwFiQe8GGt9'
    'vVt6LT0GNtM8/K9HPQdZSD0lN/c8LCN1PYNHEj1cAD+9fZ8pvP0cQr2z6hA9kOLnPOq6Bb0Z7s68'
    'A3NxPVxNGrwVS1G9GaalPL/YUb21Diq9ImglvXDGSb0/GUG9SKimPOz4GD0ZucW823gNvWANyTyQ'
    'kBY9XNkKPUtmSL28kOk8r0ITvdKiOj3cKBU8MFBuPGs6tbtfH7Q8npYBvTdoCb2Tijc9ab8dvZ35'
    'iDxE7aY8HPGSvH4Us7wYjiO8HiwjPTPo57yV3gm9GaZBvZC82bzGPX29sKAFvAgbRzwRx+C7PfnT'
    'PM2+nzzGhBM9MNt8PQykKDwTEQe9TVuRvIaUEbw5LPC8tV6BO7Gk67yExuk8OfI/vZwKLD014ji8'
    'x3tLvQaYqTzZk0q9LieFPW3R/zyxAMQ8c2USvQGDMz2npE29iB8zu0WmNT09sAg8eGFGvQ+C27uP'
    'Il69+uOFvIggar0BmU08wQ00PbUiDr0+Y0q95tUcPZGbHz1Ptle8d48APddB+7zxqkI9JGoavUP8'
    'gTwY8bI8BGQlvIDEej2jmCo9ltQMvQu3HT3Q/Te9lmWuu+rLPL1KC608A0KePHHfTD2KPCW8GlU2'
    'vQ8/rryFMPU8e0aIvNi7Rz2z33o81vJbvZhtCT332dy8Zo5APURBlbxRtM+8qQIZPYRlLDuXO1C9'
    'fTKAPZN4ZD2SoGG96c2MPNBTAbybbfS8PSQivQTQgb0+oOy89GoePTNwgLxfVuS7IXtMPNKvPz3N'
    'ueq8MM4avDimA73s/RY7GKaRulHRWz2C5XK8w5tavTJM5Dz4zMo8Kr7qOuLEJzycmhE9IAF1uvBX'
    'KDyG0Bg92SLbPHQeoTuHxB49bktSu+/yN7yqYFg9gGIVPfjB8DzvLyy8/LGxO3CCWb099pa6rzgW'
    'PJZQL7zA6S29pEQLPCwrHz2VwA29X6IMPZ1noLzIVKe8PLRTPW17HL1guF49u2kgPQW4Nbpq4cO8'
    'pnZdvKxbEj2RuWC9+xZMPc7GrTxP9+e8O1AFPa6d3zyS0zW5+3wWvdCCdT2lKia9oTYwvcGvtTpB'
    'Jr+7F/e6vA3aET3+T0u8Mky9O1X2r7wIBZu8zgOCvej0QT2mEz08QPSDPVJQEL3B+us8bG/9vA2D'
    'Eb0daUy7dlLgPAqS8ztJOyO9/640vbA7aD1yhRI9YDEEu6n5Pj3MUEC9/0KwvDxK8Tx7dX69hPNE'
    'PQQXYD3PSZy8buIOPDfBFzwp0ok8iJ6mvJPWK73IMRM8p8CqPOjj6bz22lk9HVCMvCIBVD0qJ3S8'
    'sPutvH/F/jy9/Vs9UqUKPWb5tz3a/BI9xe8nvRzJMj1ctjy93LVdPC6DPb299448OJYmvGAdVj2H'
    'oFw9f2UuPHpgWr3Rj5i8AAYrvCInMLw4ZDs8f4cXvE+fZj1ZYSK9J7OuvAS4wDwRjzY9uZ+SPFIH'
    'Hb1/GFU9s7KRPHdKbj33s0i9UHz5vKuYEL3xllA9Fgc/vQSEWj0RLiI7Yq9Uuzh2n7uuzx29RrI+'
    'PWTHf71brYC8uV5KPV2myby7iXS8WSJtvGEaKz2pP2u88KArvEeTIj0AwE29lxfru5mvHD1FAoe8'
    '6kj1PPdVAjyAqdk8I8kEO6+rmTx3gjw8VrWZu1jhF71qU806CX5dPc1NHD1bwYg9xJyaOvu0Yjzc'
    'tpE87k4OvXZOy7xRZ/080VWDPJK17Dw5XXE8WMNiPCs9Hbwh5zq97uiuPADLo7xLKT29dLm5vF9q'
    'Nz1AT2u80cG2uxW5lbpouBS9rblfPEJRR7wtKAG839wcvcm5CT3T/UM9BcVQPR7nTT1LGx28+fiu'
    'vJ2ftbudb+I8jxH6vAzudr2NSJG8YU1+PSsePjztbQw9XHNgPfKk7jtRzlQ8RwzZvHoscrso8jy8'
    'rbYbvTDPmbzL95i82TtAvE8SV70FY228vTYmPTAceLzXEpq8TTBXOS5/8DyNA089mCsoPawmAL3p'
    'Nzs8+WM8vDLBY7yzwzm8TSG3vJNMPL0TDUi8qBUkvFadwDvCYEw7bEfpvHeXJj0TeDe8m1thPUVk'
    'rjyD1X08SsP7ux6e07yOaLK7IJ0LvaSCEj2NkEE8moTkvJeikLw42Ka8ZkLMvL2Wurz0rni8PbCH'
    'Pfv6Sz2Thwg9ecB8PV6Ra72hxCG9QBY9PThSDT19U4c7HN3sPCkSNr2LiwA9/YHGPKMaLb0KoQy9'
    '3UGPPBgxAb0owk29p1XfvFcmiDzabKc8BKbzO5cycL3V3yw9N9ZnPVukNb2Sv+a8NP0mux/qvbyX'
    '5Bs9hbY8vU43frwXVzk9THJhPYmSMz1AK3A9IrjlvI4ihzwyVCo7UO7/vEcjDb1CABK9iTPZOn+q'
    'QT1LslU9jm46PbjSWDyz/fi7ZekGvf0JP71u/CO9t1eeu1QVbL3Vzzy9t7LxvK5mhr3drKc8SoqV'
    'vHBOYj1jirm8J8I7PQImADzdCv88cSWTPOn5Zr23MnG8Syu8PGojPL0SKCw9eC0qvLRhYL0uAj09'
    'qJBSvSQv6zxJhhw9Xsv3uofoDb0tHSQ92YNKPUMMSj04cqG9R6GzPElsgj1NbKa8OKvfvNMcNrzF'
    'AoS9V7BZvTqegL05/4683NUGvQDD3TxvZqU7KNWgOm2SGr2TbtY8gTX0PHeaqDzbOoS9r9kIPXyu'
    '6jx2AzO8Gf0IPXTjkbwKkDg9h4BdPVDrMLyxNIg6ku4SvQUJ0TxhiJc9DK4KPSsxiL0I3IA95dSj'
    'vBrjJT2gCEk98sx+vR61Kj3cPWG90tSzvCJ6tby+bdg8A286PUHC/TsIzFs9OHodPT4KMz3TrA48'
    'bGpcvETKC7zoigC9ttJHPczDEL0tkrS8ZUxJu38c5Dxh9Ew9BJwQPXm0ZL0Gn2+9TJWtPI9U6rtB'
    'lAA9ztU1PZB2k7ya1GC9rYMbvX+6R71W6++8CxczPd9iVL39QRm9pvJFvfAsYL2iAha9J4qwOpA+'
    'QT1fELa7f9EtPUYyhL3srz28mukqveP0I70zYgs9Ss8Qu5ACsLx16im8u+3hvFa9+zzfJ4u9arWU'
    'vRDCJzyElzY92fK+O/SOBTw52p+8o8xVvRyxlTx++q47e8SsvBkWCj3JWsw8vRkWO0AlrTwKL7q8'
    '9FyMPPDOQz2uj/w8W82suzK2xjxhFoO8ceZlvQ9WAb1MUTA94XXzO9i0sj1QYEG8VPuKPR2GkTyE'
    'SIA98D0pPYLwr7uisfa7+HTQPIffETwAwTm9/E7OvJd66byzowa9xjh+vXNYODwME7y8owezPHTM'
    'B72kfMq8xku/vIRcMT3P3wq91xNuvdtyJz0vbV28qzuCvbd8pbyXSu27gdM6OW5THL2ukPC8sEP5'
    'uTmBE7uHQKm8sTzdO0C/AzxJUju9u05vvRD4pDtQtIs741yKPXRrMr36Ylw8l6ULPOHHobt6jCq9'
    'dSCNPA7UKDzV6n+9pWzjO4klk719eXG9AF5CvD2NRz2CDWY9aSYJPdyYCDyLn968q68gvcv3bL2I'
    'Mgs9/tbuu+9uDr1AYrE8SdBcvSA8g7yVobo7oCLdPFBMVL3EWtk8wpAHvBjDdDwuT608q1hgvfCs'
    'pTws8z69pSHoO5jrUr1T9FG9HrF4PIWiRb12XRA9mPLbO3gPYD2jLYq6aaV+PIHCm7tfyzs9ojrs'
    'vEX4hzxgCCa9/UX8PF7tbz2z71S9dvE/vVa7Izx6sai67XViPWCKH7wXZiQ9pK2TvAwfe7wBCta8'
    '1tSkPPxAQjyfUD898/DxPMSaYTybsbW6EQ6yu6iY/rxu21c9PNwtvZgHaD1HVH+8ep4LvVXwCDzf'
    '6Zi8u6kQvTWKPrw2m848o7mcOqgFYz2CAZ48KQNQvaTugb14nvK8yQcVvSimjjxdsoS8M9MbveDa'
    'aT08HA69YXSRvCVD/zwuIZI8EjFEPfdWKjyn6VO9MxAlPAXWAr0I64K9Wp95PYqU3TvvbAO9/cJz'
    'PYNb9bxN9YO9YvHCPAVriz3dOAa9iU47PThDt7xfwh68r18YPcA5+jzSkj09jyOBvM4PCD0wJCY9'
    '8R2evPCmS73nkoQ7ineFPFiWYb32xlC9mm0tPXG9G7zjBD69J8FFPfDwlz10Uoq8BQyyvL73J7wg'
    'TgM9rxBDPIOUKz0sT0s9/QDJO4wILb3lAku9hlfWO26wyTzw6P68KMVrOpeDH73BsEM81cXOPDvB'
    'Vb2WDiK8f6w1vZN8F73CTyS91ESEPZMgB70OIDg6PgfCPbSkaL2K+Ck9ssYTPG7j1jvh/UI9NjiX'
    'veQkmjz4Qg498aAju9oEoDydCk49i8coveyCXj0YyFi9Idq8PLl8Aj1wLKM83NCPvVmAfj2uQwg8'
    'g6OpvKYeJz1CCQo9R+PEPClFTr3TzBw8kZEHvbv97zyS7qA9SBG1vMPUwrwzM5m8CEmJPb3iLruW'
    'tz29u6TevF8n3rxPt6Q82GyuPKp9Iz0aoiQ8YyUkvRneS72W6BI9bPJYvfB9MD2SkYe9wOwuO8j5'
    'P72uFxM9qbirPIl0EL0lc6O9gJjjPJjb8TyPwH68f8ifvIw/Mr3x8r08JW4xPUTxPjzI2Bi8vis2'
    'vWpfyrzSQU+9D3AbvJFZAL0S4dS8M1ZFO0edxbzL3TA9xbAzPYB8GT2vEeM8SRQxvQsu8TydaRc8'
    'D5BbvKHj1Lstsdy85tEZOXYWcL2aTDy97DDpPEIdHztrNKe7C02AvX/pFL28s3C9bw9OPYq5db3S'
    'kgw98SByuyH9tDx6lcU8TGhXvW9uxbxXwfY85a8aPJ2mE72RgsM8hQKsvP2V2bzYthY9H7IyPflT'
    'Lj2jGnM9NfsGPVSAO72aP8E8iSbfvAsdXTzDSic9tCW+vBrxdb1rdWi9xsSDPGrfsjwV7B09pU1C'
    'PThkjTy7kSk973KqvM7S4TxRnEC9ZAc1vYAAuTzOexU9OBscPKL/MzyR9ys9lDocPakZAD22zbA8'
    'O6xevWe8Fblbpa07uz7wPKYPO7zF9ve84CWTPO482LxOe828GidIPeBaSb1Q2XE9+AjAvGtShbyL'
    'XlA9F0g0PBoGeT1BBms9Xf+CO0VXKb0TZNu8Zlq0u2qn4LxxZVc8NpZ0PULzwbz6JQG9W39DPYFC'
    'V73aMny9YasdvdovUD1dIKu8BYZZPSSCz7mUPbs3dVFDu4Oci7wJLYO9MbqxuotBxTxvXmo8b/US'
    'vMMIITxExQg9/QTdPCgtFr2TZ7K7o754O7waELyzui69tz5avCcbvjwofg884+FSPSnb8zycmA28'
    'OywiPELJRj0brq08vwdPPb6wgz3o9By9yJYvvQbAXz0DC0y9t4s1PQclVL2tgD28ndOcPAA/UL3C'
    'YQo9WyyBPEOOP73mtGW9FO00PLniTb2fGtS8w2G8vFqFEz2o6SO94g76vI9YrDxg2la9jj3JPNQa'
    'urrgRMQ7Hf5WPAUeHL33WBQ8fKcPPfPENj2VkTE7PkkMPSFqiL3qn9c8KZv4PFHB67yVUFM9xtN1'
    'PXu8WL0fiEo9hwdQPaIMhb2DuMC6GhIFvSprLb0v+9+8rKISvUUBubzYtxY94GvTPPu78Lysuza9'
    'IOGpO0Nw0ryMmQC9HPw7PLtlBL34Juc8I4yxPGIZD73yMvm7GoVOPfLWVj0baIw9eQMhPNyzRjwD'
    'low9g2KuvDKES73K6ic9yu83vRIbA7w5Xuc87bTxvOavcTx1eoi83iQ7PN0W7Tx2bhG8sUzZvIuC'
    'UD14rl28RAqQvGnIRT3dLPw6R12bPPfbADxUnEq9ZhPovINtLbzKjE0931WGvTmNQb015FW86ItH'
    'vIsNob0iOys9OUjsPP2OK71aodW869j5PCA2AL1VQwY973mAPQNCJrxD6MC8e1kYvZmB5ryiMzU7'
    '8/1evYzwQj3z44W9rgCiO+LGKz05ox08pibkvKcAIzxfpSW9uKgKvYNc4DwCrra7X5YkvB/jkzxc'
    'lVo9uDaJPBqIL71Yj8Q8g5VAPSqAWLzZMlA9+OhCPIM5Zb0Hcwg8ZuZfvZFL0TyLElA9kVMovZft'
    '8TwzAVk9cQwdvTSi8DtDUFY9dLN/PG75vjzK0zK8usV7PRf/zjsNcAS9INMlvR9vab05yhe9Aug6'
    'PAOBED1Renq5FNU/vVKpz7xJSEE9o/ghPF03v7xzSU09f0FMPUgv+zxR2Qy9mSGkvPMD/zw8Hjk9'
    'WprXPDPoMr3NGlu9UDqoPM7HT7wMVPq6H0ZwPW5xFDwu4oC9ibLGO93MBL2RecQ8ThYHvFxTmTzA'
    'nlK9QmeNPFBLBwguVM0SAJAAAACQAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAABMAPwBhel92'
    'MzdfY2xlYW4vZGF0YS85RkI7AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpa+ZTYPOnjTT3vTl+987ZkvE9Ofbza+Fi9nbMpvbbGSb2qeE87'
    'K5RjvHtwuryBQRi9l/BlvRdtUryT6xu9BQ84PVPqBr2cNTG9mzsuvXZiH719H1495cXaPCmK0jwm'
    'JqQ7C1c9u5qbvDzXnUq9MHcPvcLEtLyxnb+8vqi3PJd70LxQSwcIAoXQwIAAAACAAAAAUEsDBAAA'
    'CAgAAAAAAAAAAAAAAAAAAAAAAAAUAD4AYXpfdjM3X2NsZWFuL3ZlcnNpb25GQjoAWlpaWlpaWlpa'
    'WlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWjMKUEsHCNGe'
    'Z1UCAAAAAgAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAIwAtAGF6X3YzN19jbGVhbi8uZGF0'
    'YS9zZXJpYWxpemF0aW9uX2lkRkIpAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa'
    'WlpaWlpaMTAzMjQ5MzIxODI5NDI5MTk1ODMxNjMzMjQwMzkwMTE3Mzk4MjkzOVBLBwgSBeUpKAAA'
    'ACgAAABQSwECAAAAAAgIAAAAAAAASi1l7DELAAAxCwAAFQAAAAAAAAAAAAAAAAAAAAAAYXpfdjM3'
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

_bundle_cfg.rollout_policy = 'heuristic'

_bundle_cfg.anchor_improvement_margin = 0.0

_bundle_cfg.total_sims = 128

_bundle_cfg.num_candidates = 4

_bundle_cfg.hard_deadline_ms = 850.0


# --- agent entry point ---

agent = MCTSAgent(gumbel_cfg=_bundle_cfg, rng_seed=0, move_prior_fn=_bundle_move_prior_fn, value_fn=_bundle_value_fn, nn_rollout_factory=_bundle_nn_rollout_factory).as_kaggle_agent()
