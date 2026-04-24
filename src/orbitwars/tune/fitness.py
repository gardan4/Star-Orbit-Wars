"""Fitness function for TuRBO/EvoTune: weights → win rate vs opponent pool.

We instantiate a fresh HeuristicAgent with the candidate weights, run N games
against each opponent in the pool, and return the average win rate. Uses
kaggle_environments for ground truth (slow but correct).

Design notes:
  * Each game gets a unique seed derived from (seed_base, opp_idx, game_idx).
  * Seats alternated to cancel first-move bias (even games: hero seat 0).
  * Draws (mutual elim at step 500) count 0.5 in win_rate, 0 in hard_win_rate.
  * HeuristicAgent instance is fresh per game — state-reset is already handled
    by the agent itself (W1 regression test covers this), but we also
    materialize fresh factories each call to eliminate any cross-game state.
  * Returns per-game records so callers can compute Wilson CIs or seat stats.
  * Opponents are specified via picklable ``OpponentSpec`` records so the
    evaluator can run games in parallel via ``multiprocessing.Pool`` on
    Windows (spawn), where closures/lambdas would fail to pickle.
"""
from __future__ import annotations

import multiprocessing as mp
import random as _r
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from kaggle_environments import make

from orbitwars.bots.heuristic import HeuristicAgent


# ---- Opponent spec (picklable) -----------------------------------------

@dataclass(frozen=True)
class OpponentSpec:
    """Pickle-safe description of a single fitness opponent.

    Factories/closures don't round-trip through ``multiprocessing.Pool``
    on Windows (spawn start method re-imports each module fresh and
    expects everything on the task boundary to be importable at top
    level). A plain dataclass with ``kind`` + ``param`` fields does
    round-trip cleanly; each worker calls ``make_kaggle_agent()`` to
    reconstruct the actual agent.

    Kinds:
      * ``kaggle_builtin`` — pass-through string like ``"starter"``.
      * ``random`` — ``orbitwars.bots.base.RandomAgent(seed=param)``.
      * ``noop`` — ``orbitwars.bots.base.NoOpAgent()``.
      * ``archetype`` — ``orbitwars.opponent.archetypes.make_archetype(param)``.
    """
    name: str
    kind: str
    param: Any = None  # archetype name (str) or random seed (int); None for builtin/noop

    def make_kaggle_agent(self) -> Any:
        """Reconstruct a fresh kaggle-compatible agent handle.

        Call per-game, never cache — an archetype's internal state
        (last-launch turn, sun-avoidance counters) must not leak
        across games.
        """
        if self.kind == "kaggle_builtin":
            return self.name
        if self.kind == "random":
            from orbitwars.bots.base import RandomAgent
            seed = int(self.param) if self.param is not None else 0
            return RandomAgent(seed=seed).as_kaggle_agent()
        if self.kind == "noop":
            from orbitwars.bots.base import NoOpAgent
            return NoOpAgent().as_kaggle_agent()
        if self.kind == "archetype":
            from orbitwars.opponent.archetypes import make_archetype
            return make_archetype(str(self.param)).as_kaggle_agent()
        raise ValueError(f"unknown opponent kind: {self.kind!r}")


# Back-compat alias — callers still unpack ``name, spec = pool[i]``; the
# second element is now a spec, not a callable. Tests call
# ``spec.make_kaggle_agent()`` instead of ``factory()``.
Opponent = Tuple[str, OpponentSpec]


# ---- Configuration + result records ------------------------------------

@dataclass
class FitnessConfig:
    """Configuration for a fitness evaluation pass."""
    opponents: List[Opponent]
    games_per_opponent: int = 10
    step_timeout: float = 5.0
    seed_base: int = 0
    workers: int = 1  # 1 = serial (as before); >1 = multiprocessing.Pool
    # EvoTune hook: if set, each worker (or the serial path) installs
    # the compiled scorer as the active ``heuristic._score_target`` for
    # the duration of the evaluation. Contract: must be a string that
    # ``evotune_bridge.install_evo_scorer`` accepts; None = baseline.
    scorer_source: Optional[str] = None


@dataclass
class GameRecord:
    opp_name: str
    seed: int
    hero_seat: int
    reward: int           # +1 win, -1 loss, 0 draw
    steps: int
    wall_seconds: float


@dataclass
class FitnessResult:
    win_rate: float                    # with draws counted as 0.5
    hard_win_rate: float               # draws counted as 0
    games: List[GameRecord] = field(default_factory=list)

    @property
    def n_games(self) -> int:
        return len(self.games)

    def by_opponent(self) -> Dict[str, Tuple[float, int]]:
        """Return {opp_name: (win_rate_with_draws_half, n_games)}."""
        by: Dict[str, List[int]] = {}
        for g in self.games:
            by.setdefault(g.opp_name, []).append(g.reward)
        out: Dict[str, Tuple[float, int]] = {}
        for k, rewards in by.items():
            wins = sum(1 for r in rewards if r == 1)
            draws = sum(1 for r in rewards if r == 0)
            n = len(rewards)
            out[k] = ((wins + 0.5 * draws) / max(1, n), n)
        return out

    def to_json(self) -> Dict[str, Any]:
        return {
            "win_rate": self.win_rate,
            "hard_win_rate": self.hard_win_rate,
            "games": [asdict(g) for g in self.games],
        }


# ---- Worker (top-level so Pool can pickle it) --------------------------

def _run_one_game(
    weights: Dict[str, float],
    spec: OpponentSpec,
    opp_name: str,
    seed: int,
    hero_seat: int,
    step_timeout: float,
) -> GameRecord:
    """Run a single ground-truth Kaggle game and return its record.

    Must be a top-level function (not a closure) so ``multiprocessing``
    can pickle it for spawn-method workers on Windows.

    Determinism: global ``random`` and ``numpy.random`` are both
    seeded per game so serial and parallel paths produce identical
    GameRecord streams for the same (weights, seed_base).
    """
    # Seed both python-random and numpy-random so env.run() is reproducible
    # regardless of whether we're in the parent or a spawned worker. Kaggle's
    # orbit_wars step function reads np.random for planet spawn + comet paths
    # when no explicit seed is passed in configuration.
    import numpy as _np
    _r.seed(seed)
    _np.random.seed(seed & 0xFFFFFFFF)

    hero = HeuristicAgent(weights=weights).as_kaggle_agent()
    opp = spec.make_kaggle_agent()
    agents = [hero, opp] if hero_seat == 0 else [opp, hero]

    env = make(
        "orbit_wars",
        configuration={"actTimeout": step_timeout},
        debug=False,
    )
    env.reset(num_agents=2)
    t0 = time.perf_counter()
    env.run(agents)
    dt = time.perf_counter() - t0

    hero_reward = int(env.state[hero_seat].reward or 0)
    steps = int(env.state[0].observation.step)
    return GameRecord(
        opp_name=opp_name, seed=seed, hero_seat=hero_seat,
        reward=hero_reward, steps=steps, wall_seconds=dt,
    )


def _build_task_list(
    weights: Dict[str, float], cfg: FitnessConfig,
) -> List[Tuple[Dict[str, float], OpponentSpec, str, int, int, float]]:
    """Flatten the (opp × game) cartesian product into a starmap-ready list.

    Factored out so ``evaluate`` and the parity test can share the exact
    same task generation — the seed formula lives here once.
    """
    tasks: List[Tuple[Dict[str, float], OpponentSpec, str, int, int, float]] = []
    for opp_idx, (opp_name, spec) in enumerate(cfg.opponents):
        for g in range(cfg.games_per_opponent):
            seed = cfg.seed_base + 997 * opp_idx + g
            hero_seat = g % 2
            tasks.append((weights, spec, opp_name, seed, hero_seat, cfg.step_timeout))
    return tasks


# ---- Public entry point ------------------------------------------------

def evaluate(weights: Dict[str, float], cfg: FitnessConfig) -> FitnessResult:
    """Run the full fitness pass; return aggregated stats + per-game records.

    If ``cfg.workers > 1`` the per-game work runs on a ``multiprocessing.Pool``.
    Windows spawn re-imports the workers' modules each call, which adds a
    ~1-2 s/worker cold-start; amortized across a 45-game trial it's
    negligible compared to the 5-10 s/game Kaggle cost.
    """
    tasks = _build_task_list(weights, cfg)

    # EvoTune: optionally swap `_score_target` in every worker (or in-
    # process for serial) before running any games. The evotune_bridge
    # import is deferred so importing fitness.py in a non-evotune context
    # doesn't require the sandbox / evotune module to be available.
    serial_installed = False
    if cfg.scorer_source is not None and cfg.workers <= 1:
        from orbitwars.tune import evotune_bridge as _eb
        _eb.install_evo_scorer(cfg.scorer_source)
        serial_installed = True

    try:
        if cfg.workers > 1:
            # `maxtasksperchild` controls child lifetime: None keeps the worker
            # alive for the full pool lifetime (default). For our 45-game trials
            # keeping the worker pool alive across the whole evaluate() call
            # avoids repeated spawn cost.
            pool_kwargs: Dict[str, Any] = {"processes": int(cfg.workers)}
            if cfg.scorer_source is not None:
                # Import the initializer by fully-qualified name so spawn
                # workers can re-import it cleanly on Windows.
                from orbitwars.tune.evotune_bridge import _worker_init_evo_scorer
                pool_kwargs["initializer"] = _worker_init_evo_scorer
                pool_kwargs["initargs"] = (cfg.scorer_source,)
            with mp.Pool(**pool_kwargs) as pool:
                records = pool.starmap(_run_one_game, tasks)
        else:
            records = [_run_one_game(*t) for t in tasks]
    finally:
        if serial_installed:
            from orbitwars.tune import evotune_bridge as _eb
            _eb.uninstall_evo_scorer()

    wins = 0.0
    hard_wins = 0.0
    for g in records:
        if g.reward == 1:
            wins += 1.0
            hard_wins += 1.0
        elif g.reward == 0:
            wins += 0.5

    n = max(1, len(records))
    result = FitnessResult(
        win_rate=wins / n,
        hard_win_rate=hard_wins / n,
        games=list(records),
    )
    return result


# ---- Convenience opponent pools ----

def starter_pool() -> List[Opponent]:
    """The W1/W2 primary target: just the engine's starter_agent.

    Kaggle ranks bots vs. each other, but starter_agent is our nearest-
    neighbor baseline and the first gate (plan: ≥95% win rate).
    """
    return [("starter", OpponentSpec(name="starter", kind="kaggle_builtin"))]


def w1_pool() -> List[Opponent]:
    """Full W1 pool: a spread of difficulties."""
    return [
        ("starter", OpponentSpec(name="starter", kind="kaggle_builtin")),
        ("random",  OpponentSpec(name="random",  kind="random", param=0)),
        ("noop",    OpponentSpec(name="noop",    kind="noop")),
    ]


def w2_pool() -> List[Opponent]:
    """W2 pool: starter + archetype portfolio.

    The first TuRBO run on `starter_pool` saw a ~16.7% uniform win rate
    across 25 trials — starter_agent is too strong a single opponent
    for small weight perturbations to separate from noise. This pool
    broadens evaluation with a spread of heuristic styles so good
    weights can beat *some* opponents even when they tie starter.

    Each archetype is a frozen HeuristicAgent with parameter overrides;
    no extra dependencies.
    """
    from orbitwars.opponent.archetypes import ARCHETYPE_NAMES
    pool: List[Opponent] = [
        ("starter", OpponentSpec(name="starter", kind="kaggle_builtin")),
        ("random",  OpponentSpec(name="random",  kind="random", param=0)),
    ]
    for name in ARCHETYPE_NAMES:
        pool.append((name, OpponentSpec(name=name, kind="archetype", param=name)))
    return pool
