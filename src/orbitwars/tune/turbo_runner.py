"""TuRBO / random-search tuning loop for HeuristicAgent weights.

Usage:
    python -m orbitwars.tune.turbo_runner \
        --strategy ax --n-trials 20 --pool starter \
        --games-per-opp 10 --out runs/ax_001.json

Strategies:
    random  — uniform over param bounds; baseline and plumbing sanity check.
    ax      — Ax-platform Bayesian optimization (qNEI). Acts as our TuRBO
              stand-in until we plumb in explicit trust regions.

The loop is sequential (one fitness eval at a time). Each trial flushes the
full run record to disk so long runs are inspectable / resumable. A 20-trial
run with 10 games/trial against starter is ~50-80 min wall clock.

Design:
  * PARAM_BOUNDS is the single source of truth for the search space.
    Adding a heuristic weight means: (1) add to HEURISTIC_WEIGHTS, (2) add
    to PARAM_BOUNDS (or leave fixed), (3) done.
  * Baseline defaults from HEURISTIC_WEIGHTS fill in any weight NOT in
    PARAM_BOUNDS — bounded-only params are "active"; the rest stay at their
    hand-tuned values.
  * Strategy is swappable: add a new class implementing TuningStrategy.
"""
from __future__ import annotations

import argparse
import json
import math
import random as _r
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from orbitwars.bots.heuristic import HEURISTIC_WEIGHTS
from orbitwars.tune.fitness import (
    FitnessConfig,
    FitnessResult,
    evaluate,
    starter_pool,
    w1_pool,
    w2_pool,
)


# ---- Parameter space ----
#
# Bounds chosen to cover a wide plausible range around current defaults.
# TuRBO / BO handles scale mismatches fine (GPs normalize internally), so we
# don't bother with log-scale transforms here.
#
# A few bounds are deliberately *tight* to avoid pathological no-launch
# regimes that drag every game to step 500 (the first tuning attempt hit a
# Sobol point with min_launch_size=38 + keep_reserve_ships=42 → the bot
# needed 80 ships to ever fire, games burned through fitness budget at
# >10 min/game). The compound constraint is: min_launch_size +
# keep_reserve_ships ≲ 40, reachable from turn-~30 production.
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "w_production":            (0.5,  20.0),
    "w_ships_cost":            (0.0,   1.0),
    "w_distance_cost":         (0.0,   0.5),
    "w_travel_cost":           (0.0,   3.0),
    "mult_neutral":            (0.1,   3.0),
    "mult_enemy":              (0.1,   5.0),
    "mult_comet":              (0.1,   5.0),
    "mult_reinforce_ally":     (0.0,   2.0),
    "ships_safety_margin":     (0.0,   5.0),   # was 10 (overkill)
    "min_launch_size":         (5.0,  30.0),   # was 60 (pathological)
    "max_launch_fraction":     (0.3,   1.0),   # was 0.1 (stalls play)
    "expand_cooldown_turns":   (0.0,   5.0),   # was 20 (throttles dev)
    "keep_reserve_ships":      (0.0,  15.0),   # was 50 (pathological)
    "agg_early_game":          (0.5,   3.0),
    "early_game_cutoff_turn":  (0.0, 200.0),   # was 300
    "sun_avoidance_epsilon":   (0.0,   0.2),
    "comet_max_time_mismatch": (0.0,   5.0),
    "expand_bias":             (0.0,   2.0),
}


# ---- Strategy interface ----

class TuningStrategy:
    """Base class for tuning strategies. Override next_point / observe.

    `observe(point, value, noise_se)` reports an outcome. `noise_se` is the
    estimated standard error of `value` (binomial SE = sqrt(p(1-p)/n) for
    win-rate fitness). Strategies like AxTurbo feed it to the GP's noise
    model; strategies like RandomSearch ignore it.
    """

    def next_point(self) -> Dict[str, float]:
        raise NotImplementedError

    def observe(
        self,
        point: Dict[str, float],
        value: float,
        noise_se: Optional[float] = None,
    ) -> None:
        raise NotImplementedError


class RandomSearch(TuningStrategy):
    """Uniform random baseline — zero dependencies, our sanity check."""

    def __init__(self, bounds: Dict[str, Tuple[float, float]], seed: int = 0):
        self.bounds = bounds
        self.rng = _r.Random(seed)

    def next_point(self) -> Dict[str, float]:
        return {
            k: self.rng.uniform(lo, hi)
            for k, (lo, hi) in self.bounds.items()
        }

    def observe(
        self,
        point: Dict[str, float],
        value: float,
        noise_se: Optional[float] = None,
    ) -> None:
        return  # no memory


class AxTurbo(TuningStrategy):
    """Ax Bayesian optimization (service API, qNEI) as our TuRBO stand-in.

    Until we plumb in real trust-region heuristics, Ax's default Sobol →
    GP+qNEI pipeline is a sensible 20D optimizer. On small budgets (≤30
    trials) qNEI is a known-reasonable acquisition; at larger budgets we'd
    switch in `SaasboStrategy` (high-dim BO) or wrap it with the BoTorch
    TuRBO state class.

    Noise is declared as an SE on each observation; BO handles the rest.
    `default_noise_se` is the fallback when observe() is called without
    noise_se (e.g. from unit tests).
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        seed: int = 0,
        default_noise_se: float = 0.15,
    ):
        try:
            from ax.service.ax_client import AxClient
            from ax.service.utils.instantiation import ObjectiveProperties
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "Ax not installed. Install with: pip install ax-platform"
            ) from e

        self.default_noise_se = default_noise_se
        self.client = AxClient(random_seed=seed)
        self.client.create_experiment(
            name="orbitwars_heuristic_tune",
            parameters=[
                {
                    "name": name,
                    "type": "range",
                    "bounds": [float(lo), float(hi)],
                    "value_type": "float",
                }
                for name, (lo, hi) in bounds.items()
            ],
            objectives={
                "win_rate": ObjectiveProperties(minimize=False),
            },
        )
        # Ax returns trial_index when we sample; we need to remember it to
        # complete the trial with observed data.
        self._pending: Dict[Tuple[Tuple[str, float], ...], int] = {}

    @staticmethod
    def _key(point: Dict[str, float]) -> Tuple[Tuple[str, float], ...]:
        return tuple(sorted(point.items()))

    def next_point(self) -> Dict[str, float]:
        params, trial_index = self.client.get_next_trial()
        self._pending[self._key(params)] = trial_index
        return dict(params)

    def observe(
        self,
        point: Dict[str, float],
        value: float,
        noise_se: Optional[float] = None,
    ) -> None:
        trial_index = self._pending.pop(self._key(point))
        se = float(noise_se) if noise_se is not None else self.default_noise_se
        self.client.complete_trial(
            trial_index=trial_index,
            raw_data={"win_rate": (float(value), se)},
        )


# ---- Orchestration ----

@dataclass
class TrialRecord:
    trial: int
    weights: Dict[str, float]
    win_rate: float
    hard_win_rate: float
    n_games: int
    fitness_wall_seconds: float
    by_opponent: Dict[str, Tuple[float, int]] = field(default_factory=dict)

    def as_json(self) -> Dict[str, Any]:
        d = asdict(self)
        # by_opponent has tuple values — normalize for JSON.
        d["by_opponent"] = {k: list(v) for k, v in self.by_opponent.items()}
        return d


@dataclass
class RunResult:
    strategy: str
    n_trials: int
    trials: List[TrialRecord] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def best(self) -> Optional[TrialRecord]:
        return max(self.trials, key=lambda t: t.win_rate) if self.trials else None

    def to_json(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "n_trials": self.n_trials,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "best_index": self.best.trial if self.best else None,
            "best_win_rate": self.best.win_rate if self.best else None,
            "trials": [r.as_json() for r in self.trials],
        }


def run(
    strategy: TuningStrategy,
    fitness_cfg: FitnessConfig,
    n_trials: int,
    out_path: Optional[Path] = None,
    verbose: bool = True,
) -> RunResult:
    """Core tuning loop. Flushes per-trial JSON to disk if out_path set."""
    res = RunResult(
        strategy=type(strategy).__name__,
        n_trials=n_trials,
        started_at=time.time(),
    )

    baseline = dict(HEURISTIC_WEIGHTS)

    for t in range(n_trials):
        t0 = time.perf_counter()
        sampled = strategy.next_point()
        # Merge sampled over the full weight dict so heuristic never sees a
        # missing key (silent KeyError → no-op, the W1 bug we fixed).
        weights = dict(baseline)
        weights.update(sampled)

        fit: FitnessResult = evaluate(weights, fitness_cfg)
        dt = time.perf_counter() - t0

        # Binomial SE of win_rate at this n. Floor at 1e-3 to avoid
        # perfectly-confident observations at rare 0/N or N/N extremes.
        p = fit.win_rate
        n = max(1, fit.n_games)
        noise_se = max(1e-3, math.sqrt(max(0.0, p * (1.0 - p)) / n))
        strategy.observe(sampled, fit.win_rate, noise_se=noise_se)

        rec = TrialRecord(
            trial=t,
            weights=weights,
            win_rate=fit.win_rate,
            hard_win_rate=fit.hard_win_rate,
            n_games=fit.n_games,
            fitness_wall_seconds=dt,
            by_opponent=fit.by_opponent(),
        )
        res.trials.append(rec)
        res.finished_at = time.time()

        if verbose:
            best = res.best
            best_str = f"{best.win_rate:.3f}" if best else "n/a"
            print(
                f"[t={t:3d}] win={fit.win_rate:.3f} hard={fit.hard_win_rate:.3f} "
                f"n={fit.n_games} {dt:5.1f}s best_so_far={best_str}",
                flush=True,
            )

        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(res.to_json(), indent=2))

    return res


# ---- CLI ----

def _build_fitness_cfg(
    pool_name: str,
    games_per_opp: int,
    step_timeout: float,
    seed_base: int,
    workers: int = 1,
) -> FitnessConfig:
    if pool_name == "starter":
        opps = starter_pool()
    elif pool_name == "w1":
        opps = w1_pool()
    elif pool_name == "w2":
        opps = w2_pool()
    else:
        raise ValueError(f"unknown pool: {pool_name}")
    return FitnessConfig(
        opponents=opps,
        games_per_opponent=games_per_opp,
        step_timeout=step_timeout,
        seed_base=seed_base,
        workers=workers,
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="TuRBO/random heuristic tuner")
    ap.add_argument("--strategy", choices=["random", "ax"], default="random")
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument(
        "--pool", choices=["starter", "w1", "w2"], default="starter",
        help="w2 = starter + random + 7 archetypes (richer signal than starter-only)",
    )
    ap.add_argument("--games-per-opp", type=int, default=10)
    ap.add_argument("--step-timeout", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers for per-game fitness eval. 1=serial (legacy). "
             "7-8 gives ~7x speedup on an 8-core CPU.",
    )
    args = ap.parse_args(argv)

    out_path = Path(args.out) if args.out else None
    fitness_cfg = _build_fitness_cfg(
        args.pool, args.games_per_opp, args.step_timeout, args.seed,
        workers=args.workers,
    )

    if args.strategy == "random":
        strat: TuningStrategy = RandomSearch(PARAM_BOUNDS, seed=args.seed)
    else:
        strat = AxTurbo(PARAM_BOUNDS, seed=args.seed)

    res = run(strat, fitness_cfg, n_trials=args.n_trials, out_path=out_path)
    best = res.best
    print(f"\n=== Best of {len(res.trials)} trials ===")
    if best:
        print(f"win_rate={best.win_rate:.3f} hard={best.hard_win_rate:.3f}")
        print("weights:")
        for k in sorted(best.weights.keys()):
            print(f"  {k:<28} = {best.weights[k]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
