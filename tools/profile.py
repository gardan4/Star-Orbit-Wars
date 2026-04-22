"""Microbenchmarks for the orbit_wars hot paths.

Tracks speed of:
  * FastEngine.step vs official env.step (steps/sec)
  * fleet_speed batched vs per-fleet
  * orbiting_intercept Newton solver
  * Heuristic agent turn time p50/p95

Usage:
    python -m tools.profile --quick          # ~5s
    python -m tools.profile --full           # ~30s (all benches)
    python -m tools.profile --engine-only    # just engine step rate
"""
from __future__ import annotations

import argparse
import gc
import math
import random
import statistics
import time
from typing import Dict, List, Optional

import numpy as np


def _time_ms(fn, iters: int = 1) -> List[float]:
    """Run fn() `iters` times, return per-call millis. GC disabled inside."""
    gc.collect()
    gc.disable()
    try:
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)
        return times
    finally:
        gc.enable()


def _p(vals: List[float], p: float) -> float:
    if not vals:
        return float("nan")
    vals = sorted(vals)
    k = max(0, min(len(vals) - 1, int(math.ceil(p / 100 * len(vals))) - 1))
    return vals[k]


def bench_engine_step_rate(turns: int = 500, seeds: int = 5) -> Dict[str, float]:
    """Compare FastEngine.step vs official env.step over full games."""
    from kaggle_environments import make
    from orbitwars.engine.fast_engine import FastEngine

    official_times = []
    fast_times = []

    for seed in range(seeds):
        # Official
        random.seed(seed)
        env = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
        env.reset(num_agents=2)
        env.step([[], []])  # init
        t0 = time.perf_counter()
        for _ in range(turns - 1):
            env.step([[], []])
            if env.done:
                break
        official_times.append(time.perf_counter() - t0)

        # Fast (from the same init state)
        random.seed(seed)
        env2 = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
        env2.reset(num_agents=2)
        env2.step([[], []])
        fast = FastEngine.from_official_obs(env2.state[0].observation, num_agents=2)
        t0 = time.perf_counter()
        for _ in range(turns - 1):
            fast.step([[], []])
            if fast.done:
                break
        fast_times.append(time.perf_counter() - t0)

    mean_official = statistics.mean(official_times)
    mean_fast = statistics.mean(fast_times)
    return {
        "official_s": mean_official,
        "fast_s": mean_fast,
        "speedup": mean_official / mean_fast if mean_fast > 0 else float("nan"),
        "fast_steps_per_s": (turns - 1) / mean_fast if mean_fast > 0 else float("nan"),
        "official_steps_per_s": (turns - 1) / mean_official if mean_official > 0 else float("nan"),
    }


def bench_fleet_speed() -> Dict[str, float]:
    """Batched numpy vs Python per-call fleet_speed."""
    from orbitwars.engine.fast_engine import _fleet_speed_batched
    from orbitwars.engine.intercept import fleet_speed

    N = 1000
    ships = np.random.randint(1, 1000, size=N)

    t0 = time.perf_counter()
    _ = _fleet_speed_batched(ships, 6.0)
    t_np = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    _ = [fleet_speed(int(s)) for s in ships]
    t_py = (time.perf_counter() - t0) * 1000.0

    return {
        "numpy_ms": t_np,
        "python_ms": t_py,
        "speedup": t_py / t_np if t_np > 0 else float("nan"),
        "N": N,
    }


def bench_orbiting_intercept() -> Dict[str, float]:
    """Time Newton solver over a batch of targets."""
    from orbitwars.engine.intercept import OrbitingTarget, orbiting_intercept

    N = 1000
    rng = random.Random(0)
    src = (10.0, 10.0)
    total_iters = 0

    t0 = time.perf_counter()
    for _ in range(N):
        r = rng.uniform(15.0, 45.0)
        ia = rng.uniform(-math.pi, math.pi)
        omega = rng.uniform(0.025, 0.05)
        step = rng.randint(0, 400)
        ot = OrbitingTarget(
            orbital_radius=r, initial_angle=ia,
            angular_velocity=omega, current_step=step,
        )
        _, _, iters = orbiting_intercept(src, ot, ships=50)
        total_iters += iters
    t_total = (time.perf_counter() - t0) * 1000.0
    return {
        "total_ms": t_total,
        "per_call_us": t_total * 1000.0 / N,
        "avg_iters": total_iters / N,
        "N": N,
    }


def bench_heuristic_turn_time(games: int = 3, turns: int = 200) -> Dict[str, float]:
    from kaggle_environments import make
    from orbitwars.bots.heuristic import HeuristicAgent

    per_turn_ms: List[float] = []
    for seed in range(games):
        random.seed(seed)
        env = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
        env.reset(num_agents=2)
        env.step([[], []])  # init
        agent = HeuristicAgent()
        kag = agent.as_kaggle_agent()
        for _ in range(turns - 1):
            obs = env.state[0].observation
            t0 = time.perf_counter()
            kag(obs)
            per_turn_ms.append((time.perf_counter() - t0) * 1000.0)
            env.step([[], []])
            if env.done:
                break

    return {
        "turns_measured": len(per_turn_ms),
        "p50_ms": statistics.median(per_turn_ms) if per_turn_ms else float("nan"),
        "p95_ms": _p(per_turn_ms, 95),
        "p99_ms": _p(per_turn_ms, 99),
        "mean_ms": statistics.mean(per_turn_ms) if per_turn_ms else float("nan"),
        "max_ms": max(per_turn_ms) if per_turn_ms else float("nan"),
    }


def _print_block(title: str, data: Dict[str, float]) -> None:
    print(f"\n--- {title} ---")
    for k, v in data.items():
        if isinstance(v, float):
            if abs(v) < 0.01 and v != 0:
                print(f"  {k:<24} {v:.4e}")
            else:
                print(f"  {k:<24} {v:.3f}")
        else:
            print(f"  {k:<24} {v}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Fast subset only")
    ap.add_argument("--full", action="store_true", help="All benches, more seeds")
    ap.add_argument("--engine-only", action="store_true")
    args = ap.parse_args()

    if args.engine_only:
        _print_block("engine_step_rate", bench_engine_step_rate(turns=500, seeds=3))
        return 0

    if args.quick:
        _print_block("fleet_speed", bench_fleet_speed())
        _print_block("orbiting_intercept", bench_orbiting_intercept())
        _print_block("engine_step_rate", bench_engine_step_rate(turns=200, seeds=2))
        return 0

    # Default = full
    _print_block("fleet_speed", bench_fleet_speed())
    _print_block("orbiting_intercept", bench_orbiting_intercept())
    _print_block("engine_step_rate", bench_engine_step_rate(turns=500, seeds=5))
    _print_block("heuristic_turn_time", bench_heuristic_turn_time(games=3, turns=200))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
