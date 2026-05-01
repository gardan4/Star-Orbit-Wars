"""Round-robin between bundled submission files.

Usage:
    python -m tools.diag_bundle_round_robin \
        --bundles submissions/mcts_v2.py,submissions/mcts_v3.py,\
submissions/mcts_v4.py,submissions/mcts_v5.py \
        --games 6 --seed 42

Each bundle is a single-file module that defines an ``agent(obs, cfg)``
callable at module scope. We load each as an isolated module (fresh
module per game to defeat any accidental cross-game state) and run
them against each other via ``kaggle_environments``.

Output: Elo + head-to-head matrix + per-bot p50/p95 turn time.

Why this exists: submissions/mcts_v4.py scored 475 on the ladder after
v2.1 scored 539. We need a local signal on which bundle is strongest
before spending another Kaggle submission slot.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import math
import os
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from kaggle_environments import make


def _load_bundle_as_agent_factory(bundle_path: Path) -> Callable[[], Callable]:
    """Return a zero-arg factory that freshly imports the bundle and
    returns its ``agent`` callable.

    Fresh import per match (unique module name) — bundles tend to cache
    agent state at module scope (this was a noted failure mode during
    the v3/v4 iterations; caching across games would skew H2H).
    """
    def factory() -> Callable:
        # Unique module name so each import is a clean slate.
        mod_name = f"_bundle_{bundle_path.stem}_{uuid.uuid4().hex[:8]}"
        spec = importlib.util.spec_from_file_location(mod_name, bundle_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load bundle {bundle_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        agent_fn = getattr(mod, "agent", None)
        if not callable(agent_fn):
            raise RuntimeError(f"Bundle {bundle_path} has no 'agent' callable")
        return agent_fn
    return factory


def _timed(turn_times: List[float], fn: Callable) -> Callable:
    def wrapped(obs, cfg=None):
        t0 = time.perf_counter()
        try:
            result = fn(obs, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"  [agent error] {type(e).__name__}: {e}", flush=True)
            result = []
        turn_times.append((time.perf_counter() - t0) * 1000.0)
        return result
    return wrapped


def play_one(
    a_factory: Callable,
    b_factory: Callable,
    seed: int,
    step_timeout: float = 1.0,
) -> Tuple[List[int], int, List[List[float]]]:
    """Play a single game; return (rewards, steps, per-player turn times)."""
    import random as _pyr

    _pyr.seed(seed)
    cfg = {"actTimeout": step_timeout}
    env = make("orbit_wars", configuration=cfg, debug=False)

    turn_times_0: List[float] = []
    turn_times_1: List[float] = []
    agent_a = _timed(turn_times_0, a_factory())
    agent_b = _timed(turn_times_1, b_factory())

    env.run([agent_a, agent_b])
    rewards = [int(s.reward if s.reward is not None else 0) for s in env.state]
    steps = int(env.state[0].observation.step)
    return rewards, steps, [turn_times_0, turn_times_1]


def _elo_update(ra: float, rb: float, score_a: float, k: float = 16.0) -> Tuple[float, float]:
    ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
    return ra + k * (score_a - ea), rb + k * ((1 - score_a) - (1 - ea))


def _pct(vals: List[float], p: float) -> float:
    if not vals:
        return float("nan")
    vals = sorted(vals)
    k = max(0, min(len(vals) - 1, int(math.ceil(p / 100 * len(vals))) - 1))
    return vals[k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", required=True,
                    help="comma-separated paths to bundled .py submission files")
    ap.add_argument("--games", type=int, default=6,
                    help="games per pair (seats alternated)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--step_timeout", type=float, default=1.0)
    args = ap.parse_args()

    bundles = [Path(p.strip()) for p in args.bundles.split(",")]
    for b in bundles:
        if not b.exists():
            raise SystemExit(f"Bundle not found: {b}")

    names = [b.stem for b in bundles]
    factories = {n: _load_bundle_as_agent_factory(b) for n, b in zip(names, bundles)}

    elos: Dict[str, float] = {n: 1500.0 for n in names}
    pair_wins: Dict[Tuple[str, str], List[int]] = {
        (a, b): [0, 0, 0] for a, b in itertools.permutations(names, 2)
    }
    turn_times_all: Dict[str, List[float]] = {n: [] for n in names}

    t_start = time.perf_counter()
    total = 0
    print(f"Round-robin: {len(names)} bundles, {args.games} games/pair, "
          f"timeout={args.step_timeout}s", flush=True)

    for a, b in itertools.combinations(names, 2):
        print(f"\n=== {a} vs {b} ({args.games} games) ===", flush=True)
        for g in range(args.games):
            swap = g % 2 == 1
            first, second = (b, a) if swap else (a, b)
            rewards, steps, tt = play_one(
                factories[first], factories[second],
                seed=args.seed + 1000 * total,
                step_timeout=args.step_timeout,
            )
            total += 1
            # Map back to a/b perspective.
            a_r = rewards[1] if swap else rewards[0]
            b_r = rewards[0] if swap else rewards[1]

            if a_r == 1 and b_r == -1:
                pair_wins[(a, b)][0] += 1; pair_wins[(b, a)][1] += 1
                elos[a], elos[b] = _elo_update(elos[a], elos[b], 1.0)
                outcome = "WIN"
            elif a_r == -1 and b_r == 1:
                pair_wins[(a, b)][1] += 1; pair_wins[(b, a)][0] += 1
                elos[a], elos[b] = _elo_update(elos[a], elos[b], 0.0)
                outcome = "LOSS"
            else:
                pair_wins[(a, b)][2] += 1; pair_wins[(b, a)][2] += 1
                elos[a], elos[b] = _elo_update(elos[a], elos[b], 0.5)
                outcome = "TIE"

            for name, times in zip([first, second], tt):
                turn_times_all[name].extend(times)

            seating = f"{first}=seat0, {second}=seat1"
            p95_a = _pct(tt[1 if swap else 0], 95)
            p95_b = _pct(tt[0 if swap else 1], 95)
            print(f"  g{g}: seed={args.seed + 1000 * (total - 1)} {seating} "
                  f"steps={steps} {a} {outcome} "
                  f"[p95 {a}={p95_a:.0f}ms {b}={p95_b:.0f}ms]", flush=True)

    wall = time.perf_counter() - t_start
    print(f"\n--- Summary ({total} games, {wall:.1f}s) ---", flush=True)

    print("\nElo ranking:")
    for n in sorted(names, key=lambda n: -elos[n]):
        p50 = statistics.median(turn_times_all[n]) if turn_times_all[n] else float("nan")
        p95 = _pct(turn_times_all[n], 95)
        print(f"  {n:<20} elo={elos[n]:7.1f}  p50={p50:6.0f}ms  p95={p95:6.0f}ms",
              flush=True)

    print("\nHead-to-head (W-L-T, row vs col):")
    col_hdr = "  " + " " * 20 + "".join(f"{c:>15}" for c in names)
    print(col_hdr)
    for a in names:
        row = f"  {a:<20}"
        for b in names:
            if a == b:
                row += f"{'—':>15}"
            else:
                w, l, t = pair_wins[(a, b)]
                row += f"{w}-{l}-{t:>13}"
        print(row, flush=True)


if __name__ == "__main__":
    main()
