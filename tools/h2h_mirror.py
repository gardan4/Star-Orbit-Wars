"""Fair head-to-head between two bundles via mirror matches.

The companion tool ``diag_bundle_round_robin.py`` alternates seats
(``swap = g % 2 == 1``) but uses a *different* seed per game. Under
anchor-locked MCTS the wire actions are ~identical between bundles, so
the test ends up measuring "which seat is favored in each individual
seed" — and our default seed sequence (42, 1042, 2042, ...) happens to
favor whichever bundle plays seat-0 in even seeds + seat-1 in odd seeds
(the harness's first-bundle position). This is the +51.8 Elo phantom
signal documented in STATUS.md (2026-04-26).

This tool runs **mirror matches**: for each seed, BOTH (a=seat0,
b=seat1) and (a=seat1, b=seat0) are played. The seat advantage cancels
exactly. With ``--games N``, you get ``2N`` actual games per pair
(N seeds × 2 mirrors).

Usage:
    python -m tools.h2h_mirror \\
        --bundles submissions/foo.py,submissions/bar.py \\
        --games 20 --seed 42 --step_timeout 1.0
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import math
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from kaggle_environments import make


def _load_bundle_as_agent_factory(bundle_path: Path) -> Callable[[], Callable]:
    def factory() -> Callable:
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
    seat0_factory: Callable,
    seat1_factory: Callable,
    seed: int,
    step_timeout: float = 1.0,
) -> Tuple[List[int], int, List[List[float]]]:
    import random as _pyr
    # Reset ALL global RNG sources before factory calls so cross-call
    # state contamination doesn't leak into mirror halves. The bundle's
    # import-time code can advance torch/numpy state via NN init even
    # when load_state_dict overwrites the result; without per-game reset
    # the second mirror sees a different starting RNG state.
    _pyr.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    cfg = {"actTimeout": step_timeout}
    env = make("orbit_wars", configuration=cfg, debug=False)
    tt0: List[float] = []
    tt1: List[float] = []
    agent0 = _timed(tt0, seat0_factory())
    agent1 = _timed(tt1, seat1_factory())
    env.run([agent0, agent1])
    rewards = [int(s.reward if s.reward is not None else 0) for s in env.state]
    steps = int(env.state[0].observation.step)
    return rewards, steps, [tt0, tt1]


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
    ap.add_argument("--games", type=int, default=20,
                    help="number of seeds (each plays 2 mirror matches → 2× games total)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--step_timeout", type=float, default=1.0)
    args = ap.parse_args()

    bundles = [Path(p.strip()) for p in args.bundles.split(",")]
    if len(bundles) != 2:
        raise SystemExit("h2h_mirror requires exactly 2 bundles")
    for b in bundles:
        if not b.exists():
            raise SystemExit(f"Bundle not found: {b}")

    name_a, name_b = bundles[0].stem, bundles[1].stem
    fac_a = _load_bundle_as_agent_factory(bundles[0])
    fac_b = _load_bundle_as_agent_factory(bundles[1])

    elo_a = elo_b = 1500.0
    a_w = a_l = a_t = 0
    tt_a: List[float] = []
    tt_b: List[float] = []
    pair_outcomes: List[Tuple[int, str, str, int, int]] = []  # (seed, mirror, outcome, steps, p95s)

    print(f"Mirror H2H: {name_a} vs {name_b}, {args.games} seeds × 2 mirrors = "
          f"{2 * args.games} games, timeout={args.step_timeout}s", flush=True)
    t_start = time.perf_counter()

    for g in range(args.games):
        seed = args.seed + 1000 * g

        for mirror_idx in range(2):
            # mirror 0: a=seat0, b=seat1
            # mirror 1: b=seat0, a=seat1
            if mirror_idx == 0:
                seat0_fac, seat1_fac = fac_a, fac_b
                rewards, steps, tts = play_one(seat0_fac, seat1_fac, seed,
                                                args.step_timeout)
                a_seat = 0
                a_reward = rewards[0]
                tt_a.extend(tts[0]); tt_b.extend(tts[1])
            else:
                seat0_fac, seat1_fac = fac_b, fac_a
                rewards, steps, tts = play_one(seat0_fac, seat1_fac, seed,
                                                args.step_timeout)
                a_seat = 1
                a_reward = rewards[1]
                tt_b.extend(tts[0]); tt_a.extend(tts[1])

            if a_reward == 1:
                a_w += 1
                elo_a, elo_b = _elo_update(elo_a, elo_b, 1.0)
                outcome = "WIN"
            elif a_reward == -1:
                a_l += 1
                elo_a, elo_b = _elo_update(elo_a, elo_b, 0.0)
                outcome = "LOSS"
            else:
                a_t += 1
                elo_a, elo_b = _elo_update(elo_a, elo_b, 0.5)
                outcome = "TIE"

            pair_outcomes.append((seed, "AB" if mirror_idx == 0 else "BA",
                                   outcome, steps, 0))
            mirror_label = f"a=seat{a_seat}"
            print(f"  s{seed:6d} m{mirror_idx} {mirror_label} steps={steps} "
                  f"{name_a} {outcome}", flush=True)

    wall = time.perf_counter() - t_start
    total_games = 2 * args.games
    a_score = a_w + 0.5 * a_t
    a_wr = a_score / total_games

    print(f"\n--- Summary ({total_games} games, {wall:.1f}s) ---", flush=True)
    print(f"\n{name_a:25s} W-L-T = {a_w}-{a_l}-{a_t}  wr={a_wr:.3f}  "
          f"elo={elo_a:.1f}  p50={_pct(tt_a, 50):5.0f}ms  p95={_pct(tt_a, 95):5.0f}ms")
    print(f"{name_b:25s} W-L-T = {a_l}-{a_w}-{a_t}  wr={1.0 - a_wr:.3f}  "
          f"elo={elo_b:.1f}  p50={_pct(tt_b, 50):5.0f}ms  p95={_pct(tt_b, 95):5.0f}ms")

    # Deltas — proper Elo-from-wr formula gives an unbiased estimator.
    if 0.0 < a_wr < 1.0:
        elo_delta = -400.0 * math.log10(1.0 / a_wr - 1.0)
    elif a_wr == 0.0:
        elo_delta = -800.0
    else:
        elo_delta = 800.0
    se_wr = math.sqrt(a_wr * (1 - a_wr) / total_games)
    print(f"\nMirror-fair Elo delta ({name_a} - {name_b}): {elo_delta:+.1f}  "
          f"(wr={a_wr:.3f} ± {se_wr:.3f})", flush=True)


if __name__ == "__main__":
    main()
