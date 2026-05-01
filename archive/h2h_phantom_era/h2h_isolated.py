"""Subprocess-isolated mirror H2H — true determinism across games.

Each game runs in its own fresh Python subprocess via
``tools/_h2h_one_game.py``. This eliminates ALL forms of cross-game
state contamination (numpy/torch RNG drift, kaggle_environments cached
state, module-level singletons). The `tools/h2h_mirror.py` companion
runs all games in the same process and was found to produce phantom
non-cancellations for byte-identical bundles when wall-clock-dependent
MCTS decisions or shared module state leak between games.

For each seed, runs both mirror halves: (a=seat0, b=seat1) and
(b=seat0, a=seat1). Identical bundles MUST cancel to 0.500 wr — if they
don't, the harness has a bug.

Usage:
    python -m tools.h2h_isolated \\
        --bundles A.py,B.py --games 20 --seed 42 --step_timeout 1.0
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple


_HELPER = Path(__file__).parent / "_h2h_one_game.py"


def _run_one(
    bundle_seat0: Path,
    bundle_seat1: Path,
    seed: int,
    step_timeout: float,
    venv_python: Path,
) -> Tuple[int, int, int, float, float]:
    """Run one game in a fresh subprocess. Returns (s0_reward, s1_reward,
    steps, p95_s0_ms, p95_s1_ms). Raises on failure."""
    cmd = [
        str(venv_python),
        "-u",
        str(_HELPER),
        str(bundle_seat0),
        str(bundle_seat1),
        str(seed),
        str(step_timeout),
    ]
    res = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed (rc={res.returncode}): "
            f"stderr={res.stderr[-500:]}"
        )
    # Parse last RESULT line from stdout.
    last = None
    for line in res.stdout.splitlines():
        if line.startswith("RESULT "):
            last = line
    if last is None:
        raise RuntimeError(f"No RESULT line in stdout: {res.stdout[-500:]}")
    parts = last.split()
    return (int(parts[1]), int(parts[2]), int(parts[3]),
            float(parts[4]), float(parts[5]))


def _elo_update(ra: float, rb: float, score_a: float, k: float = 16.0) -> Tuple[float, float]:
    ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
    return ra + k * (score_a - ea), rb + k * ((1 - score_a) - (1 - ea))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", required=True,
                    help="comma-separated paths to two bundle .py files")
    ap.add_argument("--games", type=int, default=20,
                    help="number of seeds (each plays 2 mirror matches → 2× games total)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--step_timeout", type=float, default=1.0)
    ap.add_argument("--venv-python", type=str, default="",
                    help="path to python.exe (defaults to current sys.executable)")
    args = ap.parse_args()

    bundles = [Path(p.strip()) for p in args.bundles.split(",")]
    if len(bundles) != 2:
        raise SystemExit("h2h_isolated requires exactly 2 bundles")
    for b in bundles:
        if not b.exists():
            raise SystemExit(f"Bundle not found: {b}")
    venv_python = Path(args.venv_python) if args.venv_python else Path(sys.executable)

    name_a, name_b = bundles[0].stem, bundles[1].stem
    elo_a = elo_b = 1500.0
    a_w = a_l = a_t = 0
    p95_a_all = []
    p95_b_all = []

    print(f"Isolated mirror H2H: {name_a} vs {name_b}, "
          f"{args.games} seeds × 2 mirrors = {2 * args.games} games  "
          f"timeout={args.step_timeout}s  venv={venv_python.name}", flush=True)
    t_start = time.perf_counter()

    for g in range(args.games):
        seed = args.seed + 1000 * g
        for mirror_idx in range(2):
            if mirror_idx == 0:
                seat0_b, seat1_b = bundles[0], bundles[1]
            else:
                seat0_b, seat1_b = bundles[1], bundles[0]

            try:
                s0_r, s1_r, steps, p95_s0, p95_s1 = _run_one(
                    seat0_b, seat1_b, seed, args.step_timeout, venv_python,
                )
            except Exception as e:
                print(f"  s {seed:6d} m{mirror_idx} ERROR: {e}", flush=True)
                continue

            # Map back to a's perspective (a is bundles[0]).
            if mirror_idx == 0:
                a_r = s0_r; p95_a = p95_s0; p95_b = p95_s1
            else:
                a_r = s1_r; p95_a = p95_s1; p95_b = p95_s0
            p95_a_all.append(p95_a); p95_b_all.append(p95_b)

            if a_r == 1:
                a_w += 1
                elo_a, elo_b = _elo_update(elo_a, elo_b, 1.0)
                outcome = "WIN"
            elif a_r == -1:
                a_l += 1
                elo_a, elo_b = _elo_update(elo_a, elo_b, 0.0)
                outcome = "LOSS"
            else:
                a_t += 1
                elo_a, elo_b = _elo_update(elo_a, elo_b, 0.5)
                outcome = "TIE"
            seat = "seat0" if mirror_idx == 0 else "seat1"
            print(f"  s {seed:6d} m{mirror_idx} a={seat} steps={steps} "
                  f"{name_a} {outcome} [p95 a={p95_a:.0f}ms b={p95_b:.0f}ms]",
                  flush=True)

    wall = time.perf_counter() - t_start
    total = a_w + a_l + a_t
    a_score = a_w + 0.5 * a_t
    a_wr = a_score / max(total, 1)
    print(f"\n--- Summary ({total} games, {wall:.0f}s) ---", flush=True)
    print(f"{name_a:25s} W-L-T = {a_w}-{a_l}-{a_t}  wr={a_wr:.3f}  elo={elo_a:.1f}",
          flush=True)
    print(f"{name_b:25s} W-L-T = {a_l}-{a_w}-{a_t}  wr={1.0 - a_wr:.3f}  elo={elo_b:.1f}",
          flush=True)
    if 0.0 < a_wr < 1.0:
        elo_delta = -400.0 * math.log10(1.0 / a_wr - 1.0)
    elif a_wr == 0.0:
        elo_delta = -800.0
    else:
        elo_delta = 800.0
    se_wr = math.sqrt(a_wr * (1 - a_wr) / max(total, 1))
    print(f"\nIsolated-fair Elo delta ({name_a} - {name_b}): {elo_delta:+.1f}  "
          f"(wr={a_wr:.3f} ± {se_wr:.3f})", flush=True)


if __name__ == "__main__":
    main()
