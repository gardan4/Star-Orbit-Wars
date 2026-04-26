"""Helper: run ONE game (in its own fresh subprocess) and print result.

Used by ``tools/h2h_isolated.py`` which spawns this script per-game so
state cannot leak between games (no RNG drift, no kaggle_environments
shared state, no cached imports).

Args (positional):
  bundle_path_seat0 bundle_path_seat1 seed step_timeout

Output (stdout, last line):
  ``RESULT seat0_reward seat1_reward steps p95_seat0_ms p95_seat1_ms``
"""
from __future__ import annotations

import importlib.util
import math
import sys
import time
import uuid
from pathlib import Path

# Force deterministic-ish RNG in this fresh process.
import random as _pyr
try:
    import numpy as _np
except Exception:
    _np = None
try:
    import torch as _torch
except Exception:
    _torch = None


def _load_agent(bundle_path: Path):
    mod_name = f"_h2h_bundle_{bundle_path.stem}_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(mod_name, bundle_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def _timed(times, fn):
    def wrapped(obs, cfg=None):
        t0 = time.perf_counter()
        try:
            r = fn(obs, cfg)
        except Exception:
            r = []
        times.append((time.perf_counter() - t0) * 1000.0)
        return r
    return wrapped


def _pct(vals, p):
    if not vals:
        return float("nan")
    vals = sorted(vals)
    k = max(0, min(len(vals) - 1, int(math.ceil(p / 100 * len(vals))) - 1))
    return vals[k]


def main() -> int:
    if len(sys.argv) != 5:
        print("usage: _h2h_one_game.py bundle_seat0 bundle_seat1 seed step_timeout",
              file=sys.stderr)
        return 2
    bundle_seat0 = Path(sys.argv[1])
    bundle_seat1 = Path(sys.argv[2])
    seed = int(sys.argv[3])
    step_timeout = float(sys.argv[4])

    # Reset all RNG sources up-front for determinism.
    _pyr.seed(seed)
    if _np is not None:
        _np.random.seed(seed)
    if _torch is not None:
        try:
            _torch.manual_seed(seed)
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    # Load agents AFTER seeding so any factory-time RNG use is reproducible.
    fn0 = _load_agent(bundle_seat0)
    fn1 = _load_agent(bundle_seat1)
    tt0, tt1 = [], []
    a0 = _timed(tt0, fn0)
    a1 = _timed(tt1, fn1)

    from kaggle_environments import make
    cfg = {"actTimeout": step_timeout}
    env = make("orbit_wars", configuration=cfg, debug=False)
    env.run([a0, a1])
    rewards = [int(s.reward if s.reward is not None else 0) for s in env.state]
    steps = int(env.state[0].observation.step)
    p95_0 = _pct(tt0, 95)
    p95_1 = _pct(tt1, 95)
    # Print one structured line on stdout for parent to parse.
    print(f"RESULT {rewards[0]} {rewards[1]} {steps} {p95_0:.0f} {p95_1:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
