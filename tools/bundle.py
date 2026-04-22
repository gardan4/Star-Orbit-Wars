"""Single-file submission packager.

Kaggle's orbit_wars submission is ONE Python file exposing a
`def agent(obs, config=None) -> list` callable. This tool inlines our
`orbitwars` package code into such a file.

Design decisions:
  * We inline by module, in explicit dependency order — simpler than AST
    rewriting and keeps diagnostics readable in Kaggle's log (tracebacks
    still point to meaningful source lines).
  * Any `from orbitwars.xxx import y` import that references an inlined
    module is replaced with a harmless pass-through (the names are already
    in scope).
  * External imports (numpy, math, random, ...) are preserved as-is. We do
    NOT inline anything from stdlib or site-packages.
  * Emit a `def agent(obs, config=None):` at the end wrapping
    `HeuristicAgent().as_kaggle_agent()`.

Usage:
    python -m tools.bundle --bot heuristic --out submissions/heuristic_v1.py
    python -m tools.bundle --bot noop --out submissions/noop.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src" / "orbitwars"


# Which bots we know how to package (name -> (module path, agent factory line))
#
# Module order matters: each module may reference names defined in earlier
# modules (those imports are stripped by the bundler). The listed order
# must satisfy dependency topo-sort.
BOT_RECIPES: Dict[str, Tuple[List[str], str]] = {
    "noop": (
        ["bots.base"],
        "agent = NoOpAgent().as_kaggle_agent()",
    ),
    "random": (
        ["bots.base"],
        "agent = RandomAgent(seed=0).as_kaggle_agent()",
    ),
    "heuristic": (
        ["engine.intercept", "bots.base", "bots.heuristic"],
        "agent = HeuristicAgent().as_kaggle_agent()",
    ),
    # Path B bot. Currently (W2/W3) equivalent to heuristic at the wire
    # level because `anchor_improvement_margin=2.0` locks the heuristic
    # floor — search still runs but never overrides the anchor. Keep
    # this recipe so the moment we flip search net-positive (W4-5 neural
    # prior) we can ship it with one bundle command. Bundle size: ~3k
    # lines; stays well under Kaggle's per-file limit.
    "mcts_bot": (
        [
            "engine.intercept",
            "bots.base",
            "bots.heuristic",
            "bots.fast_rollout",  # referenced by gumbel_search when rollout_policy="fast"
            "engine.fast_engine",
            "mcts.bokr_widen",    # provides BOKRKernelSelector used by mcts.actions
            "mcts.actions",
            "mcts.gumbel_search",
            "opponent.archetypes",
            "opponent.bayes",
            "bots.mcts_bot",
        ],
        "agent = MCTSAgent(rng_seed=0).as_kaggle_agent()",
    ),
}


def _strip_orbitwars_imports(src: str) -> str:
    """Remove `from orbitwars.xxx import ...` (including parenthesized
    multi-line import blocks). Those names are already defined in the
    bundled file after inlining."""
    lines = src.splitlines()
    out_lines: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith("from orbitwars.") or stripped.startswith("import orbitwars."):
            # Multi-line case: `from orbitwars.x import (` spans to matching `)`.
            if "(" in line and ")" not in line:
                # Skip until the line containing the matching `)`.
                while i < len(lines) and ")" not in lines[i]:
                    i += 1
                # Skip the closing line too
                i += 1
                continue
            # Single-line case: skip this line only.
            i += 1
            continue
        out_lines.append(line)
        i += 1
    return "\n".join(out_lines)


def _strip_leading_docstring_module_header(src: str) -> str:
    """Drop the module-level docstring and `from __future__` line if present —
    they don't belong mid-file after the first module."""
    # This is a soft heuristic; simplest approach is to keep them. A bundled
    # file with multiple `from __future__` lines is fine in Python 3.13.
    return src


def _read_module(rel_path: str) -> str:
    """rel_path e.g. 'bots.base' -> reads src/orbitwars/bots/base.py"""
    parts = rel_path.split(".")
    path = SRC_ROOT.joinpath(*parts).with_suffix(".py")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


HEADER = """\
# Auto-generated Orbit Wars submission. Do not edit by hand.
# Built by tools/bundle.py on {timestamp}.
# Bot: {bot_name}
#
# Single-file submission: Kaggle will import this and call agent(obs, cfg).
from __future__ import annotations

"""


def bundle_bot(bot_name: str, out_path: Path) -> None:
    if bot_name not in BOT_RECIPES:
        raise SystemExit(f"Unknown bot '{bot_name}'. Known: {list(BOT_RECIPES)}")
    modules, agent_factory = BOT_RECIPES[bot_name]
    parts: List[str] = [HEADER.format(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), bot_name=bot_name)]
    seen_future: List[str] = []

    for mod in modules:
        src = _read_module(mod)
        src = _strip_orbitwars_imports(src)
        # Consolidate multiple `from __future__ import annotations` into one
        # at the top; python requires them to be at the start of file.
        src = re.sub(r"^from __future__ import annotations\s*$", "", src, flags=re.MULTILINE)
        parts.append(f"# --- inlined: orbitwars/{mod.replace('.', '/')}.py ---\n")
        parts.append(src.rstrip() + "\n")
        parts.append("\n")

    # Agent entry point
    parts.append("\n# --- agent entry point ---\n")
    parts.append(f"{agent_factory}\n")

    content = "\n".join(parts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


def _smoke_test(out_path: Path) -> None:
    """Import the bundled file in a subprocess and run one game vs noop."""
    import subprocess
    code = f"""
import sys, traceback
sys.path.insert(0, r'{out_path.parent}')
mod_name = '{out_path.stem}'
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(mod_name, r'{out_path}')
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m  # dataclasses need the module in sys.modules
    spec.loader.exec_module(m)
    assert callable(m.agent), 'agent not callable'
    from kaggle_environments import make
    env = make('orbit_wars', configuration={{'actTimeout': 60}}, debug=False)
    env.run([m.agent, 'random'])
    rewards = [s.reward for s in env.state]
    steps = env.state[0].observation.step
    print(f'OK rewards={{rewards}} steps={{steps}}')
except Exception:
    traceback.print_exc()
    sys.exit(1)
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        raise SystemExit("Smoke test failed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", required=True, choices=list(BOT_RECIPES))
    ap.add_argument("--out", required=True, help="Output .py path")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Run a single game against 'random' to verify the bundle works")
    args = ap.parse_args()

    out = Path(args.out)
    bundle_bot(args.bot, out)
    print(f"Bundled {args.bot} -> {out} ({out.stat().st_size} bytes)")

    if args.smoke_test:
        _smoke_test(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
