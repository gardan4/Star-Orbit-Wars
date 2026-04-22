"""Simulate Kaggle's kernel execution locally.

When Kaggle runs a Simulation submission notebook it:
  1. Imports the kernel.
  2. Executes each cell in order.
  3. Expects a `submission.py` file on disk after execution.
  4. Imports `submission.py` and calls `agent(obs, cfg)` each turn.

This verifier reproduces steps 1-4 locally to catch bundling / escaping
bugs before we push the kernel and burn a ladder slot.

Usage:
    python -m tools.verify_kaggle_notebook \
        --notebook submissions/kaggle_notebook/orbit-wars-mcts-v2.ipynb
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def extract_writefile(notebook_path: Path) -> str:
    """Return the `submission.py` content from the first
    %%writefile-submission.py code cell."""
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if src.startswith("%%writefile submission.py"):
            # Drop the magic line; return the rest verbatim.
            return src.split("\n", 1)[1]
    raise SystemExit(f"No %%writefile submission.py cell in {notebook_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", required=True)
    ap.add_argument(
        "--compare-bundle", default=None,
        help="Optional: bundle file to diff against the extracted cell",
    )
    args = ap.parse_args()

    nb_path = Path(args.notebook)
    submission_src = extract_writefile(nb_path)

    print(f"Extracted submission.py from {nb_path}: {len(submission_src)} chars")

    # Optional byte-level diff against the source bundle.
    if args.compare_bundle:
        bundle_src = Path(args.compare_bundle).read_text(encoding="utf-8")
        if submission_src == bundle_src:
            print(f"  Exact match vs {args.compare_bundle}")
        else:
            # Give enough signal to debug without dumping both files.
            print(
                f"  MISMATCH vs {args.compare_bundle} "
                f"(notebook: {len(submission_src)}, bundle: {len(bundle_src)})"
            )
            # Find first diverging index.
            lim = min(len(submission_src), len(bundle_src))
            for i in range(lim):
                if submission_src[i] != bundle_src[i]:
                    # Show ±40 chars around the first difference.
                    lo, hi = max(0, i - 40), min(lim, i + 40)
                    print(f"    first diff at char {i}:")
                    print(f"      notebook: {submission_src[lo:hi]!r}")
                    print(f"      bundle  : {bundle_src[lo:hi]!r}")
                    break
            else:
                # Same prefix, lengths differ.
                print(
                    f"    prefix matches; trailing diff: "
                    f"{submission_src[lim:lim+60]!r} vs {bundle_src[lim:lim+60]!r}"
                )
            return 1

    # Write extracted content to a temp file and simulate Kaggle import.
    with tempfile.TemporaryDirectory() as td:
        sub_path = Path(td) / "submission.py"
        sub_path.write_text(submission_src, encoding="utf-8")
        print(f"  Wrote simulated submission.py ({sub_path.stat().st_size} bytes)")

        smoke = f"""
import sys, traceback
sys.path.insert(0, r'{td}')
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location('submission', r'{sub_path}')
    m = importlib.util.module_from_spec(spec)
    sys.modules['submission'] = m
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
        res = subprocess.run(
            [sys.executable, "-c", smoke],
            capture_output=True, text=True, timeout=300,
        )
        print(res.stdout.strip())
        if res.returncode != 0:
            print(res.stderr, file=sys.stderr)
            print("SMOKE FAIL — notebook would not run on Kaggle")
            return 1
    print("OK — notebook is safe to push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
