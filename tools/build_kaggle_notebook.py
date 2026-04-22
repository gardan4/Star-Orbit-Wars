"""Wrap a bundled single-file submission in a Kaggle Notebook for push.

Kaggle's Simulation track doesn't accept `-f submission.py` directly — it
requires a Kaggle Notebook (kernel) that, when executed, produces a
`submission.py` in its output directory. Kaggle then imports that
`submission.py` and calls `agent(obs, cfg)` for each turn.

The simplest faithful wrapper is a 1-cell notebook with
`%%writefile submission.py` followed by the bundle contents — Kaggle
runs the cell, the file gets written, the grader picks it up.

Usage:
    python -m tools.build_kaggle_notebook --bundle submissions/mcts_v2.py \
        --title "Orbit Wars MCTS v2" \
        --slug gardan4/orbit-wars-mcts-v2 \
        --out submissions/kaggle_notebook/

This writes:
    submissions/kaggle_notebook/kernel-metadata.json
    submissions/kaggle_notebook/orbit-wars-mcts-v2.ipynb

Then:
    .venv\\Scripts\\kaggle.exe kernels push -p submissions/kaggle_notebook/
    .venv\\Scripts\\kaggle.exe competitions submit -c orbit-wars \\
        -k gardan4/orbit-wars-mcts-v2 -m "<msg>"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List


def _cell_markdown(text: str) -> dict:
    # Split on newlines, preserve \n on every line except the last — the
    # nbformat canonical form.
    lines = text.splitlines(keepends=True)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def _cell_code(text: str) -> dict:
    lines = text.splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def build_notebook(bundle_path: Path, title: str) -> dict:
    bundle_src = bundle_path.read_text(encoding="utf-8")
    header = (
        f"# {title}\n"
        f"\n"
        f"Auto-generated submission notebook for the [Orbit Wars]"
        f"(https://www.kaggle.com/competitions/orbit-wars) Kaggle "
        f"Simulation competition.\n"
        f"\n"
        f"The next cell writes `submission.py` containing a "
        f"self-contained single-file bot. Kaggle's grader imports "
        f"that file and calls `agent(obs, cfg)` each turn.\n"
        f"\n"
        f"Rebuild with:\n"
        f"```\n"
        f"python -m tools.build_kaggle_notebook --bundle {bundle_path.as_posix()}\n"
        f"```\n"
    )
    write_cell = "%%writefile submission.py\n" + bundle_src

    return {
        "cells": [
            _cell_markdown(header),
            _cell_code(write_cell),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_metadata(slug: str, title: str, notebook_file: str) -> dict:
    # Canonical kernel-metadata.json for a competition-bound notebook.
    # - is_private: true so the kernel stays unlisted while we iterate.
    # - enable_internet: false — Kaggle submission kernels run offline;
    #   our bundle has no remote-fetch dependencies.
    # - competition_sources links the kernel to the comp for submission.
    # See https://github.com/Kaggle/kaggle-api for schema.
    return {
        "id": slug,
        "title": title,
        "code_file": notebook_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["orbit-wars"],
        "kernel_sources": [],
    }


def _slug_to_kernel_filename(slug: str) -> str:
    """gardan4/orbit-wars-mcts-v2 -> orbit-wars-mcts-v2.ipynb"""
    # Kaggle kernels push expects `code_file` to match a file in `-p` dir.
    # We name the notebook after the kernel name (slug after /).
    name = slug.split("/", 1)[-1]
    # Sanity: kernel name should be alnum + dashes
    if not re.match(r"^[a-z0-9-]+$", name):
        raise SystemExit(f"Invalid kernel name '{name}' in slug — lowercase + dashes only")
    return f"{name}.ipynb"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bundle", required=True,
        help="Path to the bundled single-file submission "
             "(e.g. submissions/mcts_v2.py)",
    )
    ap.add_argument(
        "--slug", required=True,
        help="Kaggle kernel slug, e.g. gardan4/orbit-wars-mcts-v2",
    )
    ap.add_argument(
        "--title", required=True,
        help="Human-readable kernel title",
    )
    ap.add_argument(
        "--out", default="submissions/kaggle_notebook",
        help="Output directory for notebook + metadata",
    )
    args = ap.parse_args()

    bundle = Path(args.bundle)
    if not bundle.exists():
        raise SystemExit(f"Bundle not found: {bundle}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nb_filename = _slug_to_kernel_filename(args.slug)
    nb_path = out_dir / nb_filename
    md_path = out_dir / "kernel-metadata.json"

    nb = build_notebook(bundle, args.title)
    md = build_metadata(args.slug, args.title, nb_filename)

    nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    md_path.write_text(json.dumps(md, indent=2), encoding="utf-8")

    bundle_bytes = bundle.stat().st_size
    nb_bytes = nb_path.stat().st_size
    print(f"Wrote {nb_path} ({nb_bytes} bytes, wraps {bundle_bytes}-byte bundle)")
    print(f"Wrote {md_path}")
    print()
    print("Next steps:")
    print(
        f"  .venv\\Scripts\\kaggle.exe kernels push -p {out_dir.as_posix()}"
    )
    print(
        f"  .venv\\Scripts\\kaggle.exe competitions submit -c orbit-wars "
        f"-k {args.slug} -m \"<msg>\""
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
