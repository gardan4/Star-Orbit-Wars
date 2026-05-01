"""Build the v7 Kaggle notebook from mcts_v7.py bundle.

Produces:
  submissions/kaggle_notebook_v7/orbit-wars-mcts-v7.ipynb
  submissions/kaggle_notebook_v7/kernel-metadata.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "submissions" / "kaggle_notebook_v7"
BUNDLE = ROOT / "submissions" / "mcts_v7.py"

MD = """# Orbit Wars MCTS v7

Auto-generated submission notebook for the [Orbit Wars](https://www.kaggle.com/competitions/orbit-wars) competition.

**v7 vs v6:** hardens `FastEngine._maybe_spawn_comets` against a global-`random`
state leak during MCTS rollouts. Previously, rollouts that crossed comet-spawn
steps {50, 150, 250, 350, 450} consumed up to ~900 `random.uniform` calls per
spawn from the Kaggle judge's entropy stream (via the official-engine
`generate_comet_paths` helper, which retries up to 300 times per path). This
perturbed the judge's own comet spawns downstream and biased outcomes by as
much as 9x background consumption. The fix snapshots and restores
`random.getstate()` around the `generate_comet_paths` call when the engine's
rng is not the global `random` module (i.e. everywhere except the parity
validator). Parity validator still passes 3/3 seeds x 100 turns.
"""


def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    bundle_src = BUNDLE.read_text(encoding="utf-8")
    code_cell_src = "%%writefile submission.py\n" + bundle_src

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": MD,
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code_cell_src,
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NB_DIR / "orbit-wars-mcts-v7.ipynb").write_text(
        json.dumps(nb, indent=1), encoding="utf-8"
    )

    meta = {
        "id": "gardan4/orbit-wars-mcts-v7",
        "title": "Orbit Wars MCTS v7",
        "code_file": "orbit-wars-mcts-v7.ipynb",
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
    (NB_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    nb_size = (NB_DIR / "orbit-wars-mcts-v7.ipynb").stat().st_size
    print(f"Wrote {NB_DIR}/orbit-wars-mcts-v7.ipynb ({nb_size} bytes)")
    print(f"Wrote {NB_DIR}/kernel-metadata.json")


if __name__ == "__main__":
    main()
