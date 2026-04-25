"""Diagnostic: compare BC-trained priors vs random-init priors on a
fixed observation set.

What this answers (faster than running an H2H):
* Is the BC checkpoint loadable? Tests both partial and full formats.
* Does the BC prior produce a meaningfully *different* distribution
  than the random-init policy? (If they're indistinguishable, BC
  failed to learn anything useful and v12 H2H will be a coin flip.)
* On a per-planet basis, does the BC prior agree with the heuristic's
  raw-score-based prior (which is what `generate_per_planet_moves`
  already attaches)? Strong agreement = BC just memorized the
  heuristic; partial agreement = there's a chance MCTS will surface
  candidates the heuristic missed.

Output: a single-screen summary table comparing the four prior modes
(heuristic, random-init NN, BC NN, uniform) on the same observation.

Run:
  $env:PYTHONPATH="src;."; .venv\\Scripts\\python.exe tools\\diag_nn_prior.py \\
      --ckpt runs\\bc_warmstart.pt
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import torch  # type: ignore[import-not-found]

from orbitwars.bots.heuristic import (
    ArrivalTable, build_arrival_table, parse_obs, HEURISTIC_WEIGHTS,
)
from orbitwars.mcts.actions import ActionConfig, generate_per_planet_moves
from orbitwars.nn.conv_policy import ConvPolicy, ConvPolicyCfg
from orbitwars.nn.nn_prior import (
    load_conv_policy, make_nn_prior_fn,
)


def _make_obs_fixture():
    """Mid-game-ish obs: 3 owned planets, mixed enemies, fleets in flight."""
    return {
        "player": 0,
        "step": 80,
        "angular_velocity": 0.03,
        "planets": [
            [0, 0, 22.0, 50.0, 1.5, 80, 3],
            [1, 1, 78.0, 50.0, 1.5, 60, 3],
            [2, 0, 50.0, 22.0, 1.0, 40, 2],
            [3, 1, 50.0, 78.0, 1.0, 40, 2],
            [4, -1, 30.0, 30.0, 1.0, 12, 1],
            [5, 0, 70.0, 30.0, 1.0, 25, 1],
        ],
        "initial_planets": [
            [0, 0, 22.0, 50.0, 1.5, 10, 3],
            [1, 1, 78.0, 50.0, 1.5, 10, 3],
            [2, 0, 50.0, 22.0, 1.0, 10, 2],
            [3, 1, 50.0, 78.0, 1.0, 10, 2],
            [4, -1, 30.0, 30.0, 1.0, 10, 1],
            [5, 0, 70.0, 30.0, 1.0, 10, 1],
        ],
        "fleets": [],
        "next_fleet_id": 0,
        "comet_planet_ids": [],
        "comets": [],
    }


def _entropy(p: List[float]) -> float:
    s = 0.0
    for q in p:
        if q > 0:
            s -= q * math.log(q)
    return s


def _kl(p: List[float], q: List[float]) -> float:
    """KL(p||q). Both must sum to 1. Skip terms where p==0."""
    s = 0.0
    eps = 1e-12
    for pi, qi in zip(p, q):
        if pi > 0:
            s += pi * (math.log(pi + eps) - math.log(qi + eps))
    return s


def _summarize_priors(
    label: str, priors_by_pid: Dict[int, List[float]], moves_by_pid: Dict[int, list],
) -> None:
    n_planets = len(priors_by_pid)
    if n_planets == 0:
        print(f"{label:24s}  (no owned planets)")
        return
    entropies: List[float] = []
    n_moves: List[int] = []
    top_p: List[float] = []
    for pid, ps in priors_by_pid.items():
        entropies.append(_entropy(ps))
        n_moves.append(len(ps))
        top_p.append(max(ps) if ps else 0.0)
    h_mean = sum(entropies) / len(entropies)
    h_unif = sum(math.log(max(n, 1)) for n in n_moves) / len(n_moves)
    top_mean = sum(top_p) / len(top_p)
    print(
        f"{label:24s}  "
        f"planets={n_planets}  "
        f"H_mean={h_mean:.3f} (unif={h_unif:.3f})  "
        f"max_p_mean={top_mean:.3f}  "
        f"n_moves_mean={sum(n_moves)/len(n_moves):.1f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/bc_warmstart.pt",
                    help="BC warm-start checkpoint path (skip if missing)")
    args = ap.parse_args()

    obs = _make_obs_fixture()
    po = parse_obs(obs)
    table = build_arrival_table(po)
    per_planet_heur = generate_per_planet_moves(
        po, table, weights=HEURISTIC_WEIGHTS, cfg=ActionConfig(),
    )
    avail = {int(pid): int(po.planet_by_id[pid][5]) for pid in per_planet_heur.keys()}

    # 1. Heuristic priors as already attached by generate_per_planet_moves.
    heur_priors = {
        pid: [m.prior for m in moves]
        for pid, moves in per_planet_heur.items()
    }
    # Renormalize per-planet (priors may sum to >1 in raw form).
    for pid, ps in heur_priors.items():
        s = sum(ps) or 1.0
        heur_priors[pid] = [p / s for p in ps]

    # 2. Random-init NN prior.
    cfg_default = ConvPolicyCfg()
    rand_model = ConvPolicy(cfg_default)
    rand_model.eval()
    rand_fn = make_nn_prior_fn(rand_model, cfg_default)
    rand_pp = rand_fn(obs, 0, per_planet_heur, avail)
    rand_priors = {pid: [m.prior for m in moves] for pid, moves in rand_pp.items()}

    # 3. BC NN prior (if checkpoint exists).
    bc_priors: Dict[int, List[float]] | None = None
    ckpt_path = Path(args.ckpt)
    if ckpt_path.exists():
        try:
            model, cfg = load_conv_policy(ckpt_path, device="cpu")
            bc_fn = make_nn_prior_fn(model, cfg)
            bc_pp = bc_fn(obs, 0, per_planet_heur, avail)
            bc_priors = {pid: [m.prior for m in moves] for pid, moves in bc_pp.items()}
            print(f"loaded BC checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"BC checkpoint load failed: {e!r}", file=sys.stderr)
    else:
        print(f"BC checkpoint not found at {ckpt_path}; comparing heur vs random-init only")

    # 4. Uniform reference.
    unif_priors = {
        pid: [1.0 / len(moves)] * len(moves)
        for pid, moves in per_planet_heur.items()
    }

    print()
    print("Per-planet prior summary (averaged over owned planets):")
    print("-" * 80)
    _summarize_priors("heuristic (renorm)", heur_priors, per_planet_heur)
    _summarize_priors("uniform", unif_priors, per_planet_heur)
    _summarize_priors("random-init NN", rand_priors, per_planet_heur)
    if bc_priors is not None:
        _summarize_priors("BC NN", bc_priors, per_planet_heur)

    # KL distances per planet.
    if bc_priors is not None:
        print()
        print("KL divergences (per planet, averaged):")
        print("-" * 80)
        kl_bc_heur = []
        kl_bc_rand = []
        kl_bc_unif = []
        for pid in per_planet_heur.keys():
            kl_bc_heur.append(_kl(bc_priors[pid], heur_priors[pid]))
            kl_bc_rand.append(_kl(bc_priors[pid], rand_priors[pid]))
            kl_bc_unif.append(_kl(bc_priors[pid], unif_priors[pid]))
        print(f"  KL(BC || heuristic) = {sum(kl_bc_heur)/len(kl_bc_heur):.3f}")
        print(f"  KL(BC || random)    = {sum(kl_bc_rand)/len(kl_bc_rand):.3f}")
        print(f"  KL(BC || uniform)   = {sum(kl_bc_unif)/len(kl_bc_unif):.3f}")
        print()
        print("Interpretation:")
        print("  * KL(BC||uniform) > 1     -> BC has learned non-trivial structure (good)")
        print("  * KL(BC||heuristic) < 0.3 -> BC has memorized heuristic (limits ceiling)")
        print("  * KL(BC||random) > 1      -> BC differs meaningfully from init (training did something)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
