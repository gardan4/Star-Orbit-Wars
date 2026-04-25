"""End-to-end v12 ship pipeline: bundle + smoke + H2H vs v11.

The v12 candidate is the **first** Kaggle submission to actually use a
neural prior at the search root. Default config (matches the tested
v11 winner + the only knobs we change for v12):

  weights:           TuRBO-v3 trial 42 (runs/turbo_v3_20260424.json)
  sim_move_variant:  exp3 (eta=0.3) — same as v11
  rollout_policy:    fast — gives 13× more sims so the prior actually
                     drives Q-values
  anchor_margin:     0.5 — relaxes the anchor lock so MCTS can override
                     the heuristic's wire pick when the search gap
                     justifies it
  nn_checkpoint:     runs/bc_warmstart.pt
  nn_temperature:    1.0
  nn_hold_neutral_prob: 0.05

The v10 experiment showed that anchor=0.5 + rollout=fast at v8 weights
produced a **byte-identical** outcome sequence to the unmodified
agent — i.e. without a learned prior the extra sims don't change wire
actions. The hypothesis v12 tests is: with a real BC prior, the
extra sims DO surface override-worthy candidates the heuristic
missed.

Run:
  $env:PYTHONPATH="src;."; .venv\\Scripts\\python.exe tools\\build_v12.py \\
      --ckpt runs\\bc_warmstart.pt \\
      --out submissions\\v12.py \\
      --h2h-games 20

If --h2h-games > 0, also runs a bundled H2H vs submissions/v11.py and
reports W/L/T + Elo delta. Pass --no-smoke to skip the import-vs-random
smoke test for faster iteration.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from tools.bundle import bundle_bot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/bc_warmstart.pt",
                    help="BC warm-start checkpoint path")
    ap.add_argument("--weights-json", default="runs/turbo_v3_20260424.json",
                    help="TuRBO weights JSON to inject")
    ap.add_argument("--out", default="submissions/v12.py")
    ap.add_argument("--anchor-margin", type=float, default=0.5)
    ap.add_argument("--rollout-policy", default="fast",
                    choices=["heuristic", "fast"])
    ap.add_argument("--exp3-eta", type=float, default=0.3)
    ap.add_argument("--nn-temperature", type=float, default=1.0)
    ap.add_argument("--nn-hold-neutral-prob", type=float, default=0.05)
    ap.add_argument("--no-smoke", action="store_true",
                    help="Skip the import-vs-random smoke test")
    ap.add_argument("--h2h-games", type=int, default=0,
                    help="If >0, run H2H vs v11.py and print summary")
    ap.add_argument("--vs-bundle", default="submissions/v11.py",
                    help="Bundle to compare against in H2H")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found at {ckpt}", file=sys.stderr)
        return 2

    # Auto-patch: an eager-save checkpoint from an old `bc_warmstart.py`
    # (pre-cfg-fix) lacks the cfg key — without it, `load_conv_policy`
    # falls back to ConvPolicyCfg() defaults (64ch/6 blocks) and the
    # state_dict load fails for non-default backbones (e.g. 32ch/3 blocks).
    # Try to infer cfg from the state_dict shape and re-save in place.
    import torch  # type: ignore[import-not-found]
    raw = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    needs_patch = (
        isinstance(raw, dict)
        and ("cfg" not in raw or raw.get("cfg") is None)
        and "model_state_dict" in raw
    )
    if needs_patch:
        from dataclasses import asdict
        from orbitwars.nn.conv_policy import ConvPolicyCfg
        sd = raw["model_state_dict"]
        # stem.weight is (backbone_channels, n_channels=12, 3, 3).
        backbone_channels = int(sd["stem.weight"].shape[0])
        # blocks.N.gn1.weight only exists for N in [0..n_blocks).
        n_blocks = sum(1 for k in sd.keys() if k.endswith(".gn1.weight") and k.startswith("blocks."))
        cfg = ConvPolicyCfg(
            backbone_channels=backbone_channels, n_blocks=n_blocks,
        )
        raw["cfg"] = asdict(cfg)
        raw["model_state"] = raw["model_state_dict"]
        torch.save(raw, str(ckpt))
        print(
            f"auto-patched checkpoint {ckpt}: inferred "
            f"cfg=backbone_channels={backbone_channels}, n_blocks={n_blocks}"
        )

    weights_path = Path(args.weights_json) if args.weights_json else None
    weights_override = None
    if weights_path is not None and weights_path.exists():
        import json
        raw = json.loads(weights_path.read_text(encoding="utf-8"))
        # TuRBO runner shape extraction (same logic as bundle.py main).
        for key in ("best_weights", "best_point", "best", "weights"):
            if isinstance(raw, dict) and key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break
        if isinstance(raw, dict) and "best_index" in raw and isinstance(raw.get("trials"), list):
            best_i = int(raw["best_index"])
            for tr in raw["trials"]:
                if int(tr.get("trial", -1)) == best_i and isinstance(tr.get("weights"), dict):
                    raw = tr["weights"]
                    break
        if not isinstance(raw, dict):
            print(f"WARNING: couldn't unwrap weights from {weights_path}; "
                  "using defaults", file=sys.stderr)
        else:
            weights_override = {k: float(v) for k, v in raw.items()}
            print(f"weights override: {len(weights_override)} keys "
                  f"from {weights_path}")

    out_path = Path(args.out)
    print(f"\n=== bundling v12 -> {out_path} ===")
    t0 = time.perf_counter()
    bundle_bot(
        "mcts_bot", out_path,
        weights_override=weights_override,
        sim_move_variant="exp3",
        exp3_eta=args.exp3_eta,
        rollout_policy=args.rollout_policy,
        anchor_margin=args.anchor_margin,
        nn_checkpoint=ckpt,
        nn_temperature=args.nn_temperature,
        nn_hold_neutral_prob=args.nn_hold_neutral_prob,
    )
    bundle_dt = time.perf_counter() - t0
    print(f"bundle wall: {bundle_dt:.1f}s  size: {out_path.stat().st_size / 1024:.0f} KB")

    if not args.no_smoke:
        print(f"\n=== smoke vs random ===")
        from tools.bundle import _smoke_test
        _smoke_test(out_path)

    if args.h2h_games > 0:
        vs = Path(args.vs_bundle)
        if not vs.exists():
            print(f"ERROR: --vs-bundle not found at {vs}", file=sys.stderr)
            return 3
        print(f"\n=== H2H vs {vs} ({args.h2h_games} games) ===")
        # Defer to the existing diag_bundle_round_robin script (used for
        # all prior version-pair H2Hs in this project — see v8 vs v7,
        # v9 vs v8, v11 vs v9 logs in runs/).
        import subprocess
        proc = subprocess.run(
            [
                sys.executable, "-m", "tools.diag_bundle_round_robin",
                "--bundles", f"{out_path},{vs}",
                "--games", str(args.h2h_games),
                "--seed", "42",
            ],
            check=False,
        )
        if proc.returncode != 0:
            print(
                f"WARNING: H2H exited {proc.returncode}; bundle is still "
                "ready, just inspect H2H separately.",
                file=sys.stderr,
            )

    print(
        f"\n=== v12 ready at {out_path} ===\n"
        "Next: build kaggle notebook, push, submit:\n"
        "  python -m tools.build_kaggle_notebook --bundle "
        f"{out_path} --slug gardan4/orbit-wars-mcts-v12 "
        "--title \"Orbit Wars MCTS v12\" --out submissions/kaggle_notebook_v12/\n"
        "  kaggle.exe kernels push -p submissions/kaggle_notebook_v12/\n"
        "  kaggle.exe kernels status gardan4/orbit-wars-mcts-v12\n"
        "  kaggle.exe competitions submit -c orbit-wars -k gardan4/orbit-wars-mcts-v12 "
        "-v 1 -f submission.py -m \"v12: NN prior + v11 weights + margin=0.5 + fast rollouts\""
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
