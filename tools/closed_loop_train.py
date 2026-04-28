"""Closed-loop self-play training orchestrator (Stage 4 of RL plan).

Iterates:
  iter 0  → collect demos with `--bc-checkpoint <initial_ckpt>` (the
            current best NN); train value head on those demos →
            checkpoint_iter1
  iter 1  → collect demos with checkpoint_iter1; train value head →
            checkpoint_iter2
  ...

Each iteration's demos reflect the latest (presumably stronger) policy's
self-play outcomes. Value head trained on those demos then targets
PROGRESSIVELY STRONGER play, which (in theory) lets MCTS-with-value-head
make progressively better moves, which makes the next iteration's
demos even stronger. Standard AlphaZero recipe.

Why this works ONLY post-Phantom-4.0:
Before the fix, the demo collector + MCTSAgent silently reverted
rollout_policy to "heuristic" + sim_move_variant to "ucb" + use_macros
to False at every act() call. So all "self-play with strong MCTS" was
actually self-play with a buggy heuristic-anchor-locked agent. Iterating
that produced no improvement. Post-fix, the agent ACTUALLY runs the
configured rollout_policy (fast or nn_value), so each iteration's demos
reflect a real change.

Run:
    python -m tools.closed_loop_train \\
        --initial-checkpoint runs/bc_warmstart_small_cpu.pt \\
        --n-iterations 3 --games-per-iter 15 \\
        --sims 128 --deadline-ms 850 \\
        --out-dir runs/closed_loop/

After 3-5 iterations, eval the final checkpoint via:
  - Bundle with --rollout-policy=nn_value + final ckpt
  - Ladder ship + ground-truth Elo

Single-iteration time estimate (CPU, post-fix):
  - Demo collection: 15 games × ~3 min/game = ~45 min
  - Value head training (frozen backbone): 30 epochs × ~30s = ~15 min
  - Total: ~60 min/iter

For 5 iterations: ~5h wall. Tractable overnight on RTX 3070 (would be
~2-3h with GPU NN inference during demo + training).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _run_cli(cmd: list, log_path: Path, label: str) -> int:
    """Run a subprocess and stream output to both stdout and a log file.

    Returns exit code. Logs are kept per-iteration for offline inspection.
    """
    print(f"\n{'=' * 60}\n[{label}] Running: {' '.join(str(c) for c in cmd)}\n{'=' * 60}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd, cwd=_ROOT, stdout=logf, stderr=subprocess.STDOUT, text=True,
        )
    print(f"[{label}] exit code: {proc.returncode}; log: {log_path}", flush=True)
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--initial-checkpoint", required=True, type=Path,
                    help="Starting BC checkpoint (e.g. bc_warmstart_small_cpu.pt)")
    ap.add_argument("--n-iterations", type=int, default=3,
                    help="Number of closed-loop iterations to run")
    ap.add_argument("--games-per-iter", type=int, default=15,
                    help="Demo games to collect per iteration")
    ap.add_argument("--sims", type=int, default=128,
                    help="MCTS total_sims for demo collection")
    ap.add_argument("--deadline-ms", type=float, default=850.0)
    ap.add_argument("--epochs-per-iter", type=int, default=20,
                    help="Value head training epochs per iteration")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/closed_loop"))
    ap.add_argument("--seed-base", type=int, default=2000,
                    help="Base seed; iteration N uses base + N*10000")
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    ap.add_argument(
        "--rollout-policy",
        choices=["heuristic", "fast", "nn_value"],
        default="fast",
        help=(
            "MCTS leaf evaluator for the demo-collection step of each "
            "iteration. 'fast' is the cheap default. 'nn_value' uses the "
            "ITER-N value head as leaf eval, so each iteration's demos "
            "reflect the NN value head's strategic assessment — this is "
            "the AlphaZero-style closed loop (stronger value head -> "
            "stronger demos -> stronger next value head)."
        ),
    )
    args = ap.parse_args()

    if not args.initial_checkpoint.exists():
        raise SystemExit(f"--initial-checkpoint not found: {args.initial_checkpoint}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable
    iter_log_dir = args.out_dir / "logs"
    iter_log_dir.mkdir(parents=True, exist_ok=True)

    # Track checkpoints across iterations.
    current_ckpt = args.initial_checkpoint
    history = {
        "initial_checkpoint": str(args.initial_checkpoint),
        "iterations": [],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    history_path = args.out_dir / "history.json"
    t_total = time.perf_counter()

    for it in range(args.n_iterations):
        iter_start = time.perf_counter()
        print(f"\n\n{'#' * 70}")
        print(f"# CLOSED-LOOP ITERATION {it + 1}/{args.n_iterations}")
        print(f"#   current_ckpt: {current_ckpt}")
        print(f"{'#' * 70}", flush=True)

        # ---- Stage A: collect demos with the current checkpoint ----
        seed = args.seed_base + it * 10000
        demos_path = args.out_dir / f"demos_iter{it + 1}.npz"
        cmd_demos = [
            python_exe, "-u", "-m", "tools.collect_mcts_demos",
            "--games", str(args.games_per_iter),
            "--sims", str(args.sims),
            "--deadline-ms", str(args.deadline_ms),
            "--seed", str(seed),
            "--out", str(demos_path),
            "--bc-checkpoint", str(current_ckpt),
            "--rollout-policy", args.rollout_policy,
        ]
        rc = _run_cli(cmd_demos, iter_log_dir / f"iter{it + 1}_demos.log",
                       label=f"iter{it+1} demos")
        if rc != 0 or not demos_path.exists():
            print(f"  iter{it + 1} demo collection failed (rc={rc}). Aborting loop.")
            return 1

        # ---- Stage B: train value head on those demos ----
        next_ckpt = args.out_dir / f"value_head_iter{it + 1}.pt"
        cmd_train = [
            python_exe, "-u", "-m", "tools.train_value_head",
            "--demos", str(demos_path),
            "--bc-checkpoint", str(current_ckpt),
            "--out", str(next_ckpt),
            "--epochs", str(args.epochs_per_iter),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--device", args.device,
            "--seed", str(seed + 1),
        ]
        rc = _run_cli(cmd_train, iter_log_dir / f"iter{it + 1}_train.log",
                       label=f"iter{it+1} train")
        if rc != 0 or not next_ckpt.exists():
            print(f"  iter{it + 1} value head training failed (rc={rc}). Aborting.")
            return 1

        # Record + advance.
        iter_wall = time.perf_counter() - iter_start
        sidecar_path = next_ckpt.with_suffix(".json")
        sidecar = {}
        if sidecar_path.exists():
            try:
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        history["iterations"].append({
            "iter": it + 1,
            "demos": str(demos_path),
            "checkpoint_in": str(current_ckpt),
            "checkpoint_out": str(next_ckpt),
            "training_history": sidecar.get("training_history"),
            "wall_seconds": iter_wall,
        })
        history["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        history["total_wall_seconds"] = time.perf_counter() - t_total
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        print(f"\n  iter{it + 1} complete in {iter_wall:.0f}s. "
              f"Next checkpoint: {next_ckpt}", flush=True)
        current_ckpt = next_ckpt

    print(f"\n\n=== Closed-loop training complete ===")
    print(f"Final checkpoint: {current_ckpt}")
    print(f"Total wall: {time.perf_counter() - t_total:.0f}s")
    print(f"History: {history_path}")
    print(f"\nNext steps:")
    print(f"  1. Bundle with --rollout-policy=nn_value --nn-checkpoint {current_ckpt}")
    print(f"  2. Run mirror H2H vs current ladder leader")
    print(f"  3. If wins meaningfully, ship to ladder")
    return 0


if __name__ == "__main__":
    sys.exit(main())
