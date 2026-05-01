#!/usr/bin/env bash
# Auto-run the iter-2 closed-loop pipeline once the big demo set is ready.
# Assumes runs/closed_loop_iter1_postfix/demos_iter1_big.npz exists.
#
# Steps:
#   1. Train v5 (joint policy+value AZ) on the big demos
#   2. Bundle v36 (heuristic rollouts + v5 prior + nn_value option)
#   3. Run 16-game H2H v36 vs v32b for tighter CI
#
# Run: bash tools/run_iter2_chain.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DEMOS=runs/closed_loop_iter1_postfix/demos_iter1_big.npz
V5_CKPT=runs/closed_loop_iter1_postfix/az_iter2.pt
V36_PRIOR_BUNDLE=submissions/v36_priorv5.py
V36_NNV_BUNDLE=submissions/v36b_nnv5.py

if [[ ! -f "$DEMOS" ]]; then
    echo "ERROR: $DEMOS not found. Run demo collection first."
    exit 1
fi

echo "=== Step 1: Train v5 (joint policy+value AZ) ==="
.venv/Scripts/python.exe -m tools.train_az_head \
    --demos "$DEMOS" \
    --bc-checkpoint runs/bc_warmstart_small_cpu.pt \
    --out "$V5_CKPT" \
    --epochs 30 --batch-size 256 --lr 1e-3 \
    --policy-tau 1.5 --policy-eps 0.5 \
    --lambda-p 1.0 --lambda-v 1.0

echo
echo "=== Step 2a: Bundle v36 (v5 prior + heuristic rollouts) ==="
.venv/Scripts/python.exe -m tools.bundle \
    --out "$V36_PRIOR_BUNDLE" \
    --bot mcts_bot \
    --weights-json runs/turbo_v5_w3pool.json \
    --total-sims 128 --hard-deadline-ms 850 \
    --rollout-policy heuristic \
    --anchor-margin 0.0 \
    --sim-move-variant exp3 --exp3-eta 0.3 \
    --nn-checkpoint "$V5_CKPT"

echo
echo "=== Step 2b: Bundle v36b (v5 prior + v5 value as leaf eval) ==="
.venv/Scripts/python.exe -m tools.bundle \
    --out "$V36_NNV_BUNDLE" \
    --bot mcts_bot \
    --weights-json runs/turbo_v5_w3pool.json \
    --total-sims 128 --hard-deadline-ms 850 \
    --rollout-policy nn_value \
    --anchor-margin 0.3 \
    --sim-move-variant exp3 --exp3-eta 0.3 \
    --nn-checkpoint "$V5_CKPT"

echo
echo "=== Step 3a: 16-game H2H v36 (heuristic) vs v32b ==="
.venv/Scripts/python.exe -m tools.h2h_mirror \
    --bundles "$V36_PRIOR_BUNDLE,submissions/v32b_heur.py" \
    --games 8 --seed 42 --step_timeout 1.0

echo
echo "=== Step 3b: 16-game H2H v36b (nn_value) vs v32b ==="
.venv/Scripts/python.exe -m tools.h2h_mirror \
    --bundles "$V36_NNV_BUNDLE,submissions/v32b_heur.py" \
    --games 8 --seed 42 --step_timeout 1.0

echo
echo "=== DONE ==="
echo "Decision: ship slot 2 only if either H2H shows v36 winning >=10/16 (62.5%)."
