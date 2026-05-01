#!/usr/bin/env bash
# Closed-loop iteration: v41 = AZ retrained on v40d-style NN-rollout demos.
#
# Prereq: runs/v41_demos.npz exists (from `tools/collect_mcts_demos.py
# --rollout-policy nn --bc-checkpoint runs/az_v39_bigger.pt`).
#
# Stages:
#   1. Train v41 AZ checkpoint from those demos (heads-only on top of
#      the v39_bigger backbone; backbone unfreeze destabilizes value head
#      at this scale).
#   2. Upload as Kaggle Dataset.
#   3. Bundle v41 with NN rollouts + v40d-style tight compute.
#   4. Push notebook + submit.

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="../../.."
VENV_PY="$REPO_ROOT/.venv/Scripts/python.exe"
VENV_GPU_PY="$REPO_ROOT/.venv-gpu/Scripts/python.exe"
KAGGLE="$REPO_ROOT/.venv/Scripts/kaggle.exe"

export KAGGLE_API_TOKEN=$(grep "KAGGLE_API_TOKEN=" "$REPO_ROOT/.env" | head -1 | cut -d= -f2-)

DEMOS="$REPO_ROOT/runs/v41_demos.npz"
[ -f "$DEMOS" ] || { echo "ERROR: $DEMOS not found. Run tools/collect_mcts_demos.py first." >&2; exit 1; }

echo "=== Stage 1: Train v41 AZ from $DEMOS ==="
PYTHONPATH=src "$VENV_GPU_PY" -m tools.train_az_head \
  --demos "$DEMOS" \
  --bc-checkpoint "$REPO_ROOT/runs/az_v39_bigger.pt" \
  --out "$REPO_ROOT/runs/az_v41.pt" \
  --epochs 18 --batch-size 256 --lr 1e-3 \
  --policy-tau 1.5 --policy-eps 0.5 \
  --lambda-p 1.0 --lambda-v 1.0 \
  --unfreeze-backbone-after-epoch -1 \
  --device cuda

[ -f "$REPO_ROOT/runs/az_v41.pt" ] || { echo "Training did not produce $REPO_ROOT/runs/az_v41.pt" >&2; exit 1; }

echo "=== Stage 2: Upload v41 as Kaggle Dataset ==="
DS_DIR="submissions/orbit_wars_az_v41_dataset"
mkdir -p "$DS_DIR"
cp "$REPO_ROOT/runs/az_v41.pt" "$DS_DIR/az_v41.pt"
cat > "$DS_DIR/dataset-metadata.json" <<EOF
{
  "title": "Orbit Wars AZ v41",
  "id": "gardan4/orbit-wars-az-v41",
  "licenses": [{"name": "CC0-1.0"}],
  "description": "AZ checkpoint v41 — closed-loop iteration on top of v39_bigger backbone, retrained on demos collected with v40d's NN-driven-rollout MCTS config (rollout_policy=nn, anchor_margin=0.5, sims=48). 179k params (48ch x 4 blocks). Tau-smoothed visit-distribution targets (tau=1.5, eps=0.5)."
}
EOF
(cd "$DS_DIR" && "$KAGGLE" datasets create -p . 2>&1 | tail -3)

echo "=== Stage 3: Bundle v41 ==="
MSYS_NO_PATHCONV=1 PYTHONPATH=src "$VENV_PY" -m tools.bundle \
  --bot mcts_bot --out submissions/v41.py \
  --weights-json "$REPO_ROOT/runs/turbo_v3_20260424.json" \
  --sim-move-variant exp3 --exp3-eta 0.3 \
  --rollout-policy nn --anchor-margin 0.5 \
  --nn-dataset-path /kaggle/input/orbit-wars-az-v41/az_v41.pt \
  --total-sims 24 --hard-deadline-ms 400 --num-candidates 4

echo "=== Stage 4: Build notebook + push ==="
PYTHONPATH=src "$VENV_PY" -m tools.build_kaggle_notebook \
  --bundle submissions/v41.py \
  --slug gardan4/orbit-wars-mcts-v41 \
  --title "Orbit Wars MCTS v41" \
  --out submissions/kaggle_notebook_v41/ \
  --dataset-source gardan4/orbit-wars-az-v41

"$KAGGLE" kernels push -p submissions/kaggle_notebook_v41

echo "=== Wait for kernel COMPLETE then run: ==="
echo "  $KAGGLE competitions submit -c orbit-wars -k gardan4/orbit-wars-mcts-v41 -v 1 -f submission.py -m 'v41: closed-loop iter from v40d-style demos'"
