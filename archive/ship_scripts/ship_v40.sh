#!/usr/bin/env bash
# One-shot v40 ship: bundle, build notebook, push, submit.
# Run from worktree root after runs/az_v39_bigger.pt exists.
set -euo pipefail

cd "$(dirname "$0")/.."

REPO_ROOT="../../.."
VENV_PY="$REPO_ROOT/.venv/Scripts/python.exe"
KAGGLE="$REPO_ROOT/.venv/Scripts/kaggle.exe"

# Load Kaggle token from .env at the repo root.
export KAGGLE_API_TOKEN=$(grep "KAGGLE_API_TOKEN=" "$REPO_ROOT/.env" | head -1 | cut -d= -f2-)

CKPT="$REPO_ROOT/runs/az_v39_bigger.pt"
if [ ! -f "$CKPT" ]; then
  echo "ERROR: $CKPT not found. Run tools/train_az_bigger.py first." >&2
  exit 1
fi

echo "=== Bundle v40 ==="
PYTHONPATH=src "$VENV_PY" -m tools.bundle \
  --bot mcts_bot \
  --weights-json "$REPO_ROOT/runs/turbo_v3_20260424.json" \
  --sim-move-variant exp3 --exp3-eta 0.3 \
  --rollout-policy nn \
  --anchor-margin 0.5 \
  --nn-checkpoint "$CKPT" \
  --total-sims 48 --hard-deadline-ms 800 --num-candidates 4 \
  --out submissions/v40.py \
  --smoke-test

echo "=== Build Kaggle notebook for v40 ==="
PYTHONPATH=src "$VENV_PY" -m tools.build_kaggle_notebook \
  --bundle submissions/v40.py \
  --slug gardan4/orbit-wars-mcts-v40 \
  --title "Orbit Wars MCTS v40" \
  --out submissions/kaggle_notebook_v40/

echo "=== Push v40 to Kaggle ==="
"$KAGGLE" kernels push -p submissions/kaggle_notebook_v40

echo
echo "Wait for kernel to finish, then submit with:"
echo "  $KAGGLE competitions submit -c orbit-wars -k gardan4/orbit-wars-mcts-v40 -v 1 -f submission.py -m 'v40: NN-driven rollouts, 250k AZ backbone (tau-smoothed), exp3, anchor_margin=0.5'"
