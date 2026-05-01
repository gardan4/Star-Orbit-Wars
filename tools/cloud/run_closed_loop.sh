#!/usr/bin/env bash
# Cloud closed-loop AlphaZero runner. Designed for Lambda Labs / RunPod
# A100 80GB instances. Idempotent — run again with --resume to pick up.
#
# Pipeline per iteration:
#   1. Self-play with current agent (collect_mcts_demos.py)
#      → runs/<out>/iter<N>/demos.npz
#   2. Joint AZ training (train_az_head.py) starting from previous iter's ckpt
#      → runs/<out>/iter<N>/checkpoint.pt
#   3. Update "current agent" pointer to the new ckpt
#
# After --iters runs, the final ckpt at runs/<out>/iter<N>/checkpoint.pt is
# the teacher. Distill it via tools/cloud/distill.py to ship inline.

set -euo pipefail

# ---- defaults ----
BACKBONE_CHANNELS=64
N_BLOCKS=16
ITERS=10
GAMES_PER_ITER=1000
EPOCHS_PER_ITER=30
SIMS=64                 # MCTS sims per turn for self-play
DEADLINE_MS=800
WARMSTART=""            # path to existing .pt to seed iter-0 (else BC from random init)
OUT="runs/cloud_az"
RESUME=""               # if set, pick up at the highest existing iter
DEVICE="cuda"
WORKERS=8               # parallel self-play workers (CPU cores)
POLICY_TAU=1.5
POLICY_EPS=0.5
LAMBDA_P=1.0
LAMBDA_V=1.0
UNFREEZE_AFTER=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backbone-channels) BACKBONE_CHANNELS="$2"; shift 2 ;;
    --n-blocks)          N_BLOCKS="$2"; shift 2 ;;
    --iters)             ITERS="$2"; shift 2 ;;
    --games-per-iter)    GAMES_PER_ITER="$2"; shift 2 ;;
    --epochs-per-iter)   EPOCHS_PER_ITER="$2"; shift 2 ;;
    --sims)              SIMS="$2"; shift 2 ;;
    --deadline-ms)       DEADLINE_MS="$2"; shift 2 ;;
    --warmstart)         WARMSTART="$2"; shift 2 ;;
    --out)               OUT="$2"; shift 2 ;;
    --resume)            RESUME=1; shift ;;
    --device)            DEVICE="$2"; shift 2 ;;
    --workers)           WORKERS="$2"; shift 2 ;;
    --policy-tau)        POLICY_TAU="$2"; shift 2 ;;
    --policy-eps)        POLICY_EPS="$2"; shift 2 ;;
    --unfreeze-after)    UNFREEZE_AFTER="$2"; shift 2 ;;
    --help|-h)
      grep -E "^# " "$0" | sed 's/^# //'
      exit 0 ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p "$OUT"

# Prefer the cloud venv if it exists, else fall back to local venv-gpu
if [ -f ".venv/bin/python" ]; then
  PY=".venv/bin/python"
elif [ -f ".venv-gpu/Scripts/python.exe" ]; then
  PY=".venv-gpu/Scripts/python.exe"
else
  echo "No venv found. Run tools/cloud/install.sh first." >&2
  exit 1
fi

export PYTHONPATH="src:."

# Determine starting iter
START=0
if [ -n "$RESUME" ]; then
  for d in "$OUT"/iter*/; do
    [ -f "$d/checkpoint.pt" ] || continue
    n=$(basename "$d" | sed 's/iter//')
    if [ "$n" -gt "$START" ]; then START="$n"; fi
  done
  echo "Resuming from iter $START (next: iter $((START + 1)))"
  CURRENT_CKPT="$OUT/iter$START/checkpoint.pt"
  START=$((START + 1))
elif [ -n "$WARMSTART" ]; then
  echo "Warmstarting from $WARMSTART"
  mkdir -p "$OUT/iter0"
  cp "$WARMSTART" "$OUT/iter0/checkpoint.pt"
  CURRENT_CKPT="$OUT/iter0/checkpoint.pt"
  START=1
else
  echo "No warmstart, no resume — bootstrapping iter-0 with a random-init ckpt"
  mkdir -p "$OUT/iter0"
  $PY -c "
import torch
from orbitwars.nn.conv_policy import ConvPolicy, ConvPolicyCfg
from dataclasses import asdict
cfg = ConvPolicyCfg(backbone_channels=$BACKBONE_CHANNELS, n_blocks=$N_BLOCKS)
m = ConvPolicy(cfg)
torch.save({'model_state': m.state_dict(), 'cfg': asdict(cfg), 'az_trained_jointly': False}, '$OUT/iter0/checkpoint.pt')
print(f'Wrote bootstrap ckpt: {sum(p.numel() for p in m.parameters()):,} params')
"
  CURRENT_CKPT="$OUT/iter0/checkpoint.pt"
  START=1
fi

START_TIME=$(date +%s)
for n in $(seq $START $ITERS); do
  ITER_DIR="$OUT/iter$n"
  mkdir -p "$ITER_DIR"

  echo
  echo "============================================================"
  echo "[iter $n / $ITERS]   $(date)"
  echo "  current ckpt: $CURRENT_CKPT"
  echo "  out dir: $ITER_DIR"
  echo "============================================================"

  # ---- Stage A: self-play demo collection ----
  if [ ! -f "$ITER_DIR/demos.npz" ]; then
    echo "[$n][collect] $GAMES_PER_ITER games at sims=$SIMS deadline=$DEADLINE_MS"
    $PY -m tools.collect_mcts_demos \
      --games "$GAMES_PER_ITER" --sims "$SIMS" --deadline-ms "$DEADLINE_MS" \
      --rollout-policy nn \
      --bc-checkpoint "$CURRENT_CKPT" \
      --visit-temperature "$POLICY_TAU" --visit-smoothing "$POLICY_EPS" \
      --out "$ITER_DIR/demos.npz" \
      --seed $((1000 + n * 100))
  else
    echo "[$n][collect] demos.npz already exists; skipping"
  fi

  # ---- Stage B: joint AZ training ----
  if [ ! -f "$ITER_DIR/checkpoint.pt" ]; then
    echo "[$n][train] $EPOCHS_PER_ITER epochs from $CURRENT_CKPT"
    $PY -m tools.train_az_head \
      --demos "$ITER_DIR/demos.npz" \
      --bc-checkpoint "$CURRENT_CKPT" \
      --out "$ITER_DIR/checkpoint.pt" \
      --epochs "$EPOCHS_PER_ITER" \
      --batch-size 256 --lr 1e-3 \
      --policy-tau "$POLICY_TAU" --policy-eps "$POLICY_EPS" \
      --lambda-p "$LAMBDA_P" --lambda-v "$LAMBDA_V" \
      --unfreeze-backbone-after-epoch "$UNFREEZE_AFTER" \
      --device "$DEVICE"
  else
    echo "[$n][train] checkpoint.pt already exists; skipping"
  fi

  # Roll forward
  CURRENT_CKPT="$ITER_DIR/checkpoint.pt"
  ELAPSED=$(( $(date +%s) - START_TIME ))
  echo "[$n] elapsed: $((ELAPSED / 60)) min  ckpt: $CURRENT_CKPT"
done

# ---- Final summary ----
FINAL_CKPT="$OUT/iter$ITERS/checkpoint.pt"
echo
echo "============================================================"
echo "🎉 Closed-loop done. Final teacher: $FINAL_CKPT"
echo "Total wall: $(( ($(date +%s) - START_TIME) / 60 )) min"
echo
echo "Next steps:"
echo "  python -m tools.cloud.distill --teacher $FINAL_CKPT --student-channels 48 --student-blocks 8 --demos $OUT/iter$ITERS/demos.npz --out runs/cloud_az_distilled.pt"
echo "  python -m tools.cloud.quantize --in runs/cloud_az_distilled.pt --out runs/cloud_az_distilled_int8.pt"
echo "  MSYS_NO_PATHCONV=1 python -m tools.bundle --bot mcts_bot --rollout-policy nn --nn-checkpoint runs/cloud_az_distilled_int8.pt ..."
echo "============================================================"
