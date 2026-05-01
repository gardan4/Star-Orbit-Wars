#!/usr/bin/env bash
# Ship a cloud-trained AZ checkpoint to Kaggle.
#
# Pipeline:
#   1. Distill teacher (5M+ params) -> student (1M params, fits inline)
#   2. Int8 quantize student
#   3. Bundle with our standard NN-priors + heuristic-rollouts config
#      (the proven v22/v41 recipe — closed-loop iter VALIDATED this beats v32b)
#   4. Build Kaggle notebook
#   5. Push notebook
#   6. Wait for kernel COMPLETE
#   7. Submit to competition
#
# Usage:
#   bash tools/cloud/ship.sh \
#       --teacher runs/cloud_az/iter10/checkpoint.pt \
#       --demos runs/cloud_az/iter10/demos.npz \
#       --version v50 \
#       --message "v50: cloud AZ + distillation"

set -euo pipefail

# ---- defaults ----
TEACHER=""
DEMOS=""
VERSION=""
MESSAGE=""
STUDENT_CHANNELS=48
STUDENT_BLOCKS=8
DISTILL_EPOCHS=30
TOTAL_SIMS=128
DEADLINE_MS=850
ROLLOUT_POLICY="heuristic"  # priors-only is the proven recipe
ANCHOR_MARGIN=0.0
DRY_RUN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --teacher)            TEACHER="$2"; shift 2 ;;
    --demos)              DEMOS="$2"; shift 2 ;;
    --version)            VERSION="$2"; shift 2 ;;
    --message)            MESSAGE="$2"; shift 2 ;;
    --student-channels)   STUDENT_CHANNELS="$2"; shift 2 ;;
    --student-blocks)     STUDENT_BLOCKS="$2"; shift 2 ;;
    --distill-epochs)     DISTILL_EPOCHS="$2"; shift 2 ;;
    --total-sims)         TOTAL_SIMS="$2"; shift 2 ;;
    --deadline-ms)        DEADLINE_MS="$2"; shift 2 ;;
    --rollout-policy)     ROLLOUT_POLICY="$2"; shift 2 ;;
    --anchor-margin)      ANCHOR_MARGIN="$2"; shift 2 ;;
    --dry-run)            DRY_RUN=1; shift ;;
    --help|-h)
      grep -E "^# " "$0" | sed 's/^# //'
      exit 0 ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

[ -n "$TEACHER" ] || { echo "ERROR: --teacher required" >&2; exit 1; }
[ -n "$DEMOS" ] || { echo "ERROR: --demos required" >&2; exit 1; }
[ -n "$VERSION" ] || { echo "ERROR: --version required (e.g. v50)" >&2; exit 1; }
[ -n "$MESSAGE" ] || MESSAGE="$VERSION: cloud AZ + distillation, $(date +%Y-%m-%d)"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [ -f ".venv/bin/python" ]; then
  PY=".venv/bin/python"; KAGGLE=".venv/bin/kaggle"
elif [ -f ".venv-gpu/Scripts/python.exe" ]; then
  PY=".venv-gpu/Scripts/python.exe"; KAGGLE=".venv/Scripts/kaggle.exe"
else
  echo "No venv found." >&2; exit 1
fi

if [ -f .env ]; then
  export KAGGLE_API_TOKEN=$(grep "KAGGLE_API_TOKEN=" .env | head -1 | cut -d= -f2-)
fi

DISTILLED="runs/${VERSION}_distilled.pt"
INT8="runs/${VERSION}_distilled_int8.pt"
BUNDLE="submissions/${VERSION}.py"
NOTEBOOK_DIR="submissions/kaggle_notebook_${VERSION}"
SLUG="gardan4/orbit-wars-mcts-${VERSION}"

echo "=========================================="
echo "  SHIP $VERSION"
echo "=========================================="
echo "  teacher: $TEACHER"
echo "  demos:   $DEMOS"
echo "  student: ${STUDENT_CHANNELS}ch x ${STUDENT_BLOCKS} blocks"
echo "  distill: $DISTILL_EPOCHS epochs"
echo "  bundle:  rollout=$ROLLOUT_POLICY anchor=$ANCHOR_MARGIN sims=$TOTAL_SIMS dl=${DEADLINE_MS}ms"
echo "  out:     $BUNDLE"
echo "=========================================="
[ -n "$DRY_RUN" ] && { echo "(dry-run, exiting)"; exit 0; }

export PYTHONPATH="src:."

echo
echo "[1/6] Distilling..."
$PY -m tools.cloud.distill \
  --teacher "$TEACHER" \
  --demos "$DEMOS" \
  --student-channels "$STUDENT_CHANNELS" \
  --student-blocks "$STUDENT_BLOCKS" \
  --epochs "$DISTILL_EPOCHS" \
  --out "$DISTILLED"

echo
echo "[2/6] Int8 quantizing..."
$PY -m tools.cloud.quantize --in "$DISTILLED" --out "$INT8"

echo
echo "[3/6] Bundling..."
MSYS_NO_PATHCONV=1 $PY -m tools.bundle \
  --bot mcts_bot \
  --weights-json runs/turbo_v3_20260424.json \
  --sim-move-variant exp3 --exp3-eta 0.3 \
  --rollout-policy "$ROLLOUT_POLICY" \
  --anchor-margin "$ANCHOR_MARGIN" \
  --nn-checkpoint "$INT8" \
  --total-sims "$TOTAL_SIMS" --hard-deadline-ms "$DEADLINE_MS" --num-candidates 4 \
  --out "$BUNDLE" \
  --smoke-test

# Local game smoke vs heuristic to catch regressions before pushing
echo
echo "[3.5/6] Local timing test vs heuristic..."
$PY - <<PY
import sys, importlib.util, time, statistics
src = open("$BUNDLE", encoding="utf-8").read()
spec = importlib.util.spec_from_loader("ship_test", loader=None)
mod = importlib.util.module_from_spec(spec)
sys.modules["ship_test"] = mod
exec(compile(src, "$BUNDLE", "exec"), mod.__dict__)
from kaggle_environments import make
from orbitwars.bots.heuristic import HeuristicAgent
heur = HeuristicAgent().as_kaggle_agent()
orig = mod.agent
times = []
def timed(obs, *a, **k):
    t0 = time.perf_counter(); r = orig(obs, *a, **k); times.append((time.perf_counter() - t0) * 1000)
    return r
env = make("orbit_wars", configuration={"actTimeout": 1.0, "episodeSteps": 500}, debug=False)
env.reset(num_agents=2); env.run([timed, heur])
mean = statistics.mean(times); mx = max(times)
ok = mx < 950
print(f"  rewards={[s.reward for s in env.state]} steps={env.state[0].observation.step}")
print(f"  mean={mean:.0f}ms max={mx:.0f}ms (Kaggle ~1.5x slowdown -> max~{mx*1.5:.0f}ms; budget 1000ms)")
if not ok:
    print(f"WARN: max>{950}ms locally — Kaggle will likely timeout. Reduce --total-sims or --deadline-ms.", file=sys.stderr)
PY

echo
echo "[4/6] Building Kaggle notebook..."
$PY -m tools.build_kaggle_notebook \
  --bundle "$BUNDLE" \
  --slug "$SLUG" \
  --title "Orbit Wars MCTS $VERSION" \
  --out "$NOTEBOOK_DIR/"

echo
echo "[5/6] Pushing kernel..."
"$KAGGLE" kernels push -p "$NOTEBOOK_DIR"

echo
echo "[6/6] Waiting for kernel COMPLETE..."
while true; do
  s=$("$KAGGLE" kernels status "$SLUG" 2>&1)
  echo "  $s"
  if echo "$s" | grep -qE "KernelWorkerStatus\.(COMPLETE|ERROR)"; then break; fi
  sleep 30
done

if echo "$s" | grep -q "ERROR"; then
  echo "Kernel ERRORED. See https://www.kaggle.com/code/$SLUG"
  exit 1
fi

echo
echo "[6/6] Submitting..."
"$KAGGLE" competitions submit -c orbit-wars -k "$SLUG" -v 1 -f submission.py -m "$MESSAGE"

echo
echo "=========================================="
echo "  ✅ SHIPPED $VERSION"
echo "=========================================="
echo "  Watch the ladder:"
echo "    $KAGGLE competitions submissions -c orbit-wars | head -5"
