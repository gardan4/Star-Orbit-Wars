#!/usr/bin/env bash
# One-shot setup for a Lambda Labs / RunPod / Vast.ai instance.
# Assumes Ubuntu 22+ with CUDA 12.x preinstalled (Lambda's "PyTorch 2.6"
# AMI works out of the box; RunPod's pytorch:2.6.0-cuda12.4 image does too).
#
# Usage on a fresh instance:
#   git clone <orbit-wars-repo> orbit-wars && cd orbit-wars
#   bash tools/cloud/install.sh
#
# After this, the env has:
#   * uv installed (Python project manager)
#   * project deps in .venv (incl. torch, kaggle CLI, ax/botorch)
#   * PYTHONPATH defaulting to src/
#   * KAGGLE_API_TOKEN read from .env (you put it there) or asked at runtime
#
# 5–8 min total on a Lambda A100 instance.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
echo "Installing into $ROOT"

# Quick sanity: not already running?
if [ -d ".venv" ] && [ -f ".venv/bin/python" ]; then
  echo "WARN: .venv already exists. Skipping creation."
else
  if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  echo "Creating .venv with Python 3.13..."
  uv venv --python 3.13
fi

# Activate for this script
. .venv/bin/activate

echo "Installing project + extras..."
uv sync --extra rl --extra kaggle

# Sanity: torch sees the GPU?
python - <<'PY'
import torch
assert torch.cuda.is_available(), "No CUDA visible — wrong instance type?"
print(f"torch={torch.__version__}  cuda={torch.version.cuda}  device={torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
PY

# Sanity: tests pass?
echo "Smoke-testing the project..."
PYTHONPATH=src python -m pytest tests/test_intercept.py tests/test_intercept_vec.py tests/test_mcts_actions.py -q --no-header 2>&1 | tail -3

echo
echo "✅ Install OK. Next:"
echo "  bash tools/cloud/run_closed_loop.sh --help"
