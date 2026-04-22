# Orbit Wars

A competitive bot for the [Kaggle Orbit Wars Simulation competition](https://www.kaggle.com/competitions/orbit-wars). The core hypothesis: **heuristic-backed search beats blind RL** in the 1s/turn CPU budget regime, validated by Halite III, Kore 2022, and Lux S1–S3 precedents.

## Strategy

Four parallel research tracks:

| Track | Approach |
|-------|----------|
| **A** | Parameterized heuristic bot with TuRBO-tuned weights and LLM-evolved priority functions |
| **B** | Gumbel MCTS with kernel-aggregated continuous angles over heuristic rollouts |
| **C** | Self-play PPO + PFSP distilled to a neural prior for Track B (W4–5) |
| **D** | Online Bayesian opponent modeling over a 7-archetype portfolio |

Current submission ships A+B+D.

## Project Structure

```
src/orbitwars/
  engine/       # Numpy SoA engine + intercept math, parity-validated vs Kaggle reference
  bots/         # HeuristicAgent (A), MCTSAgent (B), FastRolloutAgent
  mcts/         # Gumbel root search, Sequential Halving, action generation
  tune/         # TuRBO/BoTorch tuner, LLM-driven EvoTune
  opponent/     # 7-archetype portfolio + Bayesian posterior (D)
  features/     # Observation → tensors (stub, W4)
  nn/           # Conv policy + set-transformer (stub, W4–5)

tournaments/    # Round-robin harness, Elo scoring
tools/          # bundle.py, profilers, smoke runners, diagnostics
submissions/    # Versioned single-file Kaggle submissions
tests/          # pytest suite (~128 tests)
runs/           # Tuning output and experiment artifacts
docs/           # Architecture status and notes
```

## Setup

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
uv sync --extra turbo   # Ax/BoTorch for TuRBO tuning (W2+)
uv sync --extra rl      # PyTorch for PPO self-play (W4+)
```

Copy `.env.example` to `.env` and set your Kaggle API token:

```
KAGGLE_API_TOKEN=KGAT_...
```

## Running

**Tests:**
```bash
PYTHONPATH=src python -m pytest tests/ -q
```

**Smoke test (MCTS vs Heuristic):**
```bash
PYTHONPATH=src python tools/smoke_mcts_default_vs_heuristic.py
```

**Multi-seed ladder (3 seeds × 2 seats):**
```bash
PYTHONPATH=src python tools/smoke_mcts_multi_seed.py
```

**TuRBO weight tuning:**
```bash
PYTHONPATH=src python -m orbitwars.tune.turbo_runner --trials 25 --games 20
```

**Bundle a submission:**
```bash
python tools/bundle.py --bot mcts_bot --out submissions/mcts_v2.py --smoke-test
```

See [submissions/README.md](submissions/README.md) for the Kaggle Notebook submission flow.

## Key MCTS Config

```python
GumbelConfig(
    num_candidates=4,
    total_sims=32,
    rollout_depth=15,
    hard_deadline_ms=300.0,
    anchor_improvement_margin=2.0,  # 2.0 = play heuristic floor
    rollout_policy="heuristic",     # swap to "fast" for ~27 sims/turn
)
```

## Current Status

See [docs/STATUS.md](docs/STATUS.md) for the full architecture breakdown, week-by-week roadmap, and known rough edges.
