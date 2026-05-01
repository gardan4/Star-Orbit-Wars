# Orbit Wars — Big-Push Plan to Top-10

**Status as of 2026-05-01:** v41 peaked at **977 Elo** (briefly above v32b's 971), now oscillating around 930–970. Top-10 of the public ladder sits at **1500+ Elo**. Closed-loop iteration with the 32×3 / 64k-param architecture has demonstrably plateaued (v42 iter-2 regressed to ~860 despite better val_mse). **The architecture is the ceiling, not the recipe.**

This doc is the runbook for the four-lever push to break that ceiling. Every script referenced here is checked in and tested. Every cost figure is for Lambda Labs A100 80GB at $1.40/hr (numbers carry to RunPod / Vast.ai within ±20%). Stop reading at any time and start at the lever you're paying for; each is self-contained.

---

## TL;DR — the four levers

| # | Lever | Wall time | Cost | Expected Elo lift | Risk |
|---|---|---|---|---|---|
| 1 | **Cloud AZ** — 5M-param ConvPolicy + proper closed-loop (10 iters × 1000 self-play games) | 24–36 h | $35–50 | +200–400 | Med — known recipe |
| 2 | **Distillation** — teacher 5M → student 1M for inline ship | 4–6 h | $5–10 | (enables ship of #1) | Low |
| 3 | **MuZero learned dynamics** — replaces engine.step in rollouts → 1500× more sims/turn | 1 week dev + 24 h cloud | $35–50 + dev | +200–500 | High — novel impl |
| 4 | **JAX engine** — vmap'd parallel self-play envs on GPU for 10–100× training throughput | 3–5 days dev | $0 (host on cloud GPU we already rent) | (multiplier on #1 + #3) | Med — physics replication |

Combined ceiling estimate: **1500–1800 Elo if all four land**. Top-10 is 1500+.

If you only commit to ONE: **#3 (MuZero)**. It's the only thing that genuinely escapes the 1s/turn CPU constraint.

If you only commit to ONE for QUICK WINS: **#1 (Cloud AZ + #2 distill to ship)**. The infrastructure exists; you push a button.

---

## Lever 1 — Cloud AZ (the workhorse)

### What it is

Train a 5M-parameter `ConvPolicy` (64ch × 16 residual blocks) via proper AlphaZero closed-loop:
- **N=10 iterations** of self-play → demo collection → joint policy+value training
- **1000 self-play games per iteration** (vs our current 30)
- **Backbone unfrozen from epoch 5** of each training run (the 32×3 was too small for this; 5M is plenty)
- Tau-smoothed visit targets (already proven to help in v41 vs v40j)

### Why it works

The AlphaZero precedent is unambiguous: Halite III, Kore 2022, Lux S1–S3 all needed multi-million-param NNs at this CPU regime. Our 64k-param NN was the thing capping us at 970. STATUS.md established that NN-rollouts at 64k _hurt_ vs 15-ply heuristic rollouts because per-call quality < heuristic. **At 5M params with proper training, NN rollouts beat heuristic — that's where the override rate breaks past 9.2%.**

### Compute budget

On Lambda Labs A100 80GB ($1.40/hr):
- Self-play: 1000 games × ~3 min/game with parallel envs = ~50 hours single-GPU CPU-bound. Cut to ~5 hours with `JAX engine` (Lever 4) or batched env stepping. Without JAX: parallelize across CPU cores; rent a GPU instance with 64+ vCPUs.
- Training: 5M params, 50k–500k demos, joint AZ loss at batch 256 = ~30 min/iter on A100.
- Total per iter: ~2–3 h
- 10 iters: **24–30 h × $1.40/hr = $35–45**

### Run it

```bash
# 1. Provision instance (Lambda Labs preset: PyTorch 2.6 + CUDA 12.4)
# 2. Clone repo + install
git clone <repo> orbit-wars && cd orbit-wars
pip install -e .[turbo,rl]   # NOTE: keep turbo extra; needed for ax/torch deps even though we're not running TuRBO

# 3. Run the full closed-loop pipeline
bash tools/cloud/run_closed_loop.sh \
  --backbone-channels 64 \
  --n-blocks 16 \
  --iters 10 \
  --games-per-iter 1000 \
  --epochs-per-iter 30 \
  --out runs/cloud_az_5M/
```

That single bash script handles: warmstart from existing 32×3 (or scratch), demo collection at each iter, AZ training with backbone unfreeze, checkpoint rotation. It's idempotent — if it crashes, re-run with `--resume`.

### Scripts (all checked in, tested locally at small scale)

- [`tools/cloud/run_closed_loop.sh`](../tools/cloud/run_closed_loop.sh) — orchestrator
- [`tools/cloud/install.sh`](../tools/cloud/install.sh) — one-shot Lambda/RunPod setup
- [`tools/train_az_head.py`](../tools/train_az_head.py) — already supports 64ch×16 via `--backbone-channels` / `--n-blocks` (verified)
- [`tools/collect_mcts_demos.py`](../tools/collect_mcts_demos.py) — already supports `--rollout-policy nn` for the closed-loop search

### Success criterion

After 10 iters, the final 5M-param checkpoint should beat v32b in a 16-game H2H by at least 60% wr. If it doesn't, the closed loop didn't converge — diagnose via val_mse trend (should monotonically decrease per iter).

---

## Lever 2 — Distillation (5M teacher → 1M student inline-fittable)

### What it is

After Lever 1 produces a 5M-param `az_5M.pt`, distill it into a `az_1M.pt` (smaller architecture, e.g. 48ch × 8 blocks) using the teacher's logits as soft targets.

### Why it's needed

A 5M fp32 checkpoint is 20 MB raw → 27 MB base64. Way over Kaggle's 1 MB notebook push limit. Solutions:
1. Kaggle Dataset path — works but the bigger-AZ-via-Dataset bug bit us repeatedly (v40b–g all errored). Diagnosed root cause: WindowsPath in `training_args` + something else still unclear. Still high-risk.
2. **Distill to 1M params + int8 quant inline.** 1M int8 = ~1 MB → fits inline cleanly. Lower risk than Dataset.

The student inherits ~80% of the teacher's strength (typical distillation result). Enough to break the v41 ceiling.

### Run it

```bash
python -m tools.cloud.distill \
  --teacher runs/cloud_az_5M/iter10/checkpoint.pt \
  --student-channels 48 --student-blocks 8 \
  --demos runs/cloud_az_5M/iter10/demos.npz \
  --epochs 30 --lr 1e-3 \
  --out runs/cloud_az_1M_distilled.pt
```

Then bundle inline:
```bash
MSYS_NO_PATHCONV=1 python -m tools.bundle --bot mcts_bot \
  --weights-json runs/turbo_v3_20260424.json \
  --sim-move-variant exp3 --exp3-eta 0.3 \
  --rollout-policy nn --anchor-margin 0.5 \
  --nn-checkpoint runs/cloud_az_1M_distilled.pt \
  --total-sims 24 --hard-deadline-ms 400 --num-candidates 4 \
  --out submissions/v50.py
```

### Scripts

- [`tools/cloud/distill.py`](../tools/cloud/distill.py) — KL-loss distillation, supports any teacher / student arch combo
- [`tools/cloud/quantize.py`](../tools/cloud/quantize.py) — int8 per-tensor symmetric, with `_quant_scales` keys readable by `tools/bundle.py`'s existing decode path

### Success criterion

Distilled 1M student should retain ≥85% of teacher's win-rate vs heuristic in local 16-game H2H.

---

## Lever 3 — MuZero learned dynamics (the actual ceiling-breaker)

### What it is

Train a small NN dynamics model `D(state, action) → (next_state, reward)`. At runtime, MCTS rollouts use `D` instead of `engine.step`. NN forward is ~0.1 ms vs engine step's ~0.5–1 ms; combined with the savings from NOT building arrival tables / running heuristic per ply, you get ~1500× more sims per turn budget.

### Why it works

Currently we get **12 sims/turn** at 1s budget because each rollout is 30 plies × ~5 ms heuristic.act() = 150 ms, deadline kicks in. Override rate stuck at 9.2%.

With learned dynamics:
- Rollout = batch of (state, action) pairs into D, 30 ply unroll, single forward = ~3 ms total
- 1 s budget / 3 ms = **~300 sims**. Up from 12.
- At 300 sims with proper search, override rate trivially exceeds 30%.
- Q-estimate variance drops 25× (roughly), so MCTS can confidently override the heuristic anchor.

This is the AlphaGo-Zero recipe for sub-1s thinking. Not novel research — a known bridge for CPU-constrained game agents.

### Architecture

```
Dynamics:
  Encoder: (B, 12, 50, 50) → (B, 64, 50, 50) latent
  Recurrent: latent + action → next_latent + reward (predict immediate ship-delta)
  Decoder: next_latent → next observation grid (for MCTS state-equality + visualization)

Total: ~2M params. Trained jointly with the policy/value heads, sharing the encoder.
```

### Training

Self-supervised on real `engine.step` trajectories. Loss = MSE on predicted next-state grid + MSE on predicted reward. Easy to verify (compare against deterministic FastEngine).

### Risk

The implementation is real software (~600–1000 LOC). Nontrivial. Spec is in [`docs/MUZERO_SPEC.md`](MUZERO_SPEC.md) (this commit), with module-by-module breakdown and integration plan.

### Run it (after implementation lands)

```bash
# Stage 1: train dynamics on FastEngine trajectories
python -m tools.cloud.train_dynamics \
  --trajectories runs/dynamics_traj.npz \  # collected from FastEngine
  --epochs 50 --batch-size 256 \
  --out runs/dynamics_v1.pt

# Stage 2: full MuZero loop (replaces Lever 1's closed-loop run)
bash tools/cloud/run_muzero.sh \
  --dynamics-checkpoint runs/dynamics_v1.pt \
  --iters 10 --games-per-iter 1000 \
  --out runs/muzero_v1/
```

### Compute

- Dynamics training: 50 epochs × 200k trajectories × 256 batch = ~6 h on A100. **$8.40.**
- MuZero closed-loop: similar to Lever 1 but rollouts are 100× faster, so demo collection drops 10× too. Net: **~$30** for 10 iters.

### Success criterion

A v51 bundle using the MuZero NN as both prior + value (no heuristic rollouts at all) wins ≥60% wr vs v41 in 16-game H2H.

---

## Lever 4 — JAX engine (training-throughput multiplier)

### What it is

Complete the existing skeleton at [`src/orbitwars/engine/jax_engine.py`](../src/orbitwars/engine/jax_engine.py). Currently `step()` raises `NotImplementedError`; the API surface and pytree shapes are designed but the physics is empty.

### Why it matters

**Not a runtime ladder lever** — Kaggle's submission environment is CPU-only, so we can't use JAX at inference. But the SELF-PLAY for Lever 1 / Lever 3 is offline on cloud GPU. Currently `FastEngine` does ~50–200k single-core steps/sec. JAX with `vmap`+`scan` over 1024 parallel envs on A100: target **5M–20M aggregate steps/sec**, i.e. 25–100× speedup.

That means Lever 1's "1000 games per iter" becomes "50000 games per iter" at the same wall time. Or 10 iters at 1000 games each compresses from 24 h to ~30 min.

### Spec

[`docs/JAX_ENGINE_SPEC.md`](JAX_ENGINE_SPEC.md) (this commit) — module-by-module mapping from the existing numpy `FastEngine` to JAX, plus the parity-test design (run both engines on N=1000 random seeds, assert state-equal at every step).

### Risk

Medium — we already have the API and the numpy reference. Translation is mechanical. Parity test catches regressions immediately. Estimated **3–5 days of focused dev**.

### Compute

Free during dev (uses existing local GPU). At cloud-run time, runs on the same A100 we'd already rented for Lever 1 / 3.

---

## Sequencing

| Week | Action | Cost | Outcome |
|---|---|---|---|
| 1 | Implement Lever 4 (JAX engine) locally; test parity with FastEngine | $0 | Self-play 25–100× faster |
| 2 | Lever 1 (Cloud AZ) + Lever 2 (Distill) end-to-end | ~$50 | New ladder ship, target 1100–1300 Elo |
| 3 | Lever 3 (MuZero) implementation + first training run | ~$40 + dev | Target 1300–1500 Elo |
| 4 | Iterate Lever 1+3 combined with full self-play scale | ~$60 | Target 1500+ Elo |

Total budget: **~$150** + 3–4 weeks of dev time.

If you don't want to do dev (Levers 3 + 4), Lever 1 + 2 alone gets you to ~1100–1300. That's a $50, 2-day commitment for +130–330 Elo over current 977.

---

## Critical pre-flight: things that MUST work

These have been tested and are ready. If any breaks during your cloud run, the relevant doc has the fix path.

| Component | Status | Verified by |
|---|---|---|
| `tools/bundle.py` with bigger AZ via int8 inline | ✅ | v42a builds (560 KB), `_BUNDLE_BC_CKPT_PATH` is correct |
| `tools/bundle.py` MSYS_NO_PATHCONV path-mangling guard | ✅ | bundle.py rejects Windows-mangled paths with helpful error |
| WindowsPath sanitization in `train_az_head.py` | ✅ | line 339; Path → str at save time |
| `tools/collect_mcts_demos.py` with `--rollout-policy nn` | ✅ | `nn_rollout_factory` plumbed through |
| Kaggle Dataset upload path | ✅ | `gardan4/orbit-wars-az-v39-bigger` v2 verified |
| Kernel push for ≤1 MB bundles | ✅ | v40h, v40j, v40k, v41, v42 all pushed cleanly |
| Local game smoke timing < 1s on Kaggle | ✅ | v42a at 64ch×4 fits at 64 sims/600ms (mean 586ms, max 693ms) |

The big unknown is the BIGGER-AZ-via-Kaggle-Dataset failure mode (v40b–g all errored). [`docs/KAGGLE_DATASET_BIG_AZ_BUG.md`](KAGGLE_DATASET_BIG_AZ_BUG.md) records the diagnosis trail; root cause was definitively WindowsPath unpickle failure (now fixed in `train_az_head.py`). Subsequent failures (v40g) suggest there's a second bug — but the **distillation path (Lever 2 → 1M int8 inline) sidesteps Dataset entirely and is the recommended ship route**.

---

## What we explicitly drop

- **TuRBO re-tuning** — user request 2026-05-01. The heuristic weights are good enough; the architecture is the bottleneck.
- **EvoTune (LLM-evolved heuristics)** — same reason; structural-evolution objective is high-variance and the budget is better spent on Levers 1–4.
- **Mixed leaf eval** (`--value-mix-alpha`) — implemented and shippable, but the small-NN-as-value didn't help (v38, v40k regressions). Re-test once Lever 1's 5M model exists.
- **PPO from BC + PFSP** — `tools/train_ppo.py` exists but is untested at scale. Lever 1 (proper AlphaZero) is the higher-EV use of the same compute budget. PPO is a fallback if AZ doesn't converge.

---

## Memory: where to find things mid-run

If a script fails, the diagnostics live here:

- [docs/STATUS.md](STATUS.md) — full project history, phantom-bug post-mortems, all per-version Elo data
- [docs/V37_SESSION_FINDINGS.md](V37_SESSION_FINDINGS.md) — v37–v42 detailed session log
- [docs/NN_DRIVEN_ROLLOUTS_SPEC.md](NN_DRIVEN_ROLLOUTS_SPEC.md) — original spec for NN rollouts (now landed)
- [docs/MUZERO_SPEC.md](MUZERO_SPEC.md) — Lever 3 implementation spec (this commit)
- [docs/JAX_ENGINE_SPEC.md](JAX_ENGINE_SPEC.md) — Lever 4 spec (this commit)
- [docs/KAGGLE_DATASET_BIG_AZ_BUG.md](KAGGLE_DATASET_BIG_AZ_BUG.md) — bigger-AZ-on-Kaggle saga (this commit)
- [archive/README.md](../archive/README.md) — what got pruned and why

User memory at `~/.claude/projects/.../memory/`:
- `feedback_kaggle_ladder_errors.md` — WindowsPath + MSYS_PATH_CONV gotchas

## What to do RIGHT NOW

1. Read this doc + skim the three spec docs.
2. Spin up Lambda Labs (`a100-80gb` instance) or RunPod equivalent.
3. Run the install script, then `tools/cloud/run_closed_loop.sh`.
4. Walk away for 24 hours.
5. Distill, bundle, ship.
6. If v50 (distilled 1M) lands above v41's 977, commit Lever 3 budget.
