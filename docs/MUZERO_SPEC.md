# MuZero-style Learned Dynamics — Implementation Spec

## Why this exists

The current MCTS hits **12 sims/turn** at the 1s Kaggle budget because each rollout is `30 plies × ~5ms heuristic.act() = 150ms`. The override rate is stuck at ~9% — the heuristic anchor wins almost every turn because Q-estimates from heuristic-vs-heuristic rollouts agree with the heuristic by construction.

A learned dynamics model `D(state, action) → (next_state, reward)` runs in ~0.1ms on CPU. **Same budget → ~1000+ sims/turn.** That's the AlphaGo-Zero recipe in spirit; MuZero formalized it. Override rate then becomes search-quality-bounded, not compute-bound.

This is the single biggest architectural lever we have not yet pulled. Estimated lift on top of a competent NN: **+200-500 Elo**.

---

## High-level architecture

Three NN modules. They share a **representation encoder** that maps observation → latent state, then act on latent space (faster + smoother gradient flow).

```
obs (B, 12, 50, 50)
  │
  ▼
┌──────────────────┐
│  Representation  │  h(obs) → latent (B, 64, 50, 50)
└──────────────────┘
  │
  ▼
┌────────────────────────────┐
│  Prediction (policy+value) │  f(latent) → (logits, value)
└────────────────────────────┘  ← used at root + every leaf

  │
  │ (during MCTS rollout, per simulated ply)
  ▼
┌──────────────────────────┐
│  Dynamics                │  g(latent, action) → (next_latent, reward)
└──────────────────────────┘  ← replaces engine.step in rollouts
```

**Shapes:**
- `latent`: `(B, 64, 50, 50)` — same spatial dims as obs, channels widened.
- `action`: `(B, 8)` — one-hot over 8 ACTION_LOOKUP channels (or sum of one-hots if multiple planets fire). For simplicity, encode as 8-channel attention scalars added to specific planet cells; see §Action Encoding below.
- `reward`: `(B,)` scalar, predicted ship-count delta normalized to [-1, 1].

**Param budget:**
- Representation: 64ch×6 blocks ≈ 600k params
- Prediction: lightweight policy + value heads, ≈100k
- Dynamics: 64ch×4 blocks with action conditioning ≈ 500k

Total: **~1.2M params**. Comfortably distillable to ~500k for inline ship.

---

## Action encoding (load-bearing)

The game has a **factored** action space: each owned planet independently chooses 1 of 8 actions (or no-op). At the wire-action level, an action is a list of `(planet_id, angle, ships)` tuples.

For dynamics conditioning, we need a fixed-dim representation. Options:

1. **Spatial action map** (preferred): build a `(B, 8, 50, 50)` tensor where `action[b, ch, gy, gx] = 1` if planet at (gy, gx) fires action channel `ch`. Concatenate to latent → dynamics conv input is `(B, 64+8, 50, 50)`.
   - Pros: spatially aligned with the rest of the model, no separate planet-id embedding needed.
   - Cons: slightly larger first conv layer.

2. **Per-planet action vector + cell injection**: for each owned planet, embed (action_channel, ship_frac) → 16-dim, scatter into latent at the planet's grid cell.
   - Pros: smaller. Cons: scatter is awkward in batched tensors.

**Decision: option 1.** Trades a small fp32 layer for cleaner code.

---

## Training

### Stage 1 — Representation + Prediction joint training

Bootstrap from existing demos (post-Phantom MCTS visit demos). Standard AZ joint loss:
```
L_pred = lambda_p * CE(visit_dist, softmax(policy_head(h(obs))))
       + lambda_v * MSE(terminal_value, value_head(h(obs)))
```

Same training script as `tools/train_az_head.py`, just with the MuZero model. Reuse the existing 38k-57k demo dataset.

### Stage 2 — Dynamics training (NEW)

Self-supervised on `FastEngine` trajectories. For each `(state_t, action_t, state_{t+1}, reward_t)` tuple in our self-play games:

```
latent_t = h(state_t)
latent_t1_pred, reward_pred = g(latent_t, action_t)
latent_t1_target = h(state_t+1).detach()  # representation is frozen now

L_dyn = MSE(latent_t1_pred, latent_t1_target) + MSE(reward_pred, reward_t)
```

We don't need to predict the obs grid itself — just the latent. Reward is the per-turn ship-count delta normalized to `[-1, 1]`.

**Important**: train dynamics to N-step unrolls, not 1-step. MuZero paper uses 5-step. Each unroll predicts the next latent given the previous predicted latent (not the ground-truth latent). This forces the dynamics model to be self-consistent over multi-step rollouts.

```python
# 5-step unroll loss
latent = h(state_0)
loss = 0
for k in range(5):
    latent_pred, reward_pred = g(latent, actions[k])
    latent_target = h(states[k+1]).detach()
    loss += MSE(latent_pred, latent_target) + MSE(reward_pred, rewards[k])
    latent = latent_pred  # use PREDICTED latent, not target
```

### Stage 3 — Joint AZ training with dynamics in the loop

The full MuZero closed-loop:
1. Self-play with current model: rollouts use `g` (learned dynamics) instead of `engine.step`.
2. Collect `(state, mcts_visit_dist, terminal_value, action_seq)` demos.
3. Train all three modules jointly. Loss combines `L_pred` (root + every k-step prediction) + `L_dyn` (k-step latent + reward consistency).
4. Repeat.

Expected wall: ~3-4h per iter on A100. 5-10 iters total = ~$50-100 cloud.

---

## MCTS integration

Replace the rollout dispatch in `gumbel_search.py` with a **learned-rollout** path that uses `g` instead of `engine.step` for all rollout plies after the first.

### Current code path (heuristic rollout, simplified)

```python
# In gumbel_search.py::_rollout_value
state = base_state.copy()
for _ in range(rollout_depth):
    obs = engine.observation(state)
    my_action = my_agent.act(obs)
    opp_action = opp_agent.act(obs)
    state = engine.step(state, [my_action, opp_action])
return value_at(state)
```

### New code path (MuZero rollout)

```python
# In gumbel_search.py::_muzero_rollout
latent = repr_net(base_obs)
total_reward = 0
for k in range(rollout_depth):
    logits, _ = pred_net(latent)
    action = argmax(logits)  # or softmax-sample
    latent, reward = dyn_net(latent, action)
    total_reward += discount**k * reward
_, value = pred_net(latent)
return total_reward + discount**rollout_depth * value
```

A single rollout is **O(rollout_depth × dyn_forward_time) = O(15 × 0.5ms) = ~8ms**. At 850ms search budget, that's **~100 rollouts/turn**. Up from 12 with heuristic rollouts.

### Batched rollouts (the actual win)

Even better: batch ALL N candidate joints in a single forward pass. With `batch_size=N_candidates`, dynamics forward processes them simultaneously. Wall-clock for N=24 candidates × 15 plies = single forward of `(24, 64+8, 50, 50)` tensor through the dynamics conv — also ~8ms total. **24× more useful info per ms.**

This is the actual ceiling-breaker: **batched MuZero rollouts** scale near-flat with candidate count.

---

## File layout (what to create)

```
src/orbitwars/nn/
  muzero/
    __init__.py
    representation.py    # h(obs) → latent
    prediction.py        # f(latent) → (policy, value)
    dynamics.py          # g(latent, action) → (next_latent, reward)
    full_model.py        # combines the 3 + provides .root() / .recurrent() entry points
    train.py             # joint training script

src/orbitwars/mcts/
  muzero_search.py       # parallel to gumbel_search.py, uses dyn for rollouts

tools/cloud/
  train_dynamics.py      # offline training of the dynamics module
  run_muzero.sh          # closed-loop MuZero orchestrator (parallel to run_closed_loop.sh)
```

Skeleton modules are in this commit at the locations above. They have full module docstrings explaining the NN architecture, but the `forward()` bodies are stubs that raise `NotImplementedError`. Filling them in is the implementation work; the design decisions are pinned.

---

## Implementation order (estimate: 1 week focused)

| Day | Task |
|---|---|
| 1 | `representation.py` + `prediction.py` — these are basically `ConvPolicy` repackaged. Should compile + smoke immediately. |
| 2 | `dynamics.py` — write architecture, action encoding, forward pass. Test in isolation: confirm shape contracts hold for batch=1, batch=N. |
| 3 | `train_dynamics.py` — load existing demo `(x, action, next_x, reward)` trajectories from `runs/v42_demos.npz`. Train 1-step first. Confirm latent-MSE drops. |
| 4 | Extend training to N-step unroll. This is where MuZero papers are most prescriptive — copy their exact recipe. |
| 5 | `muzero_search.py` — the MCTS replacement. Plumb dyn_net + pred_net through. Smoke test: 1 game vs heuristic, confirm rollout times <10ms. |
| 6 | `run_muzero.sh` — closed-loop orchestrator. Test 1 iter (100 games). |
| 7 | Bug-fix + first 5-iter cloud run. |

---

## Risks + mitigations

1. **Dynamics divergence**: latent representation drifts during multi-step unrolls, predicted states become unrealistic. *Mitigation*: stop-gradient on target latents at each unroll step (already in the loss formulation). MuZero paper uses LayerNorm at every layer to stabilize this.

2. **Reward misprediction**: per-turn ship-count deltas are sparse (most turns nothing dies). *Mitigation*: split reward into "smooth" (production accumulation) and "jump" (combat) components, predict separately. Or just normalize aggressively and hope.

3. **MCTS state mismatch**: real engine state and learned latent diverge under repeated rollouts. *Mitigation*: at the root, always use real `engine.step` for the first few plies, then switch to learned dynamics for deep rollouts. Hybrid approach used in many MuZero variants.

4. **Action encoding doesn't generalize**: factored actions might confuse the dynamics. *Mitigation*: spatial action map (option 1 above) should work because each planet's action is local to its grid cell — the conv layers see the exact spatial coupling.

---

## Success criteria

- After Stage 2 (dynamics-only training): predicted latent matches `h(real_next_state)` to MSE < 0.05 on a held-out validation set.
- After Stage 3 (full joint training): a v51 bundle using muzero rollouts wins ≥60% wr vs v41 in 16-game H2H.
- Stretch: muzero rollouts at 200 sims/turn beat heuristic rollouts at 12 sims/turn in head-to-head Elo by **+150-300**.

---

## What we DON'T copy from the MuZero paper

- **No reanalysis** — the paper re-runs MCTS on stored states with the new model. We can add this later if needed; first iter without is fine.
- **No discrete state factorization** — MuZero Atari uses categorical reward heads (51-bin distributional). We use scalar MSE because Orbit Wars rewards are roughly continuous (ship counts).
- **No model unrolling at inference past depth-15** — staying matched to current rollout_depth is enough; deeper unrolls accumulate dynamics error.
