# PPO from BC v3 — Design Notes

Planning doc for `tools/train_ppo.py`. Drafted 2026-04-25 evening from
public Kaggle PPO tutorial (`kashiwaba/orbit-wars-reinforcement-learning-tutorial`)
+ our existing infrastructure.

## Why

Public leaderboard reality check (writeup §10.1): tier 2 starts at 1300
Elo, we're at ~880. +50 Elo per H2H ablation step is too slow. PPO
self-play is the recipe public competitors are demonstrably using to
clear the baseline; our BC v3 prior is a strong warmstart.

## Architecture choice: reuse ConvPolicy, not a new MLP

The Kaggle tutorial uses a small MLP (~20k params) on scalar features.
We have BC v3 (~460k-param ConvPolicy) trained on 107k MCTS-distilled
demos with val_acc 0.434 (3.5× random baseline). Throwing that away to
start from random-init MLP would waste a day of training.

Plan: **finetune ConvPolicy via PPO**, starting from the BC v3
checkpoint. The ConvPolicy already has `policy_head` and `value_head`;
both produce per-cell outputs that PPO can read at the source-planet
cell.

Action space stays the same as today's BC training: 8 channels per
cell = 4 angle buckets × 2 ship-fraction buckets. Heuristic computes
the actual angle from `(angle_bucket, source, nearest_target_in_quadrant)`
via `intercept.orbiting_intercept` and ship count from
`max(target_ships + 1, 20)`. The policy ONLY outputs the categorical
distribution over 8 channels per source planet.

## Loop structure

```text
for update in range(total_updates):
  rollouts = []
  for env_idx in range(num_envs):
    state = engine.fresh_state(seed)
    for step in range(rollout_steps):
      for player in (my_seat, opp_seat):
        obs = state.obs(player)
        for src_planet in obs.my_planets:
          gy, gx = grid_cell(src_planet)
          x_grid = encode_grid(obs, player)
          logits[src_planet] = ConvPolicy(x_grid)[gy, gx]  # (8,)
          if player == my_seat:
            sample action via Categorical(logits)
            log_prob, entropy
          else:
            opponent = pfsp_pool.sample()
            action = opponent.act(obs)
        step engine with both actions

  compute GAE advantages (γ=0.995, λ=0.95)
  PPO update over rollouts (4 epochs, minibatch=256, clip=0.2)
  if update % sync_interval: sync opponent weights
  every save_interval: write checkpoint
```

## Self-play opponent pool (PFSP)

Plan §"Path C" calls for **prioritized fictitious self-play**. Pool:
- `archetypes.RUSHER`, `TURTLER`, `ECONOMY`, `HARASSER`,
  `COMET_CAMPER`, `OPPORTUNIST`, `DEFENDER` (already shipped)
- Frozen heuristic baseline (TuRBO-v3 weights)
- Last K self-play checkpoints (K=10, sampled with PFSP weight ∝
  loss-rate against current learner)
- Live learner itself (50% of matches)

PFSP weighting: `p(x) ∝ (1 - win_rate(learner, x))^p` with p=2 so we
play opponents that are "just hard enough" but not impossible.

## Reward shaping

- **Terminal**: +1 win / -1 loss / 0 tie (engine reward).
- **Shaped**: optional ship-lead delta per turn:
  `shaped_r_t = (my_ships(t) - opp_ships(t)) / total_ships(t) * 0.01`
  Scaled small so terminal dominates. Set 0 by default; turn on if
  pure-terminal training stalls.

## Data efficiency targets

- 1M env steps for a usable signal (BC v3 already gives competence)
- 20M steps to clear v15's 879 ladder Elo (best guess)
- 100M steps to reach tier 2 (~1300 ladder Elo)
- FastEngine throughput ≈ 50k steps/sec per core × 8 cores = 400k/sec
  → 20M / 400k = 50 sec wall (optimistic, may be 5-10× slower under
  PPO update overhead)

## File layout

- `src/orbitwars/nn/ppo_algo.py` — `sample_actions`, `ppo_update`,
  `compute_gae` (algorithm primitives, no I/O).
- `src/orbitwars/nn/ppo_features.py` — wraps `obs_encode.encode_grid`
  + per-source-planet decision unrolling (matches tutorial's
  `encode_turn` shape but uses our spatial features).
- `src/orbitwars/nn/pfsp_pool.py` — opponent pool with sampling and
  win-rate update.
- `tools/train_ppo.py` — top-level loop. CleanRL-style: one file,
  config dataclass, all in one place. Loads BC v3 checkpoint,
  trains, saves checkpoints + curve.
- `tools/eval_ppo.py` — runs trained checkpoint vs frozen v15 in
  20-game H2H. Same shape as `diag_bundle_round_robin`.

## Test plan

- Unit: `ppo_algo` primitives (sample_actions matches torch.distributions,
  GAE numeric correctness, PPO update changes weights monotonically
  on a hand-crafted positive-reward batch).
- Integration: `tools/train_ppo.py --total-updates 5 --num-envs 2` —
  smoke test, completes in <2 min, asserts loss decreases.
- Eval gate: PPO checkpoint must beat BC v3 (no PPO) by ≥0.55 wr
  over 20 paired-seed H2H games before shipping a v22.

## Risk register

1. **Convergence**: PPO is finicky. Mitigations: BC warmstart (already
   functional), small lr (3e-4), strong baseline value head pre-trained.
2. **Speed**: FastEngine self-play in pure Python may be slow for PPO
   sample volume. Backstop: drop to 4 envs + 16 rollout steps =
   ~300 transitions per update, 100 updates per hour.
3. **PPO ↔ BC priors mismatch**: PPO may un-learn the BC prior in
   early updates if reward signal is weak. Mitigation: KL penalty
   against BC v3 distribution for first N updates; standard "trust
   region from BC" trick.
4. **Kaggle ship size**: PPO net is 460k params (same as BC v3 big);
   ship via int8 quantization — already shipped in v20i path.

## What we're NOT doing (yet)

- Full league training à la AlphaStar (too compute-hungry)
- World models / MuZero (too compute-hungry; no signs the public
  leaderboard needs it)
- Distributed training across multiple machines (single A10/3070 fine)
- Decision Transformer / offline RL (we already have BC for that)

## Next concrete actions tomorrow

1. (~2h) Implement `ppo_algo.py` + tests.
2. (~2h) Implement `pfsp_pool.py` + smoke test sampling.
3. (~3h) Implement `train_ppo.py` shell that runs 5 updates as a smoke.
4. (~30m) Hook up BC v3 checkpoint as warmstart.
5. (~1h) Launch a 100-update background run to test convergence.
6. **Decision point**: if loss decreases monotonically and entropy
   stays > 0.5, schedule a 2000-update overnight run.
