# v37/v38/v40 session — vectorization, tau-smoothed AZ, mixed eval, NN-driven rollouts

## Current status (autonomous run, mid-session)

- **v37b SHIPPED** to Kaggle ladder (kernel COMPLETE; initial 600.0, settling).
  Cache-only speedup, byte-identical to v32b heuristic.
- **v38 SHIPPED but ERRORED on ladder** (kernel COMPLETE; ladder play failed,
  likely actTimeout on slower Kaggle CPU under doubled mixed-eval cost).
  Mixed-eval pathway works locally (smoke vs random in 123 steps) but the
  config + value_mix_alpha=0.5 + small AZ ckpt was too tight for Kaggle.
  Lesson: mix doubles per-leaf cost; reduce total_sims when shipping.
- **NN-driven rollouts shipped as code** (`src/orbitwars/bots/nn_rollout.py`,
  wired through gumbel_search + mcts_bot + bundle.py). 5 unit tests pass.
- **Bigger AZ training in progress** (background task; 48-channel × 4-block
  ConvPolicy = 179k params, BC stage at epoch 5/8 with va_acc=0.568).
- **v40 ship pipeline ready** (`tools/ship_v40.sh`); will fire as soon as
  training completes.

---

# v37 session — vectorization attempt + tau-smoothed AZ training

Single session work targeting two levers from the top-10 plan:
1. **Vectorized scoring refactor** (was advertised at "+50%, byte-identical, 1-2 days").
2. **Tau-smoothed AlphaZero closed-loop** (the visit-distillation fix from STATUS.md "tomorrow plan").

## What landed

### 1. Lossless intercept caches (shippable, byte-identical, ~24% rollout speedup)

Three changes in [intercept.py](src/orbitwars/engine/intercept.py) and [heuristic.py](src/orbitwars/bots/heuristic.py):

- `fleet_speed()` memoized via dict cache (632k → ~50 actual computes per profile run).
- `ParsedObs` caches `is_orbiting_by_pid` and `orbit_params_by_pid` once per turn.
- The three call sites (`build_arrival_table`, `_travel_turns`, `_intercept_position`) read from the cache; defensive `getattr` keeps test-stub objects working.

**Measured at turn-200 mid-game state, 30 rollouts × 15 plies:**

| | Baseline | Cache changes |
|---|---|---|
| Per-rollout heuristic time | 150 ms | 114 ms |
| `fleet_speed` self-time | 0.50s | 0.24s |
| `orbiting_intercept` self-time | 2.29s | 1.65s |

**Byte-identical regression check**: heuristic-vs-heuristic at seed 42 produces `rewards=[1,-1] scores=[1880,1834] steps=499` exactly — same as pre-change baseline.

Translated to v32b's 850ms search budget: 12.5 → ~16 sims/turn average. Estimated +15-30 Elo on the ladder.

### 2. Batch Newton attempt — REVERTED

Wrote [intercept_vec.py](src/orbitwars/engine/intercept_vec.py) with `orbiting_intercept_batch()` that produces float-bit-equal output to scalar Newton on 200-pair random suites (5e-15 absolute drift, see [tests/test_intercept_vec.py](tests/test_intercept_vec.py)).

Two integration attempts both failed for different reasons:

- **Per-my-planet batch (10-pair calls)**: numpy per-call overhead (250µs/call) exceeded scalar speedup (15µs/call). Net ~50% **regression** on heuristic.act() wall time.
- **Global cross-my-planet batch (250-pair single call)**: hoisted defense-check + cooldown out of main loop into precompute. Broke byte-identicality at seed 42 (1880:1834 → 2346:333). Subtle scope/state bug not worth further debugging given we already have the +24% cache win.

The batch infrastructure (intercept_vec.py + tests) is left in place for a future attempt that batches the second-pass Newton inside `_score_target` — the structurally cleaner target, since per-pair `ships_to_send` already varies and the algorithm doesn't depend on per-my-planet defense state.

**Lesson learned**: numpy vectorization needs ≥100-element batches to beat scalar libm calls in tight loops. For ~10-element batches at this small size, scalar Python is faster. Future-me: prove batch size before committing to refactor.

### 3. Tau-smoothed AZ training — works, modest improvement

Used [train_az_head.py](tools/train_az_head.py) with `--policy-tau 1.5 --policy-eps 0.5` on the existing 40,823-demo post-fix dataset (`runs/closed_loop_iter1_postfix/demos_iter1_big.npz`).

**Key effect of smoothing**: visit_dist max-channel mean dropped 0.948 → 0.223 — targets are no longer one-hot, so the policy head has actual signal to learn from.

**Training results (15 epochs heads-only, GPU, 70s wall):**

| epoch | val CE | val MSE |
|---|---|---|
| 1 | 2.10 | 0.65 |
| 6 | 2.07 | 0.48 |
| 15 | 2.07 | 0.43 |

Compared to v5 value head's val_mse=0.39 noted in STATUS.md, this is comparable. The CE plateau at 2.07 means the small (64k-param) policy head is saturating — it can't extract more from the smoothed targets without backbone unfreezing.

**Backbone unfreeze attempt failed**: enabling `--unfreeze-backbone-after-epoch 5` caused value MSE to jump 0.48 → 1.18 immediately and plateau. The backbone learning policy-relevant features destabilizes the value head at this small scale. **A larger backbone (200k+ params) is the unblocker** — the AZ paper itself uses ~10M params for Go.

### 4. v37 bundled, smoke vs random PASS

[submissions/v37.py](submissions/v37.py) (688 KB):
- TuRBO-v3 weights
- `sim_move_variant=exp3`, eta=0.3
- `rollout_policy=nn_value` with the AZ checkpoint as leaf eval
- `anchor_margin=0.5`
- 128 sims, 850ms deadline, 4 candidates

Smoke vs `random`: rewards=[1,-1], 93 steps. NN value head + MCTS + heuristic load and play correctly.

H2H vs v32b (8-game mirror) — see results below.

## What's next for top-10

The current path (smaller NN + tau-smoothing) is unlikely to clear v32b on its own. STATUS.md established that v3/v4/v5 value heads as leaf eval all lost -190 to -800 Elo against heuristic-15-ply rollouts. Tau-smoothing plausibly recovers some of that gap (better policy distillation) but the **structural ceiling at 64k params hasn't moved**.

**Three next moves, ranked by EV for top-10 push:**

1. **Mixed leaf eval** (untested, 2 days): `V_leaf = α·V_NN + (1-α)·V_heuristic_rollout`. Add `value_mix_alpha` to GumbelConfig, modify the leaf-eval dispatch. Use the AZ checkpoint as the NN side. Variance reduction from combining a long-horizon NN prior with short-horizon heuristic rollouts. The single biggest "explicitly listed as untested" lever from STATUS.md.

2. **Bigger NN backbone via Kaggle Dataset path** (3-5 days): the existing `bc_warmstart_v2_dataset` infrastructure already loads big (>1MB) checkpoints from Kaggle Datasets. Train at 200-500k params (vs 64k current). The backbone-unfreeze instability above goes away at this scale — the value and policy heads have enough capacity to coexist. This is the real precondition for **NN-driven rollouts** (the AlphaZero structural fix).

3. **NN-driven rollouts** (3-5 days, requires move 2): replace `HeuristicAgent` rollouts with NN-greedy rollouts in MCTS. Q estimates become "how the NN plays from here" instead of "how the heuristic plays from here" — the actual structural unlock for NN-on-wire.

**Combined estimated lift**: from 939 Elo → 1100-1300 Elo if all three land. Top-10 needs ~1500+ Elo per current leaderboard.

## Files added this session

- [src/orbitwars/engine/intercept_vec.py](src/orbitwars/engine/intercept_vec.py) — batch Newton solver (works, currently unused; kept for the second-pass batching follow-up).
- [tests/test_intercept_vec.py](tests/test_intercept_vec.py) — 5 bit-equivalence tests, all pass at 1e-9 abs.
- [tools/profile_heuristic_cprofile.py](tools/profile_heuristic_cprofile.py) — line-level cProfile of heuristic.act() across rollout plies.
- [tools/smoke_heuristic_byte_identical.py](tools/smoke_heuristic_byte_identical.py) — multi-seed regression check with seed-42 baseline encoded.
- runs/az_v37.pt — tau-smoothed AZ checkpoint (val_mse=0.43, val_ce=2.07).
- submissions/v37.py — v37 bundle.

## Files modified

- [src/orbitwars/engine/intercept.py](src/orbitwars/engine/intercept.py) — `fleet_speed` memo.
- [src/orbitwars/bots/heuristic.py](src/orbitwars/bots/heuristic.py) — `ParsedObs` orbit-metadata cache + call-site reads.
