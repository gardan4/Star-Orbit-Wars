# NN Value-Head Q in MCTS — Design Doc

## Why this exists

Today's investigation (2026-04-26, see `STATUS.md` "PHANTOM" sections)
established two facts:

1. **The NN doesn't reach the wire** under our current MCTS config.
   `move_prior_fn` only modifies PUCT exploration weights; Q values
   come from `_rollout_value` (heuristic-driven 15-ply rollouts).
   Because rollouts use the heuristic, all Q estimates are ~"how good
   would the heuristic play this from here," and the heuristic anchor
   wins on Q nearly every time. NN logits differ by mean abs 1.83
   between BC v1 small and PPO 500-update, but wire actions are
   byte-identical across 20-game H2H.

2. **The NN value head is currently untrained.** `bc_warmstart.py`
   loads `(logits, _value) = model(x)` and discards `_value`. The
   value head outputs noise (~uniform [-0.07, 0]) regardless of input.

These two combine to make any "PPO/BC/AZ" attempt structurally
ineffective on the ladder — confirmed by v25 (PPO ladder = -141 Elo
vs v22). To realize AlphaZero-style ceiling we need the **value head
to drive search, not the heuristic**.

## What this design proposes

Add a `value_fn` pathway alongside the existing `_rollout_value`
pathway. Configurable via `GumbelConfig.rollout_policy`:

| Value of rollout_policy | Leaf evaluation |
|---|---|
| `"heuristic"` (default, current) | `_rollout_value` w/ HeuristicAgent on both sides, depth=15 |
| `"fast"` (current) | `_rollout_value` w/ FastRolloutAgent on both sides, depth=15 |
| **`"nn_value"` (NEW)** | Apply 1 ply (joint action) → encode resulting state → NN value head → scalar |

`"nn_value"` requires `value_fn` to be plumbed through. If unset and
the policy is selected, fall back to `"heuristic"` with a warning.

## Implementation plan

### Phase 1 — pathway plumbing (1-2 days, no training required)

1. **`src/orbitwars/nn/nn_value.py` (new)**: `make_nn_value_fn(model, cfg)`
   returns `(state, my_player) -> float`. Encodes obs via existing
   `obs_encode.encode_grid`, runs `model(x)`, returns
   `value.item()`. Scalar in [-1, 1] convention (positive = good for
   `my_player`).

2. **`gumbel_search.py`**:
   - Extend `GumbelConfig.rollout_policy` literal to include `"nn_value"`.
   - Add `GumbelRootSearch.value_fn: Optional[Callable]`.
   - In the leaf-eval dispatch (where `rollout_fn` is constructed),
     when `rollout_policy == "nn_value"` and `value_fn is not None`:
     substitute `_value_fn_eval(joint)` that:
       a. Clones `base_state`.
       b. Applies `joint.to_wire()` (my action) and a single
          `opp_factory()` ply (heuristic opp turn-0).
       c. Steps engine 1 ply.
       d. Calls `value_fn(state, my_player)`.
       e. Returns scalar.
     This reuses the same FastEngine plumbing as `_rollout_value`'s
     first ply but stops there; subsequent depth comes from value-head
     extrapolation, not actual rollouts.

3. **`MCTSAgent` (`bots/mcts_bot.py`)**: pass `value_fn` through to
   `GumbelRootSearch`.

4. **`bundle.py`**: extend `--rollout-policy` choices to include
   `nn_value`, plumb `value_fn` factory into the bootstrap (parallel
   to the existing `move_prior_fn` factory).

5. **Smoke test**: bundle `v_value_dummy` with a constant
   `value_fn = lambda s, p: 0.0`. Should run, MCTS Q values should
   collapse to similar values, anchor wins on tie-breaking. Wire
   actions should be byte-identical to a no-NN `--rollout-policy=fast`
   run *because the value function is constant*. This proves the
   pathway works end-to-end.

6. **Sanity test**: bundle `v_value_random` with random value_fn.
   Wire actions MUST differ from v_value_dummy. Confirms value_fn
   actually steers Q estimates.

### Phase 2 — value head training (3-5 days)

The bottleneck. Two paths:

**Path A — distillation from MCTS rollouts (faster, lower ceiling)**:
Take a strong existing bot (v22's MCTS), run it in self-play and
record `(state, true_outcome)` pairs. Train value head on these via
MSE: `loss = (V_pred - V_true)^2`. Uses the rollout value as a
*free* training signal — no separate self-play infra needed.
- 100k-500k state-outcome pairs
- 10-30 epochs
- 1-3h on RTX 3070
- Result: value head approximates "what would v22 score from here."

**Path B — joint policy + value training via PPO/AZ (slower,
higher ceiling)**:
Standard AlphaZero: self-play with the network in the loop, train
both heads jointly on (visit_distribution, game_outcome) targets.
Bottlenecked by self-play throughput; needs working `value_fn`
pathway from Phase 1 to be useful.
- 20-100M env steps (PPO infra is already debugged)
- 1-3 days on RTX 3070
- Result: value head learns its own preferences, not bounded by
  v22's heuristic strength.

**Recommendation: Path A first** as a proof-of-life. If Path A's
distilled value head, plugged into the Phase 1 pathway, produces
wire actions that beat v22 on the ladder, that validates the entire
direction and we move to Path B.

If Path A's value head is wire-equivalent to v22 (MCTS-derived
targets just relearned the heuristic anyway), the structural fix
isn't enough — would need different action encoding or more
fundamental redesign.

### Phase 3 — ship and iterate

1. Bundle `v28_value_head` = v22 config + `--rollout-policy=nn_value`
   + the trained value head.
2. Paired compare against v22 on w3 pool.
3. Ship if real signal.

## Risks and unknowns

1. **Value head accuracy at init is critical.** A noisy value head
   under nn_value regime can produce *worse* wire actions than
   heuristic rollouts because the search has nothing reliable to
   maximize. Cold-start handling: temperature/entropy on the value
   head, or fall back to heuristic rollout when value head is
   uncertain (e.g. abs(value) < 0.1).

2. **MCTS sim count vs value-head latency.** A NN forward pass is
   ~3-5ms on CPU. Currently 128 sims × 1ms heuristic rollout =
   128ms. With nn_value at 4ms/sim, 128 sims = 512ms — over the
   850ms hard deadline once you add MCTS overhead. Need to either
   (a) drop sims to 64, (b) batch NN forward passes across sims
   within a round, or (c) write a faster CPU forward pass.

3. **Distribution shift in value targets.** If we train Path A on
   v22-vs-v22 games but ladder opponents are very different,
   value head generalizes poorly. Mitigation: include w3-pool games
   in the training set (varied opponents).

4. **The wall-clock determinism issue persists** for any MCTS with
   `time.perf_counter()` budget allocation. Local H2H stays
   unreliable. Ladder is still the only honest signal.

## Why this is the right next bet

The ceiling argument (AlphaZero, MuZero) holds: with a working
NN-driven search, the policy improvement loop has no hard ceiling
short of compute. Heuristic-engineering plateaus around the human-
domain-knowledge limit; learning systems compound past that.

Concretely: TuRBO has gotten us +113 (v8) → +30 (v11) → +0/marginal
(v22 → v27 PENDING) Elo on the ladder. Diminishing returns. v22's
strength is bounded by what TuRBO can extract from a parameterized
heuristic. To break tier 2 (1300+) we need a system that can
learn things the heuristic doesn't encode.

## Open question for next session

Should we wait for v27 ladder result before starting Phase 1?

Arguments to start Phase 1 immediately:
- v27 settles overnight; Phase 1 is mostly mechanical, can run in
  parallel
- Phase 1 doesn't depend on which weights win — the pathway is
  structural
- Earlier we have it, sooner we can iterate

Arguments to wait:
- If v27 lands well above v22 (≥+50 Elo), maybe TuRBO + w3 pool can
  keep extracting value and Phase 1 can be deferred
- Phase 1 + 2 is ~1 week of focused work; opportunity cost matters

**Default: start Phase 1 now (low cost, high value).** Phase 2
gating depends on v27 outcome.

---

## RESULT: Phase 1 + Phase 2 (Path A) implemented (2026-04-26 evening)

### What shipped

* Phase 1 plumbing: `src/orbitwars/nn/nn_value.py` (`make_nn_value_fn`,
  diagnostic helpers), `_value_fn_eval` in `gumbel_search.py`,
  `GumbelRootSearch.value_fn` field, `MCTSAgent` value_fn passthrough,
  `bundle.py --rollout-policy=nn_value` flag. 125 tests pass, 0 regressions.
* Phase 2 step 1: `tools/collect_mcts_demos.py` extended to record
  per-state `terminal_value` (+1/-1/0 from decision-maker's perspective).
* Phase 2 step 2: `tools/train_value_head.py` shipped — frozen-backbone
  value-head training via MSE.
* Path A demo collection: 50,027 demos at sims=64, deadline=400ms
  (`runs/mcts_demos_v6_with_outcomes.npz`, 64.6 MB).
* Value head trained: 20 epochs, val_mse 0.250 → **0.045** (94% below
  mean-prediction baseline 0.713). Genuine learning. (`runs/bc_v_v1.pt`)
* `submissions/v28.py` bundled with `--rollout-policy=nn_value` and
  `bc_v_v1.pt`. Loads cleanly, wins vs random in 4s / 63 steps.

### The negative result

**Wire actions are byte-identical to v22 across all 5 test seeds**, both
at `anchor_margin=0.0` (same as v22) and at `anchor_margin=1.0`
(stricter override threshold). Same step counts, same rewards, same
trajectory.

Sanity check: the random-init value head (`v_value_smoke`) and the
trained value head (`v28`) produce the same wire actions. Training
quality is *irrelevant to the wire* in this configuration.

### Why this happened (the closed-loop problem)

Path A (rollout distillation) trains the value head to predict outcomes
*from heuristic-style play*, because that's what the demo collector's
self-play produced. So the value head's Q estimates rank candidates
*the same way* heuristic-rollout Q estimates do — by definition, since
it learned to predict heuristic-rollout outcomes.

Even when the value head's argmax differs slightly from the rollout's
argmax (it must, given the residual val_mse > 0), the:
1. heuristic anchor candidate is protected from SH pruning, gets the
   most visits, gets the most reliable Q;
2. heuristic decoder collapses MCTS's chosen channel back to the same
   wire action a heuristic player would emit.

This is the AlphaZero policy-improvement assumption: each iteration's
value head is trained on data from a *better* policy than the previous
iteration's, so the value head's preferences pull MCTS toward stronger
play. Single-iteration distillation cannot do that — it's a fixed point.

### What this means for the design

Path A was intended as proof-of-life for the pathway, not a
competitive ship. The pathway IS proven (the value_fn is being called
during search, val_mse drops as expected), but realizing the AlphaZero
ceiling requires the closed loop:

1. Run self-play with the new value-head bot.
2. Collect (state, outcome) demos from that self-play.
3. Re-train value head on the new demos (which now reflect *the value-
   head bot's* outcomes, not heuristic-rollout outcomes).
4. Iterate. After K rounds, the value head reflects the best policy in
   the iterative chain.

This is Path B in spirit (PPO/AZ joint training) but via a simpler
distillation loop. ~3-7 iteration days, ~5-10x compute of single-shot.

### Alternatives if AZ-loop is too expensive

* **Different action space**: the heuristic decoder's collapse is
  upstream of all of this. Replace `ACTION_LOOKUP` with finer angle
  resolution (16-32 channels) so non-anchor candidates produce
  meaningfully different wire actions.
* **Hand-crafted bootstrap data**: collect demos from a deeper-search
  bot (rollout depth 50, sim count 1000+). Value head learns from
  stronger-than-current play.
* **Ship without value head, focus elsewhere**: the structural NN work
  is honest research progress regardless of ladder outcome. v28 can
  ship as an identity test (will land ~v22 by hypothesis).
