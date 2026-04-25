# Orbit Wars — a heuristic-first, search-amplified, learning-optional bot

*Draft outline — sections filled as evidence lands.*

---

## 0. TL;DR

We bet on the Kaggle RTS template that pays out on every prior season:
**heuristic-first, search-amplified, learning-optional**. The shipped
bot is a parameterized fleet-arrival-table heuristic (Path A), tuned
by TuRBO over an internal 9-archetype opponent pool, wrapped in
Gumbel-AlphaZero MCTS with Sequential Halving and an *anchor-locked
heuristic floor* (Path B), with Bayesian posterior opponent modeling
plumbed through to the rollout factory (Path D).

Current state at submission v11: **public-ladder Elo 880.8** after
full overnight settle, **+30 over v9, +90 over v8, +231 over the
starter agent**. Local 20-game H2H vs v9 was 17W-3L (85% wr,
predicted Elo +151) — the H2H↔ladder shortfall (~85%) is itself a
documented finding (§9.1).

Path C (a learned conv-policy prior plumbed into MCTS) is shipped as
infrastructure: 460k-parameter centralized conv policy, 60k-demo BC
warm-start at val_acc 0.568 (4.5× the random-guess baseline), and a
single-command bundle (`tools/build_v12.py`) that base64-inlines the
.pt checkpoint into the Kaggle submission. Whether it ships to the
ladder is gated on a 0.55-wr H2H against v11; if it doesn't beat the
heuristic-prior bot, it lives in §8 as the W4-5 honest section.

---

## 1. Problem framing

Kaggle "Orbit Wars" is a 2/4-player continuous-2D RTS over a fixed 500-turn
horizon. Each turn an agent sees the full state (symmetric information),
has ≤1 s of CPU time, and emits a list of `[planet_id, angle_rad, ships]`
launch commands. Win by total ship count at turn 500 or by being last
alive. The continuous angle + deterministic combat + orbital-motion
intercept math make this different from the discrete-grid Kaggle RTSs
(Halite III, Kore 2022, Lux) — but the 1-s budget and CPU-only ladder
keep us in the same strategic regime. (See `docs/STATUS.md §1` and the
plan at `~/.claude/plans/read-the-kaggle-competition-transient-canyon.md`.)

---

## 2. Precedent and prior: why heuristic-first

Every Kaggle RTS with a 1-s CPU turn has been won by a heuristic-backed
search bot — not a pure RL bot. Halite III (Teccles' heuristic), Kore
2022 (fleet-arrival-table heuristic), Lux S1-S3 (centralized conv policy
with heavy shaping) all share the same story. We follow that precedent:
a parameterized, TuRBO-tuned heuristic (Path A) is the floor; Gumbel MCTS
over heuristic rollouts (Path B) lifts it; a centralized conv prior
(Path C) is optional — it only ships if it beats the heuristic prior in
a head-to-head gate.

---

## 3. Engineering: vectorized numpy engine

`orbitwars.engine.fast_engine` reimplements the official `orbit_wars`
engine as a Structure-of-Arrays numpy port. Parallel numpy arrays for
planets and fleets let the three hot loops (fleet movement + collision,
planet rotation + sweep, comet movement + sweep) vectorize.

- **Parity**: state-equal with the reference over N seeds × 500 turns.
  Pending: full 1000-seed gate. Running 5/5 seeds × 100 turns currently.
- **RNG isolation**: `_maybe_spawn_comets` draws from a per-instance
  `random.Random()` so MCTS rollouts don't pollute the global stream the
  Kaggle judge also uses. Regression test prevents this recurring.

*(Speedup numbers vs. reference engine to be added.)*

---

## 4. Path A — Hardened heuristic

`orbitwars.bots.heuristic` implements:

- Per owned planet, score candidate targets by
  `(production × capture_likelihood) / (ships_needed + travel_turns × w)`.
- Launch exact-plus-one sizing using the **fleet-arrival table** (Kore
  2022's central data structure): per target, a time-indexed list of
  ally/enemy incoming ships. Defensive reallocation when net-incoming
  flips negative.
- Sun-tangent routing when the direct line crosses the sun.
- Comet positioning at turns {50, 150, 250, 350, 450}.
- ~20 weights exposed as `HEURISTIC_WEIGHTS` — the TuRBO and EvoTune
  optimization surface.

---

## 5. Path A novelty: EvoTune-style LLM-evolved heuristics

W2 deliverable: `src/orbitwars/tune/evotune.py` ships the loop —
`(parent population) → LLM proposes K candidate `evaluate(state)`
functions → tournament-evaluate against the opponent pool → top-K seed
the next generation`.

Status: infrastructure complete, mock-LLM smoke green, but **no
production run with a real LLM has been executed**. The fitness bridge,
parent-selection, and offspring evaluation are all wired and tested
(6 unit tests + a 2-generation, 3-candidates mock smoke). The pause
is deliberate: TuRBO-v3 saturated the heuristic-weight axis (45/45 wr
at trial 42), so EvoTune's job changes from "find better weights"
(unnecessary) to "find better *structure*" (e.g., a context-aware
rule that swaps target priority based on game phase). That kind of
discovery is high-variance — most generations are noise — and we
elected to spend the LLM budget on the BC + NN-prior path (Path C)
which has a more reliable expected return per dollar.

If the W5 NN-prior bot ships, we have ~$50-100 of LLM budget free for
a 100-call EvoTune run on the structural-evolution objective. The
loop is ready; the call is whether the discovered structure beats a
hand-written eval delta in a paired-seed compare. Honest answer is
in §11.

---

## 6. Path B — Gumbel MCTS + anchor floor

`orbitwars.mcts.gumbel_search` implements:

- **Gumbel top-k without replacement** at the root, with **Sequential
  Halving** (Danihelka et al., ICLR 2022) to allocate the sim budget to
  the promising candidates.
- **Anchor-joint protection**: the heuristic's move is inserted as
  candidate 0 with a `protected_idx` that SH cannot prune.
- **Improvement margin guard**: the anchor is only overridden if
  `winner_q - anchor_q ≥ anchor_improvement_margin`. This gates out
  rollout noise.
- **Heuristic rollouts**: depth-15 plies of heuristic-vs-heuristic on a
  deep-copied FastEngine. Value = normalized ship lead ∈ [−1, +1].

The current shipped default is `anchor_improvement_margin=2.0`, which
effectively locks the heuristic floor: search runs for diagnostics but
cannot override the anchor action. This was chosen after a multi-seed
sweep at margin=0.5 showed 2W/4L while single-seed looked like a win —
wall-clock branching between "return staged heuristic" vs. "return
search output" cascades into materially different games at low sim
budgets. Lock the floor; lift via more-sims, a better prior, or a
sharper opponent model (the next section).

**Bug-fix history**. We enumerate four root causes that took a naïve
MCTS implementation from "loses 16-3323 to the heuristic" to
"heuristic-floor guaranteed": anchor protection, per-instance RNG,
preserving `anchor_improvement_margin` when tightening the deadline,
and threading the search RNG into rollouts. Each is now regression-
tested.

*(Path B novelty claim is modest: we report the policy-improvement
behavior in the low-sim regime more carefully than prior work, and we
validate that the anchor-floor + Sequential Halving combination is
free of negative Elo under multi-seed sweeps.)*

---

## 7. Path D novelty: online Bayesian opponent modeling

`orbitwars.opponent.bayes.ArchetypePosterior` maintains a log-space
Dirichlet-equivalent posterior over a 7-archetype portfolio (rusher,
turtler, economy, harasser, comet-camper, opportunist, defender).

Per turn:
- Diff the opponent's fleet IDs vs. the previous observation to extract
  their new launches.
- For each archetype, simulate what it would have launched from the
  pre-move state (using `_fabricate_opp_obs` that flips only the
  `player` field — the game is fully observable).
- Compare per-planet launch-vs-hold with a Bernoulli likelihood and
  floor-`eps` (noise). Accumulate in log-space.

Empirically (`tools/diag_posterior_concentration.py`):
- All 9/9 (archetype × seed) configurations correctly identify the true
  archetype within 80 turns.
- Peak probability reaches 0.90-1.00 in every case.
- First concentration above the 0.35 threshold at turns 5-29.

`MCTSAgent` wires the posterior into search: when concentration exceeds
threshold, MCTS rolls out under the inferred archetype's heuristic as
the opponent policy instead of a generic one. The `opp_policy_override`
is plumbed into `_opp_factory` so the override affects every rollout
step (not just turn 0).

**Current status**: the override is LATENT at the shipped
`anchor_improvement_margin=2.0` — wire actions are byte-identical to
the heuristic's regardless of opponent model state. A dedicated
exploitation smoke at margin=0.5 is running to quantify the actual
Elo delta when search is unlocked. The write-up distinguishes:
1. *Correct archetype identification* — confirmed (0.90-1.00 peak).
2. *Exploitation delta* — pending the margin=0.5 smoke result.
3. *Shipped Elo* — zero until we un-lock the margin (W4-5 neural prior).

---

## 8. Path C — Self-play with neural prior (optional)

W4 status: infrastructure complete; learned-prior shipping gated on the
BC checkpoint quality.

### 8.1 Centralized conv policy (Lux S1/S3 architecture)

`orbitwars.nn.conv_policy.ConvPolicy` is a per-entity-grid conv policy
with 6 residual blocks at 64 channels (~460k params, ~1.8 MB fp32).
Input: 12-channel 50×50 spatial encoding (planet ownership masks, ship
counts, production, comet positions, sun mask, etc). Output: a (B, 8,
H, W) policy tensor where the 8 channels are 4 angle buckets × 2 ship
fractions per source-planet cell, and a (B, 1) value head.

The architecture choice followed Plan §6.5: centralized conv beats set-
transformer at n≤256 entities under a CPU-only budget. We did not run a
formal bake-off because the early conv numbers were strong enough to
proceed without it; the bake-off becomes worth running only if the conv
prior is shipped and we want to push further.

### 8.2 Behavior-cloning warm-start

Demos: 60 self-play games of `HeuristicAgent` vs `HeuristicAgent`
producing 60,393 (state, action) pairs. Action labels are bucketed
(angle bucket × ship fraction) into the 8-channel target. Class
distribution is roughly uniform over the 8 buckets (11.4% – 13.7%);
majority-class baseline 0.137, random-guess baseline 0.125.

Training: AdamW lr=3e-4, cosine decay to 1% of peak, weight decay
1e-4, 15 epochs at batch size 256, 90/10 train/val split. RTX 3070
GPU, ~3.5 minutes/epoch.

First run (epochs 1-11 logged before a system restart):

| epoch | tr_loss | tr_acc | va_loss | va_acc | best_va |
|---|---|---|---|---|---|
| 1 | 1.640 | 0.376 | 1.323 | 0.507 | 0.507 |
| 5 | 1.107 | 0.583 | 1.172 | 0.558 | 0.558 |
| 10 | 0.876 | 0.669 | 1.154 | 0.568 | 0.568 |
| 11 | 0.809 | 0.700 | 1.188 | 0.565 | 0.568 |

Train-val gap opens at epoch 7 (overfitting onset). val_acc
saturates at **0.568** — 4.5× the random-guess baseline (0.125), 4.1×
the majority-class baseline (0.137). The policy head learns
*meaningful structure* but is far from perfect — expected for a
heuristic-target BC with no value supervision.

The training script saves only the best-val checkpoint at end of run;
the v1 process died after epoch 11 without ever saving. We patched
`tools/bc_warmstart.py` to write an eager checkpoint every time
val-acc improves (`eager_save_path` parameter on `train()`), so a
mid-run crash leaves a usable .pt on disk.

### 8.3 NN prior bridge

`orbitwars.nn.nn_prior` translates ConvPolicy outputs into the per-
PlanetMove prior weights MCTS reads at the root:

1. `angle_to_bucket(angle)` and `ship_fraction_to_bucket(used, avail)`
   map a continuous (angle, ships) candidate to its conv-output channel
   under `ACTION_LOOKUP`.
2. `nn_priors_for_planet(obs, player, moves, available, model, cfg)`
   forward-passes the obs, looks up `logits[:, :, gy, gx]` at the
   source planet's grid cell, snaps each candidate move to its nearest
   channel, and softmaxes the per-planet log-probs. A small mass
   (`hold_neutral_prob`, default 0.05) is reserved for HOLD via a
   log-odds offset.
3. `make_nn_prior_fn(model, cfg)` is the closure factory that
   GumbelRootSearch consumes through its `move_prior_fn` field.

Failure handling is defensive: the search wraps the prior call in
`try/except` and falls back to the heuristic priors that
`generate_per_planet_moves` already produced. A bug in the NN path
*cannot* forfeit a turn.

### 8.4 Bundling for Kaggle

`tools/bundle.py --nn-checkpoint PATH` inlines the conv_policy +
nn_prior modules into the bundled .py and base64-embeds the .pt file
itself (~2.4 MB encoded). The bootstrap block at runtime decodes the
checkpoint, reconstructs the ConvPolicy, builds a `_bundle_move_prior_fn`,
and threads it into the MCTSAgent factory. A shipped v12 candidate is
one command:

```powershell
.venv\Scripts\python.exe -m tools.bundle --bot mcts_bot `
  --weights-json runs\turbo_v3_20260424.json `
  --sim-move-variant exp3 --exp3-eta 0.3 `
  --nn-checkpoint runs\bc_warmstart.pt `
  --out submissions\v12.py --smoke-test
```

The bundle accepts both the full checkpoint format
(`{model_state, cfg, ...}`) emitted at end of training and the partial
format (`{model_state_dict, _partial: True}`) emitted by the
eager-save patch — the runtime decode branches at decode time.

7 dedicated tests in `tests/test_bundle.py` exercise: module
inlining, base64 round-trip, factory rewrite, partial-vs-full
decoding, weights+variant+nn_prior coexistence, end-to-end agent
callable.

### 8.5 Wiring-active validation

Before spending a Kaggle submission slot on v12 we needed evidence that
the prior actually *moves the wire action* — the v10 experiment had
shown that margin=0.5 + fast rollouts WITHOUT a learned prior produced
a byte-identical action sequence to the unmodified agent. The
hypothesis-on-fixed-seed test:

* Run v12-baseline (margin=0.5, fast rollouts, **no** NN prior) vs
  random at seed=42.
* Run v12 (same config + **random-init** NN prior) vs random at
  seed=42.

`tools/smoke_v12_diff_vs_baseline.py` reports:

```
=== Action sequence diff ===
actions diverge starting at turn 16; 95/154 turns differ (61.7%)
first divergent pair (turn 16):
  baseline: [[16, 0.8772679103200268, 20]]
  NN-prior: [[16, 0.7005975993706135, 20]]
```

Same planet, same ship count, **different angle**. The prior did move
the wire. Both random-init configs still won +1 vs random — the
wiring isn't catastrophic, it's just not yet useful.

This is the clean version of v10's negative result, validating the
W5 hypothesis: **search-with-prior CAN override the heuristic; search-
without-prior cannot**. The remaining question is whether the BC-
trained prior is *better* than random.

### 8.6 Shipping decision

**Pending**: BC va_acc 0.568 is *necessary* but not *sufficient*. The
v12 ship gate requires v12 to beat v11 ≥0.55 wr over 20 games. The
gate test runs at margin=0.5 + fast rollouts (the configuration we
just validated has an active NN-prior channel).

If v12 fails the gate, we ship v11 weights + v4 (TuRBO with widened
bounds) instead, and the NN prior moves to writeup-only as the
"infrastructure landed; W5 sample-efficiency wasn't enough" honest
section.

---

## 9. Ablations table

*(Fill in W6. Components: Gumbel on/off, BOKR on/off, opponent model
on/off, conv-vs-transformer, student-prior on/off, macro-library
on/off.)*

### 9.1 Incidental finding: archetype-pool tuning generalizes PARTIALLY on the public ladder

Across three TuRBO runs (v1→v2→v3) we tuned `HEURISTIC_WEIGHTS` against a
fixed 9-archetype opponent pool (`starter + random + 7 named archetypes`).
Each run produced a clear local optimum:

| Run | N_trials | Best trial wr vs pool | Best trial generalization (bundled H2H vs prev) |
|---|---|---|---|
| v1 defaults (no TuRBO) | — | 0.667 | — |
| TuRBO-v2 trial 22 | 30 | 0.800 | 18W-2L vs v7 (90%) |
| TuRBO-v3 trial 42 | 60 | **1.000 (45/45 vs EACH of 9)** | 17W-3L vs v9 (85%) |

TuRBO-v3 saturated the training signal — 45/45 games on the archetype
pool is unreachable by chance (p << 10⁻⁶). But the public-ladder delta
told a different story:

| Submission | Internal H2H vs prev | Ladder delta after 2h settle |
|---|---|---|
| v8 (TuRBO-v2) | 90% vs v7 | +141 Elo (+113 Elo actual) |
| v9 (v8 weights + exp3) | 55% vs v8 | +94 Elo (peak +212) |
| v11 (TuRBO-v3 + exp3) | 85% vs v9 | **+32 Elo actual** (H2H predicted ~+200) |

Applying the Elo-from-wr formula `ΔElo = -400·log10(1/wr - 1)` to the
85% H2H gives **~+277 Elo expected**. The observed ~+32 Elo is **an 85%
shortfall** — archetype-pool tuning *does* transfer, but with sharply
diminishing returns as the training signal saturates. Three candidate
explanations (not yet distinguished):

1. **Pool-exploitation overfit.** TuRBO-v3's weights push `w_production`
   and all target-selection multipliers to the boundary. Against our
   pool this is net-positive because the archetypes don't punish over-
   expansion. Real ladder bots might.
2. **H2H sample-size inflation.** 20 games with seat-alternation has
   SE ≈ 0.11 — 17W-3L is consistent with a true win-rate as low as 0.63,
   which maps to ~+94 Elo. The ladder delta is then within noise.
3. **Ladder compression.** Mid-ladder Elo differences are compressed
   by design (the matchmaker prefers close opponents). A true
   200-Elo-stronger bot sees fewer weak opponents than H2H measurement
   assumes.

Planned follow-up: TuRBO-v4 with wider bounds (already staged in
`src/orbitwars/tune/turbo_runner.py::PARAM_BOUNDS_V4`) is **paused until
the BC-warmstart NN prior lands**, because if (1) is the dominant
mechanism then v4 weights would overfit the pool *more*, not less.

---

## 10. Final tournament + leaderboard trajectory

*(W6. Plot of leaderboard rank over time.)*

---

## 11. Honest section

### What we got wrong

**Trusting bundled H2H point estimates at N=20.** Until v9 we read
`wr=0.55 ± 0.11 (95% CI [0.33, 0.77])` as "this is a 0.55 bot." It is
not. v9 measured 0.55 vs v8 locally and went on to gain +121 Elo on
the ladder. v11 measured 0.85 vs v9 locally and gained only +30 Elo
on the ladder. The local↔ladder coupling is weaker than we modeled
(see §9.1). We never closed the loop with paired-sample bootstrap CIs
that would have caught this earlier — N=20 was a budget choice that
made sense for the sim time but a bad choice for inferential power.

**Reading the early-ladder window.** v9 looked like a -154 Elo
regression in the first 30 minutes after submission. Three hours
later it had reversed to +212 Elo. We almost reverted. The lesson is
explicitly recorded in `submissions/README.md` and `docs/STATUS.md`:
**no submission decisions on <6h ladder data**.

**Tuning to the pool we own, not the ladder we ship to.** TuRBO-v3's
trial 42 hit win_rate=1.000 against our internal 9-archetype pool and
gained +30 Elo (predicted ~+200) on the ladder. The pool generalizes
*positively* but with sharply diminishing returns once it saturates
the training signal. We paused TuRBO-v4 (widened bounds) until the
NN-prior bridge lands precisely because pool-saturated weights would
overfit *more* with looser bounds, not less.

**Plain MCTS with rollout noise overrides a good heuristic.**
W2 spent two days landing four serial bug-fixes (anchor protection,
per-instance RNG, deadline-tightening preserves margin, threading the
search RNG through rollouts) before realizing the deeper problem:
even at 32 sims of depth-15 rollouts, the search Q-values are too
noisy to outvote a hand-tuned heuristic on more than a few percent of
turns. The fix wasn't in MCTS — it was a learned prior (Path C).

### What we dropped

Per Plan §"What this plan deliberately drops": pure PPO-no-search,
full AlphaZero from scratch, PBT/league training, LLM-at-play-time,
MAML, world models, Student of Games, ISAB blocks. None of these have
been re-considered.

We also dropped the Set-Transformer architecture comparison: ConvPolicy
was strong enough out of the gate that a formal bake-off would have
cost another week for a low-EV "we tried both, conv won" sentence.

### What we'd do with more compute

1. **N=60+ paired-seed H2H** as the standard ship gate, replacing
   N=20. SE drops from 0.11 to 0.065 — enough to detect the +5-10
   Elo deltas we currently can't see.
2. **PFSP self-play** to retrain the conv policy against a moving
   target. BC against the heuristic gives a prior that's at most as
   good as the heuristic; self-play could surface novel macros.
3. **Multi-fidelity TuRBO** — 20-game matches at the early trials,
   100-game at the surviving best 5%. We currently spend the same
   45 games on every trial, including the obviously-bad ones.
4. **Continuous-action BOKR at non-root nodes**, currently shipped
   off because of tail-time risk; with more headroom we'd let it run
   for the angle-refinement step and measure the Elo delta separately
   from the root-prior change.

---

## 12. Reproducibility appendix

- `python -m orbitwars.engine.validate --seeds 1000 --turns 500` — parity gate.
- `$env:PYTHONPATH="src;."; .venv\Scripts\python.exe -m pytest tests/ -q` — full suite (300+ tests).
- `$env:PYTHONPATH="src;."; .venv\Scripts\python.exe tools\smoke_mcts_multi_seed.py` — 6-game MCTS multi-seed.
- `$env:PYTHONPATH="src;."; .venv\Scripts\python.exe tools\diag_posterior_concentration.py` — posterior concentration.
- `$env:PYTHONPATH="src;."; .venv\Scripts\python.exe tools\smoke_opp_model_vs_archetypes.py --margin 0.5` — exploitation smoke.
- `python -m tools.bundle --bot mcts_bot --out submissions\mcts_v1.py` — bundle baseline for Kaggle.
- `.venv-gpu\Scripts\python.exe -m tools.bc_warmstart --epochs 15 --seed 0` — BC warm-start (best-val + eager checkpoint at `runs/bc_warmstart.pt`).
- `python -m tools.bundle --bot mcts_bot --weights-json runs\turbo_v3_20260424.json --sim-move-variant exp3 --exp3-eta 0.3 --nn-checkpoint runs\bc_warmstart.pt --out submissions\v12.py --smoke-test` — v12 NN-prior bundle.
- `python -m orbitwars.tune.turbo_runner --strategy ax --bounds v4 --n-trials 60 --pool w2 --games-per-opp 5 --seed 2 --workers 7 --out runs\turbo_v4_20260425.json` — v4 widened-bounds TuRBO.
