# Orbit Wars — Repo Status & Architecture

## NIGHT-OF-2026-04-28 — PHANTOM 5.0 + 6.0 FOUND, REAL MCTS FINALLY SHIPS (v32b 939.1 Elo)

**TL;DR**: Phantom 4.0 was the FIRST of three compounding bugs that
silently disabled MCTS. After fixing all three, v32b is the first
bundle in this repo's history to actually run MCTS rollouts. It
shipped to the ladder and landed at **939.1 Elo** (+53 over v27's
886.1 yesterday, the previous heuristic-only leader).

### PHANTOM 5.0 — `fresh_game` rebuild drops `move_prior_fn` and `value_fn`

**Root cause**: `mcts_bot.py` lines 341-350 detect a fresh game
(turn 0) and rebuild `self._search = GumbelRootSearch(...)` to reset
per-match state. The rebuild **didn't pass `move_prior_fn` or
`value_fn`**. Both default to `None` on the dataclass, so even though
the bundle's MCTSAgent was constructed with both NN functions, they
were silently dropped at the start of every match.

**Symptom in the wild**: every `--rollout-policy=nn_value` bundle
fell back to heuristic rollouts via the
`rollout_policy='nn_value' but no value_fn supplied` warning path.
The warning fires once per process, so multi-game self-play didn't
surface it after game 1.

**Fix**: thread `move_prior_fn=self._search.move_prior_fn,
value_fn=self._search.value_fn` through the rebuild. + regression
test `test_phantom5_fresh_game_preserves_move_prior_fn_and_value_fn`.

### PHANTOM 6.0 — `_softmax` name collision

**Root cause**: two source modules defined a top-level helper named
`_softmax`:
- `mcts/actions.py:125` — `def _softmax(xs: List[float], temperature: float) -> List[float]:`
- `opponent/bayes.py:89` — `def _softmax(x: np.ndarray) -> np.ndarray:`

When `tools/bundle.py` inlines them, both end up in the SAME module
namespace. Python's last-definition-wins rule meant the bayes 1-arg
version shadowed the actions 2-arg version. When
`generate_per_planet_moves` (in actions.py) called
`_softmax(scores, cfg.softmax_temperature)`, it dispatched to the
1-arg bayes version and **raised TypeError**. The TypeError was
caught by the outer `except Exception` in `mcts_bot.act()` and
returned `heuristic_move`. **Net effect: search ran 0 rollouts on
every turn**, regardless of `total_sims` or `hard_deadline_ms`.

**Symptom in the wild** (post-Phantom-4 fix only):
- `_value_fn_eval` calls per match: 0
- `sequential_halving` calls per match: 0 (entered, returned with `n_rollouts=0`)
- Wire actions byte-identical to a pure-heuristic agent

**Fix**: rename `bayes._softmax` to `_softmax_np`. + regression test
`test_bundle_no_top_level_function_name_collisions` that scans the
bundled `.py` for any duplicate top-level `def` names (allowlists
`build` since both `heuristic.build` and `mcts_bot.build` are
factory entrypoints, never called within the bundle).

### Bundler bug — stray `from orbitwars.nn.nn_value import ...`

**Root cause**: commit 24d784b (2026-04-26 21:01) added a
`make_nn_value_fn(...)` closure to the bundler's NN-bootstrap output
but emitted a runtime `from orbitwars.nn.nn_value import
make_nn_value_fn` line instead of inlining the module. Kaggle
sandbox has no `orbitwars` package, so the import raised
ModuleNotFoundError at notebook-load time (line 11498 of v30c).
Every v30* submission ERROR'd; v27 (built before this commit) still
worked.

**Fix**: add `nn.nn_value` AND `features.obs_encode` (the latter
needed by both `nn_prior` and `nn_value` for `encode_grid`) to the
bundler's inlined-modules list when `--nn-checkpoint` is set. Remove
the stray runtime-import line. + regression test that asserts the
bundle has zero remaining `from orbitwars.*` imports.

### Compounding effect

The three phantoms hid each other:
1. **Phantom 4** dropped `rollout_policy` at search time, so even if
   `value_fn` had been wired correctly, the search would have run
   the heuristic rollout path.
2. **Phantom 5** dropped `value_fn` at every match start, so even if
   `rollout_policy=nn_value` had been preserved, the search would
   warn-and-fall-back to heuristic rollouts.
3. **Phantom 6** raised TypeError before any rollout fired, so even
   if both 4 and 5 had been fixed, the search would still return the
   heuristic anchor.

**Pre-fix MCTS strength on the ladder = heuristic strength.** All
TuRBO weight tuning, all NN training, all macro experiments,
all opponent-model A/B runs landed on a phantom that played the
heuristic regardless of the bundle config. The heuristic itself
got tuned (TuRBO weights translate to anchor moves), so the ladder
was sensible — just not measuring what we thought.

### Numbers

| Test | Pre-Phantom-fix v32b | Post-Phantom-fix v32b |
|---|---|---|
| `_value_fn_eval` calls / 125-turn match | 0 | (N/A — heuristic rollouts) |
| `_rollout_value` calls / 125-turn match | 0 | 1,072 |
| `sequential_halving` rollouts / turn | 0 | ~8-12 |
| Mirror H2H vs v27 (8 games) | tied (phantom-vs-phantom) | **7-1, +338 Elo** |
| Public ladder Elo | 886.1 (v27 yesterday) | **939.1 (v32b today)** |

### Slot 2 candidates explored, all rejected

- **v32a (NN+v3 value head, anchor margin 0)**: lost 0-8 to v32b,
  -800 Elo. The v3 value head was trained on Phantom-stricken
  (heuristic-only) demos, so it confidently mis-evaluates real-MCTS
  candidate states.
- **v32c (NN+v3 + anchor margin 0.3)**: lost 2-6 to v32b, -190 Elo.
  Anchor margin reduces the damage but doesn't make a bad value
  head useful.
- **v32d (heuristic rollouts + macros)**: lost 0-8 to v32b, -800
  Elo. Macros add candidate joints that aren't anchor-protected,
  diluting visit count per real candidate. Net negative under the
  current ~12 sims/round budget.

**Conclusion**: slot 2 saved for v33 once a coherent value head is
trained on post-Phantom-fix self-play demos. Closed-loop iter 1
running.

### Iter 1 closed-loop result (2026-04-29 morning)

Collected 12,607 demos from 12 self-play games with v32b config
(real MCTS, heuristic rollouts, total_sims=128, deadline 850ms).
Trained v4 value head (val_mse=0.20) and v4 policy head (val_ce=1.46)
via frozen-backbone iteration.

**v33 (NN value head as leaf eval)** — `rollout_policy=nn_value`
with v4 value head. 8-game H2H lost 0-8 (-800 Elo). Try v33b with
`anchor_margin=0.5` to defensively gate overrides; lost 2-6 (-190).
Conclusion: at current value-head quality, NN-as-leaf is dominated
by 15-ply heuristic rollouts even with anchor protection.

**v34 (visit-distilled prior + heuristic rollouts)** — kept
heuristic rollouts but swapped in the policy-head trained on visit
distributions. 8-game H2H showed +88 Elo (62.5% wr) but a 16-game
confirmation flipped to -44 Elo (43.8%, CI ±12). The 8-game CI was
±17 — too noisy to make slot decisions.

**Diagnosis**: visit_dist concentration analysis shows **85% of
demos have max-channel > 0.9** (essentially one-hot). MCTS+SH
converges very aggressively at 128 sims × 8 channels per planet,
so the visit distribution is already a hard label. Distilling onto
that gives ~the same signal as BC's heuristic-pick imitation. The
policy head's 264 trainable params couldn't extract a meaningful
update.

**Next-day plan**:
1. Collect demos with `tau > 0` (visit-temperature smoothing) so
   the policy target has more entropy. Look at AlphaZero's tau=1
   for first 30 plies; we should at least try tau=0.5 across all
   plies on this small grid.
2. Scale to 50+ games (vs 12) so the policy head sees more
   per-planet position diversity.
3. Joint policy+value training (unfreeze the backbone after first
   few epochs of head-only) to get a real policy improvement, not
   a final-layer-only nudge.

### Slot 2 (2026-04-29) was held — no candidate beat v32b reliably.

Open follow-ups (pre-existing):
- Re-run TuRBO over heuristic weights now that real MCTS visits
  meaningfully different heuristic states.
- Audit `enumerate_joints` / `generate_per_planet_moves` early-exit
  paths (e.g. `len(joints) == 1` early return at
  gumbel_search.py:886 — innocuous but worth knowing).

---

## DAY-OF-2026-04-27 — PHANTOM 4.0 FOUND (and it explains EVERYTHING)

**The biggest bug discovery in this project.** Found while diagnosing
why v_v2value (NN value head + nn_value rollout) was wire-identical
to v22.

**Root cause**: `mcts_bot.py` line ~449 builds a `tight_cfg = GumbelConfig(...)`
on every `act()` call to inject the safe-budget deadline. The
constructor **only copied 5 fields** (num_candidates, total_sims,
rollout_depth, hard_deadline_ms, anchor_improvement_margin) and
silently REVERTED all other fields to GumbelConfig defaults.

**Specifically dropped at search time**:
| Field | Bundle setting | Actual runtime value |
|---|---|---|
| `rollout_policy` | "fast" / "nn_value" | "heuristic" (default) |
| `sim_move_variant` | "exp3" | "ucb" (default) |
| `exp3_eta` | 0.3 | 0.3 (irrelevant since ucb) |
| `use_decoupled_sim_move` | False (in some tests) | True (default — always fired in mcts_bot.__init__) |
| `use_macros` | True (v16) | False (default) |
| `per_rollout_budget_ms` | (whatever) | None (default) |
| `num_opp_candidates` | (whatever) | 5 (default) |

**Confirmed via diagnostic**:
- Pre-fix: nn_value bundle → 4 sims/turn, anchor wins, q_values=[0.04, -inf, -inf, -inf]
- Post-fix: nn_value bundle → 128 sims/turn (full budget), best_idx=2 (NOT anchor), q_values=[-0.55, -0.55, -0.49, -0.60]

**This bug has existed since `--sim-move-variant` and `--rollout-policy`
flags were introduced (~v9, ~2026-04-24). Every Kaggle ladder
submission since then has been running with**:
- `rollout_policy="heuristic"` (full HeuristicAgent rollouts, 30-100ms each → only 4-15 sims fit per turn → anchor wins by default)
- `sim_move_variant="ucb"` (the v9 "exp3 wins +52.5" A/B was real but never shipped)
- `use_macros=False` (v16 macros never actually fired)
- `use_decoupled_sim_move=True` (this WAS actually in default, so this part was unaffected)

**Retroactively explains**:
1. **All +51.8 Elo H2H phantoms.** Different bundles produced byte-identical play because all silently reverted to identical defaults. The "first bundle in list" win was bundle-load-order RNG variance, not bot strength.
2. **Why MCTS bundles always behaved like the heuristic.** The bundle's claim of "fast rollouts" + "NN priors" was advertising; the runtime ran 10-15 sims of full HeuristicAgent rollouts then anchor-locked.
3. **Why NN bundles were wire-identical.** No matter what NN you trained, the runtime ran heuristic rollouts and anchor protection wins.
4. **Why we couldn't find a "real signal"** — every config we tried got dropped to defaults. We were testing `default-vs-default` with cosmetically-different bundle files.

**v22 (current ladder leader before fix)** actual runtime:
- BC v1 small NN as `move_prior_fn` (this WAS preserved — passed via constructor not gumbel_cfg)
- TuRBO v3 weights (preserved — same path)
- 128 sims requested, but only ~10-15 actually completed due to slow heuristic rollouts
- ucb sim-move bandit (not exp3 as advertised)
- decoupled sim-move ON (correct)
- anchor_margin=0 (preserved)
- `use_opponent_model=False` for v26: this DID work (set on agent directly, not gumbel_cfg)

**Implications for next ship**:
- A v30 with the fix that ACTUALLY runs `rollout_policy=fast` should do
  ~100-200 sims/turn (vs 10-15) and produce meaningfully different Q
  estimates. Could be a real ladder gain.
- A v30 with `rollout_policy=nn_value` + `bc_v_v2_deep.pt` is the
  first time NN actually drives wire actions on Kaggle. Real test
  of the AlphaZero direction.
- All past "didn't help" findings need re-validation with the fix.

**Fix landed**: tight_cfg now copies all relevant fields. 100/100
tests pass (`test_mcts_bot.py + test_gumbel_search.py + test_bundle.py`).

---

## NIGHT-OF-2026-04-26 SUMMARY (the strategic-pivot night)

**Where we stand**: v22 = 936.9 ladder Elo, frozen. 5 ladder ships today
(v22, v24, v25, v26, v27); all of v24-v27 settled below v22. **The
heuristic-tuning lever has plateaued** — TuRBO v3 → v5 produces wider
spread of weight values but no ladder lift.

**Five heuristic-improvement attempts tonight, all regressive in
heuristic-vs-heuristic A/B**:
| Attempt | Δ Elo |
|---|---|
| Exact-plus-one neutrals (line 978 fix) | **-107** |
| Full smart sizing (race+counter+speed minimum-viable) | **-458** |
| Soft sizing (OPP_COMMITMENT=0.5 dampening) | **-458** |
| Context-aware (legacy in race, smart isolated) | **-800** |
| Cluster-aware target selection | **-207** |

**Lesson**: legacy heuristic with TuRBO-tuned weights is approximately
Pareto-optimal at this complexity level. Local rule tweaks lose to
"send 30 ships to nearest target with positive score." The implicit
race dynamics (snowball wins by claiming neutrals fastest) are encoded
into `min_launch_size=20` in a way that's hard to improve via tighter
math without OPPONENT MODELING — knowing what the opponent will
actually do, not just bounding their worst case.

**NN/PPO findings (Phase 1+2A done)**:
- `src/orbitwars/nn/nn_value.py` — value-fn factory, pathway plumbed
  through `GumbelRootSearch.value_fn` + `bundle.py --rollout-policy=nn_value`
- 50k demos collected with terminal_value labels (Path A)
- Value head trained val_mse=0.045 (94% below baseline), genuinely learned
- **v28 (trained value head, anchor_margin=1.0) wire-identical to v22**
  — value_fn IS being called but anchor-lock + heuristic-rollouts
  smother it at the wire across 5 seeds × 2 mirrors
- Implies single-step rollout-distillation cannot push past current
  policy strength. Closed-loop AlphaZero iteration required.

**Infrastructure shipped tonight (toggleable, default OFF)**:
- `_min_viable_attack_fleet`, `_enemy_fastest_arrival_at`,
  `_race_min_ships`, `_estimate_counter_attack_threat`,
  `_find_neutral_clusters`, `_score_cluster_for_claim`
  — all in `bots/heuristic.py`, gated by `legacy_neutral_floor`,
  `legacy_enemy_floor`, `cluster_strategy_weight` weights
- Tools: `tools/h2h_mirror.py`, `tools/h2h_isolated.py`,
  `tools/_h2h_one_game.py` (HEURISTIC-ONLY validated; unreliable for
  MCTS bundles due to wall-clock phantom)
- `tools/compare_neutral_fix.py`, `tools/compare_cluster_strategy.py`,
  `tools/train_value_head.py`
- Demos: `runs/mcts_demos_v6_with_outcomes.npz` (50k, sims=64/dl=400ms),
  `runs/mcts_demos_v7_deep.npz` (24k, sims=128/dl=850ms)
- Trained checkpoint: `runs/bc_v_v1.pt` (BC v1 with trained value head)

**Strategic pivot — committing to RL track**:
The hand-tuning era is over. Path forward (multi-day):
1. **Architecture fix**: multiple-size candidates per target in
   `mcts/actions.py` + `--neural-rollout-policy` so MCTS Q reflects
   NN strategy not heuristic strategy. Unblocks NN-on-wire.
2. **Real PPO + PFSP**: 50-100M env steps with bug-fixed `train_ppo.py`
   (bootstrap value, shaped reward, per-env GAE). RTX 3070 makes
   24-48h training tractable.
3. **Closed-loop self-play**: snapshot policy every N steps, add to
   PFSP pool, iterate. AlphaZero in spirit.
4. **Distill + ship**: <2MB student via Bayesian Policy Distillation,
   int8 quantize, ship v30+ with NN actually driving wire.

Realistic outcome at our compute scale: +50 to +200 ladder Elo.
Tier-2 (1300+) is plausible-stretch.

**Cleanup pass done**: removed ephemeral bundles
(v_value_smoke, v_no_oppmodel, v_ucb_no_oppmodel, v_no_nn_diag,
v22_clone, v_ppo_test, v_ppo_v3_dead, v28_m1) and smoke checkpoints
(ppo_smoke*.pt, bc_v_smoke.pt). 83 tests still pass.

**Don't ship tonight** — v22 stable floor; RL track work begins
tomorrow.

---



## URGENT FINDING (2026-04-26 morning) — `+51.8 Elo H2H` IS A FALSE POSITIVE

**Six "+51.8 Elo H2H" wins reported in this STATUS doc on 2026-04-25 are all the
same harness/seat-ordering artifact, not bot strength.**

Today's reproduction: `v22 vs v22_clone` (literally `cp v22.py v22_clone.py` —
identical bundle) produces **EXACTLY** 9W-4L-7T = +51.8 Elo. Same per-game step
counts as `v_ppo_test vs v22`, `ppo_v3_dead vs v22`, `v_no_nn_diag vs v22`. The
"first bundle in the round-robin list" always wins 9-4-7 regardless of content.

**Mechanism**:
1. `anchor_improvement_margin=0.0` + `rollout_policy='fast'` (heuristic-driven
   rollouts) means MCTS Q-estimates are biased toward the heuristic action.
   Anchor (heuristic pick) is anchor-locked from SH pruning. Search visits
   non-anchor candidates but rollout returns under heuristic continuation are
   not enough to overcome anchor's low-variance Q.
2. The 8-channel `ACTION_LOOKUP` collapses to ~1-3 unique wire actions per
   state (most channels map to "no matching heuristic move" → HOLD). NN logits
   differ by mean abs 1.83 between BC and PPO checkpoints, but the heuristic
   decoder picks the same wire action regardless of channel choice.
3. Result: every NN-on bundle plays byte-identical wire actions to every other
   NN-on bundle (and to no-NN). Only RNG state in the harness differs.
4. Our seed sequence (42, 1042, 2042, ...) happens to favor whichever bundle
   plays seat-0 in even seeds + seat-1 in odd seeds. The harness puts the
   "first bundle in --bundles list" on those favored seats. Hence the
   reproducible 9-4-7.

**Falsified claims** (revisit with skepticism):
- "BC v3 small (MCTS-distilled) +51.8 H2H" — fake, PPO showed same with completely different NN.
- "Macros +51.8 H2H lift" — fake, same harness artifact (and macros HURT on ladder).
- "v_ppo_test +51.8 vs v22" — fake; v25 ladder result will likely match v22, not exceed.
- Any past "MCTS variant beats heuristic by ~50 Elo locally" claim — re-test required.

**Real signals** (still trustworthy):
- The actual Kaggle ladder (v22@943, v15@880, v8@762, etc.).
- TuRBO win-rate against the archetype pool (different opponents, real diversity).
- BC val_acc on held-out demos (model-quality metric, not gameplay).

**Action**:
- Killed PPO v4 (compute waste — NN doesn't reach the wire under current anchor-lock).
- Don't trust any "+50 Elo H2H" gate from `diag_bundle_round_robin.py` for two
  bundles that differ only in NN/MCTS config under anchor_margin=0.
- Honest H2H requires either: (a) MUCH more games (200+ to average out seat
  bias), (b) random per-game seat assignment, or (c) opponent variety — H2H
  vs heuristic_v0/archetypes/older versions where the bundles' wire actions
  *actually* diverge.
- To make NN matter on the wire: lower anchor's protection OR use NN-driven
  rollouts OR pure value-head Q (no rollouts, AlphaZero-style). All of these
  are riskier than current anchor-locked play and need their own gates.

## DOWNSTREAM FINDING (2026-04-26 morning) — `use_opponent_model=False` IS A REAL +70 ELO WIN

After `tools/h2h_mirror.py` was built (mirror matches eliminate seat bias),
re-ran v_no_nn_diag vs v22 → **+70.4 Elo (wr=0.600, SE=0.110, 20 games)**.

Then disambiguated by building `v_no_oppmodel` (v22 config but
`use_opponent_model=False`, NN STILL ON) and running mirror vs v22:
**EXACTLY the same +70.4 Elo, byte-identical per-game outcomes** as
v_no_nn_diag vs v22.

Conclusion: the +70 Elo signal comes from disabling the Bayesian opponent
model, NOT from removing the NN. The NN provides zero wire-action influence
either way (consistent with the upstream finding above).

**Mechanism (hypothesis)**: opp_model uses a Dirichlet posterior over archetypes
to bias rollouts toward an exploitative opponent. If the posterior wrongly
concentrates on a hard archetype (e.g. "rusher"), MCTS Q estimates become
pessimistic → bot plays too conservatively → loses long games on ship-count
tiebreaks. Indeed, both winning-mirror seeds (3042, 6042) end at step=499
(max length) where ship totals decide.

**Built `submissions/v26.py`** = `v22` config but `use_opponent_model=False`.
Bundle is 658,616 bytes (matches v22 size since both still embed the BC NN —
the NN is benign even though it provides no wire signal). Ready to ship.

**Historical context**: `v10_fast_m0p5_noopp` (2026-04-25) had
`use_opponent_model=False` and was DECIDED NOT TO SHIP based on byte-identical
H2H vs v9. We now know that's a false negative — the harness was masking the
real +70 Elo signal. **v10 was probably a real winner that we left on the
table.** v26 is essentially v10's contribution applied to the v22 weight stack.

**30-seed CI confirmation** (`runs/v26_vs_v22_mirror_30seeds.log`): v26 vs
v22 at seed=100, 30 seeds × 2 mirrors = 60 games → **47W-13L = wr 0.783 ±
0.053, Elo +223.3**. The 10-seed test was unlucky — its specific seed offsets
landed in a "stalemate cluster" where mirrors mostly cancel; the 30-seed
test reveals 17/30 seeds give v26 a dominant double-win. Effect is robust.

**v25 LADDER LANDED (2026-04-26 ~02:45 UTC submission, ~5h settle)**: 795.6.
**This is -141 Elo BELOW v22 at 936.9.** v25 = PPO 50-update fine-tune of
BC v1 + v22 config — the PPO training that we now know didn't change wire
actions in MOST games but does in some via NN-influenced search. The
ladder result confirms: NN-on-wire is **actively harmful** in the
configurations we tested. **v25 is the strongest direct evidence yet that
the NN-prior path under anchor_margin=0 is value-destroying, not value-
adding.** Combined with the local +223 Elo signal for `use_opponent_model
=False` (with NN still embedded but benign), the next ship is unambiguous.

**v26 SHIPPED (2026-04-26 08:15 UTC, status PENDING)**: same bundle as
the +223-Elo mirror winner (`submissions/v26.py`, kernel
`gardan4/orbit-wars-mcts-v26`, version 1, 760 KB notebook). Submission
message records the local mirror result and the v25 falsification.
Awaiting ladder settle (typically 4-12h to stabilize). If v26 settles
above v22's 936.9, that's the strongest single-ship gain in this whole
effort and validates both (a) the mirror-fair H2H methodology and
(b) the opp_model regression hypothesis.

## PHANTOM 2.0 (2026-04-26 ~11:30 UTC) — `tools/h2h_mirror.py` ALSO has a seed-dependent bias

A follow-up control test ran `v22 vs v22_clone` (literally identical bundle
via `cp v22.py v22_clone.py`) under mirror H2H at **seed=100, 5 seeds = 10
games** → **8W-2L = wr 0.800, +240 Elo for the FIRST bundle in `--bundles`**.

Earlier v22 vs v22_clone at seed=42 with 3 seeds gave 3-3 cancellation.
But at seed=100, mirror does NOT cancel. The phantom is seed-dependent:

| seed=42, 3 seeds | seed=100, 5 seeds |
|---|---|
| s 42 m0 LOSS m1 WIN — cancels | s 100 m0 WIN m1 LOSS — cancels |
| s 1042 m0 LOSS m1 WIN — cancels | **s 1100 m0 WIN m1 WIN — first wins both** |
| s 2042 m0 WIN m1 LOSS — cancels | s 2100 m0 WIN m1 LOSS — cancels |
| | **s 3100 m0 WIN m1 WIN — first wins both** |
| | **s 4100 m0 WIN m1 WIN — first wins both** |

For seeds 1100/3100/4100, the FIRST bundle wins both halves of the mirror.
This violates the mirror-fairness assumption that "if A=B byte-identical,
swapping seats just swaps the outcome." The fact that v22 vs v22_clone
(literally the same code) shows this means the mirror harness has another
artifact for these seeds — either the env's internal random state isn't
fully reset between calls, or kaggle_environments has shared module state
that depends on factory call order.

**Falsified results** (revisit with extreme skepticism):
- "v_no_oppmodel beats v22 by +223 Elo" — same exact pattern as v22
  vs v22_clone in the same seed range. PHANTOM.
- "UCB beats EXP3 by +223 Elo" — same exact per-game outcomes as the
  v_no_oppmodel result. PHANTOM.
- "v26 should land ~150 Elo above v22 on ladder" — wrong; v26 is wire-
  equivalent to v22 in most games.

**Real signal**:
- v25 ladder = 795.6 (-141 vs v22 at 936.9). PPO-on-wire genuinely harmful.
- v26 ladder = 901.3 (early, settling) → likely lands near v22, not above.

**Action**:
- Don't run any more `tools/h2h_mirror.py` H2H tests until the seed-100
  artifact is understood. The seed=42 control passed only by coincidence.
- The honest H2H methodology probably requires either: (a) full RNG state
  reset including numpy/torch globals between mirrors, (b) running each
  game in a fresh subprocess, (c) trusting only the actual Kaggle ladder.
- For now, **the actual Kaggle ladder is the only honest signal**. v26
  result will tell us if `use_opponent_model=False` truly does anything.

**The methodology lesson here is the same as the original phantom**: any
"+50 to +250 Elo H2H signal" that shows the same per-game outcomes
across multiple bundle pairs should be treated as a harness artifact
until proven otherwise. The mirror-cancellation property only holds when
ALL stochasticity sources (Python random, numpy random, torch RNG, env
internal state) are fully reset between mirror halves.

## PHANTOM 3.0 (2026-04-26 ~11:45 UTC) — Subprocess isolation does NOT fix it

Built `tools/h2h_isolated.py` + `tools/_h2h_one_game.py` to run each
game in a completely fresh Python subprocess (no shared state, no RNG
drift, no kaggle_environments cached singletons). Re-ran v22 vs
v22_clone at seed=100, 5 seeds = 10 games:
**8W-2L = wr 0.800 = +240 Elo for first bundle**. SAME EXACT PATTERN.

This rules out:
- Shared module state in the parent Python process
- numpy/torch global RNG drift between mirrors (fix didn't help)
- kaggle_environments cached state across env.run() calls
- Bundle import-time state leakage

The non-determinism is inherent to **MCTS wall-clock-dependent decisions**.
Even when `hard_deadline_ms` never binds (p95 turn times 50-75ms vs
850ms deadline), the search's internal `time.perf_counter()` use for
round-budget allocation in Sequential Halving cascades microsecond
variations into different visit counts → different actions → different
final ship counts in long games. Over 500 turns the variations compound.

**Implication: local H2H is unreliable for any bundle that uses MCTS
with wall-clock timing — the production config.** No amount of harness
isolation can fix this without changing the bundle itself.

## v26 LADDER (2026-04-26 ~11:45 UTC) — settling around 856-902, BELOW v22

v26 = v22 config + `use_opponent_model=False`. Submitted ~08:15 UTC.
Trajectory: 600 → 901 → 856 → 902 (bouncing in 856-902 range).
v22 = 936.9. **v26 is consistently ~30-80 Elo BELOW v22.**

Conclusion: `use_opponent_model=True` (the v22 default) is genuinely
helpful on the ladder despite being silenced by anchor-locked play in
local tests. The Bayesian archetype posterior probably helps in long-
running games or against specific opponent types — exactly what would
NOT show up in our seed=42/100 mirror tests.

## CONSOLIDATED FINDING 2026-04-26

After today's investigation, the honest picture is:
- **v22 (936.9) is approximately optimal among tested configurations**.
- **NN/PPO under anchor_margin=0 + heuristic-fast rollouts cannot reach
  the wire**: confirmed by byte-identical wire actions across BC v1 small,
  PPO 50-update, PPO 500-update collapsed checkpoints.
- **Active harm from PPO experimental ship**: v25 ladder = 795.6, -141 Elo.
- **opp_model is mildly helpful**: disabling it landed v26 at -50 Elo.
- **All recent +50/+223 Elo local "wins"** (BC v3, macros, PPO, opp_model
  disabled, UCB-vs-EXP3) **were phantom signals from harness order bias**.

To genuinely improve beyond v22 we need to escape the anchor-lock trap:
either (a) widen TuRBO bounds (`PARAM_BOUNDS_V4` is staged but not
launched), (b) use NN to drive rollouts (so search Q reflects NN strategy,
not heuristic), (c) AlphaZero-style pure value-head Q (no rollouts), or
(d) redesign the action space so the heuristic decoder doesn't collapse
NN preferences to identical wire actions.

---

*Last updated: 2026-04-25 Day 3 LATE evening — **5 NN-prior bots shipped today (v12/v15/v16 frozen, v19+v20i active settling, v18 ERROR). v15 frozen 879.9 (best). 6 H2H ablations all +51.8 Elo, N=60 confirmed signal real (+63 Elo). int8 quantization (4× compression) shipped — unblocks BIG BC ship path (~1MB Kaggle bundle limit empirical). MCTS-as-teacher (107k demos), soft-target BC training, 4-fold augmentation, macros library, Kaggle Dataset path all infrastructure shipped (136 tests across changed modules pass). BC v3 small (va_acc=0.421, MCTS-distilled) trained on teacher demos. **LEADERBOARD REALITY CHECK: #1 kovi at 2560 (outlier), tier 2 at 1300-1600, us at ~880. Gap to tier 2 is +400-700 Elo — +50 Elo H2H steps are too slow.** Strategy pivots: PPO from BC (Plan W5) becomes priority #1, game-specific math improvements #2, TuRBO with ladder-mimicking pool #3.** Earlier today (W4 live; **v11 LADDER FLOOR HOLDING AT 880.8 AFTER FULL OVERNIGHT SETTLE (+30 Elo over v9 at 850.7, +90 over v8 at 790.7).** Peaked 926.0 at 23:48 UTC; 6.5h gap (PC restart) and v11 still came back UP from 875.5 → 880.8 = the +30 Elo signal is durable. **NN prior bridge SHIPPED**: `src/orbitwars/nn/nn_prior.py` (15 unit tests) + `move_prior_fn` plumbed through `GumbelRootSearch` + `MCTSAgent` (3 integration tests, defensive fallback on NN failure). **`tools/bundle.py --nn-checkpoint` SHIPPED** with base64-inline checkpoint embedding (~2.4 MB encoded for 1.8 MB raw); 7 new bundle tests, full v12 ship pipeline orchestrated by `tools/build_v12.py`. **End-to-end smoke PASS** with random-init fake checkpoint: bundle 2.9 MB, import+bootstrap 0.37s, full game vs random WIN +1 in 175 steps. **CRITICAL: NN-prior-active validation passed** — `tools/smoke_v12_diff_vs_baseline.py` runs v12-with-NN vs v12-without-NN at the same seed and confirms action sequences diverge at turn 16, 95/154 turns differ (61.7% divergence). This is the test that v10 *failed* (byte-identical to v8). v10's compute-bound finding ("more sims + lower margin doesn't help without a learned prior") is now confirmed: the prior makes margin=0.5 + fast rollouts a real lever. Both random-init configs still won +1 vs random. `tools/diag_nn_prior.py` compares heuristic vs random vs BC priors on a fixed obs (KL divergences) — runs the moment the real checkpoint lands. **BC warmstart relaunched with `python -u` + GPU isolation** (the previous run shared GPU with concurrent v12 smoke runs, hitting 7.7GB/8GB pressure; cleaned up to 758 MB before restart); same seed 0, 15 epochs, eager-save patch in place. **TuRBO-v4 60-trial relaunched** (first cycle reached best=0.933 at trial 13 before being killed alongside the BC restart; same seed 2, BoTorch trust-region kicks in around trial 20). v4 plausibly doesn't saturate at 1.000 like v3 did — bound expansion may be revealing genuine optima beyond v3's edge. **2-hour settle complete.** TuRBO-v3 trial-42 weights + exp3 eta=0.3; H2H vs v9 17W-3L = 85% wr Elo+151, the strongest H2H signal in the v5→v11 lineage. **Key observation: ladder delta (+32 Elo) is MUCH smaller than H2H predicted (~+200). v3 archetype-pool overfitting is real but not catastrophic — weights still generalize positively.** Extended overnight poll running for 10h to confirm final settle band.** Also noted during submission: ladder is still volatile — v9 moved 934.4 → 864.3, v8 moved 769.5 → 790.7 between 21:36 and 22:53 UTC, so even the "settled" v9 floor is still being re-computed. **TuRBO-v4 bounds config staged** in `src/orbitwars/tune/turbo_runner.py::PARAM_BOUNDS_V4` (5 v3 edge-hits expanded: w_production 20→40, mult_enemy 5→10, mult_comet 5→10, min_launch_size 30→50, comet_max_time_mismatch 5→10; stall-safety preserved by tightening keep_reserve_ships 15→3). Launchable via `--bounds v4 --strategy ax --n-trials 60 --seed 2 --pool w2 --games-per-opp 5 --workers 7`. **NOT launched yet** — decision gated on v11 ladder settle: if v11 wins the ladder clearly we launch v4; if v11 regresses (v3 weights may overfit archetype pool) we pivot off TuRBO to BC-warmstart-PPO. 4 new tests (v4-keys-match-v3, v4-valid-intervals, v4-stall-safety, bounds-dispatch) pass; 29/29 turbo+bundle tests green. **TuRBO-v3 COMPLETE: 60-trial fresh Sobol init + BoTorch, trial 42 hit win_rate=1.000 (45/45: 5/5 vs EACH of 9 archetypes).** Trust-region narrowed around trial 42 gave 18 follow-up trials oscillating 0.756-0.978 → this is a stable optimum, not a single lucky trial. Weights are MUCH more aggressive than v2: `mult_comet=5.0`, `mult_enemy=5.0`, `mult_reinforce_ally=0.0` (disables reinforce), `keep_reserve_ships=0.0` (disables reserve), `w_production=20.0` (maxed), `min_launch_size=30.0` (maxed). Multiple weights at search-bound edges = TRUE optimum may be outside; worth a follow-on TuRBO run with wider bounds. **v11 bundled = v3 weights + sim_move_variant=exp3 (232,710 bytes); smoke test WIN vs random rewards=[1,-1] steps=99**. v11 vs v9 H2H is running (20 games, ~19min wall). Still need the ladder defense confirmation — 45/45 vs our training pool is strong evidence of generalization but not proof it beats v9's exp3-on-v8-weights specifically. **v9 IS THE NEW LADDER FLOOR AT 934.4 AND CLIMBING — +165 Elo over v8 at 769.5, peaked at 981.9 (+212 Elo) at 21:26**, reversing the early-ladder -154 Elo signal. v9's marginal bundled H2H (+7 Elo, wr=0.550) was weak but not noise — the ladder saw +121 Elo of real signal once v9 played a full settle batch. Lesson: 20-game bundled H2H has SE ~0.112 and cannot detect gains under ~50 Elo; the ladder at N~50-100 replays per settle is ~2× more sensitive. Early-ladder scores in the first 30 min are near-useless — you need overnight settle before calling a ladder trend. **Do NOT make submission decisions on <6h ladder data.** Pre-v9-settle status below preserved. **v8 LIVE ON KAGGLE at 762.3, +113 Elo over v7 at 649.1** — confirmed ladder win for the TuRBO-v2 tuned bot. Submission trail: initial "COMPLETE" landing at 600.0 within ~5 min of push, then settled up to 762.3 over subsequent ladder games. v7 itself ticked up from 630.1 → 649.1 on the same day (ladder replays with fresh opponents). Local H2H (v8 18-2-0 vs v7, +175.8 locally) **predicted the ladder direction correctly for the first time in three submissions** — paired-seed compare_weights (61.7%) + TuRBO-v2 tuning (+0.233 wr vs defaults at N=90) both held on the public ladder. Next submission floor: v8 @ 762.3. **W3→W4 transition fully complete.** Pre-v8 status below preserved: v7 at 630.1 back ahead of v6 at 594.1 by +36 Elo after overnight settle, H1 noise-hypothesis confirmed. **NN torch forward passes live**, 13 forward-pass tests green. **TuRBO-v2 complete: 30 trials, best trial 22 win=0.800** (+0.133 over 0.667 defaults baseline, well above the 0.07 ship margin). **compare_weights ran at N=90 games/side, workers=7: turbo_v2 0.833 vs defaults 0.600, delta_wr=+0.233, 95% CI [+0.106, +0.361], paired: turbo_v2 wins on 61.7% of seeds. Worst per-opp regression: rusher -0.100 (exactly at the -0.10 boundary), all other 8 opps positive or tied. `tools/ship_gate.py` VERDICT: SHIP turbo_v2 (both gates PASS).** **v8 bundled: `submissions/v8.py` (229714 bytes) with HEURISTIC_WEIGHTS.update from trial-22 weights; smoke test `rewards=[1,-1] steps=114` vs random.** **v8 vs v7 bundled H2H (20 games, 1.0s timeout, alternating seats): v8 18-2-0 (90%), Elo delta +175.8 locally** — WAY above the 55% defense gate. Turn times: v8 p50=36ms / p95=66ms, v7 p50=46ms / p95=103ms, both comfortably under the 1s budget. **W4 Exp3 A/B infrastructure shipped** — `decoupled_exp3_root` now accepts `protected_my_idx` (anchor-lock contract matches UCB) + `rng` (deterministic tests); `GumbelSearchConfig.sim_move_variant: "ucb"|"exp3"` dispatches at the wiring point with a graceful warn-and-fallback on typo; 5 new tests green (2 sim_move + 3 gumbel_search dispatch), 47/47 pass in the touched modules. **Full regression suite green: 285 passed, 3 skipped, 4 warnings in 4683s + 8 new `tests/test_bundle.py` tests for `--sim-move-variant` + `--weights-json` + their interaction (293 total now)** — Exp3 wiring + v8 bundle clean across the full matrix, not just the touched modules. **Exp3 live-smoke passed**: MCTSAgent with `sim_move_variant="exp3"` completed a 500-turn real game vs HeuristicAgent (seat 1), winning +1 with p50=332ms / p95=389ms / max=579ms turn times — all well under the 1.0s step_timeout. **Exp3 2-game AB smoke PASS** (seed 1000-1001 at hard_deadline_ms=150): exp3 2W-0L-0D vs ucb at matched compute (p50 exp3=176ms, ucb=175ms — identical budget usage). **W4 Exp3 A/B gate: PASS.** 20-game head-to-head at production settings (total_sims=32, num_candidates=4, rollout_depth=15, hard_deadline_ms=300ms, eta=0.3) closed 103.3 min wall with exp3 **11W-8L-1D → wr=0.575, Elo +52.5** over ucb. Seat split is asymmetric — seat 0 wr=0.450 / seat 1 wr=0.700 — so seat-1 carries all the signal and seat-0 regresses slightly, but the point-estimate gate (≥0.55) is clean. Turn-time overhead negligible: p50 exp3=339ms / ucb=332ms (+2%), max-max exp3=1168ms / ucb=1176ms (both well under 1s Kaggle budget, opening-turn overage-bank usage is identical). **`tools/bundle.py` extended with `--sim-move-variant {ucb,exp3}` + `--exp3-eta`** flags so v9 is a 1-command bundle; 8 new tests in `tests/test_bundle.py` (default-no-inject, exp3-injects-cfg, default-eta, explicit-ucb, reject-unknown, require-mcts-bot, end-to-end-importable, weights+variant-coexist-and-ordered) all pass. **v9 bundled + defense H2H: marginally passes.** `submissions/v9.py` (232,861 bytes) built via `tools.bundle --weights-json runs/turbo_v2_20260424.json --sim-move-variant exp3 --exp3-eta 0.3`; smoke test `rewards=[1,-1] steps=137` vs random. v9 vs v8 bundled H2H (20 games, 1.0s step_timeout, alternating seats): v9 **8W-6T-6L = 0.550 wr, Elo delta +7** — *exactly on the 55% defense gate*. Turn times identical between versions (p50=37ms, p95=69ms for both), confirming compute parity. Note: the exp3 edge is much smaller here (+7 Elo) than in the A/B test (+52.5 Elo at default weights) — TuRBO-tuned weights appear to capture most of what exp3 gained at defaults, leaving a small residual. Both gates pass at point-estimate criterion → shipping v9. SE on 20 games at p=0.5 is ~0.112 so the true WR could plausibly sit in [0.33, 0.77]; if v9 doesn't beat v8 on the ladder within 24h we revert to v8 as the floor. **Shipping v9 = v8 TuRBO-v2 weights + `sim_move_variant="exp3"`.** **v9 SHIPPED, EARLY LADDER SIGNAL IS NEGATIVE.** Submitted at 18:17 UTC (kernel version 1 → submission "COMPLETE" within ~5 min at initial 600.0). Over the first ~20 min of ladder replay v9 wobbled 600.0 → 511.3 → 615.2, meanwhile v8 *rose* 752.9 → 769.5 on the same replay window. **v9 currently 615.2 vs v8 769.5: -154 Elo vs v8.** Final settle not yet known (v8 took several hours to settle). Likely mechanism: exp3 adds exploration noise that helped at DEFAULT weights (+52.5 Elo A/B) but **duplicates the exploration that TuRBO-tuned weights already encode**, so the marginal H2H gain (+7 Elo, wr=0.550 on exactly-55% gate) is noise around 0, not signal. Plan: **keep v8 as defended floor (769.5)**; let v9 fully settle overnight; if v9 stays below v8 by >50 Elo, revert next submission to weights-only-no-variant and redirect W4 effort to BC-warmstart NN prior + more TuRBO trials rather than sim-move bandit variants. The 0.55 point-estimate gate was too loose for a marginal change — future gates should require wr ≥ 0.60 over 20 games OR wr ≥ 0.55 over 60+ games to have enough power. **`.venv-gpu` online with CUDA 12.4 torch + RTX 3070**, BC-warmstart training on GPU. **EvoTune fitness bridge + runner CLI shipped** with 6 passing tests; mock-LLM smoke run end-to-end. **Overage-bank opening-turn deadline lift shipped + empirically validated** — all 3 live-game gates PASS. **Path D W4 audit: archetype-overrides-in-FastRolloutAgent + 0.99 posterior freeze already shipped** — no action needed. **v10 experiment = NEGATIVE RESULT, margin-lock is compute-bound, not margin-bound.** Built `v10_fast_m0p5_noopp.py` (232,896 bytes) with `rollout_policy='fast'` (13× more sims: 27 vs 2 at 300ms) + `anchor_improvement_margin=0.5` (4× looser override threshold) + `use_opponent_model=False`. Required extending `tools/bundle.py` with `--rollout-policy`, `--anchor-margin`, `--use-opponent-model` flags (+8 new tests, 16 bundle tests total pass in 0.57s). v10 vs v8 H2H (20 games, same seeds as v9 H2H): **8W-6T-6L = 0.550 wr, Elo +7 — game-by-game outcome sequence is BYTE-IDENTICAL to v9 vs v8.** Even with 13× more rollouts + looser margin, MCTS cannot build a Q-gap exceeding the anchor on enough turns to change wire actions. H3 margin-lock hypothesis is now **compute-bound, not margin-bound** — the fix is a learned prior (NN), not a threshold tweak. Do NOT ship v10. **BC-warmstart + PPO is now the highest-EV W4 lever** (TuRBO-v3 60-trial run is launched in background but the heuristic ceiling is anchor-locked, not weight-shaped).)*

**Kaggle submission trail** (public score, most recent first; full settle trajectory in brackets):
- **v11** (TuRBO-v3 weights + `sim_move_variant='exp3'`, eta=0.3): **880.8 (full overnight settle, floor confirmed; peak 926.0 at 23:48)** — **+30 Elo over v9 (850.7), +90 over v8 (790.7). CURRENT LADDER LEADER.** [submitted 2026-04-24 22:53 UTC → initial 600.0 → early dip 562.2 → climb 748.2 → 829.6 → 894.4 → **924.0 (23:23) → 926.0 (23:48, PEAK)** → settle band 875-912, closing 875.5 at 00:49 UTC. 2-hour trajectory recorded in `runs/v11_ladder_poll_20260424.log`; overnight settle continuing in `runs/v11_ladder_poll_part2_20260425.log`. Notable: v9 dropped 864.3 → 843.1 (-21) during the same window, i.e. the ladder is net-transferring Elo from v9 to v11 — confirms they ARE playing each other and H2H direction holds on the ladder. **Delta is MUCH smaller than the 85% H2H predicted (+32 Elo actual vs ~+200 Elo from Elo-from-wr formula)** — archetype-pool overfitting is real but not catastrophic; v3 weights generalize positively. Local H2H vs v9 at same seeds: v11 **17W-3L = 85% wr Elo+151** (`runs/v11_vs_v9_bundled_h2h.log`). Turn times: v11 p50=33ms / p95=59ms, v9 p50=41ms / p95=85ms — v11 is FASTER too.]
- **v9** (v8 weights + `sim_move_variant='exp3'`, eta=0.3): 864.3 (latest, still shifting; peaked 981.9 at 21:26) [submitted 2026-04-24 18:17 UTC → initial COMPLETE 600.0 → early-ladder dip 511.3/615.2 at ~20min (MISLEADING, -154 vs v8 early) → settle trajectory (1-hour poll, `runs/v9_ladder_poll_20260424.log`): 702.3 (20:41) → 849.6 (20:46, +147 in 5min) → 873.6 (20:56) → 855.1 (21:06) → 920.3 (21:11) → 907.9 (21:16) → 957.5 (21:21) → **981.9 (21:26, PEAK, +212 Elo)** → 956.8 (21:31) → 934.4 (21:36) → 864.3 (22:53, -70 from peak over 90min). Ladder oscillation persists well past the first hour; the "234.4 floor" claim was premature — true settle needs >6h. Still above v8 (+73 Elo), still the operational floor until v11 settles]
- v8 (TuRBO-v2 tuned HEURISTIC_WEIGHTS, 18 keys, trial-22 `win=0.800`): **769.5** [initial COMPLETE 600.0 at 2026-04-24 14:51 UTC → same-day settle 762.3 → stable at 769.5; **+113 Elo over v7** at the time of settle, now -121 Elo under v9]
- v7 (entropy-leak fix in FastEngine._maybe_spawn_comets): **649.1 (day-2 settle)** [initial 708.2 → day-1 settle 599.0 → day-2 630.1 → post-v8-replay 649.1]
- v6 (seat-1 anchor-lock off-by-one fix): 594.1 [initial 667.4 → day-1 settle 623.3 → day-2 settle 594.1]
- v5 (seat-1 step inference + comet path bookkeeping + zero-miss gate): 523.0
- v4 (outer_hard_stop_at threading): 475.2 (regression)
- v3 (decoupled UCT + Bayes + fast-archetype rollouts): 535.0
- v2.1 (hardened per-ply deadlines): 539.1
- v1 (W3 first real: MCTS v2 Gumbel SH + Bayes + margin=2.0 floor): 455.1

**v7 day-2 settle: +36 Elo over v6. Noise-hypothesis (H1) confirmed, exploration-hypothesis (H2) falsified.**
Day 1 showed v7 < v6 by 24 Elo, which prompted a three-hypothesis analysis. Day 2
ratings flipped: v7 gained +31 overnight while v6 lost 29, putting v7 ahead by 36.
The magnitude of the flip (60-Elo swing in 24h) is classic small-N variance in
a thin-sample ladder phase. Walking the three hypotheses forward:

1. **H1 (small-N noise) — CONFIRMED.** A 60-Elo one-day swing is only explicable
   by limited game counts at this rating level. Each bot has played an estimated
   15-30 games across the two-day window, well below the ~200 needed for
   <20 Elo CI. The day-1 anomaly was the noise; day-2 settlement is the signal.
2. **H2 (lost accidental exploration) — FALSIFIED.** If v6's entropy leak were
   doing real work disrupting over-fit opponents, v6 would HOLD the lead after
   more games. It didn't — v6 collapsed -29 while v7 rose +31. The leak wasn't
   helping.
3. **H3 (MCTS-collapses-to-heur trap) — STILL OPEN, orthogonal to v6-vs-v7.**
   At margin=2.0 anchor-lock MCTS wire-actions are byte-identical to heur in
   >95% of turns. H3 isn't refuted; it just doesn't explain v6-vs-v7. It IS
   the reason neither v6 nor v7 is above ~630 Elo — W4 NN priors are the fix.

**Decision: defend v8 as the new floor (762.3, +113 over v7).** v8 settled
well above v7 on the public ladder — first confirmed Kaggle win in the v5→v8
lineage where local H2H (90%, Elo +175.8) predicted the ladder direction
correctly. Next submission must beat v8 locally (≥55% over 200 games H2H)
before shipping. W4 NN work (BC-warmstart + PPO-from-heuristic, Exp3 A/B at
sim-move nodes) remains the path forward; the tuned heuristic floor is no
longer the bottleneck.

The **v6 → v7** fix mechanism: `FastEngine._maybe_spawn_comets`
calls `kaggle_environments.envs.orbit_wars.generate_comet_paths`, which internally
calls `random.uniform` up to ~900 times (per-path retry loop bounded at 300 iters).
In isolation mode (`self._rng is not random`, i.e. all MCTS rollouts) that global
consumption LEAKS entropy from the Kaggle judge's stream — the same stream used
by the real env for *its* comet spawns. Measurement (`tools/diag_who_touches_global_random.py`):
heur-vs-heur at seed=123 made 3166 global random calls; MCTS-vs-heur (pre-fix)
made **28888** — a 9x leak, all attributable to `orbit_wars.py:{233,234,242}`
inside `generate_comet_paths`. Fix: snapshot/restore `random.getstate()` around
the call in isolation mode only (parity validator still explicitly shares global
state via `rng=random`). Multi-seed smoke went **3W/3L cum=-12540 → 3W/3L cum=0**
(perfect mirror symmetry — MCTS is now byte-identical to heur across all 6
seeds x seats). All paired games differ only by the map's inherent per-seed
seat advantage.

The v5 → v6 delta (+9.7) validates the `_turn_counter` ordering fix: moving the
fresh-game detection + `self._fallback` replacement BEFORE the first
`self._fallback.act()` call eliminates a 1-turn off-by-one in the seat-1 step
inference path (obs.step is None at seat 1). Pre-fix diagnostic showed 3/30
wire-action divergences at seat 1; post-fix is 0/30.

### TuRBO-v3 complete: trial 42 = 45/45 wr=1.000 across all 9 archetypes

Launched in parallel with v9 settling (`runs/turbo_v3_20260424.{json,log}`, 60
trials, Ax AxClient + BoTorch, seed=1, w2 opp pool, 5 games/opp × 9 opps = 45
games/trial). Ran ~4h total wall; the Sobol init phase (trials 0-11 at
~180-225s/trial) transitioned to BoTorch at trial 12, which dropped per-trial
wall to ~45-55s as the trust region narrowed. Best-so-far:
  trial  0 → 0.444  (Sobol init)
  trial  2 → 0.511
  trial 10 → 0.644
  trial 42 → **1.000** (45/45, perfect against EACH of 9 archetypes: starter,
              random, rusher, turtler, economy, harasser, comet_camper,
              opportunist, defender)
  trials 43-59 → 0.756-0.978 (trust-region follow-ups, stable optimum)

Best weights (trial 42) vs v8 (TuRBO-v2 trial 22):
  * `w_production`: 19.44 → **20.0** (maxed at bound)
  * `mult_comet`: 1.70 → **5.0** (maxed)
  * `mult_enemy`: 2.26 → **5.0** (maxed)
  * `mult_neutral`: 1.37 → 2.0
  * `mult_reinforce_ally`: 0.29 → **0.0** (disables reinforcement)
  * `w_ships_cost`: 0.14 → **0.0**
  * `w_distance_cost`: 0.10 → **0.0**
  * `keep_reserve_ships`: 12.5 → **0.0** (disables reserve)
  * `min_launch_size`: 29.1 → **30.0** (maxed)
  * `max_launch_fraction`: 0.78 → 0.99
  * `comet_max_time_mismatch`: → **5.0** (maxed)
  * `agg_early_game`: 0.71 → 0.50
  * `ships_safety_margin`: 3.35 → 0.99
  * Others: similar or unchanged.

This is a qualitatively different strategy: v2 balanced production/distance
costs with a small reinforcement bias; v3 is pure mass-attack with disabled
reinforce/reserve and max weights on comet/enemy captures. **Interpretation**:
at MCTS margin=2.0 (anchor-locks to heuristic), the heuristic BECOMES the
bot. TuRBO-v3 found a more aggressive heuristic that wins 100% on our opp
pool. Multiple weights at search-bound edges (20.0, 5.0, 30.0, 0.0) means the
TRUE optimum may be outside current bounds — follow-on TuRBO-v4 with wider
bounds is a reasonable next step if v11 H2H clears the gate.

**v11 bundle built**: `submissions/v11.py` (232,710 bytes) = TuRBO-v3 weights
+ `sim_move_variant=exp3` + eta=0.3. Smoke test vs `random` WIN: `rewards=[1,
-1] steps=99` — v11 finishes games much faster than v8 (steps=114), consistent
with the all-in profile. **v11 vs v9 H2H (20 games, 1.0s step_timeout) running
in background.** Gate: wr ≥ 0.55 over 20 games vs v9 (the current floor at
934.4). With the weights this different from v9's, ≥0.60 or a 60+ game run
would be ideal, but 0.55@20 is the shipped gate.

### v10 experimental bundle — negative result: margin-lock holds even with 13× sims

To stress-test the **margin-lock hypothesis** (H3: MCTS wire-actions collapse to
the heuristic anchor at `anchor_improvement_margin=2.0`) we built `v10_fast_m0p5_noopp.py`
(232,896 bytes) by extending `tools/bundle.py` with three new CLI flags:
`--rollout-policy {heuristic,fast}`, `--anchor-margin FLOAT`, and
`--use-opponent-model {true,false}`. 8 new tests in `tests/test_bundle.py`
(16 total, all pass in 0.57s) cover injection, factory-rewriting, bot-type
validation, reject-unknown, and the "use_opp_model alone skips dead _bundle_cfg"
edge case.

v10 = v8 TuRBO-v2 weights + `rollout_policy='fast'` (FastRolloutAgent, gives
~13× more sims: 27 vs 2 at 300 ms budget) + `anchor_improvement_margin=0.5`
(lowered from 2.0 to let a small Q-gap override the heuristic) +
`use_opponent_model=False` (skips the ~27 ms/turn Bayesian posterior update).

**v10 vs v8 bundled H2H (20 games, alternating seats, same seed pool as v9 H2H):
v10 8W-6T-6L = 0.550 wr, Elo +7.** The game-by-game outcome sequence is
**byte-identical to v9 vs v8 on those same seeds** (g0=W, g1=T, g2=W, g3=T,
g4=T, g5=W, g6=T, g7=W, g8=T, g9=L, g10=L, g11=W, g12=W, g13=L, g14=L,
g15=W, g16=L, g17=T, g18=W, g19=L). v10 turn times p50=52ms / p95=83ms vs v8
p50=56ms / p95=100ms — MCTS is spending a similar amount of compute, but NOT
using it to wire different actions.

**Interpretation: this is the strongest margin-lock evidence yet.** If MCTS
were actually overriding the heuristic more often at margin=0.5 than at
margin=2.0, v10 would diverge from v9 (which kept margin=2.0) on at least a
few seeds. It doesn't. That means even with 13× more rollouts *and* the
override threshold lowered 4×, MCTS still cannot produce a Q-gap that exceeds
anchor-margin on enough turns to change wire actions. **Conclusion: H3 is
compute-bound, not margin-bound.** The fix is better Q-estimates (NN prior or
bigger sim budget), not a looser anchor threshold. Do NOT ship v10 — it would
reproduce v9's likely ladder regression.

**Caveat (revised after v9 ladder peaked at +212 Elo)**: the v10 experiment
proves the margin-lock at these *specific H2H seeds vs v8* — but says nothing
about broader ladder dynamics. v9's ladder trajectory (+165 Elo settled,
+212 Elo peak vs v8 on public leaderboard) shows that exp3 *does* produce
different wire actions against ladder-scale heterogeneous opponents, just
not against the v8 bundle in our specific seed-pool H2H. Plausible
mechanism: against unknown opponents the Bayesian posterior stays diffuse
→ exp3's opp marginalization produces different Q-values than ucb →
margin-lock breaks more often → MCTS genuinely overrides heuristic. v10
disabled opp_model AND switched to ucb, so it lost this effect and went
back to heuristic-clone behavior. The H3 margin-bound-vs-compute-bound
framing is right AT THE LOCAL LEVEL (can't break with knobs alone), but
opens a NEW axis at the LADDER LEVEL: Bayes + exp3 does break margin-lock
on diffuse-prior states, just not on pinned-prior (vs known opp) states.

**Implication for W4 path**: BC-warmstart + PPO NN prior is still the
highest-EV lever — a learned policy gives better Q-estimates across ALL
states, not just diffuse-prior ones. But v9's ladder win also validates
the Bayes+exp3 axis: dumping opp_model like v10 did is a regression, not
an optimization. TuRBO-v3 (60 trials, launched in background) can squeeze
more out of the heuristic; the current ceiling is the cross-product of
{learned prior, Bayes+exp3 decoupled root}.

This is the project map. If you just cloned the repo, read this first.

---

## 1. The plan in one paragraph

Kaggle "Orbit Wars" is a 2/4-player real-time strategy game under a **1 s/turn CPU budget**. Our bet is that **heuristic-backed search beats blind RL** in this regime — the precedent (Halite III, Kore 2022, Lux S1-S3) is consistent. The plan has four tracks:

- **Path A**: a parameterized heuristic bot + TuRBO-tuned weights + LLM-evolved priority components (primary novelty).
- **Path B**: Gumbel MCTS with kernel-aggregated continuous angles over heuristic rollouts.
- **Path C**: self-play PPO + PFSP → distilled prior for Path B (optional; writeup-only if it doesn't ship).
- **Path D**: online Bayesian opponent modeling over an archetype portfolio (secondary novelty).

The full plan lives at `C:\Users\Marc\.claude\plans\read-the-kaggle-competition-transient-canyon.md`. First real Kaggle submission targeted at **Week 3-4**; Weeks 1-2 are infrastructure + Paths A & B v1.

---

## 2. Directory layout

```
orbit-wars/
  src/orbitwars/
    engine/             # W1: our fast numpy engine, intercept math, parity validator
    bots/               # bot implementations (Agent contract)
    mcts/               # action generation, Gumbel root search, rollouts
    tune/               # TuRBO, EvoTune, fitness harness, LLM client
    features/           # (stub) obs → tensors for future NN
    opponent/           # (stub) archetype portfolio + Bayesian posterior (W3)
    nn/                 # (stub) conv policy + set-transformer (W4)
  tournaments/          # round-robin harness, seeded games, Elo
  tools/                # bundle.py, profile.py, smoke_* runners, trace_mcts_turn.py
  tests/                # pytest: 90 passing, 1 skipped
  submissions/          # versioned submission.py files + README on Kaggle submit flow
  writeup/              # (empty — populated in W6)
  training/             # (empty — PPO configs live here in W4-5)
  notebooks/            # scratch analysis
  runs/                 # gitignored — tuning output, tournament artifacts
```

Total source: ~3k lines of real code, ~1.6k lines of tests.

---

## 3. Module-by-module

### `orbitwars.engine.fast_engine` (891 lines)
Numpy Structure-of-Arrays re-implementation of `kaggle_environments.envs.orbit_wars`. Planets & fleets are parallel numpy arrays (int64 ships, float64 positions) so the three hot loops — fleet movement + collision, planet rotation + sweep, comet movement + sweep — are vectorized. Comet groups stay list-of-dicts (branchy, not hot).

**Two construction paths:**
- `from_scratch(num_agents, seed)` — for scenarios: delegates to reference `generate_planets()` so we share its RNG stream and produce identical initial maps. Uses global `random` (intentional — the reference does).
- `from_official_obs(obs, rng=None)` — for MCTS rollouts and validation: snapshots a running env's state. **`rng` defaults to a per-instance `random.Random()` so rollouts don't pollute global random state.** (This was a critical bug fix — see §6.)

**Parity gate**: `orbitwars.engine.validate` runs us step-by-step vs the reference over N seeds with synchronized global random state, diffing planets, fleets, step, next_fleet_id. **5/5 seeds pass × 100 turns currently.** Run the full W1 gate with `python -m orbitwars.engine.validate --seeds 1000 --turns 500`.

### `orbitwars.engine.intercept` (316 lines)
Analytic intercept solvers: closed-form for static targets; Newton iterations for orbiting planets and comet paths. Symplectic velocity-Verlet for any gravity integration (sun-influenced flights, if we add them later — the reference engine doesn't).

### `orbitwars.bots.base` (159 lines)
The `Agent` contract and `Deadline` scaffolding. Key constants:
- `HARD_DEADLINE_MS = 900` — absolute return-by wall time per turn.
- `SEARCH_DEADLINE_MS = 850` — MCTS internal ceiling.
- `EARLY_FALLBACK_MS = 200` — must have a valid action staged by this point.

Every agent must stage a valid action within 200 ms and be willing to return it if anything later exceeds budget. `gc.disable()` at import; `gc.collect()` between turns in `as_kaggle_agent`. Exception-safe wrapper converts any raise into the staged fallback.

### `orbitwars.bots.heuristic` (487 lines)
Path A bot — the FLOOR. For each owned planet scores candidate targets by `(production × capture_likelihood) / (ships_needed + travel_turns × w)`; launches exact-plus-one; defensive reallocation when net-incoming flips negative; sun-tangent routing; comet positioning at `{50, 150, 250, 350, 450}`; fleet-arrival table (Kore 2022 insight). **~20 weights exposed as `HEURISTIC_WEIGHTS` dict → consumed by TuRBO and EvoTune.**

### `orbitwars.bots.mcts_bot` (148 lines)
Path B bot. Flow per turn:
1. Stage `no_op` immediately.
2. Compute heuristic action (the anchor) and stage it.
3. On step==0, refresh fallback HeuristicAgent + GumbelRootSearch (fresh match state).
4. If outer Deadline budget is tight, return the staged heuristic.
5. Else tighten the GumbelConfig's `hard_deadline_ms` to fit the remaining budget — **including carrying `anchor_improvement_margin` forward** (a bug fixed in W2; see §6).
6. Call `self._search.search(obs, my_player, anchor_action=heuristic_move)`. Return search's wire action on success, or the heuristic on any exception.

### `orbitwars.mcts.actions` (263 lines)
Generates the candidate action set per turn. Per owned planet: analytic intercept angles to reachable targets (planets, comets, enemy fleets), sun-tangent angles ±ε when direct path is blocked, ship-count fractions `{25%, 50%, 100%}`. Joint actions are built by sampling per-planet decisions — not a full Cartesian product, which would blow up on 20+ planets.

### `orbitwars.mcts.gumbel_search` (569 lines)
The heart of Path B. Implements:
- **Gumbel top-k without replacement** at the root.
- **Sequential Halving** (Danihelka et al., ICLR 2022) for simple-regret-optimal budget allocation.
- **Anchor-joint protection**: the heuristic's move is prepended as candidate 0 and marked with `protected_idx` so SH can't prune it across rounds.
- **Anchor improvement margin guard**: post-SH, only override the anchor if `winner_q - anchor_q ≥ anchor_improvement_margin`. This gates out rollout noise.
- **Rollouts**: `depth`-plies heuristic-vs-heuristic simulation in a deep-copied FastEngine. Value = normalized ship-lead in [-1, +1], or terminal reward if game ended.
- **Per-rollout RNG**: the search's `random.Random` is forwarded into each rollout engine, so rollouts are **reproducible given the search seed** (another bug fixed in W2; see §6).

Default config (post-W2 tuning, post-multi-seed):
```python
num_candidates = 4
total_sims = 32
rollout_depth = 15
hard_deadline_ms = 300.0
anchor_improvement_margin = 2.0   # effectively always-defer-to-anchor
```

**Why margin=2.0** (effectively "always play the heuristic"): the single-seed=42 smoke showed MCTS beating heuristic at margin=0.5 (1415-787), but the multi-seed smoke (seeds {42,123,7} × both seats, 6 matches) at the same config was **2W/4L cum_score_diff=-7174**. Root cause: under wall-clock pressure some turns hit `HARD_DEADLINE_MS` and return the staged heuristic while others use search output; those branching decisions cascade into materially different games, and at low sim budgets the search output is worse than the heuristic more often than it's better. Until W4-5 gives us a proper neural prior (or the overage-bank discipline lets us afford many more sims), margin=2.0 locks in the Path A floor. Search still runs and exposes statistics in `SearchResult` for diagnostics.

### `orbitwars.opponent.archetypes` (175 lines, W3)
Frozen archetype portfolio: 7 stylistic variants of `HeuristicAgent` built by parameter override. Each is a plausible strategy a real competitor might ship — `rusher, turtler, economy, harasser, comet_camper, opportunist, defender`. Doubles as (a) the support set for the Bayesian posterior, (b) permanent opponent pool for PFSP in W4-5. Module-import time asserts every override key is in `HEURISTIC_WEIGHTS` (a silently-ignored typo would make the archetype identical to default).

### `orbitwars.opponent.bayes` (245 lines, W3)
Online Bayesian posterior over archetypes (Path D, secondary novelty). Per turn:
- Diff fleet ids against the previous obs to extract the opponent's new launches.
- For each archetype, simulate what it would launch from the pre-move state (using `_fabricate_opp_obs` that flips only the `player` field — orbit_wars is fully observable).
- Compare per-planet launch-vs-hold decisions; per-planet Bernoulli likelihood with noise floor `eps=0.1`; accumulate in log-space.
- `distribution()` returns softmax over log-alpha. `most_likely()` for the argmax.

Unit tests include a concentration test: after 40 turns of watching an opponent play archetype-X, the posterior's P(X) exceeds the uniform 1/7 baseline for all tested archetypes. Per-turn cost <50 ms on dense mid-game obs — fits the MCTS sim budget.

**Wired into `MCTSAgent`**: observation every turn (defensive try/except); posterior-driven `opp_policy_override` on the search when concentration exceeds thresholds (turns_observed ≥ 15 AND max probability > 0.35). When the override fires, MCTS rollouts use the inferred archetype's heuristic as the opponent policy instead of the generic default.

### `orbitwars.tune.fitness` (155 lines)
Tournament harness-as-fitness-function for parameter search. Takes a `weights` dict, plays N games vs an opponent pool with alternating seats, returns win-rate (or score margin). Used by both TuRBO and EvoTune.

### `orbitwars.tune.turbo_runner` (358 lines)
Ax/BoTorch `qNoisyExpectedImprovement` + trust-region. Tightened `PARAM_BOUNDS` (lines 59-78) from the original pathological ranges. Handles Sobol bootstrap (8 trials) + BoTorch optimization (remaining trials). Logs per-trial win-rate to `runs/turbo_*.json`.

### `orbitwars.tune.evotune` (345 lines)
LLM-driven evolution of `evaluate(state) -> float` priority functions. LLM proposes candidates → tournament scores them → top-k seed the next generation. Parser hardening (line 79 onward) rejects unsafe Python (no imports, no eval, no network).

### `orbitwars.tune.llm_client` (204 lines)
Minimal Anthropic SDK wrapper; used by EvoTune. Swap via env var.

### `tournaments.harness` (312 lines)
`play_game(agents, seed, players, step_timeout) -> GameResult`. Round-robin driver built on top for ladders. Catches agent exceptions (returns `[]`). Uses `_pyr.seed(seed)` to set the reference engine's map RNG. Records `turn_times_ms` per player. **Critical**: this is the canonical way to score any two bots.

### `tools/`
- `bundle.py` — single-file submission packager (imports + flattens `src/orbitwars/...` into one `submission.py`).
- `profile.py` — microbenchmarks per hot path.
- `smoke_*.py` — scripted "play one game and print result" runners, used during W2 debugging (see §6).
- `trace_mcts_turn.py` — single-turn deterministic trace for verifying `act()` output matches heuristic under forced-anchor config.

---

## 4. Entry points

```bash
# 1. Run the full test suite (should be 90 passed, 1 skipped)
$env:PYTHONPATH="src;."; .\.venv\Scripts\python.exe -m pytest tests/ -q

# 2. W1 parity gate
$env:PYTHONPATH="src;."; .\.venv\Scripts\python.exe -m orbitwars.engine.validate --seeds 1000 --turns 500

# 3. Default MCTS vs heuristic smoke (~3 min, expected: MCTS WIN)
$env:PYTHONPATH="src;."; .\.venv\Scripts\python.exe tools/smoke_mcts_default_vs_heuristic.py

# 4. Multi-seed mini-ladder
$env:PYTHONPATH="src;."; .\.venv\Scripts\python.exe tools/smoke_mcts_multi_seed.py

# 5. Bundle a submission
python tools/bundle.py --bot mcts_bot --out submissions/v1.py

# 6. TuRBO tuning run
python -m orbitwars.tune.turbo_runner --trials 25 --games 20
```

---

## 5. What's done (W1-W2 so far)

### W1 infrastructure ✅
1. **Fast numpy engine** (`fast_engine.py`) — parity with reference engine; 5/5 seeds pass × 100 turns in our smoke; 1000-seed full gate pending.
2. **Intercept solver** (`intercept.py`) — analytic + Newton solvers for moving targets.
3. **Heuristic v1** (`heuristic.py`) — parameterized weights, fleet-arrival table, sun-tangent routing, comet positioning.
4. **Tournament harness** (`tournaments/harness.py`) — seeded games, turn-time tracking, Elo wiring.
5. **Bundle tool** (`tools/bundle.py`) — single-file submission packager.

### W2 Paths A + B ✅
6. **TuRBO wiring** (`turbo_runner.py`) — Ax/BoTorch driver with tightened param bounds; first run completed 25 trials.
7. **EvoTune skeleton** (`evotune.py`) — LLM-evolved priority functions; parser hardening.
8. **MCTS action generator** (`actions.py`) — per-planet candidate sets with analytic intercepts.
9. **Gumbel MCTS v1** (`gumbel_search.py`) — top-k + Sequential Halving + anchor protection.
10. **MCTSAgent** (`mcts_bot.py`) — Path B bot integrating everything.

### W2 critical bug fixes 🔧
The rest of W2 was spent hunting a mystery: MCTSAgent lost 16-3323 to the heuristic despite the anchor-joint floor. Four root causes found, all fixed in sequence:

- **11. Anchor-joint floor** — prepend heuristic's move as candidate 0; `protected_idx` keeps it alive across SH rounds; margin guard (`winner_q - anchor_q ≥ margin`) prevents rollout-noise overrides.
- **12. Global RNG pollution (partial fix in W2)** — `FastEngine._maybe_spawn_comets` called `random.randint(1, 99)` on the *global* `random` module. Fix: instance-local `random.Random()` per FastEngine; validator opts into sharing via `rng=random` kwarg. **NOTE: this only closed HALF the leak — the OTHER half was `generate_comet_paths` itself, caught in v7; see §5 item 23.**
- **13. tight_cfg strip** — `mcts_bot.act` rebuilds GumbelConfig to tighten `hard_deadline_ms`, but the rebuild dropped `anchor_improvement_margin`, silently reverting to the default (0.15) every turn. Fix: copy the field explicitly. Regression test added.
- **14. Non-deterministic rollouts** — After the RNG-isolation fix, each FastEngine rollout used an os.urandom-seeded RNG → different runs of MCTSAgent produced different outcomes on the same seed. Fix: thread `self._rng` from the search into each rollout's FastEngine.

### W3 Path D shipped ✅ (2026-04-22)
- **15. Archetype portfolio** — 7 archetypes in `orbitwars.opponent.archetypes` built on `HeuristicAgent` with parameter overrides. Module-level assertion catches typos in override keys.
- **16. Bayesian posterior** — `ArchetypePosterior` in `orbitwars.opponent.bayes`; log-space per-planet Bernoulli updates; concentration test passes for rusher/turtler/defender after 40 turns.
- **17. MCTSAgent observation wiring** — posterior observed every turn; defensive try/except; opt-out via `use_opponent_model=False`.
- **18. MCTSAgent search-override wiring** — `GumbelRootSearch.opp_policy_override` plumbed; when posterior concentrates (15+ turns AND top prob > 0.35), MCTS rollouts use the inferred archetype's heuristic as opponent policy. Integration test in `test_gumbel_search.py` confirms the factory is actually called. **Currently latent** in wire action because `anchor_improvement_margin=2.0` keeps the heuristic floor locked — the override shapes search Q-values and diagnostics but not the returned action until we relax that margin (W4-5 with neural priors, or manual override for exploitation experiments).
- **19. Posterior telemetry** — MCTSAgent exposes `self.telemetry` dict with `{turns_observed, override_fires, override_clears, last_top_name, last_top_prob}`. Resets per match on turn 0. Two unit tests lock in both transition (fires/clears) and reset behavior. This is what self-diagnoses exploitation smokes — null deltas with `games_with_override=0` mean "posterior never concentrated", which is a different pathology than "posterior concentrated but wrong archetype" or "right archetype but rollouts didn't exploit".
- **20. Empirical posterior concentration** — `tools/diag_posterior_concentration.py` measures peak probability across rusher/turtler/defender × 3 seeds × 80-turn matches. **All 9/9 runs correctly identify the archetype with peak probability 0.90–1.00**; first crossing 0.35 between turns 5–29. Justifies the MIN_TURNS=15, MIN_TOP_PROB=0.35 gate as reachable in practice.
- **21. Exploitation smoke harness + two null results** — `tools/smoke_opp_model_vs_archetypes.py` runs MCTSAgent (model on vs off) × archetypes at a configurable margin. Rolls up telemetry per cell so null results have diagnosis, not just a shrug.

  **Run A — margin=0.5, heuristic rollouts, rusher/turtler/defender × 2 seeds × 2 seats, 24 games, 4382 s wall**:

  | archetype | no_model wr | with_model wr | delta |
  |---|---|---|---|
  | rusher | 0.50 | 0.50 | +0.00 |
  | turtler | 1.00 | 1.00 | +0.00 |
  | defender | 1.00 | 1.00 | +0.00 |

  **Run B — same config but `rollout_policy="fast"` (~13× more sims/turn), 24 games, 4510 s wall**:

  | archetype | no_model wr | with_model wr | delta |
  |---|---|---|---|
  | rusher | 0.50 | 0.50 | +0.00 |
  | turtler | 1.00 | 1.00 | +0.00 |
  | defender | 1.00 | 1.00 | +0.00 |

  Posterior is correct on every single game (142-483 fires/game, p=1.00, correct archetype top-ranked). Game-level score diffs are *mostly identical* across the two rollout policies (typical <1% relative difference) — i.e. the WIRE ACTIONS are converging even at 13× the sim budget. This FALSIFIES hypothesis #1 (sim-budget-starvation). The surviving candidates are #2 (margin=0.5 is still too protective — the margin guard blocks opp-model-driven overrides regardless of sim count) and #3 (turtler/defender hit the 1.00 WR ceiling, so no room for the model to add Elo).

  **Run C — margin=0.0, fast, rusher-only (ceiling-room archetype), 8 games, 597 s wall**:

  | seed | seat | no_model diff | with_model diff | per-game delta |
  |---|---|---|---|---|
  | 42 | 0 | −2581 | −2942 | −361 |
  | 42 | 1 | −2304 | −2930 | −626 |
  | 7 | 0 | +7121 | +7021 | −100 |
  | 7 | 1 | +5327 | +3705 | −1622 |
  | **cum** | | **+7563** | **+4854** | **−2709** |

  All four paired games show `use_model=True` WORSE than `use_model=False` (monotonic, not random). 2W/2L both cells, so the W/L delta is still 0, but score drifts −2709 / 4 games ≈ −677 units/game in the wrong direction — 95% CI is wide with N=4 but the direction is uniform. **This is a weak-negative empirical signal**, not a null.

  **Mechanism diagnosed via `tools/profile_posterior_observe.py`**: `ArchetypePosterior.observe()` costs **27 ms/turn** mean (27.1 ms / 26.9 ms / 28.3 ms / 29.2 ms at mean / median / p95 / p99, N=50 on a mid-game obs). That's **9% of the 300 ms MCTS budget** paid unconditionally every turn under `use_opponent_model=True`. Further, when the override fires, `opp_policy_override` returns `HeuristicAgent(archetype_weights)` — the slow full-featured heuristic (~3.2 ms/`act()`) — which bypasses the "fast" branch of `_opp_factory` and makes opponent rollouts **~400× slower per ply** than under `rollout_policy="fast"` + `use_model=False`. Per-rollout cost under `use_model=True + fast` is therefore ≈ 15 plies × (8 µs my + 3.2 ms opp) ≈ 50 ms/rollout, vs. 11 ms/rollout under `use_model=False + fast`. Net: **`use_model=True` runs with ~5× fewer effective rollouts per turn than `use_model=False`** under the same `rollout_policy="fast"` config. This compounded sim-budget loss fully explains the −2709 drift on Run C without needing "the archetype override is wrong" as an explanation.

  Operational read:
  - Under shipped margin=2.0 the override is INERT (margin blocks the wire action from changing), so there is no ladder-side risk. Posterior still logs correctly for diagnostics. **Safe to keep as shipped.**
  - If/when we lower margin (post-W4 neural priors), the override integration must be fixed: (a) port the archetype's two relevant knobs (`min_launch_size`, `max_launch_fraction`) onto `FastRolloutAgent` so fast-override stays fast, and/or (b) cache / early-exit the posterior once concentration reaches 0.99+ to kill the 27 ms/turn overhead.
  - Good writeup material — the instrumentation works, the posterior concentrates correctly, the negative empirical signal has a concrete and measurable mechanism attached (not "we don't know why it's bad").
- **30. EvoTune fitness bridge + runner CLI (2026-04-24 evening)** — Path A primary-novelty plumbing is now complete end-to-end. Three artifacts landed:
  - `src/orbitwars/tune/evotune_bridge.py` (~185 lines) — bridges the EvoTune sandbox (`compile_scorer(source) -> callable`) into `HeuristicAgent`'s game loop via module-attribute monkey-patch of `heuristic._score_target`. The wrapper reuses the baseline's `_travel_turns` + `_intercept_position` + sun-routing + arrival-table logic; only the *priority score formula* is swapped for the LLM candidate. Also preserves the "can't capture" −10 penalty so under-sized bluff-attacks can't outrank well-sized ones. Install/uninstall API with saved-original pointer; idempotent against double-install.
  - `FitnessConfig.scorer_source: Optional[str]` — new field threaded into `tune.fitness.evaluate()`. When set, workers get installed via `Pool(initializer=_worker_init_evo_scorer, initargs=(source,))` so Windows spawn processes each compile + monkey-patch independently; serial path installs in-process and uninstalls via a `try/finally`.
  - `tools/run_evotune.py` (~280 lines) — CLI driver. Supports `--llm mock` (deterministic scripted sources for plumbing) and `--llm anthropic` (claude-sonnet-4-5-20250929 default). Streaming JSONL log written incrementally so a tail -f shows live candidate scores. Outputs: `runs/evotune_TIMESTAMP.jsonl` + `_best.py` + `_summary.json`.
  - **6 new bridge tests** (`tests/test_evotune_bridge.py`) — install/uninstall identity round-trip, double-install idempotency, constant-42 scorer changes `_score_target`'s return, capture-penalty survives evolved scorer, unsafe source leaves baseline in place, spawn-worker `Pool(initializer=…)` reports `is_installed()` True in every worker. All 6 green in 0.52 s.
  - **End-to-end smoke**: `--llm mock --generations 2 --candidates-per-gen 3 --games 2 --pool starter --workers 1` completes in ~90 s, producing a best candidate (squared-production variation at 1.0 WR vs starter over 2 games — noisy by design, but the plumbing works). Ready for a real `--llm anthropic --generations 30 --candidates-per-gen 8 --games 20 --pool w2 --workers 7` run post-TuRBO.
  - **Test count: 161 → 167** (+6 from evotune_bridge).
- **29. GPU venv + full-size BC warm-start on RTX 3070 (2026-04-24 evening)** — the CUDA unblock for W4-5. Separate `.venv-gpu` created (main `.venv` retains CPU torch for TuRBO/Ax — cleanest separation since torch_cpu.dll was already loaded in the TuRBO parent process). Install dance: `torch==2.6.0+cu124` from the PyTorch CUDA 12.4 index (2.11.0 isn't built for CUDA), `numpy 2.4.4`, project in editable mode, `kaggle-environments --no-deps` (avoids open_spiel Windows build failure). `torch.cuda.is_available()` returns True; `NVIDIA GeForce RTX 3070` detected. `tools/bc_warmstart.py` (~220 lines, shipped today) adds: full-size ConvPolicyCfg (64 backbone ch × 6 ResBlocks ≈ 460K params vs prototype's 45K), demo cache at `runs/bc_demos.npz` with SHA-256 provenance, train/val split with best-val checkpointing, AdamW + CosineAnnealingLR, checkpoint header recording `{model_state, cfg, curve, demo_hash, n_demos, torch_version, cuda_available, seed, hparams}` for W5 PPO audit. Gate: best_val_acc ≥ max(2×random_guess, majority_class + 0.05) — must exceed ~28.6% to pass. Currently running `--games 60 --epochs 15 --batch-size 256 --device cuda` in background (task bb74snoc3).
- **28. TuRBO-v2 parallelized (2026-04-24 afternoon)** — the single most impactful wall-clock win of the project. Serial TuRBO-v2 was running at ~85 min/trial (total ETA 42h for 30 trials); relaunched with `multiprocessing.Pool(processes=7)` — one worker per physical core — dropping trial time to **~203 s/trial** (25× speedup, total ETA ~1.7h). Three load-bearing changes:
  - `OpponentSpec` dataclass in `tune.fitness` replaces the prior closures/lambdas. Each spec is `(name, kind, param)` where kind ∈ {kaggle_builtin, random, noop, archetype}; `make_kaggle_agent()` reconstructs a fresh kaggle-wire callable in the worker. Lambdas aren't pickleable across Windows spawn; dataclasses are.
  - `_run_one_game(weights, spec, opp_name, seed, hero_seat, step_timeout) -> GameRecord` extracted as a top-level worker function (spawn requires it). Seeds both `random` and `numpy.random` in-worker so serial and parallel paths produce byte-identical GameRecord streams for any (weights, seed_base).
  - `FitnessConfig.workers: int = 1` added; when >1, `evaluate()` uses `mp.Pool(...).starmap(_run_one_game, tasks)`; when ==1, no Pool is ever instantiated (regression-tested via a `_Poison` monkey-patch that raises if touched).
  - **2 new parallelism tests** (`tests/test_fitness_parallel.py`): serial vs 2-worker parity test plays identical noop games and asserts byte-identical `GameRecord.reward + steps`; serial-no-pool test monkey-patches `mp.Pool` to a raising object to prove workers=1 never instantiates a pool. **6 pool-spec tests** updated (`tests/test_fitness_pools.py`): shape checks, late-binding lambda safety, pickle round-trip over every spec. All 17/17 tests green including parity. CPU utilization jumped 25% → 55% under load (7-core saturation with TuRBO's BoTorch single-core optimization on the 8th core). — the prior TuRBO run (`runs/turbo_ax_w2_20260422.json`) only completed **2 of 25 trials** before wall-time ran out at `step_timeout=5.0` × `games_per_opp=10` (trial 0: 2.85 h, trial 1: 2.40 h). Earlier STATUS claim "all 25 trials scored 16.7%" was wrong — only 2 trials evaluated, scoring 0.400 and 0.178. **Baseline eval** (`tools/baseline_eval.py`) now records the default `HEURISTIC_WEIGHTS` reference: **0.667 win_rate over 45 games vs w2 pool** (wall 594 s, `step_timeout=1.0`). Per-opp: defender/random/turtler=100%, economy/starter=80%, harasser=60%, rusher=40%, comet_camper/opportunist=20%. **Bar TuRBO-v2 must beat**: 0.667 baseline + 7 pp binomial SE → ≥0.74 win_rate before we ship a weights update. **TuRBO-v2** relaunched with `--step-timeout 1.0 --games-per-opp 5 --n-trials 30 --seed 1 --out runs/turbo_v2_20260424.json`; trial 0 confirmed generated past Ax Sobol init; expected wall ~8-24 h in background.
- **26. NN torch switchover (2026-04-24 afternoon)** — both W4 architecture scaffolds activated. Both previously had the torch module definitions commented out; today they ship live forward passes with 13 new smoke tests pinning shape, dtype, finiteness, value range, param count, determinism, masked-padding isolation, and end-to-end obs_encode → model.forward for both candidates. Key design notes locked in: (a) **conv_policy** uses `GroupNorm` (groups=min(8, C)) in ResBlocks rather than `BatchNorm2d` so batch=1 MCTS leaf inference is statistically identical to training; (b) **set_transformer** `SetTransformerPolicy(cfg, n_features=None)` defaults `n_features` to `len(entity_feature_schema())` so typical callers write `SetTransformerPolicy(SetTransformerCfg())` without schema-drift risk; (c) padding-row features cannot leak into valid-row outputs — new `test_set_transformer_padding_doesnt_leak` perturbs padding features with `randn*100` and asserts valid-row logits + scalar value are byte-identical, the clearest mask-correctness contract for variable-N training later. **Actual param counts** (differ slightly from estimates because the estimator overcounts biases): conv 460,617 / set-t 554,265 — both well under the W5 2M-param distillation gate even before distillation. Torch 2.11.0+cpu confirmed on install; CUDA build deferred to GPU training (RTX 3070 local). **Test count: 148 → 161** (+13 from nn forward). BC prototype `tools/bc_prototype.py` shipped alongside: dumps ~1000 demos/game from HeuristicAgent self-play, trains a tiny 45K-param conv on (grid, gy, gx, action_label) tuples with gather-cross-entropy, reports per-epoch accuracy; label histogram near-uniform (11.5-14% per class, majority-class baseline 14.0%). Uncovered one base-class gotcha during write: `Deadline(1.0)` interprets the float as an *absolute* `perf_counter()` time (not a duration), so every bare `.act(obs, Deadline(1.0))` call fires `should_stop()` immediately and returns `no_op()` — calling `Deadline()` with no argument is correct (defaults to elapsed-time path). Worth remembering before more new tools hit this.
- **25. Feature encoder (`src/orbitwars/features/obs_encode.py`, 2026-04-24)** — the W4 NN candidates need a pure-numpy observation encoder; shipped with 12 regression tests locked in. Two entry points, sharing one obs source:
  - `encode_grid(obs, player_id, cfg=ConvPolicyCfg()) -> (C=12, H=50, W=50) float32`. Channel order matches `conv_policy.feature_channels()` exactly (asserted at runtime). Perspective-normalized so `player_id=0` and `player_id=1` swap the my/enemy channels — single encoder serves both seats.
  - `encode_entities(obs, player_id, cfg=SetTransformerCfg()) -> (features, mask)` with `features: (320, 19) float32` and `mask: (320,) bool`. Schema pinned to `set_transformer.entity_feature_schema()`; padding rows are fully zeroed (tested). Includes velocity decomposition via the game's ship-count speed formula so the model sees `(vx, vy)` directly rather than `(angle, speed)`.
  - Design choices: (a) sqrt-scale ship counts (`sqrt(x)/sqrt(1000)`) — gentler than the game's log-based speed scaling; keeps small fleets distinguishable. (b) No radius-splat — planets rasterize to their single (gy, gx) cell because H=50 on a 100x100 board (2x2 units/cell) already matches planet radius 1-3. (c) Fourier positional encoding lives INSIDE the set-transformer (`_fourier_encode`), not in the encoder, so encoder output is model-agnostic. (d) `turn_phase` broadcasts to every cell of the grid and every valid entity row — models learn the temporal gating from one channel without having to propagate it through attention.
  - Regression tests in `tests/test_obs_encode.py` (12 passed, 45s wall): shape + dtype, no NaN/inf, turn_phase broadcast correctness, perspective swap symmetry on seed=42 mid-game state (p0_me == p1_enemy), one-hot exclusivity for type and owner, `owned_planet_positions()` agrees with `type_planet & owner_me` entity-encoder filter, schema offsets match `entity_feature_schema()` ordering, hardcoded POS_X=7/POS_Y=8 in the set-transformer forward pass match the schema.
  - Bench on mid-game obs (step 100, 24 planets + 2 fleets): grid 12 channels populated 9/12 (comet + neutral channels empty because no comets spawned yet and all neutrals captured), entity tensor 26/320 rows valid. Timing not yet profiled — W4 training will batch these; ~0.5-1 ms/obs on single-core is the rough target for it not to bottleneck PPO.
  - **Test count: 136 → 148** (+12 from obs_encode).
- **24. W4 architecture scaffolds (2026-04-24)** — shipped both candidates for the W4 bake-off as importable skeletons with pinned signatures and estimated param counts:
  - **Candidate A (`src/orbitwars/nn/conv_policy.py`)** — centralized per-entity-grid conv over a (B, 12, 50, 50) feature tensor. 12 channels documented in `feature_channels()` (ship density p0/p1, production p0/p1/neutral, planet_radius, is_orbiting, is_comet, sun_distance, fleet_angle_cos/sin, turn_phase). 6 ResBlocks at 64 channels + policy head + value head. `param_count_estimate(ConvPolicyCfg())` = 503,923. Lux S1/S3 winning architecture; translation-equivariant under 4-fold mirror symmetry (free data augmentation).
  - **Candidate B (`src/orbitwars/nn/set_transformer.py`)** — SAB blocks (plain self-attention, ISAB explicitly dropped per plan — N≤320 makes ISAB slower) over a (B, 320, 19) per-entity tensor. 19 features in `entity_feature_schema()` (identity: is_valid + 3-way type one-hot + 3-way ownership; kinematics: pos_x/y, velocity_x/y, is_orbiting, orbital_angular_vel; capacity: ships, production, radius, sun_distance; globals: turn_phase, score_diff). 4 SAB blocks at d_model=128, 4 heads, learned per-head distance bias (`logits_ij -= scale * euclidean_dist(pos_i, pos_j)`), NeRF-style Fourier coord encoding with 8 bands. `param_count_estimate(SetTransformerCfg())` = 603,187.
  - **Both models emit the same 8-channel per-owned-planet action distribution** (`ACTION_LOOKUP` identical across both files: 4 angle buckets × 2 ship fractions). MCTS integration is therefore architecture-agnostic — the Path C → Path B integration layer can load either without conditional logic.
  - **Torch blocks are commented out in both files** until the feature encoder (`orbitwars.features.obs_encode`) and action decoder land. Torch 2.11.0+cpu IS installed; the scaffolds stay disabled by choice to avoid a half-wired module. Switchover is uncomment + instantiate. *(Update: the encoder landed same-day — see §5 item 25. Switchover completed same-day — see §5 item 26. Both candidates now ship live forward passes with 13 smoke tests.)*
  - Bake-off plan per W4 todo: train each to 1M PPO env steps against the same PFSP opponent pool, ship the higher-Elo winner; the loser becomes an ablation data point in the W6 writeup.
- **23. Full entropy-leak fix (v7, 2026-04-23)** — traced the residual seat-1 catastrophic losses (v6 multi-seed 3W/3L cum=-12540 with seed=123 seat=1 mcts=0) to `FastEngine._maybe_spawn_comets` calling `kaggle_environments.envs.orbit_wars.generate_comet_paths`, which consumes `random.uniform` up to 3x300=900 times per spawn step on the GLOBAL `random` module. Under env.run the judge's comet spawns at steps {50, 150, 250, 350, 450} draw from the same global stream, so every MCTS rollout crossing a spawn boundary desynchronized the real game's future spawns. Measurement (`tools/diag_who_touches_global_random.py`): heur-vs-heur seed=123 consumed 3166 global random calls; MCTS-vs-heur pre-fix consumed 28888 (9x leak). Fix: snapshot/restore `random.getstate()` around the `generate_comet_paths` call when `self._rng is not random` — keeps parity-validator semantics (explicitly passes `rng=random` to share global state). New `tests/test_fast_engine_isolation.py` locks the invariant at every spawn step (6 passed, 2 skipped due to game ending before steps 350/450 on seed=42). **Multi-seed smoke post-fix: 3W/3L cum=0** — perfect mirror symmetry across seeds {42,123,7} × both seats. MCTS at margin=2.0 anchor-lock is now byte-identical to heur in wire actions; remaining score deltas are pure map per-seat advantage (±46 on seed=42, ±3882 on seed=123, ±2185 on seed=7). Bundled as `submissions/mcts_v7.py` (+1581 bytes vs v6), notebook `gardan4/orbit-wars-mcts-v7`, submitted 2026-04-23 15:34 UTC.
- **22. Fast rollout policy** — `orbitwars.bots.fast_rollout.FastRolloutAgent` is a ~50-line nearest-target static-push agent (single `atan2` per launch, no arrival table, no Newton solve). Selectable via `GumbelConfig.rollout_policy={"heuristic", "fast"}`; default stays `"heuristic"` to preserve shipped mcts_v1 behavior. **Measured head-to-head at mid-game obs, depth=15 rollouts**:

| | per `act()` | per rollout | sims @ 300 ms budget |
|---|---|---|---|
| HeuristicAgent (default) | 3.2 ms | 104 ms | **2** |
| FastRolloutAgent | **8 µs** (~380×) | **11 ms** (~9.4×) | **27** |

  This is the fix for item 1 in §7: going from 2 sims/turn to 27 sims/turn at 300 ms budget means SH finally has signal above rollout noise. Expected impact: real policy improvement in MCTS at the same CPU budget; unlocks Path D exploitation signal that the running smoke is currently NOT showing (rusher cell: posterior identifies rusher at p=1.00 for 275+ fires/game, but use_model=True/False both give 2W/2L — consistent with "search was budget-starved" rather than "opponent model doesn't help"). 6 new tests lock in action validity (ownership, ship cap, finite angle) plus the sim-count inequality vs. the heuristic policy at matched budget.

Test count: 117 → 128 (+11 from Path D empirical + fast rollout work) → 136 (+8 from v7 isolation regression tests, 6 passed + 2 skipped) → 148 (+12 from W4-prep obs_encode regression tests) → 161 (+13 from nn forward-pass tests covering both W4 candidates).

### W2 empirical validation 📊
Seed=42, 500 turns, MCTSAgent (defaults) vs HeuristicAgent:

| Config | P0 MCTS Score | P0 Heur Score | Verdict |
|---|---|---|---|
| Pre-fix | 48 | 2965 | MCTS crushed |
| +anchor only | 900 | 1089 | MCTS lost even with forced anchor |
| +RNG isolated, forced anchor | 1675 | 698 | MCTS wins ✓ |
| +real search (margin=0.15) | 692 | 1525 | noise overrides anchor |
| +real search (margin=0.5) | 1356 | 874 | MCTS wins ✓ |
| +deterministic rollouts, defaults (margin=0.5) | 1415 | 787 | MCTS wins (single seed) |
| **multi-seed** (3 seeds × 2 seats, margin=0.5) | — | — | **2W/4L, cum=-7174** ✗ |
| **default (margin=2.0, anchor-locked)** | ~= heuristic | ~= heuristic | **heuristic floor** ✓ |
| **multi-seed re-verification (margin=2.0)** | — | — | **4W/2L, cum=+7373** ✓ |

Reference: heuristic vs heuristic on seed=42 is 2221-252 — P0 has a real map advantage.

Anchor-lock sanity: `tools/diag_mcts_vs_heur_actions.py` confirms **0/30 turns diverge** in wire action between MCTSAgent(defaults) and HeuristicAgent. Search runs, posterior observes, override plumbs — but the emitted action is byte-identical to the heuristic's. Exactly the floor guarantee we wanted.

**Lesson from the multi-seed failure**: single-seed wins are untrustworthy. Wall-clock pressure on the real run (not just the validator) causes branching between "return staged heuristic" and "return search output" on different turns, and those branches cascade. Until we fix the underlying problem (more sims or better priors), locking margin=2.0 gives us the Path A floor with zero downside.

---

## 6. What's pending

### Immediate (this week, W2 tail)
- [x] **Multi-seed MCTS verification** (margin=2.0): 4W/2L, cum_score_diff=+7373 across seeds {42,123,7} × 2 seats. Anchor-lock floor verified; search does not diverge from heuristic. `tools/diag_mcts_vs_heur_actions.py` adds a deterministic 0/30-turn wire-action equality check.
- [ ] **Second Ax tuning run** — first run saw 16.7% win-rate across trials, suggesting starter_agent is too strong a single opponent. `orbitwars.tune.fitness.w2_pool()` now returns `starter + random + 7 archetypes` (9 opponents) — run `python -m orbitwars.tune.turbo_runner --strategy ax --n-trials 25 --pool w2` to rerun with a richer fitness signal. Tests in `tests/test_fitness_pools.py` lock in pool shape + late-binding lambda safety.
- [ ] **W1 GATE: Kaggle 403 on no-op submit** — the Kaggle Simulation track needs rules accepted in the UI + `kaggle kernels push -f notebook.ipynb`; we've been hitting a 403 on CLI submit. Human-gated (rules).

### W3 (this week)
- [x] **Archetype portfolio** — 7 stylistic variants in `orbitwars.opponent.archetypes`.
- [x] **Bayesian posterior** — `ArchetypePosterior` in `orbitwars.opponent.bayes`; concentration-tested.
- [x] **MCTSAgent posterior wiring** — observation + search override (both under `use_opponent_model`).
- [x] **Posterior telemetry + self-diagnosing smoke** — `telemetry` dict on MCTSAgent; `diag_posterior_concentration.py` verifies 0.90-1.00 peak probability empirically (all 9 tested scenarios correctly identify the opponent).
- [x] **Exploitation smoke results (heuristic rollouts)** — 24 games, margin=0.5, 4382 s wall. Result: delta=+0.00 for rusher/turtler/defender.
- [x] **Fast rollout policy** — 9.4× faster rollouts, 27 sims/turn vs 2; `FastRolloutAgent` in `orbitwars.bots.fast_rollout`. See §5 item 22.
- [x] **Re-ran exploitation smoke with `rollout_policy="fast"`** (13× more sims/turn) — *same* null delta. Falsifies sim-budget-starvation. Most individual game scores are within <1% of the heuristic-rollout run — the wire actions are margin-locked, not Q-starved. See §5 item 21.
- [x] **Margin=0.0 + fast + rusher-only diagnostic** — 8 games, 597 s wall. Result: **weak-negative** — all 4 paired games with `use_model=True` scored worse than `use_model=False` (cum. −2709 score units, monotonic direction). W/L unchanged at 2W/2L both cells. Margin was NOT the primary bottleneck. See §5 item 21 "Run C".
- [x] **Profile posterior-observe cost** — `tools/profile_posterior_observe.py` measures 27 ms/turn mean (9% of 300 ms MCTS budget) on a mid-game obs. Combined with slow opp-rollouts under override (HeuristicAgent 3.2 ms vs FastRolloutAgent 8 µs per ply), `use_model=True + rollout_policy=fast` runs with ~5× fewer effective rollouts per turn than `use_model=False + rollout_policy=fast` — fully explains Run C's −2709 drift.
- [ ] **Path D W4 follow-up** — fix the two compounded sim-budget leaks: (a) parameterize `FastRolloutAgent` with `min_launch_size` + `max_launch_fraction` from the inferred archetype so fast-override stays fast; (b) cache/early-exit `ArchetypePosterior.observe()` once concentration ≥ 0.99 to kill the 27 ms/turn unconditional cost. Deferred until W4-5 neural priors make low-margin wire actions viable in the first place; under shipped margin=2.0 the override is inert and the bleeds don't affect Elo.
- [ ] **Decoupled UCT at simultaneous-move nodes**. Meaningless until we have non-root tree expansion; big rewrite.
- [ ] **Regret-matching variant** (flag-gated A/B test — after decoupled UCT).
- [ ] **Time-budget audit + overage-bank** usage (spend ~10 s on opening turn for deep planning).
- [ ] **First real Kaggle submission** — the A+B+D bot. This is the floor we defend.
- [ ] **BOKR kernel sub-action selector** for continuous-angle UCB.

### W4 (GPU work — local RTX 3070 available free, no rental needed)
- [x] **Compute source**: local RTX 3070 (8GB) — 2026-04-24 update. Removes the €100-200 rental constraint from the original plan. 0.5-0.6M param teachers should fit comfortably; 1M-step bake-off est. 1-2 days wall-clock, full 20-100M PPO run est. 4-7 days.
- [x] JAX engine skeleton (`src/orbitwars/engine/jax_engine.py`) — 28-field State, reset/step stubs, 300 lines. Bodies to be filled this week now that compute isn't budget-gated.
- [x] Conv policy scaffold (`src/orbitwars/nn/conv_policy.py`) — ~504K params estimate at default cfg, 12 input channels documented, commented torch module pinned for W4 switchover. **(2026-04-24 afternoon: torch blocks activated; GroupNorm swapped in for BatchNorm2d per Wu & He 2018 recommendation for batch=1 inference; 460,617 actual params.)**
- [x] Set-Transformer scaffold (`src/orbitwars/nn/set_transformer.py`) — ~603K params estimate at default cfg, 19 per-entity features, learned relative-distance attention bias, Fourier-encoded coords, same 8-channel ACTION_LOOKUP as conv (architecture-agnostic decode). **(2026-04-24 afternoon: torch blocks activated; 554,265 actual params; padding-mask correctness regression-tested.)**
- [x] NN forward-pass smoke tests (`tests/test_nn_forward.py`) — 13 tests pin shape, dtype, finiteness, value range, param count, determinism, padding isolation, and end-to-end obs_encode → model.forward for both candidates. All green in 16.87 s.
- [x] BC prototype (`tools/bc_prototype.py`, 2026-04-24) — tiny 45K-param conv trained on gather-cross-entropy of heuristic self-play demos. **Pipeline validated** (29,511 demos from 30 games → tensors → gather at (gy, gx) → softmax loss): epochs 1→6 loss 1.71→1.40, train accuracy **34.2% → 46.2%** (random-guess 12.5%, majority-class 14.3%). Model learns strictly above both floors in 13 minutes CPU — confirms obs_encode + ConvPolicy + gather semantics are correct and the action-label mapping (angle → bucket in {0, π/2, π, 3π/2}; ship fraction → {0.5, 1.0}) captures meaningful signal. Larger-scale BC warm-start on default-size (460K-param) conv with GPU + 500K demos is the W4 deliverable this unblocks.
- [ ] Arch bake-off: centralized per-entity-grid conv vs set-transformer — train 1M steps each, pick winner.
- [ ] Fuller BC warm-start from Path A (GPU, default-size conv, >500K demos).
- [ ] Begin PPO + PFSP run.
- [ ] Install CUDA torch build on the RTX 3070 (`pip install torch --index-url https://download.pytorch.org/whl/cu121`) — currently on `torch 2.11.0+cpu`.

### W5 (GPU)
- [ ] PPO to 20-100M env steps.
- [ ] Distillation to <2M-param student via Bayesian Policy Distillation.
- [ ] Integrate student as Gumbel-AZ prior.
- [ ] **Submission v2** — ships only if it beats v1 ≥55% over 200 games.

### W6 (no GPU)
- [ ] Ablations table: Gumbel on/off, BOKR on/off, opponent model on/off, conv-vs-transformer, student-prior on/off, macro-library on/off.
- [ ] Final tournament + leaderboard trajectory.
- [ ] **Writeup**: `writeup/report.md`.

---

## 7. Known rough edges

1. **MCTS search is still net-NEGATIVE at low sim budgets** — single-seed margin=0.5 looked like a win, multi-seed showed 2W/4L. Under wall-clock pressure, turns randomly branch between "staged heuristic" and "search output" depending on timing, and at low sim budgets search < heuristic more often than the reverse. We ship margin=2.0 (anchor-locked) as the default; search still runs for diagnostics but does not flip the wire action unless a candidate beats the anchor by an unusually clear margin. Fixing this is what W3 (overage-bank for more sims, Path D archetype priors) and W4-5 (neural-net prior) are for.
2. **Ax tuning — prior run wall-time starved; v2 relaunched** — the earlier STATUS claim "all 25 trials scored 16.7%" was factually wrong: prior run (`runs/turbo_ax_w2_20260422.json`) completed only **2 of 25 trials** at `step_timeout=5.0` × `games_per_opp=10` (5.2 h total wall for 2 trials; trial 0 scored 0.400, trial 1 scored 0.178). Default weights against the same w2 pool scored **0.667** (45 games) in today's baseline — so both trial-0 and trial-1 Sobol samples were substantially WORSE than the hand-tuned defaults, explaining the low scores. **v2 running** at `step_timeout=1.0 --games-per-opp 5 --n-trials 30 --seed 1` → `runs/turbo_v2_20260424.json`. To ship a TuRBO-derived weights update, v2 must beat 0.667 baseline + 7 pp binomial SE → ≥0.74 win_rate. Plan fallback if v2 also plateaus under 0.74: seed Ax with the default-weights vector as trial 0 (currently Ax starts with pure Sobol), and/or prune the hardest opponents (comet_camper / opportunist / rusher) from the tuning pool and tune narrowly against the weak half where defaults already crush.
3. **`from_scratch` still uses global random** — intentional (needs reference-matching map), but means test setup affects global state. Call `random.seed(seed)` explicitly when ordering matters.
4. **4-player coverage validated** (2026-04-24) — `tools/smoke_4p.py` ran v7 MCTS in each of 4 seats across 2 seeds = 8 matches, ~20 min wall total. Result: **2 wins, 4 top-2, 2 last-place, 0 timeouts, 0 crashes**. Seed=42 spread: 1st/2nd/3rd/4th across seats 1/3/2/0; seed=123 spread: 1st/2nd/4th/3rd across seats 3/0/2/1 (with seats 1 and 2 scoring 0 — positional wipeouts). Pattern confirms the 4p regime is seat-position-dominated rather than strategy-dominated: MCTS@margin=2.0 plays identically to heur, so ranking is determined by map advantage at each seat. Target (rank ≥ 2 of 4) achieved in 6/8 matches. Not-last in 75%. Safe to ship; no 4p-specific regression beyond what heur alone would exhibit.
5. **Submission flow is human-gated** — the Kaggle Simulation track requires rules acceptance in the browser, not via API. Nothing to fix code-side.

---

## 8. Status summary

Week 2 closed with **MCTSAgent shipped at the heuristic-floor default** (margin=2.0) — the four critical bugs that caused the initial catastrophic losses are fixed, tested, and regression-guarded.

**W3 shipped seven ladder versions** (v1→v7). Each was gated on local wins + zero-timeout soaks. **Day-2 settle: v7 at 630.1, v6 at 594.1 — v7 ahead by +36 Elo** (+175 over v1's 455.1). The day-1 apparent regression (v7 < v6 by 24) reversed overnight with a 60-Elo swing (+31 v7, -29 v6) — classic small-N variance in a thin-sample ladder phase. See §1 for the full three-hypothesis post-mortem. Local semantics are unchanged: v7 is strictly more correct (perfect mirror parity vs v6's -12540 cum). Time-budget audit post-v7: p95=396ms, max=594ms, 0 turns >= 850ms (v4 was p95=423, max=882) — the fix also slightly improved tail latency by removing the retry-loop overhead inside comet spawn.

W3→W4 transition checkpoint (2026-04-24): the infrastructure for the W4 bake-off is now in place. Shipped today (morning): (a) Set-Transformer scaffold as candidate-B, (b) feature encoder `obs_encode.py` serving both candidates with 12 tests + perspective-symmetry proof, (c) 4-player smoke validated (2W/4 top-2/0 crashes across 8 matches — §7 item 4 closed), (d) JAX engine skeleton for offline self-play was shipped yesterday. Shipped today (afternoon): (e) **torch forward passes activated** on both NN candidates with 13 regression tests locking shape/dtype/finiteness/mask-correctness; (f) **baseline eval** recording defaults vs w2 pool at 0.667 win_rate — the number TuRBO-v2 must beat to ship a weights update; (g) **TuRBO-v2 running** in the background at corrected-budget params (`step_timeout=1.0 × games_per_opp=5 × n_trials=30`, expected wall 8-24 h); (h) **BC prototype** wired end-to-end demonstrating the obs_encode → conv_policy gather-cross-entropy loop works on the heuristic's own launch decisions. Remaining for W4: implement `jax_engine.reset()`/`step()` bodies, install the CUDA torch build on the local RTX 3070, run the 1M-step bake-off. BOKR kernel wiring decision stays deferred to W4 (default off; revisit when NN prior lowers mean turn time). **RTX 3070 availability** (disclosed 2026-04-24) removes the €100-200 GPU rental constraint from the original plan — all W4-5 training runs fit on local hardware.
