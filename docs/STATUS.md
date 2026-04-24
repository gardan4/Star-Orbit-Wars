# Orbit Wars — Repo Status & Architecture

*Last updated: 2026-04-24 Day 2 evening (W3→W4 transition; **v7 at 630.1, back ahead of v6 at 594.1 by +36 Elo** after overnight settle; H1 noise-hypothesis confirmed. **NN torch forward passes live**, 13 forward-pass tests green. **TuRBO-v2 complete: 30 trials, best trial 22 win=0.800** (+0.133 over 0.667 defaults baseline, well above the 0.07 ship margin). **compare_weights ran at N=90 games/side, workers=7: turbo_v2 0.833 vs defaults 0.600, delta_wr=+0.233, 95% CI [+0.106, +0.361], paired: turbo_v2 wins on 61.7% of seeds. Worst per-opp regression: rusher -0.100 (exactly at the -0.10 boundary), all other 8 opps positive or tied. `tools/ship_gate.py` VERDICT: SHIP turbo_v2 (both gates PASS).** **v8 bundled: `submissions/v8.py` (229714 bytes) with HEURISTIC_WEIGHTS.update from trial-22 weights; smoke test `rewards=[1,-1] steps=114` vs random.** **v8 vs v7 bundled H2H (20 games, 1.0s timeout, alternating seats): v8 18-2-0 (90%), Elo delta +175.8 locally** — WAY above the 55% defense gate. Turn times: v8 p50=36ms / p95=66ms, v7 p50=46ms / p95=103ms, both comfortably under the 1s budget. **v8 is ready to submit to Kaggle.** v8 is our A+B+D bot (TuRBO-v2 tuned heuristic + Gumbel MCTS + Bayesian opponent model) — next Kaggle submission candidate. **W4 Exp3 A/B infrastructure shipped** — `decoupled_exp3_root` now accepts `protected_my_idx` (anchor-lock contract matches UCB) + `rng` (deterministic tests); `GumbelSearchConfig.sim_move_variant: "ucb"|"exp3"` dispatches at the wiring point with a graceful warn-and-fallback on typo; 5 new tests green (2 sim_move + 3 gumbel_search dispatch), 47/47 pass in the touched modules. Ready to run the 200-game A/B gate. **`.venv-gpu` online with CUDA 12.4 torch + RTX 3070**, BC-warmstart training on GPU (60+ min in, PowerShell Out-File buffering delays epoch visibility but GPU SM remains 100%, 7794/8192 MiB). **EvoTune fitness bridge + runner CLI shipped** with 6 passing tests; mock-LLM smoke run end-to-end. **Overage-bank opening-turn deadline lift shipped + empirically validated** — Deadline.extra_budget_ms plumbed through Agent.deadline_boost_ms hook, MCTSAgent override reads obs.remainingOverageTime + amortizes (bank-2s) across first 10 turns capped at 2s/turn; 12 unit tests green + `tools/smoke_overage_bank.py` confirms all 3 live-game gates PASS (opening boost=2000 ms, opening search 1.60x midgame, midgame boost exactly 0). Key finding: boosted step 4 completes the FULL total_sims=32 budget in 644ms vs unboosted step 10 deadline-capped at 13 rollouts in 301ms. **Path D W4 audit: archetype-overrides-in-FastRolloutAgent + 0.99 posterior freeze already shipped** — no action needed.)*

**Kaggle submission trail** (public score, most recent first; full settle trajectory in brackets):
- v7 (entropy-leak fix in FastEngine._maybe_spawn_comets): **630.1** [initial 708.2 → day-1 settle 599.0 → day-2 settle 630.1]
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

**Decision: defend v7 as the floor.** Next submission must beat v7 locally
(≥55% over 200 games H2H) before shipping. W4 NN work is the path forward.

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
