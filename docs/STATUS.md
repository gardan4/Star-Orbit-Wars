# Orbit Wars — Repo Status & Architecture

*Last updated: 2026-04-22 (W2→W3 transition, 6-week plan)*

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
- **12. Global RNG pollution** — `FastEngine._maybe_spawn_comets` called `random.randint(1, 99)` on the *global* `random` module. Every MCTS rollout that crossed a comet spawn step drained the same stream the Kaggle judge uses, desynchronizing the real game's subsequent comet ship counts. Fix: instance-local `random.Random()` per FastEngine; validator opts into sharing via `rng=random` kwarg.
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
- **22. Fast rollout policy** — `orbitwars.bots.fast_rollout.FastRolloutAgent` is a ~50-line nearest-target static-push agent (single `atan2` per launch, no arrival table, no Newton solve). Selectable via `GumbelConfig.rollout_policy={"heuristic", "fast"}`; default stays `"heuristic"` to preserve shipped mcts_v1 behavior. **Measured head-to-head at mid-game obs, depth=15 rollouts**:

| | per `act()` | per rollout | sims @ 300 ms budget |
|---|---|---|---|
| HeuristicAgent (default) | 3.2 ms | 104 ms | **2** |
| FastRolloutAgent | **8 µs** (~380×) | **11 ms** (~9.4×) | **27** |

  This is the fix for item 1 in §7: going from 2 sims/turn to 27 sims/turn at 300 ms budget means SH finally has signal above rollout noise. Expected impact: real policy improvement in MCTS at the same CPU budget; unlocks Path D exploitation signal that the running smoke is currently NOT showing (rusher cell: posterior identifies rusher at p=1.00 for 275+ fires/game, but use_model=True/False both give 2W/2L — consistent with "search was budget-starved" rather than "opponent model doesn't help"). 6 new tests lock in action validity (ownership, ship cap, finite angle) plus the sim-count inequality vs. the heuristic policy at matched budget.

Test count: 117 → 128 (+11 from Path D empirical + fast rollout work).

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

### W4 (GPU rental begins)
- [ ] JAX engine (`src/orbitwars/engine/jax_engine.py`) for offline self-play.
- [ ] Arch bake-off: centralized per-entity-grid conv vs set-transformer — train 1M steps each, pick winner.
- [ ] BC warm-start from Path A.
- [ ] Begin PPO + PFSP run.

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
2. **Ax tuning underperforming** — all 25 trials scored 16.7% win-rate in the first run. Likely the opponent pool (`{starter_agent, heuristic_v0}`) is too strong relative to the weight perturbations. Plan: expand pool, widen param bounds, rerun.
3. **`from_scratch` still uses global random** — intentional (needs reference-matching map), but means test setup affects global state. Call `random.seed(seed)` explicitly when ordering matters.
4. **No 4-player coverage in smokes yet** — all W2 smoke runs are 2-player. 4-player is a distinct regime; needs its own validation in W3.
5. **Submission flow is human-gated** — the Kaggle Simulation track requires rules acceptance in the browser, not via API. Nothing to fix code-side.

---

## 8. Status summary

Week 2 closes with **MCTSAgent shipped at the heuristic-floor default** (margin=2.0) — the four critical bugs that caused the initial catastrophic losses are fixed, tested, and regression-guarded. The single-seed win turned out to be a timing-variance artifact once we ran the multi-seed sweep; rather than chase phantom wins, we lock in the Path A floor and let W3 (archetypes, opponent model, more sims) and W4-5 (neural prior) do the lifting that actually makes search net-positive. Next week: archetype portfolio + Bayesian opponent model, then first real Kaggle submission.
