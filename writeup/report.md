# Orbit Wars — a heuristic-first, search-amplified, learning-optional bot

*Draft outline — sections filled as evidence lands.*

---

## 0. TL;DR

(Fill on submission.)

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

*(To be filled once the EvoTune run has produced candidates worth
evaluating.)*

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

*(Drafted in W5 if we ship; else a retrospective "here's what we tried,
here's where it broke.")*

---

## 9. Ablations table

*(Fill in W6. Components: Gumbel on/off, BOKR on/off, opponent model
on/off, conv-vs-transformer, student-prior on/off, macro-library
on/off.)*

---

## 10. Final tournament + leaderboard trajectory

*(W6. Plot of leaderboard rank over time.)*

---

## 11. Honest section

*(W6. What we dropped, what we got wrong, what we'd do with more
compute. First draft: see §6 bug-fix history — most of our W2 was
spent learning that MCTS with rollout noise overrides a good
heuristic.)*

---

## 12. Reproducibility appendix

- `python -m orbitwars.engine.validate --seeds 1000 --turns 500` — parity gate.
- `$env:PYTHONPATH="src;."; .venv\Scripts\python.exe -m pytest tests/ -q` — 117 tests.
- `$env:PYTHONPATH="src;."; .venv\Scripts\python.exe tools\smoke_mcts_multi_seed.py` — 6-game MCTS multi-seed.
- `$env:PYTHONPATH="src;."; .venv\Scripts\python.exe tools\diag_posterior_concentration.py` — posterior concentration.
- `$env:PYTHONPATH="src;."; .venv\Scripts\python.exe tools\smoke_opp_model_vs_archetypes.py --margin 0.5` — exploitation smoke.
- `python -m tools.bundle --bot mcts_bot --out submissions\mcts_v1.py` — bundle for Kaggle.
