# Submissions

Auto-built single-file Kaggle submissions. Regenerate from source via:

```powershell
.venv\Scripts\python.exe -m tools.bundle --bot <name> --out submissions\<name>.py --smoke-test
```

| File | Purpose | Bot |
|---|---|---|
| `noop.py` | Pipeline dry-run (W1 gate) — confirms the judging machine accepts our bundle format | always-pass |
| `noop_v0.py` | Earliest archived noop for the first Kaggle Notebook push | always-pass |
| `heuristic_v1.py` | Path A bot v1 — parameterized heuristic w/ intercept math + arrival table | HeuristicAgent |
| `mcts_v1.py` | Path B bot: Gumbel MCTS + anchor-locked heuristic floor (margin=2.0) + posterior-aware rollouts (no fast_rollout module) | MCTSAgent |
| `mcts_v2.py` | Same default behavior as v1 but the `FastRolloutAgent` module is inlined, so switching `GumbelConfig.rollout_policy="fast"` works without a re-bundle. 141KB (+7KB vs v1). Smoke-tested vs `random` (step 111, W). | MCTSAgent |
| `mcts_v3` … `mcts_v7` | Successor MCTS bundles (see `docs/STATUS.md` for per-version deltas). | MCTSAgent |
| `v8.py` | Previous floor at 769.5; superseded by v9. `mcts_bot` + TuRBO-v2 tuned HEURISTIC_WEIGHTS (18 keys, +0.233 wr vs defaults). `--weights-json runs/turbo_v2_20260424.json`. | MCTSAgent |
| `v9.py` | Previous floor at 934.4; superseded by v11. v8 weights + `sim_move_variant=exp3`, eta=0.3. Peaked 981.9 (+212 Elo over v8 at 769.5). Passed W4 A/B gate at default weights (wr=0.575 Elo +52.5, N=20). Defense H2H vs v8 = 8W-6T-6L wr=0.550 (marginal). Early ladder (first 30 min) mis-signaled at -154 Elo vs v8; after 3-4h of replays reversed to +150 to +212 Elo. **Lesson: bundled H2H at N=20 has SE ~0.112 and is ~3× less sensitive than a full ladder settle batch; early-ladder (<6h) scores are near-useless.** | MCTSAgent |
| `v11.py` | **CURRENT LADDER FLOOR AT 880.8** (full overnight settle, peak 926.0 at 23:48 UTC, +30 Elo over v9 at 850.7, +90 over v8 at 790.7). TuRBO-v3 60-trial HEURISTIC_WEIGHTS (trial 42 wr=1.000 vs 9-archetype pool, 45/45 games) + `sim_move_variant=exp3`, eta=0.3. Aggressive weights: `w_production=20.0` (maxed), `mult_comet=5.0` (maxed), `mult_enemy=5.0` (maxed), `mult_reinforce_ally=0.0`, `keep_reserve_ships=0.0`, `min_launch_size=30.0` (maxed). Multiple weights at bound edges → TuRBO-v4 staged in `turbo_runner.py::PARAM_BOUNDS_V4` (not launched pending v11 final settle). **v11 vs v9 bundled H2H = 17W-3L = 85% wr, Elo delta +150.6** — strongest H2H signal in the v5→v11 lineage. **Ladder delta (+32 Elo) is smaller than H2H predicted (~+200 from the Elo-from-wr formula): archetype-pool overfitting is real but not catastrophic.** v11 is also FASTER than v9 (p95=59ms vs 85ms). `--weights-json runs/turbo_v3_20260424.json`. | MCTSAgent |

## Building bundles with config overrides

`tools/bundle.py` supports two kinds of overrides:

1. **`--weights-json PATH`** — injects a `HEURISTIC_WEIGHTS.update({...})` block after the heuristic module. Accepts top-level dicts or TuRBO-runner shapes (`best_weights`/`best_point`/`best_index`+`trials`).
2. **`--sim-move-variant {ucb,exp3}` + `--exp3-eta FLOAT`** — applies only to `--bot mcts_bot`. Emits a `_bundle_cfg = GumbelConfig()` shim and rewrites the factory to `MCTSAgent(gumbel_cfg=_bundle_cfg, rng_seed=0)`. Use `exp3` only after the W4 A/B gate has passed (ship_gate ≥0.55 wr vs ucb).

Example (future v9 with weights + exp3 variant):
```powershell
.venv\Scripts\python.exe -m tools.bundle --bot mcts_bot `
  --weights-json runs\turbo_v2_20260424.json `
  --sim-move-variant exp3 --exp3-eta 0.3 `
  --out submissions\v9.py --smoke-test
```

## W1 GATE: Kaggle dry-run (noop)

**Orbit Wars is a Kaggle Simulation competition** (aka "code
competition"). Simulation comps do NOT accept a raw `-f file.py`
submission — they take a Kaggle Notebook that defines the agent. Our
first attempt at `kaggle competitions submit -c orbit-wars -f file.py
-m "msg"` returned **HTTP 403 Forbidden**; this is the expected failure
mode for that flow, not an auth or bundling problem.

### One-time setup (human-gated)

1. **Accept rules in the web UI.** There is no API for this — open
   https://www.kaggle.com/competitions/orbit-wars/rules and click
   "I Understand and Accept."
2. **API token.** Either:
   - `KAGGLE_API_TOKEN=KGAT_...` (bearer token) in `.env` (our current
     setup), or
   - Legacy `kaggle.json` at `%USERPROFILE%\.kaggle\kaggle.json`.

### Smoke-check auth

```powershell
$env:KAGGLE_API_TOKEN = (Get-Content .env | ?{$_ -match "KAGGLE_API_TOKEN="}) -replace "KAGGLE_API_TOKEN=",""
.venv\Scripts\kaggle.exe competitions list -s "orbit"
```

Should list `orbit-wars` with `userHasEntered: True`.

### Submit via Kaggle Notebook (correct flow)

1. **Build the notebook wrapper** for a bundled .py. The generic builder
   handles v2+ (slug + title + ipynb name all derived from flags):
   ```powershell
   .venv\Scripts\python.exe -m tools.build_kaggle_notebook `
     --bundle submissions\v8.py `
     --slug gardan4/orbit-wars-mcts-v8 `
     --title "Orbit Wars MCTS v8" `
     --out submissions\kaggle_notebook_v8\
   ```
2. **Push the notebook** (Kaggle runs the cell, producing
   `submission.py` as the kernel output):
   ```powershell
   .venv\Scripts\kaggle.exe kernels push -p submissions\kaggle_notebook_v8\
   ```
3. **Wait for the kernel to finish running** before submitting. Poll
   until status is COMPLETE:
   ```powershell
   .venv\Scripts\kaggle.exe kernels status gardan4/orbit-wars-mcts-v8
   ```
4. **Submit by kernel slug + version + output filename** (all three
   are required — the CLI errors if any are missing with "Code
   competition submissions require both the output file name and the
   version number"):
   ```powershell
   .venv\Scripts\kaggle.exe competitions submit -c orbit-wars `
     -k gardan4/orbit-wars-mcts-v8 -v 1 -f submission.py `
     -m "v8: TuRBO-v2 tuned weights"
   ```
   (`-v` is the **kernel** version, usually `1` for the first push;
   `-f submission.py` is the **output filename** the kernel produced,
   which matches our `%%writefile submission.py` cell.)
5. **Poll submission status** (score can update for hours as the
   ladder replays):
   ```powershell
   .venv\Scripts\kaggle.exe competitions submissions -c orbit-wars
   ```

**Gate passes** when the submission appears in the list with status
`complete` (not `error`). Score irrelevant — we just need to know the
pipeline works end-to-end.

## Versioning

- `noop.py` — pipeline validation only. Never update.
- `heuristic_v1.py` — track per weight change during W1/W2 tuning. Replace in-place.
- `mcts_v1.py` — Path B bot with Path D opponent model. Replace in-place when source changes; re-run the smoke harness before each refresh.
- `mcts_v2.py` — v1 + fast_rollout module inlined. No behavior change at shipped default; future-proofs toggling `rollout_policy="fast"` in a weight override without a re-bundle. Smoke-test before each refresh.
- Future: `heuristic_v2.py` (TuRBO-tuned, pending second Ax run), `nn_mcts_v1.py` (Path C).

## Ship-gate criterion (post-v9 update, rewritten after v9 settled at 890.8)

The original gate was "H2H wr ≥ 0.55 over 20 games". v9 hit **exactly 0.550**
locally (8W-6T-6L vs v8) — then went on to gain **+121 Elo on the ladder**
(890.8 vs v8 769.5) after overnight settle. **So the 0.55@N=20 gate did work**:
it correctly flagged v9 as a ship-candidate. What DIDN'T work was the early-ladder
call (first 30 min showed -154 Elo) — that window is too noisy to inform decisions.

**What we now know**:
1. Bundled H2H wr=0.55@N=20 is a NECESSARY-but-WEAK signal. SE=0.112, so true wr
   could be anywhere in [0.33, 0.77]. Pass → eligible to ship; don't interpret the
   point estimate as a reliable prediction.
2. **Early-ladder scores (<6h after submit) are NOISE** and should not influence
   ship decisions on already-submitted bots. Only full-settle (overnight minimum)
   scores are signal.
3. If a change is expected to be marginal (e.g. ±10 Elo H2H), consider running a
   larger N (60+ games) to tighten the SE to ~0.065, but it's not strictly
   required.

**Gate for post-v9 submissions**:
- Minimum: H2H wr ≥ 0.55 over ≥20 games vs current ladder floor (same as before).
- Preferred when N=20 gets a borderline result (0.50-0.60): re-run to 60+ games
  and require the 95% CI lower bound to clear 0.50.
- Decision horizon: allow ≥12h of ladder settle before reverting or declaring
  regression. 30-min ladder dips do not refute a 0.55 H2H pass.
