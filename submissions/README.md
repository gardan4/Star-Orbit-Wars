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

1. Wrap the bundled single-file submission (e.g. `noop_v0.py`) inside a
   Kaggle Notebook that defines `def agent(obs, cfg): ...`.
2. Push the notebook:
   ```powershell
   .venv\Scripts\kaggle.exe kernels push -p submissions\kaggle_notebook\
   ```
3. Submit by kernel slug:
   ```powershell
   .venv\Scripts\kaggle.exe competitions submit -c orbit-wars -k marcmeijers01/orbit-wars-noop -m "W1 dry-run"
   ```
4. Poll status:
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
