# Archive

Stale code and artifacts from the v1–v42 era, preserved for history but no
longer in active rotation. Don't add new files here. If something here turns
out to be useful, move it back into the live tree (and re-test it).

## Layout

| Dir | Contents |
|---|---|
| `diagnostics/` | One-shot `diag_*.py` scripts written to investigate specific bugs (Phantom 4/5/6, comet entropy leak, posterior concentration, etc.). All findings either fixed or recorded in [docs/STATUS.md](../docs/STATUS.md). |
| `h2h_phantom_era/` | `h2h_*.py`, `_h2h_one_game.py`, `ab_exp3_vs_ucb.py`. The `+51.8 Elo H2H` results from these were proven to be harness artifacts (Phantom 1.0 / 2.0 / 3.0); see STATUS.md. Don't trust historical numbers from them without re-running on the post-Phantom-fix harness. |
| `build_one_shot/` | Per-version bundle builders (`build_v11_1.py`, `build_v12.py`, `build_notebook_v7.py`). Replaced by the generic `tools/build_kaggle_notebook.py`. |
| `smoke_old/` | Outdated smoke tests (`smoke_v12_*`, `smoke_4p`, `smoke_overage_bank`, `smoke_bokr_ablation`, etc.). The active smokes live in `tools/`. |
| `text_dumps/` | Output captures from past pytest runs / smoke runs / diagnostic walks. Keep for grep-able history; nothing in here is consumed by code. |
| `ps_scripts/` | Windows PowerShell wait-loops (`wait_kaggle_kernel.ps1`, `wait_kaggle_submission.ps1`). Replaced by inline `kaggle.exe kernels status` polling in shell scripts. |
| `ship_scripts/` | Per-version ship orchestrators (`ship_v40.sh`, `ship_v41.sh`, `poll_ladder_part*.sh`, `run_iter2_chain.sh`). The new pipeline is in `tools/cloud/` (see [docs/BIG_PUSH.md](../docs/BIG_PUSH.md)). |
| `profile_old/` | Stale per-turn / posterior / turn-time profilers. Kept the cprofile + rollout-components ones in `tools/`. |
| `old_submissions/` | Notebooks + bundles for v37–v40k. Current submissions tree only retains v41 (ladder leader), v42 (iter-2 baseline), v42a (bigger int8 AZ ready). |

## Why this lives in-tree (not deleted)

Two reasons:
1. The phantom-bug post-mortems in STATUS.md cite specific files here. Easier to reason about "what did v22 actually do" if the code is grep-able.
2. The closed-loop iter results (v40j → v41 → v42) are the empirical ground truth for "what works"; preserving the v40* bundles helps reproduce H2H if that ever becomes useful.

If the repo gets too heavy later, this whole directory can be deleted as a single commit — no live code imports it.
