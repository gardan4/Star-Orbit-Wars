# Bigger AZ via Kaggle Dataset — diagnosis trail

This doc records the multi-attempt debugging of why every NN-on-wire submission errored on the Kaggle ladder before v40h finally passed. Recorded so future sessions don't redo the work.

## Symptom

Submissions that loaded a 5M+ param AZ-trained checkpoint via the Kaggle Dataset path errored on the LADDER (kernel build phase succeeded; `submission.py` was correctly written). Each attempt had a different theory; some fixed real bugs, some didn't.

| Submission | Config | Result | Root cause |
|---|---|---|---|
| v38 | small AZ via inline + value_mix=0.5 | ERROR | mix doubles per-leaf cost; Kaggle CPU exceeds actTimeout |
| v40b | bigger AZ via Dataset + NN rollouts | ERROR | Git Bash path mangling: `/kaggle/input/...` → `C:/Program Files/Git/kaggle/...` |
| v40c | same as v40b but path fixed via `MSYS_NO_PATHCONV=1` | ERROR | timing — local mean 484 ms × 1.5x Kaggle ≈ 730 ms, hit tail |
| v40d | total_sims 24 / dl 400ms | ERROR | timing — max 821 ms locally → ~1230 ms Kaggle worst case |
| v40e | total_sims 16 / dl 300ms | ERROR | NOT timing — local p99=434 ms / max=447 ms; **another bug** |
| v40f | DIAGNOSTIC: heuristic rollouts + bigger AZ as priors only | ERROR | confirmed: not specific to NN rollouts. Loading bigger AZ at all is the issue. |
| v40g | "cleaned" bigger AZ via Dataset (stripped non-essential fields) | ERROR | `WindowsPath` in `training_args` was the smoking gun, but cleaning didn't fix it on its own |
| v40h | DIAGNOSTIC: small bc_warmstart inline + heuristic | **COMPLETE @ 947** | confirmed: bundle code is fine; problem is specific to AZ-trained ckpts via Dataset OR bigger arch |
| v40j | DIAGNOSTIC: small AZ inline (`az_v37_clean`) + heuristic | **COMPLETE @ 920** | AZ-trained checkpoints CAN load on Kaggle when inlined and cleaned |
| v40k | small AZ inline + NN rollouts | **COMPLETE @ 859** | works, but small NN as rollout policy hurts (doesn't beat 15-ply heuristic) |

## Confirmed bugs (fixed)

### Bug 1: Git Bash MSYS path translation

When `tools/bundle.py` is invoked from Git Bash on Windows, the shell auto-translates Linux-style absolute paths (`/kaggle/input/...`) to Windows paths (`C:/Program Files/Git/kaggle/input/...`). The translated path is hard-coded into the bundled `.py` at `_BUNDLE_BC_CKPT_PATH = '...'`. On Kaggle's Linux container, `torch.load(...)` with that path raises `FileNotFoundError`.

**Fix landed**: `tools/bundle.py` now refuses any `--nn-dataset-path` whose 2nd char is `:` (Windows drive letter). Suggests `MSYS_NO_PATHCONV=1` prefix.

**How to apply**: when shipping a Dataset-backed bundle from Git Bash, prefix the bundle command with `MSYS_NO_PATHCONV=1`. Or invoke from PowerShell. Always grep:
```bash
grep "_BUNDLE_BC_CKPT_PATH =" submissions/<bundle>.py
```

### Bug 2: WindowsPath unpickle failure on Kaggle Linux

`tools/train_az_head.py` saved `training_args = vars(args)`, which embedded `pathlib.WindowsPath` objects directly. On Kaggle's Linux container, `torch.load(...)` tries to unpickle these and raises `AttributeError: Can't get attribute 'WindowsPath' on <module 'pathlib'>`. Fails BEFORE any agent code runs.

**Fix landed**: `tools/train_az_head.py` now stringifies any `Path` in `training_args` before save (line 339-348). Existing checkpoints can be re-cleaned via:

```python
import torch
ckpt = torch.load(src, map_location='cpu', weights_only=False)
clean = {'model_state': ckpt['model_state'], 'cfg': ckpt['cfg']}
torch.save(clean, dst)
```

Same fix landed for `tools/cloud/distill.py` and `tools/cloud/quantize.py`.

## Unconfirmed (suspected)

### "v40g still errored after WindowsPath strip"

After cleaning the Dataset-uploaded bigger AZ, v40g still errored. The kernel build succeeded; the failure was during ladder play. We never got concrete logs (Kaggle CLI's `competitions logs` had a `KeyError: 'content-length'` bug; the API endpoint returned HTML; the Python client raised on the same key).

**Working theory**: there's a SECOND issue with the bigger 48ch×4 architecture that we couldn't isolate. v40i (bigger AZ inline) couldn't be pushed (1.38 MB exceeds Kaggle's silent ~1 MB notebook limit). v42a (bigger AZ int8 inline at 560 KB) builds locally but was never shipped because v40j-style heuristic-rollout config with smaller AZ was already winning.

**Why we don't care**: the **distillation path sidesteps this entirely**. After Lever 1 (cloud AZ) produces a 5M-param teacher, Lever 2 (distill to 1M student + int8) yields a 250-300 KB checkpoint that fits inline cleanly. v40j proves AZ-trained checkpoints up to ~64k params load fine when inlined; v42a (191 KB int8 of the bigger AZ) builds but was never ladder-tested. The distilled output should be in the same size envelope.

**If we ever do need to ship via Dataset**: re-investigate by:
1. Push a TINY model via Dataset (just to confirm the path itself works for AZ-trained ckpts).
2. Bisect by checkpoint size (not architecture) — strip layers until one size succeeds.
3. Try `weights_only=True` in the bundle's `torch.load` — narrows the unpickle surface (only allows tensors + a stdlib whitelist).

## Operational checklist (next time you ship via Dataset)

```bash
# 1. Save fp16 / int8 checkpoint with NO Path objects, NO heavy metadata
python -c "
import torch
ckpt = torch.load('runs/<src>.pt', map_location='cpu', weights_only=False)
torch.save({'model_state': ckpt['model_state'], 'cfg': ckpt['cfg']}, 'runs/<src>_clean.pt')
"

# 2. Upload as Kaggle Dataset
mkdir -p submissions/<dataset_dir>
cp runs/<src>_clean.pt submissions/<dataset_dir>/<file>.pt
cat > submissions/<dataset_dir>/dataset-metadata.json <<EOF
{"title": "...", "id": "<user>/<slug>", "licenses": [{"name": "CC0-1.0"}], "description": "..."}
EOF
(cd submissions/<dataset_dir> && kaggle datasets create -p .)

# 3. Bundle WITH MSYS_NO_PATHCONV (or from PowerShell)
MSYS_NO_PATHCONV=1 python -m tools.bundle --bot mcts_bot --nn-dataset-path \
    /kaggle/input/<slug>/<file>.pt ...

# 4. Verify path constant in the bundled .py
grep "_BUNDLE_BC_CKPT_PATH =" submissions/<version>.py
# Must be EXACTLY: _BUNDLE_BC_CKPT_PATH = '/kaggle/input/<slug>/<file>.pt'

# 5. Build notebook with --dataset-source
python -m tools.build_kaggle_notebook --dataset-source <user>/<slug> ...

# 6. Push, wait, submit. If it errors despite all of the above, fall
#    back to int8-inline distillation (Lever 2 in BIG_PUSH.md).
```
