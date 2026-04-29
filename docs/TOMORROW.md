# Tomorrow's Runbook (2026-04-30)

Quick-reference for picking up where 2026-04-29 left off. Full
context in `STATUS.md`.

## Where we are

- **Ladder**: v32b at 933.2 Elo (slot 1). Slot 2 held — no candidate
  beat v32b under tight-CI scrutiny over ~15 variants.
- **Phantom bugs 4/5/6 found and fixed** — first real MCTS in this
  repo's history landed today as v32b.
- **NN value-as-leaf-eval is dead** at our 64k-param model size
  (v3/v4/v5 all -190 Elo regardless of training data quality).
- **TuRBO v6 ran overnight** — check
  `runs/turbo_v6_w3pool.json` and the corresponding `.log` to see
  if it found weights better than v5's (v5 best win-rate=0.97 vs
  v6 best at last check 0.82, so likely **no improvement**).

## First-thing-tomorrow checklist

```bash
# 1. Check v32b ladder Elo (might have shifted overnight)
.venv/Scripts/kaggle.exe competitions submissions -c orbit-wars | head -3

# 2. Check TuRBO v6 finished + best result
grep "Best of\|best win_rate\|best_index" runs/turbo_v6_w3pool.log | tail -5
.venv/Scripts/python.exe -c "import json; d=json.load(open('runs/turbo_v6_w3pool.json')); print('best_win_rate:', d.get('best_win_rate'), 'best_index:', d.get('best_index'))"
```

## Decision tree based on TuRBO v6 result

### If v6 best_win_rate > 0.92 (decent improvement vs v5)

```bash
# Bundle v38 with v6 weights
.venv/Scripts/python.exe -m tools.bundle \
    --out submissions/v38_turbo_v6.py --bot mcts_bot \
    --weights-json runs/turbo_v6_w3pool.json \
    --total-sims 128 --hard-deadline-ms 850 \
    --rollout-policy heuristic --anchor-margin 0.0 \
    --sim-move-variant exp3 --exp3-eta 0.3 \
    --nn-checkpoint runs/bc_warmstart_small_cpu.pt

# 32-game H2H (16 + 16 with different seeds)
.venv/Scripts/python.exe -m tools.h2h_mirror \
    --bundles submissions/v38_turbo_v6.py,submissions/v32b_heur.py \
    --games 8 --seed 100 --step_timeout 1.0  # = 16 games
.venv/Scripts/python.exe -m tools.h2h_mirror \
    --bundles submissions/v38_turbo_v6.py,submissions/v32b_heur.py \
    --games 8 --seed 5000 --step_timeout 1.0  # = another 16 games

# If COMBINED 32-game score is positive AND CI lower bound > 0:
#   ship v38 to slot 2
```

### If v6 best_win_rate <= 0.95 (no meaningful improvement)

TuRBO with the w3 pool is saturated. Next steps in priority order:

1. **TuRBO with v32b in the pool** — currently TuRBO evaluates a
   HeuristicAgent against opponents, never against v32b itself.
   The fitness eval has nothing to push against. Build a custom
   pool that includes `submissions/v32b_heur.py` as an opponent,
   so TuRBO weights are explicitly optimized for "beat v32b".
   ~2h to wire up + ~2h to run.
2. **Bigger NN model** — push from 64k to ~500k params. Bundle
   size cap is ~700KB of weights = ~175k fp32 / ~700k int8 params.
   Re-train value head on 40k demos. Expected: maybe -100 instead
   of -190 Elo, but unlikely to flip net positive without a much
   bigger model.
3. **Algorithmic heuristic improvements** — careful re-attempt of
   the smart-sizing variants. Past attempts regressed -107 to -800
   Elo, but those were Phantom-stricken evaluations. Real-MCTS
   eval might tell a different story. Risky.

## Known files and their roles

```
runs/closed_loop_iter1_postfix/
  demos_iter1.npz       (12,607 demos, 12 games — too small for distillation)
  demos_iter1_big.npz   (40,823 demos, 30 games — current best dataset)
  value_head_iter1.pt   (v4: 12k demos, val_mse=0.20)
  value_head_iter2.pt   (v5: 40k demos, val_mse=0.39)
  policy_head_iter1.pt  (12k demos, didn't help in v34)

runs/turbo_v5_w3pool.json     (current shipped weights, v32b uses)
runs/turbo_v6_w3pool.json     (overnight tuning attempt — verify if improved)

submissions/v32b_heur.py      (the ladder-shipped bundle)
submissions/v3{3,4,5,6,7}*.py (rejected variants)
```

## Critical lessons from today

1. **16-game H2H is too noisy** in near-50% regime (CI ±70 Elo).
   v37 went +44 over 16 games and -89 over the next 16. Default
   to 32-64 games for ship/no-ship decisions.
2. **MCTS overrides only 9.2% of turns** in v32b. The other 91% IS
   the heuristic. So heuristic improvements > MCTS tweaks for
   ladder Elo gains.
3. **Bundle-time vs runtime divergence is dangerous**. Three
   different Phantom-class bugs (tight_cfg field-drop, fresh_game
   NN-fn-drop, _softmax name-collision TypeError) all silently
   ran the heuristic regardless of bundle config. Regression tests
   added for each.
4. **NN-as-leaf-eval doesn't work** at our model size. Three value
   heads trained on increasingly favorable data all gave -190 Elo
   in MCTS. Structural fix needed (bigger model, different
   architecture, or mixture eval), not more data.
