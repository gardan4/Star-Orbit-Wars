# NN-driven rollouts — implementation spec

The structural reason every NN-as-leaf experiment (v33-v36) lost to v32b's
heuristic rollouts: **MCTS rollouts use `HeuristicAgent` on both sides**,
so Q estimates measure "how the heuristic plays from here" — exactly
what the heuristic anchor already represents. Search has no information
that disagrees with the anchor; the override rate stays at 9.2%.

Mixed leaf eval (now shipped at [gumbel_search.py:175](src/orbitwars/mcts/gumbel_search.py),
GumbelConfig.value_mix_alpha) softens this by blending NN + heuristic
at the leaf, but doesn't fix the rollout policy itself.

The fix: **replace `HeuristicAgent` with an NN-greedy agent inside
rollouts** when configured. Q estimates then measure "how the NN plays
from here", which is genuinely different from the heuristic anchor.
This is the AlphaZero recipe in spirit (their MCTS uses pure NN; we
keep the heuristic anchor as a Pareto floor and use NN inside rollouts).

## Pre-requisites

1. A trained NN policy head good enough that an NN-greedy agent beats
   the random / fast-rollout baseline. The current 64k-param head
   (val_acc 0.568, val_ce 2.07 with tau-smoothed targets) is on the
   edge; the bigger 250k+ backbone trained via [train_az_bigger.py](tools/train_az_bigger.py)
   is the realistic target.
2. The `move_prior_fn` pathway (already shipped in [nn_prior.py](src/orbitwars/nn/nn_prior.py))
   gives us NN-policy logits per (planet, candidate). We re-use the
   same logits for rollouts.

## Code change

Add a new rollout policy `"nn"` and an NN-rollout-agent factory.

### 1. New agent: `NNRolloutAgent` in `src/orbitwars/bots/nn_rollout.py`

A minimal `Agent` subclass that takes the same NN logits used for
priors, computes argmax (or top-K Gumbel sample) per planet, and
emits the resulting wire action. ~80 lines of code, mirroring
`fast_rollout.py`'s shape.

```python
class NNRolloutAgent(Agent):
    """NN-greedy rollout policy.

    Given a ConvPolicy + obs, computes per-planet candidate moves via
    the heuristic action generator (cheap; ~0.5 ms), reads the NN
    logits at each planet's grid cell, and selects the argmax-channel
    move per planet. Skips the heuristic's expensive scoring +
    trajectory walk — all decisions come from the NN.

    Cost: ~1-2 ms per act() — between fast_rollout (~0.02 ms) and
    full heuristic (~4-5 ms). Roughly 30-50 sims/turn at 850 ms
    budget vs heuristic's 12-16.
    """

    def __init__(self, model, cfg, weights):
        self.model = model
        self.cfg = cfg
        self.weights = weights

    def act(self, obs, deadline):
        po = parse_obs(obs)
        if not po.my_planets:
            return no_op()
        # 1. Generate candidates (cheap; reuses heuristic action gen).
        per_planet = generate_per_planet_moves(po, ArrivalTable(), self.weights)
        # 2. NN forward pass.
        grid = encode_grid(obs, po.player, self.cfg)
        x = torch.from_numpy(grid).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(x)
        # 3. Per-planet argmax.
        moves = []
        for pid, candidates in per_planet.items():
            if not candidates:
                continue
            mp = po.planet_by_id[pid]
            gy, gx = planet_to_grid_coords(mp)
            cell_logits = logits[0, :, gy, gx].numpy()
            best_score = -float('inf')
            best_move = None
            for cand in candidates:
                ch = candidate_to_channel(mp, cand)
                if cell_logits[ch] > best_score:
                    best_score = cell_logits[ch]
                    best_move = cand
            if best_move and not best_move.is_hold:
                moves.append(best_move.to_move())
        return moves
```

### 2. Plumbing in `gumbel_search.py`

- Extend `GumbelConfig.rollout_policy` literal to include `"nn"`.
- Extend `GumbelRootSearch` to accept an `nn_rollout_factory` callable
  that returns a fresh `NNRolloutAgent` per rollout.
- In the rollout-policy dispatch ([gumbel_search.py:735-750](src/orbitwars/mcts/gumbel_search.py)),
  branch on `rollout_policy == "nn"`:
  - `_my_future_factory = nn_rollout_factory`
  - `_opp_factory = nn_rollout_factory` (or keep heuristic/posterior;
    A/B both)

### 3. Plumbing in `mcts_bot.py`

- Add `nn_rollout_factory: Optional[Callable] = None` to `MCTSAgent.__init__`.
- Pass it through to `GumbelRootSearch.__init__`.
- Mirror in the `tight_cfg` Phantom-4 fix at [mcts_bot.py:482](src/orbitwars/bots/mcts_bot.py)
  (no-op for the cfg itself, but make sure we don't drop `nn_rollout_factory`
  from `self._search` on game restart — Phantom 5 lesson).

### 4. Plumbing in `bundle.py`

- Extend `--rollout-policy` choices to include `"nn"`.
- When `--rollout-policy=nn`, the bundle bootstrap creates an
  `NNRolloutAgent` factory closure analogous to `_bundle_move_prior_fn`
  (but using the same checkpoint).

### 5. Tests

Mirror the test pattern at [tests/test_gumbel_search.py:1075](tests/test_gumbel_search.py):

- `test_nn_rollout_changes_q_vs_heuristic`: same seed, swap rollout
  policy heuristic → nn, Q values must differ.
- `test_nn_rollout_falls_back_when_factory_none`: rollout_policy="nn"
  with factory=None → warns and falls back to heuristic.
- `test_nn_rollout_byte_identical_at_constant_logits`: factory that
  returns argmax of constant logits should produce predictable
  wire actions (sanity).

## Estimated effort

- Code: ~200-300 LOC across 3 files + 1 new file.
- Tests: ~150 LOC (3 unit tests + 1 integration test).
- Bundle test: existing `tests/test_bundle.py` pattern.
- **1.5 - 2 days** of focused implementation.

## Estimated lift

- **+50-100 Elo** if the bigger backbone (250k+ params) is the rollout
  policy. Q-values now reflect NN strategy; the 9.2% override rate
  should rise to 25-40%, multiplying the search advantage.
- **Zero or negative** if the rollout policy is the small 64k checkpoint
  — the fast_rollout test already showed weak rollout quality dominates
  quantity. NN-rollouts only win when their per-call quality exceeds
  the heuristic's.

## Combined v40 config (the realistic top-of-the-stack)

After both [train_az_bigger.py](tools/train_az_bigger.py) and this spec
land, the v40 config is:

```bash
python -m tools.bundle --bot mcts_bot \
  --weights-json runs/turbo_v3_20260424.json \
  --sim-move-variant exp3 --exp3-eta 0.3 \
  --rollout-policy nn \
  --nn-checkpoint runs/az_v39_bigger.pt \
  --anchor-margin 0.5 \
  --total-sims 128 --hard-deadline-ms 850 --num-candidates 8 \
  --out submissions/v40.py --smoke-test
```

Anchor-margin=0.5 is the sweet spot once NN rollouts are working — gives
search permission to override but keeps a margin for safety.
`num-candidates=8` (vs 4) doubles search width, justified because each
NN-rollout sim is 2-4× cheaper than a heuristic rollout, so we can
afford the wider tree.
