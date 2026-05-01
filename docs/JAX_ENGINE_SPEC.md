# JAX Engine — Implementation Spec

## Purpose

Replace `FastEngine` (numpy, single-core, ~50-200k steps/sec) with a `jax`-based engine that runs **vmap'd over thousands of parallel envs** on a single GPU. Target: **5-20M aggregate env steps/sec**.

This is a **training-throughput multiplier**, not a runtime ladder lever (Kaggle is CPU-only). The win: Lever 1's "1000 self-play games per iter" can become "50,000 games per iter" at the same wall time. Or 10 iters compress from 24h to ~30min.

## Status

[`src/orbitwars/engine/jax_engine.py`](../src/orbitwars/engine/jax_engine.py) is a SKELETON. The API surface (`State` pytree, `StaticCfg`, `reset`/`step` signatures) is locked. The physics is empty. This doc is the implementation plan.

## Constraints (non-negotiable)

1. **Fixed-shape arrays, no growth/shrink.** All planet / fleet / comet pools are `N_MAX`-sized with an `alive` mask. JAX's `vmap` requires identical shapes across batch.
2. **Pure-functional `step`.** No mutation. Use `jax.lax.cond` / `jnp.where` instead of Python control flow.
3. **Randomness threaded explicitly via `jax.random.PRNGKey`.** No globals. The `state.key` field splits at every step.
4. **No Kaggle-engine parity.** The training engine only needs to be self-consistent + a reasonable inductive bias for the real game. Parity-matched rollouts go through `FastEngine` (already 1000-seed gated).
5. **dtype `float32` everywhere; `int32` for counts and slot indices.** Best A100 throughput.

## Module-by-module mapping (numpy `FastEngine` → JAX)

The reference implementation lives at [`src/orbitwars/engine/fast_engine.py`](../src/orbitwars/engine/fast_engine.py). Each method below maps to its JAX equivalent.

| `FastEngine` method | Behavior | JAX implementation |
|---|---|---|
| `_move_fleets_and_collide` | Step each in-flight fleet by its speed; check collision with planets/comets; resolve combat | `jax.lax.scan` over fleet-pool with `jnp.where`-masked collisions. Order matters; copy the deterministic resolution sequence from numpy. |
| `_rotate_planets_and_sweep` | Advance orbital phase; sweep fleets that pass through new planet position | Pure `cos`/`sin` update + masked sweep. Trivial with `jnp.where`. |
| `_maybe_spawn_comets` | At fixed steps spawn 1-3 comets with random paths | Pre-compute ALL comet paths at `reset()` (not at runtime) — drops the retry-up-to-300-iters pattern. Pre-computed array is `(MAX_COMETS, MAX_PATH_LEN, 2)`. |
| `_apply_actions` | Insert new fleets from agent actions into first free slot | `jax.lax.scan` over agent actions, mask-aware insert. |
| `_check_done` / `_score` | Win condition + final ship-count tally | `jnp.where` + `jnp.sum`. |

## Pseudocode for the inner loop

```python
import jax
import jax.numpy as jnp

def step(state: State, actions: jnp.ndarray) -> tuple[State, jnp.ndarray, jnp.ndarray]:
    """One game tick. actions: (num_agents, MAX_LAUNCHES, 3)
       with action[..., 0] = -1 for invalid slots."""
    # 1. Insert new fleets from actions (mask-aware)
    state = _apply_actions(state, actions)

    # 2. Move fleets + collide
    state = _move_fleets_and_collide(state)

    # 3. Rotate planets + sweep
    state = _rotate_planets_and_sweep(state)

    # 4. Comet movement (use pre-computed path)
    state = _step_comets(state)

    # 5. Production accumulation on owned planets
    state = _accumulate_production(state)

    # 6. Step counter + done check
    state = state._replace(step=state.step + 1)
    reward = _per_step_reward(state)  # ship-count delta, normalized
    done = _check_done(state)

    # 7. Split RNG key
    new_key, _ = jax.random.split(state.key)
    state = state._replace(key=new_key)

    return state, reward, done


# Vmap'd over batch of envs, scanned over time
def play_episode(initial_states: State, agent_fn, max_steps: int):
    def step_fn(state, _):
        actions = agent_fn(state)  # (B, num_agents, MAX_LAUNCHES, 3)
        next_state, reward, done = jax.vmap(step)(state, actions)
        return next_state, (reward, done)

    final_state, (rewards, dones) = jax.lax.scan(
        step_fn, initial_states, jnp.arange(max_steps)
    )
    return final_state, rewards, dones
```

The whole episode runs in a single `jit`-compiled `scan` — minimal Python overhead. With `vmap` over B=1024 envs, A100 throughput should hit 5-20M aggregate steps/sec.

## Parity testing — load-bearing

Without parity to the numpy reference, the JAX engine could silently train the agent on a different game. Test design:

1. `tests/test_jax_engine_parity.py` — for N=1000 random seeds:
   - Reset both `FastEngine` and `jax_engine` from same seed.
   - Step both with the SAME random agent's actions for 500 turns.
   - Assert state-equal at every step (to fp32 last-bit precision).

The numpy `FastEngine` is the source of truth — it's parity-matched against the official `kaggle_environments.envs.orbit_wars` engine on a 5/5 × 100-turn gate (per [docs/STATUS.md](STATUS.md) §3 Engineering). So JAX-vs-FastEngine parity gives transitive parity to Kaggle.

If parity fails, the diff diagnostics live in `_seg_dist_many_points_single_seg` and `_handle_combat` — the two non-trivial ordering points in `FastEngine`.

## Implementation order (3-5 days focused)

| Day | Task |
|---|---|
| 1 | `reset` — set up State pytree, deterministic planet/comet placement, pre-compute comet paths. Smoke: state shapes match `FastEngine.from_official_obs`. |
| 2 | `_apply_actions` + `_move_fleets_and_collide` — the heaviest piece. Translate combat resolution exactly. Parity-test with hand-crafted scenarios first (1 fleet vs 1 planet). |
| 3 | `_rotate_planets_and_sweep` + `_step_comets` + `_accumulate_production`. Parity test full step. |
| 4 | `play_episode` with `vmap`+`scan`. Parity test on 100 seeds × 500 turns. |
| 5 | Throughput benchmark on local GPU; tune `N_MAX_FLEETS` / `N_MAX_COMETS` / batch size. Target single-A100 5M+ steps/sec. |

## Risks

1. **Collision ordering** — numpy iterates fleets in a deterministic order; JAX-`vmap`'d ops are unordered within a batch. Fleets can't collide with each other directly (only with planets/comets), so the per-fleet collision is independent and `vmap`s cleanly. But ordering of "first-hit-wins" within a single fleet's path still matters — implement via `jnp.argmin(distance_to_intersection)` which is deterministic.

2. **Comet path pre-computation** — the official engine spawns comet paths via a retry loop (up to 300 iters per path). At pre-compute time we need the same expected distribution. Workaround: at `reset()` use a Python loop (NOT jit'd) to generate paths, pad to `MAX_PATH_LEN`. Costs ~10 ms once; never blocks the hot loop.

3. **Float32 vs Kaggle's float64** — the official engine uses float64 for some intercept math. Our `FastEngine` already runs float32 and parity-tests pass. JAX float32 should be safe.

4. **JAX install footprint** — adds ~600 MB to the venv (jax + jaxlib + cudnn). Not a problem on cloud, mildly annoying locally. Install via `uv sync --extra jax` (already declared in `pyproject.toml`).

## Integration points with the closed-loop pipeline

After JAX engine lands, [`tools/cloud/run_closed_loop.sh`](../tools/cloud/run_closed_loop.sh)'s self-play stage gains a `--jax-self-play` flag:

```bash
# Before: 1000 games × 3 min/game CPU = 50 hours
# After (with JAX): 1024-batch × 500 turns × 1 ms/step = ~9 min total
bash tools/cloud/run_closed_loop.sh --jax-self-play --games-per-iter 50000 ...
```

The MCTS itself stays on the CPU side (it's not the bottleneck for self-play; the game-step is). MCTS-with-NN-leaf-eval queries the trained NN over batches of MCTS-pending leaves; that batches well to GPU too via the existing `value_fn` interface.

## Success criteria

* Parity test passes on 1000 seeds × 500 turns.
* Single A100 sustained throughput ≥ 5M aggregate env steps/sec at batch 1024.
* `tools/cloud/run_closed_loop.sh --jax-self-play` produces demos byte-equivalent to the numpy path within 1e-4 absolute on visit_dist (allowing for FP rounding).
