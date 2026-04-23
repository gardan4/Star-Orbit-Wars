"""JAX vmap/scan engine for offline self-play training (W4-5, Path C).

**Status**: SKELETON. The API surface is fixed; the step() internals are
stubs that raise ``NotImplementedError``. This file exists so W4 work can
start by filling in one hot loop at a time without re-litigating the
data-model decisions every time.

**Why JAX, not numpy?** `fast_engine.FastEngine` tops out at ~50-200k
steps/sec single-core (good enough for MCTS rollouts at 1 s/turn, useless
for PPO). Under `jax.jit` + `jax.vmap` + `jax.lax.scan` on a T4/A100 we
target **1M-20M aggregate env steps/sec** for self-play. That requires:
  * All shapes static across the whole episode (no growing/shrinking arrays).
  * Pure-functional step (no mutation, no Python control flow on state values).
  * Randomness threaded explicitly via ``jax.random.PRNGKey``.

**Design decisions — load-bearing, read before changing:**

1. Fixed-capacity Structure-of-Arrays. Planets / fleets / comets live in
   `N_MAX`-sized arrays with an `alive` mask. Nothing is ever compacted;
   dead slots are simply masked out. This is mandatory for ``vmap`` to
   produce identically-shaped state trees across seeds.

2. No parity with ``kaggle_environments.envs.orbit_wars``. The training
   engine only needs to be **self-consistent** and have a reasonable
   inductive bias for the real game. Parity-matched rollouts go through
   ``fast_engine`` (which already has a 1000-seed parity gate).

3. Comet paths pre-computed at ``reset()``. Drops the retry-up-to-300-iters
   pattern from the reference engine — one-shot batched sampler at
   episode start. No runtime spawn branching.

4. Actions as fixed-shape tensors: ``(num_agents, MAX_LAUNCHES_PER_TURN, 3)``
   with a validity bit. MCTS / PPO policies emit padded action tensors;
   the engine uses mask-aware inserts into the first free fleet slots.

5. JAX PRNG threaded through state. ``state.key`` splits at every ``step``
   so vmap over seeds is trivially parallel and reproducible.

6. dtype ``float32`` everywhere (best T4/A100 throughput); ``int32`` for
   counts and slot indices.

**Scope of v0 (this file):**
  * `State` pytree (frozen dataclass registered with JAX's pytree registry).
  * `StaticCfg` struct — capacity constants + episode length, hashable so
    ``jit`` doesn't recompile on every call.
  * `reset(key, cfg) -> State` stub.
  * `step(state, actions) -> (state, reward, done)` stub.
  * Fully typed; every field carries a comment explaining shape + dtype.

**Scope NOT in v0:**
  * Any actual physics — all step() internals are stubs.
  * JIT / vmap wrappers — those live in ``src/orbitwars/train/jax_rollout.py``
    (also not yet created; see module docstring).
  * PPO loop — ``src/orbitwars/train/ppo_jax.py``.

**Dependencies:** ``jax`` is NOT currently in ``requirements.txt``. This
file imports ``numpy`` as a stand-in so the module is importable today
and the API can be unit-tested. W4 swap is a single find-replace of the
``_xp`` alias to ``jax.numpy``. All shape / semantics decisions translate
1:1 because we constrained ourselves to array ops ``jnp`` also supports.

**Reference**: see ``docs/STATUS.md`` §5 for the full plan and the
``fast_engine.py`` module for the numpy analogue of most operations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple

import numpy as _np


# ---------------------------------------------------------------------------
# The ``_xp`` alias. In W4 this becomes ``import jax.numpy as _xp``. For v0
# we use plain numpy so the module imports cleanly on the dev machine.
# ``jax.random`` analogues go via ``_rand`` — a minimal shim that will be
# swapped to ``jax.random`` in W4. Anything NOT covered by ``_xp`` / ``_rand``
# (e.g. ``jax.lax.scan``) is deferred to the rollout module.
# ---------------------------------------------------------------------------
_xp = _np


class _NumpyRandShim:
    """Tiny stand-in for the subset of ``jax.random`` we use at reset-time."""

    def key(self, seed: int):
        # JAX returns a (2,) uint32 array; we use a simple int for numpy.
        return _np.random.default_rng(int(seed))

    def split(self, key, num: int = 2):
        # For numpy the "split" returns independent generators seeded from
        # the parent's next draw. Keeps the API shape identical.
        children = []
        for _ in range(num):
            seed = int(key.integers(0, 2**31 - 1))
            children.append(_np.random.default_rng(seed))
        return children

    def uniform(self, key, shape=(), minval: float = 0.0, maxval: float = 1.0):
        return key.uniform(low=minval, high=maxval, size=shape).astype(_np.float32)

    def randint(self, key, shape, minval: int, maxval: int):
        return key.integers(low=minval, high=maxval, size=shape).astype(_np.int32)


_rand = _NumpyRandShim()


# ---------------------------------------------------------------------------
# Capacity constants. Tuned empirically against the 1000-seed parity gate:
#   * planets per game: observed 20–40; ``N_MAX_PLANETS=48`` has 20%
#     headroom above the observed max.
#   * fleets per game: observed peak ~80; ``N_MAX_FLEETS=256`` covers 3×
#     that for late-game swarms.
#   * comets: reference caps at 5 spawn windows × 4 quadrants = 20 per
#     game but at any instant usually ≤ 8. ``C_MAX=12`` is cautious.
#   * comet path length: reference paths observed 60–115 steps;
#     ``T_PATH_MAX=128`` pads to a power-of-2.
#   * per-turn launches per agent: heuristic emits ≤ 1 launch per owned
#     planet per turn → ≤ N_MAX_PLANETS per agent. ``MAX_LAUNCHES_PER_TURN=48``
#     matches N_MAX_PLANETS (we never launch more than we have planets).
# Any capacity that is exceeded at runtime should raise loudly from
# ``reset`` — silently truncating would produce invisibly-wrong training
# data.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StaticCfg:
    """Hashable, immutable per-experiment constants.

    Must be hashable because JAX's ``jit`` uses it as a static argument;
    changing any field forces a recompile.
    """

    # Capacity (see module comment).
    n_max_planets: int = 48
    n_max_fleets: int = 256
    c_max: int = 12
    t_path_max: int = 128
    max_launches_per_turn: int = 48

    # Game rules mirroring the reference engine.
    num_agents: int = 2                      # 2 or 4
    episode_steps: int = 500                 # matches orbit_wars.py DEFAULT_STEPS
    board_width: float = 100.0
    board_height: float = 100.0
    sun_x: float = 50.0
    sun_y: float = 50.0
    sun_radius: float = 10.0

    # Reward shaping knobs (tuned in W4 PPO).
    # Dense per-turn reward = shaping_weight * delta_ship_lead; terminal
    # reward adds ``terminal_bonus`` on the winner. Set shaping_weight=0
    # for sparse-only.
    shaping_weight: float = 0.0
    terminal_bonus: float = 1.0


# ---------------------------------------------------------------------------
# State pytree.
#
# *Every* leaf is an array with static shape and static dtype. The shapes
# are fully determined by ``StaticCfg``; no Python-list / dict leaves allowed
# (those break vmap and scan).
# ---------------------------------------------------------------------------
@dataclass
class State:
    """Full game state as a JAX pytree (once ``_xp`` swaps to jnp).

    Planet SoA (shape ``[n_max_planets]`` unless noted):
      p_alive        bool       slot occupied by a real planet
      p_owner        int32      -1 for neutral, else 0..num_agents-1
      p_x, p_y       float32    absolute positions (orbiting planets updated each step)
      p_radius       float32
      p_ships        int32
      p_production   int32      ships added per turn when owned
      p_is_rotating  bool       true iff orbital_r + radius < 50 (per ref rule)
      p_init_angle   float32    angle at step 0 for orbital reconstruction
      p_omega        float32    rad/turn, in [0.025, 0.05]; 0 when not rotating
      p_orbit_cx     float32    center-of-orbit x (usually the sun)
      p_orbit_cy     float32    center-of-orbit y
      p_orbit_r      float32    orbital radius

    Fleet SoA (shape ``[n_max_fleets]``):
      f_alive        bool
      f_owner        int32      0..num_agents-1
      f_x, f_y       float32
      f_angle        float32    heading in radians
      f_from_slot    int32      planet slot this fleet launched from (not pid)
      f_ships        int32      larger = faster (see ref speed formula)

    Comet SoA (shape ``[c_max]``):
      c_alive        bool
      c_planet_slot  int32      planet slot of the moving comet-bearing planet
      c_paths_xy     float32[c_max, t_path_max, 2]
      c_path_len     int32      valid length along t_path_max
      c_path_index   int32      current position along the path

    Scalars:
      step        int32
      key         PRNG key (parent for all RNG this turn)
      reward_acc  float32[num_agents]  cumulative shaping since reset

    Derived (recomputed each step, not carried):
      combat_accum  int32[n_max_planets, num_agents]  ships reaching this planet this turn
    """

    # Planet SoA
    p_alive: Any
    p_owner: Any
    p_x: Any
    p_y: Any
    p_radius: Any
    p_ships: Any
    p_production: Any
    p_is_rotating: Any
    p_init_angle: Any
    p_omega: Any
    p_orbit_cx: Any
    p_orbit_cy: Any
    p_orbit_r: Any

    # Fleet SoA
    f_alive: Any
    f_owner: Any
    f_x: Any
    f_y: Any
    f_angle: Any
    f_from_slot: Any
    f_ships: Any

    # Comet SoA
    c_alive: Any
    c_planet_slot: Any
    c_paths_xy: Any
    c_path_len: Any
    c_path_index: Any

    # Scalars / aggregates
    step: Any
    key: Any
    reward_acc: Any


# ---------------------------------------------------------------------------
# reset / step stubs.
#
# reset() is a real generator — it produces a StaticCfg-shaped State with
# valid initial data (random planet layout within the mirror-symmetry
# constraint, fleets empty, comets pre-scheduled).
#
# step() is explicitly a stub. Flesh out one hot loop at a time:
#   1. maybe_spawn_comet (consume pre-computed path at spawn step).
#   2. apply_actions (mask-gated fleet insert into first-free slots).
#   3. production (p_ships += p_prod on owned planets).
#   4. move_fleets_collide (fleet movement + all-pairs planet distance).
#   5. rotate_planets_sweep (orbital update + planet-hits-fleet sweep).
#   6. move_comets_sweep (comet update + comet-hits-fleet sweep).
#   7. resolve_combat (per-planet top-2 reduction, survivor vs garrison).
#   8. compute_reward_delta (dense shaping + sparse terminal bonus).
#
# Each of the 8 numbered items above gets its own PR; by the time all 8
# are real, we can delete the NotImplementedError and run PPO.
# ---------------------------------------------------------------------------

def reset(key, cfg: StaticCfg) -> State:
    """Produce an initial ``State`` with random planet layout + comet paths.

    Deterministic given ``key``. Respects the reference engine's
    **mirror-symmetry constraint**: 2-player games are mirrored through
    (sun_x, sun_y); 4-player games have 4-fold symmetry. Initial planet
    count is drawn from ``U(20, min(40, n_max_planets))``; excess slots
    have ``p_alive=False``.

    TODO(W4):
      * Implement mirror-symmetric random planet placement.
      * Implement ``_generate_comet_paths_batched`` — fixed 16-candidate
        batched sampler replacing the reference's 300-iter retry loop.
      * Seed the ``c_*`` arrays with the pre-scheduled spawn path for
        each of the 5 spawn windows × num-quadrants slots; set
        ``c_alive=False`` until spawn step.
    """
    del key, cfg  # unused in stub
    raise NotImplementedError(
        "jax_engine.reset: TODO(W4) — see module docstring step 0"
    )


def step(state: State, actions: Any, cfg: StaticCfg) -> Tuple[State, Any, Any]:
    """Advance one turn.

    Args:
      state: current State.
      actions: ``int32[num_agents, max_launches_per_turn, 4]`` tensor.
        Last axis is ``(validity_bit, from_slot, angle_scaled, ships)``.
        ``angle_scaled`` is ``int32`` in ``[0, 65536)`` mapping to
        ``[0, 2*pi)`` — integer encoding keeps the action tensor pure
        ``int32`` which plays better with policy-network output heads.
      cfg: static capacities / rules.

    Returns:
      next_state: State with ``step`` incremented.
      reward: ``float32[num_agents]`` per-agent reward for this turn.
      done: ``bool`` scalar — True iff this was the last turn of the episode.

    TODO(W4): all eight numbered hot loops above.
    """
    del state, actions, cfg  # unused in stub
    raise NotImplementedError(
        "jax_engine.step: TODO(W4) — see module docstring steps 1-8"
    )


# ---------------------------------------------------------------------------
# Utility helpers (placeholders).
#
# These are pure functions that will be JIT-friendly once bodies are real.
# Keeping them as module-level functions (not State methods) matches the
# JAX-functional style — no hidden ``self``.
# ---------------------------------------------------------------------------

def compute_scores(state: State, cfg: StaticCfg):
    """Per-agent total ships (on owned planets + in fleets).

    This is the scoreboard that decides the terminal winner. Cheap: two
    segment sums over alive masks, so safe to call every turn if needed.

    TODO(W4): implement. Should be a ~5-line ``_xp.where`` + ``sum``.
    """
    del state, cfg
    raise NotImplementedError("jax_engine.compute_scores: TODO(W4)")


def observation(state: State, cfg: StaticCfg, agent_id: int):
    """Return the per-agent observation view.

    In orbit_wars the game is fully observable, so this is just a perspective
    swap (``player`` field) — no hiding. For NN input we'll probably encode
    a different feature tensor anyway (see ``features/obs_encode.py``).

    TODO(W4): decide whether to emit the raw state subset or pre-encoded
    features here. Leaning toward: raw subset + a separate
    ``features.jax_obs_encode(state, agent_id, cfg)`` that produces the
    policy-network input.
    """
    del state, cfg, agent_id
    raise NotImplementedError("jax_engine.observation: TODO(W4)")
