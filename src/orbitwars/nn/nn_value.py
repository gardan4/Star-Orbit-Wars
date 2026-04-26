"""NN value-head bridge: ConvPolicy.value -> scalar leaf evaluation.

Mirrors ``nn_prior.py`` for the value head. Where ``move_prior_fn``
re-weights PUCT exploration via NN policy logits, ``value_fn``
replaces the ``_rollout_value`` heuristic-rollout with a single NN
forward pass that returns the value head's estimate of state-value
for ``my_player``.

Why this exists (see ``docs/NN_VALUE_HEAD_DESIGN.md``):

* ``move_prior_fn`` alone CANNOT change the wire action under
  anchor-locked play with heuristic rollouts. The Q values come from
  rollouts using HeuristicAgent on both sides → all candidates'
  Q ≈ "how the heuristic would play this from here," and the
  heuristic anchor wins on Q nearly every time.
* ``value_fn`` lets the NN drive the Q estimates directly. With a
  trained value head, Q reflects the NN's strategy assessment, not
  the heuristic's. This is the "value head as Q estimator" path
  (AlphaZero leaf evaluation, MuZero terminal value).

Status (2026-04-26): the BC v1 small checkpoint's value head was
NEVER TRAINED (``bc_warmstart.py`` does ``logits, _value = model(x)``
and discards _value). Plugging this module's ``make_nn_value_fn``
into MCTS today would feed garbage to the search — the value head
outputs ~uniform [-0.07, 0] noise for any input. Phase 2 of the
design doc covers training a useful value head.

This module is the Phase 1 deliverable: the *pathway* from value
head to MCTS leaf eval. Smoke-testable with constant or random
value_fn to verify wire actions diverge from a heuristic-rollout
baseline.

Convention: value is a scalar in [-1, 1] from ``my_player``'s
perspective. +1 = win, -1 = loss. Matches the ``_rollout_value``
sign convention so the two pathways are interchangeable in the
search.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import torch

from orbitwars.features.obs_encode import encode_grid
from orbitwars.nn.conv_policy import ConvPolicy, ConvPolicyCfg


# Public type: caller supplies (obs, my_player) and gets a scalar.
ValueFn = Callable[[Any, int], float]


def make_nn_value_fn(
    model: ConvPolicy,
    cfg: ConvPolicyCfg,
    *,
    clip: float = 1.0,
) -> ValueFn:
    """Build a value_fn closure over a loaded ConvPolicy.

    Args:
      model: a ``ConvPolicy`` checkpoint with both heads. The policy
        head's logits are ignored here; only the value head's scalar
        output is used.
      cfg: the matching ``ConvPolicyCfg`` used for grid dimensions.
      clip: max absolute value to return. The value head is in
        principle in [-1, 1] but training instability or distribution
        shift can produce out-of-range values; clipping keeps the
        downstream MCTS Q-aggregation well-behaved.

    Returns:
      A callable ``fn(obs, my_player: int) -> float`` that returns
      the NN's estimate of state-value for ``my_player``. Errors
      (encoding failures, NaN outputs) return 0.0 — neutral leaf
      value, search will still anchor-lock on the heuristic so a
      defective value_fn cannot forfeit a turn.
    """
    model.eval()  # idempotent; ensure dropout/BN are in eval mode

    def fn(obs: Any, my_player: int) -> float:
        try:
            grid = encode_grid(obs, my_player, cfg)
            x = torch.from_numpy(grid).unsqueeze(0)  # (1, C, H, W)
            with torch.no_grad():
                _logits, value = model(x)  # value: (1, 1) by ConvPolicy contract
            v = float(value.squeeze().item())
            if not np.isfinite(v):
                return 0.0
            # Clamp to [-clip, clip].
            if v > clip:
                return clip
            if v < -clip:
                return -clip
            return v
        except Exception:
            return 0.0

    return fn


def make_constant_value_fn(value: float = 0.0) -> ValueFn:
    """Diagnostic: return the same scalar for every state.

    Used by smoke tests to verify the value_fn pathway is wired up
    correctly. With ``value=0.0``, all candidates' Q estimates collapse
    to ~0; anchor-lock should make the heuristic anchor win on tie-
    breaking. This produces wire actions identical to a no-NN
    heuristic-rollout-only run if the pathway is wired correctly.
    """
    def fn(obs: Any, my_player: int) -> float:
        return float(value)
    return fn


def make_random_value_fn(seed: int = 0) -> ValueFn:
    """Diagnostic: per-call random value in [-1, 1].

    Used to confirm the value_fn actually steers Q estimates: wire
    actions under this fn MUST differ from a constant-value run.
    """
    import random as _r
    rng = _r.Random(seed)
    def fn(obs: Any, my_player: int) -> float:
        return rng.uniform(-1.0, 1.0)
    return fn
