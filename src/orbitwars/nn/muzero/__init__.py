"""MuZero-style learned dynamics for Orbit Wars.

See ``docs/MUZERO_SPEC.md`` for the full design rationale.

Three modules:
  * :mod:`representation` — ``h(obs) -> latent`` (encoder).
  * :mod:`prediction`     — ``f(latent) -> (policy, value)`` (heads).
  * :mod:`dynamics`       — ``g(latent, action) -> (next_latent, reward)``
                            (replaces ``engine.step`` in MCTS rollouts).

Plus :mod:`full_model` which composes them and provides MCTS-friendly
entry points (``initial_inference`` and ``recurrent_inference``).
"""

from orbitwars.nn.muzero.representation import RepresentationNet, RepresentationCfg
from orbitwars.nn.muzero.prediction import PredictionNet, PredictionCfg
from orbitwars.nn.muzero.dynamics import DynamicsNet, DynamicsCfg
from orbitwars.nn.muzero.full_model import MuZeroNet, MuZeroCfg

__all__ = [
    "RepresentationNet", "RepresentationCfg",
    "PredictionNet", "PredictionCfg",
    "DynamicsNet", "DynamicsCfg",
    "MuZeroNet", "MuZeroCfg",
]
