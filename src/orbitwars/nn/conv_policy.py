"""Centralized per-entity-grid conv policy for Orbit Wars (W4 candidate A).

**Status**: SKELETON. Architecture decisions frozen; forward pass body is
stubbed. This is the *primary* candidate for the W4 architecture bake-off
per the plan — Set-Transformer (see ``set_transformer.py``) is candidate
B. Pick winner by 1M-step training result, not a priori.

**Why this architecture?** Lux S1/S3 winning submissions used centralized
per-entity-grid conv policies over a dense spatial tensor. The recipe:

  1. Render the game state onto a ``(C, H, W)`` grid where each channel
     is one entity feature (ships_owned_0, ships_owned_1, production,
     is_comet, sun_distance, ...). H=W=50 or 100 gives a spatial
     resolution that captures intercept geometry without blowing up
     the activation volume.
  2. Run a small conv backbone (4-8 residual blocks, 64-128 channels).
  3. Emit two heads:
       * **Policy head**: per-planet action distribution, decoded by
         indexing the output grid at each owned planet's (x, y) slot.
       * **Value head**: scalar game value via global average pool.

**Why it beats a flat MLP / set-transformer on RTS:**
  * Spatial locality is the dominant structure (nearby planets interact,
    far planets don't). Conv's inductive bias matches the game.
  * Translation equivariance on the mirror-symmetric board is free data
    augmentation: a conv filter trained in one quadrant generalizes
    automatically to the other three.
  * O(H*W) compute vs O(N^2) attention — cheaper at N=40 planets and
    scales better if we move to 100+ entity boards in later iterations.

**Parameter budget:**
  * Target: <2M params total after distillation (W5 deliverable —
    Bayesian Policy Distillation to <2M student).
  * W4 teacher: 5-20M params is fine; student is what ships.

**Training pipeline (not in this file):**
  * Feature encoding: ``orbitwars.features.obs_encode`` (stub today).
  * PPO loop: ``orbitwars.train.ppo_jax`` (not yet created).
  * PFSP opponent pool: ``orbitwars.training.pfsp_pool`` (not yet created).
  * Distillation: ``orbitwars.nn.distill`` (not yet created).

**Dependencies:** ``torch`` 2.11.0+cpu is installed as of W3 tail (CPU-only
for skeleton work; GPU build deferred to W4 when training starts). The
torch module definitions below are still commented out — they will be
enabled once the feature encoder (``features/obs_encode.py``) and the
action decoder are in place, so the first import error surfaces at the
plumbing layer rather than at module-load.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ConvPolicyCfg:
    """Hyperparameters for the conv policy.

    Load-bearing values picked from Lux S3 winner analysis:
      * `grid_h=grid_w=50`: half the 100x100 board. Each cell covers a
        2x2 unit area — fine enough to localize planets (radius 1-3) and
        fleets, coarse enough that the activation volume stays manageable
        (50x50 @ 64 channels = 160KB per layer).
      * `n_channels=12`: see ``feature_channels()`` for the breakdown.
      * `backbone_channels=64` / `n_blocks=6`: ~1M params at H=W=50.
        Distills cleanly to <2M final; fits within the W4 GPU budget
        (1-2 days training on one T4).
      * `n_action_channels=8`: per-planet action distribution — 4 angle
        buckets x 2 ship-fraction buckets. Continuous angle gets
        re-introduced via BOKR-style sampling around the arg-max angle
        at inference time (see ``mcts/bokr_widen.py``).
    """

    grid_h: int = 50
    grid_w: int = 50
    n_channels: int = 12              # input feature channels
    backbone_channels: int = 64
    n_blocks: int = 6
    n_action_channels: int = 8        # per-cell action distribution
    value_hidden: int = 128


def feature_channels() -> Tuple[str, ...]:
    """Documented list of the ``n_channels`` input features.

    The feature tensor shape is ``(batch, C, H, W)`` where ``C = len(...)``.
    Order is load-bearing — the encoder in ``features/obs_encode.py``
    and the decoder heads must agree. Adding a channel requires a
    retrain; prefer slotting unused fields in at the END rather than
    reordering.
    """
    return (
        "ship_count_p0",           # 0. my-side ship density (sqrt-scaled)
        "ship_count_p1",           # 1. enemy ship density (sqrt-scaled)
        "production_p0",           # 2. owned-planet production rate, mine
        "production_p1",           # 3. owned-planet production rate, enemy
        "production_neutral",      # 4. neutral planet production
        "planet_radius",           # 5. planet radius at cell (or 0)
        "is_orbiting",             # 6. 1 if rotating planet occupies cell
        "is_comet",                # 7. 1 if comet-bearing planet
        "sun_distance",            # 8. pre-computed distance to (50,50)
        "fleet_angle_cos",         # 9. cos(angle) of any fleet at cell
        "fleet_angle_sin",         # 10. sin(angle)
        "turn_phase",              # 11. step / 500 (broadcast scalar)
    )


# ---------------------------------------------------------------------------
# Torch module skeleton — body stubbed, signatures pinned.
#
# torch 2.11.0+cpu is installed. The block below stays commented until the
# surrounding glue (``features/obs_encode.py``, action decoder) lands so we
# don't carry a half-wired module. W4 switchover: uncomment, replace the
# ``nn.BatchNorm2d`` with ``nn.GroupNorm`` if batch size < 16 is expected
# at inference, and instantiate via ``ConvPolicy(ConvPolicyCfg())``.
# ---------------------------------------------------------------------------

# TODO(W4): uncomment and implement.
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class ResBlock(nn.Module):
#     """Standard 3x3 conv residual block, pre-activation variant."""
#     def __init__(self, c: int):
#         super().__init__()
#         self.bn1 = nn.BatchNorm2d(c)
#         self.conv1 = nn.Conv2d(c, c, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(c)
#         self.conv2 = nn.Conv2d(c, c, 3, padding=1)
#
#     def forward(self, x):
#         y = self.conv1(F.relu(self.bn1(x)))
#         y = self.conv2(F.relu(self.bn2(y)))
#         return x + y
#
# class ConvPolicy(nn.Module):
#     """Centralized per-entity-grid conv policy + value network."""
#     def __init__(self, cfg: ConvPolicyCfg):
#         super().__init__()
#         self.cfg = cfg
#         self.stem = nn.Conv2d(cfg.n_channels, cfg.backbone_channels, 3, padding=1)
#         self.blocks = nn.ModuleList([ResBlock(cfg.backbone_channels)
#                                      for _ in range(cfg.n_blocks)])
#         self.policy_head = nn.Conv2d(cfg.backbone_channels,
#                                      cfg.n_action_channels, 1)
#         self.value_head = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(cfg.backbone_channels, cfg.value_hidden),
#             nn.ReLU(),
#             nn.Linear(cfg.value_hidden, 1),
#             nn.Tanh(),
#         )
#
#     def forward(self, x):
#         """Args:
#           x: (B, n_channels, H, W) input grid.
#         Returns:
#           policy_logits: (B, n_action_channels, H, W) per-cell action logits.
#           value: (B, 1) scalar game value in [-1, 1].
#         """
#         h = self.stem(x)
#         for block in self.blocks:
#             h = block(h)
#         policy = self.policy_head(h)
#         value = self.value_head(h)
#         return policy, value


def param_count_estimate(cfg: ConvPolicyCfg) -> int:
    """Rough parameter count sanity-check.

    Dominant terms:
      * Stem: C_in * C * 9
      * Each ResBlock: 2 * (C * C * 9)
      * Policy head: C * n_action_channels
      * Value head: C * value_hidden + value_hidden

    Returns the integer estimate. Useful for gate-checks like "student
    must be <2M params" without actually constructing the torch module.
    """
    c = cfg.backbone_channels
    c_in = cfg.n_channels
    stem = c_in * c * 9
    blocks = cfg.n_blocks * 2 * c * c * 9
    policy = c * cfg.n_action_channels
    value = c * cfg.value_hidden + cfg.value_hidden
    # Add ~10% for biases + batchnorm scale/shift.
    total = int((stem + blocks + policy + value) * 1.10)
    return total


# ---------------------------------------------------------------------------
# Decode helpers — convert policy grid -> per-planet action distribution.
#
# The NN emits a (C_action, H, W) grid. At inference time we read it at
# the (grid_x, grid_y) of each owned planet: `probs[C_action] = softmax(logits)`.
# Then map the C_action index to (angle_bucket, ship_fraction) via
# ``ACTION_LOOKUP`` below. BOKR-style continuous-angle refinement happens
# AFTER decode (mcts/bokr_widen.py expands the top-k angles into a fine
# grid and runs MCTS over them).
# ---------------------------------------------------------------------------

# n_action_channels = 4 angle buckets x 2 ship fractions = 8 channels.
# angle_bucket_0..3 = {0, pi/2, pi, 3pi/2} with BOKR expanding to the
# continuous angle at inference. ship_frac_0 = 50%, ship_frac_1 = 100%.
# Quarter-angle resolution is coarse on purpose: intercepts land within
# +/- ~90 deg of the target direction, so each of the 4 buckets maps
# cleanly to "toward quadrant X". BOKR refines.
ACTION_LOOKUP = (
    # (angle_bucket, ship_frac)  description
    (0, 0.5),  # 0: East, 50%
    (0, 1.0),  # 1: East, 100%
    (1, 0.5),  # 2: North, 50%
    (1, 1.0),  # 3: North, 100%
    (2, 0.5),  # 4: West, 50%
    (2, 1.0),  # 5: West, 100%
    (3, 0.5),  # 6: South, 50%
    (3, 1.0),  # 7: South, 100%
)


def planet_to_grid_coords(
    x: float, y: float, cfg: ConvPolicyCfg
) -> Tuple[int, int]:
    """Map continuous planet position -> (grid_y, grid_x) cell index.

    Board is ``[0, 100] x [0, 100]``; grid is ``[0, grid_h] x [0, grid_w]``.
    Standard conv convention uses (row, col) i.e. (y, x) order.

    Clamps to ``[0, grid_h-1]`` / ``[0, grid_w-1]`` to handle edge cases
    where a planet sits exactly on the boundary.
    """
    gy = int(y * cfg.grid_h / 100.0)
    gx = int(x * cfg.grid_w / 100.0)
    gy = max(0, min(cfg.grid_h - 1, gy))
    gx = max(0, min(cfg.grid_w - 1, gx))
    return gy, gx
