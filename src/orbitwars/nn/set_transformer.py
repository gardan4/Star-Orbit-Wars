"""Set-Transformer with relative-distance attention bias (W4 candidate B).

**Status**: SKELETON. Architecture pinned; forward-pass bodies stubbed in
commented torch blocks exactly like ``conv_policy.py``. This is candidate
*B* in the W4 bake-off — the centralized per-entity-grid conv in
``conv_policy.py`` is candidate A. Pick winner by 1M-step PPO win-rate,
not a priori.

**Why set-transformer over conv for Orbit Wars** (the case for B):

  1. Orbit Wars is a *set*, not a grid. Planets + fleets + comets are
     discrete entities with continuous positions, not dense spatial
     observations. A 50x50 conv grid wastes 99% of its cells (20-40
     planets + 50-200 fleets = <5% fill factor).
  2. Variable N generalizes cleanly — a conv must decide ``grid_h`` at
     training time and is stuck with it; a set-transformer handles
     25-planet and 40-planet boards with the same weights.
  3. Pairwise interactions are first-class: attention A_{ij} directly
     models "how much should planet i care about planet j", which is
     exactly the fleet-intercept calculus of the game. A conv only
     reasons locally via receptive field, and 3x3 kernels don't see
     the 80-unit diagonal of the board until 30+ layers deep.
  4. Relative-distance attention bias (Ji et al., 2024 and the
     arXiv 2504.08195 reference from the plan) bakes euclidean distance
     into the attention logits: ``logits_ij += -w * dist(pos_i, pos_j)``.
     Nearby entities attend more strongly. This is the structural prior
     the game rewards without forcing the model to learn it from scratch.

**Why conv might still win** (the case for A):

  * Lux S1/S3 winners used conv. Empirical precedent matters.
  * Spatial translation equivariance is free data augmentation under
     the 4-fold mirror symmetry; attention gets it only via explicit
     position-embedding augmentation.
  * Compute cost: conv is O(C^2 * H * W), attention is O(N^2 * d).
     At N=300 entities with d=128, attention is ~10x more FLOPs than
     a comparable-capacity conv.

**Verdict: bake-off decides.** Train both to 1M PPO steps vs the same
PFSP pool; ship whichever has higher Elo. Honest W5 decision gate.

**Parameter budget:**
  * Target: <2M params final (W5 deliverable — distill to student).
  * Set-transformer teacher: 1-5M is fine.

**ISAB vs SAB**: ISAB (Inducing-point Self-Attention Block, Lee et al.
2019) is O(N*m) for m inducing points instead of SAB's O(N^2). At our
N<=300 entities, SAB is faster in wall-clock despite the asymptotic
disadvantage. Plan explicitly drops ISAB; if later iterations push N
into the low-thousands, Perceiver-IO is the upgrade path (not ISAB).

**Dependencies:** ``torch`` 2.11.0+cpu is installed. Torch blocks stay
commented until the feature encoder and action decoder glue lands.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SetTransformerCfg:
    """Hyperparameters for the set-transformer policy.

    Load-bearing values:
      * ``d_model=128``: matches conv_policy.backbone_channels=64 by
        roughly doubling (attention layers have ~2x params per unit
        width vs conv because of Q/K/V projections).
      * ``n_heads=4``: standard; 128/4=32 head-dim. Higher head-count
        hurts wall-clock at our N.
      * ``n_blocks=4``: each block is SAB(attn → MLP → residual). 4 blocks
        gives ~8x receptive-field equivalent (every entity has attended
        to every other entity 4 times). Matches conv's 6-block depth in
        expressive power.
      * ``mlp_ratio=2.0``: MLP hidden = 2 * d_model. Smaller than ViT's
        4x because our N<300 is attention-compute-bound, not MLP-bound.
      * ``n_max_entities=320``: upper-bound buffer (40 planets + 256
        fleets + 5 comets + 4 team-summary tokens + slack). Static shape
        for JAX vmap / torch.compile. Mask excess via attention mask.
      * ``n_fourier_bands=8``: Fourier features for (x, y) coords. 8
        bands at frequencies 2^{0..7} cover board scale 1-100 units
        evenly.
      * ``dist_bias_init=0.02``: initial per-head scale for the
        relative-distance attention bias. Learned per-head; small init
        so the bias adds to stock attention rather than swamping it.
      * ``n_action_channels=8``: same decoding as conv_policy — 4 angle
        buckets x 2 ship-fraction buckets.
    """

    d_model: int = 128
    n_heads: int = 4
    n_blocks: int = 4
    mlp_ratio: float = 2.0
    n_max_entities: int = 320
    n_fourier_bands: int = 8
    dist_bias_init: float = 0.02
    n_action_channels: int = 8
    value_hidden: int = 128


def entity_feature_schema() -> Tuple[str, ...]:
    """Documented per-entity input features.

    The feature tensor is ``(batch, n_max_entities, F)`` where
    ``F = len(entity_feature_schema())``. Padding entities get
    ``is_valid=0`` and are masked out of attention.

    Order is load-bearing — ``features/obs_encode.py`` must agree.
    Adding a feature requires retraining; prefer slotting in at the
    end rather than reordering.

    A note on coordinates: ``pos_x`` / ``pos_y`` are the raw continuous
    positions in [0, 100]. They are NOT Fourier-encoded here; the model
    applies Fourier encoding as the first learned layer, which lets
    the learned bands differ from the fixed conv-style grid.
    """
    return (
        # --- Identity ---
        "is_valid",            # 0. 1 if entity is present; 0 for padding
        "type_planet",         # 1. one-hot: entity is a planet
        "type_fleet",          # 2. one-hot: entity is a fleet
        "type_comet",          # 3. one-hot: entity is a comet (static-pos)
        "owner_me",            # 4. 1 if owned by agent perspective
        "owner_enemy",         # 5. 1 if owned by enemy
        "owner_neutral",       # 6. 1 if neutral (planets/comets only)
        # --- Kinematics ---
        "pos_x",               # 7. x in [0, 100] — Fourier-encoded later
        "pos_y",               # 8. y in [0, 100]
        "velocity_x",          # 9. fleet vx (0 for planets)
        "velocity_y",          # 10. fleet vy (0 for planets)
        "is_orbiting",         # 11. planet orbits sun (moves predictably)
        "orbital_angular_vel", # 12. rad/turn (0 for static entities)
        # --- Capacity / economy ---
        "ships",               # 13. sqrt-scaled ship count
        "production",          # 14. ships/turn (0 for fleets)
        "radius",              # 15. planet radius (0 for fleets)
        "sun_distance",        # 16. pre-computed |pos - (50,50)|
        # --- Global broadcast (same value across all entity rows) ---
        "turn_phase",          # 17. step / 500 — broadcast scalar
        "score_diff",          # 18. (my_ships - enemy_ships) / 1000
    )


# ---------------------------------------------------------------------------
# Torch module skeleton — body stubbed, signatures pinned.
#
# Fill in when the obs encoder lands. The key design elements:
#   * Fourier positional encoding applied to pos_x, pos_y -> 2*n_bands extra
#     dims, concatenated to the raw feature vector before the stem linear.
#   * Relative-distance attention bias: compute pairwise euclidean distances
#     once per forward pass, broadcast-add to attention logits per head
#     with a learned per-head scale.
#   * Entity mask: standard additive mask (-inf on padding rows) so
#     padding entities get zero attention weight.
#   * Two heads:
#       - Policy: per-entity linear -> (n_max_entities, n_action_channels).
#         At decode time we only read the rows corresponding to owned
#         planets (type_planet=1 & owner_me=1).
#       - Value: masked-mean pool over valid entities -> MLP -> scalar.
# ---------------------------------------------------------------------------

# TODO(W4): uncomment and implement.
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# def _fourier_encode(coords: torch.Tensor, n_bands: int) -> torch.Tensor:
#     """Standard NeRF-style sinusoidal encoding.
#
#     Args:
#       coords: (..., 2) raw (x, y) in [0, 100].
#       n_bands: number of frequency bands.
#
#     Returns:
#       (..., 4 * n_bands) tensor of [sin(2^k * x), cos(2^k * x),
#       sin(2^k * y), cos(2^k * y)] for k=0..n_bands-1. Coords are
#       normalized to [-1, 1] first so the period of the highest
#       band covers the whole board.
#     """
#     # Normalize [0, 100] -> [-1, 1]
#     xy = (coords / 50.0) - 1.0
#     freqs = 2.0 ** torch.arange(n_bands, device=coords.device, dtype=coords.dtype)
#     # (..., 2, n_bands)
#     scaled = xy.unsqueeze(-1) * freqs * torch.pi
#     sin = torch.sin(scaled)
#     cos = torch.cos(scaled)
#     # Flatten last two dims
#     return torch.cat([sin, cos], dim=-1).flatten(-2, -1)
#
# class SABWithDistBias(nn.Module):
#     """Set Attention Block with learned relative-distance bias.
#
#     Forward:
#       1. Multi-head self-attention with mask + per-head distance bias.
#       2. Residual + LayerNorm.
#       3. MLP with GELU + residual + LayerNorm.
#     """
#     def __init__(self, d_model: int, n_heads: int, mlp_ratio: float,
#                  dist_bias_init: float):
#         super().__init__()
#         self.n_heads = n_heads
#         self.qkv = nn.Linear(d_model, 3 * d_model)
#         self.proj = nn.Linear(d_model, d_model)
#         # Per-head learned scale for the distance bias. Initialized small.
#         self.dist_scale = nn.Parameter(
#             torch.full((n_heads,), dist_bias_init)
#         )
#         self.ln1 = nn.LayerNorm(d_model)
#         self.ln2 = nn.LayerNorm(d_model)
#         hidden = int(d_model * mlp_ratio)
#         self.mlp = nn.Sequential(
#             nn.Linear(d_model, hidden),
#             nn.GELU(),
#             nn.Linear(hidden, d_model),
#         )
#
#     def forward(self, x: torch.Tensor, pairwise_dist: torch.Tensor,
#                 mask: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#           x: (B, N, d_model) entity embeddings.
#           pairwise_dist: (B, N, N) precomputed euclidean distances.
#           mask: (B, N) bool; True = valid, False = padding.
#
#         Returns:
#           (B, N, d_model) updated embeddings. Padded rows retain input.
#         """
#         B, N, D = x.shape
#         H = self.n_heads
#         qkv = self.qkv(x).reshape(B, N, 3, H, D // H).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, N, d_head)
#         attn = (q @ k.transpose(-2, -1)) / (D // H) ** 0.5  # (B, H, N, N)
#         # Add per-head negative-distance bias.
#         attn = attn - self.dist_scale.view(1, H, 1, 1) * pairwise_dist.unsqueeze(1)
#         # Mask invalid keys.
#         mask_kv = mask.view(B, 1, 1, N)  # (B, 1, 1, N)
#         attn = attn.masked_fill(~mask_kv, float("-inf"))
#         attn = F.softmax(attn, dim=-1)
#         out = (attn @ v).transpose(1, 2).reshape(B, N, D)
#         x = self.ln1(x + self.proj(out))
#         x = self.ln2(x + self.mlp(x))
#         return x
#
# class SetTransformerPolicy(nn.Module):
#     """Set-transformer policy + value network.
#
#     Inputs are per-entity feature vectors; outputs are per-entity
#     policy logits and a scalar value. Read owned-planet rows from
#     the policy output at decode time.
#     """
#     def __init__(self, cfg: SetTransformerCfg, n_features: int):
#         super().__init__()
#         self.cfg = cfg
#         # Stem: raw features + Fourier(pos) -> d_model.
#         fourier_dim = 4 * cfg.n_fourier_bands
#         self.stem = nn.Linear(n_features + fourier_dim, cfg.d_model)
#         self.blocks = nn.ModuleList([
#             SABWithDistBias(cfg.d_model, cfg.n_heads, cfg.mlp_ratio,
#                             cfg.dist_bias_init)
#             for _ in range(cfg.n_blocks)
#         ])
#         self.policy_head = nn.Linear(cfg.d_model, cfg.n_action_channels)
#         self.value_head = nn.Sequential(
#             nn.Linear(cfg.d_model, cfg.value_hidden),
#             nn.ReLU(),
#             nn.Linear(cfg.value_hidden, 1),
#             nn.Tanh(),
#         )
#
#     def forward(self, features: torch.Tensor, mask: torch.Tensor):
#         """
#         Args:
#           features: (B, N, F) entity features. pos_x / pos_y are
#             at fixed offsets (see entity_feature_schema) and are
#             used both as raw features AND Fourier-encoded.
#           mask: (B, N) bool; True = valid.
#
#         Returns:
#           policy_logits: (B, N, n_action_channels) per-entity logits.
#             Decode by indexing owned-planet rows.
#           value: (B, 1) scalar in [-1, 1].
#         """
#         # Fixed offsets from entity_feature_schema.
#         POS_X, POS_Y = 7, 8
#         pos = features[..., [POS_X, POS_Y]]  # (B, N, 2)
#         fourier = _fourier_encode(pos, self.cfg.n_fourier_bands)
#         x = torch.cat([features, fourier], dim=-1)
#         x = self.stem(x)
#         # Precompute pairwise distance once per forward pass.
#         pairwise_dist = torch.cdist(pos, pos, p=2)  # (B, N, N)
#         for block in self.blocks:
#             x = block(x, pairwise_dist, mask)
#         policy = self.policy_head(x)
#         # Masked mean pool for value.
#         m = mask.float().unsqueeze(-1)  # (B, N, 1)
#         pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
#         value = self.value_head(pooled)
#         return policy, value


def param_count_estimate(cfg: SetTransformerCfg, n_features: int = 19) -> int:
    """Rough parameter count sanity-check.

    Dominant terms:
      * Stem linear: (n_features + 4*n_fourier_bands) * d_model
      * Each SAB block:
          - QKV projection: 3 * d_model * d_model
          - Output projection: d_model * d_model
          - Per-head dist_scale: n_heads (negligible)
          - LN x 2: 2 * d_model (negligible)
          - MLP: 2 * d_model * (d_model * mlp_ratio)
      * Policy head: d_model * n_action_channels
      * Value head: d_model * value_hidden + value_hidden

    Returns the integer estimate. Useful for the gate-check "set-transformer
    student must be <2M params after W5 distillation" without actually
    constructing the torch module.
    """
    d = cfg.d_model
    fourier_dim = 4 * cfg.n_fourier_bands
    stem = (n_features + fourier_dim) * d
    per_block = (4 * d * d) + int(2 * d * d * cfg.mlp_ratio)
    blocks = cfg.n_blocks * per_block
    policy = d * cfg.n_action_channels
    value = d * cfg.value_hidden + cfg.value_hidden
    # +10% for biases, LN, dist_scale, residual path scales.
    total = int((stem + blocks + policy + value) * 1.10)
    return total


# ---------------------------------------------------------------------------
# Decode helpers — mirror conv_policy's ACTION_LOOKUP so the downstream
# MCTS integration is architecture-agnostic. Both models emit
# (n_action_channels) logits per owned planet; the same
# ``(angle_bucket, ship_frac)`` mapping applies.
# ---------------------------------------------------------------------------

# Same layout as conv_policy.ACTION_LOOKUP — keep these in sync.
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


def owned_planet_indices(
    features_row: Tuple[float, ...],
    type_planet_idx: int = 1,
    owner_me_idx: int = 4,
) -> bool:
    """Predicate: does this entity row represent an owned planet?

    Used at decode time to read the policy logits for only the rows
    where the agent has a valid action. The model emits per-entity
    logits; we filter to owned planets.

    Args:
      features_row: a single (F,) entity feature tuple / array.
      type_planet_idx: feature index of the ``type_planet`` flag.
        Defaults match ``entity_feature_schema`` order.
      owner_me_idx: feature index of the ``owner_me`` flag.

    Returns:
      True iff the row is an owned planet.
    """
    return bool(features_row[type_planet_idx]) and bool(features_row[owner_me_idx])


def feature_offsets() -> dict:
    """Map from feature name to its offset in ``entity_feature_schema()``.

    Useful for the torch forward pass (indexing pos_x / pos_y) and for
    the decoder (reading owner_me / type_planet flags). Computed once
    from the schema to avoid magic-number drift.

    Returns:
      dict mapping feature name -> int index.
    """
    return {name: i for i, name in enumerate(entity_feature_schema())}
