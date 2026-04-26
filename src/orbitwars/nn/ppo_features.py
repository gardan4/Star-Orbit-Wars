"""Feature encoding for PPO rollout collection.

The PPO trainer collects transitions where each "decision row" is one
owned-planet-in-one-state. This module wraps the existing spatial
``encode_grid`` (12-channel 50×50 conv input) plus the per-decision
``(gy, gx)`` source-cell index, so the trainer can batch decisions
across players and timesteps without redoing the encoding.

Design choice — share encode with BC training:

* Same ``encode_grid`` used for BC v1/v3 means the BC checkpoint is
  drop-in as the PPO warmstart. No re-train needed.
* Same ``ConvPolicy`` at the head; the policy head outputs (B, 8, H, W)
  and we read at ``(gy, gx)`` to get 8 per-cell logits → categorical
  distribution → channel index → ``ACTION_LOOKUP`` → (angle_bucket,
  ship_frac).
* Heuristic intercept math chooses the ACTUAL angle from
  (angle_bucket, source_planet, nearest_target_in_quadrant) so the
  policy doesn't need to learn continuous angles — just direction-
  bucket and ship-fraction-bucket.

Compare to the public Kaggle PPO tutorial which builds candidate-list
+ scalar features per planet. Their action space is "pick a target
index from the K nearest"; ours is "pick an angle bucket from 4 ×
2 ship-fractions = 8 channels". Both are valid; we keep ours so the
BC warmstart works.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from orbitwars.bots.heuristic import ParsedObs, parse_obs
from orbitwars.features.obs_encode import encode_grid
from orbitwars.nn.conv_policy import ConvPolicyCfg


@dataclass
class DecisionRow:
    """One (state, source-planet) decision unit with an associated
    grid cell. Used as input to the policy at rollout-collection time.
    """
    obs_x: np.ndarray              # (C, H, W) float32 — encoded obs
    gy: int                        # source planet grid row
    gx: int                        # source planet grid col
    source_planet_id: int          # planet id (for action decoding)
    available_ships: int           # ships at the source (clip target)


def encode_decisions(
    obs: Any, player_id: int, cfg: ConvPolicyCfg | None = None,
) -> List[DecisionRow]:
    """Encode all per-planet decisions for ``player_id`` from a single obs.

    Returns one ``DecisionRow`` per owned planet with at least one ship.
    Empty list if the player has no owned planets (e.g. eliminated).

    The same ``obs_x`` tensor is shared across all rows for a given turn
    — we do NOT make N copies. The PPO trainer should batch DecisionRows
    from multiple turns/players together, NOT rely on deduplication
    inside this function.
    """
    if cfg is None:
        cfg = ConvPolicyCfg()
    po = parse_obs(obs)
    if po.player != player_id:
        # parse_obs uses obs.player; perspective mismatch is caller bug.
        # Keep it tolerant — we still respect the requested player_id.
        po = parse_obs(_obs_with_player(obs, player_id))
    obs_x = encode_grid(obs, player_id=player_id, cfg=cfg)
    rows: List[DecisionRow] = []
    grid_h = cfg.grid_h
    grid_w = cfg.grid_w
    for plnt in po.my_planets:
        ships = int(plnt[5])
        if ships <= 0:
            continue
        # Same gy/gx convention as encode_grid: y * H/100, x * W/100.
        gy = int(min(grid_h - 1, max(0, float(plnt[3]) * grid_h / 100.0)))
        gx = int(min(grid_w - 1, max(0, float(plnt[2]) * grid_w / 100.0)))
        rows.append(DecisionRow(
            obs_x=obs_x, gy=gy, gx=gx,
            source_planet_id=int(plnt[0]),
            available_ships=ships,
        ))
    return rows


def _obs_with_player(obs: Any, player_id: int) -> Any:
    """Return a copy of obs with `player` overwritten. Used when the
    caller queries for a player that's NOT the seat the obs is from
    (e.g. encoding the opponent's view in self-play)."""
    # Both dict and AttrDict-style obs are tolerated.
    if isinstance(obs, dict):
        new = dict(obs)
        new["player"] = int(player_id)
        return new
    # Fall back: monkey-patch the attribute. ParsedObs is already an
    # alias used by callers that want this; for a general object we
    # construct a minimal dict shim.
    return {
        "player": int(player_id),
        "step": getattr(obs, "step", 0),
        "angular_velocity": getattr(obs, "angular_velocity", 0.03),
        "planets": getattr(obs, "planets", []),
        "initial_planets": getattr(obs, "initial_planets", []),
        "fleets": getattr(obs, "fleets", []),
        "next_fleet_id": getattr(obs, "next_fleet_id", 0),
        "comet_planet_ids": getattr(obs, "comet_planet_ids", []),
        "comets": getattr(obs, "comets", []),
    }


def stack_decisions(rows: List[DecisionRow]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack a list of DecisionRows into batched arrays for the policy.

    Returns:
      ``(obs_x, gy, gx)`` with shapes ``(N, C, H, W)``, ``(N,)``,
      ``(N,)``. ``obs_x`` IS duplicated per-row (the encoder shares the
      same array reference across rows of the same turn, but stack
      makes a contiguous copy). Caller can dedupe by source-turn if
      memory is tight.
    """
    if not rows:
        return (
            np.zeros((0, 0, 0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    obs_x = np.stack([r.obs_x for r in rows], axis=0).astype(np.float32)
    gy = np.array([r.gy for r in rows], dtype=np.int64)
    gx = np.array([r.gx for r in rows], dtype=np.int64)
    return obs_x, gy, gx
