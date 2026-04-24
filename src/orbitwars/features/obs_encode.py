"""Observation -> feature tensor encoders for both W4 arch candidates.

Two encoders sharing a single source obs, pinned to the schemas in
``orbitwars.nn.conv_policy`` and ``orbitwars.nn.set_transformer``:

  * ``encode_grid(obs, player_id) -> np.ndarray (C=12, H=50, W=50)``
    matches ``conv_policy.feature_channels()``. Each channel is a
    dense rasterization over the 100x100 board downsampled to 50x50.

  * ``encode_entities(obs, player_id) -> (features, mask)`` matches
    ``set_transformer.entity_feature_schema()``. ``features`` is
    ``(n_max_entities, 19)`` with padding rows zeroed; ``mask`` is
    ``(n_max_entities,)`` bool — True where the row is valid.

Both encoders operate on a Kaggle obs dict (or a ``ParsedObs`` — we
tolerate either). They are pure numpy, no torch dependency, so they run
at MCTS-rollout speed. The training path wraps them with a torch tensor
conversion.

**Perspective-normalization**: both encoders take ``player_id`` and
encode "me vs. everyone else" regardless of the raw seat id. Under the
4-fold symmetry the board is identical up to rotation, so the encoder
alone is enough — no extra rotation is applied here. Data augmentation
(flip/rotate) happens at the training-loop level, not inside this
module, so inference and training share one canonical encoder.

**Sqrt scaling**: ship counts go through ``sqrt(x) / sqrt(1000)``. The
game's fleet speed formula is ``1 + 5 * (log(ships)/log(1000))^1.5``,
so fleet dynamics are already log-scaled. Sqrt is a gentler compression
that keeps small fleets distinguishable (1 vs 10 ships matters early
game) while capping large fleets (1000 vs 5000 is the same for policy
purposes — both just move fast).

**What's NOT here** (out of scope for the skeleton):
  * 4-fold symmetry augmentation helpers (rotate/mirror). Training-side.
  * Batched encoding (``encode_batch``). Trivial wrapper once this
    single-obs path is validated.
  * Fourier positional encoding of pos_x/y for the set-transformer —
    applied INSIDE the model (``set_transformer._fourier_encode``), not
    here, so the encoder stays model-agnostic.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from orbitwars.nn.conv_policy import ConvPolicyCfg, feature_channels
from orbitwars.nn.set_transformer import (
    SetTransformerCfg,
    entity_feature_schema,
    feature_offsets,
)


# ---------------------------------------------------------------------------
# Small helpers — tolerate either a kaggle obs (dict/AttrDict) or ParsedObs.
# ---------------------------------------------------------------------------


def _obs_get(obs: Any, key: str, default: Any = None) -> Any:
    """Dict-or-attr accessor matching ``heuristic.obs_get`` semantics.

    Kaggle passes an AttrDict whose values can also be accessed via
    attribute; ``ParsedObs`` is a regular dataclass. One helper fits
    both so callers don't need to branch.
    """
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


_BOARD_SIZE = 100.0
_SUN_POS = (50.0, 50.0)
_MAX_STEPS = 500
_SHIP_SCALE = float(np.sqrt(1000.0))


def _sqrt_scale_ships(ships: float) -> float:
    """``sqrt(ships) / sqrt(1000)``; saturates gracefully beyond 1000."""
    return float(np.sqrt(max(0.0, ships))) / _SHIP_SCALE


def _planet_to_grid(x: float, y: float, cfg: ConvPolicyCfg) -> Tuple[int, int]:
    """Continuous (x, y) in [0, 100] -> (gy, gx) cell index, clamped."""
    gy = int(y * cfg.grid_h / _BOARD_SIZE)
    gx = int(x * cfg.grid_w / _BOARD_SIZE)
    gy = max(0, min(cfg.grid_h - 1, gy))
    gx = max(0, min(cfg.grid_w - 1, gx))
    return gy, gx


# ---------------------------------------------------------------------------
# Grid encoder (conv_policy candidate A).
# ---------------------------------------------------------------------------


def encode_grid(
    obs: Any,
    player_id: int,
    cfg: ConvPolicyCfg | None = None,
) -> np.ndarray:
    """Encode obs as a ``(C, H, W)`` conv input tensor.

    Channel order is locked to ``conv_policy.feature_channels()``. Each
    planet and fleet rasterizes to its single (gy, gx) cell — we do NOT
    splat into a radius, because grid resolution (H=50 over a 100x100
    board, 2x2 units per cell) already matches planet radius 1-3, and
    the conv itself learns the effective radius from neighboring cells.

    Args:
      obs: Kaggle obs or ``ParsedObs`` for ``player_id``.
      player_id: seat id (0, 1, 2, 3) — perspective for me/enemy channels.
      cfg: optional override of conv cfg (grid size + channel count).

    Returns:
      float32 ``(n_channels, grid_h, grid_w)`` numpy array.

    Contract:
      * Output dtype is ``np.float32`` (torch-friendly; 4x smaller than
        float64 and negligible precision loss at our scale).
      * Channel dimension comes FIRST (PyTorch convention), not last.
      * No NaN / inf produced — callers can trust the tensor is safe to
        pass directly to a torch conv without masking.
    """
    if cfg is None:
        cfg = ConvPolicyCfg()
    names = feature_channels()
    C = len(names)
    assert C == cfg.n_channels, f"channel count drift: {C} != {cfg.n_channels}"

    grid = np.zeros((C, cfg.grid_h, cfg.grid_w), dtype=np.float32)

    # Channel indices.
    CH = {name: i for i, name in enumerate(names)}

    # --- Planets ---
    comet_pids = set(_obs_get(obs, "comet_planet_ids", []) or [])
    planets = _obs_get(obs, "planets", []) or []
    for pl in planets:
        # Planet row: [id, owner, x, y, radius, ships, production]
        pid = int(pl[0])
        owner = int(pl[1])
        x = float(pl[2])
        y = float(pl[3])
        radius = float(pl[4])
        ships = float(pl[5])
        production = float(pl[6])
        gy, gx = _planet_to_grid(x, y, cfg)

        if owner == player_id:
            grid[CH["ship_count_p0"], gy, gx] += _sqrt_scale_ships(ships)
            grid[CH["production_p0"], gy, gx] = max(
                grid[CH["production_p0"], gy, gx], production
            )
        elif owner == -1:
            grid[CH["production_neutral"], gy, gx] = max(
                grid[CH["production_neutral"], gy, gx], production
            )
        else:
            grid[CH["ship_count_p1"], gy, gx] += _sqrt_scale_ships(ships)
            grid[CH["production_p1"], gy, gx] = max(
                grid[CH["production_p1"], gy, gx], production
            )

        grid[CH["planet_radius"], gy, gx] = max(
            grid[CH["planet_radius"], gy, gx], radius / 5.0
        )
        # Orbiting = initial (orbital_r + radius) < 50. Approximated here
        # via distance-to-sun < 50 - radius, which is the condition at
        # spawn; it holds for the life of the game since orbits are
        # circular. Exact threshold may drift by one cell for planets
        # sitting right on the boundary — rare; the conv learns around it.
        dist_sun = float(np.hypot(x - _SUN_POS[0], y - _SUN_POS[1]))
        if dist_sun + radius < 50.0:
            grid[CH["is_orbiting"], gy, gx] = 1.0

        if pid in comet_pids:
            grid[CH["is_comet"], gy, gx] = 1.0

        # sun_distance: normalize by board half-diagonal (~71) so it
        # fits in [0, 1] with headroom.
        grid[CH["sun_distance"], gy, gx] = dist_sun / 71.0

    # --- Fleets ---
    fleets = _obs_get(obs, "fleets", []) or []
    for fl in fleets:
        # Fleet row: [id, owner, x, y, angle, from_planet_id, ships]
        owner = int(fl[1])
        x = float(fl[2])
        y = float(fl[3])
        angle = float(fl[4])
        ships = float(fl[6])
        gy, gx = _planet_to_grid(x, y, cfg)

        if owner == player_id:
            grid[CH["ship_count_p0"], gy, gx] += _sqrt_scale_ships(ships)
        else:
            grid[CH["ship_count_p1"], gy, gx] += _sqrt_scale_ships(ships)

        # Angle components: sum into cell (multiple fleets in one cell
        # get a vector sum — conv learns to decode).
        grid[CH["fleet_angle_cos"], gy, gx] += float(np.cos(angle))
        grid[CH["fleet_angle_sin"], gy, gx] += float(np.sin(angle))

    # --- Broadcast scalar: turn_phase ---
    step = int(_obs_get(obs, "step", 0) or 0)
    grid[CH["turn_phase"], :, :] = step / _MAX_STEPS

    return grid


# ---------------------------------------------------------------------------
# Entity encoder (set_transformer candidate B).
# ---------------------------------------------------------------------------


def encode_entities(
    obs: Any,
    player_id: int,
    cfg: SetTransformerCfg | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode obs as a ``(n_max_entities, F)`` per-entity feature tensor + mask.

    Entity order: planets first, then fleets. Padding rows (is_valid=0)
    fill up to ``cfg.n_max_entities``. Order within each type is stable
    across turns (by id), so the set-transformer's attention is the
    only mechanism that establishes entity relationships — position in
    the tensor itself carries no information beyond identity.

    Args:
      obs: Kaggle obs or ``ParsedObs``.
      player_id: seat id — perspective for owner_me / owner_enemy.
      cfg: optional override.

    Returns:
      ``(features, mask)`` where features is float32
      ``(n_max_entities, F=19)`` and mask is bool ``(n_max_entities,)``.

    Raises:
      No explicit raise; if the obs has more entities than fit, the
      extras are DROPPED. A diag warning would be appropriate in W4
      training code; this skeleton stays silent.
    """
    if cfg is None:
        cfg = SetTransformerCfg()

    schema = entity_feature_schema()
    F = len(schema)
    N = cfg.n_max_entities
    offsets = feature_offsets()

    features = np.zeros((N, F), dtype=np.float32)
    mask = np.zeros((N, ), dtype=bool)

    # Global broadcast scalars computed once, applied to each valid row.
    step = int(_obs_get(obs, "step", 0) or 0)
    turn_phase = step / _MAX_STEPS

    my_ships_total = 0.0
    enemy_ships_total = 0.0

    comet_pids = set(_obs_get(obs, "comet_planet_ids", []) or [])
    planets = _obs_get(obs, "planets", []) or []
    fleets = _obs_get(obs, "fleets", []) or []
    initial_by_id = {
        int(ip[0]): ip for ip in (_obs_get(obs, "initial_planets", []) or [])
    }

    angular_velocity = float(_obs_get(obs, "angular_velocity", 0.0) or 0.0)

    # Pre-scan for ship totals (for score_diff broadcast).
    for pl in planets:
        owner = int(pl[1])
        ships = float(pl[5])
        if owner == player_id:
            my_ships_total += ships
        elif owner != -1:
            enemy_ships_total += ships
    for fl in fleets:
        owner = int(fl[1])
        ships = float(fl[6])
        if owner == player_id:
            my_ships_total += ships
        else:
            enemy_ships_total += ships
    score_diff = (my_ships_total - enemy_ships_total) / 1000.0

    cursor = 0

    # --- Planets ---
    for pl in planets:
        if cursor >= N:
            break
        pid = int(pl[0])
        owner = int(pl[1])
        x = float(pl[2])
        y = float(pl[3])
        radius = float(pl[4])
        ships = float(pl[5])
        production = float(pl[6])

        row = features[cursor]
        row[offsets["is_valid"]] = 1.0
        row[offsets["type_planet"]] = 1.0
        if owner == player_id:
            row[offsets["owner_me"]] = 1.0
        elif owner == -1:
            row[offsets["owner_neutral"]] = 1.0
        else:
            row[offsets["owner_enemy"]] = 1.0
        row[offsets["pos_x"]] = x
        row[offsets["pos_y"]] = y

        # is_orbiting: same check as the grid encoder for consistency.
        dist_sun = float(np.hypot(x - _SUN_POS[0], y - _SUN_POS[1]))
        if dist_sun + radius < 50.0:
            row[offsets["is_orbiting"]] = 1.0
            # Only orbiting planets have a nonzero angular velocity;
            # non-orbiters are fixed.
            row[offsets["orbital_angular_vel"]] = angular_velocity

        row[offsets["ships"]] = _sqrt_scale_ships(ships)
        row[offsets["production"]] = production
        row[offsets["radius"]] = radius / 5.0
        row[offsets["sun_distance"]] = dist_sun / 71.0

        # Globals.
        row[offsets["turn_phase"]] = turn_phase
        row[offsets["score_diff"]] = score_diff

        if pid in comet_pids:
            # Tag comet flag via type_comet; planet flag still on (comets
            # are planet-backed in the engine). Models can disambiguate.
            row[offsets["type_comet"]] = 1.0

        mask[cursor] = True
        cursor += 1

    # --- Fleets ---
    for fl in fleets:
        if cursor >= N:
            break
        owner = int(fl[1])
        x = float(fl[2])
        y = float(fl[3])
        angle = float(fl[4])
        ships = float(fl[6])

        row = features[cursor]
        row[offsets["is_valid"]] = 1.0
        row[offsets["type_fleet"]] = 1.0
        if owner == player_id:
            row[offsets["owner_me"]] = 1.0
        else:
            row[offsets["owner_enemy"]] = 1.0
        row[offsets["pos_x"]] = x
        row[offsets["pos_y"]] = y

        # Fleet speed depends on ship count (game formula). We store
        # raw vx, vy derived from (angle, speed) so the model sees
        # velocity directly without having to re-derive it.
        # Speed formula matches orbit_wars.py.
        speed = 1.0 + 5.0 * (np.log(max(ships, 1.0)) / np.log(1000.0)) ** 1.5
        row[offsets["velocity_x"]] = float(np.cos(angle) * speed)
        row[offsets["velocity_y"]] = float(np.sin(angle) * speed)

        row[offsets["ships"]] = _sqrt_scale_ships(ships)
        # production / radius / is_orbiting / angular_vel stay 0 for fleets.
        row[offsets["sun_distance"]] = float(
            np.hypot(x - _SUN_POS[0], y - _SUN_POS[1])
        ) / 71.0

        row[offsets["turn_phase"]] = turn_phase
        row[offsets["score_diff"]] = score_diff

        mask[cursor] = True
        cursor += 1

    return features, mask


# ---------------------------------------------------------------------------
# Convenience helper used by both encoders + callers that want a single
# "owned planet" list for decoding the policy output.
# ---------------------------------------------------------------------------


def owned_planet_positions(
    obs: Any, player_id: int
) -> list[Tuple[int, float, float]]:
    """Return ``[(planet_id, x, y), ...]`` for planets owned by ``player_id``.

    Both decoders (conv grid indexing by planet_to_grid; set-transformer
    row indexing by (type_planet & owner_me)) need to know *which*
    planets we can launch from. This helper keeps the decode logic
    aligned with the encode logic — if we change how "owned planet" is
    determined (e.g. if we introduce allied play in a future mode),
    we change it once here.

    The list preserves the engine's planet order (by id), so training
    labels (policy target = visit distributions at each owned planet)
    and inference output align.
    """
    out: list[Tuple[int, float, float]] = []
    planets = _obs_get(obs, "planets", []) or []
    for pl in planets:
        if int(pl[1]) == player_id:
            out.append((int(pl[0]), float(pl[2]), float(pl[3])))
    return out
