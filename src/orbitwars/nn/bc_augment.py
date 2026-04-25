"""4-fold rotational data augmentation for BC training.

The Orbit Wars board is 100x100 with the sun at the center, so the game
state has C4 symmetry (rotations by 90/180/270 degrees around the
center are equivalent to relabeling positions). For BC training we
exploit this for free: every (state, action, source_cell) demo can be
rotated into 3 additional valid demos. Quadruples the effective dataset
without touching the demo collector.

The transformation has three parts:

1. **Spatial rotation** of the (C, H, W) tensor.
2. **Angle-vector rotation** of the ``fleet_angle_cos`` and
   ``fleet_angle_sin`` channels (their per-cell scalar values are
   themselves direction-encoded and need updating, not just moving).
3. **Source-cell + action-bucket relabeling**.

Convention: ``np.rot90(arr, k=K, axes=(-2, -1))`` rotates the array
counter-clockwise K times when viewing the spatial axes (H=row, W=col)
as an image. Under this convention, a planet at grid (gy, gx) with
H=W=N moves to:

    K=1 (90° CCW):  (gy', gx') = (N-1-gx, gy)
    K=2 (180°):     (gy', gx') = (N-1-gy, N-1-gx)
    K=3 (270° CCW): (gy', gx') = (gx, N-1-gy)

The action label is an index into ``ACTION_LOOKUP`` = 4 angle buckets
× 2 ship-fraction buckets. Ship fraction is rotation-invariant; angle
bucket rotates as ``(b + K) % 4``. The compound channel index moves
``c → (c + 2*K) % 8`` (since each pair of channels covers one angle
bucket).

The per-cell vector channels (``fleet_angle_cos``, ``fleet_angle_sin``)
encode a 2D direction. Under rotation by angle φ:
    new_cos = cos(θ + φ) = cos(θ)cos(φ) - sin(θ)sin(φ)
    new_sin = sin(θ + φ) = sin(θ)cos(φ) + cos(θ)sin(φ)
With φ = K * π/2 the matrix simplifies; we compute the rotated values
on the *already-spatially-rotated* tensor.

Tests in ``tests/test_bc_augment.py`` verify the round-trip with
hand-built demos and check the equivariance against the live encoder.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

# Channel indices that are vector-valued (need angle rotation, not just
# spatial reposition). Pinned to ``feature_channels()`` order in
# ``orbitwars.nn.conv_policy``.
_FLEET_COS_CH = 9
_FLEET_SIN_CH = 10

# Action lookup pairs (angle_bucket, ship_frac). Pinned to ACTION_LOOKUP
# in conv_policy.py: 8 channels = 4 buckets * 2 fractions, layout is
# bucket-major (channel = bucket*2 + frac_idx), so a 90-deg rotation
# shifts the channel index by +2.
_ACTION_CHANNELS = 8
_ANGLE_BUCKETS = 4


def rotate_demo(
    x: np.ndarray, gy: int, gx: int, label: int, k: int,
) -> Tuple[np.ndarray, int, int, int]:
    """Apply a K * 90° CCW rotation to a single (state, source-cell,
    action-bucket) demo.

    Args:
      x: (C, H, W) float array, the conv input encoding.
      gy, gx: source planet's grid cell.
      label: action bucket index in [0, 8).
      k: rotation count (0=identity, 1=90 CCW, 2=180, 3=270 CCW). k can
         be any int — taken mod 4.

    Returns:
      (x_rot, gy_rot, gx_rot, label_rot)
    """
    assert x.ndim == 3, f"expected (C,H,W), got {x.shape}"
    k = int(k) % 4
    if k == 0:
        return x.copy(), int(gy), int(gx), int(label)

    h, w = x.shape[1], x.shape[2]
    assert h == w, f"4-fold aug requires square grid, got H={h} W={w}"

    # 1. Spatial rotation. np.rot90 with axes=(-2,-1) rotates (H, W).
    x_rot = np.rot90(x, k=k, axes=(-2, -1)).copy()

    # 2. Angle-vector channels need their per-cell direction rotated.
    # After np.rot90, the vector at (h', w') was originally at the
    # source cell (h, w); its (cos, sin) values still encode the
    # ORIGINAL angle θ, but they should now encode θ + k*π/2.
    # We update in place on x_rot using the rotation matrix.
    cos_k = float(np.cos(k * np.pi / 2))
    sin_k = float(np.sin(k * np.pi / 2))
    cos_ch = x_rot[_FLEET_COS_CH].copy()
    sin_ch = x_rot[_FLEET_SIN_CH].copy()
    x_rot[_FLEET_COS_CH] = cos_ch * cos_k - sin_ch * sin_k
    x_rot[_FLEET_SIN_CH] = cos_ch * sin_k + sin_ch * cos_k

    # 3. Source cell remap.
    n = h
    if k == 1:
        gy_new, gx_new = (n - 1 - gx), gy
    elif k == 2:
        gy_new, gx_new = (n - 1 - gy), (n - 1 - gx)
    else:  # k == 3
        gy_new, gx_new = gx, (n - 1 - gy)

    # 4. Action-bucket remap: channel = bucket*2 + frac_idx, so a CCW
    # rotation shifts the bucket by +k and the channel by +2k mod 8.
    label_new = (int(label) + 2 * k) % _ACTION_CHANNELS

    return x_rot, gy_new, gx_new, label_new


def augment_4fold(
    x: np.ndarray, gy: np.ndarray, gx: np.ndarray, labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack the 4 rotations of every demo. Output size = 4 * input size.

    Args:
      x: (N, C, H, W) float
      gy, gx: (N,) int
      labels: (N,) int

    Returns:
      (x_4N, gy_4N, gx_4N, labels_4N)
    """
    assert x.ndim == 4, f"expected (N,C,H,W), got {x.shape}"
    n = x.shape[0]
    h, w = x.shape[2], x.shape[3]
    assert h == w, f"4-fold aug requires square grid, got H={h} W={w}"
    out_x = np.empty((4 * n, x.shape[1], h, w), dtype=x.dtype)
    out_gy = np.empty(4 * n, dtype=gy.dtype)
    out_gx = np.empty(4 * n, dtype=gx.dtype)

    # Two label formats supported:
    #   * 1D integer labels (heuristic-target BC): rotate via mod-8.
    #   * 2D float visit-distribution labels (MCTS-teacher BC): rotate
    #     by rolling the 8 channels by 2*k (since each pair of channels
    #     covers one of 4 angle buckets). np.roll with axis=-1.
    soft = labels.ndim == 2
    if soft:
        out_lb = np.empty((4 * n, labels.shape[1]), dtype=labels.dtype)
    else:
        out_lb = np.empty(4 * n, dtype=labels.dtype)

    # k = 0 (identity).
    out_x[:n] = x
    out_gy[:n] = gy
    out_gx[:n] = gx
    out_lb[:n] = labels

    # k = 1, 2, 3.
    for k in (1, 2, 3):
        s = k * n
        # Spatial rotation: vectorized via np.rot90 over axes (-2, -1).
        rot = np.rot90(x, k=k, axes=(-2, -1)).copy()
        # Angle-vector channel update.
        cos_k = float(np.cos(k * np.pi / 2))
        sin_k = float(np.sin(k * np.pi / 2))
        cos_ch = rot[:, _FLEET_COS_CH].copy()
        sin_ch = rot[:, _FLEET_SIN_CH].copy()
        rot[:, _FLEET_COS_CH] = cos_ch * cos_k - sin_ch * sin_k
        rot[:, _FLEET_SIN_CH] = cos_ch * sin_k + sin_ch * cos_k
        out_x[s:s + n] = rot

        if k == 1:
            out_gy[s:s + n] = (h - 1 - gx)
            out_gx[s:s + n] = gy
        elif k == 2:
            out_gy[s:s + n] = (h - 1 - gy)
            out_gx[s:s + n] = (h - 1 - gx)
        else:
            out_gy[s:s + n] = gx
            out_gx[s:s + n] = (h - 1 - gy)
        if soft:
            # Roll channels: c_new = (c_orig + 2*k) % 8 means the value at
            # original channel `c` moves to channel `(c + 2*k) % 8`. With
            # np.roll, shift=+2*k along axis=-1 produces this exact mapping.
            out_lb[s:s + n] = np.roll(labels, shift=2 * k, axis=-1)
        else:
            out_lb[s:s + n] = (labels + 2 * k) % _ACTION_CHANNELS

    return out_x, out_gy, out_gx, out_lb
