"""NN-greedy rollout policy.

The structural reason every NN-as-leaf experiment (v33-v36) lost to
v32b's heuristic rollouts: MCTS rollouts use ``HeuristicAgent`` on
both sides, so Q estimates measure "how the heuristic plays from
here" — exactly what the heuristic anchor already represents. Search
has no information that disagrees with the anchor; the override rate
stays at 9.2%.

This file provides ``NNRolloutAgent`` — a rollout policy that uses
the NN's policy logits directly to pick moves. When MCTS rollouts use
this agent on both sides, Q estimates measure "how the NN plays from
here". That's genuinely different from the heuristic anchor, so search
gets meaningful disagreement signal.

Cost per ``act()``: ~1-2 ms — between fast_rollout (~0.02 ms) and full
heuristic (~4-5 ms). At 850 ms search budget, ~30-50 sims/turn vs
heuristic's 12-16. The quality is hopefully better than fast_rollout
(which lost -190 Elo in the v35a A/B) because it draws on a trained
policy head rather than nearest-target geometry.

Invariants (mirrors fast_rollout.py):
  * Only my planets launch.
  * ``ships <= planet.ships`` always.
  * Angle is finite.
  * Falls back to no-op if NN forward fails (defensive — must never
    forfeit a turn).

This is consumed by ``GumbelRootSearch`` when
``GumbelConfig.rollout_policy == "nn"``. The root anchor is still
provided by ``HeuristicAgent``; only rollout plies swap in this agent.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from orbitwars.bots.base import Action, Agent, Deadline, no_op, obs_get
from orbitwars.features.obs_encode import encode_grid
from orbitwars.nn.conv_policy import (
    ACTION_LOOKUP,
    ConvPolicy,
    ConvPolicyCfg,
    planet_to_grid_coords,
)


# Pre-compute angle bucket centers (in radians, [0, 2π)) for the 4
# canonical directions used by ACTION_LOOKUP. East=0, North=π/2,
# West=π, South=3π/2.
_BUCKET_ANGLES = (
    0.0,
    0.5 * math.pi,
    math.pi,
    1.5 * math.pi,
)


class NNRolloutAgent(Agent):
    """NN-greedy per-planet rollout policy.

    Reads the trained ConvPolicy's logits at each owned-planet's grid
    cell and selects the argmax-channel action per planet. Channels
    correspond to (angle_bucket, ship_fraction) per ACTION_LOOKUP.

    Min-launch + send-fraction constraints mirror fast_rollout so the
    actions remain valid under the engine's combat math regardless of
    NN output quality.

    Attributes:
        model: a loaded ``ConvPolicy``. ``model.eval()`` is enforced
            once at construction.
        cfg: matching ``ConvPolicyCfg``.
        min_launch_size: don't launch a fleet smaller than this many
            ships. Matches HeuristicAgent's default to avoid dribbles.
    """

    name = "nn_rollout"

    def __init__(
        self,
        model: ConvPolicy,
        cfg: ConvPolicyCfg,
        min_launch_size: int = 20,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.min_launch_size = int(min_launch_size)
        # Idempotent — but cheap enough to call every time.
        self.model.eval()

    def act(self, obs: Any, deadline: Deadline) -> Action:
        # Always stage a safe fallback first; if anything below blows
        # up we still return a valid action.
        deadline.stage(no_op())

        player = obs_get(obs, "player", 0)
        planets = obs_get(obs, "planets", [])
        if not planets:
            return no_op()

        # Forward pass once per act(). Defensive: any failure in the
        # NN path falls back to no-op (still a valid action, never
        # forfeits a turn).
        try:
            import torch
            grid = encode_grid(obs, player, self.cfg)
            x = torch.from_numpy(grid).unsqueeze(0)  # (1, C, H, W)
            with torch.no_grad():
                logits, _value = self.model(x)
            # logits: (1, 8, H, W). Drop batch dim.
            logits_np = logits[0].cpu().numpy()  # (8, H, W)
        except Exception:
            return no_op()

        moves: Action = []
        min_size = self.min_launch_size
        # Single pass over my planets — no defensive scoring, no arrival
        # table, no sun-tangent. Per-planet argmax over 8 channels.
        for p in planets:
            if p[1] != player:
                continue
            available = int(p[5])
            if available < min_size:
                continue
            mp_x = float(p[2])
            mp_y = float(p[3])
            gy, gx = planet_to_grid_coords(mp_x, mp_y, self.cfg)
            # logits shape (8, H, W); pick argmax over the 8 channels
            # at this cell.
            cell = logits_np[:, gy, gx]
            best_ch = int(np.argmax(cell))
            angle_bucket, ship_frac = ACTION_LOOKUP[best_ch]
            angle = _BUCKET_ANGLES[angle_bucket]
            ships = int(available * float(ship_frac))
            if ships < min_size:
                ships = min_size
            if ships > available:
                ships = available
            moves.append([int(p[0]), float(angle), int(ships)])

        deadline.stage(moves)
        return moves
