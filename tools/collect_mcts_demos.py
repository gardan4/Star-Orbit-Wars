"""Collect (state, MCTS visit distribution) demos for distillation BC.

Today's BC uses ``HeuristicAgent`` self-play and labels each example
with a *single* bucketed action — the best the heuristic plays. That
caps the policy at "as good as the heuristic" in the limit. AlphaZero's
recipe swaps in **MCTS visit distributions** as the target: at each
turn run search to N sims, record the (root_visits / total_visits)
distribution, distill into the policy head with KL loss.

This script collects exactly that. For each self-play game we instrument
both ``MCTSAgent`` instances so they expose their last
``SearchResult`` via ``.last_search_result``. Per turn:

* For each candidate joint that the search evaluated, walk its
  per-planet ``PlanetMove`` and bin the visit count into the planet's
  8-channel ACTION_LOOKUP slot.
* Per source planet, emit one example: ``(x_grid, gy, gx, visits[8])``
  where ``visits`` is the softmax-normalized visit histogram across
  the 8 channels (4 angle buckets × 2 ship-frac buckets).

Output schema (``runs/mcts_demos.npz``):

    x:           (N, 12, 50, 50)  float32   — same encoding as bc_demos
    gy, gx:      (N,)             int64     — source planet grid cell
    visit_dist:  (N, 8)           float32   — softmax-normalized visits

Run:
  $env:PYTHONPATH="src;."; .venv\\Scripts\\python.exe tools\\collect_mcts_demos.py \\
      --games 60 --sims 256 --out runs/mcts_demos.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch  # type: ignore[import-not-found]

from orbitwars.features.obs_encode import encode_grid as encode_obs_grid
from orbitwars.bots.heuristic import parse_obs
from orbitwars.bots.mcts_bot import MCTSAgent
from orbitwars.mcts.gumbel_search import GumbelConfig
from orbitwars.nn.nn_prior import candidate_to_channel
from kaggle_environments import make


def _planet_grid_cell(planet: list, grid_h: int = 50, grid_w: int = 50) -> Tuple[int, int]:
    """Map a planet's (x, y) to grid (gy, gx). Same convention as
    encode_obs_grid: x in [0, 100), grid is 50×50 with each cell = 2 units."""
    gx = int(min(grid_w - 1, max(0, int(float(planet[2]) / 2))))
    gy = int(min(grid_h - 1, max(0, int(float(planet[3]) / 2))))
    return gy, gx


def _harvest_visits(
    obs: Any, player: int, agent: Any,
    out_x: List[np.ndarray], out_gy: List[int], out_gx: List[int],
    out_visits: List[np.ndarray],
    out_player_ids: Optional[List[int]] = None,
) -> int:
    """Pull (state, visit-dist) demos from ``agent.last_search_result``.

    Returns the number of demos appended (one per owned planet that
    accumulated nonzero visits across non-HOLD candidates).

    If ``out_player_ids`` is provided, also records the player_id for
    each emitted demo. The caller (``_run_self_play``) uses this after
    the game ends to stamp each demo with the corresponding player's
    terminal reward — that's the (state, outcome) pair the value head
    needs to learn V(s).
    """
    sr = getattr(agent, "last_search_result", None)
    if sr is None or not sr.visits or not sr.best_joint:
        return 0
    # Encode obs once per turn — same channel layout as bc_demos.
    x_grid = encode_obs_grid(obs, player_id=player)
    if x_grid is None:
        return 0

    po = parse_obs(obs)
    # Available ships per source planet for candidate_to_channel (ship-
    # fraction bucketing depends on it).
    avail_by_pid = {
        int(p[0]): int(p[5]) for p in po.my_planets
    }

    # Aggregate per-planet 8-channel visit histograms across all candidate
    # joints. SearchResult.candidates and .visits are parallel-indexed.
    candidates = getattr(sr, "candidates", None) or [sr.best_joint]
    visits = sr.visits if sr.visits else [1] * len(candidates)
    per_planet: dict[int, np.ndarray] = {}
    for joint, n_visits in zip(candidates, visits):
        if joint is None or n_visits <= 0:
            continue
        for move in joint.moves:
            pid = int(move.from_pid)
            avail = avail_by_pid.get(pid, 0)
            ch = candidate_to_channel(move, avail)
            if ch < 0:  # HOLD — skip (no policy-channel target)
                continue
            if pid not in per_planet:
                per_planet[pid] = np.zeros(8, dtype=np.float32)
            per_planet[pid][ch] += float(n_visits)

    n_emitted = 0
    for pid, hist in per_planet.items():
        plnt = po.planet_by_id.get(pid)
        if plnt is None:
            continue
        s = float(hist.sum())
        if s <= 0:
            continue
        target = hist / s
        gy, gx = _planet_grid_cell(plnt)
        out_x.append(x_grid.astype(np.float32))
        out_gy.append(gy)
        out_gx.append(gx)
        out_visits.append(target)
        if out_player_ids is not None:
            out_player_ids.append(int(player))
        n_emitted += 1
    return n_emitted


def make_agent(
    seed: int, sims: int, deadline_ms: float, bc_ckpt: Path,
    rollout_policy: str = "fast",
) -> MCTSAgent:
    cfg = GumbelConfig()
    cfg.sim_move_variant = "exp3"
    cfg.exp3_eta = 0.3
    cfg.rollout_policy = rollout_policy
    cfg.anchor_improvement_margin = 0.0
    cfg.use_macros = True
    cfg.total_sims = sims
    cfg.num_candidates = max(8, getattr(cfg, "num_candidates", 4))
    cfg.hard_deadline_ms = deadline_ms
    cfg.rollout_depth = 15

    prior_fn = None
    value_fn = None
    if bc_ckpt.exists():
        try:
            from orbitwars.nn.nn_prior import load_conv_policy, make_nn_prior_fn
            model, mcfg = load_conv_policy(bc_ckpt, device="cpu")
            prior_fn = make_nn_prior_fn(model, mcfg)
            # When rollout_policy == "nn_value" we ALSO need to wire the
            # value_fn so MCTS replaces leaf rollouts with NN-value lookups.
            # This is the closed-loop iteration N+1 path: collect demos
            # with the iter-N value head guiding search.
            if rollout_policy == "nn_value":
                from orbitwars.nn.nn_value import make_nn_value_fn
                value_fn = make_nn_value_fn(model, mcfg)
        except Exception as e:
            print(f"  BC prior load failed: {e!r} — using uniform", flush=True)
    return MCTSAgent(
        gumbel_cfg=cfg, rng_seed=seed,
        move_prior_fn=prior_fn, value_fn=value_fn,
    )


def _run_self_play(agents: List[MCTSAgent], harvest_each_turn: bool,
                   out_x, out_gy, out_gx, out_visits,
                   out_terminal_values: Optional[List[float]] = None,
                   ) -> Tuple[int, list, int]:
    """Custom self-play loop that lets us inspect each agent's state
    *between* turns (kaggle_environments' env.run is opaque).

    Returns (steps, rewards, n_demos_this_game).

    If ``out_terminal_values`` is provided, after the game ends each
    demo collected during this game gets stamped with the corresponding
    player's terminal reward (+1 win / -1 loss / 0 tie). Demos
    collected from player 0's perspective get rewards[0]; demos from
    player 1's perspective get rewards[1]. This is the value-head
    training target for AlphaZero-style learning (V(s) = expected
    outcome from this state).
    """
    env = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
    env.reset()

    from orbitwars.bots.base import Deadline
    n_demos_this_game = 0
    steps = 0
    # Track per-demo player_id so we can stamp terminal values once
    # the game finishes. This is a *local* list; it gets resolved into
    # out_terminal_values at the end of the game.
    demo_player_ids: List[int] = []
    while not all(s.status in ("DONE", "ERROR", "INVALID") for s in env.state):
        actions = []
        for player_id, agent in enumerate(agents):
            obs = env.state[player_id].observation
            if env.state[player_id].status != "ACTIVE":
                actions.append([])
                continue
            dl = Deadline()
            try:
                action = agent.act(obs, dl)
            except Exception:
                action = []
            actions.append(action)
            if harvest_each_turn:
                n_demos_this_game += _harvest_visits(
                    obs, player_id, agent,
                    out_x, out_gy, out_gx, out_visits,
                    out_player_ids=demo_player_ids,
                )
        env.step(actions)
        steps += 1
        if steps > 600:  # safety: should never hit (game is 500-turn capped)
            break
    rewards = [s.reward for s in env.state]

    # Stamp terminal values per demo. Convert env.state.reward (which
    # can be None / partial) to a clean +1/-1/0 vector.
    if out_terminal_values is not None:
        clean_rewards = [
            (1.0 if r == 1 else (-1.0 if r == -1 else 0.0))
            for r in rewards
        ]
        for pid in demo_player_ids:
            if 0 <= pid < len(clean_rewards):
                out_terminal_values.append(clean_rewards[pid])
            else:
                out_terminal_values.append(0.0)
    return steps, rewards, n_demos_this_game


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=256)
    ap.add_argument("--deadline-ms", type=float, default=2000.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="runs/mcts_demos.npz")
    ap.add_argument("--bc-checkpoint", type=str, default="runs/bc_warmstart_v2.pt")
    ap.add_argument(
        "--rollout-policy",
        choices=["heuristic", "fast", "nn_value"],
        default="fast",
        help=(
            "MCTS leaf evaluator. 'fast'=tiny rollout (default, cheap demos); "
            "'heuristic'=HeuristicAgent rollouts; "
            "'nn_value'=skip rollouts, query NN value head 1 ply ahead "
            "(requires --bc-checkpoint with a trained value head)."
        ),
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bc_ckpt = Path(args.bc_checkpoint)
    print(f"BC prior: {bc_ckpt}  ({'exists' if bc_ckpt.exists() else 'MISSING — uniform prior'})", flush=True)

    out_x: List[np.ndarray] = []
    out_gy: List[int] = []
    out_gx: List[int] = []
    out_visits: List[np.ndarray] = []
    out_terminal_values: List[float] = []

    t_total = time.perf_counter()
    print(f"rollout_policy: {args.rollout_policy}"
          + (" (NN value head used as leaf eval)"
             if args.rollout_policy == "nn_value" else ""),
          flush=True)
    for g in range(args.games):
        a0 = make_agent(
            args.seed + g * 2,     args.sims, args.deadline_ms, bc_ckpt,
            rollout_policy=args.rollout_policy,
        )
        a1 = make_agent(
            args.seed + g * 2 + 1, args.sims, args.deadline_ms, bc_ckpt,
            rollout_policy=args.rollout_policy,
        )
        t0 = time.perf_counter()
        steps, rewards, n_demos = _run_self_play(
            [a0, a1], harvest_each_turn=True,
            out_x=out_x, out_gy=out_gy, out_gx=out_gx, out_visits=out_visits,
            out_terminal_values=out_terminal_values,
        )
        wall = time.perf_counter() - t0
        print(
            f"  game {g + 1:>3}/{args.games}  "
            f"steps={steps}  rewards={rewards}  "
            f"demos+={n_demos}  wall={wall:.0f}s",
            flush=True,
        )

    if not out_x:
        print(
            "WARN: no demos collected — agent may not be exposing "
            "last_search_result. Check that mcts_bot.act() sets it.",
            file=sys.stderr,
        )
        return 1

    print(f"\ncollected {len(out_x):,} demos in {time.perf_counter() - t_total:.0f}s")
    # Sanity: terminal values must be parallel-indexed with the demos
    # (one stamp per emitted demo). If they're not equal-length, the
    # game-end stamping logic missed a turn — fail loudly so we don't
    # silently train on broken value targets.
    if out_terminal_values and len(out_terminal_values) != len(out_x):
        print(
            f"ERROR: terminal_values len ({len(out_terminal_values)}) != "
            f"demos len ({len(out_x)}). Refusing to write inconsistent "
            f"targets.", file=sys.stderr,
        )
        return 1
    save_kwargs = dict(
        x=np.stack(out_x).astype(np.float32),
        gy=np.array(out_gy, dtype=np.int64),
        gx=np.array(out_gx, dtype=np.int64),
        visit_dist=np.stack(out_visits).astype(np.float32),
    )
    if out_terminal_values:
        save_kwargs["terminal_value"] = np.array(
            out_terminal_values, dtype=np.float32,
        )
        # Quick distribution summary so log shows the mix.
        tvs = save_kwargs["terminal_value"]
        n_win = int((tvs > 0.5).sum())
        n_loss = int((tvs < -0.5).sum())
        n_tie = int(((tvs >= -0.5) & (tvs <= 0.5)).sum())
        print(
            f"terminal_value mix: {n_win} wins / {n_loss} losses / "
            f"{n_tie} ties (mean={tvs.mean():.3f})"
        )
    np.savez_compressed(out_path, **save_kwargs)
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
