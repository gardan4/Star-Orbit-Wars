"""Single-launch forensic probe.

Replay a game up to a given launch turn, then for the emitted launch:
  1. Run `_trajectory_obstruction` on the launch; report verdict.
  2. Continue stepping the engine; observe when / where the fleet dies.
  3. Print side-by-side so we can see whether the walk agreed with
     engine reality.

Usage:
    python tools/diag_one_shot.py --seed 42 --launch-turn 50
"""
from __future__ import annotations

import argparse
import math
import random
import sys

import numpy as np
from kaggle_environments import make

from orbitwars.bots.heuristic import (
    HeuristicAgent,
    _trajectory_obstruction,
    _OBSTR_CLEAR, _OBSTR_SUN, _OBSTR_OOB, _OBSTR_WASTED,
)


def _obstr_name(code: int) -> str:
    return {
        _OBSTR_CLEAR: "CLEAR",
        _OBSTR_SUN: "SUN",
        _OBSTR_OOB: "OOB",
        _OBSTR_WASTED: "WASTED",
    }.get(code, f"planet#{code}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--launch-turn", type=int, default=50,
                    help="Game step at which to inspect fresh launches.")
    ap.add_argument("--fleet-idx", type=int, default=0,
                    help="Which emitted launch on that turn to follow.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    env = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
    env.reset(num_agents=2)
    env.step([[], []])

    me = HeuristicAgent()
    opp = HeuristicAgent()
    k_me = me.as_kaggle_agent()
    k_opp = opp.as_kaggle_agent()

    # Play to launch-turn-1.
    for step in range(args.launch_turn):
        if env.state[0]["status"] != "ACTIVE":
            print(f"game ended before step {args.launch_turn}", file=sys.stderr)
            return 1
        obs0 = env.state[0]["observation"]
        obs1 = env.state[1]["observation"]
        a0 = k_me(obs0, env.configuration)
        a1 = k_opp(obs1, env.configuration)
        env.step([a0, a1])

    # On this turn, emit and inspect.
    if env.state[0]["status"] != "ACTIVE":
        print("game inactive at launch turn", file=sys.stderr)
        return 1
    obs0 = env.state[0]["observation"]
    pre_next_id = int(obs0.get("next_fleet_id", 0))
    a0 = k_me(obs0, env.configuration)
    intents = list(me.last_launch_intents)

    if not intents:
        print(f"no launches at turn {args.launch_turn}")
        return 0
    if args.fleet_idx >= len(intents):
        print(f"only {len(intents)} intents this turn; idx out of range")
        return 1

    it = intents[args.fleet_idx]
    fleet_id = pre_next_id + args.fleet_idx
    print(f"=== Launch #{fleet_id} at turn {args.launch_turn} ===")
    print(f"  from_pid={it.from_pid}  target_pid={it.target_pid}")
    print(f"  angle={it.angle:.4f}  ships={it.ships}")
    print(f"  predicted_travel={it.predicted_travel_turns:.2f}  predicted_arrival={it.predicted_arrival_turn}")
    print(f"  score={it.score:.3f}")

    # Re-parse obs to get planet positions/initial-planet; call walk.
    from orbitwars.bots.heuristic import parse_obs
    po = parse_obs(obs0)
    mp = po.planet_by_id[it.from_pid]
    verdict = _trajectory_obstruction(
        source_center=(float(mp[2]), float(mp[3])),
        source_radius=float(mp[4]),
        angle=float(it.angle),
        ships=int(it.ships),
        target_pid=int(it.target_pid),
        po=po,
    )
    print(f"  obstruction verdict: {_obstr_name(verdict)} (code={verdict})")

    a1 = k_opp(env.state[1]["observation"], env.configuration)
    env.step([a0, a1])

    # Track this specific fleet until it dies.
    max_follow = 80
    print(f"\n=== Following fleet #{fleet_id} ===")
    print(f"{'step':>4}  {'fx':>7}  {'fy':>7}  dist_to_target  target_pos")
    for follow in range(max_follow):
        if env.state[0]["status"] != "ACTIVE":
            print("  [game ended]")
            break
        obs_me = env.state[0]["observation"]
        fleets = {int(f[0]): f for f in (obs_me.get("fleets") or [])}
        planets = {int(p[0]): p for p in (obs_me.get("planets") or [])}
        f = fleets.get(fleet_id)
        tp = planets.get(it.target_pid)
        if f is None:
            print(f"  step={obs_me['step']}: fleet died")
            if tp is not None:
                fx, fy = 0.0, 0.0  # unknown
                tx, ty = float(tp[2]), float(tp[3])
                print(f"    target still exists at ({tx:.2f},{ty:.2f}), owner={tp[1]}, ships={tp[5]}")
            else:
                print("    target is gone from obs")
            # Try to guess what killed it: iterate all planets and check
            # which is nearest to what would be its position.
            break
        fx, fy = float(f[2]), float(f[3])
        if tp is not None:
            tx, ty = float(tp[2]), float(tp[3])
            d = math.hypot(fx - tx, fy - ty)
            print(f"  {obs_me['step']:>4}  {fx:7.2f}  {fy:7.2f}  {d:14.2f}  ({tx:.2f},{ty:.2f})")
        else:
            print(f"  {obs_me['step']:>4}  {fx:7.2f}  {fy:7.2f}  target-vanished")

        if env.state[0]["status"] != "ACTIVE":
            break
        a0 = k_me(obs_me, env.configuration)
        a1 = k_opp(env.state[1]["observation"], env.configuration)
        env.step([a0, a1])

    return 0


if __name__ == "__main__":
    sys.exit(main())
