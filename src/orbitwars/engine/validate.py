"""Parity validator: compare FastEngine state against kaggle_environments
step-by-step over many random seeds.

This is the blocking gate for Week 1. If FastEngine doesn't match the reference
over 1000 seeds we either fix bugs or fall back to the reference engine with
monkey-patched hot paths.

Usage:
    python -m orbitwars.engine.validate --seeds 10 --turns 20 --verbose
    python -m orbitwars.engine.validate --seeds 1000 --turns 500   # full Week 1 gate

The "run alongside" model:
  1. Build an official env, run init (call env.run with no-op bots for 1 step).
  2. Snapshot the resulting state into a FastEngine via from_official_obs().
  3. For each turn:
       a. Generate random actions for both players.
       b. Apply those actions to BOTH engines via a step.
       c. Compare state (planets, fleets, step, next_fleet_id, comets).
  4. Report any mismatches with diffing detail.

We drive random actions from a seeded RNG to keep runs reproducible.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from kaggle_environments import make

from orbitwars.engine.fast_engine import FastEngine, GameConfig


@dataclass
class Mismatch:
    seed: int
    turn: int
    field: str
    official: Any
    fast: Any

    def __repr__(self) -> str:
        return f"[seed={self.seed} turn={self.turn}] {self.field}:\n  official={self.official}\n  fast    ={self.fast}"


def _random_actions(obs, rng: random.Random) -> List[Any]:
    """Generate a moderate-activity random action list for each turn.

    We want non-trivial behavior (fleets actually launching) but not
    guaranteed-launch-everything-every-turn so we exercise the no-op branch.
    """
    player = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0)
    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    moves: List[Any] = []
    for p in planets:
        if p[1] == player and p[5] > 0 and rng.random() < 0.35:
            angle = rng.uniform(0, 2 * math.pi)
            max_ships = p[5]
            send = rng.randint(1, max_ships)
            if send >= 10:  # below 10 ships is too anemic to matter for tests
                moves.append([p[0], angle, send])
    return moves


def _compare_planets(seed: int, turn: int, ref: List[List[Any]], fast: List[List[Any]]) -> List[Mismatch]:
    mm: List[Mismatch] = []
    if len(ref) != len(fast):
        mm.append(Mismatch(seed, turn, "planet_count", len(ref), len(fast)))
        return mm
    # Planets don't necessarily appear in the same order (e.g., after comet
    # removal). Key by id.
    ref_by_id = {p[0]: p for p in ref}
    fast_by_id = {p[0]: p for p in fast}
    if set(ref_by_id.keys()) != set(fast_by_id.keys()):
        mm.append(Mismatch(seed, turn, "planet_ids",
                           sorted(ref_by_id.keys()), sorted(fast_by_id.keys())))
        return mm
    for pid, rp in ref_by_id.items():
        fp = fast_by_id[pid]
        # [id, owner, x, y, radius, ships, production]
        for name, i in [("owner", 1), ("radius", 4), ("ships", 5), ("production", 6)]:
            if rp[i] != fp[i]:
                mm.append(Mismatch(seed, turn, f"planet[{pid}].{name}", rp[i], fp[i]))
        # Positions: allow tiny float drift
        for name, i in [("x", 2), ("y", 3)]:
            if abs(rp[i] - fp[i]) > 1e-9:
                mm.append(Mismatch(seed, turn, f"planet[{pid}].{name}", rp[i], fp[i]))
    return mm


def _compare_fleets(seed: int, turn: int, ref: List[List[Any]], fast: List[List[Any]]) -> List[Mismatch]:
    mm: List[Mismatch] = []
    if len(ref) != len(fast):
        mm.append(Mismatch(seed, turn, "fleet_count", len(ref), len(fast)))
    ref_by_id = {f[0]: f for f in ref}
    fast_by_id = {f[0]: f for f in fast}
    if set(ref_by_id.keys()) != set(fast_by_id.keys()):
        mm.append(Mismatch(seed, turn, "fleet_ids",
                           sorted(ref_by_id.keys()), sorted(fast_by_id.keys())))
        return mm
    for fid, rf in ref_by_id.items():
        ff = fast_by_id[fid]
        # [id, owner, x, y, angle, from_planet_id, ships]
        for name, i in [("owner", 1), ("from_pid", 5), ("ships", 6)]:
            if rf[i] != ff[i]:
                mm.append(Mismatch(seed, turn, f"fleet[{fid}].{name}", rf[i], ff[i]))
        for name, i in [("x", 2), ("y", 3), ("angle", 4)]:
            if abs(rf[i] - ff[i]) > 1e-9:
                mm.append(Mismatch(seed, turn, f"fleet[{fid}].{name}", rf[i], ff[i]))
    return mm


def run_one_seed(seed: int, turns: int, verbose: bool = False) -> Tuple[int, List[Mismatch]]:
    """Run one validation match. Returns (turns_completed, mismatches)."""
    # Use seeded RNG for action generation; map init already seeded at build.
    action_rng = random.Random(seed * 7919 + 13)

    # Build the official env (this seeds its own random via our prior call).
    random.seed(seed)
    env = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)

    # Step the env once with no-op agents to trigger init. We use callable
    # agents to capture observations across calls.
    captured_obs = [None, None]

    def agent_factory(i):
        def agent(obs, cfg=None):
            captured_obs[i] = obs
            # First turn return empty; later turns use the driver below.
            # We'll control actions by returning from outside via a nonlocal.
            return pending_actions[i]
        agent.__name__ = f"val_agent_{i}"
        return agent

    pending_actions = [[], []]
    agents = [agent_factory(0), agent_factory(1)]

    # Start env: we set it running step by step via env.step().
    env.reset(num_agents=2)
    # After reset, the env has called interpreter with the init branch (or not —
    # kaggle-environments actually triggers init on first .run()). We call step
    # with empty actions once to trigger init.
    # Use env.step([[]]*2) — this runs agents just once.
    # Actually cleaner: emulate agent loop via env.run of a finite number of steps.

    # Strategy: wrap the loop manually with env.step().
    try:
        env.step([[], []])  # init call (agents see no planets, return nothing)
    except Exception as e:
        return 0, [Mismatch(seed, 0, "env.step init failed", None, str(e))]

    # Snapshot state into FastEngine. Pass rng=random (the module) to
    # share global random state with the reference engine — without this
    # comet ship generation would diverge since FastEngine normally
    # isolates its own RNG to avoid polluting MCTS rollouts.
    obs0 = env.state[0].observation
    fast = FastEngine.from_official_obs(
        obs0, num_agents=2, config=GameConfig(), rng=random,
    )

    mismatches: List[Mismatch] = []
    completed = 0

    for turn in range(1, turns + 1):
        # Build actions for this turn from current state (we use the SAME obs
        # for both engines so actions are deterministic — use official's obs).
        action_p0 = _random_actions(env.state[0].observation, action_rng)
        action_p1 = _random_actions(env.state[1].observation, action_rng)

        # Snapshot global random state so both engines consume random from the
        # same starting point. Without this, comet spawn paths diverge —
        # env.step advances `random`, fast.step then starts from a different
        # position and generates different paths.
        pre_state = random.getstate()

        # Apply to official
        try:
            env.step([action_p0, action_p1])
        except Exception as e:
            mismatches.append(Mismatch(seed, turn, "official env.step raised", None, str(e)))
            break
        post_official_state = random.getstate()

        # Rewind random state and apply to fast
        random.setstate(pre_state)
        try:
            fast.step([action_p0, action_p1])
        except Exception as e:
            mismatches.append(Mismatch(seed, turn, "fast.step raised", None, str(e)))
            break
        post_fast_state = random.getstate()

        # Sanity check: engines should have consumed the same amount of random.
        # (Not strictly required for parity, but a useful invariant.)
        if post_official_state != post_fast_state:
            mismatches.append(Mismatch(
                seed, turn, "random_consumption",
                "official and fast consumed different random amounts",
                "",
            ))

        # Compare
        ref_obs = env.state[0].observation
        ref_planets = list(ref_obs.planets)
        ref_fleets = list(ref_obs.fleets)
        fast_planets = fast.state.to_official_planets()
        fast_fleets = fast.state.to_official_fleets()

        turn_mm = []
        turn_mm.extend(_compare_planets(seed, turn, ref_planets, fast_planets))
        turn_mm.extend(_compare_fleets(seed, turn, ref_fleets, fast_fleets))

        # Compare step
        ref_step = getattr(ref_obs, "step", None)
        if ref_step is not None and ref_step != fast.state.step:
            turn_mm.append(Mismatch(seed, turn, "step", ref_step, fast.state.step))

        # Compare next_fleet_id
        ref_nfid = getattr(ref_obs, "next_fleet_id", None)
        if ref_nfid is not None and ref_nfid != fast.state.next_fleet_id:
            turn_mm.append(Mismatch(seed, turn, "next_fleet_id", ref_nfid, fast.state.next_fleet_id))

        mismatches.extend(turn_mm)
        completed = turn
        if turn_mm:
            if verbose:
                for m in turn_mm[:5]:
                    print(m)
                    print()
            # Stop on first divergence to keep output tractable.
            break

        # End if either engine reports done
        if env.done or fast.done:
            break

    return completed, mismatches


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10, help="Number of random seeds")
    ap.add_argument("--turns", type=int, default=30, help="Max turns per seed")
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--stop-on-fail", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    fail_seeds: List[int] = []
    total_mm = 0
    for s in range(args.start_seed, args.start_seed + args.seeds):
        completed, mm = run_one_seed(s, args.turns, verbose=args.verbose)
        if mm:
            fail_seeds.append(s)
            total_mm += len(mm)
            print(f"[seed={s}] MISMATCH at turn {completed}: {len(mm)} issue(s)")
            if not args.verbose:
                for m in mm[:3]:
                    print("  ", m)
            if args.stop_on_fail:
                break
        else:
            if args.verbose:
                print(f"[seed={s}] OK — {completed} turns match")
    elapsed = time.perf_counter() - t0
    ok = args.seeds - len(fail_seeds)
    print(f"\n{ok}/{args.seeds} seeds passed ({total_mm} mismatches) in {elapsed:.1f}s")
    return 0 if not fail_seeds else 1


if __name__ == "__main__":
    sys.exit(main())
