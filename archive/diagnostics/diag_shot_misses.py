"""Shot-miss diagnostic with engine-hooked ground truth.

We should have ~100% hit rate via analytic intercept math. If fleets miss,
something in the intercept -> engine pipeline is wrong.

Classification approach: instead of guessing from ship-delta heuristics
(which false-positive on coincidental combat), we monkey-patch the engine's
``interpreter`` function so ``combat_lists`` (the per-turn
``{planet_id: [fleet, ...]}`` dict that step-3/4/5 populate for step-6
combat) is captured after every step. That dict is ground truth for
"which fleet hit which planet this turn" and requires no guessing.

Usage:
    python -m tools.diag_shot_misses [--seed 42] [--opp heuristic]
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from kaggle_environments import make
import kaggle_environments.envs.orbit_wars.orbit_wars as _ow_mod

from orbitwars.bots.heuristic import (
    HeuristicAgent,
)
from orbitwars.bots.base import Deadline


# --- Engine instrumentation -----------------------------------------------

# Monkey-patch the orbit_wars interpreter so ``combat_lists`` (local inside
# ``interpreter``) is captured after every env.step. We install a tracer
# that watches the interpreter frame's return event and snapshots the
# dict's fleet-ids right before the frame is destroyed. This is the
# cleanest hook — no need to replicate ~200 lines of engine code.
#
# Note: kaggle_environments caches ``env.interpreter`` at ``make()`` time,
# so patching the module attribute doesn't flow through. We patch
# ``env.interpreter`` on the env instance in ``install_engine_hook``.

_captured: Dict[str, Any] = {"combat_lists": {}}


def _build_patched_interpreter(orig_interpreter):
    def _patched_interpreter(state, env):
        captured = {"ref": None}

        def tracer(frame, event, arg):
            if event != "call":
                return None
            if frame.f_code.co_name != "interpreter":
                return None

            def local_tracer(f, e, a):
                if e == "return":
                    cl = f.f_locals.get("combat_lists")
                    if cl is not None:
                        # Snapshot fleet ids per planet — copy out of the
                        # frame since refs die after return.
                        captured["ref"] = {
                            int(pid): [int(fl[0]) for fl in fleets]
                            for pid, fleets in cl.items()
                        }
                return local_tracer

            return local_tracer

        old_trace = sys.gettrace()
        sys.settrace(tracer)
        try:
            result = orig_interpreter(state, env)
        finally:
            sys.settrace(old_trace)
        _captured["combat_lists"] = captured["ref"] or {}
        return result

    # kaggle_environments inspects ``__code__.co_argcount`` to slice args,
    # so our wrapper must preserve the 2-argument signature. The def
    # above already has argcount=2, but alias __code__ explicitly for
    # defensive clarity.
    return _patched_interpreter


def install_engine_hook(env) -> None:
    """Replace env.interpreter with our capturing wrapper."""
    orig = env.interpreter
    env.interpreter = _build_patched_interpreter(orig)


# --- Per-fleet tracking ----------------------------------------------------

@dataclass
class ShotRecord:
    # Intent (from manifest)
    launch_turn: int
    from_pid: int
    target_pid: int
    angle: float
    ships: int
    predicted_arrival_turn: int
    # Outcome (filled in as we step)
    fleet_id: int = -1
    final_status: str = "in_flight"
    # Statuses: "hit", "miss", "wrong_planet", "alive_at_end",
    #           "target_vanished"
    actual_arrival_turn: int = -1
    actual_collided_pid: int = -1
    last_pos: Tuple[float, float] = (0.0, 0.0)


# --- Driver ----------------------------------------------------------------

def _fleets_by_id(obs) -> Dict[int, List[Any]]:
    return {int(f[0]): f for f in (obs.get("fleets") or [])}


def _planet_by_id(obs) -> Dict[int, List[Any]]:
    return {int(p[0]): p for p in (obs.get("planets") or [])}


def run(seed: int, opp: str, max_turns: int = 499, verbose: bool = False,
        my_seat: int = 0) -> List[ShotRecord]:
    # Kaggle's orbit_wars seeds initial layout from Python stdlib random
    # and numpy random. Seed both before `make` for reproducibility.
    import random as _pyrand
    import numpy as _np
    _pyrand.seed(seed)
    _np.random.seed(seed)
    env = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
    install_engine_hook(env)
    env.reset(num_agents=2)
    env.step([[], []])  # Kaggle reset convention

    my_agent = HeuristicAgent()
    opp_agent = HeuristicAgent()
    k_my = my_agent.as_kaggle_agent()
    k_opp = opp_agent.as_kaggle_agent() if opp == "heuristic" else opp

    shots: List[ShotRecord] = []
    by_fleet_id: Dict[int, int] = {}

    for step in range(max_turns):
        if env.state[0]["status"] != "ACTIVE":
            break
        obs_me = env.state[my_seat]["observation"]
        obs_opp = env.state[1 - my_seat]["observation"]

        pre_next_id = int(obs_me.get("next_fleet_id", 0))

        a_me = k_my(obs_me, env.configuration)
        intents = list(my_agent.last_launch_intents)

        a_opp = k_opp(obs_opp, env.configuration) if callable(k_opp) else []

        # Record intended shots. Engine assigns fleet IDs in wire order,
        # starting from ``pre_next_id``. Critically, ``process_moves`` is
        # called for each seat in order ``range(num_agents)`` — so
        # player 0's valid launches consume the first slots and player 1's
        # start at ``pre_next_id + (number of valid launches from player 0)``.
        #
        # We can't infer the count from opp_intents alone (opp is either a
        # HeuristicAgent or a raw callable without intents). So we use the
        # action list directly: ``len(a_opp)`` gives the wire-order count
        # of launches from the opponent. (Invalid launches — insufficient
        # ships, bad planet, etc. — consume no id, so this is a ceiling:
        # my fleet_id <= pre_next_id + len(a_opp) + i. Under-attribution
        # risk is minor; we prefer the correct ceiling to an always-wrong
        # offset of zero.)
        #
        # Edge case: for my_seat == 0 the offset is 0 by definition
        # (player 0 == me is processed first).
        if my_seat == 0:
            id_offset = 0
        else:
            # Count only valid-looking move entries (list of 3). Matches
            # ``process_moves``'s filter exactly.
            id_offset = sum(
                1 for m in (a_opp or [])
                if isinstance(m, (list, tuple)) and len(m) == 3
            )
        for i, it in enumerate(intents):
            shot = ShotRecord(
                launch_turn=it.turn,
                from_pid=it.from_pid,
                target_pid=it.target_pid,
                angle=it.angle,
                ships=it.ships,
                predicted_arrival_turn=it.predicted_arrival_turn,
                fleet_id=pre_next_id + id_offset + i,
            )
            shots.append(shot)
            by_fleet_id[shot.fleet_id] = len(shots) - 1

        # Step the env.
        if my_seat == 0:
            env.step([a_me, a_opp])
        else:
            env.step([a_opp, a_me])

        # Ground-truth collision map from the engine's combat_lists.
        combat_lists: Dict[int, List[int]] = _captured.get("combat_lists", {}) or {}
        # Invert to fleet_id -> planet_id for fast lookup.
        fid_to_pid: Dict[int, int] = {}
        for pid, fids in combat_lists.items():
            for fid in fids:
                fid_to_pid[fid] = pid

        obs_after = env.state[my_seat]["observation"]
        alive_fleets = _fleets_by_id(obs_after)
        planets_after = _planet_by_id(obs_after)

        for fid, idx in list(by_fleet_id.items()):
            shot = shots[idx]
            if shot.final_status != "in_flight":
                continue
            f = alive_fleets.get(fid)
            if f is not None:
                # Still alive — cache last seen position for forensics.
                shot.last_pos = (float(f[2]), float(f[3]))
                continue
            # --- Fleet died this turn. Classify from combat_lists. ---
            shot.actual_arrival_turn = step
            collided_pid = fid_to_pid.get(fid)
            if collided_pid is not None:
                shot.actual_collided_pid = collided_pid
                if collided_pid == shot.target_pid:
                    shot.final_status = "hit"
                else:
                    # Distinguish "target vanished mid-flight" from a real
                    # wrong-planet error: if the target no longer exists in
                    # planets_after, the comet expired while our fleet was
                    # en route. Engine-ground-truth says we hit *something*,
                    # but since the target disappeared first, this isn't a
                    # bot bug.
                    if shot.target_pid not in planets_after:
                        shot.final_status = "target_vanished"
                    else:
                        shot.final_status = "wrong_planet"
            else:
                # Fleet gone, not in any combat_list this turn -> died to
                # sun or out-of-bounds.
                if shot.target_pid not in planets_after:
                    shot.final_status = "target_vanished"
                else:
                    shot.final_status = "miss"
            del by_fleet_id[fid]

    # Any still in_flight at end-of-game never resolved.
    for shot in shots:
        if shot.final_status == "in_flight":
            shot.final_status = "alive_at_end"

    # Summary
    n = len(shots)
    hits = sum(1 for s in shots if s.final_status == "hit")
    misses = sum(1 for s in shots if s.final_status == "miss")
    wrong = sum(1 for s in shots if s.final_status == "wrong_planet")
    alive_end = sum(1 for s in shots if s.final_status == "alive_at_end")
    vanished = sum(1 for s in shots if s.final_status == "target_vanished")

    print(f"\nseed={seed} opp={opp}  shots={n}")
    print(f"  hit:            {hits}  ({100.0*hits/max(1,n):.1f}%)")
    print(f"  miss:           {misses}  ({100.0*misses/max(1,n):.1f}%)")
    print(f"  wrong_planet:   {wrong}  ({100.0*wrong/max(1,n):.1f}%)")
    print(f"  target_vanished:{vanished}  ({100.0*vanished/max(1,n):.1f}%)")
    print(f"  alive_at_end:   {alive_end}")

    miss_shots = [s for s in shots if s.final_status in ("miss", "wrong_planet")]
    if miss_shots:
        print(f"\nFirst {min(10, len(miss_shots))} misses:")
        for s in miss_shots[:10]:
            print(
                f"  fleet#{s.fleet_id} turn={s.launch_turn} "
                f"from={s.from_pid} -> target={s.target_pid} "
                f"ships={s.ships} angle={s.angle:.4f} "
                f"predicted_arrival={s.predicted_arrival_turn} "
                f"actual={s.actual_arrival_turn} "
                f"status={s.final_status}"
                + (f" collided={s.actual_collided_pid}"
                   if s.actual_collided_pid >= 0 else "")
            )

    return shots


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--opp", type=str, default="heuristic",
                    choices=["heuristic", "random"])
    ap.add_argument("--max-turns", type=int, default=499)
    ap.add_argument("--my-seat", type=int, default=0, choices=[0, 1])
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    run(args.seed, args.opp, args.max_turns, args.verbose, my_seat=args.my_seat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
