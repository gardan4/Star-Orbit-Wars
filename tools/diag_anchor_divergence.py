"""Diagnostic: MCTS should anchor-lock at margin=2.0, yet it loses to the
same heuristic it uses as its floor. Why?

This script runs a full 499-turn game with MCTSAgent vs HeuristicAgent on
the seed/seat combo that failed the audit (20260422 seat=0), while also
capturing the standalone heuristic's action on the SAME obs. For every
turn we log:
  * whether MCTS's returned wire equals the fallback's staged wire
  * whether search overrode the anchor (best_i != 0 and guard didn't fire)
  * anchor_q, winner_q, gap (the data the guard makes its decision on)

If divergences cluster late-game at Q-gap == 2.0, the fix is to either
make the margin strict-inequality `>=` (already is `<` so guard fires at
exactly 2.0) or tighten the margin to e.g. `< 2.1` to cover the boundary.

If divergences appear mid-game, something subtler is going on \u2014 e.g.
the anchor's wire doesn't round-trip through _build_anchor_joint exactly,
or the search returns a JointAction with a wire that differs from the
heuristic's even when best_i == 0.

Usage:
    python -m tools.diag_anchor_divergence
"""
from __future__ import annotations

import random
import sys
from typing import Any, Dict, List, Optional, Tuple

from kaggle_environments import make

from orbitwars.bots.base import Deadline, HARD_DEADLINE_MS
from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.bots.mcts_bot import MCTSAgent
from orbitwars.mcts.gumbel_search import GumbelRootSearch, SearchResult


# --- Instrumented search subclass -------------------------------------------

class InstrumentedSearch(GumbelRootSearch):
    """Captures anchor_q, winner_q, and guard state from each search."""
    last_stats: Dict[str, Any] = {}

    def search(self, obs, my_player, num_agents=2, start_time=None,
               anchor_action=None):
        result = super().search(
            obs, my_player, num_agents=num_agents,
            start_time=start_time, anchor_action=anchor_action,
        )
        if result is None:
            InstrumentedSearch.last_stats = {"result": None}
            return result
        q = list(result.q_values)
        anchor_q = q[0] if q else None
        winner_q = max(q) if q else None
        gap = (winner_q - anchor_q) if (q and anchor_q is not None) else None
        # Figure out if the guard fired (anchor returned). We don't have
        # the pre-guard best_joint, but we can infer: if the winner's Q
        # was confidently better than anchor yet best_joint IS anchor,
        # guard fired. If best_joint is NOT the anchor, guard did not fire.
        # We'll just log whether best_joint is anchor-by-identity via the
        # wire round-trip (best_joint's wire vs the anchor's wire).
        InstrumentedSearch.last_stats = {
            "n_rollouts": result.n_rollouts,
            "q_values": q,
            "anchor_q": anchor_q,
            "winner_q": winner_q,
            "gap": gap,
            "aborted": result.aborted,
            "duration_ms": result.duration_ms,
        }
        return result


class InstrumentedMCTS(MCTSAgent):
    """MCTSAgent that routes all search through InstrumentedSearch and
    logs, per turn, what the staged heuristic wire was vs. what we
    ultimately returned."""

    name = "mcts_instr"

    last_turn_info: Dict[str, Any] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the search with an instrumented variant. Same config
        # object reference so mcts_bot's act() swap-restore of gumbel_cfg
        # still works.
        inst = InstrumentedSearch(
            weights=self._search.weights,
            action_cfg=self._search.action_cfg,
            gumbel_cfg=self._search.gumbel_cfg,
            rng_seed=0,
        )
        inst.opp_policy_override = self._search.opp_policy_override
        inst.opp_candidate_builder = self._search.opp_candidate_builder
        self._search = inst

    def act(self, obs, deadline):
        # Capture the heuristic's wire BEFORE any search by calling an
        # ephemeral second HeuristicAgent with the same state as _fallback.
        # We can't just introspect _fallback mid-turn without mutating it,
        # so we instrument the staged action via deadline.best() after
        # super().act completes.
        action = super().act(obs, deadline)
        # `deadline._best` was set to `heuristic_move` on line 256 of
        # mcts_bot.py, BEFORE search. If search overrode successfully,
        # `action` differs from deadline.best() at that moment \u2014 but
        # deadline.best() has since been set to `action` by line 387.
        # Workaround: we can read _best AFTER line 387 to see the final
        # stage, but that's the returned action, not the pre-search one.
        # Cleaner approach: monkey-patch deadline.stage to snapshot the
        # FIRST non-trivial stage (which is the heuristic move). But
        # even simpler \u2014 replicate the heuristic call here using the
        # same _fallback and the same obs. Since _fallback's state has
        # already mutated from the super().act call, we cannot. So we
        # have to rely on what super().act set.
        #
        # For this diagnostic we'll just log the search stats; the
        # match-test done at the driver level compares MCTS's returned
        # wire to a separately-run standalone heuristic's wire.
        InstrumentedMCTS.last_turn_info = {
            "returned_wire": action,
            "search_stats": dict(InstrumentedSearch.last_stats or {}),
        }
        return action


# --- Driver -----------------------------------------------------------------

def _norm_wire(a):
    """Canonicalize a wire list for order-insensitive comparison."""
    out = []
    for m in a or []:
        if isinstance(m, (list, tuple)) and len(m) >= 3:
            out.append((int(m[0]), round(float(m[1]), 4), int(m[2])))
    return sorted(out)


def run(seed: int, my_seat: int, max_turns: int = 499) -> None:
    random.seed(seed)
    env = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
    env.reset(num_agents=2)
    env.step([[], []])  # Kaggle's reset convention \u2014 must run the reset step

    # Two fresh HeuristicAgents: one "shadow" that sees the same obs as
    # MCTS, independently of the MCTS bot's internal _fallback. We use
    # the shadow as the ground-truth "what would the heuristic do here".
    mcts = InstrumentedMCTS(rng_seed=0)
    shadow_heur = HeuristicAgent()
    opp_heur = HeuristicAgent()

    k_mcts = mcts.as_kaggle_agent()
    k_shadow = shadow_heur.as_kaggle_agent()
    k_opp = opp_heur.as_kaggle_agent()

    divergences: List[Tuple[int, Any]] = []

    for step in range(max_turns):
        if env.state[0]["status"] != "ACTIVE":
            break
        obs_me = env.state[my_seat]["observation"]
        obs_opp = env.state[1 - my_seat]["observation"]

        # MCTS gets obs_me. Shadow heuristic also gets obs_me so it sees
        # EXACTLY what MCTS saw this turn.
        a_mcts = k_mcts(obs_me, env.configuration)
        a_shadow = k_shadow(obs_me, env.configuration)

        info = InstrumentedMCTS.last_turn_info
        ss = info.get("search_stats", {})

        n_m, n_h = _norm_wire(a_mcts), _norm_wire(a_shadow)
        if n_m != n_h:
            divergences.append((step, {
                "gap": ss.get("gap"),
                "anchor_q": ss.get("anchor_q"),
                "winner_q": ss.get("winner_q"),
                "n_rollouts": ss.get("n_rollouts"),
                "aborted": ss.get("aborted"),
                "extra": sorted(set(n_m) - set(n_h)),
                "missing": sorted(set(n_h) - set(n_m)),
            }))

        # Step env: MCTS plays its seat, opp heuristic plays the other.
        a_opp = k_opp(obs_opp, env.configuration)
        if my_seat == 0:
            env.step([a_mcts, a_opp])
        else:
            env.step([a_opp, a_mcts])

    # Summary
    print(f"\nseed={seed} seat={my_seat}  total_steps={step}")
    scores_final = [int(env.state[i]["observation"].get("reward", 0) or 0) for i in range(2)]
    try:
        # Ship totals
        obs_final = env.state[0]["observation"]
        my_ships = sum(p[5] for p in obs_final.get("planets", []) if p[1] == my_seat)
        opp_ships = sum(p[5] for p in obs_final.get("planets", []) if p[1] == (1 - my_seat))
        my_fleet = sum(f[6] for f in obs_final.get("fleets", []) if f[1] == my_seat)
        opp_fleet = sum(f[6] for f in obs_final.get("fleets", []) if f[1] == (1 - my_seat))
        print(f"  my={my_ships}+{my_fleet}={my_ships+my_fleet}  "
              f"opp={opp_ships}+{opp_fleet}={opp_ships+opp_fleet}")
    except Exception as e:
        print(f"  ship tally failed: {e}")
    print(f"  divergences: {len(divergences)} / {step}")

    # Classify divergences
    gap_2_plus = [d for d in divergences if (d[1].get("gap") or 0) >= 2.0]
    gap_under_2 = [d for d in divergences if 0 < (d[1].get("gap") or 0) < 2.0]
    gap_zero_or_negative = [d for d in divergences if (d[1].get("gap") or 0) <= 0]
    aborted = [d for d in divergences if d[1].get("aborted")]

    print(f"  gap>=2.0: {len(gap_2_plus)}  gap(0,2): {len(gap_under_2)}  "
          f"gap<=0: {len(gap_zero_or_negative)}  aborted: {len(aborted)}")

    print("\nFirst 10 divergences:")
    for step, d in divergences[:10]:
        print(f"  step={step} gap={d['gap']} "
              f"anchor_q={d['anchor_q']} winner_q={d['winner_q']} "
              f"n_roll={d['n_rollouts']} abort={d['aborted']}")
        print(f"    extra={d['extra'][:3]}  missing={d['missing'][:3]}")

    if divergences:
        print("\nLast 5 divergences:")
        for step, d in divergences[-5:]:
            print(f"  step={step} gap={d['gap']} "
                  f"anchor_q={d['anchor_q']} winner_q={d['winner_q']}")


def main():
    # Failing seed from the audit.
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20260422
    seat = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    max_turns = int(sys.argv[3]) if len(sys.argv) > 3 else 499
    run(seed, seat, max_turns)


if __name__ == "__main__":
    main()
