"""Per-turn breakdown profile: split MCTSAgent.act wall time into buckets.

Goal: attribute tail-time to a specific stage. The audit revealed turns
>= 900 ms on some seeds; action-gen micro-profiling showed only ~1.65 ms
difference from BOKR, so the tail source must be elsewhere (rollouts,
posterior update, heuristic fallback, or engine init).

Buckets measured per turn:
  * ``heur_fallback_ms``  — self._fallback.act() (HeuristicAgent)
  * ``posterior_ms``      — opp_posterior.observe + route-to-search
  * ``search_prep_ms``    — parse_obs + build_arrival_table + generate_moves
                            + anchor + enumerate_joints + FastEngine.from_official_obs
  * ``rollouts_ms``       — sequential_halving / decoupled_ucb_root
  * ``wrapup_ms``         — to_wire + stage
  * ``total_ms``          — measured act() wall time (sanity; should = sum)

Usage:
  python tools/profile_per_turn_breakdown.py --seed 20260423 --seat 1 \
         --opponent heuristic

Output: per-bucket p50/p95/max across one full game + the top-10 slowest
turns with full bucket breakdown. This is what tells us "the 1156-ms
max was 800 ms of rollouts + 50 ms of posterior" vs. "50 ms rollouts +
950 ms of posterior blowup".

Defaults match the shipped MCTSAgent config: hard_deadline_ms=300,
margin=2.0, use_opponent_model=True, rollout_policy=heuristic.
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Any, Dict, List

from orbitwars.bots.base import Deadline
from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.bots.mcts_bot import MCTSAgent
from orbitwars.opponent.archetypes import make_archetype
from tournaments.harness import play_game


# ---------------------------------------------------------------------------
# Instrumented MCTSAgent subclass — wraps each stage with perf_counter.
# ---------------------------------------------------------------------------

class InstrumentedMCTSAgent(MCTSAgent):
    """MCTSAgent that writes per-stage wall times to self.breakdown[].

    The breakdown list grows one entry per turn. Each entry is a dict
    with keys: step, total_ms, heur_fallback_ms, posterior_ms,
    search_prep_ms, rollouts_ms, wrapup_ms, n_rollouts, search_aborted.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.breakdown: List[Dict[str, Any]] = []

    def act(self, obs: Any, deadline: Deadline):
        from orbitwars.bots.base import HARD_DEADLINE_MS, no_op, obs_get
        from orbitwars.bots.heuristic import HeuristicAgent
        from orbitwars.mcts.gumbel_search import GumbelConfig, GumbelRootSearch
        from orbitwars.opponent.bayes import ArchetypePosterior

        step = int(obs_get(obs, "step", 0))
        entry: Dict[str, Any] = {
            "step": step,
            "total_ms": 0.0,
            "heur_fallback_ms": 0.0,
            "posterior_ms": 0.0,
            "search_prep_ms": 0.0,
            "rollouts_ms": 0.0,
            "wrapup_ms": 0.0,
            "n_rollouts": 0,
            "search_aborted": False,
        }

        act_t0 = time.perf_counter()
        deadline.stage(no_op())

        # --- Heuristic fallback -------------------------------------------
        t = time.perf_counter()
        try:
            heuristic_move = self._fallback.act(obs, deadline)
            deadline.stage(heuristic_move)
        except Exception:
            heuristic_move = no_op()
        entry["heur_fallback_ms"] = (time.perf_counter() - t) * 1000.0

        # --- Turn-0 reset (cheap, folded into posterior bucket) -----------
        if step == 0:
            self._fallback = HeuristicAgent(weights=self.weights)
            self._search = GumbelRootSearch(
                weights=self.weights,
                action_cfg=self.action_cfg,
                gumbel_cfg=self.gumbel_cfg,
                rng_seed=None,
            )
            if self._use_opponent_model:
                self.opp_posterior = ArchetypePosterior()
            self._search.opp_policy_override = None
            self._search.opp_candidate_builder = None
            self.telemetry = {
                "turns_observed": 0,
                "override_fires": 0,
                "override_clears": 0,
                "builder_fires": 0,
                "builder_clears": 0,
                "last_top_name": None,
                "last_top_prob": 0.0,
            }

        my_player = int(obs_get(obs, "player", 0))

        # --- Posterior observe + route ------------------------------------
        t = time.perf_counter()
        if self._use_opponent_model and self.opp_posterior is not None:
            try:
                opp_player = 1 - my_player
                self.opp_posterior.observe(obs, opp_player=opp_player)
                self._maybe_route_posterior_to_search()
            except Exception:
                pass
        entry["posterior_ms"] = (time.perf_counter() - t) * 1000.0

        remaining = deadline.remaining_ms(HARD_DEADLINE_MS)
        if remaining < 50.0:
            entry["total_ms"] = (time.perf_counter() - act_t0) * 1000.0
            self.breakdown.append(entry)
            return heuristic_move

        _ROLLOUT_OVERSHOOT_BUDGET_MS = 260.0
        _WRAPUP_BUDGET_MS = 40.0
        safe_budget = min(
            self.gumbel_cfg.hard_deadline_ms,
            remaining - _ROLLOUT_OVERSHOOT_BUDGET_MS - _WRAPUP_BUDGET_MS,
        )
        if safe_budget <= 10.0:
            entry["total_ms"] = (time.perf_counter() - act_t0) * 1000.0
            self.breakdown.append(entry)
            return heuristic_move

        tight_cfg = GumbelConfig(
            num_candidates=self.gumbel_cfg.num_candidates,
            total_sims=self.gumbel_cfg.total_sims,
            rollout_depth=self.gumbel_cfg.rollout_depth,
            hard_deadline_ms=safe_budget,
            anchor_improvement_margin=self.gumbel_cfg.anchor_improvement_margin,
        )

        # --- Search: split into prep vs. rollouts via monkey-patch -------
        # We intercept `sequential_halving` and `decoupled_ucb_root` on the
        # gumbel_search module — any time inside them is the "rollouts"
        # bucket; the rest of search() is "prep". This is clean and
        # doesn't require rewriting the search flow.
        from orbitwars.mcts import gumbel_search as _gs
        from orbitwars.mcts import sim_move as _sm

        real_sh = _gs.sequential_halving
        real_dc = _sm.decoupled_ucb_root
        rollout_ms_holder = {"ms": 0.0, "n": 0, "aborted": False}

        def timed_sh(*a, **kw):
            t0 = time.perf_counter()
            r = real_sh(*a, **kw)
            rollout_ms_holder["ms"] += (time.perf_counter() - t0) * 1000.0
            rollout_ms_holder["n"] += int(r.n_rollouts or 0)
            rollout_ms_holder["aborted"] = rollout_ms_holder["aborted"] or bool(r.aborted)
            return r

        def timed_dc(*a, **kw):
            t0 = time.perf_counter()
            r = real_dc(*a, **kw)
            rollout_ms_holder["ms"] += (time.perf_counter() - t0) * 1000.0
            rollout_ms_holder["n"] += int(r.n_rollouts or 0)
            rollout_ms_holder["aborted"] = rollout_ms_holder["aborted"] or bool(r.aborted)
            return r

        saved_cfg = None
        search_t0 = time.perf_counter()
        result = None
        try:
            saved_cfg = self._search.gumbel_cfg
            self._search.gumbel_cfg = tight_cfg
            _gs.sequential_halving = timed_sh
            _sm.decoupled_ucb_root = timed_dc
            result = self._search.search(
                obs, my_player, start_time=search_t0,
                anchor_action=heuristic_move,
            )
        except Exception:
            pass
        finally:
            _gs.sequential_halving = real_sh
            _sm.decoupled_ucb_root = real_dc
            if saved_cfg is not None:
                try:
                    self._search.gumbel_cfg = saved_cfg
                except Exception:
                    pass

        search_total_ms = (time.perf_counter() - search_t0) * 1000.0
        entry["rollouts_ms"] = rollout_ms_holder["ms"]
        entry["search_prep_ms"] = max(0.0, search_total_ms - rollout_ms_holder["ms"])
        entry["n_rollouts"] = rollout_ms_holder["n"]
        entry["search_aborted"] = rollout_ms_holder["aborted"]

        # --- Wrapup --------------------------------------------------------
        t = time.perf_counter()
        if result is None:
            out = heuristic_move
        else:
            out = result.best_joint.to_wire()
            deadline.stage(out)
        entry["wrapup_ms"] = (time.perf_counter() - t) * 1000.0

        entry["total_ms"] = (time.perf_counter() - act_t0) * 1000.0
        self.breakdown.append(entry)
        return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _percentiles(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"n": 0}
    xs = sorted(xs)
    n = len(xs)
    def pct(p):
        k = max(0, min(n - 1, int(round(p * (n - 1)))))
        return xs[k]
    return {
        "n": n,
        "mean": statistics.fmean(xs),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": xs[-1],
    }


def _mk_opponent(name: str):
    if name == "heuristic":
        return HeuristicAgent().as_kaggle_agent()
    if name == "random":
        return "random"
    return make_archetype(name).as_kaggle_agent()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260423)
    ap.add_argument("--seat", type=int, choices=(0, 1), default=1)
    ap.add_argument("--opponent", type=str, default="heuristic")
    ap.add_argument("--hard-deadline-ms", type=float, default=300.0)
    ap.add_argument("--step-timeout", type=float, default=2.0)
    ap.add_argument("--top-slowest", type=int, default=10)
    args = ap.parse_args()

    from orbitwars.mcts.gumbel_search import GumbelConfig
    mcts = InstrumentedMCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=4, total_sims=32, rollout_depth=15,
            hard_deadline_ms=args.hard_deadline_ms,
            anchor_improvement_margin=2.0,
            rollout_policy="heuristic",
        ),
        rng_seed=0,
    )

    opp = _mk_opponent(args.opponent)
    agents = [mcts.as_kaggle_agent(), opp] if args.seat == 0 else [opp, mcts.as_kaggle_agent()]

    print(
        f"Seed={args.seed} seat={args.seat} opp={args.opponent} "
        f"hard_deadline_ms={args.hard_deadline_ms}"
    )
    t0 = time.perf_counter()
    result = play_game(agents, seed=args.seed, players=2, step_timeout=args.step_timeout)
    wall = time.perf_counter() - t0
    print(
        f"Game: steps={result.steps} scores={result.final_scores} "
        f"wall={wall:.1f}s"
    )
    print()

    b = mcts.breakdown
    if not b:
        print("No breakdown captured — did the game even run?")
        return 1

    buckets = [
        "total_ms", "heur_fallback_ms", "posterior_ms",
        "search_prep_ms", "rollouts_ms", "wrapup_ms",
    ]
    print(f"Per-turn bucket percentiles (n={len(b)} turns):")
    print(f"  {'bucket':>18s}  {'mean':>7s}  {'p50':>6s}  {'p90':>6s}  {'p95':>6s}  {'p99':>6s}  {'max':>7s}")
    for name in buckets:
        vs = [e[name] for e in b]
        pct = _percentiles(vs)
        print(
            f"  {name:>18s}  {pct['mean']:>7.2f}  {pct['p50']:>6.2f}  "
            f"{pct['p90']:>6.2f}  {pct['p95']:>6.2f}  {pct['p99']:>6.2f}  "
            f"{pct['max']:>7.2f}"
        )

    over_850 = [e for e in b if e["total_ms"] >= 850.0]
    over_900 = [e for e in b if e["total_ms"] >= 900.0]
    print()
    print(f"Turns >= 850ms: {len(over_850)}")
    print(f"Turns >= 900ms: {len(over_900)}")

    print()
    print(f"Top-{args.top_slowest} slowest turns (bucket breakdown, ms):")
    print(
        f"  {'step':>5s} {'total':>7s} {'heur':>6s} {'post':>6s} "
        f"{'prep':>6s} {'roll':>7s} {'wrap':>5s} {'n_sim':>5s} {'abort':>5s}"
    )
    for e in sorted(b, key=lambda e: -e["total_ms"])[: args.top_slowest]:
        print(
            f"  {e['step']:>5d} {e['total_ms']:>7.1f} "
            f"{e['heur_fallback_ms']:>6.1f} {e['posterior_ms']:>6.1f} "
            f"{e['search_prep_ms']:>6.1f} {e['rollouts_ms']:>7.1f} "
            f"{e['wrapup_ms']:>5.1f} {e['n_rollouts']:>5d} "
            f"{'Y' if e['search_aborted'] else 'N':>5s}"
        )

    # Attribute the worst spike: which bucket is driving it?
    if over_900 or over_850:
        spike = max(b, key=lambda e: e["total_ms"])
        rem = {
            "heur_fallback_ms": spike["heur_fallback_ms"],
            "posterior_ms": spike["posterior_ms"],
            "search_prep_ms": spike["search_prep_ms"],
            "rollouts_ms": spike["rollouts_ms"],
            "wrapup_ms": spike["wrapup_ms"],
        }
        worst_bucket = max(rem.items(), key=lambda kv: kv[1])
        print()
        print(
            f"Worst spike step={spike['step']} total={spike['total_ms']:.0f}ms — "
            f"top bucket: {worst_bucket[0]} = {worst_bucket[1]:.0f}ms"
        )

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
