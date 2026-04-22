"""End-to-end time-budget audit for MCTSAgent under shipped defaults.

W2 verification gate requires p95 turn time < 800 ms (within Kaggle's
1-s actTimeout) and zero hard timeouts. The existing
`tools/profile_turn_times.py` runs one game; this audit aggregates
across multiple games and opponents so the percentiles are stable and
reflect ladder-style heterogeneity (HeuristicAgent isn't the only
opponent we'll face).

What we're measuring:
  * Total per-turn wall time for MCTSAgent(shipped defaults) — this is
    search + posterior.observe + action encoding + whatever else sits
    on the hot path. This is the number Kaggle compares to actTimeout.
  * Per-opponent breakdowns: some opponents produce more fleets,
    which inflates posterior observation cost. A ladder average is
    what matters for submission safety.

Output:
  * Aggregate p50/p90/p95/p99/max across all games.
  * Per-opponent percentiles so we can see if a specific opponent
    pushes us near-timeout.
  * Count of turns >= 850 ms (SEARCH_DEADLINE_MS — our internal guard
    should have kicked in) and >= 900 ms (HARD_DEADLINE_MS — we're
    risking Kaggle forfeit). Non-zero on either is a red flag.

Exit: 0 if p95 < 800 ms across all opponents AND zero turns >= 900 ms.
Otherwise 1 (shell-friendly gate).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Dict, List

from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.bots.mcts_bot import MCTSAgent
from orbitwars.mcts.gumbel_search import GumbelConfig
from orbitwars.opponent.archetypes import make_archetype
from tournaments.harness import play_game


def _percentiles(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {}
    xs = sorted(xs)
    n = len(xs)
    def pct(p: float) -> float:
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
    """Produce a fresh kaggle-agent callable for a given opponent name."""
    if name == "heuristic":
        return HeuristicAgent().as_kaggle_agent()
    if name == "random":
        # kaggle_environments built-in; passed as a string to play_game.
        return "random"
    if name == "starter":
        return "starter"
    # Otherwise assume it's an archetype name.
    return make_archetype(name).as_kaggle_agent()


def run_audit(
    opponents: List[str],
    n_games: int,
    seed_base: int,
    hard_deadline_ms: float,
    use_opponent_model: bool,
    anchor_improvement_margin: float,
    rollout_policy: str,
    step_timeout: float,
) -> Dict[str, Dict[str, float]]:
    """Play n_games against each opponent (alternating seats) and
    collect MCTSAgent turn times per opponent.

    Returns a dict opponent -> percentiles dict, plus an "all" key with
    the merged distribution.
    """
    per_opp: Dict[str, List[float]] = {o: [] for o in opponents}
    all_times: List[float] = []
    over_850_total = 0
    over_900_total = 0

    print(
        f"Shipped defaults: margin={anchor_improvement_margin} "
        f"use_opponent_model={use_opponent_model} rollout_policy={rollout_policy} "
        f"hard_deadline_ms={hard_deadline_ms}"
    )
    print(
        f"Opponents: {opponents}   N-games (each seat): {n_games}   "
        f"seed_base={seed_base}"
    )
    print()

    t_start = time.perf_counter()
    for opp_name in opponents:
        print(f"=== vs {opp_name} ===", flush=True)
        for g in range(n_games):
            seat = g % 2  # alternate seats: 0,1,0,1,...
            seed = seed_base + 10000 * opponents.index(opp_name) + g
            mcts = MCTSAgent(
                gumbel_cfg=GumbelConfig(
                    num_candidates=4, total_sims=32, rollout_depth=15,
                    hard_deadline_ms=hard_deadline_ms,
                    anchor_improvement_margin=anchor_improvement_margin,
                    rollout_policy=rollout_policy,
                ),
                rng_seed=0,
                use_opponent_model=use_opponent_model,
            )
            mcts_agent = mcts.as_kaggle_agent()
            opp = _mk_opponent(opp_name)

            if seat == 0:
                agents = [mcts_agent, opp]
            else:
                agents = [opp, mcts_agent]

            result = play_game(agents, seed=seed, players=2, step_timeout=step_timeout)
            mcts_times = result.turn_times_ms[seat]
            per_opp[opp_name].extend(mcts_times)
            all_times.extend(mcts_times)

            over_850 = sum(1 for t in mcts_times if t >= 850.0)
            over_900 = sum(1 for t in mcts_times if t >= 900.0)
            over_850_total += over_850
            over_900_total += over_900

            tag = ""
            if over_900:
                tag = f"  !! {over_900} turns >= 900ms"
            elif over_850:
                tag = f"  ! {over_850} turns >= 850ms"
            print(
                f"  seed={seed} seat={seat}: steps={result.steps} "
                f"scores={result.final_scores} "
                f"mcts_turns={len(mcts_times):>3d}{tag}",
                flush=True,
            )

    wall = time.perf_counter() - t_start
    print()
    print(f"Total wall: {wall:.0f} s")
    print()

    print("Per-opponent turn-time percentiles (ms):")
    print(f"  {'opp':>13s}  {'n':>5s}  {'mean':>7s}  {'p50':>7s}  {'p90':>7s}  {'p95':>7s}  {'p99':>7s}  {'max':>7s}")
    results: Dict[str, Dict[str, float]] = {}
    for opp in opponents:
        pct = _percentiles(per_opp[opp])
        results[opp] = pct
        if not pct:
            continue
        print(
            f"  {opp:>13s}  {pct['n']:>5d}  {pct['mean']:>7.1f}  "
            f"{pct['p50']:>7.1f}  {pct['p90']:>7.1f}  "
            f"{pct['p95']:>7.1f}  {pct['p99']:>7.1f}  {pct['max']:>7.1f}"
        )
    print()

    # Merged distribution across opponents is the honest "ladder-average"
    # view. Per-opponent is the tail-risk view.
    pct_all = _percentiles(all_times)
    results["_all"] = pct_all
    print(
        f"ALL (merged):  n={pct_all['n']}  mean={pct_all['mean']:.1f}  "
        f"p50={pct_all['p50']:.1f}  p90={pct_all['p90']:.1f}  "
        f"p95={pct_all['p95']:.1f}  p99={pct_all['p99']:.1f}  "
        f"max={pct_all['max']:.1f}"
    )
    print(
        f"Timeout-zone turns: "
        f">= 850 ms: {over_850_total}   >= 900 ms: {over_900_total}"
    )

    # Gate: p95 < 800 ms and zero 900ms turns.
    gate_pass = pct_all["p95"] < 800.0 and over_900_total == 0
    print()
    if gate_pass:
        print("PASS: W2 budget gate held (p95 < 800 ms, 0 turns >= 900 ms)")
    else:
        reasons = []
        if pct_all["p95"] >= 800.0:
            reasons.append(f"p95={pct_all['p95']:.0f} ms >= 800 ms")
        if over_900_total:
            reasons.append(f"{over_900_total} turns >= 900 ms")
        print("FAIL: " + " ; ".join(reasons))

    return results, gate_pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--opponents", nargs="+", type=str,
        default=["heuristic", "random", "rusher", "turtler", "defender"],
        help="Opponent names. Built-ins: heuristic, random, starter. Archetypes: "
             "rusher, turtler, defender, harasser, economy, opportunist, comet_camper, sun_gambler.",
    )
    ap.add_argument("--n-games", type=int, default=4,
                    help="Games per opponent (seats alternate)")
    ap.add_argument("--seed-base", type=int, default=20260422)
    ap.add_argument("--hard-deadline-ms", type=float, default=300.0)
    ap.add_argument(
        "--step-timeout", type=float, default=2.0,
        help="Engine actTimeout for the harness. Set well above "
             "hard-deadline-ms so we can actually measure long turns "
             "without the engine forfeiting them.",
    )
    ap.add_argument("--margin", type=float, default=2.0,
                    help="anchor_improvement_margin (2.0 = shipped default)")
    ap.add_argument(
        "--use-opponent-model", dest="use_opponent_model",
        action="store_true", default=True,
    )
    ap.add_argument(
        "--no-opponent-model", dest="use_opponent_model",
        action="store_false",
    )
    ap.add_argument(
        "--rollout-policy", type=str, default="heuristic",
        choices=["heuristic", "fast"],
    )
    args = ap.parse_args()

    _, gate_pass = run_audit(
        opponents=args.opponents,
        n_games=args.n_games,
        seed_base=args.seed_base,
        hard_deadline_ms=args.hard_deadline_ms,
        use_opponent_model=args.use_opponent_model,
        anchor_improvement_margin=args.margin,
        rollout_policy=args.rollout_policy,
        step_timeout=args.step_timeout,
    )
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
