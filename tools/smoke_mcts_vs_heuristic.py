"""Live-game smoke test: MCTSAgent vs HeuristicAgent.

Runs a single 500-turn game with a generous per-turn timeout (so search
gets to complete rather than hit its internal deadline prematurely on a
cold start). Prints the final score and rewards; a healthy result after
the anchor fix is MCTS >= heuristic (never more than marginally behind).
"""
from __future__ import annotations

import time

from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.bots.mcts_bot import MCTSAgent
from orbitwars.mcts.actions import ActionConfig
from orbitwars.mcts.gumbel_search import GumbelConfig
from tournaments.harness import play_game


def main() -> None:
    mcts = MCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=2,      # tiny — speed up
            total_sims=2,          # tiny — speed up
            rollout_depth=1,
            hard_deadline_ms=50.0,  # short so turn cost is near-instant
            # Diagnostic: force MCTS to always return the anchor. If it
            # still loses, anchor reconstruction or MCTSAgent wiring is
            # broken.
            anchor_improvement_margin=10.0,
        ),
        rng_seed=0,
    )
    heur = HeuristicAgent()

    t0 = time.perf_counter()
    result = play_game(
        [mcts.as_kaggle_agent(), heur.as_kaggle_agent()],
        seed=42,
        players=2,
        step_timeout=2.0,
    )
    wall = time.perf_counter() - t0

    print(f"RESULT: rewards={result.rewards} final_scores={result.final_scores} "
          f"steps={result.steps} wall={wall:.1f}s")
    # Turn-time summary for the MCTS agent (already in ms).
    tt = result.turn_times_ms[0]
    if tt:
        n = len(tt)
        tts = sorted(tt)
        p50 = tts[n // 2]
        p95 = tts[int(n * 0.95)]
        p99 = tts[int(n * 0.99)]
        print(f"MCTS turn-time ms: p50={p50:.0f} p95={p95:.0f} p99={p99:.0f} "
              f"max={max(tt):.0f}")


if __name__ == "__main__":
    main()
