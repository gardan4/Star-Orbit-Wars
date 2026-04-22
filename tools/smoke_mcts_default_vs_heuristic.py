"""Definitive smoke: MCTSAgent with FACTORY DEFAULTS vs HeuristicAgent.

Uses MCTSAgent() with no config overrides to verify the shipped
default GumbelConfig beats HeuristicAgent on a canonical seed. This
is the "if users pip-install and run, what do they get?" test.
"""
from __future__ import annotations

import time

from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.bots.mcts_bot import MCTSAgent
from tournaments.harness import play_game


def main() -> None:
    # No config overrides — verify the default GumbelConfig is tuned.
    mcts = MCTSAgent(rng_seed=0)
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
