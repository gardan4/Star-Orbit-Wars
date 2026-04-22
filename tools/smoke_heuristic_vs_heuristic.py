"""Reference run: HeuristicAgent vs HeuristicAgent on the same seed used
for the MCTS smoke test. If the MCTS smoke test result matches this
one closely, the anchor+margin guard is working (MCTS has degenerated
to heuristic-equivalent play), and any apparent "loss" is just seed-
dependent first-mover advantage — not an MCTS bug.
"""
from __future__ import annotations

import time

from orbitwars.bots.heuristic import HeuristicAgent
from tournaments.harness import play_game


def main() -> None:
    h1 = HeuristicAgent()
    h2 = HeuristicAgent()
    t0 = time.perf_counter()
    result = play_game(
        [h1.as_kaggle_agent(), h2.as_kaggle_agent()],
        seed=42,
        players=2,
        step_timeout=2.0,
    )
    wall = time.perf_counter() - t0
    print(f"RESULT: rewards={result.rewards} final_scores={result.final_scores} "
          f"steps={result.steps} wall={wall:.1f}s")


if __name__ == "__main__":
    main()
