"""Play MCTS vs Heuristic in both seat orders to control for map bias.

On seed=42 player 0 has a significant map advantage (heuristic self-play
there produces 2221-252). So a single MCTS-as-P0 or MCTS-as-P1 smoke is
a biased estimator. Run both and average.
"""
from __future__ import annotations

import time

from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.bots.mcts_bot import MCTSAgent
from tournaments.harness import play_game


def run_one(mcts_seat: int, seed: int) -> None:
    mcts = MCTSAgent(rng_seed=0)
    heur = HeuristicAgent()
    agents = [mcts.as_kaggle_agent(), heur.as_kaggle_agent()]
    if mcts_seat == 1:
        agents = [heur.as_kaggle_agent(), mcts.as_kaggle_agent()]

    t0 = time.perf_counter()
    result = play_game(agents, seed=seed, players=2, step_timeout=2.0)
    wall = time.perf_counter() - t0

    mcts_reward = result.rewards[mcts_seat]
    mcts_score = result.final_scores[mcts_seat]
    heur_score = result.final_scores[1 - mcts_seat]
    tag = "WIN " if mcts_reward > 0 else ("LOSS" if mcts_reward < 0 else "TIE ")
    print(f"  mcts_seat={mcts_seat} seed={seed}: {tag} "
          f"mcts={mcts_score} heur={heur_score} wall={wall:.0f}s")


def main() -> None:
    print("MCTS (defaults) vs HeuristicAgent, both seats, seed=42:")
    run_one(0, 42)
    run_one(1, 42)
    print("\nSame, seed=123:")
    run_one(0, 123)
    run_one(1, 123)


if __name__ == "__main__":
    main()
