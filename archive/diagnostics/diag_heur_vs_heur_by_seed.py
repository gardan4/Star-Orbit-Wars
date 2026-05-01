"""Baseline: HeuristicAgent vs HeuristicAgent on the same seeds/seats.

Purpose: the multi-seed MCTS smoke shows seat-1 MCTS scoring 0 on
seeds 123 and 7. Since MCTSAgent is anchor-locked (margin=2.0), its
wire actions should be byte-identical to HeuristicAgent's at 0/30
divergence rate. If heuristic-vs-heuristic on the same seeds shows
seat 1 also scoring ~0, the "blowout" is map imbalance, not a bug.
"""
from __future__ import annotations

import time

from orbitwars.bots.heuristic import HeuristicAgent
from tournaments.harness import play_game


def main() -> None:
    seeds = [42, 123, 7]
    print(f"Heur-vs-heur baseline: seeds={seeds} x 2 seats", flush=True)
    for s in seeds:
        for seat in (0, 1):
            h0 = HeuristicAgent()
            h1 = HeuristicAgent()
            agents = [h0.as_kaggle_agent(), h1.as_kaggle_agent()]
            t0 = time.perf_counter()
            result = play_game(agents, seed=s, players=2, step_timeout=2.0)
            wall = time.perf_counter() - t0
            # Report from the perspective of "if MCTSAgent were at `seat`".
            our_score = int(result.final_scores[seat])
            their_score = int(result.final_scores[1 - seat])
            tag = "WIN " if our_score > their_score else (
                "LOSS" if our_score < their_score else "TIE "
            )
            print(f"  seed={s} seat_of_interest={seat}: {tag} "
                  f"us={our_score} them={their_score} wall={wall:.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()
