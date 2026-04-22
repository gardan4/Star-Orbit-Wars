"""Multi-seed MCTSAgent vs HeuristicAgent smoke.

Runs the default bot across several seeds in both seat positions to
check the win rate is stable (not a lucky seed=42 artifact). This is
the closest thing to a mini-ladder we have locally.
"""
from __future__ import annotations

import time
from typing import List, Tuple

from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.bots.mcts_bot import MCTSAgent
from tournaments.harness import play_game


def run_match(mcts_seat: int, seed: int) -> Tuple[int, int, float]:
    mcts = MCTSAgent(rng_seed=0)
    heur = HeuristicAgent()
    agents = [mcts.as_kaggle_agent(), heur.as_kaggle_agent()]
    if mcts_seat == 1:
        agents = [heur.as_kaggle_agent(), mcts.as_kaggle_agent()]

    t0 = time.perf_counter()
    result = play_game(agents, seed=seed, players=2, step_timeout=2.0)
    wall = time.perf_counter() - t0
    return (
        int(result.final_scores[mcts_seat]),
        int(result.final_scores[1 - mcts_seat]),
        wall,
    )


def main() -> None:
    seeds = [42, 123, 7]
    matches: List[Tuple[int, int, int, float, str]] = []
    print(f"Starting multi-seed smoke: seeds={seeds} x 2 seats = "
          f"{len(seeds) * 2} matches", flush=True)
    for s in seeds:
        for seat in (0, 1):
            mcts_score, heur_score, wall = run_match(seat, s)
            tag = "WIN " if mcts_score > heur_score else (
                "LOSS" if mcts_score < heur_score else "TIE "
            )
            print(f"  seed={s} mcts_seat={seat}: {tag} "
                  f"mcts={mcts_score} heur={heur_score} wall={wall:.0f}s",
                  flush=True)
            matches.append((s, seat, mcts_score - heur_score, wall, tag))

    wins = sum(1 for m in matches if m[2] > 0)
    losses = sum(1 for m in matches if m[2] < 0)
    ties = len(matches) - wins - losses
    total_diff = sum(m[2] for m in matches)
    print(f"\nSummary: {wins}W/{losses}L/{ties}T across {len(matches)} matches, "
          f"cum_score_diff={total_diff:+d}", flush=True)


if __name__ == "__main__":
    main()
