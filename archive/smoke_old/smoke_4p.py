"""4-player smoke: MCTS in seat=K vs Heur in the other 3 seats.

Rationale: STATUS.md §7 item 4 flags no 4-player coverage in W1-W3 smokes.
The entropy-leak fix in v7 might behave differently in 4p because comet
spawns are more frequent (4x quadrants) and there are 4 competing ship
totals instead of 2. We validate:
  1. MCTSAgent doesn't crash or timeout in 4p.
  2. MCTSAgent does not lose catastrophically to HeuristicAgent in 4p
     (target: not-worst, i.e. rank >= 2 of 4 in the final ship total).

Runs MCTS in each of the 4 seats, one at a time, then prints a summary.
Each game is ~250-400s wall-clock. Total ~20-25 min for all 4 seats.
"""
from __future__ import annotations

import time
from typing import List, Tuple

from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.bots.mcts_bot import MCTSAgent
from tournaments.harness import play_game


def run_match(mcts_seat: int, seed: int) -> Tuple[List[int], int, float]:
    agents = []
    for s in range(4):
        if s == mcts_seat:
            agents.append(MCTSAgent(rng_seed=0).as_kaggle_agent())
        else:
            agents.append(HeuristicAgent().as_kaggle_agent())
    t0 = time.perf_counter()
    result = play_game(agents, seed=seed, players=4, step_timeout=2.0)
    wall = time.perf_counter() - t0
    scores = list(result.final_scores)
    # rank: 0 = best, 3 = worst
    sorted_idx = sorted(range(4), key=lambda i: -scores[i])
    rank = sorted_idx.index(mcts_seat)
    return scores, rank, wall


def main() -> None:
    seeds = [42, 123]
    print(f"Starting 4p smoke: seeds={seeds} x 4 seats = {len(seeds) * 4} matches",
          flush=True)
    summary: List[Tuple[int, int, int, List[int], float]] = []
    for s in seeds:
        for seat in (0, 1, 2, 3):
            scores, rank, wall = run_match(seat, s)
            tag_rank = ["1st", "2nd", "3rd", "4th"][rank]
            mcts_score = scores[seat]
            max_other = max(scores[i] for i in range(4) if i != seat)
            delta = mcts_score - max_other
            print(
                f"  seed={s} mcts_seat={seat}: {tag_rank} "
                f"mcts={mcts_score} best_opp={max_other} "
                f"(delta={delta:+d}) wall={wall:.0f}s",
                flush=True,
            )
            summary.append((s, seat, rank, scores, wall))

    wins = sum(1 for m in summary if m[2] == 0)
    top2 = sum(1 for m in summary if m[2] <= 1)
    worst = sum(1 for m in summary if m[2] == 3)
    print(
        f"\nSummary: {wins} wins, {top2} top-2, {worst} last-place "
        f"across {len(summary)} matches",
        flush=True,
    )
    if worst >= len(summary) // 2:
        print("WARN: MCTS is last-place in >=50% of 4p games; 4p regime may have "
              "a residual bug beyond the v7 entropy-leak fix.")


if __name__ == "__main__":
    main()
