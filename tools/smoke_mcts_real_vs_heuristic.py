"""Live-game smoke test: MCTSAgent (REAL search, normal margin) vs HeuristicAgent.

Unlike `smoke_mcts_vs_heuristic.py` which forces anchor (margin=10.0),
this runs MCTS with its real search config:
  * Realistic sim budget.
  * anchor_improvement_margin=0.15 (default).

Expected after the RNG-isolation + anchor-floor fixes: MCTS should
beat the heuristic, or at least draw level. Previously (pre-fix) this
match lost 48-2965. The RNG-isolation fix restored heuristic parity;
this run confirms that search can now ALSO find improvements.
"""
from __future__ import annotations

import time

from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.bots.mcts_bot import MCTSAgent
from orbitwars.mcts.gumbel_search import GumbelConfig
from tournaments.harness import play_game


def main() -> None:
    mcts = MCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=4,          # 4 joint candidates at root
            total_sims=64,             # 16 sims/candidate after SH
            rollout_depth=20,          # deep rollouts
            hard_deadline_ms=500.0,    # 500ms per turn budget
            anchor_improvement_margin=0.3,  # balanced floor
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
