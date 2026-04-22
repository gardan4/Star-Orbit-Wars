"""A/B smoke: MCTSAgent with BOKR angle refinement ON vs OFF.

Configuration:
  * A = MCTSAgent() with shipped defaults — angle_refinement_n_grid=3,
        range=0.1 rad (base + 2 offsets per target+fraction pair).
  * B = MCTSAgent(action_cfg=ActionConfig(n_grid=1)) — single-angle
        (pre-BOKR) behavior.

For each seed we run both seat positions and tally score deltas. The
null hypothesis is ``A <= B``; if angle refinement is net-positive the
A-wins-minus-B-wins should trend positive (and cumulative score diff
positive).

With only a handful of seeds the signal is noisy — treat this as a
"rough direction" check, not a statistically significant result. The
real signal comes from the Kaggle ladder.
"""
from __future__ import annotations

import time
from typing import List, Tuple

from orbitwars.bots.mcts_bot import MCTSAgent
from orbitwars.mcts.actions import ActionConfig
from tournaments.harness import play_game


def _mk_bokr_on():
    # Default MCTSAgent — BOKR refinement is shipped on.
    return MCTSAgent(rng_seed=0)


def _mk_bokr_off():
    # Explicitly disable refinement (back-compat single-angle behavior).
    cfg = ActionConfig(
        max_per_planet=10,  # match the shipped MCTSAgent max
        angle_refinement_n_grid=1,
        angle_refinement_range=0.1,
    )
    return MCTSAgent(action_cfg=cfg, rng_seed=0)


def run_pair(seed: int, a_seat: int) -> Tuple[int, int, float]:
    """A (BOKR on) in ``a_seat``, B (BOKR off) in the other.
    Returns (a_score, b_score, wall_sec)."""
    a = _mk_bokr_on()
    b = _mk_bokr_off()
    agents = [a.as_kaggle_agent(), b.as_kaggle_agent()]
    if a_seat == 1:
        agents = [b.as_kaggle_agent(), a.as_kaggle_agent()]
    t0 = time.perf_counter()
    result = play_game(agents, seed=seed, players=2, step_timeout=2.0)
    wall = time.perf_counter() - t0
    return (
        int(result.final_scores[a_seat]),
        int(result.final_scores[1 - a_seat]),
        wall,
    )


def main() -> None:
    seeds = [42, 123, 7, 2026, 314159]
    rows: List[Tuple[int, int, int, int, str]] = []
    print(f"BOKR ablation: {len(seeds)} seeds x 2 seats = "
          f"{len(seeds) * 2} matches", flush=True)

    for s in seeds:
        for a_seat in (0, 1):
            a_score, b_score, wall = run_pair(s, a_seat)
            diff = a_score - b_score
            tag = "A_WIN" if diff > 0 else ("B_WIN" if diff < 0 else "TIE")
            print(
                f"  seed={s} a_seat={a_seat}: {tag} "
                f"a={a_score} b={b_score} diff={diff:+d} wall={wall:.0f}s",
                flush=True,
            )
            rows.append((s, a_seat, a_score, b_score, tag))

    wins = sum(1 for r in rows if r[4] == "A_WIN")
    losses = sum(1 for r in rows if r[4] == "B_WIN")
    ties = sum(1 for r in rows if r[4] == "TIE")
    cum_diff = sum(r[2] - r[3] for r in rows)
    print(
        f"\nBOKR-ON vs BOKR-OFF: {wins}W/{losses}L/{ties}T across "
        f"{len(rows)} matches, cum_score_diff={cum_diff:+d}",
        flush=True,
    )
    if wins > losses:
        print("  ⇒ angle refinement trending net-positive (tiny sample)")
    elif wins < losses:
        print("  ⇒ angle refinement trending net-negative (reconsider defaults)")
    else:
        print("  ⇒ ties; no signal from this sample")


if __name__ == "__main__":
    main()
