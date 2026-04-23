"""Zero-miss gate.

Intent: every fleet the HeuristicAgent launches should reach its intended
target planet. We use ``tools.diag_shot_misses`` (which hooks the engine's
``combat_lists`` for ground-truth collision attribution) to measure.

"miss" means the fleet died without colliding with any planet — sun crossing,
out-of-bounds, or walked off-path. That is a bug in our intercept/obstruction
math and fails the gate.

"wrong_planet" is relaxed: we accept collisions with pre-spawn comets
(planet ids 28-31 and equivalents that weren't in ``obs.comets`` at the
launching turn — comet paths are literally not revealed to agents until
``(step+1) in COMET_SPAWN_STEPS``). Any wrong_planet with a non-comet
target fails the gate.

Both seats are exercised: seat 0 and seat 1 observe subtly different obs
shapes (``obs.step`` is None for seat 1), and the engine processes
player 0's moves before player 1's, so the fleet-id → intent mapping in
the verifier must account for the opponent's launch count. Regressing on
either seat is a bug and fails the gate.
"""
from __future__ import annotations

import pytest

from tools.diag_shot_misses import run


# Seeds cover several distinct board layouts.
_SEEDS = [42, 123, 7, 99, 2024]
_SEATS = [0, 1]


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("seat", _SEATS)
def test_zero_pure_misses(seed: int, seat: int) -> None:
    """No fleet should die without hitting a planet."""
    shots = run(seed=seed, opp="heuristic", max_turns=499, verbose=False,
                my_seat=seat)
    pure_misses = [s for s in shots if s.final_status == "miss"]
    # Every pure miss is a bug — print forensics if any slip through.
    assert not pure_misses, (
        f"seed={seed} seat={seat}: {len(pure_misses)} pure misses "
        f"(sun / oob / wayward). First: {pure_misses[0]}"
    )


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("seat", _SEATS)
def test_no_non_comet_wrong_planet(seed: int, seat: int) -> None:
    """Wrong-planet collisions must involve a comet that wasn't visible at
    launch. Hitting a static/orbiting planet we didn't aim at is a bug.
    """
    # Comet planet ids live in the 28+ range (one group per 4-comet spawn).
    # We approximate "was a comet" by: actual_collided_pid >= 28 (the first
    # comet pid in a 20-planet layout). This is permissive; if a future
    # layout has >=28 non-comet planets, tighten to an explicit comet set.
    shots = run(seed=seed, opp="heuristic", max_turns=499, verbose=False,
                my_seat=seat)
    non_comet_wrong = [
        s for s in shots
        if s.final_status == "wrong_planet" and s.actual_collided_pid < 28
    ]
    assert not non_comet_wrong, (
        f"seed={seed} seat={seat}: {len(non_comet_wrong)} wrong-planet hits "
        f"on non-comet targets. First: {non_comet_wrong[0]}"
    )


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("seat", _SEATS)
def test_high_hit_rate(seed: int, seat: int) -> None:
    """Sanity floor: of fleets that resolved (hit/miss/wrong/vanished),
    >= 95% should be clean hits.
    """
    shots = run(seed=seed, opp="heuristic", max_turns=499, verbose=False,
                my_seat=seat)
    resolved = [s for s in shots if s.final_status != "alive_at_end"]
    if not resolved:
        pytest.skip(f"seed={seed} seat={seat}: no resolved shots")
    hits = sum(1 for s in resolved if s.final_status == "hit")
    rate = hits / len(resolved)
    assert rate >= 0.95, (
        f"seed={seed} seat={seat}: hit rate {rate:.3f} "
        f"({hits}/{len(resolved)} resolved)"
    )
