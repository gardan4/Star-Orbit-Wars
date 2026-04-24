"""Parity test: fitness.evaluate serial == parallel (same seed → same records).

The cheap invariant: for a fixed (weights, cfg.seed_base, cfg.opponents) the
serial path (workers=1) and the multiprocessing.Pool path (workers>1) must
produce the same GameRecord list. We verify this against the cheapest
opponent we have — NoOpAgent — so the test runs quickly on CPU (≤30 s).

Why this matters:
  * TuRBO observations are binomial win-rates; if parallel silently
    drifted from serial, we'd be tuning a subtly different objective.
  * Per-worker determinism is already a subtle thing on Windows
    (spawn), where each worker re-imports modules; we seed both
    python-random and numpy-random inside ``_run_one_game`` so each
    game reproduces identically regardless of worker context.
"""
from __future__ import annotations

import pytest

from orbitwars.bots.heuristic import HEURISTIC_WEIGHTS
from orbitwars.tune.fitness import (
    FitnessConfig,
    Opponent,
    OpponentSpec,
    evaluate,
)


def _noop_only_pool() -> list[Opponent]:
    """Smallest real pool — just noop. Each game terminates fast
    (no launches → tie at turn 500 → draw) so the parity test
    completes quickly."""
    return [("noop", OpponentSpec(name="noop", kind="noop"))]


@pytest.mark.slow
def test_parallel_matches_serial_for_same_seed():
    """Same seed_base + same weights → identical game records in both
    serial and parallel paths.

    We keep this tiny — 1 opponent × 2 games — because a full Kaggle
    game is ~5 s even for noop-vs-heuristic. Marked slow so default
    CI doesn't run it, but `pytest -m slow` picks it up.
    """
    weights = dict(HEURISTIC_WEIGHTS)
    base_cfg = dict(
        opponents=_noop_only_pool(),
        games_per_opponent=2,
        step_timeout=1.0,
        seed_base=42,
    )

    serial = evaluate(weights, FitnessConfig(**base_cfg, workers=1))
    parallel = evaluate(weights, FitnessConfig(**base_cfg, workers=2))

    assert serial.n_games == parallel.n_games == 2
    # Order preservation: starmap returns in input order.
    for s, p in zip(serial.games, parallel.games):
        assert s.opp_name == p.opp_name
        assert s.seed == p.seed
        assert s.hero_seat == p.hero_seat
        assert s.reward == p.reward, (
            f"reward mismatch at seed={s.seed}: serial={s.reward} parallel={p.reward}"
        )
        assert s.steps == p.steps, (
            f"step count mismatch at seed={s.seed}: "
            f"serial={s.steps} parallel={p.steps}"
        )
    # Aggregates match modulo float precision.
    assert abs(serial.win_rate - parallel.win_rate) < 1e-9
    assert abs(serial.hard_win_rate - parallel.hard_win_rate) < 1e-9


def test_evaluate_workers_1_does_not_use_pool():
    """Regression guard: workers=1 must skip multiprocessing entirely.

    Otherwise every test file that calls evaluate() with default config
    would eat the ~1-2 s Windows spawn cost. We verify by patching out
    mp.Pool to a poison object — if serial code path tries to use it,
    the test fails loudly.
    """
    import orbitwars.tune.fitness as fit

    weights = dict(HEURISTIC_WEIGHTS)
    cfg = FitnessConfig(
        opponents=[],               # zero games → evaluate is trivially serial
        games_per_opponent=0,
        step_timeout=1.0,
        seed_base=0,
        workers=1,
    )

    class _Poison:
        def __call__(self, *a, **kw):
            raise AssertionError("workers=1 must not instantiate a Pool")

    orig = fit.mp.Pool
    fit.mp.Pool = _Poison()  # type: ignore[assignment]
    try:
        res = evaluate(weights, cfg)
    finally:
        fit.mp.Pool = orig  # type: ignore[assignment]

    assert res.n_games == 0
    assert res.win_rate == 0.0
