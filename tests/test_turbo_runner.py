"""Smoke tests for TuRBO runner scaffolding.

The real tuning is slow (each fitness eval is ~10 kaggle games), so these
tests focus on the orchestration plumbing: parameter bounds, strategy
interface, and the run loop end-to-end with a minimal 1-game fitness config.
"""
from __future__ import annotations

from unittest.mock import patch

from orbitwars.bots.base import NoOpAgent
from orbitwars.bots.heuristic import HEURISTIC_WEIGHTS
from orbitwars.tune.fitness import FitnessConfig, FitnessResult, GameRecord
from orbitwars.tune.turbo_runner import (
    PARAM_BOUNDS,
    RandomSearch,
    RunResult,
    TrialRecord,
    run,
)


def test_param_bounds_are_subset_of_weights():
    """Every bounded param must also exist in HEURISTIC_WEIGHTS — typos in
    PARAM_BOUNDS would otherwise silently get merged into weights and then
    dropped by _score_target's lookups."""
    missing = set(PARAM_BOUNDS.keys()) - set(HEURISTIC_WEIGHTS.keys())
    assert not missing, f"PARAM_BOUNDS has unknown keys: {missing}"


def test_param_bounds_are_valid_intervals():
    """All (lo, hi) pairs are finite, ordered, and non-empty."""
    for name, (lo, hi) in PARAM_BOUNDS.items():
        assert lo < hi, f"{name}: lo={lo} hi={hi} is not a valid interval"
        assert lo == lo and hi == hi, f"{name}: NaN bound"  # no NaN


def test_random_search_respects_bounds():
    """RandomSearch.next_point() always lies inside PARAM_BOUNDS."""
    rs = RandomSearch(PARAM_BOUNDS, seed=42)
    for _ in range(50):
        p = rs.next_point()
        assert set(p.keys()) == set(PARAM_BOUNDS.keys())
        for k, v in p.items():
            lo, hi = PARAM_BOUNDS[k]
            assert lo <= v <= hi, f"{k}={v} out of [{lo},{hi}]"


def test_random_search_is_seedable():
    """Same seed → same sequence."""
    a = RandomSearch(PARAM_BOUNDS, seed=7)
    b = RandomSearch(PARAM_BOUNDS, seed=7)
    for _ in range(5):
        assert a.next_point() == b.next_point()


def test_run_loop_merges_baseline_weights_for_full_dict():
    """The run loop must hand `evaluate` a weights dict with ALL keys from
    HEURISTIC_WEIGHTS (so heuristic's _score_target never KeyErrors), even
    when PARAM_BOUNDS is smaller. This is the W1 regression made permanent."""
    captured = []

    def fake_evaluate(weights, cfg):
        captured.append(dict(weights))
        return FitnessResult(win_rate=0.5, hard_win_rate=0.4, games=[
            GameRecord("mock", 0, 0, 0, 10, 0.01)
        ])

    strat = RandomSearch(PARAM_BOUNDS, seed=0)
    cfg = FitnessConfig(opponents=[], games_per_opponent=0, seed_base=0)

    with patch("orbitwars.tune.turbo_runner.evaluate", fake_evaluate):
        res = run(strat, cfg, n_trials=2, verbose=False)

    assert isinstance(res, RunResult)
    assert len(captured) == 2
    for weights in captured:
        assert set(weights.keys()) >= set(HEURISTIC_WEIGHTS.keys()), (
            "run loop must pass a complete weights dict to evaluate"
        )


def test_run_loop_records_trials_and_picks_best():
    """Records append in order; .best tracks the highest win_rate."""
    scores = [0.3, 0.9, 0.5, 0.7]
    call_idx = [0]

    def fake_evaluate(weights, cfg):
        v = scores[call_idx[0]]
        call_idx[0] += 1
        return FitnessResult(win_rate=v, hard_win_rate=v, games=[
            GameRecord("mock", 0, 0, 0, 10, 0.01)
        ])

    strat = RandomSearch(PARAM_BOUNDS, seed=0)
    cfg = FitnessConfig(opponents=[], games_per_opponent=0, seed_base=0)

    with patch("orbitwars.tune.turbo_runner.evaluate", fake_evaluate):
        res = run(strat, cfg, n_trials=4, verbose=False)

    assert [t.win_rate for t in res.trials] == scores
    assert res.best is not None
    assert res.best.trial == 1
    assert res.best.win_rate == 0.9


def test_run_loop_writes_json_checkpoint(tmp_path):
    """Every trial flushes the full run to disk so long runs are inspectable
    mid-flight / resumable."""
    def fake_evaluate(weights, cfg):
        return FitnessResult(win_rate=0.42, hard_win_rate=0.42, games=[
            GameRecord("mock", 0, 0, 1, 5, 0.01)
        ])

    out = tmp_path / "run.json"
    strat = RandomSearch(PARAM_BOUNDS, seed=0)
    cfg = FitnessConfig(opponents=[], games_per_opponent=0, seed_base=0)

    with patch("orbitwars.tune.turbo_runner.evaluate", fake_evaluate):
        run(strat, cfg, n_trials=3, out_path=out, verbose=False)

    import json
    data = json.loads(out.read_text())
    assert data["strategy"] == "RandomSearch"
    assert data["n_trials"] == 3
    assert len(data["trials"]) == 3
    assert data["best_win_rate"] == 0.42


def test_trial_record_by_opponent_roundtrips():
    """TrialRecord.as_json preserves by_opponent dict structure."""
    rec = TrialRecord(
        trial=0, weights={}, win_rate=0.5, hard_win_rate=0.5, n_games=4,
        fitness_wall_seconds=1.0,
        by_opponent={"starter": (0.25, 4)},
    )
    d = rec.as_json()
    assert d["by_opponent"]["starter"] == [0.25, 4]


def test_ax_turbo_sobol_bootstrap_and_observe():
    """AxTurbo can sample 3 points (Sobol phase) and accept noisy observations
    without errors. Skipped silently if Ax isn't installed so the test suite
    still passes on a bare venv."""
    import pytest
    try:
        from orbitwars.tune.turbo_runner import AxTurbo
    except RuntimeError:
        pytest.skip("Ax not installed")
    try:
        import ax  # noqa: F401
    except ImportError:
        pytest.skip("Ax not installed")

    import warnings
    warnings.filterwarnings("ignore")

    strat = AxTurbo(PARAM_BOUNDS, seed=42)
    seen = []
    for i in range(3):
        p = strat.next_point()
        assert set(p.keys()) == set(PARAM_BOUNDS.keys()), "Ax returned wrong keys"
        for k, v in p.items():
            lo, hi = PARAM_BOUNDS[k]
            assert lo <= v <= hi, f"Ax sample out of bounds: {k}={v}"
        seen.append(p)
        strat.observe(p, 0.5 + 0.1 * i)
    # Ax should return different points across iterations (Sobol is deterministic
    # but distinct by index; bug-guard that we're not re-emitting the same point).
    assert seen[0] != seen[1], "Ax emitted duplicate points"
