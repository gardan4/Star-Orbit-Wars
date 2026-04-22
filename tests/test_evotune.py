"""Tests for EvoTune: sandbox safety, pool semantics, and main-loop flow.

Real LLM calls are substituted with MockLLMClient; real fitness calls are
mocked with a simple callable. Together these cover the plumbing end-to-end
without running any game.
"""
from __future__ import annotations

import math

import pytest

from orbitwars.tune.evotune import (
    BASELINE_SOURCE,
    Candidate,
    CandidatePool,
    EvoTuneConfig,
    EvoTuneResult,
    MockLLMClient,
    TargetFeatures,
    UnsafeCandidateError,
    compile_scorer,
    run,
)


# ---- TargetFeatures ----

def _mk_features(**overrides) -> TargetFeatures:
    defaults = dict(
        target_production=3.0, target_defender_now=10.0,
        projected_defender_at_arrival=15.0,
        is_enemy=1.0, is_neutral=0.0, is_ally=0.0, is_comet=0.0,
        distance=30.0, travel_turns=8.0, ships_to_send=20.0,
        source_ships=40.0, source_production=2.0, step=20.0,
    )
    defaults.update(overrides)
    return TargetFeatures(**defaults)


# ---- Sandbox ----

def test_compile_scorer_runs_baseline():
    """The hand-written baseline compiles and produces finite positive scores
    for a typical attack scenario."""
    fn = compile_scorer(BASELINE_SOURCE)
    features = _mk_features()
    weights = {
        "w_production": 5.0, "w_ships_cost": 0.02, "w_distance_cost": 0.05,
        "w_travel_cost": 0.3, "mult_enemy": 1.8, "mult_neutral": 1.0,
        "mult_comet": 1.5, "mult_reinforce_ally": 0.0,
    }
    s = fn(features, weights)
    assert math.isfinite(s)
    assert s > 0


def test_compile_scorer_rejects_import():
    """Imports are banned — LLM-generated `import os; os.system(...)` dies
    at parse time."""
    src = "import os\ndef score(f, w): return 1.0"
    with pytest.raises(UnsafeCandidateError):
        compile_scorer(src)


def test_compile_scorer_rejects_dunder_attr():
    """`obj.__class__.__bases__` escape hatch is blocked."""
    src = "def score(f, w): return f.__class__\n"
    with pytest.raises(UnsafeCandidateError):
        compile_scorer(src)


def test_compile_scorer_rejects_no_score_function():
    """Source must define a `score` callable."""
    src = "def not_score(f, w): return 1.0"
    with pytest.raises(UnsafeCandidateError):
        compile_scorer(src)


def test_compile_scorer_swallows_runtime_errors():
    """Runtime errors inside the scorer return -inf (candidate dies, loop
    keeps going)."""
    src = "def score(f, w): return 1.0 / 0"
    fn = compile_scorer(src)
    features = _mk_features()
    assert fn(features, {}) == float("-inf")


def test_compile_scorer_coerces_nan_to_minus_inf():
    """Candidates that return NaN are killed (can't compare NaN cleanly)."""
    src = "def score(f, w): return float('nan')"
    fn = compile_scorer(src)
    assert fn(_mk_features(), {}) == float("-inf")


def test_compile_scorer_blocks_open():
    """Removed from __builtins__ → NameError → caught → -inf."""
    src = "def score(f, w): return open('/tmp/pwn').read()"
    fn = compile_scorer(src)
    assert fn(_mk_features(), {}) == float("-inf")


# ---- CandidatePool ----

def test_candidate_pool_dedupes_by_source_hash():
    pool = CandidatePool()
    a = Candidate(source="def score(f, w): return 1.0")
    b = Candidate(source="def score(f, w): return 1.0")  # same source
    c = Candidate(source="def score(f, w): return 2.0")
    assert pool.add(a) is True
    assert pool.add(b) is False  # dup
    assert pool.add(c) is True
    assert len(pool) == 2


def test_candidate_pool_top_k_order():
    pool = CandidatePool()
    for i, s in enumerate([0.1, 0.9, 0.5, 0.7]):
        c = Candidate(source=f"def score(f, w): return {i}", score=s)
        pool.add(c)
    top = pool.top_k(2)
    assert [c.score for c in top] == [0.9, 0.7]


# ---- Mock LLM ----

def test_mock_llm_cycles_sources():
    llm = MockLLMClient(["src_a", "src_b"])
    out = llm.propose([], 5)
    assert out == ["src_a", "src_b", "src_a", "src_b", "src_a"]
    assert llm.calls == [((), 5)]


# ---- End-to-end loop ----

def test_run_loop_passes_top_k_to_llm_and_scores_candidates():
    """Generation loop: seed gen-0 with baseline, then each gen asks LLM for
    N candidates, scores them via the fitness_fn, and tracks top so far.
    """
    # Trivial low-scoring seed so the LLM-proposed candidates can actually
    # overtake it — otherwise BASELINE_SOURCE would stay on top of the pool
    # and the "best so far" curve would be flat.
    seed_src = "def score(f, w): return 0.0"
    # Four distinct sources (matches candidates_per_gen=2 × generations=2 so
    # the cyclic mock never returns a duplicate and the pool stays
    # deterministic).
    sources = [
        "def score(f, w): return 0.2",
        "def score(f, w): return 0.5",
        "def score(f, w): return 0.9",
        "def score(f, w): return 0.7",
    ]
    llm = MockLLMClient(sources)

    # Fitness fn: score = the constant the scorer returns (test-only trick).
    def fitness(scorer):
        v = scorer(_mk_features(), {})
        return (v, 4, 0.01)

    cfg = EvoTuneConfig(
        generations=2, candidates_per_gen=2, top_k_survivors=2,
        seed_sources=[seed_src], seed=0,
    )
    res: EvoTuneResult = run(cfg, llm, fitness, verbose=False)

    # 1 seed + 2 gen-1 + 2 gen-2 = 5 total
    assert len(res.pool) == 5
    # Best candidate is 0.9 (from sources[2])
    assert res.pool.top_k(1)[0].score == 0.9
    # LLM saw elders on both generations it was called
    assert len(llm.calls) == 2
    assert llm.calls[0][1] == 2  # candidates_per_gen
    # Best-per-generation strictly increases: 0.0 → 0.5 → 0.9
    per_gen = [c.score if c else float("-inf") for c in res.best_per_generation]
    assert per_gen == [0.0, 0.5, 0.9]


def test_run_loop_survives_unsafe_candidates():
    """A generation that includes an unsafe candidate scores -inf for it and
    keeps going with the remaining ones."""
    sources = [
        "import os\ndef score(f, w): return 1.0",  # unsafe
        "def score(f, w): return 0.42",
    ]
    llm = MockLLMClient(sources)

    def fitness(scorer):
        v = scorer(_mk_features(), {})
        return (v, 4, 0.01)

    cfg = EvoTuneConfig(
        generations=1, candidates_per_gen=2, top_k_survivors=2,
        seed_sources=["def score(f, w): return 0.0"], seed=0,
    )
    res = run(cfg, llm, fitness, verbose=False)

    # Best should not be -inf — the safe sibling won.
    best = res.pool.top_k(1)[0]
    assert math.isfinite(best.score)
    assert best.score == 0.42
    # Unsafe candidate is in the pool but with score -inf.
    unsafe = [c for c in res.pool if c.score == float("-inf")]
    assert len(unsafe) == 1
