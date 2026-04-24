"""Tests for ``orbitwars.tune.evotune_bridge`` — the glue that makes
LLM-evolved priority functions usable inside ``HeuristicAgent``'s game
loop.

We exercise:

  * install / uninstall round-trip restores ``heuristic._score_target``
    to the exact callable it was (identity check).
  * An evolved scorer that returns a constant actually *changes*
    ``_score_target``'s return (so the monkey-patch really landed).
  * The "can't capture" penalty survives — an under-sized attack on a
    defended planet still receives the -10 penalty even with the
    evolved scorer (the baseline contract we explicitly preserve).
  * The pool initializer ``_worker_init_evo_scorer`` is picklable and
    actually compiles + installs in a fresh process. We verify via
    ``multiprocessing.Pool(initializer=...)`` so Windows spawn is
    exercised.
"""
from __future__ import annotations

import math
import multiprocessing as mp

import pytest

from orbitwars.bots import heuristic as H
from orbitwars.tune import evotune_bridge as EB
from orbitwars.tune.evotune import BASELINE_SOURCE


# A trivial scorer that returns a known constant — lets us prove
# the monkey-patch took effect.
_CONST_SOURCE = """
def score(f, w):
    return 42.0
"""


@pytest.fixture(autouse=True)
def _reset_installed():
    """Guarantee every test starts + ends with the baseline scorer.

    Without this, a test that raises mid-install-round-trip would leak
    a monkey-patch into downstream tests and they'd silently stop
    testing ``heuristic._score_target``."""
    EB.uninstall_evo_scorer()
    yield
    EB.uninstall_evo_scorer()


def test_install_uninstall_roundtrip_restores_identity():
    """install → uninstall leaves ``heuristic._score_target`` bound to
    the *same object* it was before. Any identity-drift would mean
    future code paths silently use a stale wrapper."""
    original = H._score_target
    assert not EB.is_installed()

    EB.install_evo_scorer(BASELINE_SOURCE)
    assert EB.is_installed()
    assert H._score_target is not original

    EB.uninstall_evo_scorer()
    assert not EB.is_installed()
    assert H._score_target is original


def test_install_is_idempotent_wrt_original_pointer():
    """Double-install saves the original exactly once — otherwise a
    second install would overwrite ``_ORIGINAL_SCORE_TARGET`` with the
    first install's wrapper, and uninstall would then restore the
    wrapper instead of the real baseline."""
    original = H._score_target
    EB.install_evo_scorer(BASELINE_SOURCE)
    EB.install_evo_scorer(_CONST_SOURCE)
    EB.uninstall_evo_scorer()
    assert H._score_target is original


def test_evolved_scorer_changes_return_value(monkeypatch):
    """Plug the constant-42 scorer in and call ``_score_target`` on a
    plausible-looking context. Expect the returned score to be ~42
    (minus the baseline capture penalty if it fires)."""
    EB.install_evo_scorer(_CONST_SOURCE)

    # Fabricate the minimum context the scorer's features need. We
    # can't easily construct a real ParsedObs + ArrivalTable without
    # running a game, so we use stubs that respond to the specific
    # attribute reads the wrapper performs.

    class _StubObs:
        step = 10
        player = 0
        comet_planet_ids = set()
        comet_path_by_pid: dict = {}
        comet_path_index_by_pid: dict = {}
        angular_velocity = 0.0
        initial_planet_by_id: dict = {}

    class _StubTable:
        def projected_defender_at(self, pid, owner, ships, production, arrival):
            return 5  # low — so ships_to_send=50 is more than enough

    mp_planet = [0, 0, 30.0, 50.0, 2.0, 100, 3]  # owner=0 (hero)
    tp_planet = [1, 1, 70.0, 50.0, 2.0, 5, 2]    # owner=1 (enemy)
    ip_planet = tp_planet                         # static

    score, angle, projected = H._score_target(
        mp_planet, tp_planet, ip_planet,
        _StubObs(), _StubTable(),
        dict(H.HEURISTIC_WEIGHTS),
        ships_to_send=50,
    )
    # The evolved score is 42; the baseline penalty doesn't fire
    # (ships_to_send=50 > projected+margin=6), so we see 42 exactly.
    assert score == pytest.approx(42.0)
    assert projected == 5
    assert math.isfinite(angle)


def test_capture_penalty_survives_evolved_scorer():
    """Under-sized attack on a defended enemy planet still gets -10
    after the evolved scorer, so a hallucinated score-one-ship plan
    can't outrank a well-sized attack.

    With the constant-42 scorer and projected=50, an attack sizing
    10 ships against an enemy should score 42 - 10 = 32, not 42."""
    EB.install_evo_scorer(_CONST_SOURCE)

    class _StubObs:
        step = 10
        player = 0
        comet_planet_ids = set()
        comet_path_by_pid: dict = {}
        comet_path_index_by_pid: dict = {}
        angular_velocity = 0.0
        initial_planet_by_id: dict = {}

    class _StubTable:
        def projected_defender_at(self, pid, owner, ships, production, arrival):
            return 50  # entrenched defender

    mp_planet = [0, 0, 30.0, 50.0, 2.0, 100, 3]
    tp_planet = [1, 1, 70.0, 50.0, 2.0, 50, 2]
    ip_planet = tp_planet

    score, _, projected = H._score_target(
        mp_planet, tp_planet, ip_planet,
        _StubObs(), _StubTable(),
        dict(H.HEURISTIC_WEIGHTS),
        ships_to_send=10,   # under-sized!
    )
    assert projected == 50
    assert score == pytest.approx(32.0), (
        "under-sized enemy attack should lose the -10 penalty baked in"
    )


def test_unsafe_source_leaves_baseline_in_place():
    """A syntactically-OK but sandbox-forbidden scorer (e.g. one that
    imports a module) must be rejected at install time without mutating
    ``_score_target``."""
    original = H._score_target
    unsafe = "import os\ndef score(f, w): return 1.0\n"
    with pytest.raises(Exception):
        EB.install_evo_scorer(unsafe)
    assert H._score_target is original


# ---- Multiprocessing pool init -----------------------------------------

def _probe_is_installed(_):
    """Top-level so Pool can pickle it. Returns True iff the worker
    has an evolved scorer active. We can't return a function object
    (not picklable) so we return the boolean."""
    return EB.is_installed()


@pytest.mark.slow
def test_worker_init_installs_in_spawn_workers():
    """Spawn a tiny pool with ``_worker_init_evo_scorer`` as the
    initializer and verify every worker reports the scorer is
    installed.

    This is the contract that makes ``FitnessConfig.scorer_source``
    actually work in parallel on Windows — where a monkey-patch in
    the parent is NOT visible to spawn children. Marked slow because
    spawning a pool costs ~2 s on Windows; meaningless on CI Linux
    but cheap enough to keep in the default suite."""
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=2,
        initializer=EB._worker_init_evo_scorer,
        initargs=(_CONST_SOURCE,),
    ) as pool:
        flags = pool.map(_probe_is_installed, range(4))
    assert all(flags), (
        f"expected every worker to report is_installed=True, got {flags}"
    )
