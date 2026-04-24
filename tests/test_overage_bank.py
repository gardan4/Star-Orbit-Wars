"""Tests for the overage-bank opening-turn deadline lift (plan \xa7W3).

The lift adds ``Agent.deadline_boost_ms(obs, step)`` which the Kaggle
wrapper reads from ``obs.remainingOverageTime`` and passes into
``Deadline(extra_budget_ms=...)``. ``MCTSAgent`` overrides this hook to
lift the search deadline on the opening turns when the bank is fat
enough; every other shipped agent inherits the zero default.

These tests pin:
  * ``Deadline.remaining_ms`` / ``should_stop`` observe the extra budget.
  * ``MCTSAgent.deadline_boost_ms`` returns 0 outside the opening
    window, below the reserve, and on missing fields.
  * The per-turn boost is bounded by ``_OVERAGE_MAX_BOOST_MS`` so a
    huge bank can't blow the full budget on turn 0.
  * The Kaggle wrapper of a malformed agent falls back to a 0 boost
    instead of forfeiting.
"""
from __future__ import annotations

import time

import pytest

from orbitwars.bots.base import Agent, Deadline, HARD_DEADLINE_MS, NoOpAgent
from orbitwars.bots.mcts_bot import MCTSAgent


# ---- Deadline boost wiring --------------------------------------------


def test_deadline_extra_budget_defaults_to_zero():
    dl = Deadline()
    assert dl.extra_budget_ms == 0.0
    # With zero boost, behavior is identical to the historical Deadline.
    assert dl.remaining_ms(850) == pytest.approx(850 - dl.elapsed_ms(), abs=5)


def test_deadline_remaining_includes_extra_budget():
    dl = Deadline(extra_budget_ms=2000.0)
    # Sleep a minimal slice to make the elapsed_ms non-trivial; the
    # remaining should still report ~2850 - epsilon.
    time.sleep(0.01)
    remaining = dl.remaining_ms(850)
    assert 2830 < remaining < 2855, remaining


def test_deadline_should_stop_respects_boost():
    dl = Deadline(extra_budget_ms=500.0)
    # At t=0, should_stop(1) is immediately False because the boost
    # lifts the effective deadline to 501 ms.
    assert dl.should_stop(1.0) is False
    # A deadline of -500 means "immediately stop"; +500 boost brings
    # the effective deadline to 0, i.e. elapsed (>=0) >= 0 → True.
    assert dl.should_stop(-500.0) is True


def test_deadline_negative_extra_budget_is_clamped():
    # Guard: an override that returns a negative value must NOT shrink
    # the caller's base deadline; we clamp to zero.
    dl = Deadline(extra_budget_ms=-100.0)
    assert dl.extra_budget_ms == 0.0


# ---- Agent hook default ------------------------------------------------


def test_noop_agent_reports_zero_boost_by_default():
    """Every baseline agent inherits the zero default; no behavior
    change under the new hook."""
    a = NoOpAgent()
    obs = {"step": 0, "remainingOverageTime": 60.0}
    assert a.deadline_boost_ms(obs, step=0) == 0.0


# ---- MCTSAgent hook ----------------------------------------------------


def _mk_obs(step: int, bank: float) -> dict:
    return {
        "step": step,
        "remainingOverageTime": bank,
        "planets": [],
        "fleets": [],
        "comets": [],
        "player": 0,
    }


def test_mcts_boost_zero_outside_opening_window():
    a = MCTSAgent(rng_seed=0)
    # Any turn at/after the window closes → no boost, even with big bank.
    assert a.deadline_boost_ms(_mk_obs(step=10, bank=60.0), step=10) == 0.0
    assert a.deadline_boost_ms(_mk_obs(step=499, bank=60.0), step=499) == 0.0


def test_mcts_boost_zero_when_bank_below_reserve():
    a = MCTSAgent(rng_seed=0)
    # Bank is 3 s — above reserve (2 s) but below min_bank (5 s). Don't
    # take the risk.
    assert a.deadline_boost_ms(_mk_obs(step=0, bank=3.0), step=0) == 0.0
    # Default local simulator bank is 2 s; also below min.
    assert a.deadline_boost_ms(_mk_obs(step=0, bank=2.0), step=0) == 0.0
    # Zero bank (Kaggle already exhausted it on earlier turns).
    assert a.deadline_boost_ms(_mk_obs(step=0, bank=0.0), step=0) == 0.0


def test_mcts_boost_amortizes_bank_across_opening_window():
    a = MCTSAgent(rng_seed=0)
    # Bank=12 s, reserve=2 s, window=10 turns → usable 10 s → 1000 ms/turn
    # capped at 2000 ms max (our ceiling).
    boost_turn_0 = a.deadline_boost_ms(_mk_obs(step=0, bank=12.0), step=0)
    assert boost_turn_0 == pytest.approx(1000.0, abs=1.0)
    # Remaining turns shrinks as step advances; per-turn share grows.
    boost_turn_5 = a.deadline_boost_ms(_mk_obs(step=5, bank=12.0), step=5)
    assert boost_turn_5 == pytest.approx(2000.0, abs=1.0)  # 10000/5 = 2000 (at cap)


def test_mcts_boost_capped_by_max_per_turn():
    a = MCTSAgent(rng_seed=0)
    # Enormous bank — without the cap, turn 0 with 10-turn window would
    # try for (1000-2)/10 = 99.8 s/turn. Cap it at _OVERAGE_MAX_BOOST_MS.
    boost = a.deadline_boost_ms(_mk_obs(step=0, bank=1000.0), step=0)
    assert boost == pytest.approx(MCTSAgent._OVERAGE_MAX_BOOST_MS, abs=1.0)


def test_mcts_boost_robust_to_missing_field():
    """An obs without ``remainingOverageTime`` must still be handled —
    the local simulator default is 2 s but a 3rd-party harness may
    omit the field entirely."""
    a = MCTSAgent(rng_seed=0)
    obs = {"step": 0, "planets": [], "fleets": [], "player": 0}  # no bank key
    # obs_get returns 0 on missing, which is below min_bank → no boost.
    assert a.deadline_boost_ms(obs, step=0) == 0.0


# ---- Kaggle wrapper end-to-end ----------------------------------------


def test_kaggle_wrapper_uses_override_and_survives_bad_override():
    """The wrapper must:
      * plumb a positive boost through Deadline when agent allows,
      * catch exceptions in the override (degrades to zero),
      * never crash the match.
    """
    captured: dict = {}

    class SpyAgent(Agent):
        name = "spy"

        def deadline_boost_ms(self, obs, step: int) -> float:
            # Confirm the wrapper passed the right step + obs through.
            captured["step"] = step
            captured["bank"] = obs.get("remainingOverageTime")
            return 1500.0  # boost this turn by 1.5 s

        def act(self, obs, deadline):
            captured["extra_budget_ms"] = deadline.extra_budget_ms
            deadline.stage([])
            return []

    spy = SpyAgent().as_kaggle_agent()
    obs = {"step": 3, "remainingOverageTime": 30.0, "planets": [], "fleets": []}
    assert spy(obs, None) == []
    assert captured == {"step": 3, "bank": 30.0, "extra_budget_ms": 1500.0}

    class BadAgent(Agent):
        name = "bad"

        def deadline_boost_ms(self, obs, step: int) -> float:
            raise RuntimeError("boom")

        def act(self, obs, deadline):
            # Survives even though the hook raised.
            assert deadline.extra_budget_ms == 0.0
            deadline.stage([])
            return []

    bad = BadAgent().as_kaggle_agent()
    assert bad(obs, None) == []


def test_kaggle_wrapper_respects_lifted_hard_deadline():
    """When boost is positive, the wrapper's HARD_DEADLINE check lifts
    by the same boost — an act() that legitimately used the overage
    bank must still be accepted, not replaced with the fallback."""

    class SlowAgent(Agent):
        name = "slow"

        def deadline_boost_ms(self, obs, step: int) -> float:
            return 1000.0  # lift deadline by 1 s

        def act(self, obs, deadline):
            # Simulate a turn that takes ~150 ms — under HARD_DEADLINE_MS
            # regardless — so this is really testing that boost is
            # harmless when unused.
            time.sleep(0.15)
            deadline.stage([[0, 0.0, 1]])
            return [[0, 0.0, 1]]

    a = SlowAgent().as_kaggle_agent()
    obs = {"step": 0, "remainingOverageTime": 30.0, "planets": [], "fleets": []}
    out = a(obs, None)
    # Got the real action back, not the no-op fallback.
    assert out == [[0, 0.0, 1]]
