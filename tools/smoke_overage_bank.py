"""Empirical smoke test for the overage-bank opening-turn deadline lift.

Unit tests (``tests/test_overage_bank.py``) pin the *wiring* — the
``deadline_boost_ms`` hook returns the right number, ``Deadline.remaining_ms``
observes the boost, the Kaggle wrapper plumbs it through. This smoke goes
one step further and validates the effect is actually visible in a live
game:

  * ``deadline.extra_budget_ms`` is non-zero on turns 0..9 when we inject
    a realistic ``remainingOverageTime`` (local default is 2 s; Kaggle
    ladder is ~60 s).
  * Search actually **uses** the extra budget — ``result.n_rollouts``
    on boosted turns is higher than the un-boosted baseline.
  * The agent does not blow the hard 900 ms-per-turn ceiling (wrapper
    will lift it to 900 + boost — we print both numbers per turn so we
    can eyeball it).
  * From turn 10 onward the boost clamps to 0, so the agent is back on
    the default 300 ms search cap.

Usage::

    .\\.venv\\Scripts\\python.exe -m tools.smoke_overage_bank [--bank 60] [--seed 42]

The smoke prints one row per turn and a summary comparing the mean
``n_rollouts`` in the opening window (steps 0..9) against the mid-game
window (steps 10..19). A healthy result has opening_mean > midgame_mean
by at least a few ×.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from orbitwars.bots.base import Action, Agent, Deadline, obs_get
from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.bots.mcts_bot import MCTSAgent
from orbitwars.mcts.gumbel_search import GumbelConfig, SearchResult
from tournaments.harness import play_game


class TracingMCTSAgent(MCTSAgent):
    """MCTSAgent subclass that records per-turn telemetry to self.trace.

    The trace is a list of dicts, one per call to ``act``. Fields:
      * ``step``                — MCTSAgent's internal turn counter
      * ``extra_budget_ms``     — Deadline.extra_budget_ms this turn
      * ``wall_ms``             — total act() wall-clock
      * ``n_rollouts``          — how many Gumbel rollouts completed
      * ``search_duration_ms``  — Gumbel search's own self-reported time
      * ``used_fallback``       — True if search aborted / budget too tight
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.trace: List[Dict[str, Any]] = []
        self._last_search_result: Optional[SearchResult] = None
        # First-time install — MCTSAgent may replace self._search on the
        # fresh-game branch of act(), so we also re-install every turn in
        # _ensure_wrapped below.
        self._ensure_wrapped()

    def _ensure_wrapped(self) -> None:
        """Install the search-result capture wrapper on self._search.

        MCTSAgent.act rebuilds self._search on fresh_game (turn 0), so
        any wrapper installed in __init__ would be discarded on the very
        first call. We check each turn whether the wrapper is already in
        place and re-install only if it isn't — idempotent.
        """
        search = self._search.search
        if getattr(search, "_is_tracing_wrapper", False):
            return
        original = search

        def _wrapped_search(*a, **kw) -> Optional[SearchResult]:
            res = original(*a, **kw)
            self._last_search_result = res
            return res

        _wrapped_search._is_tracing_wrapper = True  # type: ignore[attr-defined]
        self._search.search = _wrapped_search  # type: ignore[assignment]

    def act(self, obs: Any, deadline: Deadline) -> Action:
        self._last_search_result = None
        # Re-install AFTER super().act() because the fresh_game branch
        # replaces self._search from inside act(). We capture the result
        # next turn onwards; turn 0 search is missed by design (the
        # wrapper is installed during that very act call — after the
        # fresh_game code ran). A single pre-act re-install handles it.
        self._ensure_wrapped()
        t0 = time.perf_counter()
        action = super().act(obs, deadline)
        # Re-install again so any fresh_game branch run during this act
        # is also covered on the NEXT turn.
        self._ensure_wrapped()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        step = getattr(self._fallback, "_turn_counter", -1)
        # `step` in the fallback increments AFTER the fallback's act() is
        # done, so by now it is (current_turn + 1). Subtract 1 so logs
        # show the 0-indexed turn that just executed.
        logged_step = max(0, step - 1)
        if self._last_search_result is not None:
            n_rollouts = self._last_search_result.n_rollouts
            sdur = self._last_search_result.duration_ms
            used_fallback = False
        else:
            n_rollouts = 0
            sdur = 0.0
            used_fallback = True
        self.trace.append({
            "step": logged_step,
            "extra_budget_ms": deadline.extra_budget_ms,
            "wall_ms": wall_ms,
            "n_rollouts": n_rollouts,
            "search_duration_ms": sdur,
            "used_fallback": used_fallback,
        })
        return action


def _inject_overage_bank_wrapper(
    kaggle_agent: Callable, bank_seconds: float
) -> Callable:
    """Wrap a kaggle-style ``agent(obs, cfg) -> action`` to inject a
    fake ``remainingOverageTime`` into obs.

    The local kaggle_environments schema hard-codes ``remainingOverageTime``
    default to 2 s, so without this injection our ``deadline_boost_ms``
    hook sees bank=2 and (correctly) refuses to boost. On the real ladder
    the bank starts at 60 s, so we simulate that here to exercise the
    opening-window boost.
    """
    def wrapped(obs, cfg=None):
        # obs from the env is a SimpleNamespace-ish object; wrap it as a
        # mutable dict copy that still satisfies obs_get.
        if hasattr(obs, "keys"):
            patched = dict(obs)
        else:
            # Object-style obs: reflect keys. SimpleNamespace exposes
            # __dict__; use that to build the dict.
            patched = dict(getattr(obs, "__dict__", {}))
        patched["remainingOverageTime"] = float(bank_seconds)
        return kaggle_agent(patched, cfg)

    wrapped.__name__ = getattr(kaggle_agent, "__name__", "wrapped")
    return wrapped


def run_smoke(
    bank_seconds: float = 60.0,
    seed: int = 42,
    n_turns: int = 20,
    gumbel_cfg: Optional[GumbelConfig] = None,
    save_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a short game and return the summary dict."""
    mcts = TracingMCTSAgent(
        gumbel_cfg=gumbel_cfg or GumbelConfig(
            num_candidates=16,
            total_sims=32,
            rollout_depth=15,
            hard_deadline_ms=300.0,
            anchor_improvement_margin=0.05,
        ),
        rng_seed=0,
    )
    heur = HeuristicAgent()

    kaggle_mcts = _inject_overage_bank_wrapper(
        mcts.as_kaggle_agent(), bank_seconds=bank_seconds
    )
    kaggle_heur = heur.as_kaggle_agent()

    t_start = time.perf_counter()
    # `step_timeout` matches Kaggle's actTimeout (1 s). Beyond that the
    # engine counts overage but the local schema clamps remainingOverageTime
    # to 2 s — which is why we inject a fresh 60 s per turn.
    result = play_game(
        [kaggle_mcts, kaggle_heur],
        seed=seed,
        players=2,
        step_timeout=5.0,  # be permissive; we're measuring, not racing.
    )
    wall = time.perf_counter() - t_start

    trace = mcts.trace[:n_turns]
    opening = [t for t in trace if t["step"] < 10]
    midgame = [t for t in trace if t["step"] >= 10]
    # Filter to turns where search actually exercised its budget. Early
    # game turns often short-circuit at the "not enough ships to launch"
    # path in HeuristicAgent, so n_rollouts=0 doesn't mean the boost
    # was wasted — it means there was no candidate action to rollout.
    # The relevant comparison is boost-on vs boost-off for turns that
    # DID call sequential_halving.
    active_opening = [t for t in opening if t["n_rollouts"] > 0]
    active_midgame = [t for t in midgame if t["n_rollouts"] > 0]

    def _mean(xs, key):
        vs = [x[key] for x in xs if x is not None]
        return sum(vs) / len(vs) if vs else 0.0

    summary = {
        "config": {
            "bank_seconds": bank_seconds,
            "seed": seed,
            "n_turns_traced": len(trace),
            "gumbel_cfg": {
                "num_candidates": mcts.gumbel_cfg.num_candidates,
                "total_sims": mcts.gumbel_cfg.total_sims,
                "rollout_depth": mcts.gumbel_cfg.rollout_depth,
                "hard_deadline_ms": mcts.gumbel_cfg.hard_deadline_ms,
            },
        },
        "per_turn": trace,
        "opening_mean_n_rollouts": _mean(opening, "n_rollouts"),
        "opening_mean_wall_ms": _mean(opening, "wall_ms"),
        "opening_mean_extra_budget_ms": _mean(opening, "extra_budget_ms"),
        "midgame_mean_n_rollouts": _mean(midgame, "n_rollouts"),
        "midgame_mean_wall_ms": _mean(midgame, "wall_ms"),
        "midgame_mean_extra_budget_ms": _mean(midgame, "extra_budget_ms"),
        # Active-search aggregates (turns where sequential_halving ran):
        "n_active_opening": len(active_opening),
        "n_active_midgame": len(active_midgame),
        "active_opening_mean_n_rollouts": _mean(active_opening, "n_rollouts"),
        "active_opening_mean_search_ms": _mean(active_opening, "search_duration_ms"),
        "active_midgame_mean_n_rollouts": _mean(active_midgame, "n_rollouts"),
        "active_midgame_mean_search_ms": _mean(active_midgame, "search_duration_ms"),
        "rewards": result.rewards,
        "final_scores": result.final_scores,
        "total_steps": result.steps,
        "game_wall_seconds": wall,
    }

    if save_json is not None:
        save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w") as f:
            json.dump(summary, f, indent=2)

    return summary


def _print_summary(s: Dict[str, Any]) -> None:
    print("=== overage-bank lift smoke test ===")
    print(f"bank: {s['config']['bank_seconds']} s, seed: {s['config']['seed']}")
    cfg = s["config"]["gumbel_cfg"]
    print(f"gumbel: candidates={cfg['num_candidates']} total_sims={cfg['total_sims']} "
          f"depth={cfg['rollout_depth']} base_deadline={cfg['hard_deadline_ms']:.0f} ms")
    print()
    print(f"{'step':>4}  {'boost(ms)':>9}  {'wall(ms)':>8}  {'rollouts':>8}  "
          f"{'srch(ms)':>8}  {'fallback':>8}")
    for t in s["per_turn"]:
        print(f"{t['step']:>4}  {t['extra_budget_ms']:>9.1f}  "
              f"{t['wall_ms']:>8.1f}  {t['n_rollouts']:>8d}  "
              f"{t['search_duration_ms']:>8.1f}  "
              f"{str(t['used_fallback']):>8}")
    print()
    print(f"opening (steps 0-9)  : all-turn mean_rollouts={s['opening_mean_n_rollouts']:.1f}  "
          f"mean_wall_ms={s['opening_mean_wall_ms']:.1f}  "
          f"mean_boost_ms={s['opening_mean_extra_budget_ms']:.1f}")
    print(f"midgame (steps 10+)  : all-turn mean_rollouts={s['midgame_mean_n_rollouts']:.1f}  "
          f"mean_wall_ms={s['midgame_mean_wall_ms']:.1f}  "
          f"mean_boost_ms={s['midgame_mean_extra_budget_ms']:.1f}")
    print()
    print(f"active-search only:")
    print(f"  opening (n={s['n_active_opening']}): mean_rollouts={s['active_opening_mean_n_rollouts']:.1f}  "
          f"mean_search_ms={s['active_opening_mean_search_ms']:.1f}")
    print(f"  midgame (n={s['n_active_midgame']}): mean_rollouts={s['active_midgame_mean_n_rollouts']:.1f}  "
          f"mean_search_ms={s['active_midgame_mean_search_ms']:.1f}")
    print()

    # Gates — informational, not blocking.
    gate_lift = s["opening_mean_extra_budget_ms"] > 100.0
    # Compare only turns where search actually ran. Early-game no-launch
    # turns inflate both means with zeros.
    if s["n_active_opening"] > 0 and s["n_active_midgame"] > 0:
        opening_ratio = (
            s["active_opening_mean_search_ms"]
            / max(1.0, s["active_midgame_mean_search_ms"])
        )
        # Boost is 2 s; default deadline is ~300 ms. Even with total_sims
        # capping SH's actual rollout count, boosted turns should search
        # for at least 1.1x the unboosted turn time.
        gate_longer_search = opening_ratio >= 1.1
    else:
        opening_ratio = float("nan")
        gate_longer_search = False
    gate_midgame_zero = s["midgame_mean_extra_budget_ms"] == 0.0
    print("gates:")
    print(f"  [{'PASS' if gate_lift else 'FAIL'}] opening boost > 100 ms")
    print(f"  [{'PASS' if gate_longer_search else 'FAIL'}] opening search >= 1.1x midgame search "
          f"(ratio={opening_ratio:.2f})")
    print(f"  [{'PASS' if gate_midgame_zero else 'FAIL'}] midgame boost is exactly 0")
    print()
    print(f"game: rewards={s['rewards']} scores={s['final_scores']} "
          f"steps={s['total_steps']} wall={s['game_wall_seconds']:.1f}s")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=float, default=60.0,
                    help="Simulated remainingOverageTime in seconds.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--turns", type=int, default=20,
                    help="How many turns of the trace to print/analyze.")
    ap.add_argument("--save-json", type=Path, default=None,
                    help="If set, write the full summary (per-turn trace) to JSON.")
    args = ap.parse_args(argv)

    s = run_smoke(
        bank_seconds=args.bank,
        seed=args.seed,
        n_turns=args.turns,
        save_json=args.save_json,
    )
    _print_summary(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
