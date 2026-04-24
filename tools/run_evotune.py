r"""CLI driver for the EvoTune LLM-evolved priority-function experiment.

**Primary-novelty experiment** (plan §Path A, EvoTune): does an LLM,
given the baseline scorer source + top-k elder scores + a strict
feature dict, mutate its way to a priority function that beats the
hand-written ``_score_target`` on a frozen opponent pool?

Usage
-----
Smoke test (mock LLM — deterministic cycling through scripted sources)::

    .\.venv\Scripts\python.exe -m tools.run_evotune \
        --llm mock --generations 2 --candidates-per-gen 3 \
        --games 4 --pool starter --workers 1

Real run (Anthropic Claude Sonnet 4.5; needs ANTHROPIC_API_KEY)::

    .\.venv\Scripts\python.exe -m tools.run_evotune \
        --llm anthropic --generations 30 --candidates-per-gen 8 \
        --games 20 --pool w2 --workers 7

Outputs (under ``runs/``):

  evotune_YYYYMMDD_HHMMSS.jsonl — one line per candidate scored,
                                  streaming so partial progress is
                                  visible mid-run.
  evotune_YYYYMMDD_HHMMSS_best.py — source of the best candidate
                                    (drop-in into ``install_evo_scorer``).
  evotune_YYYYMMDD_HHMMSS_summary.json — aggregate stats + best-per-gen.

Design
------
Fitness = win-rate (draws count 0.5) vs the chosen pool. The hero is
HeuristicAgent with default weights plus the evolved scorer; the
opponents are the frozen pool specs. We use the baseline weights
intentionally — the LLM is evolving *structure*, not the 20 tuned
coefficients. TuRBO handles the coefficients in a separate run.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, List, Tuple

from orbitwars.bots.heuristic import HEURISTIC_WEIGHTS
from orbitwars.tune.evotune import (
    BASELINE_SOURCE,
    EvoTuneConfig,
    LLMClient,
    MockLLMClient,
    run as evotune_run,
)
from orbitwars.tune.fitness import (
    FitnessConfig,
    evaluate,
    starter_pool,
    w1_pool,
    w2_pool,
)


POOL_FACTORIES = {
    "starter": starter_pool,
    "w1": w1_pool,
    "w2": w2_pool,
}


# ---- LLM client factory ------------------------------------------------

def _make_mock_client() -> LLMClient:
    """Mock client that cycles through a tiny curated library of hand-
    written variations. Used only for smoke-testing plumbing — does not
    actually 'evolve' anything."""
    sources = [
        # Variation 1: emphasize production-to-cost ratio, ignore distance.
        """
def score(f, w):
    mult = (w.get('mult_enemy', 1.8) if f.is_enemy
            else (w.get('mult_neutral', 1.0) if f.is_neutral
                  else w.get('mult_reinforce_ally', 0.0)))
    if f.is_comet:
        mult *= w.get('mult_comet', 1.5)
    denom = max(1.0, f.ships_to_send) + f.travel_turns + 1e-6
    return mult * f.target_production / denom
""",
        # Variation 2: square production (superlinear reward for big
        # planets), penalize long travels more.
        """
def score(f, w):
    mult = (w.get('mult_enemy', 1.8) if f.is_enemy
            else (w.get('mult_neutral', 1.0) if f.is_neutral
                  else w.get('mult_reinforce_ally', 0.0)))
    if f.is_comet:
        mult *= w.get('mult_comet', 1.5)
    denom = (w.get('w_ships_cost', 0.02) * max(1.0, f.ships_to_send) +
             w.get('w_travel_cost', 0.6) * f.travel_turns + 1e-6)
    return mult * (f.target_production ** 2) / denom
""",
        # Variation 3: the baseline itself (sanity check — should score
        # near parity with itself).
        BASELINE_SOURCE,
    ]
    return MockLLMClient(sources=sources)


def _make_anthropic_client(model: str) -> LLMClient:
    """Load the real Anthropic client lazily so `--llm mock` runs never
    require an API key."""
    from orbitwars.tune.llm_client import AnthropicClient
    return AnthropicClient(model=model)


def _build_llm(kind: str, model: str) -> LLMClient:
    if kind == "mock":
        return _make_mock_client()
    if kind == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: --llm anthropic requires ANTHROPIC_API_KEY env var.",
                  file=sys.stderr)
            sys.exit(2)
        return _make_anthropic_client(model)
    raise ValueError(f"unknown --llm: {kind!r}")


# ---- Fitness bridge ----------------------------------------------------

def _make_fitness_fn(
    cfg_template: FitnessConfig,
    weights: dict,
    writer: Callable[[dict], None],
):
    """Closure: given a compiled scorer, run a fitness pass and return
    (score, n_games, wall_seconds). Also emits a JSONL record per
    candidate so the live log is visible during the run.

    Note: EvoTune's ``run()`` gives us the *scorer* (compiled callable),
    but ``FitnessConfig.scorer_source`` wants the *source string* so
    pool workers can recompile from scratch. We therefore re-score via
    the source; the scorer callable we're given isn't used directly.
    This is a deliberate contract: the compiled callable is captured
    in the closure-via-source-string flow, and recompiling in each
    worker keeps the picklable surface clean.
    """
    # The scorer signature in evotune._score_batch passes `compile_scorer`
    # output. But our fitness contract pushes the source string through
    # workers. We solve this by reaching back into the Candidate cid
    # via a small proxy: the caller hands us the compiled scorer, but
    # also (out-of-band) the source — see `_score_one_candidate` below.
    # For clarity we define the fitness fn to be a *partial* over
    # source, and EvoTune calls `_score_one_candidate(source)` instead.
    raise RuntimeError(
        "internal: use _score_one_candidate, not _make_fitness_fn, as "
        "EvoTune's fitness_fn. See run() below."
    )


def _score_one_candidate(
    source: str,
    cfg_template: FitnessConfig,
    weights: dict,
    writer: Callable[[dict], None],
    cid: str = "",
    generation: int = -1,
) -> Tuple[float, int, float]:
    """Play the candidate's priority-function through the fitness pool.

    Returns ``(win_rate, n_games, wall_seconds)`` in the shape EvoTune
    expects. Writes a JSONL event after each candidate so a tail -f on
    the run file shows progress.
    """
    t0 = time.perf_counter()
    # Build a FitnessConfig with the scorer source attached so the pool
    # initializer installs it in every worker.
    cfg = FitnessConfig(
        opponents=cfg_template.opponents,
        games_per_opponent=cfg_template.games_per_opponent,
        step_timeout=cfg_template.step_timeout,
        seed_base=cfg_template.seed_base,
        workers=cfg_template.workers,
        scorer_source=source,
    )
    result = evaluate(weights, cfg)
    wall = time.perf_counter() - t0

    by_opp = result.by_opponent()
    writer({
        "event": "candidate",
        "cid": cid,
        "generation": generation,
        "win_rate": result.win_rate,
        "hard_win_rate": result.hard_win_rate,
        "n_games": result.n_games,
        "wall_seconds": wall,
        "by_opponent": {k: {"wr": wr, "n": n} for k, (wr, n) in by_opp.items()},
    })
    return (result.win_rate, result.n_games, wall)


# ---- Run driver --------------------------------------------------------

def _bind_fitness(
    cfg_template: FitnessConfig,
    weights: dict,
    writer: Callable[[dict], None],
):
    """Return a fitness_fn(compiled_scorer) closure that EvoTune calls.

    EvoTune hands us a compiled callable (from ``compile_scorer``); we
    *ignore* it and use the source string attached via the cid map
    below. This indirection exists because workers can't pickle the
    compiled callable cleanly, but can pickle its source.
    """
    # Shared dict: EvoTune hashes candidates by source; we use the same
    # hash to look up the source from its cid when scoring.
    cid_to_source: dict = {}
    # Patch EvoTune's flow: the caller uses `pool.add(c)` and then
    # `_score_batch` invokes `compile_scorer(c.source)` and calls
    # `fitness_fn(scorer)`. We need the source, not the scorer. The
    # simplest route: monkey-patch compile_scorer? No — too fragile.
    # Instead we provide `fitness_fn` that takes the scorer and writes
    # the scorer's closure back into a call counter. Since EvoTune
    # also calls our fn once per candidate in source-iteration order,
    # we can *rebuild* the source table by inspecting the pool.
    #
    # Cleaner plan: bypass the EvoTune-level fitness_fn entirely and
    # write our own scoring loop that walks the pool and scores by
    # source. That's what ``run()`` below does.
    raise RuntimeError("not used — see run() instead")


def run(args: argparse.Namespace) -> int:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / f"evotune_{stamp}.jsonl"
    best_src_path = out_dir / f"evotune_{stamp}_best.py"
    summary_path = out_dir / f"evotune_{stamp}_summary.json"

    pool_factory = POOL_FACTORIES[args.pool]
    opponents = pool_factory()

    fitness_template = FitnessConfig(
        opponents=opponents,
        games_per_opponent=args.games,
        step_timeout=args.step_timeout,
        seed_base=args.seed,
        workers=args.workers,
    )

    # Streaming JSONL writer — one dict per line, flushed immediately.
    log_f = open(jsonl_path, "a", encoding="utf-8")

    def writer(row: dict) -> None:
        row = {"ts": time.time(), **row}
        log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log_f.flush()

    writer({
        "event": "start",
        "stamp": stamp,
        "args": vars(args),
        "pool": args.pool,
        "n_opponents": len(opponents),
        "games_per_opponent": args.games,
        "workers": args.workers,
    })

    # We bypass EvoTune's abstract fitness protocol and drive the loop
    # manually: that way we keep the *source* (not just the compiled
    # scorer) and can push it through fitness pool workers. The EvoTune
    # module still owns the sandbox + pool + prompt logic; we only
    # re-own the per-candidate scoring step.
    from orbitwars.tune.evotune import (
        Candidate, CandidatePool, compile_scorer,
    )

    llm = _build_llm(args.llm, args.llm_model)

    weights = dict(HEURISTIC_WEIGHTS)
    pool_obj = CandidatePool()
    best_per_gen: List[dict] = []

    # Generation 0 — seed.
    for src in [BASELINE_SOURCE]:
        c = Candidate(source=src, generation=0)
        pool_obj.add(c)
    _score_generation(pool_obj, fitness_template, weights, writer, generation=0,
                      candidates=[c for c in pool_obj if c.generation == 0])
    top = pool_obj.top_k(1)
    best_per_gen.append(_summarize_best(top[0]) if top else None)

    # Generations 1..N
    for gen in range(1, args.generations + 1):
        elders = pool_obj.top_k(args.top_k)
        try:
            proposals = llm.propose(elders, args.candidates_per_gen)
        except Exception as e:  # noqa: BLE001
            writer({"event": "llm_error", "generation": gen, "error": str(e)})
            print(f"[gen {gen}] LLM propose failed: {e}", flush=True)
            break
        fresh: List[Candidate] = []
        for src in proposals:
            # Pre-screen source via compile_scorer so a syntactically
            # broken candidate gets logged but never reaches a worker.
            try:
                compile_scorer(src)
            except Exception as e:  # noqa: BLE001
                c = Candidate(source=src, generation=gen, score=float("-inf"))
                pool_obj.add(c)
                writer({
                    "event": "rejected",
                    "cid": c.cid,
                    "generation": gen,
                    "error": str(e),
                })
                continue
            c = Candidate(
                source=src, generation=gen,
                parent_ids=[e.cid for e in elders],
            )
            if pool_obj.add(c):
                fresh.append(c)
            else:
                writer({"event": "duplicate", "cid": c.cid, "generation": gen})
        _score_generation(pool_obj, fitness_template, weights, writer,
                          generation=gen, candidates=fresh)
        top = pool_obj.top_k(1)
        best_per_gen.append(_summarize_best(top[0]) if top else None)

    best_overall = pool_obj.top_k(1)[0] if len(pool_obj) else None
    if best_overall is not None:
        best_src_path.write_text(best_overall.source, encoding="utf-8")

    summary = {
        "stamp": stamp,
        "args": vars(args),
        "n_candidates_evaluated": len(pool_obj),
        "best_overall": _summarize_best(best_overall) if best_overall else None,
        "best_per_generation": best_per_gen,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    writer({"event": "end", "best": summary["best_overall"]})
    log_f.close()

    if best_overall is not None:
        print(
            f"\n=== Best of run: cid={best_overall.cid} "
            f"score={best_overall.score:.3f} "
            f"gen={best_overall.generation} ===\n"
            f"Source written to: {best_src_path}\n"
            f"Summary written to: {summary_path}\n"
            f"Full log: {jsonl_path}"
        )
    return 0


def _summarize_best(c) -> dict:
    return {
        "cid": c.cid,
        "score": c.score,
        "generation": c.generation,
        "parent_ids": list(c.parent_ids),
        "n_games": c.n_games,
        "wall_seconds": c.wall_seconds,
    }


def _score_generation(
    pool_obj,
    fitness_template: FitnessConfig,
    weights: dict,
    writer,
    generation: int,
    candidates,
) -> None:
    for c in candidates:
        t0 = time.perf_counter()
        try:
            wr, n_games, wall = _score_one_candidate(
                c.source, fitness_template, weights, writer,
                cid=c.cid, generation=generation,
            )
        except Exception as e:  # noqa: BLE001
            c.score = float("-inf")
            c.wall_seconds = time.perf_counter() - t0
            writer({
                "event": "fitness_error",
                "cid": c.cid,
                "generation": generation,
                "error": repr(e),
            })
            continue
        c.score = float(wr)
        c.n_games = int(n_games)
        c.wall_seconds = float(wall)


# ---- CLI ---------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the EvoTune LLM-evolved priority-function experiment."
    )
    p.add_argument("--llm", choices=["mock", "anthropic"], default="mock",
                   help="LLM backend. 'mock' cycles through scripted sources.")
    p.add_argument("--llm-model", default="claude-sonnet-4-5-20250929",
                   help="LLM model name (Anthropic only).")
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--candidates-per-gen", type=int, default=4)
    p.add_argument("--top-k", type=int, default=4,
                   help="Top-K elders shown to the LLM each generation.")
    p.add_argument("--pool", choices=list(POOL_FACTORIES), default="starter")
    p.add_argument("--games", type=int, default=10,
                   help="Games per opponent per candidate.")
    p.add_argument("--workers", type=int, default=1,
                   help="multiprocessing workers for fitness eval.")
    p.add_argument("--step-timeout", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-dir", default="runs")
    return p


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
