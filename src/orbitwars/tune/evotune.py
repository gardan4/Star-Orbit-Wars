"""EvoTune: LLM-evolved priority functions for HeuristicAgent target scoring.

Inspired by Romera-Paredes et al. (2024) "FunSearch" and the EvoTune recipe
that interleaves LLM proposal with tournament-scored selection. An LLM
proposes candidate Python `score(features, weights) -> float` functions; we
compile each in a restricted namespace, plug it into HeuristicAgent's target
scorer, and rank by win-rate vs an opponent pool. Top-k survive to seed the
next generation.

Key design:
  * Candidates are Python source strings exposing a single function named
    `score`. Safety: compiled with a restricted globals dict (math, typing,
    no __builtins__) so a hallucinated `open('/etc/passwd')` can't run.
  * Fitness reuses `tune.fitness.evaluate` — we swap the *structure* of the
    scorer, not just its coefficients.
  * Top-k elitism across generations. The LLM sees the top sources + scores
    and proposes mutations / hybrids, keyword: "beat these".
  * The LLM client is a protocol — `MockLLMClient` for tests, a real
    Anthropic/OpenAI client for production. That keeps tests deterministic.

Ablation target (writeup): after ~500 LLM calls, does the best evolved
scorer beat the hand-written `_score_target`? Any answer is writeup-worthy.
"""
from __future__ import annotations

import ast
import hashlib
import math
import random as _r
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol


# ---- Feature shape the scorer sees ----

@dataclass(frozen=True)
class TargetFeatures:
    """Inputs to an evolved target scorer.

    Frozen so candidates can't accidentally mutate shared state. Flat floats
    keep the LLM-facing interface small and hard to hallucinate-around.
    """
    # Target planet
    target_production: float
    target_defender_now: float
    projected_defender_at_arrival: float
    is_enemy: float         # 0/1 (float so math ops are well-defined)
    is_neutral: float
    is_ally: float
    is_comet: float
    # Geometry / timing
    distance: float
    travel_turns: float
    ships_to_send: float
    # Source planet
    source_ships: float
    source_production: float
    # Game state
    step: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "target_production": self.target_production,
            "target_defender_now": self.target_defender_now,
            "projected_defender_at_arrival": self.projected_defender_at_arrival,
            "is_enemy": self.is_enemy,
            "is_neutral": self.is_neutral,
            "is_ally": self.is_ally,
            "is_comet": self.is_comet,
            "distance": self.distance,
            "travel_turns": self.travel_turns,
            "ships_to_send": self.ships_to_send,
            "source_ships": self.source_ships,
            "source_production": self.source_production,
            "step": self.step,
        }


# ---- Sandbox: compile + call LLM-generated scorers ----

_AST_FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
    ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await,
    ast.Try, ast.Raise, ast.ClassDef,
)


class UnsafeCandidateError(ValueError):
    """Raised when a candidate source contains disallowed syntax."""


def _ast_is_safe(source: str) -> None:
    """Raise UnsafeCandidateError if source contains forbidden AST nodes."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, _AST_FORBIDDEN_NODES):
            raise UnsafeCandidateError(
                f"disallowed node: {type(node).__name__}"
            )
        # Block attribute access to dunder attributes (subclass escape hatch).
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeCandidateError(
                f"disallowed dunder attribute: {node.attr}"
            )


_SAFE_GLOBALS: Dict[str, Any] = {
    "__builtins__": {
        # Minimal: only things a pricing function plausibly needs.
        "abs": abs, "min": min, "max": max, "round": round,
        "int": int, "float": float, "bool": bool, "len": len,
        "sum": sum, "any": any, "all": all, "range": range,
        "sorted": sorted, "reversed": reversed, "zip": zip,
        "isinstance": isinstance,
    },
    "math": math,
}


def compile_scorer(source: str) -> Callable[[TargetFeatures, Dict[str, float]], float]:
    """Compile `source` (must define `score(features, weights)`) in a
    restricted namespace and return the function. Raises
    UnsafeCandidateError for forbidden syntax, SyntaxError for bad code."""
    _ast_is_safe(source)
    ns: Dict[str, Any] = {}
    exec(compile(source, "<evotune_candidate>", "exec"), _SAFE_GLOBALS, ns)
    fn = ns.get("score")
    if not callable(fn):
        raise UnsafeCandidateError("source must define a callable `score`")

    def _wrapped(features: TargetFeatures, weights: Dict[str, float]) -> float:
        try:
            v = fn(features, weights)
        except Exception:
            return float("-inf")  # kill the candidate on runtime errors
        if not isinstance(v, (int, float)) or math.isnan(v):
            return float("-inf")
        return float(v)

    return _wrapped


# ---- Candidate pool ----

@dataclass
class Candidate:
    source: str
    score: float = float("-inf")
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    n_games: int = 0
    wall_seconds: float = 0.0

    @property
    def cid(self) -> str:
        """Stable hash of the source — dedup across generations."""
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()[:12]


class CandidatePool:
    """Stores candidates, dedupes by source hash, returns top-k by score."""

    def __init__(self) -> None:
        self._by_cid: Dict[str, Candidate] = {}

    def add(self, c: Candidate) -> bool:
        """Add `c`; return True if new, False if duplicate."""
        if c.cid in self._by_cid:
            return False
        self._by_cid[c.cid] = c
        return True

    def top_k(self, k: int) -> List[Candidate]:
        return sorted(
            self._by_cid.values(), key=lambda c: c.score, reverse=True
        )[:k]

    def __len__(self) -> int:
        return len(self._by_cid)

    def __iter__(self):
        return iter(self._by_cid.values())


# ---- LLM client protocol ----

class LLMClient(Protocol):
    """The minimal surface EvoTune needs from any LLM.

    `propose(top_k, n) -> List[str]` returns n candidate source strings.
    Real clients (Anthropic/OpenAI) wrap this around a chat completion;
    MockLLMClient hardcodes outputs for tests.
    """

    def propose(self, top_k: List[Candidate], n: int) -> List[str]: ...


class MockLLMClient:
    """Deterministic mock: cycles through a scripted list of source strings.

    Ignores the top_k — used only for plumbing tests. `calls` records every
    (top_k, n) the loop asked for, so tests can assert the main loop passed
    the expected context.
    """

    def __init__(self, sources: List[str]):
        self.sources = list(sources)
        self.cursor = 0
        self.calls: List[tuple] = []

    def propose(self, top_k: List[Candidate], n: int) -> List[str]:
        self.calls.append((tuple(c.cid for c in top_k), n))
        out = []
        for _ in range(n):
            out.append(self.sources[self.cursor % len(self.sources)])
            self.cursor += 1
        return out


# ---- Seed scorers: what the LLM is trying to beat ----

BASELINE_SOURCE = """
# Hand-written baseline: mimics HeuristicAgent._score_target (simplified).
def score(f, w):
    mult = 1.0
    if f.is_enemy: mult = w.get('mult_enemy', 1.8)
    elif f.is_neutral: mult = w.get('mult_neutral', 1.0)
    elif f.is_ally: mult = w.get('mult_reinforce_ally', 0.0)
    if f.is_comet: mult *= w.get('mult_comet', 1.5)
    denom = (w.get('w_ships_cost', 0.02) * max(1.0, f.ships_to_send) +
             w.get('w_travel_cost', 0.3) * f.travel_turns +
             w.get('w_distance_cost', 0.05) * f.distance +
             1e-6)
    return mult * w.get('w_production', 5.0) * f.target_production / denom
"""


# ---- Fitness signature ----

# A fitness fn takes a compiled scorer and returns (score, n_games, wall_s).
# Callers supply this — it plugs the scorer into HeuristicAgent somehow and
# runs an opponent-pool pass. Kept abstract here so tests can mock it.
FitnessFn = Callable[[Callable], tuple]


# ---- Main loop ----

@dataclass
class EvoTuneConfig:
    generations: int = 10
    candidates_per_gen: int = 8
    top_k_survivors: int = 4
    seed_sources: List[str] = field(default_factory=lambda: [BASELINE_SOURCE])
    seed: int = 0


@dataclass
class EvoTuneResult:
    pool: CandidatePool
    best_per_generation: List[Optional[Candidate]] = field(default_factory=list)


def run(
    cfg: EvoTuneConfig,
    llm: LLMClient,
    fitness_fn: FitnessFn,
    verbose: bool = True,
) -> EvoTuneResult:
    """Drive the generation loop.

    `fitness_fn(scorer)` is the caller-supplied bridge into the game harness;
    it returns `(score, n_games, wall_seconds)`.
    """
    rng = _r.Random(cfg.seed)  # noqa: F841 (reserved for future stochastic ops)
    pool = CandidatePool()
    result = EvoTuneResult(pool=pool)

    # Seed generation 0.
    gen0 = []
    for src in cfg.seed_sources:
        c = Candidate(source=src, generation=0)
        if pool.add(c):
            gen0.append(c)
    _score_batch(gen0, fitness_fn, verbose=verbose)
    result.best_per_generation.append(_best(pool))

    # Generations 1..N.
    for gen in range(1, cfg.generations + 1):
        elders = pool.top_k(cfg.top_k_survivors)
        sources = llm.propose(elders, cfg.candidates_per_gen)
        fresh: List[Candidate] = []
        for src in sources:
            c = Candidate(
                source=src,
                generation=gen,
                parent_ids=[e.cid for e in elders],
            )
            if pool.add(c):
                fresh.append(c)
        _score_batch(fresh, fitness_fn, verbose=verbose)
        result.best_per_generation.append(_best(pool))
        if verbose:
            best = result.best_per_generation[-1]
            bs = f"{best.score:.3f}" if best else "n/a"
            print(f"[gen {gen}] {len(fresh)} new candidates | best so far: {bs}",
                  flush=True)

    return result


def _score_batch(
    candidates: List[Candidate],
    fitness_fn: FitnessFn,
    verbose: bool,
) -> None:
    for c in candidates:
        t0 = time.perf_counter()
        try:
            scorer = compile_scorer(c.source)
        except (UnsafeCandidateError, SyntaxError) as e:
            c.score = float("-inf")
            c.wall_seconds = time.perf_counter() - t0
            if verbose:
                print(f"  [{c.cid}] rejected: {e}", flush=True)
            continue
        try:
            score, n_games, wall = fitness_fn(scorer)
        except Exception as e:  # noqa: BLE001
            c.score = float("-inf")
            c.wall_seconds = time.perf_counter() - t0
            if verbose:
                print(f"  [{c.cid}] fitness raised: {e}", flush=True)
            continue
        c.score = float(score)
        c.n_games = int(n_games)
        c.wall_seconds = float(wall)
        if verbose:
            print(f"  [{c.cid}] score={c.score:.3f} n={c.n_games} "
                  f"t={c.wall_seconds:.1f}s", flush=True)


def _best(pool: CandidatePool) -> Optional[Candidate]:
    top = pool.top_k(1)
    return top[0] if top else None
