"""Prioritized Fictitious Self-Play (PFSP) opponent pool for PPO.

Plan §"Path C" calls for PFSP — opponents are sampled with weight
``(1 - win_rate(learner, opp))^p`` so the trainer plays opponents
that are "just hard enough" but not impossible. Learner spends most
of its time at the boundary, not blowing out weak baselines or
losing every game to overpowered ones.

This module is a pure-Python list manager with a dict of win-rate
estimates per opponent, plus a sampler. No torch dependency — the
opponents themselves are callables that take an obs and return a
wire action, matching the ``Agent`` contract.

Reference: Vinyals et al. AlphaStar, "PFSP" appendix.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# An opponent is any callable matching the kaggle_agent shape.
OpponentAgent = Callable[[Any], list]


@dataclass
class OpponentEntry:
    """One opponent in the pool.

    Args:
      name: Unique key for win-rate accounting.
      factory: Zero-arg callable that builds a fresh agent instance per
        match. Per-match construction is important because some agents
        (HeuristicAgent, MCTSAgent) hold per-match state.
      base_weight: Multiplicative prior on this opponent before PFSP
        adjustment. Use 1.0 by default; raise to oversample a key
        opponent (e.g. the current ladder floor v15).
    """
    name: str
    factory: Callable[[], OpponentAgent]
    base_weight: float = 1.0


@dataclass
class PFSPPool:
    """Pool of opponents with online win-rate tracking.

    Attributes:
      opponents: List of OpponentEntry (read-mostly after construction).
      pfsp_p: PFSP exponent. ``p=2`` is the AlphaStar default — softer
        than ``p=1`` (which oversamples the very-hardest opponent).
      ema_alpha: Exponential moving average factor for win-rate updates.
        ``0.05`` ≈ 20-game effective window.
      min_weight: Floor on the sample weight so an opponent we always
        beat doesn't go to zero (keeps the PFSP curriculum from
        forgetting solved opponents).
      seed: RNG seed for ``sample()``.
    """
    opponents: List[OpponentEntry]
    pfsp_p: float = 2.0
    ema_alpha: float = 0.05
    min_weight: float = 0.01
    seed: int = 0
    _win_rate: Dict[str, float] = field(default_factory=dict)
    _rng: random.Random = field(default_factory=lambda: random.Random(0))

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        # Initialize win-rate at 0.5 (uncertain) for any opponent
        # without a prior estimate.
        for opp in self.opponents:
            self._win_rate.setdefault(opp.name, 0.5)

    def names(self) -> List[str]:
        return [o.name for o in self.opponents]

    def win_rate(self, name: str) -> float:
        return self._win_rate.get(name, 0.5)

    def update_win_rate(self, name: str, won: bool) -> None:
        """EMA-update the learner's win-rate against ``name`` after one match.

        ``won`` is True iff the learner won. Ties count as 0.5 — caller
        should pass ``0.5`` via ``update_win_rate_value`` for ties.
        """
        self.update_win_rate_value(name, 1.0 if won else 0.0)

    def update_win_rate_value(self, name: str, outcome: float) -> None:
        """Same as ``update_win_rate`` but accepts arbitrary ``outcome``
        in [0, 1] so ties (0.5) and losses with bonuses can be passed.
        """
        prev = self._win_rate.get(name, 0.5)
        new = (1.0 - self.ema_alpha) * prev + self.ema_alpha * float(outcome)
        self._win_rate[name] = new

    def sample(self) -> OpponentEntry:
        """Sample an opponent under PFSP weights.

        Weight for opp ``o`` = ``base_weight × max(min_weight, (1 - wr(o))^p)``.
        Higher weight = sample more often.
        """
        weights = []
        for opp in self.opponents:
            wr = self._win_rate.get(opp.name, 0.5)
            pfsp = max(self.min_weight, (1.0 - wr) ** self.pfsp_p)
            weights.append(opp.base_weight * pfsp)
        total = sum(weights)
        if total <= 0:
            # Degenerate (all min_weight zeroed via base_weight=0); fall
            # back to uniform.
            return self._rng.choice(self.opponents)
        r = self._rng.uniform(0, total)
        acc = 0.0
        for opp, w in zip(self.opponents, weights):
            acc += w
            if r <= acc:
                return opp
        return self.opponents[-1]  # numerical safety

    def snapshot(self) -> Dict[str, float]:
        """Read-only copy of the current win-rate estimates. Useful for
        logging the curriculum state per update."""
        return dict(self._win_rate)


def build_default_pool(
    *,
    weights_json: Optional[str] = None,
    nn_checkpoint: Optional[str] = None,
    seed: int = 0,
) -> PFSPPool:
    """Build a sensible default opponent pool.

    Composition (Plan §"Path D"):
      * 7 named archetypes (rusher, turtler, economy, harasser,
        comet-camper, opportunist, defender)
      * Frozen heuristic baseline (TuRBO-v3 weights if provided)
      * MCTSAgent at the v15 config (BC v1 + m=0, no macros) — the
        current ladder floor

    Caller can extend with checkpoint-snapshot opponents during training.
    """
    from orbitwars.bots.heuristic import HeuristicAgent, HEURISTIC_WEIGHTS
    from orbitwars.bots.mcts_bot import MCTSAgent
    from orbitwars.opponent.archetypes import ARCHETYPE_WEIGHTS

    entries: List[OpponentEntry] = []

    # 7 named archetypes — frozen heuristic snapshots with shaped weights.
    for name, archetype_weight_overrides in ARCHETYPE_WEIGHTS.items():
        # ARCHETYPE_WEIGHTS values are *overrides* on top of HEURISTIC_WEIGHTS.
        merged = dict(HEURISTIC_WEIGHTS)
        merged.update(archetype_weight_overrides)
        entries.append(OpponentEntry(
            name=f"arch:{name}",
            factory=lambda w=merged: HeuristicAgent(weights=w).as_kaggle_agent(),
            base_weight=1.0,
        ))

    # Frozen heuristic at default weights (or TuRBO-v3 if provided).
    base_weights = dict(HEURISTIC_WEIGHTS)
    if weights_json is not None:
        import json
        from pathlib import Path
        try:
            raw = json.loads(Path(weights_json).read_text(encoding="utf-8"))
            for key in ("best_weights", "best_point", "best", "weights"):
                if isinstance(raw, dict) and key in raw and isinstance(raw[key], dict):
                    raw = raw[key]
                    break
            if isinstance(raw, dict) and "best_index" in raw and isinstance(raw.get("trials"), list):
                best_i = int(raw["best_index"])
                for tr in raw["trials"]:
                    if int(tr.get("trial", -1)) == best_i and isinstance(tr.get("weights"), dict):
                        raw = tr["weights"]
                        break
            if isinstance(raw, dict):
                base_weights.update({k: float(v) for k, v in raw.items()})
        except Exception:
            pass

    entries.append(OpponentEntry(
        name="heur:turbo_v3",
        factory=lambda w=dict(base_weights): HeuristicAgent(weights=w).as_kaggle_agent(),
        # The TuRBO-v3 heuristic is our floor — give it more weight so
        # the learner trains hard against it.
        base_weight=2.0,
    ))

    # v15-config MCTSAgent — current ladder floor at 903.8.
    # Skip if no NN checkpoint; the bare MCTSAgent at default config
    # would just degenerate to heuristic anyway.
    if nn_checkpoint is not None:
        from pathlib import Path
        if Path(nn_checkpoint).exists():
            from orbitwars.mcts.gumbel_search import GumbelConfig
            from orbitwars.nn.nn_prior import load_conv_policy, make_nn_prior_fn

            def _v15_factory():
                model, mcfg = load_conv_policy(Path(nn_checkpoint), device="cpu")
                fn = make_nn_prior_fn(model, mcfg)
                cfg = GumbelConfig()
                cfg.sim_move_variant = "exp3"
                cfg.exp3_eta = 0.3
                cfg.rollout_policy = "fast"
                cfg.anchor_improvement_margin = 0.0
                return MCTSAgent(
                    weights=base_weights, gumbel_cfg=cfg,
                    rng_seed=0, move_prior_fn=fn,
                ).as_kaggle_agent()

            entries.append(OpponentEntry(
                name="mcts:v15",
                factory=_v15_factory,
                base_weight=3.0,  # the bot we want to beat — overweight.
            ))

    return PFSPPool(opponents=entries, seed=seed)
