"""Fitness function for TuRBO/EvoTune: weights → win rate vs opponent pool.

We instantiate a fresh HeuristicAgent with the candidate weights, run N games
against each opponent in the pool, and return the average win rate. Uses
kaggle_environments for ground truth (slow but correct).

Design notes:
  * Each game gets a unique seed derived from (seed_base, opp_idx, game_idx).
  * Seats alternated to cancel first-move bias (even games: hero seat 0).
  * Draws (mutual elim at step 500) count 0.5 in win_rate, 0 in hard_win_rate.
  * HeuristicAgent instance is fresh per game — state-reset is already handled
    by the agent itself (W1 regression test covers this), but we also
    materialize fresh factories each call to eliminate any cross-game state.
  * Returns per-game records so callers can compute Wilson CIs or seat stats.
"""
from __future__ import annotations

import random as _r
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from kaggle_environments import make

from orbitwars.bots.heuristic import HeuristicAgent


# (name, factory_returning_kaggle_callable)
Opponent = Tuple[str, Callable[[], Any]]


@dataclass
class FitnessConfig:
    """Configuration for a fitness evaluation pass."""
    opponents: List[Opponent]
    games_per_opponent: int = 10
    step_timeout: float = 5.0
    seed_base: int = 0


@dataclass
class GameRecord:
    opp_name: str
    seed: int
    hero_seat: int
    reward: int           # +1 win, -1 loss, 0 draw
    steps: int
    wall_seconds: float


@dataclass
class FitnessResult:
    win_rate: float                    # with draws counted as 0.5
    hard_win_rate: float               # draws counted as 0
    games: List[GameRecord] = field(default_factory=list)

    @property
    def n_games(self) -> int:
        return len(self.games)

    def by_opponent(self) -> Dict[str, Tuple[float, int]]:
        """Return {opp_name: (win_rate_with_draws_half, n_games)}."""
        by: Dict[str, List[int]] = {}
        for g in self.games:
            by.setdefault(g.opp_name, []).append(g.reward)
        out: Dict[str, Tuple[float, int]] = {}
        for k, rewards in by.items():
            wins = sum(1 for r in rewards if r == 1)
            draws = sum(1 for r in rewards if r == 0)
            n = len(rewards)
            out[k] = ((wins + 0.5 * draws) / max(1, n), n)
        return out

    def to_json(self) -> Dict[str, Any]:
        return {
            "win_rate": self.win_rate,
            "hard_win_rate": self.hard_win_rate,
            "games": [asdict(g) for g in self.games],
        }


def _materialize(opp: Any) -> Any:
    """Turn a string or zero-arg factory into a Kaggle-compatible agent handle."""
    if isinstance(opp, str):
        return opp
    if callable(opp):
        return opp()
    return opp


def evaluate(weights: Dict[str, float], cfg: FitnessConfig) -> FitnessResult:
    """Run the full fitness pass; return aggregated stats + per-game records."""
    result = FitnessResult(win_rate=0.0, hard_win_rate=0.0)
    wins = 0.0
    hard_wins = 0.0

    for opp_idx, (opp_name, opp_factory) in enumerate(cfg.opponents):
        for g in range(cfg.games_per_opponent):
            seed = cfg.seed_base + 997 * opp_idx + g
            hero_seat = g % 2
            _r.seed(seed)

            # Fresh hero + opp per game. `evaluate` callers pass factories, so
            # each game gets an independent agent instance (no leakage).
            hero = HeuristicAgent(weights=weights).as_kaggle_agent()
            opp = _materialize(opp_factory)
            agents = [hero, opp] if hero_seat == 0 else [opp, hero]

            env = make(
                "orbit_wars",
                configuration={"actTimeout": cfg.step_timeout},
                debug=False,
            )
            env.reset(num_agents=2)
            t0 = time.perf_counter()
            env.run(agents)
            dt = time.perf_counter() - t0

            hero_reward = int(env.state[hero_seat].reward or 0)
            steps = int(env.state[0].observation.step)
            result.games.append(GameRecord(
                opp_name=opp_name, seed=seed, hero_seat=hero_seat,
                reward=hero_reward, steps=steps, wall_seconds=dt,
            ))
            if hero_reward == 1:
                wins += 1.0
                hard_wins += 1.0
            elif hero_reward == 0:
                wins += 0.5

    n = max(1, result.n_games)
    result.win_rate = wins / n
    result.hard_win_rate = hard_wins / n
    return result


# ---- Convenience opponent pools ----

def starter_pool() -> List[Opponent]:
    """The W1/W2 primary target: just the engine's starter_agent.

    Kaggle ranks bots vs. each other, but starter_agent is our nearest-
    neighbor baseline and the first gate (plan: ≥95% win rate).
    """
    return [("starter", lambda: "starter")]


def w1_pool() -> List[Opponent]:
    """Full W1 pool: a spread of difficulties."""
    from orbitwars.bots.base import NoOpAgent, RandomAgent
    return [
        ("starter", lambda: "starter"),
        ("random",  lambda: RandomAgent(seed=0).as_kaggle_agent()),
        ("noop",    lambda: NoOpAgent().as_kaggle_agent()),
    ]


def w2_pool() -> List[Opponent]:
    """W2 pool: starter + archetype portfolio.

    The first TuRBO run on `starter_pool` saw a ~16.7% uniform win rate
    across 25 trials — starter_agent is too strong a single opponent
    for small weight perturbations to separate from noise. This pool
    broadens evaluation with a spread of heuristic styles so good
    weights can beat *some* opponents even when they tie starter.

    Each archetype is a frozen HeuristicAgent with parameter overrides;
    no extra dependencies.
    """
    from orbitwars.bots.base import RandomAgent
    from orbitwars.opponent.archetypes import (
        ARCHETYPE_NAMES, make_archetype,
    )
    pool: List[Opponent] = [
        ("starter", lambda: "starter"),
        ("random",  lambda: RandomAgent(seed=0).as_kaggle_agent()),
    ]
    for name in ARCHETYPE_NAMES:
        pool.append((
            name,
            lambda n=name: make_archetype(n).as_kaggle_agent(),
        ))
    return pool
