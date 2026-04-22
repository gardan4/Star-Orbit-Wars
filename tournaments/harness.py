"""Round-robin tournament harness for Orbit Wars bots.

Usage:
    from orbitwars.bots.base import NoOpAgent, RandomAgent
    from tournaments.harness import round_robin

    bots = {"noop": NoOpAgent(), "random": RandomAgent(seed=0)}
    report = round_robin(bots, games=20, players=2, seed=0)
    report.print()

Design notes:
  * Uses the official kaggle_environments engine. This is slow (~1s/game for
    short matches, seconds for full games) but matches what the ladder sees.
  * Runs are seeded for reproducibility. Each pairing plays N games with
    positions swapped halfway through to cancel first-move bias.
  * Elo update uses K=16 and uniform prior Elo=1500. Enough signal at N=200.
  * Turn times recorded as p50 / p95 per bot; useful to audit the 1s budget.

CLI:
    python -m tournaments.harness --bots noop,random,starter --games 20 --players 2
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

from kaggle_environments import make


# ---- Bot registry ----

def _builtin_registry() -> Dict[str, Callable]:
    """Return a dict name -> factory() -> kaggle-callable, for CLI convenience.

    Real bots register themselves here via the `register` decorator below.
    """
    from orbitwars.bots.base import NoOpAgent, RandomAgent

    reg: Dict[str, Callable] = {
        "noop": lambda: NoOpAgent().as_kaggle_agent(),
        "random": lambda: RandomAgent(seed=0).as_kaggle_agent(),
        # Engine-provided baselines:
        "starter": lambda: "starter",  # uses kaggle_environments built-in
        "rnd_builtin": lambda: "random",
    }
    return reg


# ---- Match execution ----

@dataclass
class GameResult:
    seeds: int
    players: int
    agent_names: List[str]
    rewards: List[int]
    final_scores: List[int]
    steps: int
    turn_times_ms: List[List[float]]  # per-player, per-turn wall time

    def winners(self) -> List[int]:
        return [i for i, r in enumerate(self.rewards) if r == 1]


def _score_from_state(env) -> List[int]:
    """Compute score = total ships on owned planets + in fleets, per player."""
    n = len(env.state)
    scores = [0] * n
    for p in env.state[0].observation.planets:
        if p[1] != -1:
            scores[p[1]] += p[5]
    for f in env.state[0].observation.fleets:
        scores[f[1]] += f[6]
    return scores


def play_game(agents: Sequence, seed: int, players: int = 2, step_timeout: float = 5.0) -> GameResult:
    """Run a single orbit_wars game with the provided agents.

    `agents` is a list of callables (the Kaggle agent contract) or strings
    naming built-in agents.
    """
    import random as _pyr
    _pyr.seed(seed)
    # The engine uses its own random module; seeding here makes map generation
    # deterministic for reproducibility.

    cfg = {"actTimeout": step_timeout}
    env = make("orbit_wars", configuration=cfg, debug=False)

    # Crude per-turn timing by wrapping each agent.
    turn_times: List[List[float]] = [[] for _ in range(players)]

    def _timed(i, fn):
        def wrapped(obs, cfg=None):
            t0 = time.perf_counter()
            try:
                result = fn(obs, cfg) if callable(fn) else fn
            except Exception:
                result = []
            turn_times[i].append((time.perf_counter() - t0) * 1000.0)
            return result
        return wrapped

    wrapped_agents = []
    names = []
    for i, a in enumerate(agents):
        if isinstance(a, str):
            # Let engine resolve by name, but then we can't time it. Keep it simple:
            # we only time callable agents. For string-name agents we still get
            # final rewards & scores.
            wrapped_agents.append(a)
            names.append(a)
        else:
            wrapped_agents.append(_timed(i, a))
            names.append(getattr(a, "__name__", f"agent_{i}"))

    env.run(wrapped_agents)

    rewards = [int(s.reward if s.reward is not None else 0) for s in env.state]
    scores = _score_from_state(env)
    steps = int(env.state[0].observation.step)
    return GameResult(
        seeds=seed, players=players, agent_names=names,
        rewards=rewards, final_scores=scores, steps=steps, turn_times_ms=turn_times,
    )


# ---- Elo tracking ----

def _elo_update(ra: float, rb: float, score_a: float, k: float = 16.0) -> Tuple[float, float]:
    """Standard Elo: score_a in [0, 0.5, 1]. Returns (ra', rb')."""
    ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
    return ra + k * (score_a - ea), rb + k * ((1 - score_a) - (1 - ea))


# ---- Round robin ----

@dataclass
class PairStats:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    games: int = 0

    def record(self, outcome: float) -> None:
        self.games += 1
        if outcome == 1.0: self.wins += 1
        elif outcome == 0.0: self.losses += 1
        else: self.draws += 1

    def win_rate(self) -> float:
        return (self.wins + 0.5 * self.draws) / max(1, self.games)


@dataclass
class Report:
    bot_names: List[str]
    elos: Dict[str, float]
    h2h: Dict[Tuple[str, str], PairStats]  # (a, b) -> a's stats
    turn_time_p50: Dict[str, float]
    turn_time_p95: Dict[str, float]
    games_played: int
    wall_seconds: float

    def print(self) -> None:
        print(f"--- Round-robin report ({self.games_played} games, {self.wall_seconds:.1f}s) ---")
        print("Elo:")
        for name in sorted(self.bot_names, key=lambda n: -self.elos[n]):
            p50 = self.turn_time_p50.get(name, float("nan"))
            p95 = self.turn_time_p95.get(name, float("nan"))
            print(f"  {name:<16} elo={self.elos[name]:7.1f}  p50={p50:6.1f}ms  p95={p95:6.1f}ms")
        print("Head-to-head win rates (row vs col):")
        names = sorted(self.bot_names)
        header = "  " + " " * 16 + " ".join(f"{n:>10}" for n in names)
        print(header)
        for a in names:
            row = f"  {a:<16}"
            for b in names:
                if a == b:
                    row += f"  {'   - ':>8}"
                else:
                    wr = self.h2h.get((a, b), PairStats()).win_rate()
                    row += f"  {wr*100:6.1f}% "
            print(row)


def round_robin(
    bots: Dict[str, Callable],
    games: int = 20,
    players: int = 2,
    seed: int = 0,
    verbose: bool = True,
) -> Report:
    """Play every pair `games` times. 2p only for now (4p todo).

    `bots` values are Agent instances OR zero-arg factories returning a Kaggle
    agent callable. We materialize callables per match to avoid stateful leaks.
    """
    from orbitwars.bots.base import Agent

    def _materialize(v):
        if isinstance(v, Agent):
            return v.as_kaggle_agent()
        if callable(v):
            maybe = v()
            if callable(maybe):
                return maybe
            return maybe  # string, for engine-built-in
        return v

    if players != 2:
        raise NotImplementedError("4p round-robin TODO (straightforward extension)")

    names = list(bots.keys())
    elos = {n: 1500.0 for n in names}
    h2h: Dict[Tuple[str, str], PairStats] = {}
    for a, b in itertools.permutations(names, 2):
        h2h[(a, b)] = PairStats()

    all_turn_times: Dict[str, List[float]] = {n: [] for n in names}
    total_games = 0
    t_start = time.perf_counter()

    for a, b in itertools.combinations(names, 2):
        for g in range(games):
            # Alternate seating to cancel first-move bias.
            if g % 2 == 0:
                seat = [a, b]
            else:
                seat = [b, a]
            agents = [_materialize(bots[n]) for n in seat]
            result = play_game(agents, seed=seed + 1000 * total_games, players=2)
            total_games += 1

            # Record outcome (from a's perspective)
            if a == seat[0]:
                a_reward = result.rewards[0]
            else:
                a_reward = result.rewards[1]
            if a_reward == 1:
                h2h[(a, b)].record(1.0); h2h[(b, a)].record(0.0)
                elos[a], elos[b] = _elo_update(elos[a], elos[b], 1.0)
            elif a_reward == -1:
                h2h[(a, b)].record(0.0); h2h[(b, a)].record(1.0)
                elos[a], elos[b] = _elo_update(elos[a], elos[b], 0.0)
            else:
                h2h[(a, b)].record(0.5); h2h[(b, a)].record(0.5)
                elos[a], elos[b] = _elo_update(elos[a], elos[b], 0.5)

            for i, n in enumerate(seat):
                all_turn_times[n].extend(result.turn_times_ms[i])

            if verbose and total_games % 10 == 0:
                print(f"  [{total_games}] {a} vs {b}: rewards={result.rewards} steps={result.steps}")

    def _pct(vals: List[float], p: float) -> float:
        if not vals: return float("nan")
        vals = sorted(vals)
        k = max(0, min(len(vals) - 1, int(math.ceil(p / 100 * len(vals))) - 1))
        return vals[k]

    return Report(
        bot_names=names,
        elos=elos,
        h2h=h2h,
        turn_time_p50={n: statistics.median(v) if v else float("nan") for n, v in all_turn_times.items()},
        turn_time_p95={n: _pct(v, 95) for n, v in all_turn_times.items()},
        games_played=total_games,
        wall_seconds=time.perf_counter() - t_start,
    )


# ---- CLI ----

def _cli_main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bots", default="noop,random,starter",
                   help="comma-separated names (from the built-in registry)")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--players", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", help="write JSON report to this path")
    args = p.parse_args()

    reg = _builtin_registry()
    selected = {n.strip(): reg[n.strip()] for n in args.bots.split(",")}
    report = round_robin(selected, games=args.games, players=args.players, seed=args.seed)
    report.print()

    if args.json:
        with open(args.json, "w") as f:
            json.dump({
                "elos": report.elos,
                "h2h": {f"{a}_vs_{b}": {
                    "wins": s.wins, "losses": s.losses, "draws": s.draws, "games": s.games,
                } for (a, b), s in report.h2h.items()},
                "turn_time_p50": report.turn_time_p50,
                "turn_time_p95": report.turn_time_p95,
                "games_played": report.games_played,
                "wall_seconds": report.wall_seconds,
            }, f, indent=2)


if __name__ == "__main__":
    _cli_main()
