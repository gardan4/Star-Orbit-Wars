"""Tests for FastRolloutAgent — the ultra-cheap rollout policy.

These tests verify:
  * Action format is valid (list of [pid, angle, ships] triples).
  * Only my planets launch.
  * Ships never exceed what the source planet has.
  * The agent is materially faster than HeuristicAgent (the whole point
    of having it).
  * MCTSAgent with rollout_policy="fast" actually completes rollouts
    that would time out under the default "heuristic" policy.

Timing-based tests use generous thresholds; CI noise is real but the
gap we care about is 30-50×, not 2×, so generous thresholds still
cleanly separate pass from fail.
"""
from __future__ import annotations

import math
import random
import time
from typing import Any, List

import pytest
from kaggle_environments import make

from orbitwars.bots.base import Deadline
from orbitwars.bots.fast_rollout import FastRolloutAgent
from orbitwars.bots.heuristic import HeuristicAgent


def _fresh_env_obs(seed: int = 42, warmup_turns: int = 50) -> Any:
    """Seed + play a few heuristic-vs-heuristic turns to get a
    realistic mid-game observation (fleets on the board, ship counts
    varying). Turn-0 states are degenerate — they don't exercise the
    target-picking branch. We want a mid-game obs.
    """
    random.seed(seed)
    env = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
    env.reset(num_agents=2)
    env.step([[], []])
    warmup = HeuristicAgent().as_kaggle_agent()
    for _ in range(warmup_turns):
        a0 = warmup(env.state[0].observation)
        a1 = warmup(env.state[1].observation)
        env.step([a0, a1])
        if env.done:
            break
    return env.state[0].observation


def test_fast_rollout_returns_action_list():
    agent = FastRolloutAgent()
    obs = _fresh_env_obs()
    action = agent.act(obs, Deadline())
    assert isinstance(action, list), f"action must be a list, got {type(action)}"
    for m in action:
        assert isinstance(m, list) and len(m) == 3, (
            f"each move must be [pid, angle, ships], got {m!r}"
        )
        pid, angle, ships = m
        assert isinstance(pid, int)
        assert isinstance(angle, float) and math.isfinite(angle)
        assert isinstance(ships, int) and ships > 0


def test_fast_rollout_only_launches_from_my_planets():
    agent = FastRolloutAgent()
    obs = _fresh_env_obs()
    my_player = obs["player"] if isinstance(obs, dict) else obs.player
    my_pids = {
        int(p[0]) for p in (obs["planets"] if isinstance(obs, dict) else obs.planets)
        if int(p[1]) == my_player
    }
    action = agent.act(obs, Deadline())
    for m in action:
        assert int(m[0]) in my_pids, f"launched from non-owned planet {m[0]}"


def test_fast_rollout_respects_ship_cap():
    agent = FastRolloutAgent()
    obs = _fresh_env_obs()
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    ships_by_pid = {int(p[0]): int(p[5]) for p in planets}
    action = agent.act(obs, Deadline())
    for m in action:
        pid, _, ships = int(m[0]), float(m[1]), int(m[2])
        assert ships <= ships_by_pid[pid], (
            f"launched {ships} from planet {pid} which has only "
            f"{ships_by_pid[pid]} ships"
        )


def test_fast_rollout_returns_no_op_with_no_enemies():
    """If every planet belongs to us (degenerate state), there's
    nothing to push toward — agent must return no-op, not crash."""
    agent = FastRolloutAgent()
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 2.0, 100, 3],
            [1, 0, 90.0, 90.0, 2.0, 100, 3],
        ],
        "fleets": [],
        "step": 50,
        "angular_velocity": 0.03,
        "initial_planets": [],
        "next_fleet_id": 0,
        "comets": [],
        "comet_planet_ids": [],
    }
    action = agent.act(obs, Deadline())
    assert action == []


def test_fast_rollout_is_materially_faster_than_heuristic():
    """The whole point of this agent. Measure on a realistic obs
    across N calls; require at least 10× speedup (design target is
    30-50×, so 10× is a very forgiving threshold)."""
    obs = _fresh_env_obs()
    fast = FastRolloutAgent()
    slow = HeuristicAgent()

    # Warmup both (first call pays import/JIT costs).
    fast.act(obs, Deadline())
    slow.act(obs, Deadline())

    N = 30
    t0 = time.perf_counter()
    for _ in range(N):
        fast.act(obs, Deadline())
    t_fast = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(N):
        slow.act(obs, Deadline())
    t_slow = time.perf_counter() - t0

    ratio = t_slow / t_fast if t_fast > 0 else float("inf")
    assert ratio >= 10.0, (
        f"FastRolloutAgent only {ratio:.1f}× faster than HeuristicAgent "
        f"(expected >=10×). slow={t_slow*1000:.1f}ms, "
        f"fast={t_fast*1000:.1f}ms over N={N}."
    )


def test_gumbel_search_with_fast_rollout_completes_more_sims():
    """End-to-end: GumbelRootSearch(rollout_policy='fast') at a short
    budget completes more rollouts than the 'heuristic' policy.

    This is the motivating claim for the whole file: at 300 ms budget,
    the default heuristic policy can't finish a single rollout (one
    rollout costs ~560 ms). Fast-mode rollouts are ~20-30 ms each, so
    budget fits ~10. We check strict inequality with a small safety
    factor — the gap is large enough that flakes should be rare.
    """
    from orbitwars.mcts.gumbel_search import GumbelConfig, GumbelRootSearch

    obs = _fresh_env_obs()
    my_player = obs["player"] if isinstance(obs, dict) else obs.player

    def run(policy: str) -> int:
        cfg = GumbelConfig(
            num_candidates=4,
            total_sims=32,
            rollout_depth=15,
            hard_deadline_ms=300.0,
            anchor_improvement_margin=0.5,
            rollout_policy=policy,
        )
        search = GumbelRootSearch(gumbel_cfg=cfg, rng_seed=42)
        result = search.search(obs, my_player=my_player)
        return result.n_rollouts if result is not None else 0

    n_fast = run("fast")
    n_slow = run("heuristic")

    # Fast must complete at least one full rollout.
    assert n_fast >= 1, (
        f"fast rollout policy completed only {n_fast} rollouts at "
        f"300ms budget — expected >=1"
    )
    # And fast must dominate heuristic. The gap is typically 10-50×
    # in practice; we require at least 3× to keep the test robust to
    # CI noise.
    assert n_fast > n_slow * 3 or (n_slow == 0 and n_fast >= 1), (
        f"fast ({n_fast}) should complete materially more rollouts than "
        f"heuristic ({n_slow}) at the same budget"
    )
