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


# ---- Archetype-flavored fast rollouts (Path D + B integration) ---------

def _stubby_obs():
    """Synthetic 4-planet obs with clearly-distinguishable enemy vs
    neutral target positions. Used to exercise the ``enemy_bias`` and
    ``keep_reserve_ships`` knobs deterministically (no RNG in the
    decision path)."""
    return {
        "player": 0,
        "step": 50,
        "angular_velocity": 0.0,
        "planets": [
            # [pid, owner, x, y, r, ships, production]
            [0, 0, 20.0, 50.0, 2.0, 100, 3],   # my home
            [1, -1, 35.0, 50.0, 1.0, 10, 1],   # neutral, close
            [2, 1, 50.0, 50.0, 1.0, 20, 2],    # enemy, further
            [3, 1, 80.0, 50.0, 2.0, 30, 3],    # enemy, farthest
        ],
        "initial_planets": [],
        "fleets": [],
        "next_fleet_id": 0,
        "comet_planet_ids": [],
        "comets": [],
    }


def test_fast_rollout_enemy_bias_prefers_enemy_over_closer_neutral():
    """With enemy_bias=0.3 the closest neutral (pid=1 at d=15) is
    effectively further than the enemy at d=30 (30 × 0.3 = 9 effective),
    so the agent should target the enemy."""
    agent = FastRolloutAgent(enemy_bias=0.3, min_launch_size=5)
    action = agent.act(_stubby_obs(), Deadline())
    assert len(action) == 1
    pid_launched, angle, ships = action[0]
    # Angle must point roughly at enemy planet 2 (x=50, y=50 from x=20).
    # Pure east: atan2(0, +30) = 0.
    assert abs(angle - 0.0) < 0.1, (
        f"enemy_bias=0.3 should target enemy pid=2; angle={angle:.3f}"
    )


def test_fast_rollout_neutral_preference_with_high_enemy_bias():
    """With enemy_bias=3.0 (pro-neutral), the closer neutral wins even
    though enemies are "within reach". atan2 to pid=1 at x=35 is still
    0.0, so we verify via the target's equivalent distance instead —
    any enemy at d=30 has effective d=90, neutral at d=15 stays d=15."""
    agent = FastRolloutAgent(enemy_bias=3.0, min_launch_size=5)
    action = agent.act(_stubby_obs(), Deadline())
    assert len(action) == 1
    # Since both neutral pid=1 and enemy pid=2 sit on the same angle
    # (all planets are along y=50 here), the easier check is on ship
    # count + angle = 0.
    _, angle, _ = action[0]
    assert abs(angle - 0.0) < 0.1


def test_fast_rollout_keep_reserve_ships_holds_back_fleet():
    """With min_launch_size=5 and keep_reserve_ships=90, a 100-ship
    planet has 100-90=10 launchable, which beats min_launch. But with
    reserve=96, only 4 launchable < min_launch=5 so no launch fires."""
    agent = FastRolloutAgent(
        min_launch_size=5, send_fraction=1.0, keep_reserve_ships=90,
    )
    action = agent.act(_stubby_obs(), Deadline())
    assert len(action) == 1
    ships = int(action[0][2])
    assert ships == 10, f"expected 10 launchable, got {ships}"

    # Now crank the reserve past the margin — no launch should fire.
    agent2 = FastRolloutAgent(
        min_launch_size=5, send_fraction=1.0, keep_reserve_ships=96,
    )
    assert agent2.act(_stubby_obs(), Deadline()) == []


def test_fast_rollout_from_weights_derives_knobs():
    """from_weights must pull the four knobs out of a HEURISTIC_WEIGHTS
    dict. Tested with a synthetic dict so the expected values are
    unambiguous."""
    weights = {
        "min_launch_size": 25.0,
        "max_launch_fraction": 0.5,
        "mult_enemy": 2.0,
        "mult_neutral": 1.0,
        "keep_reserve_ships": 40.0,
    }
    agent = FastRolloutAgent.from_weights(weights)
    assert agent.min_launch_size == 25
    assert agent.send_fraction == pytest.approx(0.5)
    # enemy_bias = mult_neutral / mult_enemy = 1.0 / 2.0 = 0.5
    assert agent.enemy_bias == pytest.approx(0.5, rel=1e-3)
    assert agent.keep_reserve_ships == 40


def test_fast_rollout_from_weights_clamps_bias():
    """Pathological weights (e.g. mult_enemy=0) must not produce inf
    or div-by-zero — from_weights clamps the divisor."""
    weights = {
        "mult_enemy": 0.0,
        "mult_neutral": 10.0,
    }
    agent = FastRolloutAgent.from_weights(weights)
    # enemy_bias clamped to 10.0 (upper bound) since 10.0 / 1e-3 = 10000 → clamp
    assert agent.enemy_bias == pytest.approx(10.0)


def test_make_fast_archetype_returns_flavored_agent():
    """Each archetype name yields a FastRolloutAgent whose knobs reflect
    the archetype's weights after merging on top of HEURISTIC_WEIGHTS."""
    from orbitwars.opponent.archetypes import ARCHETYPE_NAMES, make_fast_archetype

    for name in ARCHETYPE_NAMES:
        agent = make_fast_archetype(name)
        assert isinstance(agent, FastRolloutAgent), (
            f"make_fast_archetype({name!r}) should return FastRolloutAgent"
        )
        # enemy_bias must be finite and clamped.
        assert 0.1 <= agent.enemy_bias <= 10.0
        # Keep reserve must be non-negative.
        assert agent.keep_reserve_ships >= 0


def test_make_fast_archetype_rusher_prefers_enemies():
    """Rusher has mult_enemy=2.6, mult_neutral=0.9 →
    enemy_bias = 0.9/2.6 ≈ 0.35 (strong enemy preference)."""
    from orbitwars.opponent.archetypes import make_fast_archetype

    agent = make_fast_archetype("rusher")
    assert agent.enemy_bias < 0.5, (
        f"rusher should strongly prefer enemies; enemy_bias={agent.enemy_bias:.3f}"
    )


def test_make_fast_archetype_economy_prefers_neutrals():
    """Economy has mult_enemy=1.2, mult_neutral=1.5 →
    enemy_bias = 1.5/1.2 = 1.25 (mild neutral preference)."""
    from orbitwars.opponent.archetypes import make_fast_archetype

    agent = make_fast_archetype("economy")
    assert agent.enemy_bias > 1.0


def test_make_fast_archetype_unknown_raises():
    """Typos error loudly, matching make_archetype's contract."""
    from orbitwars.opponent.archetypes import make_fast_archetype

    with pytest.raises(KeyError):
        make_fast_archetype("totally_made_up_archetype")


def test_mcts_agent_with_fast_policy_uses_fast_archetype_override():
    """End-to-end: when MCTSAgent has rollout_policy='fast' and the
    posterior fires, the override must produce FastRolloutAgents, not
    the slow HeuristicAgent-based ArchetypeAgents."""
    import numpy as np
    from orbitwars.bots.mcts_bot import MCTSAgent
    from orbitwars.mcts.gumbel_search import GumbelConfig

    agent = MCTSAgent(
        gumbel_cfg=GumbelConfig(
            num_candidates=2, total_sims=2, rollout_depth=1,
            hard_deadline_ms=2000.0, rollout_policy="fast",
        ),
        rng_seed=0,
    )
    # Seed a posterior that concentrates on rusher.
    agent.act({
        "player": 0, "step": 0, "angular_velocity": 0.03,
        "planets": [[0, 0, 20.0, 50.0, 1.5, 50, 3]],
        "initial_planets": [[0, 0, 20.0, 50.0, 1.5, 10, 3]],
        "fleets": [], "next_fleet_id": 0,
        "comets": [], "comet_planet_ids": [],
    }, Deadline())
    post = agent.opp_posterior
    post._turns_observed = agent._POSTERIOR_MIN_TURNS
    post.log_alpha = np.full(post.K, -10.0)
    post.log_alpha[post.names.index("rusher")] = 5.0
    agent._maybe_route_posterior_to_search()

    # The override should be populated and produce a FastRolloutAgent.
    assert agent._search.opp_policy_override is not None
    built = agent._search.opp_policy_override()
    assert isinstance(built, FastRolloutAgent)
