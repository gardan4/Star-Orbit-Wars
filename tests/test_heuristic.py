"""Tests for HeuristicAgent's arrival-table, scoring, and defense logic.

These are the regression tests for the state-reset bug (stale
`last_launch_turn` blocking launches in game 2+) that was found in W1 and
any future similar issues where the agent silently no-ops.
"""
from __future__ import annotations

import random

from kaggle_environments import make

from orbitwars.bots.heuristic import (
    HEURISTIC_WEIGHTS,
    ArrivalTable,
    HeuristicAgent,
    parse_obs,
)


def test_agent_weights_complete():
    """Every weight key referenced in _score_target must exist in the default
    weights dict — catches the w_ships_cost KeyError bug that made the bot
    silently no-op."""
    # Enumerate all keys that _score_target and _plan_moves access.
    expected = {
        "w_production", "w_ships_cost", "w_distance_cost", "w_travel_cost",
        "mult_neutral", "mult_enemy", "mult_comet", "mult_reinforce_ally",
        "ships_safety_margin", "min_launch_size", "max_launch_fraction",
        "expand_cooldown_turns", "keep_reserve_ships",
        "agg_early_game", "early_game_cutoff_turn",
        "sun_avoidance_epsilon", "comet_max_time_mismatch", "expand_bias",
    }
    missing = expected - set(HEURISTIC_WEIGHTS.keys())
    assert not missing, f"Missing weight keys: {missing}"


def test_arrival_table_projected_defender_neutral_capture():
    """Projecting past a capture-by-ally event flips owner and production."""
    t = ArrivalTable()
    # Ally sends 20 ships, arrives turn 10. Neutral has 5 ships, prod 0 (no
    # production on neutrals by engine rules).
    t.add(pid=42, turn=10, owner=0, ships=20)
    # At turn 20, the planet should be owned by me (0), production=2, ships
    # = (20 - 5) = 15 + 10*2 (production turns from 10 to 20) = 35.
    proj = t.projected_defender_at(
        pid=42, defender_owner=-1, current_ships=5, production=2, arrival_turn=20,
    )
    assert proj == 35, f"expected 35, got {proj}"


def test_arrival_table_empty_returns_base():
    """No events → just current ships + production for the elapsed turns.

    NOTE: the heuristic's projector assumes neutrals do not produce (matching
    engine rules: production only applied when owner != -1). So a neutral
    projection is unchanged by production; an owned projection grows.
    """
    t = ArrivalTable()
    # Neutral: production ignored.
    proj = t.projected_defender_at(pid=1, defender_owner=-1, current_ships=10, production=3, arrival_turn=5)
    assert proj == 10
    # Owned by player 0: production applied.
    proj = t.projected_defender_at(pid=1, defender_owner=0, current_ships=10, production=3, arrival_turn=5)
    assert proj == 25


def test_agent_launches_in_game_two_with_reused_instance():
    """Regression: in W1 a stale `last_launch_turn` from game 1 was blocking
    all game-2 launches, making the bot play like noop. A round-robin with
    reused agent instances exposed this. This test runs two full games with
    the same instance and asserts the bot takes action in both.

    We check via turn count: if the bot never launches, games drag to step
    499 (nobody gets eliminated). If it launches, it captures planets fast
    and games end well before 499 (by elimination, typically <250 steps).
    Also: heuristic should DEFINITELY beat starter_agent on a fraction of
    games even with reuse.
    """
    agent = HeuristicAgent()  # ONE instance, reused across both games
    kag = agent.as_kaggle_agent()

    steps = []
    rewards = []
    for i, seed in enumerate([11, 22]):
        random.seed(seed)
        env = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
        env.reset(num_agents=2)
        env.run([kag, "random"])
        steps.append(env.state[0].observation.step)
        rewards.append(env.state[0].reward)

    # Heuristic should beat random in both games (eliminates random inside
    # 500 turns). If stale state blocked launches, we'd see rewards=[-1, -1]
    # and max steps=499.
    assert rewards[0] == 1, f"game 1 should win, got {rewards[0]}"
    assert rewards[1] == 1, f"game 2 should win, got {rewards[1]} — state-reset regression?"
    assert steps[0] < 499 or steps[1] < 499, "both games hit step cap — bot never launched?"


def test_parse_obs_separates_owners():
    """parse_obs correctly sorts planets by owner."""
    obs = {
        "player": 1,
        "step": 5,
        "angular_velocity": 0.03,
        "planets": [
            [0, 0, 10, 10, 1.5, 20, 3],   # p0 owns
            [1, 1, 50, 50, 2.0, 15, 2],   # p1 (me) owns
            [2, -1, 80, 80, 1.0, 5, 1],   # neutral
            [3, 1, 90, 30, 1.0, 10, 1],   # p1 (me) owns
        ],
        "initial_planets": [],
        "fleets": [],
        "next_fleet_id": 0,
        "comet_planet_ids": [],
    }
    po = parse_obs(obs)
    assert po.player == 1
    assert {p[0] for p in po.my_planets} == {1, 3}
    assert {p[0] for p in po.enemy_planets} == {0}
    assert {p[0] for p in po.neutral_planets} == {2}
    assert po.planet_by_id[1][5] == 15
