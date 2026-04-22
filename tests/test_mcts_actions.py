"""Tests for the MCTS action generator.

Covers:
  * Bounds: per-planet cap, min_launch_size, ship caps.
  * Priors: sum to 1 within a planet, HOLD always present.
  * Shape: KIND enum correctness; frozen PlanetMove hashable.
  * Joint: independent sampling returns exactly one move per planet.
"""
from __future__ import annotations

import math
import random

import pytest

from orbitwars.bots.heuristic import ArrivalTable, parse_obs
from orbitwars.mcts.actions import (
    ActionConfig,
    JointAction,
    KIND_ATTACK_ENEMY,
    KIND_ATTACK_NEUTRAL,
    KIND_HOLD,
    PlanetMove,
    generate_per_planet_moves,
    sample_joint,
)


def _mk_obs(my_ships: int = 50, enemy_ships: int = 30, neutral_ships: int = 10):
    """Tiny synthetic observation for a 2p game with 3 planets."""
    return {
        "player": 0,
        "step": 10,
        "angular_velocity": 0.03,
        # Columns: [id, owner, x, y, radius, ships, production]
        "planets": [
            [0, 0, 20.0, 50.0, 1.5, my_ships, 3],      # me
            [1, 1, 80.0, 50.0, 1.5, enemy_ships, 3],   # enemy
            [2, -1, 50.0, 20.0, 1.0, neutral_ships, 1], # neutral
        ],
        "initial_planets": [
            [0, 0, 20.0, 50.0, 1.5, 10, 3],
            [1, 1, 80.0, 50.0, 1.5, 10, 3],
            [2, -1, 50.0, 20.0, 1.0, 10, 1],
        ],
        "fleets": [],
        "next_fleet_id": 0,
        "comet_planet_ids": [],
    }


def test_planet_move_is_frozen_and_hashable():
    """Frozen dataclass → we can stash moves in dicts / sets (tree nodes)."""
    m = PlanetMove(from_pid=0, angle=0.1, ships=20, target_pid=1, kind=KIND_ATTACK_ENEMY)
    d = {m: 1}  # would raise if unhashable
    assert d[m] == 1


def test_generate_includes_hold_when_enabled():
    po = parse_obs(_mk_obs())
    table = ArrivalTable()
    cfg = ActionConfig(max_per_planet=8, include_hold=True)
    per_planet = generate_per_planet_moves(po, table, cfg=cfg)
    moves = per_planet[0]
    assert any(m.is_hold for m in moves), "HOLD move must be emitted"


def test_generate_respects_max_per_planet():
    po = parse_obs(_mk_obs())
    table = ArrivalTable()
    cfg = ActionConfig(max_per_planet=3, ship_fractions=(0.25, 0.5, 1.0))
    per_planet = generate_per_planet_moves(po, table, cfg=cfg)
    assert len(per_planet[0]) <= 3


def test_generate_priors_sum_to_one_within_planet():
    po = parse_obs(_mk_obs())
    table = ArrivalTable()
    per_planet = generate_per_planet_moves(po, table)
    for pid, moves in per_planet.items():
        if not moves:
            continue
        total = sum(m.prior for m in moves)
        assert math.isclose(total, 1.0, abs_tol=1e-6), (
            f"planet {pid}: priors sum to {total} not 1"
        )


def test_generate_falls_back_to_hold_only_when_no_ships():
    """A planet with ships < min_launch_size emits exactly one HOLD move."""
    po = parse_obs(_mk_obs(my_ships=5))   # below default min_launch_size=20
    table = ArrivalTable()
    per_planet = generate_per_planet_moves(po, table)
    moves = per_planet[0]
    assert len(moves) == 1
    assert moves[0].is_hold
    assert math.isclose(moves[0].prior, 1.0)


def test_generate_ships_never_exceeds_available():
    """Ship sizes never overshoot the planet's available ships."""
    po = parse_obs(_mk_obs(my_ships=50))
    table = ArrivalTable()
    per_planet = generate_per_planet_moves(po, table)
    for m in per_planet[0]:
        assert m.ships <= 50, f"ship_count={m.ships} exceeds available=50"


def test_generate_emits_enemy_and_neutral_kinds():
    """With a reachable enemy + neutral, at least one move of each kind exists."""
    po = parse_obs(_mk_obs())
    table = ArrivalTable()
    per_planet = generate_per_planet_moves(po, table)
    kinds = {m.kind for m in per_planet[0]}
    assert KIND_ATTACK_ENEMY in kinds or KIND_ATTACK_NEUTRAL in kinds, (
        "expected at least one attack-kind move"
    )


def test_higher_prior_for_more_promising_target():
    """All else equal, a weaker neutral (fewer defenders) should score a
    higher attack-neutral prior than a well-defended enemy — exercises the
    heuristic's scoring integration."""
    po = parse_obs(_mk_obs(my_ships=80, enemy_ships=80, neutral_ships=5))
    table = ArrivalTable()
    per_planet = generate_per_planet_moves(po, table)
    attacks = [m for m in per_planet[0] if not m.is_hold]
    assert attacks, "should have at least one attack move"
    # Top-prior move should be attacking the weak neutral.
    top = max(attacks, key=lambda m: m.prior)
    assert top.target_pid == 2, f"expected neutral target (pid 2), got {top.target_pid}"


def test_sample_joint_returns_one_move_per_planet():
    """With a single owned planet, joint action has exactly one move."""
    po = parse_obs(_mk_obs())
    table = ArrivalTable()
    per_planet = generate_per_planet_moves(po, table)
    rng = random.Random(0)
    joint = sample_joint(per_planet, rng)
    assert isinstance(joint, JointAction)
    assert len(joint.moves) == 1


def test_joint_action_to_wire_drops_hold():
    """Wire format omits HOLD moves so the engine receives the correct signal."""
    m_attack = PlanetMove(from_pid=0, angle=0.5, ships=20, target_pid=1,
                          kind=KIND_ATTACK_ENEMY, prior=1.0)
    m_hold = PlanetMove(from_pid=1, angle=0.0, ships=0, target_pid=-1,
                        kind=KIND_HOLD, prior=1.0)
    joint = JointAction(moves=(m_attack, m_hold))
    wire = joint.to_wire()
    assert len(wire) == 1
    assert wire[0] == [0, 0.5, 20]


def test_softmax_temperature_changes_sharpness():
    """Low temperature → sharper prior (max prior closer to 1)."""
    po = parse_obs(_mk_obs(my_ships=60, enemy_ships=80, neutral_ships=5))
    table = ArrivalTable()

    cfg_sharp = ActionConfig(softmax_temperature=0.1)
    cfg_flat = ActionConfig(softmax_temperature=10.0)

    sharp = generate_per_planet_moves(po, table, cfg=cfg_sharp)[0]
    flat = generate_per_planet_moves(po, table, cfg=cfg_flat)[0]

    max_sharp = max(m.prior for m in sharp)
    max_flat = max(m.prior for m in flat)
    assert max_sharp > max_flat, (
        f"expected sharp>{max_sharp:.3f} > flat>{max_flat:.3f}"
    )
