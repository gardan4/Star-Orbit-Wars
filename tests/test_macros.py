"""Tests for the macro-action library at the search root."""
from __future__ import annotations

from orbitwars.bots.heuristic import parse_obs, build_arrival_table
from orbitwars.mcts.actions import (
    ActionConfig, KIND_ATTACK_ENEMY, KIND_HOLD,
    KIND_REINFORCE_ALLY, generate_per_planet_moves,
)
from orbitwars.mcts.macros import (
    DEFAULT_MACROS,
    build_macro_anchors,
    macro_hold_all,
    macro_mass_attack_nearest_enemy,
    macro_reinforce_weakest_ally,
    macro_retreat_to_largest_ally,
)


def _mk_obs(*, my_planets=2, enemy_planets=1):
    """Build a synthetic obs with configurable counts. Player 0 is us."""
    planets = []
    for i in range(my_planets):
        planets.append([i, 0, 20.0 + i * 5, 50.0, 1.5, 50 + i * 10, 3])
    for j in range(enemy_planets):
        planets.append([my_planets + j, 1, 80.0 - j * 5, 50.0, 1.5, 30, 3])
    initial = [[p[0], p[1], p[2], p[3], p[4], 10, p[6]] for p in planets]
    return {
        "player": 0,
        "step": 50,
        "angular_velocity": 0.03,
        "planets": planets,
        "initial_planets": initial,
        "fleets": [],
        "next_fleet_id": 0,
        "comet_planet_ids": [],
        "comets": [],
    }


def _per_planet(obs):
    po = parse_obs(obs)
    table = build_arrival_table(po)
    return po, generate_per_planet_moves(po, table, cfg=ActionConfig())


def test_hold_all_returns_one_hold_per_owned_planet():
    obs = _mk_obs(my_planets=3, enemy_planets=1)
    po, pp = _per_planet(obs)
    j = macro_hold_all(po, pp)
    assert j is not None
    moves = list(j.moves)
    assert len(moves) == 3
    for m in moves:
        assert m.kind == KIND_HOLD
        assert m.ships == 0


def test_hold_all_returns_none_when_no_owned_planets():
    obs = _mk_obs(my_planets=0, enemy_planets=2)
    po, pp = _per_planet(obs)
    assert macro_hold_all(po, pp) is None


def test_mass_attack_nearest_enemy_targets_nearest():
    obs = _mk_obs(my_planets=2, enemy_planets=2)
    # planet 0 at (20, 50), planet 1 at (25, 50)
    # enemy 2 at (80, 50), enemy 3 at (75, 50) — nearest to both my planets
    po, pp = _per_planet(obs)
    j = macro_mass_attack_nearest_enemy(po, pp, fraction=1.0)
    assert j is not None
    targets = sorted(m.target_pid for m in j.moves if m.kind == KIND_ATTACK_ENEMY)
    # Both my planets target enemy 3 (nearest at x=75 vs enemy 2 at x=80).
    assert targets == [3, 3]


def test_mass_attack_returns_none_when_no_enemies():
    obs = _mk_obs(my_planets=2, enemy_planets=0)
    po, pp = _per_planet(obs)
    assert macro_mass_attack_nearest_enemy(po, pp) is None


def test_mass_attack_holds_planets_below_min_launch():
    """Source planets with fewer than `min_launch` ships hold instead of
    attacking."""
    obs = _mk_obs(my_planets=2, enemy_planets=1)
    obs["planets"][0][5] = 2  # planet 0 has only 2 ships
    po, pp = _per_planet(obs)
    j = macro_mass_attack_nearest_enemy(po, pp, min_launch=5)
    assert j is not None
    move0 = next(m for m in j.moves if m.from_pid == 0)
    move1 = next(m for m in j.moves if m.from_pid == 1)
    assert move0.kind == KIND_HOLD
    assert move1.kind == KIND_ATTACK_ENEMY


def test_reinforce_weakest_ally_targets_min_ships_planet():
    obs = _mk_obs(my_planets=3, enemy_planets=1)
    obs["planets"][0][5] = 80  # strongest
    obs["planets"][1][5] = 10  # weakest
    obs["planets"][2][5] = 50
    po, pp = _per_planet(obs)
    j = macro_reinforce_weakest_ally(po, pp)
    assert j is not None
    # Every non-weakest sends to weakest; weakest itself holds.
    for m in j.moves:
        if m.from_pid == 1:
            assert m.kind == KIND_HOLD
        else:
            assert m.kind == KIND_REINFORCE_ALLY
            assert m.target_pid == 1


def test_reinforce_returns_none_when_only_one_friendly():
    obs = _mk_obs(my_planets=1, enemy_planets=1)
    po, pp = _per_planet(obs)
    assert macro_reinforce_weakest_ally(po, pp) is None


def test_retreat_to_largest_ally_targets_max_ships_planet():
    obs = _mk_obs(my_planets=3, enemy_planets=1)
    obs["planets"][0][5] = 10
    obs["planets"][1][5] = 100  # largest
    obs["planets"][2][5] = 30
    po, pp = _per_planet(obs)
    j = macro_retreat_to_largest_ally(po, pp)
    assert j is not None
    for m in j.moves:
        if m.from_pid == 1:
            assert m.kind == KIND_HOLD
        else:
            assert m.kind == KIND_REINFORCE_ALLY
            assert m.target_pid == 1


def test_build_macro_anchors_returns_default_macros():
    obs = _mk_obs(my_planets=3, enemy_planets=2)
    po, pp = _per_planet(obs)
    anchors = build_macro_anchors(po, pp)
    # All four default macros produce valid joints in this fixture.
    # `hold_all` + `mass_attack_nearest_enemy` + `reinforce_weakest` +
    # `retreat_to_largest` = 4 distinct wires (each macro produces a
    # non-degenerate joint here).
    assert len(anchors) >= 3, f"expected 3+ macros, got {len(anchors)}"
    # Every joint must have one move per owned planet.
    for j in anchors:
        assert len(j.moves) == 3


def test_build_macro_anchors_dedupes_identical_wires():
    """If two macros produce the same wire, only the first is kept."""
    obs = _mk_obs(my_planets=1, enemy_planets=2)  # only 1 owned → reinforce/retreat are None
    po, pp = _per_planet(obs)
    anchors = build_macro_anchors(po, pp)
    wires = {tuple(tuple(m) for m in j.to_wire()) for j in anchors}
    assert len(wires) == len(anchors), "duplicated wires not deduped"


def test_build_macro_anchors_swallows_macro_exceptions():
    """A macro that raises must NOT poison the rest."""
    def boom(po, pp):
        raise RuntimeError("simulated macro failure")

    obs = _mk_obs(my_planets=2, enemy_planets=1)
    po, pp = _per_planet(obs)
    anchors = build_macro_anchors(po, pp, macros=(boom, macro_hold_all))
    # boom is dropped, hold_all survives.
    assert len(anchors) == 1
    assert all(m.kind == KIND_HOLD for m in anchors[0].moves)


def test_default_macros_tuple_has_four_entries():
    """Sanity check: the shipped macros list is what we documented."""
    assert len(DEFAULT_MACROS) == 4
