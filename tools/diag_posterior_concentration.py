"""Diagnostic: does the posterior concentrate above the 0.35 threshold?

The MCTSAgent wiring requires posterior.max() >= 0.35 before the
opp_policy_override fires. The concentration test in test_bayes.py
only proves `>1/7` (uniform baseline); we need to know empirically
what actual peak is achieved over a realistic match length.

Run this before any expensive exploitation smoke — if peak probability
never crosses 0.35, the full smoke will show no delta and the fix is
either (a) lower the threshold, (b) lengthen the eps/temperature
schedule, or (c) accept that the override is latent for typical matches.
"""
from __future__ import annotations

import random

import numpy as np
from kaggle_environments import make

from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.opponent.archetypes import (
    ARCHETYPE_NAMES,
    make_archetype,
)
from orbitwars.opponent.bayes import ArchetypePosterior


THRESHOLD = 0.35
N_TURNS = 80   # bot runs mid-to-late game; take more turns than the test
SEEDS = [7, 13, 42]


def run_one(opp_name: str, seed: int) -> dict:
    random.seed(seed)
    env = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
    env.reset(num_agents=2)

    me = HeuristicAgent()
    opp = make_archetype(opp_name)
    kag_me = me.as_kaggle_agent()
    kag_opp = opp.as_kaggle_agent()

    post = ArchetypePosterior(temperature=2.0, eps=0.1)

    max_probs: list = []
    first_concentration_turn: int | None = None
    concentrated_turns = 0

    for step in range(N_TURNS):
        if env.state[0]["status"] != "ACTIVE":
            break
        obs_p0 = env.state[0]["observation"]
        obs_p1 = env.state[1]["observation"]
        post.observe(obs_p0, opp_player=1)

        dist = post.distribution()
        top_p = float(dist.max())
        top_name = ARCHETYPE_NAMES[int(np.argmax(dist))]
        max_probs.append(top_p)

        if top_p >= THRESHOLD:
            concentrated_turns += 1
            if first_concentration_turn is None:
                first_concentration_turn = step

        a0 = kag_me(obs_p0, env.configuration)
        a1 = kag_opp(obs_p1, env.configuration)
        env.step([a0, a1])

    final_dist = post.distribution()
    final_top_idx = int(np.argmax(final_dist))
    return {
        "opp": opp_name,
        "seed": seed,
        "peak_prob": float(max(max_probs)) if max_probs else 0.0,
        "final_top": ARCHETYPE_NAMES[final_top_idx],
        "final_top_prob": float(final_dist[final_top_idx]),
        "correct_archetype_prob": float(
            final_dist[ARCHETYPE_NAMES.index(opp_name)]
            if opp_name in ARCHETYPE_NAMES else 0.0
        ),
        "first_concentration_turn": first_concentration_turn,
        "concentrated_turns": concentrated_turns,
        "turns_observed": post.turns_observed(),
    }


def main() -> None:
    print(f"Threshold: {THRESHOLD}")
    print(f"Turns per match: {N_TURNS}")
    print(f"Seeds: {SEEDS}")
    print()

    for opp_name in ("rusher", "turtler", "defender"):
        for seed in SEEDS:
            r = run_one(opp_name, seed)
            correct_tag = "OK " if r["final_top"] == opp_name else "WRONG"
            first = r["first_concentration_turn"]
            first_str = f"first@{first}" if first is not None else "never"
            print(
                f"  opp={opp_name:>10s} seed={seed:>3d}: "
                f"peak={r['peak_prob']:.2f} "
                f"final_top={r['final_top']:>13s} ({correct_tag}) "
                f"p={r['final_top_prob']:.2f} "
                f"correct_p={r['correct_archetype_prob']:.2f} "
                f"{first_str} conc_turns={r['concentrated_turns']}"
            )


if __name__ == "__main__":
    main()
