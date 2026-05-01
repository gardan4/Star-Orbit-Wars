"""Full-game seat-1 anchor-lock check across multiple seeds.

The 30-turn diagnostic (diag_mcts_vs_heur_actions_seat1.py) reports
0/30 divergences after the off-by-one fix. But the multi-seed smoke
still shows catastrophic losses at seat 1 on seeds 123 and 7
(mcts=0 vs heur=9345/10556). This extends the check to the full game
duration to locate any divergence that kicks in after turn 30.

For each seed in {42, 123, 7}, play 1 full game (MCTS at seat 1,
heuristic at seat 0) while maintaining a shadow HeuristicAgent at
seat 1 and computing what IT would play each turn. Any divergence
from MCTS's wire action is logged with turn number, observation
snippet, and both action proposals.
"""
from __future__ import annotations

import random
from typing import List, Sequence, Tuple

from kaggle_environments import make

from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.bots.mcts_bot import MCTSAgent


def norm(a) -> List[Tuple[int, float, int]]:
    out = []
    for m in a or []:
        if len(m) == 3:
            out.append((int(m[0]), round(float(m[1]), 4), int(m[2])))
    return sorted(out)


def run_one_seed(seed: int, max_turns: int = 500) -> None:
    random.seed(seed)
    # Use the same budget as the multi-seed smoke (step_timeout=2.0)
    # so divergence patterns reproduce what play_game observes.
    env = make("orbit_wars", configuration={"actTimeout": 2.0}, debug=False)
    env.reset(num_agents=2)

    mcts = MCTSAgent(rng_seed=seed)
    heur_shadow = HeuristicAgent()
    k_mcts = mcts.as_kaggle_agent()
    k_heur_p0 = HeuristicAgent().as_kaggle_agent()
    k_heur_shadow = heur_shadow.as_kaggle_agent()

    divergences = 0
    total = 0
    first_diverge_turn = None
    last_status = None

    for step in range(max_turns):
        if env.state[0]["status"] != "ACTIVE":
            last_status = env.state[0]["status"]
            break

        obs0 = env.state[0]["observation"]
        obs1 = env.state[1]["observation"]

        a_mcts = k_mcts(obs1, env.configuration)
        a_heur = k_heur_shadow(obs1, env.configuration)
        total += 1

        n_m = norm(a_mcts)
        n_h = norm(a_heur)
        if n_m != n_h:
            divergences += 1
            if first_diverge_turn is None:
                first_diverge_turn = step
                # Print details only for the FIRST divergence this game.
                extra = set(n_m) - set(n_h)
                missing = set(n_h) - set(n_m)
                obs_step = getattr(obs1, "step", None)
                # Score snapshot so we can see if the game is already lost.
                my_planets = [
                    p for p in (obs1.get("planets", []) if hasattr(obs1, "get") else getattr(obs1, "planets", []))
                    if int(p[1]) == 1
                ]
                my_ships = sum(int(p[5]) for p in my_planets) if my_planets else 0
                print(f"  seed={seed} FIRST DIVERGE at step={step}: "
                      f"mcts-only={len(extra)} heur-only={len(missing)} "
                      f"our_planets={len(my_planets)} our_ships={my_ships} "
                      f"obs.step={obs_step}", flush=True)
                print(f"    mcts: {n_m[:3]}{'...' if len(n_m) > 3 else ''}", flush=True)
                print(f"    heur: {n_h[:3]}{'...' if len(n_h) > 3 else ''}", flush=True)

        a_opp = k_heur_p0(obs0, env.configuration)
        env.step([a_opp, a_mcts])

    # Final scoring.
    final_r = [env.state[i]["reward"] for i in range(2)]
    planets = env.state[0]["observation"].get("planets", [])
    # Sum our ships across planets/fleets.
    # The kaggle orbit_wars env stores reward as total ship count.
    # Use reward directly.
    print(f"  seed={seed}: {divergences}/{total} diverged "
          f"(first_at={first_diverge_turn}) status={last_status} "
          f"final_rewards={final_r}", flush=True)


def main() -> None:
    for s in [42, 123, 7]:
        print(f"=== seed={s} ===", flush=True)
        run_one_seed(s, max_turns=500)


if __name__ == "__main__":
    main()
