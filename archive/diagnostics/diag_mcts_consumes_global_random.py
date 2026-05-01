"""Check whether MCTSAgent.act() consumes entropy from the global random module.

If yes, running MCTS at any seat perturbs the engine's comet-sizing RNG
stream (orbit_wars.py:455-458 uses global random for comet ship counts,
fired at steps 49, 149, 249, 349, 449). That would cause games where
MCTS-at-seat-1 plays identically to a shadow HeuristicAgent to still
produce different outcomes than heur-vs-heur baseline — which is
exactly the symptom of the residual seed=123/7 blowouts.

Hypothesis from the full-game diagnostic: 0/N divergence between MCTS's
wire action and shadow_heur at seat 1, yet final rewards [1, -1] with
catastrophic P1 losses. The only remaining causal variable is the
engine's random stream, which MCTS search could be consuming.
"""
from __future__ import annotations

import random
import time

from orbitwars.bots.base import Deadline
from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.bots.mcts_bot import MCTSAgent


def _snapshot() -> object:
    """Opaque hash of the current global random state."""
    return hash(str(random.getstate()))


def _make_dense_obs():
    """Build a representative mid-game obs via kaggle_environments."""
    from kaggle_environments import make
    random.seed(42)
    env = make("orbit_wars", configuration={"actTimeout": 2.0}, debug=False)
    env.reset(num_agents=2)
    # Advance a few turns so we're not at obs.step==0.
    heur = HeuristicAgent()
    k = heur.as_kaggle_agent()
    for _ in range(20):
        obs0 = env.state[0]["observation"]
        obs1 = env.state[1]["observation"]
        a0 = k(obs0, env.configuration)
        a1 = k(obs1, env.configuration)
        env.step([a0, a1])
    return env.state[1]["observation"]


def main() -> None:
    obs = _make_dense_obs()

    # Warm MCTSAgent (first call has import/class init overhead we don't want
    # to attribute to random consumption).
    mcts = MCTSAgent(rng_seed=0)
    mcts.act(obs, Deadline())

    # Now take a clean measurement.
    random.seed(0xDEADBEEF)
    before = _snapshot()
    t0 = time.perf_counter()
    _ = mcts.act(obs, Deadline())
    wall = (time.perf_counter() - t0) * 1000.0
    after = _snapshot()
    print(f"MCTS.act wall={wall:.0f}ms  random state "
          f"{'CHANGED' if before != after else 'unchanged'}", flush=True)

    # Baseline: a pure HeuristicAgent.act should NOT touch global random.
    heur = HeuristicAgent()
    heur.act(obs, Deadline())  # warm

    random.seed(0xDEADBEEF)
    before = _snapshot()
    _ = heur.act(obs, Deadline())
    after = _snapshot()
    print(f"Heur.act  random state "
          f"{'CHANGED' if before != after else 'unchanged'}", flush=True)


if __name__ == "__main__":
    main()
