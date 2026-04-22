"""Diagnostic: verify that hard_stop_at propagation actually bounds a single rollout.

Profile pass 5 showed n_sim=3, rollouts_ms=1272ms on step 35 — per-rollout
avg of ~424ms. That shouldn't be possible if the hard_stop_at threading
works: a rollout starting AFTER the outer deadline has fired should
short-circuit on the very first check inside `_rollout_value`.

This script exercises three cases on a dense mid-game state:
  (1) hard_stop_at already in the past  -> expect ~1ms rollout wall cost
  (2) hard_stop_at  = now + 50ms        -> expect ~50-80ms rollout cost
  (3) no hard_stop_at / far-future      -> expect "natural" rollout cost

If (1) reports hundreds of ms, the deadline propagation is broken for
some path we haven't covered. If (1) is tight, the profile's tall-tail
must come from rollouts that STARTED before the deadline and now we
need to break earlier WITHIN a single rollout.
"""
from __future__ import annotations

import copy
import random
import time
from statistics import mean

from kaggle_environments import make

from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.engine.fast_engine import FastEngine
from orbitwars.mcts.gumbel_search import _rollout_value


def build_dense_state(n_warmup_turns: int = 200):
    """Spin up a game to step n_warmup_turns via two HeuristicAgents so
    the resulting state has many fleets and captured planets — the dense
    regime where the rollouts-blow-up shows up."""
    random.seed(42)
    env = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
    env.reset(num_agents=2)
    env.step([[], []])
    heur = HeuristicAgent().as_kaggle_agent()
    for _ in range(n_warmup_turns):
        obs0 = env.state[0].observation
        obs1 = env.state[1].observation
        a0 = heur(obs0)
        a1 = heur(obs1)
        env.step([a0, a1])
        if env.done:
            break
    eng = FastEngine.from_official_obs(env.state[0].observation, num_agents=2)
    return eng.state, env.state[0].observation


def time_rollout(base_state, my_player, hard_stop_at, depth=15, n=30):
    """Run `n` rollouts and report the distribution of per-rollout wall time.

    Each rollout deepcopy's the base state so the cloned engine is
    independent of the previous rollout's state mutations.
    """
    def opp_factory():
        return HeuristicAgent()
    def my_future_factory():
        return HeuristicAgent()

    # Build a concrete action for turn-0 — just a no-op joint, so the
    # cost we measure is dominated by rollout plies, not action-building.
    my_action = []

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        _rollout_value(
            base_state=base_state,
            my_player=my_player,
            my_action=my_action,
            opp_agent_factory=opp_factory,
            my_future_factory=my_future_factory,
            depth=depth,
            num_agents=2,
            rng=random.Random(0),
            deadline_fn=(
                (lambda: time.perf_counter() >= hard_stop_at)
                if hard_stop_at is not None else None
            ),
            hard_stop_at=hard_stop_at,
        )
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def summarize(times):
    ts = sorted(times)
    p50 = ts[len(ts) // 2]
    p95 = ts[int(len(ts) * 0.95)]
    mx = ts[-1]
    return f"n={len(ts)}  mean={mean(ts):.1f}ms  p50={p50:.1f}ms  p95={p95:.1f}ms  max={mx:.1f}ms"


def measure_at_step(n_warmup_turns):
    print(f"\n=== Warmup {n_warmup_turns} turns ===")
    base_state, obs = build_dense_state(n_warmup_turns=n_warmup_turns)
    print(f"  step={obs['step']}  fleets={len(obs['fleets'])}  planets={len(obs['planets'])}")

    # Warmup
    _ = time_rollout(base_state, 0, time.perf_counter() + 10.0, depth=15, n=3)

    # (1) hard_stop already fired — rollout should short-circuit instantly
    ts_expired = time_rollout(base_state, 0, time.perf_counter() - 0.001, depth=15, n=30)
    print(f"[expired hard_stop_at] {summarize(ts_expired)}")

    # (2) hard_stop 50ms in the future
    ts_50ms = time_rollout(base_state, 0, time.perf_counter() + 0.050, depth=15, n=30)
    print(f"[hard_stop +50ms]     {summarize(ts_50ms)}")

    # (3) far-future hard_stop — natural rollout cost
    ts_natural = time_rollout(base_state, 0, time.perf_counter() + 10.0, depth=15, n=30)
    print(f"[hard_stop +10s]      {summarize(ts_natural)}")


def main():
    # Larger sample at step 35 and step 150 (where profile showed tail)
    print("Searching for rollout-time outliers across 200 samples ...")
    for n in [35, 150]:
        print(f"\n=== Warmup {n} turns (N=200) ===")
        base_state, obs = build_dense_state(n_warmup_turns=n)
        print(f"  step={obs['step']}  fleets={len(obs['fleets'])}  planets={len(obs['planets'])}")
        _ = time_rollout(base_state, 0, time.perf_counter() + 10.0, depth=15, n=3)
        ts_natural = time_rollout(base_state, 0, time.perf_counter() + 10.0, depth=15, n=200)
        print(f"[natural] {summarize(ts_natural)}")
        outliers = [t for t in ts_natural if t > 500.0]
        if outliers:
            print(f"  outliers > 500ms: {len(outliers)}  values: {[f'{t:.0f}' for t in outliers[:10]]}")
        else:
            print("  no outliers > 500ms")


if __name__ == "__main__":
    main()
