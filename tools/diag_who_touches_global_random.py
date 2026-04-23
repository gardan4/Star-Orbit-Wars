"""Track who consumes global `random` during env.run.

Hypothesis: MCTS's search transitively consumes entropy from the global
`random` module (used by kaggle_environments's orbit_wars engine for
comet ship sizing at steps {50, 150, 250, 350, 450}). Consuming ANY
amount perturbs the comet stream, changing game outcome even when the
agent's wire action exactly matches a heuristic baseline.

Previous diagnostics showed:
- heur-vs-heur env.run on seed=123 -> rewards (-1, 1), P1 wins
- MCTS(P1)-vs-heur(P0) env.run on seed=123 -> rewards [1, -1], P1 loses
- MCTS's wire action at seat 1 exactly matches a shadow HeuristicAgent's
  on EVERY turn (0/479 diverged)

Since actions are identical, the only explanation is engine-visible state
diverging — the prime suspect is global random state.

This diag wraps `random` functions with counters, runs a game, and reports
total calls by caller (via stack inspection on a sample).
"""
from __future__ import annotations

import random
import sys
import traceback
from collections import Counter

from kaggle_environments import make

from orbitwars.bots.heuristic import HeuristicAgent
from orbitwars.bots.mcts_bot import MCTSAgent


def wrap_random(tag: str):
    """Replace random.{randint,uniform,random,choice,seed,getstate,setstate}
    with counting wrappers. Return (counter, callers) and uninstall fn.
    """
    counter: Counter[str] = Counter()
    callers: Counter[str] = Counter()

    funcs = ["randint", "uniform", "random", "choice", "seed", "getstate",
             "setstate", "choices", "sample", "shuffle", "getrandbits",
             "gauss", "triangular"]
    originals = {}

    def _top_frame():
        # Walk the stack; skip this module; return caller site.
        stk = traceback.extract_stack()
        # stk[-1] is inside wrap_call; stk[-2] is _wrapped; stk[-3] is caller.
        if len(stk) >= 3:
            f = stk[-3]
            return f"{f.filename.split(chr(92))[-1]}:{f.lineno}"
        return "?"

    for fname in funcs:
        if not hasattr(random, fname):
            continue
        orig = getattr(random, fname)
        originals[fname] = orig

        def make_wrapped(fn_name, fn):
            def _wrapped(*a, **kw):
                counter[fn_name] += 1
                # Sample callers for the most-common fn to keep overhead low.
                if counter[fn_name] <= 50 or counter[fn_name] % 500 == 0:
                    callers[_top_frame()] += 1
                return fn(*a, **kw)
            _wrapped.__name__ = fn.__name__
            return _wrapped

        setattr(random, fname, make_wrapped(fname, orig))

    def uninstall():
        for fname, orig in originals.items():
            setattr(random, fname, orig)

    return counter, callers, uninstall


def run_one(label: str, seed: int, p1_factory):
    counter, callers, uninstall = wrap_random(label)
    try:
        # Reset global random state (seed() is now wrapped — unwrap briefly).
        uninstall()
        random.seed(seed)
        counter, callers, uninstall = wrap_random(label)

        env = make("orbit_wars", configuration={"actTimeout": 2.0}, debug=False)
        p0 = HeuristicAgent().as_kaggle_agent()
        p1 = p1_factory()
        env.run([p0, p1])
        rewards = [int(env.state[i]["reward"] if env.state[i]["reward"] is not None else 0)
                   for i in range(2)]
        total_calls = sum(counter.values())
        print(f"[{label}] seed={seed} rewards={rewards} total_global_random_calls={total_calls}",
              flush=True)
        for fname, n in counter.most_common():
            print(f"    random.{fname:10s} : {n}", flush=True)
        print("    top caller sites:", flush=True)
        for site, n in callers.most_common(10):
            print(f"      {n:6d}  {site}", flush=True)
    finally:
        uninstall()


def main() -> None:
    seed = 123

    def heur_factory():
        return HeuristicAgent().as_kaggle_agent()

    def mcts_factory():
        return MCTSAgent(rng_seed=0).as_kaggle_agent()

    print("=== heur-vs-heur baseline ===", flush=True)
    run_one("heur", seed, heur_factory)
    print("", flush=True)
    print("=== mcts-vs-heur (mcts at seat 1) ===", flush=True)
    run_one("mcts", seed, mcts_factory)


if __name__ == "__main__":
    main()
