"""Verify the NN prior is *actually active* by running v12 (NN prior) and
v12-baseline (no NN prior, otherwise identical config) on the same seed
and checking that they produce different action sequences.

Why this matters: in W3 we shipped v10 and discovered margin=0.5 +
fast-rollouts produced a BYTE-IDENTICAL outcome sequence to the
unmodified agent — i.e. without a learned prior, the extra sims don't
move the wire. The hypothesis the v12 ship gate tests is: with a
prior (even random-init for now), the prior CAN move the wire.

If this script reports "actions diverge at turn T", the wiring is
active. If "actions identical", we have a wiring bug to fix BEFORE
spending a Kaggle slot on v12.

Run:
  $env:PYTHONPATH="src;."; .venv\\Scripts\\python.exe tools\\smoke_v12_diff_vs_baseline.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import torch  # type: ignore[import-not-found]

from orbitwars.bots.base import Deadline
from orbitwars.bots.mcts_bot import MCTSAgent
from orbitwars.mcts.gumbel_search import GumbelConfig
from orbitwars.nn.conv_policy import ConvPolicy, ConvPolicyCfg
from orbitwars.nn.nn_prior import make_nn_prior_fn
from kaggle_environments import make


def _mcts_cfg() -> GumbelConfig:
    """v12 default: rollout=fast, margin=0.5, exp3 sim_move."""
    cfg = GumbelConfig()
    cfg.rollout_policy = "fast"
    cfg.anchor_improvement_margin = 0.5
    cfg.sim_move_variant = "exp3"
    cfg.exp3_eta = 0.3
    return cfg


def _build_baseline_agent():
    return MCTSAgent(gumbel_cfg=_mcts_cfg(), rng_seed=0).as_kaggle_agent()


def _build_nn_agent():
    torch.manual_seed(0)
    cfg = ConvPolicyCfg()
    model = ConvPolicy(cfg)
    model.eval()
    fn = make_nn_prior_fn(model, cfg)
    return MCTSAgent(
        gumbel_cfg=_mcts_cfg(),
        rng_seed=0,
        move_prior_fn=fn,
    ).as_kaggle_agent()


def run_game(agent_a, label: str):
    """Run agent_a vs random in seed=42. Returns ordered list of seat-0
    actions per turn."""
    env = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
    env.reset()
    actions = []
    # Hijack: env.run([agent_a, "random"]) but capture actions per step.
    # Simpler: use env.step manually via the agent's __call__.
    obs0 = env.state[0].observation
    cfg = env.configuration
    env.run([agent_a, "random"])
    # Pull action history if exposed; fall back to logs.
    # kaggle_environments stores the agent's action in state[i].action
    # for the just-played turn, but we want the FULL trajectory.
    # Easiest is to instrument env.run with a per-step hook — but
    # kaggle_environments doesn't expose that cleanly. We re-run with
    # a logging wrapper.
    return env  # caller compares action-by-turn


def _wrap_with_log(agent_callable, log: list):
    def wrapped(obs, cfg=None):
        a = agent_callable(obs, cfg)
        log.append(a)
        return a
    return wrapped


def main() -> int:
    print("=== v12-baseline (no NN prior) ===", flush=True)
    base_log = []
    base_agent = _wrap_with_log(_build_baseline_agent(), base_log)
    env_b = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
    t0 = time.perf_counter()
    env_b.run([base_agent, "random"])
    print(
        f"  done in {time.perf_counter() - t0:.1f}s  "
        f"steps={env_b.state[0].observation.step}  "
        f"reward={env_b.state[0].reward}  "
        f"actions logged={len(base_log)}",
        flush=True,
    )

    print("\n=== v12 (with random-init NN prior) ===", flush=True)
    nn_log = []
    nn_agent = _wrap_with_log(_build_nn_agent(), nn_log)
    env_n = make("orbit_wars", configuration={"actTimeout": 60}, debug=False)
    t0 = time.perf_counter()
    env_n.run([nn_agent, "random"])
    print(
        f"  done in {time.perf_counter() - t0:.1f}s  "
        f"steps={env_n.state[0].observation.step}  "
        f"reward={env_n.state[0].reward}  "
        f"actions logged={len(nn_log)}",
        flush=True,
    )

    print("\n=== Action sequence diff ===", flush=True)
    diverge_at = -1
    n_diff = 0
    n_compared = min(len(base_log), len(nn_log))
    for i in range(n_compared):
        if base_log[i] != nn_log[i]:
            if diverge_at < 0:
                diverge_at = i
            n_diff += 1
    if diverge_at < 0:
        print("BYTE-IDENTICAL across all turns. NN prior is NOT moving the wire.")
        print("(Same as v10 vs v8: anchor-lock is compute-bound, not just margin-bound.)")
        return 1
    print(
        f"actions diverge starting at turn {diverge_at}; "
        f"{n_diff}/{n_compared} turns differ ({100*n_diff/max(n_compared,1):.1f}%)"
    )
    print(f"first divergent pair (turn {diverge_at}):")
    print(f"  baseline: {base_log[diverge_at]}")
    print(f"  NN-prior: {nn_log[diverge_at]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
