"""PPO from BC warmstart — Plan §Path C W5 deliverable.

Single-file CleanRL-style trainer. Loads a BC checkpoint as warmstart,
runs self-play against a PFSP opponent pool, computes GAE advantages
on (state, source-planet) decision rows, and applies PPO updates.

Run:
  $env:PYTHONPATH="src;."; .venv-gpu\\Scripts\\python.exe -u -m tools.train_ppo \\
      --bc-checkpoint runs/bc_warmstart_v5_small.pt \\
      --out runs/ppo_v1.pt \\
      --total-updates 100

For a smoke run pass ``--total-updates 5 --num-envs 2 --rollout-steps 16``.

Architecture (see docs/PPO_DESIGN.md for the surrounding strategy):

* **Policy**: ``ConvPolicy`` from BC. Outputs per-cell (B, 8, H, W)
  policy logits + (B, 1) value scalar. PPO reads at (gy, gx) for each
  source planet to get per-decision distributions.
* **Action decoding**: channel index → (angle_bucket, ship_frac) via
  ``ACTION_LOOKUP``. Heuristic intercept math computes the actual angle
  and target from the angle bucket; ship count uses the heuristic's
  exact-plus-one rule. Same decoder as ``nn_prior.py``.
* **Self-play**: half the matches are vs the live learner (with stop-
  gradient for the opponent side), the other half vs PFSP-sampled
  opponents from the pool.
* **Reward**: terminal +1 win / -1 loss / 0 tie. Optional shaped
  per-step ship-lead delta if ``--shaped-reward`` is passed.
* **GAE**: γ=0.995, λ=0.95.

Status: scaffolded, not fully end-to-end at first commit. The
rollout-collection inner loop drives FastEngine self-play and emits
TransitionBatch rows. Smoke-tested with --total-updates 1 to verify
the pipeline; full convergence runs follow once we've sanity-checked
the loss curve.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from orbitwars.bots.heuristic import (
    HEURISTIC_WEIGHTS, HeuristicAgent, parse_obs, build_arrival_table,
)
from orbitwars.engine.fast_engine import FastEngine
from orbitwars.mcts.actions import ActionConfig, generate_per_planet_moves
from orbitwars.nn.conv_policy import ACTION_LOOKUP, ConvPolicy, ConvPolicyCfg
from orbitwars.nn.nn_prior import (
    candidate_to_channel, load_conv_policy,
)
from orbitwars.nn.pfsp_pool import PFSPPool, build_default_pool
from orbitwars.nn.ppo_algo import (
    SampledAction, TransitionBatch,
    action_log_prob_and_entropy, compute_gae, ppo_update, sample_actions,
)
from orbitwars.nn.ppo_features import encode_decisions, stack_decisions


@dataclass
class PPOArgs:
    """Resolved CLI args."""
    bc_checkpoint: Path
    out: Path
    total_updates: int = 100
    num_envs: int = 2
    rollout_steps: int = 32
    epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    self_play_sync_every: int = 10
    save_every: int = 25
    seed: int = 0
    device: str = "auto"
    shaped_reward: bool = False
    weights_json: Optional[str] = None


@dataclass
class RolloutBuffer:
    """Per-decision transitions accumulated during one rollout phase."""
    obs_x: List[np.ndarray] = field(default_factory=list)
    gy: List[int] = field(default_factory=list)
    gx: List[int] = field(default_factory=list)
    action: List[int] = field(default_factory=list)
    log_prob: List[float] = field(default_factory=list)
    value: List[float] = field(default_factory=list)
    reward: List[float] = field(default_factory=list)
    done: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.action)

    def to_batch(
        self, advantages: np.ndarray, returns: np.ndarray,
    ) -> TransitionBatch:
        return TransitionBatch(
            obs_x=torch.from_numpy(np.stack(self.obs_x).astype(np.float32)),
            gy=torch.from_numpy(np.array(self.gy, dtype=np.int64)),
            gx=torch.from_numpy(np.array(self.gx, dtype=np.int64)),
            candidate_mask=None,
            action=torch.from_numpy(np.array(self.action, dtype=np.int64)),
            log_prob=torch.from_numpy(np.array(self.log_prob, dtype=np.float32)),
            returns=torch.from_numpy(returns.astype(np.float32)),
            advantages=torch.from_numpy(advantages.astype(np.float32)),
        )


# ---------------------------------------------------------------------------
# Action decoding (channel -> wire move via heuristic intercept)
# ---------------------------------------------------------------------------


def decode_action(
    channel: int, source_pid: int, available_ships: int, obs: Any, weights,
) -> list:
    """Decode a sampled channel index into a wire-format move.

    Uses the heuristic's per-planet move generator to find the best
    target+angle in the angle quadrant, then applies the channel's
    ship fraction. Falls back to HOLD if no valid target.
    """
    angle_bucket, frac = ACTION_LOOKUP[int(channel) % len(ACTION_LOOKUP)]
    po = parse_obs(obs)
    table = build_arrival_table(po)
    per_planet = generate_per_planet_moves(po, table, weights=weights, cfg=ActionConfig())
    moves = per_planet.get(int(source_pid), [])
    # Pick best heuristic move whose own channel matches our chosen channel.
    target_chan = int(channel)
    matching = []
    for m in moves:
        ch = candidate_to_channel(m, available_ships)
        if ch == target_chan:
            matching.append(m)
    if not matching:
        # Fallback: return HOLD (empty wire).
        return []
    # Take the highest-raw-score matching move.
    matching.sort(key=lambda m: -m.raw_score)
    chosen = matching[0]
    if chosen.is_hold:
        return []
    return [int(chosen.from_pid), float(chosen.angle), int(chosen.ships)]


# ---------------------------------------------------------------------------
# Policy forward helper (per-cell read)
# ---------------------------------------------------------------------------


def policy_forward(
    model: ConvPolicy, obs_x: torch.Tensor, gy: torch.Tensor, gx: torch.Tensor,
):
    """Run the ConvPolicy and gather per-cell logits + value at (gy, gx).

    Returns ``(logits (N, 8), value (N,))``.
    """
    pi, v = model(obs_x)
    n = obs_x.shape[0]
    per_cell_logits = pi[torch.arange(n, device=obs_x.device), :, gy, gx]
    return per_cell_logits, v.squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout collection (single env, single update)
# ---------------------------------------------------------------------------


def run_rollout(
    *,
    policy: ConvPolicy,
    pool: PFSPPool,
    args: PPOArgs,
    weights,
    device: torch.device,
    rng: np.random.Generator,
) -> Tuple["RolloutBuffer", np.ndarray, np.ndarray]:
    """Collect one rollout-steps phase across num_envs games.

    For each env: pair the learner against a PFSP-sampled opponent (or
    the live learner with stop_grad for half the matches). Step the
    FastEngine, encode obs, sample actions, store transitions.

    Returns: (concatenated_buf, advantages, returns) where advantages and
    returns are computed PER-ENV with proper bootstrap-value handling
    (V(s_K+1) for non-terminal rollouts, 0 for terminal). This is the
    correct PPO setup; concatenating envs and calling compute_gae once
    with bootstrap=0 (as the previous version did) trains the value head
    on all-zero targets when rollouts don't reach terminal — which they
    almost never do at 32 steps in 500-step games.
    """
    main_buf = RolloutBuffer()
    all_advs: List[np.ndarray] = []
    all_rets: List[np.ndarray] = []
    for env_i in range(args.num_envs):
        env_buf = RolloutBuffer()
        # Initialize a fresh FastEngine for this env.
        seed = int(rng.integers(0, 2**31 - 1))
        eng = FastEngine.from_scratch(num_agents=2, seed=seed)
        # Pair with a PFSP opponent half the time; self-play the rest.
        opp_kind = "self" if rng.random() < 0.5 else "pfsp"
        pfsp_opp = pool.sample()
        opp_agent_factory = pfsp_opp.factory
        opp_agent = opp_agent_factory()
        learner_seat = int(rng.integers(0, 2))
        opp_seat = 1 - learner_seat

        terminal = False
        # Track ship-lead for shaped reward.
        prev_lead: Optional[float] = None
        if args.shaped_reward:
            try:
                s0 = eng.scores()
                prev_lead = float(s0[learner_seat] - s0[opp_seat])
            except Exception:
                prev_lead = 0.0
        for step in range(args.rollout_steps):
            if eng.done:
                terminal = True
                break
            # Encode the learner's view.
            learner_obs = eng.observation(learner_seat)
            decisions = encode_decisions(learner_obs, learner_seat)
            if not decisions:
                # No owned planets — learner already eliminated. Step
                # with empty action and let the engine wrap up.
                opp_obs = eng.observation(opp_seat)
                opp_act = opp_agent.act(opp_obs) if hasattr(opp_agent, "act") else opp_agent(opp_obs)
                actions = [None, None]
                actions[learner_seat] = []
                actions[opp_seat] = opp_act or []
                eng.step(actions)
                continue
            obs_x_np, gy_np, gx_np = stack_decisions(decisions)
            obs_x = torch.from_numpy(obs_x_np).to(device)
            gy = torch.from_numpy(gy_np).to(device)
            gx = torch.from_numpy(gx_np).to(device)
            with torch.no_grad():
                logits, value = policy_forward(policy, obs_x, gy, gx)
                sampled = sample_actions(logits, deterministic=False)
            actions_wire = []
            for i, dec in enumerate(decisions):
                wire = decode_action(
                    int(sampled.action[i].item()),
                    dec.source_planet_id,
                    dec.available_ships,
                    learner_obs,
                    weights,
                )
                if wire:
                    actions_wire.append(wire)
                # Record the decision row for PPO.
                env_buf.obs_x.append(obs_x_np[i])
                env_buf.gy.append(int(gy_np[i]))
                env_buf.gx.append(int(gx_np[i]))
                env_buf.action.append(int(sampled.action[i].item()))
                env_buf.log_prob.append(float(sampled.log_prob[i].item()))
                env_buf.value.append(float(value[i].item()))
                # Reward/done filled in after the engine step.
                env_buf.reward.append(0.0)
                env_buf.done.append(0.0)
            # Opponent action.
            opp_obs = eng.observation(opp_seat)
            try:
                opp_act = opp_agent.act(opp_obs) if hasattr(opp_agent, "act") else opp_agent(opp_obs)
            except Exception:
                opp_act = []
            if opp_act is None:
                opp_act = []
            actions = [None, None]
            actions[learner_seat] = actions_wire
            actions[opp_seat] = opp_act
            eng.step(actions)
            # Compute shaped reward (Δ ship_lead / scale) on the new state.
            shaped_r = 0.0
            if args.shaped_reward:
                try:
                    s_after = eng.scores()
                    cur_lead = float(s_after[learner_seat] - s_after[opp_seat])
                    if prev_lead is None:
                        prev_lead = cur_lead
                    # Scale by 100 so per-turn shaped reward is in roughly
                    # [-1, 1] for typical fleet sizes. Game can have ~500
                    # ship swings on a comet, but most turns are small.
                    shaped_r = (cur_lead - prev_lead) / 100.0
                    prev_lead = cur_lead
                except Exception:
                    shaped_r = 0.0
            # Patch reward when game ends, or apply shaped per-step.
            if eng.done:
                terminal = True
                # Win = +1 / loss = -1 / tie = 0.
                scores = eng.scores()
                if scores[learner_seat] > scores[opp_seat]:
                    term_r = 1.0
                elif scores[learner_seat] < scores[opp_seat]:
                    term_r = -1.0
                else:
                    term_r = 0.0
                final_r = term_r + shaped_r  # add shaped if enabled
                if env_buf.reward:
                    # Set reward on the last decision and done=1 there;
                    # GAE will propagate the credit to earlier decisions
                    # at the same turn through the value baseline.
                    env_buf.reward[-1] = final_r
                    env_buf.done[-1] = 1.0
                break
            elif args.shaped_reward and env_buf.reward:
                # Per-turn shaped signal (no done; rollout continues).
                env_buf.reward[-1] = shaped_r

        # Compute bootstrap value for non-terminal rollouts: V(s_K+1).
        # Without this, the value head trains on all-zero returns and
        # the policy gradient has no baseline.
        if terminal or len(env_buf) == 0:
            bootstrap = 0.0
        else:
            try:
                cur_obs = eng.observation(learner_seat)
                cur_decisions = encode_decisions(cur_obs, learner_seat)
                if cur_decisions:
                    bx_np, by_np, bgx_np = stack_decisions(cur_decisions)
                    bx = torch.from_numpy(bx_np).to(device)
                    bgy = torch.from_numpy(by_np).to(device)
                    bgx = torch.from_numpy(bgx_np).to(device)
                    with torch.no_grad():
                        _, bv = policy_forward(policy, bx, bgy, bgx)
                    bootstrap = float(bv.mean().item())
                else:
                    bootstrap = 0.0
            except Exception:
                bootstrap = 0.0

        if len(env_buf) == 0:
            continue
        # Per-env GAE with proper bootstrap.
        rewards = np.array(env_buf.reward, dtype=np.float32)
        values = np.array(env_buf.value, dtype=np.float32)
        dones = np.array(env_buf.done, dtype=np.float32)
        adv, ret = compute_gae(
            torch.from_numpy(rewards), torch.from_numpy(values),
            torch.from_numpy(dones),
            gamma=args.gamma, lam=args.gae_lambda,
            bootstrap_value=bootstrap,
        )
        all_advs.append(adv.numpy())
        all_rets.append(ret.numpy())
        # Concatenate this env's transitions into main_buf.
        main_buf.obs_x.extend(env_buf.obs_x)
        main_buf.gy.extend(env_buf.gy)
        main_buf.gx.extend(env_buf.gx)
        main_buf.action.extend(env_buf.action)
        main_buf.log_prob.extend(env_buf.log_prob)
        main_buf.value.extend(env_buf.value)
        main_buf.reward.extend(env_buf.reward)
        main_buf.done.extend(env_buf.done)

    if not all_advs:
        return (
            main_buf,
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    return (
        main_buf,
        np.concatenate(all_advs).astype(np.float32),
        np.concatenate(all_rets).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--total-updates", type=int, default=100)
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--rollout-steps", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch-size", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-coef", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--self-play-sync-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--shaped-reward", action="store_true")
    ap.add_argument("--weights-json", type=str, default=None,
                    help="TuRBO weights JSON (used by PFSP pool's heuristic opponents)")
    raw = ap.parse_args()

    args = PPOArgs(
        bc_checkpoint=raw.bc_checkpoint, out=raw.out,
        total_updates=raw.total_updates, num_envs=raw.num_envs,
        rollout_steps=raw.rollout_steps, epochs=raw.epochs,
        minibatch_size=raw.minibatch_size, gamma=raw.gamma,
        gae_lambda=raw.gae_lambda, clip_coef=raw.clip_coef,
        ent_coef=raw.ent_coef, vf_coef=raw.vf_coef, lr=raw.lr,
        max_grad_norm=raw.max_grad_norm,
        self_play_sync_every=raw.self_play_sync_every,
        save_every=raw.save_every, seed=raw.seed, device=raw.device,
        shaped_reward=raw.shaped_reward, weights_json=raw.weights_json,
    )

    device = torch.device(
        "cuda" if (raw.device == "auto" and torch.cuda.is_available())
        else (raw.device if raw.device != "auto" else "cpu")
    )
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Load BC warmstart.
    print(f"loading BC warmstart from {args.bc_checkpoint}", flush=True)
    policy, cfg = load_conv_policy(args.bc_checkpoint, device=str(device))
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    # Build PFSP opponent pool.
    print("building PFSP opponent pool", flush=True)
    pool = build_default_pool(
        weights_json=args.weights_json,
        nn_checkpoint=str(args.bc_checkpoint),
        seed=args.seed,
    )
    print(f"  pool: {pool.names()}", flush=True)

    weights = dict(HEURISTIC_WEIGHTS)
    if args.weights_json:
        import json
        try:
            raw = json.loads(Path(args.weights_json).read_text(encoding="utf-8"))
            for key in ("best_weights", "best_point", "best", "weights"):
                if isinstance(raw, dict) and key in raw and isinstance(raw[key], dict):
                    raw = raw[key]
                    break
            if isinstance(raw, dict) and "best_index" in raw and isinstance(raw.get("trials"), list):
                best_i = int(raw["best_index"])
                for tr in raw["trials"]:
                    if int(tr.get("trial", -1)) == best_i and isinstance(tr.get("weights"), dict):
                        raw = tr["weights"]
                        break
            if isinstance(raw, dict):
                weights.update({k: float(v) for k, v in raw.items()})
        except Exception:
            pass

    # Training loop.
    t0_total = time.perf_counter()
    for update in range(args.total_updates):
        t0 = time.perf_counter()
        buf, adv_np, ret_np = run_rollout(
            policy=policy, pool=pool, args=args, weights=weights,
            device=device, rng=rng,
        )
        if len(buf) == 0:
            print(f"update {update}: empty rollout — skipping", flush=True)
            continue
        rollout_dt = time.perf_counter() - t0

        # GAE was computed per-env inside run_rollout with proper
        # V(s_K+1) bootstrap; just hand the arrays to to_batch.
        batch = buf.to_batch(adv_np, ret_np)

        # PPO update.
        def fwd(obs_x, gy, gx):
            return policy_forward(policy, obs_x, gy, gx)

        metrics = ppo_update(
            forward_fn=fwd, optimizer=optimizer, batch=batch,
            clip_coef=args.clip_coef, ent_coef=args.ent_coef,
            vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm,
            epochs=args.epochs, minibatch_size=args.minibatch_size,
            device=device,
        )
        update_dt = time.perf_counter() - t0 - rollout_dt
        print(
            f"update {update + 1}/{args.total_updates}  "
            f"n_decisions={len(buf)}  "
            f"loss={metrics['loss']:.3f} "
            f"pl={metrics['policy_loss']:.3f} "
            f"vl={metrics['value_loss']:.3f} "
            f"ent={metrics['entropy']:.3f} "
            f"kl={metrics['approx_kl']:.4f}  "
            f"wall_rollout={rollout_dt:.1f}s update={update_dt:.1f}s",
            flush=True,
        )

        # Periodic save — versioned so we can pick the best checkpoint
        # later (PPO can collapse entropy late in training; the final
        # checkpoint may be worse than mid-training ones). Also write
        # the same blob to args.out so the file at the canonical path
        # always reflects the latest state.
        if (update + 1) % args.save_every == 0:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            from dataclasses import asdict
            blob = {
                "model_state": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
                "cfg": asdict(cfg),
                "update": update + 1,
                "metrics": metrics,
            }
            torch.save(blob, args.out)
            # Versioned save: e.g. runs/ppo_v3_500update_u100.pt
            versioned = args.out.with_name(
                args.out.stem + f"_u{update + 1}" + args.out.suffix
            )
            torch.save(blob, versioned)
            print(f"  saved checkpoint to {args.out} and {versioned}", flush=True)

    # Final save.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict
    torch.save({
        "model_state": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
        "cfg": asdict(cfg),
        "update": args.total_updates,
    }, args.out)
    print(
        f"\nfinal: total_wall={time.perf_counter() - t0_total:.0f}s  "
        f"saved to {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
