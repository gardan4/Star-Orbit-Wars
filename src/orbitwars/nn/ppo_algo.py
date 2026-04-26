"""PPO algorithm primitives — sampling, GAE, update step.

Pure-PyTorch math, no I/O or env dependencies. The data shape is
designed for **per-decision** rows (one decision = one source planet
in one observation), matching how MCTSAgent decomposes a turn:

* ``logits``: ``(N_decisions, K)`` — categorical over K action
  channels per decision. For our ConvPolicy, K=8 (4 angle buckets ×
  2 ship-fractions). For an MLP-over-candidates architecture,
  K=candidate_count (e.g. 8 nearest-target slots).
* ``value``: ``(N_decisions,)`` — scalar baseline per decision.

The update treats every decision row as an independent training
example, just like CleanRL PPO at minibatch granularity.

Adapted in spirit (not copied) from the public Kaggle tutorial
``kashiwaba/orbit-wars-reinforcement-learning-tutorial`` — standard
PPO clipped objective, GAE for advantages, value MSE.

See docs/PPO_DESIGN.md for the surrounding training-loop design.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass
class SampledAction:
    """Output of ``sample_actions`` — drop-in record for the rollout buffer."""
    action: torch.Tensor       # (N,) int64 — sampled channel index
    log_prob: torch.Tensor     # (N,) float — log π(action | state)
    entropy: torch.Tensor      # (N,) float — H[π(·|state)]


@dataclass
class TransitionBatch:
    """One epoch's worth of (state, action, advantage, return) for PPO."""
    obs_x: torch.Tensor               # (N, C, H, W) — encoded obs
    gy: torch.Tensor                  # (N,) int64 — source planet grid row
    gx: torch.Tensor                  # (N,) int64 — source planet grid col
    candidate_mask: Optional[torch.Tensor]  # (N, K) bool, optional
    action: torch.Tensor              # (N,) int64
    log_prob: torch.Tensor            # (N,) old policy log prob
    returns: torch.Tensor             # (N,) GAE returns
    advantages: torch.Tensor          # (N,) GAE advantages


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def safe_logits(logits: torch.Tensor) -> torch.Tensor:
    """Replace any all-(-inf) row (e.g. all candidates masked out) with
    a uniform distribution to keep ``Categorical`` happy. Caller should
    have masked invalid rows out of training already; this is a
    backstop for inference paths."""
    invalid = ~torch.isfinite(logits).any(dim=-1)
    if not invalid.any():
        return logits
    safe = logits.clone()
    safe[invalid] = 0.0
    return safe


def sample_actions(
    logits: torch.Tensor,
    *,
    deterministic: bool = False,
    candidate_mask: Optional[torch.Tensor] = None,
) -> SampledAction:
    """Sample from per-decision logits.

    Args:
      logits: ``(N, K)``. Per-row categorical logits.
      deterministic: if True, take ``argmax`` instead of sampling. Used
        at inference time and for "Self-play opponent" pinning.
      candidate_mask: ``(N, K)`` bool. False entries are forced to
        ``-inf`` before sampling.
    """
    if candidate_mask is not None:
        masked = logits.masked_fill(~candidate_mask, float("-inf"))
    else:
        masked = logits
    masked = safe_logits(masked)
    dist = Categorical(logits=masked)
    if deterministic:
        action = masked.argmax(dim=-1)
    else:
        action = dist.sample()
    log_prob = dist.log_prob(action)
    entropy = dist.entropy()
    return SampledAction(action=action, log_prob=log_prob, entropy=entropy)


def action_log_prob_and_entropy(
    logits: torch.Tensor, action: torch.Tensor,
    *, candidate_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute (log_prob, entropy) for given actions under the current policy.

    Used inside ``ppo_update`` to evaluate the new policy on old actions.
    """
    if candidate_mask is not None:
        masked = logits.masked_fill(~candidate_mask, float("-inf"))
    else:
        masked = logits
    masked = safe_logits(masked)
    dist = Categorical(logits=masked)
    return dist.log_prob(action), dist.entropy()


# ---------------------------------------------------------------------------
# Advantage estimation (GAE)
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float = 0.995,
    lam: float = 0.95,
    bootstrap_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard GAE-λ over a single trajectory.

    Args:
      rewards: ``(T,)`` float — terminal-only or shaped per-step rewards.
      values: ``(T,)`` float — value-head estimates at each step.
      dones: ``(T,)`` bool/float — 1 if state s_t was terminal (next
        bootstrap blocked).
      gamma: discount factor. Default 0.995 (long horizon, 500-turn
        games).
      lam: GAE λ. 0.95 standard.
      bootstrap_value: V(s_T) for the final state. 0 for terminal,
        otherwise the value-head output at s_T.

    Returns:
      ``(advantages, returns)``, each ``(T,)`` float.
    """
    assert rewards.dim() == 1 and values.dim() == 1 and dones.dim() == 1
    assert rewards.shape == values.shape == dones.shape
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_adv = 0.0
    next_value = float(bootstrap_value)
    for t in reversed(range(T)):
        nonterm = 1.0 - float(dones[t])
        delta = float(rewards[t]) + gamma * next_value * nonterm - float(values[t])
        last_adv = delta + gamma * lam * nonterm * last_adv
        advantages[t] = last_adv
        next_value = float(values[t])
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------


def ppo_update(
    *,
    forward_fn,                 # (obs_x, gy, gx) -> (logits, value), shapes (B,K), (B,)
    optimizer: torch.optim.Optimizer,
    batch: TransitionBatch,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    epochs: int = 4,
    minibatch_size: int = 256,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Run a PPO update over the transition batch.

    Args:
      forward_fn: callable that takes ``(obs_x, gy, gx)`` and returns
        ``(logits, value)``. Lets the caller compose ConvPolicy +
        per-cell extraction without us coupling to a specific module.
      optimizer: torch optimizer over the policy/value parameters.
      batch: pre-computed transitions from rollout collection.
      clip_coef: PPO clip range. Default 0.2.
      ent_coef: entropy bonus weight. Default 0.01.
      vf_coef: value loss weight. Default 0.5.
      max_grad_norm: gradient clip. Default 0.5.
      epochs: PPO update epochs. Default 4.
      minibatch_size: minibatch size. Default 256.
      device: target device.

    Returns:
      Dict of mean metrics (``loss``, ``policy_loss``, ``value_loss``,
      ``entropy``, ``approx_kl``).
    """
    if batch.obs_x.shape[0] == 0:
        return {
            "loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
            "entropy": 0.0, "approx_kl": 0.0,
        }
    if device is None:
        device = batch.obs_x.device
    obs_x = batch.obs_x.to(device)
    gy = batch.gy.to(device)
    gx = batch.gx.to(device)
    cmask = batch.candidate_mask.to(device).bool() if batch.candidate_mask is not None else None
    old_lp = batch.log_prob.to(device)
    action = batch.action.to(device)
    returns = batch.returns.to(device)
    advantages = batch.advantages.to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    n = obs_x.shape[0]
    minibatch_size = min(n, max(1, minibatch_size))
    metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    updates = 0
    for _ in range(epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, minibatch_size):
            idx = order[start:start + minibatch_size]
            logits, value = forward_fn(obs_x[idx], gy[idx], gx[idx])
            mb_mask = cmask[idx] if cmask is not None else None
            new_lp, ent = action_log_prob_and_entropy(
                logits, action[idx], candidate_mask=mb_mask,
            )
            ratio = (new_lp - old_lp[idx]).exp()
            surr1 = -advantages[idx] * ratio
            surr2 = -advantages[idx] * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
            policy_loss = torch.maximum(surr1, surr2).mean()
            value_loss = 0.5 * (returns[idx] - value).pow(2).mean()
            entropy_mean = ent.mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Clip across whatever parameters the optimizer controls.
            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (old_lp[idx] - new_lp).mean()
            metrics["loss"] += float(loss.detach().cpu())
            metrics["policy_loss"] += float(policy_loss.detach().cpu())
            metrics["value_loss"] += float(value_loss.detach().cpu())
            metrics["entropy"] += float(entropy_mean.detach().cpu())
            metrics["approx_kl"] += float(approx_kl.detach().cpu())
            updates += 1
    return {k: v / max(updates, 1) for k, v in metrics.items()}
