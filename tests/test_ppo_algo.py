"""Tests for PPO algorithm primitives in orbitwars.nn.ppo_algo."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from orbitwars.nn.ppo_algo import (
    SampledAction,
    TransitionBatch,
    action_log_prob_and_entropy,
    compute_gae,
    ppo_update,
    safe_logits,
    sample_actions,
)


def test_sample_actions_returns_valid_categorical():
    """sample_actions output should be a real Categorical sample."""
    torch.manual_seed(0)
    logits = torch.randn(5, 8)
    sampled = sample_actions(logits)
    assert isinstance(sampled, SampledAction)
    assert sampled.action.shape == (5,)
    assert sampled.log_prob.shape == (5,)
    assert sampled.entropy.shape == (5,)
    assert (sampled.action >= 0).all() and (sampled.action < 8).all()
    assert torch.isfinite(sampled.log_prob).all()
    assert (sampled.entropy >= 0).all()


def test_sample_actions_deterministic_picks_argmax():
    logits = torch.tensor([[0.0, 0.0, 5.0, 0.0]])
    sampled = sample_actions(logits, deterministic=True)
    assert sampled.action.item() == 2


def test_sample_actions_respects_candidate_mask():
    """Masked-out candidates should never be sampled."""
    torch.manual_seed(0)
    logits = torch.zeros(20, 4)
    mask = torch.tensor([[True, False, True, False]] * 20)
    out = sample_actions(logits, candidate_mask=mask)
    # All samples should be in {0, 2}.
    assert ((out.action == 0) | (out.action == 2)).all()


def test_safe_logits_replaces_all_inf_rows_with_zeros():
    """A row with all -inf logits would crash Categorical; safe_logits
    swaps it out for zeros (uniform)."""
    logits = torch.tensor([
        [1.0, 2.0, 3.0],
        [float("-inf")] * 3,
        [0.0, 1.0, 0.5],
    ])
    safe = safe_logits(logits)
    # Row 0 unchanged
    assert torch.allclose(safe[0], logits[0])
    # Row 1 replaced with zeros
    assert torch.allclose(safe[1], torch.zeros(3))
    # Row 2 unchanged
    assert torch.allclose(safe[2], logits[2])


def test_log_prob_matches_torch_categorical():
    """Hand-validate the action_log_prob_and_entropy helper."""
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    action = torch.tensor([2])
    lp, ent = action_log_prob_and_entropy(logits, action)
    # Hand-compute: softmax → (~0.09, ~0.245, ~0.665), log(0.665) ≈ -0.408
    assert lp.item() == pytest.approx(-0.4076, abs=1e-3)
    assert ent.item() > 0.0  # nonzero entropy for any non-degenerate dist


def test_compute_gae_terminal_only_reward_decays_correctly():
    """For a 1-step reward at the end, GAE returns should equal the
    reward at the last step and discounted reward at earlier steps."""
    T = 4
    rewards = torch.tensor([0.0, 0.0, 0.0, 1.0])
    values = torch.zeros(T)
    dones = torch.tensor([0.0, 0.0, 0.0, 1.0])
    gamma = 0.9
    adv, ret = compute_gae(rewards, values, dones, gamma=gamma, lam=1.0)
    # With λ=1 and γ=0.9, advantage at each step = γ^(T-t-1) * 1.0.
    for t in range(T):
        expected = gamma ** (T - t - 1)
        assert ret[t].item() == pytest.approx(expected, abs=1e-5)


def test_compute_gae_zero_reward_zero_advantage():
    """No reward → no advantage anywhere."""
    T = 5
    rewards = torch.zeros(T)
    values = torch.zeros(T)
    dones = torch.zeros(T)
    adv, ret = compute_gae(rewards, values, dones)
    assert torch.allclose(adv, torch.zeros(T))
    assert torch.allclose(ret, torch.zeros(T))


def test_compute_gae_uses_bootstrap_when_no_terminal():
    """If the final state isn't terminal, GAE uses bootstrap_value
    instead of treating it as zero."""
    T = 2
    rewards = torch.tensor([0.0, 0.0])
    values = torch.tensor([0.0, 0.0])
    dones = torch.tensor([0.0, 0.0])
    bootstrap = 1.0
    adv, ret = compute_gae(
        rewards, values, dones, gamma=1.0, lam=1.0, bootstrap_value=bootstrap,
    )
    # δ_1 = 0 + 1*1 - 0 = 1; δ_0 = 0 + 1*0 - 0 = 0; with λ=1, A_1 = 1, A_0 = 1.
    assert ret[1].item() == pytest.approx(1.0, abs=1e-5)
    assert ret[0].item() == pytest.approx(1.0, abs=1e-5)


def test_ppo_update_changes_policy_when_advantage_is_positive():
    """Sanity: with all-positive advantages on the chosen action, PPO
    should INCREASE that action's log-prob after the update."""
    torch.manual_seed(0)
    K = 4
    N = 16

    class ToyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, K)
            self.v = torch.nn.Linear(2, 1)

        def forward(self, obs_x, gy, gx):
            # obs_x: (N, 2)
            return self.fc(obs_x), self.v(obs_x).squeeze(-1)

    policy = ToyPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

    obs_x = torch.randn(N, 2)
    gy = torch.zeros(N, dtype=torch.long)
    gx = torch.zeros(N, dtype=torch.long)
    action = torch.randint(0, K, (N,))
    with torch.no_grad():
        logits, value = policy(obs_x, gy, gx)
        old_lp, _ = action_log_prob_and_entropy(logits, action)
    advantages = torch.ones(N)
    returns = torch.zeros(N)
    batch = TransitionBatch(
        obs_x=obs_x, gy=gy, gx=gx, candidate_mask=None,
        action=action, log_prob=old_lp,
        returns=returns, advantages=advantages,
    )
    metrics = ppo_update(
        forward_fn=policy, optimizer=optimizer, batch=batch,
        epochs=4, minibatch_size=8,
    )
    assert metrics["loss"] != 0.0
    # After update: chosen actions should have higher log-prob.
    with torch.no_grad():
        new_logits, _ = policy(obs_x, gy, gx)
        new_lp, _ = action_log_prob_and_entropy(new_logits, action)
    # Allow some noise but expect on average new_lp >= old_lp for chosen actions
    # (positive advantage = "do this more").
    assert (new_lp - old_lp).mean().item() > 0.0


def test_ppo_update_handles_empty_batch():
    """Empty batch (no decisions in a turn) returns zero metrics
    without crashing."""
    batch = TransitionBatch(
        obs_x=torch.zeros(0, 12, 50, 50),
        gy=torch.zeros(0, dtype=torch.long),
        gx=torch.zeros(0, dtype=torch.long),
        candidate_mask=None,
        action=torch.zeros(0, dtype=torch.long),
        log_prob=torch.zeros(0),
        returns=torch.zeros(0),
        advantages=torch.zeros(0),
    )
    metrics = ppo_update(
        forward_fn=lambda *_: (torch.zeros(0, 8), torch.zeros(0)),
        optimizer=torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))]),
        batch=batch,
    )
    assert metrics["loss"] == 0.0
    assert metrics["policy_loss"] == 0.0
