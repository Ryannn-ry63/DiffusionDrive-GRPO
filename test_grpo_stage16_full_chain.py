"""Focused mathematical tests for Stage-16 full-chain DiffGRPO."""

import pytest
import torch

from navsim.agents.diffusiondrive.diffusion_grpo import (
    diagonal_gaussian_log_prob_mean,
)
from navsim.agents.diffusiondrive.transfuser_loss import (
    compute_full_chain_diffgrpo_objective,
)


def test_mean_log_probability_is_coordinate_count_invariant_and_differentiable():
    mean = torch.zeros((1, 2, 1, 1), requires_grad=True)
    value = torch.ones_like(mean)
    one = diagonal_gaussian_log_prob_mean(value, mean, torch.tensor(0.5))
    repeated = diagonal_gaussian_log_prob_mean(
        value.expand(1, 2, 8, 3), mean.expand(1, 2, 8, 3), torch.tensor(0.5)
    )
    torch.testing.assert_close(one, repeated)
    repeated.sum().backward()
    assert mean.grad is not None
    assert torch.isfinite(mean.grad).all()
    assert mean.grad.abs().sum() > 0


def test_full_chain_objective_propagates_discounted_on_policy_and_bc_gradients():
    current = torch.zeros((1, 3, 5), requires_grad=True)
    bc = torch.full((1, 3, 5), -2.0, requires_grad=True)
    rewards = torch.tensor([[0.1, 0.5, 0.9]])
    result = compute_full_chain_diffgrpo_objective(
        current, bc, rewards, torch.ones_like(rewards, dtype=torch.bool)
    )

    assert result["discount_first"].item() == pytest.approx(0.6 ** 4)
    assert result["discount_last"].item() == pytest.approx(1.0)
    assert result["loss"].ndim == 0
    assert result["reward_mean"].ndim == 0
    assert result["reward_std"].ndim == 0
    assert result["bc_loss"].item() == pytest.approx(2.0)
    result["loss"].backward()

    assert torch.isfinite(current.grad).all()
    assert current.grad[0, 0, -1] > 0
    assert current.grad[0, 2, -1] < 0
    assert current.grad[0, 2, 0].abs() < current.grad[0, 2, -1].abs()
    assert torch.all(bc.grad < 0)
    assert bc.grad.abs().sum().item() == pytest.approx(0.1)


def test_invalid_rewards_are_excluded_and_degenerate_groups_have_zero_rl_gradient():
    current = torch.zeros((2, 3, 5), requires_grad=True)
    bc = torch.zeros_like(current, requires_grad=True)
    rewards = torch.tensor([[0.2, float("nan"), 0.8], [0.5, 0.5, 0.5]])
    valid = torch.tensor([[True, False, True], [True, True, True]])
    result = compute_full_chain_diffgrpo_objective(
        current, bc, rewards, valid, bc_weight=0.0
    )
    result["loss"].backward()

    assert torch.all(current.grad[0, 1] == 0)
    assert current.grad[0, (0, 2), :].abs().sum() > 0
    assert torch.all(current.grad[1] == 0)
    assert result["policy_active_scene_fraction"].item() == pytest.approx(0.5)


@pytest.mark.parametrize(
    "override,match",
    (
        ({"bc_weight": -0.1}, "BC weight"),
        ({"step_discount": 0.0}, "step discount"),
        ({"step_discount": 1.1}, "step discount"),
    ),
)
def test_full_chain_objective_rejects_unregistered_coefficients(override, match):
    log_probs = torch.zeros((1, 2, 5))
    rewards = torch.tensor([[0.2, 0.8]])
    with pytest.raises(ValueError, match=match):
        compute_full_chain_diffgrpo_objective(
            log_probs,
            log_probs,
            rewards,
            torch.ones_like(rewards, dtype=torch.bool),
            **override,
        )


def test_full_chain_objective_rejects_shape_and_nonfinite_log_prob_errors():
    log_probs = torch.zeros((1, 2, 5))
    rewards = torch.tensor([[0.2, 0.8]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    with pytest.raises(ValueError, match="must match"):
        compute_full_chain_diffgrpo_objective(
            log_probs, torch.zeros((1, 2, 4)), rewards, valid
        )
    bad = log_probs.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="must be finite"):
        compute_full_chain_diffgrpo_objective(bad, log_probs, rewards, valid)
