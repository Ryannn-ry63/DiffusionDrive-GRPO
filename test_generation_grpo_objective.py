"""Checks for the denoising-action GRPO objective."""

import torch

from navsim.agents.diffusiondrive.transfuser_loss import (
    compute_generation_grpo_objective,
)


def test_generation_objective_backpropagates_to_log_probs():
    current = torch.zeros(1, 3, 2, requires_grad=True)
    old = torch.zeros_like(current)
    generation_kl = torch.zeros_like(current)
    rewards = torch.tensor([[0.1, 0.5, 0.9]])
    valid = torch.ones_like(rewards, dtype=torch.bool)

    objective = compute_generation_grpo_objective(
        current, old, generation_kl, rewards, valid
    )
    objective["policy_loss"].backward()

    assert torch.isfinite(objective["policy_loss"])
    assert current.grad is not None
    assert current.grad.abs().sum() > 0


def test_generation_objective_rejects_wrong_shape():
    current = torch.zeros(1, 3)
    rewards = torch.zeros(1, 3)
    valid = torch.ones_like(rewards, dtype=torch.bool)
    try:
        compute_generation_grpo_objective(current, current, current, rewards, valid)
    except ValueError as error:
        assert "shape [batch, mode, step]" in str(error)
    else:
        raise AssertionError("invalid generation log-prob shape was accepted")
