"""Numerical checks for the classification GRPO control objective."""

from dataclasses import replace

import pytest
import torch

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_loss import (
    compute_grpo_objective,
    compute_selector_consistency_kl,
    transfuser_loss,
)


def test_temperature_must_be_positive():
    tensor = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="temperature must be positive"):
        compute_grpo_objective(
            tensor, tensor, tensor, tensor, torch.ones_like(tensor, dtype=torch.bool),
            temperature=0.0,
        )


def test_identical_policies_have_zero_kl():
    logits = torch.tensor([[1.0, -0.5, 0.25]])
    objective = compute_grpo_objective(
        logits,
        logits,
        logits,
        torch.tensor([[0.1, 0.8, 0.4]]),
        torch.ones_like(logits, dtype=torch.bool),
        temperature=1.5,
    )
    torch.testing.assert_close(objective["kl_loss"], torch.zeros(()), atol=1e-7, rtol=0)


def test_selector_consistency_is_reference_to_current_kl():
    current = torch.tensor([[0.5, -0.25, 1.0]], requires_grad=True)
    reference = torch.tensor([[1.0, 0.0, -0.5]], requires_grad=True)
    loss = compute_selector_consistency_kl(current, reference)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss > 0
    assert current.grad is not None and current.grad.abs().sum() > 0
    assert reference.grad is None


def test_identical_selector_logits_have_zero_consistency_kl():
    logits = torch.tensor([[1.0, -0.5, 0.25]])
    loss = compute_selector_consistency_kl(logits, logits)
    torch.testing.assert_close(loss, torch.zeros(()), atol=1e-7, rtol=0)


def test_generation_mode_uses_only_explicit_selector_consistency():
    current_logits = torch.tensor([[1.0, -0.5, 0.25]], requires_grad=True)
    reference_logits = torch.tensor([[0.0, 0.5, -0.25]])
    generation_log_probs = torch.zeros(1, 3, 2, requires_grad=True)
    rewards = torch.tensor([[0.1, 0.8, 0.4]])
    predictions = {
        "final_poses_cls": current_logits,
        "final_old_poses_cls": current_logits.detach(),
        "final_ref_poses_cls": reference_logits,
        "rewards": rewards,
        "reward_valid_mask": torch.ones_like(rewards, dtype=torch.bool),
        "generation_current_log_probs": generation_log_probs,
        "generation_old_log_probs": torch.zeros_like(generation_log_probs),
        "generation_kl": torch.zeros_like(generation_log_probs),
    }

    config = replace(
        TransfuserConfig(),
        grpo_training_mode="generation",
        kl_loss_weight=100.0,
        selection_entropy_weight=100.0,
        selector_consistency_kl_weight=0.0,
    )
    loss = transfuser_loss({}, predictions, config)["loss"]
    loss.backward(retain_graph=True)
    assert current_logits.grad is None or current_logits.grad.abs().sum() == 0

    current_logits.grad = None
    config = replace(config, selector_consistency_kl_weight=0.2)
    loss = transfuser_loss({}, predictions, config)["loss"]
    loss.backward()
    assert current_logits.grad is not None and current_logits.grad.abs().sum() > 0


def test_entropy_bonus_pushes_nonuniform_logits_toward_uniform():
    logits = torch.tensor([[2.0, 0.0]], requires_grad=True)
    config = replace(
        TransfuserConfig(),
        policy_loss_weight=0.0,
        kl_loss_weight=0.0,
        selection_entropy_weight=1.0,
    )
    predictions = {
        "final_poses_cls": logits,
        "final_old_poses_cls": logits.detach(),
        "final_ref_poses_cls": logits.detach(),
        "rewards": torch.ones_like(logits),
        "reward_valid_mask": torch.ones_like(logits, dtype=torch.bool),
    }
    loss = transfuser_loss({}, predictions, config)["loss"]
    loss.backward()

    assert logits.grad[0, 0] > 0
    assert logits.grad[0, 1] < 0
