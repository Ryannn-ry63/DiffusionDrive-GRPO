"""Numerical checks for the classification GRPO control objective."""

from dataclasses import replace

import pytest
import torch

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_loss import (
    compute_grpo_objective,
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
