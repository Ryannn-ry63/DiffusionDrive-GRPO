"""Focused correctness tests for the Stage-11 frozen selector route."""

from dataclasses import replace

import pytest
import torch

from navsim.agents.diffusiondrive.diffusion_grpo import (
    select_inference_mode,
    validate_inference_selector_source,
)
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_loss import (
    compute_generation_grpo_objective,
)
from scripts.evaluation.check_grpo_stage11_gate import check_base_equivalence


def test_inference_selector_source_defaults_to_current_and_validates():
    assert TransfuserConfig().inference_selector_source == "current"
    assert validate_inference_selector_source("current") == "current"
    assert validate_inference_selector_source("reference") == "reference"
    with pytest.raises(ValueError, match="current.*reference"):
        validate_inference_selector_source("frozen")


def test_reference_argmax_selects_a_current_generator_candidate():
    current_logits = torch.tensor([[5.0, 1.0, 0.0]], requires_grad=True)
    reference_logits = torch.tensor([[0.0, 2.0, 3.0]], requires_grad=True)
    candidates = torch.arange(18.0).reshape(1, 3, 2, 3)

    mode, selector_logits = select_inference_mode(
        current_logits, reference_logits, "reference"
    )
    selected = candidates[torch.arange(1), mode]

    assert mode.tolist() == [2]
    torch.testing.assert_close(selected, candidates[:, 2])
    assert selector_logits.requires_grad is False


def test_current_source_does_not_require_reference_logits():
    logits = torch.tensor([[0.0, 4.0]])
    mode, selected_logits = select_inference_mode(logits, None, "current")
    assert mode.tolist() == [1]
    assert selected_logits is logits
    with pytest.raises(RuntimeError, match="requires frozen reference logits"):
        select_inference_mode(logits, None, "reference")


def test_base_equal_logits_are_exactly_selector_equivalent():
    logits = torch.tensor([[0.2, 0.8], [3.0, -1.0]])
    candidates = torch.randn(2, 2, 8, 3)
    current_mode, _ = select_inference_mode(logits, logits.clone(), "current")
    reference_mode, _ = select_inference_mode(
        logits, logits.clone(), "reference"
    )
    batch = torch.arange(2)

    assert torch.equal(current_mode, reference_mode)
    torch.testing.assert_close(
        candidates[batch, current_mode],
        candidates[batch, reference_mode],
        rtol=0.0,
        atol=0.0,
    )


def generation_loss_and_gradient(config):
    assert config.inference_selector_source in {"current", "reference"}
    current = torch.zeros(1, 3, 2, requires_grad=True)
    objective = compute_generation_grpo_objective(
        current_log_probs=current,
        old_log_probs=torch.zeros_like(current),
        generation_kl=torch.zeros_like(current),
        rewards=torch.tensor([[0.1, 0.5, 0.9]]),
        valid_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    objective["policy_loss"].backward()
    return objective["policy_loss"].detach(), current.grad.detach().clone()


def test_inference_source_does_not_change_generation_loss_or_gradient():
    base = TransfuserConfig()
    current_loss, current_grad = generation_loss_and_gradient(
        replace(base, inference_selector_source="current")
    )
    reference_loss, reference_grad = generation_loss_and_gradient(
        replace(base, inference_selector_source="reference")
    )

    torch.testing.assert_close(current_loss, reference_loss, rtol=0.0, atol=0.0)
    torch.testing.assert_close(current_grad, reference_grad, rtol=0.0, atol=0.0)
    assert current_grad.abs().sum() > 0


def selector_artifact(source, mode=1, reward=0.8, trajectory=None):
    trajectory = trajectory or [[1.0, 2.0, 0.1], [2.0, 3.0, 0.2]]
    return {
        "summary": {
            "num_tokens": 1,
            "token_set_sha256": "same",
            "selector_logits_source": source,
        },
        "records": [
            {
                "token": "token",
                "selected_mode": mode,
                "selected_reward": reward,
                "selected_trajectory": trajectory,
            }
        ],
    }


def test_base_equivalence_gate_accepts_exact_selector_match():
    result = check_base_equivalence(
        selector_artifact("current"), selector_artifact("reference")
    )
    assert result["passed"] is True
    assert result["selected_mode_agreement"] == 1.0
    assert result["selected_trajectory_max_error"] == 0.0
    assert result["selected_pdms_difference"] == 0.0


def test_base_equivalence_gate_rejects_mode_trajectory_and_reward_drift():
    result = check_base_equivalence(
        selector_artifact("current"),
        selector_artifact(
            "reference",
            mode=0,
            reward=0.7,
            trajectory=[[1.0, 2.0, 0.1], [2.0, 3.001, 0.2]],
        ),
    )
    assert result["passed"] is False
    assert len(result["failures"]) == 3
