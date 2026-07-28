import hashlib
from types import SimpleNamespace

import pytest
import torch

from navsim.agents.diffusiondrive import stage34_trace
from navsim.agents.diffusiondrive.stage34_contract import (
    STAGE34_OBJECTIVE_REVISION,
    STAGE34_PLAN_SHA256,
)
from navsim.agents.diffusiondrive.stage34_loss_bridge import (
    compute_stage34_transfuser_loss,
)
from navsim.agents.diffusiondrive.stage34_mode_aligned_frontier import (
    compute_mode_aligned_frontier_objective,
)


def _inputs(batch=2, delta=None, buckets=None):
    base = torch.linspace(0.50, 0.88, 20).repeat(batch, 1)
    if delta is None:
        delta = torch.zeros(batch, 20)
    if buckets is None:
        buckets = torch.arange(batch, dtype=torch.long) % 4
    valid = torch.ones(batch, 20, dtype=torch.bool)
    components = torch.ones(batch, 20, 6)
    return {
        "current_log_probs": torch.zeros(batch, 20, 5, requires_grad=True),
        "bc_log_probs": torch.zeros(batch, 20, 5, requires_grad=True),
        "reference_mean_kl": torch.full(
            (batch, 20, 5), 0.01, requires_grad=True
        ),
        "rewards": base + delta,
        "valid_mask": valid,
        "component_scores": components.clone(),
        "base_rewards": base,
        "base_valid_mask": valid.clone(),
        "base_component_scores": components,
        "current_selected_modes": torch.zeros(batch, dtype=torch.long),
        "public_selected_modes": torch.ones(batch, dtype=torch.long),
        "current_eligible_mask": valid.clone(),
        "public_eligible_mask": valid.clone(),
        "scene_buckets": buckets,
    }


def test_stage34_same_mode_credit_and_public_frontier_negative_protection():
    delta = torch.zeros(1, 20)
    delta[0, 18:] = -0.001
    inputs = _inputs(batch=1, delta=delta, buckets=torch.tensor([1]))
    inputs["current_eligible_mask"].zero_()
    result = compute_mode_aligned_frontier_objective(**inputs)
    advantages = result["advantages"][0]
    assert advantages[18].item() < 0
    assert advantages[19].item() < advantages[18].item()
    assert result["candidate_level_trace"].item() == 1.0
    assert result["same_mode_delta_mean"].item() == pytest.approx(-0.0001, abs=1e-8)


def test_stage34_mature_positive_is_attenuated_but_negative_is_not():
    delta = torch.zeros(2, 20)
    delta[:, 18] = 0.004
    delta[:, 19] = -0.004
    inputs = _inputs(batch=2, delta=delta, buckets=torch.tensor([0, 3]))
    inputs["current_eligible_mask"].zero_()
    result = compute_mode_aligned_frontier_objective(**inputs)
    advantages = result["advantages"]
    assert advantages[0, 18].item() > 0
    assert advantages[1, 18].item() == pytest.approx(
        advantages[0, 18].item() * 0.25, abs=2e-6
    )
    assert advantages[1, 19].item() == pytest.approx(
        advantages[0, 19].item()
    )


def test_stage34_safety_regression_overrides_positive_credit():
    inputs = _inputs(
        batch=1,
        delta=torch.full((1, 20), 0.005),
        buckets=torch.tensor([0]),
    )
    inputs["component_scores"][0, 19, 0] = 0.99
    result = compute_mode_aligned_frontier_objective(**inputs)
    assert result["advantages"][0, 19].item() == -2.0
    assert result["safety_override_fraction"].item() > 0


def test_stage34_bc_and_kl_are_mean_normalized_over_all_valid_modes():
    inputs = _inputs(batch=2)
    inputs["bc_log_probs"] = torch.full(
        (2, 20, 5), -2.0, requires_grad=True
    )
    inputs["reference_mean_kl"] = torch.full(
        (2, 20, 5), 0.5, requires_grad=True
    )
    result = compute_mode_aligned_frontier_objective(
        **inputs, step_discount=1.0
    )
    assert torch.allclose(result["policy_loss"], torch.tensor(0.0))
    assert torch.allclose(result["bc_loss"], torch.tensor(0.2))
    assert torch.allclose(result["reference_kl_loss"], torch.tensor(0.05))
    assert torch.allclose(result["loss"], torch.tensor(0.25))
    result["loss"].backward()
    assert inputs["bc_log_probs"].grad.abs().sum().item() > 0
    assert inputs["reference_mean_kl"].grad.abs().sum().item() > 0


def test_stage34_loss_bridge_requires_exact_twenty_by_twenty_trace():
    values = _inputs(batch=1, delta=torch.full((1, 20), 0.002))
    predictions = {
        "diffgrpo_current_log_probs": values.pop("current_log_probs"),
        "diffgrpo_bc_log_probs": values.pop("bc_log_probs"),
        "diffgrpo_reference_mean_kl": values.pop("reference_mean_kl"),
        "raw_rewards": values.pop("rewards"),
        "reward_valid_mask": values.pop("valid_mask"),
        "component_scores": values.pop("component_scores"),
        "diffgrpo_base_rewards": values.pop("base_rewards"),
        "diffgrpo_base_valid_mask": values.pop("base_valid_mask"),
        "diffgrpo_base_component_scores": values.pop("base_component_scores"),
        "stage34_current_selected_modes": values.pop("current_selected_modes"),
        "stage34_public_selected_modes": values.pop("public_selected_modes"),
        "stage34_current_eligible_mask": values.pop("current_eligible_mask"),
        "stage34_public_eligible_mask": values.pop("public_eligible_mask"),
        "stage34_scene_buckets": values.pop("scene_buckets"),
        "stage34_sampled_chain_count": torch.tensor(20),
        "stage34_replayed_chain_count": torch.tensor(20),
        "stage34_mode_index_alignment": torch.tensor(1.0),
        "stage34_candidate_level_trace": torch.tensor(1.0),
        "diffgrpo_group_size": torch.tensor(20),
    }
    result = compute_stage34_transfuser_loss(predictions, SimpleNamespace())
    assert torch.isfinite(result["loss"])
    assert result["stage34_candidate_level_trace"].item() == 1.0
    predictions["stage34_replayed_chain_count"] = torch.tensor(19)
    with pytest.raises(ValueError, match="20 sampled and 20 replayed"):
        compute_stage34_transfuser_loss(predictions, SimpleNamespace())


def test_stage34_trace_uses_frozen_selector_on_both_aligned_banks(monkeypatch):
    call = {"count": 0}

    def fake_select(*args, **kwargs):
        selected = torch.tensor([call["count"]])
        call["count"] += 1
        diagnostics = {
            "eligible": torch.ones(1, 20, dtype=torch.bool),
            "risk_ucb": torch.zeros(1, 20),
            "delta_lcb": torch.zeros(1, 20),
            "ood_distance": torch.zeros(1, 20),
            "switch": torch.tensor([True]),
        }
        return selected, diagnostics

    monkeypatch.setattr(stage34_trace, "_select_stage31_deployed_modes", fake_select)
    trace = {
        "trajectories": torch.zeros(1, 20, 8, 3),
        "base_trajectories": torch.ones(1, 20, 8, 3),
        "current_cls": torch.zeros(1, 20),
        "reference_cls": torch.zeros(1, 20),
    }
    diagnostics = stage34_trace.build_stage34_trace_diagnostics(
        object(), trace, *(torch.zeros(1, 1) for _ in range(5))
    )
    assert diagnostics["stage34_current_selected_modes"].item() == 0
    assert diagnostics["stage34_public_selected_modes"].item() == 1
    assert diagnostics["stage34_selector_mode_disagreement"].item() == 1.0
    assert diagnostics["stage34_candidate_level_trace"].item() == 1.0


def test_stage34_plan_and_objective_are_immutable():
    plan = open(
        "GRPO_STAGE34_MODE_ALIGNED_FRONTIER_PLAN_20260727.md", "rb"
    ).read()
    assert hashlib.sha256(plan).hexdigest() == STAGE34_PLAN_SHA256
    assert STAGE34_OBJECTIVE_REVISION == "mode_aligned_frontier_v1"
