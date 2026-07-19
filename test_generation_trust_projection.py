"""Focused tests for Phase-5 hard functional trust projection."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from navsim.agents.diffusiondrive.diffusion_grpo import (
    collect_generation_trace,
    diagonal_gaussian_kl_same_std,
    diagonal_gaussian_log_prob,
    load_generation_trust_calibration,
    normalized_rms_displacement,
    project_reference_mean_ball,
    summarize_trust_projection,
)


def make_means(scale=1.0):
    reference = torch.zeros(2, 3, 4, 2)
    current = torch.full_like(reference, scale)
    return current, reference


def test_mean_ball_keeps_inside_and_projects_outside_to_boundary():
    inside, reference = make_means(0.2)
    inside_projected, inside_diag = project_reference_mean_ball(
        inside, reference, std=1.0, radius=0.5
    )
    torch.testing.assert_close(inside_projected, inside)
    assert not inside_diag["projected"].any()

    outside, reference = make_means(2.0)
    outside_projected, outside_diag = project_reference_mean_ball(
        outside, reference, std=1.0, radius=0.5
    )
    assert outside_diag["projected"].all()
    torch.testing.assert_close(
        outside_diag["post_distance"],
        torch.full((2, 3), 0.5),
        atol=2e-7,
        rtol=0,
    )
    assert float(outside_diag["post_distance"].max()) <= 0.5 + 1e-6


def test_step_mapping_and_anchor_order_are_preserved():
    pre = torch.tensor(
        [[[0.1, 1.1], [0.2, 1.2], [0.3, 1.3]]]
    )
    result = summarize_trust_projection(
        pre,
        pre / 2,
        torch.ones_like(pre),
        torch.zeros_like(pre, dtype=torch.bool),
        torch.ones_like(pre, dtype=torch.bool),
    )
    torch.testing.assert_close(
        result["generation_trust_transition_pre_distance_mean"],
        torch.tensor(0.2),
    )
    torch.testing.assert_close(
        result["generation_trust_final_pre_distance_mean"],
        torch.tensor(1.2),
    )
    torch.testing.assert_close(
        result["generation_trust_transition_pre_distance_max"],
        torch.tensor(0.3),
    )


def test_reference_radius_and_action_targets_are_detached(tmp_path):
    current = torch.ones(1, 2, 3, 2, requires_grad=True)
    reference = torch.zeros_like(current, requires_grad=True)
    projected, _ = project_reference_mean_ball(
        current, reference, std=1.0, radius=0.5
    )
    action_target = (projected.detach() + torch.randn_like(projected)).detach()
    diagonal_gaussian_log_prob(action_target, projected, torch.tensor(1.0)).sum().backward()
    assert current.grad is not None
    assert reference.grad is None
    assert not action_target.requires_grad

    artifact = {
        "schema_version": 1,
        "formula_version": "reference_mean_ball_v1",
        "mode": "reference_mean_ball",
        "seed": 20260719,
        "percentile": 0.99,
        "steps": {
            name: {
                "radius": 0.5,
                "sigma": 0.1,
                "count": 2,
                "quantiles": {"p99": 0.5},
            }
            for name in ("transition", "final")
        },
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    path.chmod(0o444)
    loaded = load_generation_trust_calibration(str(path))
    assert isinstance(loaded["steps"]["transition"]["radius"], float)


def test_current_equals_old_has_projected_ratio_one():
    current, reference = make_means(1.0)
    old = current.detach().clone()
    current_projected, _ = project_reference_mean_ball(
        current, reference, std=0.5, radius=0.7
    )
    old_projected, _ = project_reference_mean_ball(
        old, reference, std=0.5, radius=0.7
    )
    action = old_projected + 0.5 * torch.randn_like(old_projected)
    current_log_prob = diagonal_gaussian_log_prob(
        action, current_projected, torch.tensor(0.5)
    )
    old_log_prob = diagonal_gaussian_log_prob(
        action, old_projected, torch.tensor(0.5)
    )
    torch.testing.assert_close((current_log_prob - old_log_prob).exp(), torch.ones(2, 3))


def test_projected_kl_has_gradient_when_policy_advantage_is_zero():
    current = torch.full((1, 2, 3, 2), 0.1, requires_grad=True)
    reference = torch.zeros_like(current)
    projected, _ = project_reference_mean_ball(
        current, reference, std=1.0, radius=0.5
    )
    kl = diagonal_gaussian_kl_same_std(projected, reference, torch.tensor(1.0)).mean()
    kl.backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    assert current.grad.abs().sum() > 0


def test_exact_reference_projection_backward_is_finite_identity():
    reference = torch.randn(2, 3, 4, 2)
    current = reference.detach().clone().requires_grad_(True)
    projected, diagnostics = project_reference_mean_ball(
        current, reference, std=0.1, radius=0.5
    )
    projected.sum().backward()
    torch.testing.assert_close(projected, reference)
    torch.testing.assert_close(
        diagnostics["pre_distance"], torch.zeros(2, 3)
    )
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    torch.testing.assert_close(current.grad, torch.ones_like(current))


def test_old_sample_and_same_noise_action_displacements_are_bounded():
    old, reference = make_means(1.5)
    projected_old, diagnostics = project_reference_mean_ball(
        old, reference, std=0.25, radius=0.8
    )
    noise = torch.randn_like(projected_old)
    behavior_action = (projected_old + 0.25 * noise).detach()
    base_same_noise_action = (reference + 0.25 * noise).detach()
    displacement = normalized_rms_displacement(
        behavior_action, base_same_noise_action, std=0.25
    )
    torch.testing.assert_close(displacement, diagnostics["post_distance"])
    assert float(displacement.max()) <= 0.8 + 1e-6
    assert not behavior_action.requires_grad


def test_training_and_inference_use_identical_projection_function():
    current, reference = make_means(1.2)
    train_projected, train_diag = project_reference_mean_ball(
        current, reference, std=0.3, radius=0.4
    )
    inference_projected, inference_diag = project_reference_mean_ball(
        current, reference, std=0.3, radius=0.4
    )
    torch.testing.assert_close(train_projected, inference_projected)
    for key in train_diag:
        torch.testing.assert_close(train_diag[key], inference_diag[key])


def test_invalid_reference_and_failed_post_check_raise():
    current, reference = make_means(1.0)
    with pytest.raises(ValueError, match="identical shapes"):
        project_reference_mean_ball(
            current, reference[:, :2], std=1.0, radius=0.5
        )
    invalid_reference = reference.clone()
    invalid_reference[0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        project_reference_mean_ball(
            current, invalid_reference, std=1.0, radius=0.5
        )
    with pytest.raises(RuntimeError, match="post-check failed"):
        project_reference_mean_ball(
            current,
            reference,
            std=1.0,
            radius=0.5,
            post_tolerance=-1.0,
        )

    head = SimpleNamespace(
        _generation_trust_projection_mode="reference_mean_ball",
        ref_policy=None,
        old_policy=object(),
    )
    with pytest.raises(RuntimeError, match="frozen base"):
        collect_generation_trace(
            head, None, None, None, None, None, None, None
        )


def test_legacy_mode_delegates_without_touching_projection_path():
    sentinel = {"legacy": torch.tensor(1.0)}
    head = SimpleNamespace(_generation_trust_projection_mode="none")
    with patch(
        "navsim.agents.diffusiondrive.diffusion_grpo._collect_generation_trace_legacy",
        return_value=sentinel,
    ) as legacy:
        result = collect_generation_trace(
            head, "initial", "ego", "agents", "bev", "shape", "status", "image"
        )
    assert result is sentinel
    legacy.assert_called_once_with(
        head, "initial", "ego", "agents", "bev", "shape", "status", "image"
    )
