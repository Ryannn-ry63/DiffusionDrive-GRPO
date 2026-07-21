from types import SimpleNamespace

import pytest
import torch

from navsim.agents.diffusiondrive.trajectory_value_selector import (
    TrajectoryValueSelector,
    compose_pdm_score,
    compute_value_selector_loss,
    deterministic_bootstrap_mask,
    select_conservative_top2,
)


def config():
    return SimpleNamespace(
        tf_d_model=256,
        tf_num_head=8,
        lidar_max_y=32.0,
        lidar_max_x=32.0,
    )


def test_compose_pdm_score_matches_registered_formula():
    components = torch.tensor([[1.0, 1.0, 0.8, 0.6, 0.4, 0.2]])
    expected = (10.0 * 0.8 + 5.0 * 0.6 + 2.0 * 0.4) / 17.0
    assert compose_pdm_score(components).item() == pytest.approx(expected)
    components[0, 0] = 0.5
    assert compose_pdm_score(components).item() == pytest.approx(expected * 0.5)


def test_bootstrap_mask_is_reproducible_and_never_empty():
    left = deterministic_bootstrap_mask(("a", "b", "c"), 3, torch.device("cpu"))
    right = deterministic_bootstrap_mask(("a", "b", "c"), 3, torch.device("cpu"))
    assert torch.equal(left, right)
    assert left.shape == (3, 3)
    assert left.any(dim=-1).all()


def test_value_selector_reencodes_final_trajectory():
    torch.manual_seed(4)
    selector = TrajectoryValueSelector(config(), num_heads=3).eval()
    trajectories = torch.zeros(2, 4, 8, 3)
    bev = torch.randn(2, 256, 8, 8)
    agents = torch.randn(2, 5, 256)
    ego = torch.randn(2, 1, 256)
    status = torch.randn(2, 1, 256)
    first = selector(trajectories, bev, (8, 8), agents, ego, status)
    moved = trajectories.clone()
    moved[:, 0, :, 0] = torch.linspace(0.0, 8.0, 8)
    second = selector(moved, bev, (8, 8), agents, ego, status)
    assert first["component_predictions"].shape == (2, 4, 3, 6)
    assert first["score_predictions"].shape == (2, 4, 3)
    assert not torch.allclose(
        first["score_predictions"][:, 0], second["score_predictions"][:, 0]
    )


def test_conservative_top2_switches_only_for_safe_challenger():
    logits = torch.tensor([[5.0, 4.0, 0.0], [5.0, 4.0, 0.0]])
    component_mean = torch.ones(2, 3, 6)
    component_std = torch.zeros_like(component_mean)
    score_mean = torch.tensor([[0.6, 0.8, 0.9], [0.6, 0.8, 0.9]])
    component_mean[1, 1, 0] = 0.2
    selected, diagnostics = select_conservative_top2(
        logits, component_mean, component_std, score_mean, margin=0.1
    )
    assert selected.tolist() == [1, 0]
    assert diagnostics["switch"].tolist() == [True, False]
    assert diagnostics["fallback_mode"].tolist() == [0, 0]
    assert diagnostics["challenger_mode"].tolist() == [1, 1]


def test_conservative_top2_rejects_uncalibrated_margin():
    logits = torch.tensor([[2.0, 1.0]])
    components = torch.ones(1, 2, 6)
    scores = torch.tensor([[0.5, 0.6]])
    with pytest.raises(ValueError, match="margin"):
        select_conservative_top2(
            logits, components, torch.zeros_like(components), scores, margin=-1.0
        )


def test_value_selector_loss_is_finite_and_backpropagates():
    torch.manual_seed(7)
    batch, modes, heads = 2, 40, 3
    component_logits = torch.randn(batch, modes, heads, 6, requires_grad=True)
    component_predictions = component_logits.sigmoid()
    score_predictions = compose_pdm_score(component_predictions)
    targets = torch.rand(batch, modes, 6)
    targets[..., :2] = (targets[..., :2] > 0.2).float()
    rewards = compose_pdm_score(targets)
    reference_logits = torch.randn(batch, modes)
    output = compute_value_selector_loss(
        {
            "value_component_predictions": component_predictions,
            "value_score_predictions": score_predictions,
            "component_scores": targets,
            "raw_rewards": rewards,
            "reward_valid_mask": torch.ones(batch, modes, dtype=torch.bool),
            "value_bootstrap_mask": torch.ones(batch, heads, dtype=torch.bool),
            "value_reference_logits": reference_logits,
            "value_group_size": 20,
        }
    )
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert component_logits.grad is not None
    assert component_logits.grad.abs().sum() > 0


def test_value_selector_loss_masks_invalid_candidates():
    components = torch.full((1, 40, 3, 6), 0.5, requires_grad=True)
    scores = compose_pdm_score(components)
    targets = torch.ones(1, 40, 6)
    rewards = torch.ones(1, 40)
    valid = torch.zeros(1, 40, dtype=torch.bool)
    output = compute_value_selector_loss(
        {
            "value_component_predictions": components,
            "value_score_predictions": scores,
            "component_scores": targets,
            "raw_rewards": rewards,
            "reward_valid_mask": valid,
            "value_bootstrap_mask": torch.ones(1, 3, dtype=torch.bool),
            "value_reference_logits": torch.randn(1, 40),
            "value_group_size": 20,
        }
    )
    assert output["loss"].item() == 0.0
