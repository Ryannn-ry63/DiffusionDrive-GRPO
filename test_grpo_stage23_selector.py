from types import SimpleNamespace

import torch
import pytest

from navsim.agents.diffusiondrive.diffusion_grpo import _gather_selected_set_modes
from navsim.agents.diffusiondrive.transfuser_agent import validate_formal_grpo_config
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_loss import (
    compute_selected_set_diffgrpo_objective,
)

from navsim.agents.diffusiondrive.stage23_trajectory_selector import (
    Stage23TrajectorySelector,
    compute_stage23_selector_loss,
    deterministic_log_bucket,
    gather_oof_predictions,
    member_training_mask,
    select_stage23_trajectory,
)


def _config():
    return SimpleNamespace(
        tf_d_model=32,
        tf_num_head=4,
        lidar_max_y=32.0,
        lidar_max_x=32.0,
        stage23_selector_num_members=5,
        stage23_selector_num_modes=20,
        stage23_selector_dim=32,
    )


def _inputs(batch=2):
    return (
        torch.randn(batch, 20, 8, 3),
        torch.randn(batch, 20),
        torch.randn(batch, 32, 8, 8),
        torch.randn(batch, 6, 32),
        torch.randn(batch, 1, 32),
        torch.randn(batch, 1, 32),
    )


def test_whole_log_bucket_is_deterministic_and_excludes_one_member():
    logs = ("log-a", "log-a", "log-b", "log-c")
    first = deterministic_log_bucket(logs)
    second = deterministic_log_bucket(logs)
    assert torch.equal(first, second)
    assert first[0] == first[1]
    mask = member_training_mask(logs, 5, torch.device("cpu"))
    assert mask.shape == (4, 5)
    assert (~mask).sum(dim=1).tolist() == [1, 1, 1, 1]


def test_oof_gather_uses_held_out_member_only():
    logs = ("log-a", "log-b")
    predictions = torch.arange(2 * 20 * 5).reshape(2, 20, 5)
    result = gather_oof_predictions(predictions, logs)
    buckets = deterministic_log_bucket(logs)
    assert result.shape == (2, 20)
    assert torch.equal(result[0], predictions[0, :, buckets[0]])
    assert torch.equal(result[1], predictions[1, :, buckets[1]])


def test_selector_is_permutation_equivariant_when_mode_identity_moves_too():
    torch.manual_seed(23)
    selector = Stage23TrajectorySelector(_config()).eval()
    trajectories, logits, bev, agents, ego, status = _inputs()
    mode_ids = torch.arange(20).expand(2, -1)
    output = selector(trajectories, logits, bev, (8, 8), agents, ego, status, mode_ids)
    permutation = torch.randperm(20)
    moved = selector(
        trajectories[:, permutation], logits[:, permutation], bev, (8, 8),
        agents, ego, status, mode_ids[:, permutation],
    )
    assert torch.allclose(
        output["score_predictions"][:, permutation],
        moved["score_predictions"], atol=1e-5, rtol=1e-5,
    )
    assert torch.allclose(
        output["component_predictions"][:, permutation],
        moved["component_predictions"], atol=1e-5, rtol=1e-5,
    )


def test_selector_depends_on_trajectory():
    torch.manual_seed(24)
    selector = Stage23TrajectorySelector(_config()).eval()
    trajectories, logits, bev, agents, ego, status = _inputs(batch=1)
    original = selector(trajectories, logits, bev, (8, 8), agents, ego, status)
    changed = trajectories.clone()
    changed[:, 7, :, 0] += torch.linspace(0, 15, 8)
    moved = selector(changed, logits, bev, (8, 8), agents, ego, status)
    assert not torch.allclose(
        original["score_predictions"][:, 7], moved["score_predictions"][:, 7]
    )


def test_all_pair_component_loss_is_finite_and_backpropagates():
    torch.manual_seed(25)
    logits = torch.randn(2, 20, 5, 6, requires_grad=True)
    components = logits.sigmoid()
    score_logits = torch.randn(2, 20, 5, requires_grad=True)
    scores = score_logits.sigmoid()
    targets = torch.rand(2, 20, 6)
    targets[..., [0, 1, 3]] = (
        targets[..., [0, 1, 3]] > 0.1
    ).float()
    rewards = torch.rand(2, 20)
    output = compute_stage23_selector_loss({
        "stage23_component_logits": logits,
        "stage23_component_predictions": components,
        "stage23_score_predictions": scores,
        "component_scores": targets,
        "raw_rewards": rewards,
        "reward_valid_mask": torch.ones(2, 20, dtype=torch.bool),
        "stage23_member_training_mask": member_training_mask(
            ("a", "b"), 5, torch.device("cpu")
        ),
    })
    assert torch.isfinite(output["loss"])
    assert output["stage23_active_pair_count"] > 0
    output["loss"].backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert score_logits.grad is not None and score_logits.grad.abs().sum() > 0


def test_all20_conservative_selector_uses_best_eligible_challenger():
    reference = torch.arange(20, dtype=torch.float32).flip(0).unsqueeze(0)
    components = torch.ones(1, 20, 5, 6)
    scores = torch.full((1, 20, 5), 0.5)
    scores[:, 7] = 0.56
    scores[:, 9] = 0.60
    components[:, 9, :, 0] = 0.2  # highest value is unsafe
    selected, diagnostic = select_stage23_trajectory(
        reference, components, scores, residual_margin=0.01,
        safety_threshold=0.9, confidence_z=1.96,
    )
    assert selected.item() == 7
    assert diagnostic["switch"].item()


def test_selector_falls_back_without_positive_calibrated_lcb():
    reference = torch.randn(2, 20)
    components = torch.ones(2, 20, 5, 6)
    scores = torch.full((2, 20, 5), 0.5)
    selected, diagnostic = select_stage23_trajectory(
        reference, components, scores, residual_margin=0.01,
    )
    assert torch.equal(selected, reference.argmax(dim=-1))
    assert not diagnostic["switch"].any()


def test_selected_set_gather_replays_exactly_one_mode_per_set():
    values = torch.arange(2 * 8 * 20 * 3).reshape(2, 160, 3)
    selected = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [19] * 8])
    result = _gather_selected_set_modes(values, selected, 20)
    assert result.shape == (2, 8, 3)
    for batch in range(2):
        for group in range(8):
            assert torch.equal(
                result[batch, group],
                values[batch, group * 20 + selected[batch, group]],
            )


def _objective_inputs():
    current_log_probs = torch.zeros(1, 8, 5, requires_grad=True)
    bc_log_probs = torch.zeros(1, 8, 5, requires_grad=True)
    exact_kl = torch.ones(1, 8, 5, requires_grad=True)
    base = torch.tensor([[0.50, 0.80, 0.60, 0.70, 0.55, 0.65, 0.45, 0.85]])
    rewards = base + torch.tensor([[0.02, -0.003, -0.02, 0.03, 0.0, 0.01, -0.01, -0.001]])
    valid = torch.ones(1, 8, dtype=torch.bool)
    components = torch.ones(1, 8, 6)
    base_components = torch.ones(1, 8, 6)
    components[0, 3, 0] = 0.9
    return (
        current_log_probs, bc_log_probs, exact_kl, rewards, valid,
        components, base, valid.clone(), base_components,
    )


def test_selected_set_objective_applies_all_guards_and_gradients():
    inputs = _objective_inputs()
    result = compute_selected_set_diffgrpo_objective(*inputs)
    advantages = result["advantages"]
    assert advantages[0, 2] == -2  # candidate-base < -0.01
    assert advantages[0, 1] == -2  # mature base and delta < -0.002
    assert advantages[0, 3] == -2  # safety regression dominates positive reward
    assert result["safety_override_fraction"] > 0
    assert result["mature_negative_fraction"] > 0
    result["loss"].backward()
    assert inputs[0].grad.abs().sum() > 0
    assert inputs[1].grad.abs().sum() > 0
    assert inputs[2].grad.abs().sum() > 0


def test_selected_set_exact_kl_uses_point1_and_mature_negative_point5():
    inputs = list(_objective_inputs())
    inputs[2] = torch.ones_like(inputs[2], requires_grad=True)
    result = compute_selected_set_diffgrpo_objective(*inputs)
    # mature-negative modes: base >= .75 and delta < -.002 -> only mode 1.
    expected = (7 * 0.1 + 0.5) / 8
    assert result["reference_kl_loss"].item() == pytest.approx(expected)


def test_stage23_formal_config_is_locked():
    config = TransfuserConfig()
    config.grpo_training_mode = "diffgrpo_selected_set"
    config.generation_policy_algorithm = "diffgrpo_selected_set"
    config.grpo_decoder_gradient_scope = "all_layers"
    config.grpo_decoder_layer0_lr_mult = 0.1
    config.inference_selector_source = "trajectory_oof"
    config.policy_loss_weight = 0.0
    config.kl_loss_weight = 0.0
    config.selector_generation_kl_weight = 0.0
    config.generation_kl_loss_weight = 0.0
    config.diffusion_truncation_timestep = 32
    config.diffusion_roll_timesteps = (32, 24, 16, 8, 0)
    validate_formal_grpo_config(config)
    config.diffgrpo_group_size = 7
    with pytest.raises(ValueError, match="group_size"):
        validate_formal_grpo_config(config)
