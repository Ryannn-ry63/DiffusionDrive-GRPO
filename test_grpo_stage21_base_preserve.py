import json

import pytest
import torch

from navsim.agents.diffusiondrive.transfuser_agent import (
    build_stage10_decoder_param_groups,
    validate_formal_grpo_config,
)
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_loss import (
    compute_base_preserve_diffgrpo_objective,
)
from navsim.agents.diffusiondrive.transfuser_model_v2 import (
    STAGE19_BASE_SHA256,
    STAGE19_FULL_SCHEDULE,
    load_base_preserve_manifest,
)
from scripts.evaluation.build_grpo_stage21_manifests import (
    bounded_unit_mean,
    build_folds,
)


def make_inputs(batch=2):
    current = torch.zeros(batch, 8, 3, requires_grad=True)
    bc = torch.zeros(batch, 1, 3, requires_grad=True)
    rewards = torch.tensor(
        [[0.70, 0.60, 0.505, 0.40, 0.45, 0.50, 0.52, 0.48], [0.60] * 8]
    )[:batch]
    valid = torch.ones_like(rewards, dtype=torch.bool)
    components = torch.ones(batch, 8, 6)
    base_rewards = torch.tensor([0.50, 0.50])[:batch]
    base_safety = torch.full((batch, 3), 0.90)
    scene_weights = torch.tensor([0.5, 2.0])[:batch]
    bc_weights = torch.tensor([0.1, 0.3])[:batch]
    return current, bc, rewards, valid, components, base_rewards, base_safety, scene_weights, bc_weights


def call_objective(*inputs):
    return compute_base_preserve_diffgrpo_objective(
        *inputs,
        margin=0.01,
        scale=0.10,
        advantage_clip=2.0,
        safety_tolerance=1e-6,
        step_discount=0.6,
    )


def test_stage21_advantage_signs_deadzone_and_zero_variance():
    result = call_objective(*make_inputs())
    advantages = result["advantages"]
    assert advantages[0, 0] > 0
    assert advantages[0, 3] < 0
    assert advantages[0, 2] == 0
    assert advantages[0, 5] == 0
    assert torch.all(advantages[1] > 0), "absolute base gain remains active at zero group variance"
    assert torch.isfinite(result["loss"])


def test_stage21_safety_override_forces_negative_clip():
    inputs = list(make_inputs())
    inputs[4][0, 0, 0] = 0.80
    result = call_objective(*inputs)
    assert result["advantages"][0, 0].item() == -2.0
    assert result["safety_override_fraction"] > 0


def test_stage21_invalid_group_has_no_policy_gradient_but_bc_is_active():
    inputs = list(make_inputs(batch=1))
    inputs[3][:] = False
    inputs[3][0, 0] = True
    result = call_objective(*inputs)
    result["loss"].backward()
    assert torch.count_nonzero(inputs[0].grad) == 0
    assert torch.count_nonzero(inputs[1].grad) > 0
    assert torch.all(result["advantages"] == 0)


def test_stage21_dynamic_bc_gradient_ratio():
    inputs = list(make_inputs())
    inputs[3][:] = False
    result = call_objective(*inputs)
    result["loss"].backward()
    first = inputs[1].grad[0].abs().mean()
    second = inputs[1].grad[1].abs().mean()
    assert torch.isclose(second / first, torch.tensor(3.0), atol=1e-6)


def test_stage21_rejects_wrong_group_and_nonfinite_metadata():
    inputs = list(make_inputs())
    with pytest.raises(ValueError, match="G=8"):
        call_objective(inputs[0][:, :7], inputs[1], inputs[2][:, :7], inputs[3][:, :7], inputs[4][:, :7], *inputs[5:])
    inputs = list(make_inputs())
    inputs[5][0] = float("nan")
    with pytest.raises(FloatingPointError):
        call_objective(*inputs)


def test_stage21_manifest_loader_is_fail_closed(tmp_path):
    payload = {
        "summary": {
            "base_checkpoint_sha256": STAGE19_BASE_SHA256,
            "schedule": STAGE19_FULL_SCHEDULE,
            "group_definition": "same_token_same_selected_anchor",
            "group_size": 8,
            "stage21_objective": "base_anchored_log_robust_v1",
            "base_noise_namespaces": [-1, 20260728],
        },
        "records": [{
            "token": "abc", "selected_mode": 3, "base_reward_mean": 0.8,
            "base_collision_mean": 1.0, "base_drivable_mean": 0.9,
            "base_ttc_mean": 1.0, "scene_weight": 1.2, "bc_weight": 0.14,
        }],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    loaded = load_base_preserve_manifest(str(path), 8)
    assert loaded["abc"]["selected_mode"] == 3
    payload["summary"]["base_noise_namespaces"] = [-1, 7]
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="base_noise_namespaces"):
        load_base_preserve_manifest(str(path), 8)


def test_stage21_layer0_lr_group_and_formal_constants():
    layer0 = torch.nn.Parameter(torch.ones(()))
    layer1 = torch.nn.Parameter(torch.ones(()))
    groups = build_stage10_decoder_param_groups(
        [
            ("_trajectory_head.diff_decoder.layers.0.weight", layer0),
            ("_trajectory_head.diff_decoder.layers.1.weight", layer1),
        ],
        0.1,
    )
    assert groups[0]["lr_scale"] == 1.0 and groups[1]["lr_scale"] == 0.1
    config = TransfuserConfig()
    config.grpo_training_mode = "diffgrpo_selected_anchor_base_preserve"
    config.generation_policy_algorithm = "diffgrpo_selected_anchor_base_preserve"
    config.grpo_decoder_gradient_scope = "all_layers"
    config.grpo_decoder_layer0_lr_mult = 0.1
    config.inference_selector_source = "reference"
    config.policy_loss_weight = 0.0
    config.kl_loss_weight = 0.0
    config.selector_generation_kl_weight = 0.0
    config.generation_kl_loss_weight = 0.0
    config.diffusion_truncation_timestep = 32
    config.diffusion_roll_timesteps = (32, 24, 16, 8, 0)
    validate_formal_grpo_config(config)
    config.diffgrpo_base_margin = 0.02
    with pytest.raises(ValueError, match="diffgrpo_base_margin"):
        validate_formal_grpo_config(config)


def test_stage21_scene_weights_and_whole_log_folds_are_balanced():
    weights = bounded_unit_mean([0.001, 1.0, 1000.0] * 10)
    assert weights.min() >= 0.25 and weights.max() <= 4.0
    assert abs(weights.mean() - 1.0) < 1e-9
    by_log = {}
    for index in range(780):
        by_log[f"log-{index:04d}"] = [{
            "token": f"token-{index:04d}",
            "base_reward_mean": 0.4 if index % 20 == 0 else 0.8,
            "safety_failure": index % 25 == 0,
            "selected_mode": 0,
        }]
    folds, audit, _ = build_folds(by_log, 20260728)
    assert audit["passes"]
    assert len(set().union(*(set(fold) for fold in folds))) == 780
    assert sum(len(fold) for fold in folds) == 780
