"""Focused correctness tests for the Stage-10 adaptive generation route."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from navsim.agents.diffusiondrive.diffusion_grpo import AdaptiveKLController
from navsim.agents.diffusiondrive.modules.scheduler import WarmupCosLR
from navsim.agents.diffusiondrive.transfuser_agent import (
    build_stage10_decoder_param_groups,
    validate_formal_grpo_config,
)
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig


def make_controller(**overrides):
    kwargs = dict(
        initial_coefficient=0.1,
        minimum_coefficient=0.1,
        maximum_coefficient=100.0,
        target=1e-4,
        hard_limit=2.5e-4,
        window=4,
        update_interval=2,
        adaptation_factor=2.0,
        lower_ratio=2.0 / 3.0,
        upper_ratio=1.5,
        hard_limit_patience=2,
    )
    kwargs.update(overrides)
    return AdaptiveKLController(**kwargs)


def observe(controller, value, count):
    metrics = None
    for _ in range(count):
        metrics = controller.update(torch.tensor(value))
    return metrics


def test_adaptive_kl_controller_increases_decreases_and_respects_bounds():
    controller = make_controller()
    observe(controller, 2e-4, 4)
    assert controller.coefficient.item() == pytest.approx(0.2)

    observe(controller, 2e-4, 4)
    assert controller.coefficient.item() == pytest.approx(0.8)

    observe(controller, 1e-5, 4)
    assert controller.coefficient.item() == pytest.approx(0.4)

    bounded_high = make_controller(
        initial_coefficient=0.4,
        minimum_coefficient=0.1,
        maximum_coefficient=0.4,
    )
    observe(bounded_high, 2e-4, 8)
    assert bounded_high.coefficient.item() == pytest.approx(0.4)

    bounded_low = make_controller(initial_coefficient=0.2)
    observe(bounded_low, 1e-5, 8)
    assert bounded_low.coefficient.item() == pytest.approx(0.1)


def test_adaptive_kl_controller_hard_stop_requires_consecutive_checked_windows():
    controller = make_controller()
    metrics = observe(controller, 3e-4, 4)
    assert metrics["generation_kl_controller_checked"].item() == 1
    assert metrics["generation_kl_hard_violation_count"].item() == 1
    assert metrics["generation_kl_should_stop"].item() == 0

    metrics = observe(controller, 3e-4, 2)
    assert metrics["generation_kl_controller_checked"].item() == 1
    assert metrics["generation_kl_hard_violation_count"].item() == 2
    assert metrics["generation_kl_should_stop"].item() == 1


def test_adaptive_kl_controller_resume_preserves_window_and_counters():
    uninterrupted = make_controller()
    observe(uninterrupted, 3e-4, 5)

    resumed = make_controller()
    resumed.load_state_dict(uninterrupted.state_dict())

    expected = uninterrupted.update(torch.tensor(3e-4))
    actual = resumed.update(torch.tensor(3e-4))
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])
    for key, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[key], value)


def formal_stage10_config(layer0_lr_mult):
    return replace(
        TransfuserConfig(),
        grpo_training_mode="generation_group_adaptive",
        grpo_decoder_gradient_scope="all_layers",
        grpo_reward_mode="pdms",
        grpo_scene_weight_mode="uniform",
        selection_behavior_weighting="old_policy",
        generation_advantage_mode="group_zscore",
        generation_mode_weighting="uniform",
        generation_trust_projection_mode="none",
        grpo_rollouts_per_mode=1,
        grpo_old_policy_sync_steps=32,
        grpo_priority_manifest_path="",
        grpo_priority_sample_fraction=0.0,
        selection_entropy_weight=0.0,
        selection_exploration_floor=0.0,
        selection_rank_loss_weight=0.0,
        selector_consistency_kl_weight=0.0,
        grpo_clip_ratio=0.2,
        policy_loss_weight=0.0,
        kl_loss_weight=0.0,
        selector_generation_kl_weight=0.0,
        generation_policy_loss_weight=1.0,
        generation_kl_loss_weight=0.1,
        generation_adaptive_kl_enabled=True,
        generation_kl_initial_coefficient=0.1,
        generation_kl_min_coefficient=0.1,
        generation_kl_max_coefficient=100.0,
        generation_kl_target=1e-4,
        generation_kl_hard_limit=2.5e-4,
        generation_kl_window=32,
        generation_kl_update_interval=8,
        generation_kl_adaptation_factor=2.0,
        generation_kl_lower_ratio=2.0 / 3.0,
        generation_kl_upper_ratio=1.5,
        generation_kl_hard_limit_patience=2,
        grpo_decoder_layer0_lr_mult=layer0_lr_mult,
    )


@pytest.mark.parametrize("layer0_lr_mult", (0.1, 1.0))
def test_formal_stage10_config_accepts_registered_layer_learning_rates(
    layer0_lr_mult,
):
    validate_formal_grpo_config(formal_stage10_config(layer0_lr_mult))


def test_formal_stage10_config_rejects_unregistered_layer_learning_rate():
    with pytest.raises(ValueError, match="layer-0 LR multiplier"):
        validate_formal_grpo_config(formal_stage10_config(0.5))


class TinyTwoLayerDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._trajectory_head = nn.Module()
        self._trajectory_head.diff_decoder = nn.Module()
        self._trajectory_head.diff_decoder.layers = nn.ModuleList(
            [nn.Linear(2, 2), nn.Linear(2, 2)]
        )


@pytest.mark.parametrize("layer0_lr_mult", (0.1, 1.0))
def test_stage10_optimizer_groups_train_both_layers_with_expected_lr(
    layer0_lr_mult,
):
    model = TinyTwoLayerDecoder()
    groups = build_stage10_decoder_param_groups(
        model.named_parameters(), layer0_lr_mult
    )
    optimizer = torch.optim.Adam(groups, lr=1e-6)
    scheduler = WarmupCosLR(
        optimizer=optimizer,
        min_lr=1e-6,
        lr=1e-6,
        warmup_epochs=3,
        epochs=100,
    )

    layer0_ids = {
        id(parameter) for parameter in model._trajectory_head.diff_decoder.layers[0].parameters()
    }
    layer1_ids = {
        id(parameter) for parameter in model._trajectory_head.diff_decoder.layers[1].parameters()
    }
    group_ids = [{id(parameter) for parameter in group["params"]} for group in groups]
    assert layer0_ids <= group_ids[1]
    assert layer1_ids <= group_ids[0]
    assert not layer0_ids & group_ids[0]
    assert not layer1_ids & group_ids[1]
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert scheduler.get_last_lr() == pytest.approx(
        [1e-6 / 3.0, 1e-6 * layer0_lr_mult / 3.0]
    )


def test_stage9_static_config_remains_valid_and_has_no_adaptive_controller():
    config = replace(
        formal_stage10_config(1.0),
        grpo_training_mode="generation_group",
        generation_adaptive_kl_enabled=False,
    )
    validate_formal_grpo_config(config)
    assert config.generation_kl_loss_weight == pytest.approx(0.1)
    assert config.generation_adaptive_kl_enabled is False
