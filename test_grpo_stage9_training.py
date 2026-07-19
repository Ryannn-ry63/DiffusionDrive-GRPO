"""Focused correctness tests for the Stage-9 full-decoder GRPO route."""

import copy
from dataclasses import replace
from types import MethodType

import pytest
import torch
from torch import nn

from navsim.agents.diffusiondrive.transfuser_agent import (
    validate_formal_grpo_config,
    validate_frozen_policy_state,
)
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_model_v2 import (
    CustomTransformerDecoder,
    TrajectoryHead,
)


class TinyRefinement(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2))

    def forward(
        self, traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
        agents_query, ego_query, time_embed, status_encoding, global_img=None,
    ):
        xy = noisy_traj_points + self.scale * traj_feature[..., None, :2]
        heading = xy[..., :1] * 0.0
        regression = torch.cat((xy, heading), dim=-1)
        classification = regression[..., 0, 0]
        return regression, classification


def run_tiny_decoder(decoder, traj_feature, points):
    return decoder(
        traj_feature, points, None, None, None, None, None, None
    )


def test_all_layers_changes_only_backward_not_forward():
    torch.manual_seed(0)
    legacy = CustomTransformerDecoder(
        TinyRefinement(), 2, gradient_scope="last_layer"
    )
    full = copy.deepcopy(legacy)
    full.gradient_scope = "all_layers"
    feature = torch.randn(1, 2, 2)
    points = torch.randn(1, 2, 3, 2)

    legacy_reg, legacy_cls = run_tiny_decoder(legacy, feature, points)
    full_reg, full_cls = run_tiny_decoder(full, feature, points)
    for legacy_value, full_value in zip(legacy_reg + legacy_cls, full_reg + full_cls):
        torch.testing.assert_close(legacy_value, full_value)

    legacy_reg[-1].sum().backward()
    full_reg[-1].sum().backward()
    assert legacy.layers[0].scale.grad is None
    assert torch.isfinite(legacy.layers[1].scale.grad)
    assert torch.isfinite(full.layers[0].scale.grad)
    assert torch.isfinite(full.layers[1].scale.grad)
    assert full.layers[0].scale.grad.abs() > 0
    assert full.layers[1].scale.grad.abs() > 0


class TinyPolicyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.diff_decoder = nn.Linear(2, 2, bias=False)
        self.old_policy = copy.deepcopy(self.diff_decoder)
        self._old_policy_sync_steps = 2
        self.register_buffer(
            "_old_policy_last_sync_step", torch.tensor(-1, dtype=torch.long)
        )
        self.maybe_sync_old_policy = MethodType(
            TrajectoryHead.maybe_sync_old_policy, self
        )


def test_old_policy_sync_state_survives_resume():
    head = TinyPolicyHead()
    head.maybe_sync_old_policy(0)
    with torch.no_grad():
        head.diff_decoder.weight.add_(1.0)
    head.maybe_sync_old_policy(1)
    assert not torch.equal(
        head.diff_decoder.weight, head.old_policy.weight
    )

    resumed = TinyPolicyHead()
    resumed.load_state_dict(head.state_dict())
    assert int(resumed._old_policy_last_sync_step) == 0
    resumed.maybe_sync_old_policy(1)
    assert not torch.equal(
        resumed.diff_decoder.weight, resumed.old_policy.weight
    )
    resumed.maybe_sync_old_policy(2)
    torch.testing.assert_close(
        resumed.diff_decoder.weight, resumed.old_policy.weight
    )
    assert int(resumed._old_policy_last_sync_step) == 2


def test_frozen_reference_state_validation_detects_mutation():
    policy = nn.Linear(2, 2)
    policy.requires_grad_(False)
    expected = {
        name: value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
    }
    validate_frozen_policy_state(policy, expected)
    with torch.no_grad():
        policy.weight.add_(1.0)
    with pytest.raises(RuntimeError, match="changed relative to base"):
        validate_frozen_policy_state(policy, expected)


def formal_config(mode):
    common = dict(
        grpo_training_mode=mode,
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
    )
    if mode == "selector_group":
        common.update(
            policy_loss_weight=1.0,
            kl_loss_weight=0.01,
            selector_generation_kl_weight=0.1,
            generation_policy_loss_weight=0.0,
            generation_kl_loss_weight=0.0,
        )
    else:
        common.update(
            policy_loss_weight=0.0,
            kl_loss_weight=0.0,
            selector_generation_kl_weight=0.0,
            generation_policy_loss_weight=1.0,
            generation_kl_loss_weight=0.1,
        )
    return replace(TransfuserConfig(), **common)


@pytest.mark.parametrize("mode", ("selector_group", "generation_group"))
def test_formal_stage9_config_accepts_only_registered_routes(mode):
    config = formal_config(mode)
    validate_formal_grpo_config(config)
    with pytest.raises(ValueError, match="requires grpo_clip_ratio"):
        validate_formal_grpo_config(replace(config, grpo_clip_ratio=0.1))
