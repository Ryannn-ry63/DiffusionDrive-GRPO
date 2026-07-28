import hashlib
from types import SimpleNamespace

import pytest
import torch
from navsim.agents.diffusiondrive import stage35_nested_trace

from navsim.agents.diffusiondrive.stage35_contract import (
    STAGE35_ROUTE_EXPECTED,
)
from navsim.agents.diffusiondrive.stage35_counterfactual import (
    STAGE35_OBJECTIVE_REVISION,
    STAGE35_PLAN_SHA256,
    compute_counterfactual_contributions,
    compute_nested_counterfactual_deployment_objective,
    compute_same_anchor_advantages,
    counterfactual_donor_indices,
    union_selector_top2,
)
from navsim.agents.diffusiondrive.stage35_loss_bridge import (
    compute_stage35_transfuser_loss,
)
from navsim.agents.diffusiondrive.transfuser_agent import (
    FORMAL_GRPO_MODES,
    STAGE35_GRPO_MODES,
    validate_formal_grpo_config,
)
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_model_v2 import (
    resolve_mode_coverage_bucket_manifest,
)


def _objective_inputs(batch: int = 2):
    groups, modes, active, steps = 8, 20, 4, 5
    bank_offsets = torch.linspace(-0.014, 0.014, groups)
    rewards = torch.full((batch, groups, modes), 0.80)
    rewards[:, :, 0] += bank_offsets
    base_rewards = rewards - 0.001
    valid = torch.ones(batch, groups, modes, dtype=torch.bool)
    components = torch.ones(batch, groups, modes, 6)
    active_modes = torch.arange(active).view(1, 1, active).expand(
        batch, groups, active
    ).clone()
    active_valid = torch.ones_like(active_modes, dtype=torch.bool)
    selected = torch.zeros(batch, groups, dtype=torch.long)
    donors = counterfactual_donor_indices(groups)
    hybrid_selected = torch.zeros(
        batch, groups, active, groups - 1, dtype=torch.long
    )
    return {
        "current_log_probs": torch.zeros(
            batch, groups, active, steps, requires_grad=True
        ),
        "bc_log_probs": torch.full(
            (batch, groups, active, steps), -2.0, requires_grad=True
        ),
        "reference_mean_kl": torch.full(
            (batch, groups, active, steps), 0.5, requires_grad=True
        ),
        "rewards": rewards,
        "valid_mask": valid,
        "component_scores": components.clone(),
        "base_rewards": base_rewards,
        "base_valid_mask": valid.clone(),
        "base_component_scores": components,
        "current_selected_modes": selected,
        "public_selected_modes": selected.clone(),
        "active_modes": active_modes,
        "active_valid": active_valid,
        "hybrid_selected_modes": hybrid_selected,
        "donor_indices": donors,
        "scene_buckets": torch.arange(batch) % 4,
    }


def test_stage35_active_union_is_stable_unique_and_contains_deployments():
    current = torch.tensor([[[3, 5], [1, 2]]])
    public = torch.tensor([[[3, 7], [4, 1]]])
    valid = torch.ones_like(current, dtype=torch.bool)
    modes, mask = union_selector_top2(
        current, valid, public, valid, max_modes=4
    )
    assert modes.tolist() == [[[3, 5, 7, 0], [1, 2, 4, 0]]]
    assert mask.tolist() == [
        [[True, True, True, False], [True, True, True, False]]
    ]


def test_stage35_counterfactual_credit_uses_same_anchor_donor_reward():
    values = _objective_inputs(batch=1)
    result = compute_counterfactual_contributions(
        rewards=values["rewards"],
        valid_mask=values["valid_mask"],
        selected_modes=values["current_selected_modes"],
        active_modes=values["active_modes"],
        active_valid=values["active_valid"],
        hybrid_selected_modes=values["hybrid_selected_modes"],
        donor_indices=values["donor_indices"],
    )
    mode0 = result["contributions"][0, :, 0]
    expected = torch.linspace(-0.014, 0.014, 8)
    expected = expected - (
        expected.sum() - expected
    ) / 7
    assert torch.allclose(mode0, expected, atol=1e-7)
    assert torch.allclose(
        result["contributions"][0, :, 1:], torch.zeros(8, 3), atol=2e-7
    )
    assert result["deployment_influence"][0, :, 0].all()



def test_stage35_hybrid_builder_replaces_only_same_anchor(monkeypatch):
    batch, groups, modes = 2, 8, 20
    banks = torch.zeros(batch, groups, modes, 8, 3)
    logits = torch.zeros(batch, groups, modes)
    for batch_index in range(batch):
        for group_index in range(groups):
            for mode in range(modes):
                banks[batch_index, group_index, mode].fill_(
                    batch_index * 10000 + group_index * 100 + mode
                )
                logits[batch_index, group_index, mode] = (
                    group_index + 0.01 * mode
                )
    active_modes = torch.arange(4).view(1, 1, 4).expand(
        batch, groups, 4
    ).clone()
    donors = counterfactual_donor_indices(groups)
    calls = []

    def fake_select(
        head,
        candidates,
        candidate_logits,
        repeated_bev,
        bev_spatial_shape,
        repeated_agents,
        repeated_ego,
        repeated_status,
    ):
        scene_from_bank = (candidates[:, 0, 0, 0] // 10000).long()
        assert torch.equal(scene_from_bank, repeated_status[:, 0].long())
        calls.append(candidates.shape[0])
        return candidate_logits.argmax(dim=-1), {}

    monkeypatch.setattr(
        stage35_nested_trace, "_select_stage31_deployed_modes", fake_select
    )
    head = SimpleNamespace(
        _config=SimpleNamespace(stage35_selector_micro_batch_size=13)
    )
    status = torch.arange(batch, dtype=torch.float32).unsqueeze(-1)
    selected = stage35_nested_trace._select_counterfactual_hybrids(
        head,
        banks,
        logits,
        active_modes,
        donors,
        torch.zeros(batch, 1),
        torch.zeros(batch, 1),
        torch.zeros(batch, 1),
        (1, 1),
        status,
    )
    expected = torch.empty_like(selected)
    for b in range(batch):
        for g in range(groups):
            for slot in range(4):
                for donor_slot, donor in enumerate(donors[g].tolist()):
                    hybrid_logits = logits[b, g].clone()
                    mode = int(active_modes[b, g, slot])
                    hybrid_logits[mode] = logits[b, donor, mode]
                    expected[b, g, slot, donor_slot] = hybrid_logits.argmax()
    assert torch.equal(selected, expected)
    assert max(calls) == 13
    assert sum(calls) == batch * groups * 4 * 7


def test_stage35_normalizes_only_replicas_of_the_same_anchor():
    values = _objective_inputs(batch=1)
    contribution = torch.zeros(1, 8, 4)
    contribution[0, :, 0] = torch.arange(8, dtype=torch.float32)
    contribution[0, :, 1] = 100 + torch.arange(8, dtype=torch.float32)
    result = compute_same_anchor_advantages(
        contributions=contribution,
        valid_mask=values["active_valid"],
        active_modes=values["active_modes"],
        num_modes=20,
    )
    assert result["group_count"][0, 0].item() == 8
    assert result["group_count"][0, 1].item() == 8
    assert result["advantages"][0, :, 0].mean().item() == pytest.approx(0.0)
    assert result["advantages"][0, :, 1].mean().item() == pytest.approx(0.0)
    assert torch.allclose(
        result["advantages"][0, :, 0],
        result["advantages"][0, :, 1],
    )


def test_stage35_objective_is_finite_gradient_bearing_and_mean_normalized():
    values = _objective_inputs()
    result = compute_nested_counterfactual_deployment_objective(
        **values, step_discount=1.0
    )
    assert torch.isfinite(result["loss"])
    assert torch.allclose(result["bc_loss"], torch.tensor(0.2))
    assert torch.allclose(result["reference_kl_loss"], torch.tensor(0.05))
    assert torch.allclose(result["loss"], torch.tensor(0.25))
    assert result["counterfactual_nonzero_fraction"].item() > 0.10
    assert result["nonowner_influence_scene_fraction"].item() == 0.0
    result["loss"].backward()
    assert values["current_log_probs"].grad is not None
    assert values["bc_log_probs"].grad.abs().sum().item() > 0
    assert values["reference_mean_kl"].grad.abs().sum().item() > 0


def test_stage35_loss_bridge_requires_candidate_level_accounting():
    values = _objective_inputs(batch=1)
    predictions = {
        "diffgrpo_current_log_probs": values["current_log_probs"],
        "diffgrpo_bc_log_probs": values["bc_log_probs"],
        "diffgrpo_reference_mean_kl": values["reference_mean_kl"],
        "raw_rewards": values["rewards"].reshape(1, 160),
        "reward_valid_mask": values["valid_mask"].reshape(1, 160),
        "component_scores": values["component_scores"].reshape(1, 160, 6),
        "diffgrpo_base_rewards": values["base_rewards"].reshape(1, 160),
        "diffgrpo_base_valid_mask": values["base_valid_mask"].reshape(1, 160),
        "diffgrpo_base_component_scores": values[
            "base_component_scores"
        ].reshape(1, 160, 6),
        "stage35_current_selected_modes": values["current_selected_modes"],
        "stage35_public_selected_modes": values["public_selected_modes"],
        "stage35_active_modes": values["active_modes"],
        "stage35_active_valid": values["active_valid"],
        "stage35_hybrid_selected_modes": values["hybrid_selected_modes"],
        "stage35_donor_indices": values["donor_indices"],
        "stage35_scene_buckets": values["scene_buckets"],
        "stage35_sampled_chain_count": torch.tensor(320),
        "stage35_replayed_chain_count": torch.tensor(64),
        "stage35_counterfactual_selector_count": torch.tensor(224),
        "stage35_selector_mode_disagreement": torch.tensor(0.1),
        "stage35_current_selector_switch_rate": torch.tensor(0.1),
        "stage35_public_selector_switch_rate": torch.tensor(0.1),
        "diffgrpo_group_size": torch.tensor(8),
    }
    result = compute_stage35_transfuser_loss(predictions, SimpleNamespace())
    assert torch.isfinite(result["loss"])
    predictions["stage35_counterfactual_selector_count"] = torch.tensor(223)
    with pytest.raises(ValueError, match="accounting constants"):
        compute_stage35_transfuser_loss(predictions, SimpleNamespace())


def _formal_config() -> TransfuserConfig:
    config = TransfuserConfig()
    config.grpo_training_mode = (
        "stage35_nested_counterfactual_deployment_grpo"
    )
    for name, value in STAGE35_ROUTE_EXPECTED.items():
        setattr(config, name, value)
    config.grpo_decoder_gradient_scope = "all_layers"
    config.grpo_decoder_layer0_lr_mult = 0.1
    config.inference_selector_source = "trajectory_relative_harm_v3"
    config.stage25_selector_checkpoint_path = "selector.ckpt"
    config.stage25_selector_calibration_path = "calibration.json"
    config.stage35_bucket_manifest_path = "buckets.json"
    config.policy_loss_weight = 0.0
    config.kl_loss_weight = 0.0
    config.selector_generation_kl_weight = 0.0
    config.generation_policy_loss_weight = 1.0
    config.generation_kl_loss_weight = 0.0
    config.generation_adaptive_kl_enabled = False
    config.diffgrpo_bc_weight = 0.1
    config.diffgrpo_base_advantage_clip = 2.0
    config.diffgrpo_safety_regression_tolerance = 1e-6
    config.diffgrpo_step_discount = 0.6
    config.diffgrpo_logprob_reduction = "mean"
    config.diffusion_truncation_timestep = 32
    config.diffusion_roll_timesteps = (32, 24, 16, 8, 0)
    config.diffusion_scheduler_num_inference_steps = 125
    config.weight_decay = 0.0
    return config


def test_stage35_formal_route_and_plan_are_frozen():
    config = _formal_config()
    assert STAGE35_GRPO_MODES <= FORMAL_GRPO_MODES
    validate_formal_grpo_config(config)
    stage, path = resolve_mode_coverage_bucket_manifest(
        config, config.grpo_training_mode
    )
    assert (stage, path) == ("Stage35", "buckets.json")
    config.stage35_counterfactual_weight = 0.20
    with pytest.raises(ValueError, match="stage35_counterfactual_weight"):
        validate_formal_grpo_config(config)
    with open(
        "GRPO_STAGE35_NESTED_COUNTERFACTUAL_DEPLOYMENT_PLAN_20260727.md",
        "rb",
    ) as plan_file:
        assert hashlib.sha256(plan_file.read()).hexdigest() == STAGE35_PLAN_SHA256
    assert STAGE35_OBJECTIVE_REVISION == "nested_counterfactual_deployment_v1"
