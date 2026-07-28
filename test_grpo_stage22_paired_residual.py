import copy
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import navsim.agents.diffusiondrive.diffusion_grpo as diffusion_grpo
from navsim.agents.diffusiondrive.paired_residual_lora import (
    attach_stage22_lora,
    merge_stage22_checkpoint_state_dict,
    merge_stage22_lora_inplace,
)
from navsim.agents.diffusiondrive.transfuser_agent import (
    validate_formal_grpo_config,
)
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_loss import (
    compute_paired_residual_diffgrpo_objective,
)
from navsim.agents.diffusiondrive.transfuser_model_v2 import (
    STAGE19_BASE_SHA256,
    STAGE19_FULL_SCHEDULE,
    load_paired_residual_manifest,
)
from scripts.evaluation.build_grpo_stage22_manifests import bounded_unit_mean


class TinyScheduler:
    def __init__(self):
        self.config = SimpleNamespace(num_train_timesteps=4)
        self.num_inference_steps = 4

    def set_timesteps(self, count, device):
        self.num_inference_steps = int(count)

    def add_noise(self, original_samples, noise, timesteps):
        return original_samples + noise

    def _get_variance(self, timestep, previous):
        return torch.tensor(0.04)

    def step(self, model_output, timestep, sample, eta, variance_noise):
        return SimpleNamespace(prev_sample=0.5 * sample + 0.5 * model_output)


class TinyPolicy(nn.Module):
    def __init__(self, value=0.0):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(float(value)))


class TinyHead:
    def __init__(self):
        self._roll_timesteps = (2, 1, 0)
        self._scheduler_num_inference_steps = 4
        self._generation_ddim_eta = 1.0
        self._generation_sigma_min = 1e-4
        self._generation_final_std = 0.05
        self._generation_trust_projection_mode = "none"
        self._truncation_timestep = 2
        self.diffusion_scheduler = TinyScheduler()
        self.diff_decoder = TinyPolicy(0.0)
        self.ref_policy = TinyPolicy(0.0)
        self.ref_policy.requires_grad_(False)

    @staticmethod
    def norm_odo(value):
        return value

    @staticmethod
    def denorm_odo(value):
        return value


def fake_decode(
    head, policy, sample, timestep, ego_query, agents_query, bev_feature,
    bev_spatial_shape, status_encoding, global_img,
):
    regression = sample + policy.offset
    classification = torch.zeros(
        sample.shape[:2], device=sample.device, dtype=sample.dtype
    )
    return regression, classification


def test_stage22_common_random_trace_is_identical_at_zero_residual(monkeypatch):
    monkeypatch.setattr(diffusion_grpo, "_decode_policy_step", fake_decode)
    head = TinyHead()
    clean = torch.zeros(1, 1, 3, 2)
    trace = diffusion_grpo.collect_paired_residual_diffgrpo_trace(
        head, clean, torch.tensor([3]), 8,
        None, None, None, None, None, None,
    )
    torch.testing.assert_close(
        trace["trajectories"], trace["base_trajectories"], atol=0, rtol=0
    )
    assert len(trace["transition_noises"]) == 2
    assert torch.count_nonzero(trace["reference_mean_kl"]) == 0
    assert trace["current_log_probs"].shape == (1, 8, 3)


def test_stage22_exact_kl_uses_base_states_and_backpropagates(monkeypatch):
    monkeypatch.setattr(diffusion_grpo, "_decode_policy_step", fake_decode)
    head = TinyHead()
    head.diff_decoder.offset.data.fill_(0.1)
    trace = diffusion_grpo.collect_paired_residual_diffgrpo_trace(
        head, torch.zeros(1, 1, 3, 2), torch.tensor([0]), 8,
        None, None, None, None, None, None,
    )
    assert torch.all(trace["reference_mean_kl"] > 0)
    trace["reference_mean_kl"].mean().backward()
    assert head.diff_decoder.offset.grad is not None
    assert head.diff_decoder.offset.grad.abs() > 0
    assert head.ref_policy.offset.grad is None


def make_objective_inputs():
    base = torch.tensor([[0.50, 0.80, 0.60, 0.70, 0.55, 0.65, 0.45, 0.85]])
    rewards = base + torch.tensor([[0.02, -0.005, 0.005, 0.02, 0.0, 0.0, 0.0, 0.0]])
    current_log_probs = torch.zeros(1, 8, 5, requires_grad=True)
    exact_kl = torch.zeros(1, 8, 5, requires_grad=True)
    valid = torch.ones(1, 8, dtype=torch.bool)
    current_components = torch.ones(1, 8, 6)
    base_components = torch.ones(1, 8, 6)
    current_components[0, 3, 0] = 0.9
    return (
        current_log_probs, exact_kl, rewards, valid, current_components,
        base, valid.clone(), base_components, torch.ones(1),
    )


def test_stage22_advantage_deadzone_mature_negative_and_safety_override():
    inputs = make_objective_inputs()
    result = compute_paired_residual_diffgrpo_objective(*inputs)
    advantages = result["advantages"]
    assert advantages[0, 0] > 0
    assert advantages[0, 1] < 0
    assert 0 < advantages[0, 2].abs() <= 0.25
    assert advantages[0, 3] == -2
    assert result["mature_negative_fraction"] > 0
    assert result["safety_override_fraction"] > 0
    assert result["bootstrap_advantage_fraction"] > 0
    result["loss"].backward()
    assert inputs[0].grad.abs().sum() > 0


def test_stage22_exact_kl_coefficients_are_regular_point1_mature_point5():
    inputs = list(make_objective_inputs())
    inputs[2] = inputs[5].clone()
    inputs[4] = inputs[7].clone()
    inputs[1] = torch.ones(1, 8, 5, requires_grad=True)
    result = compute_paired_residual_diffgrpo_objective(*inputs)
    mature_count = int((inputs[5] >= 0.75).sum())
    expected = (mature_count * 0.5 + (8 - mature_count) * 0.1) / 8
    assert result["reference_kl_loss"].item() == pytest.approx(expected)


def make_decoder():
    layers = []
    for _ in range(2):
        branch = nn.Sequential(
            nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4), nn.ReLU(),
            nn.Linear(4, 3),
        )
        layers.append(
            nn.ModuleDict({"task_decoder": nn.ModuleDict({"plan_reg_branch": branch})})
        )
    decoder = nn.Module()
    decoder.layers = nn.ModuleList(layers)
    return decoder


def test_stage22_lora_only_gradients_and_merge_equivalence():
    decoder = make_decoder()
    baseline = copy.deepcopy(decoder)
    names = attach_stage22_lora(decoder, rank=8, alpha=8.0)
    assert len(names) == 4
    assert all(name.endswith(("lora_A", "lora_B")) for name in names)
    inputs = torch.randn(2, 4)
    for layer in range(2):
        original = baseline.layers[layer]["task_decoder"]["plan_reg_branch"](inputs)
        adapted = decoder.layers[layer]["task_decoder"]["plan_reg_branch"](inputs)
        torch.testing.assert_close(original, adapted, atol=0, rtol=0)
        wrapper = decoder.layers[layer]["task_decoder"]["plan_reg_branch"][-1]
        wrapper.lora_B.data.normal_(std=0.01)
    outputs_before = [
        decoder.layers[layer]["task_decoder"]["plan_reg_branch"](inputs)
        for layer in range(2)
    ]
    training_state = decoder.state_dict()
    merged_state, prefixes = merge_stage22_checkpoint_state_dict(training_state)
    assert len(prefixes) == 2 and not any("lora_" in key for key in merged_state)
    merge_stage22_lora_inplace(decoder)
    outputs_after = [
        decoder.layers[layer]["task_decoder"]["plan_reg_branch"](inputs)
        for layer in range(2)
    ]
    for before, after in zip(outputs_before, outputs_after):
        torch.testing.assert_close(before, after, atol=1e-6, rtol=0)


def test_stage22_manifest_loader_and_weights_are_fail_closed(tmp_path):
    payload = {
        "summary": {
            "base_checkpoint_sha256": STAGE19_BASE_SHA256,
            "schedule": STAGE19_FULL_SCHEDULE,
            "group_definition": "same_token_same_selected_anchor_common_random",
            "group_size": 8,
            "stage22_objective": "paired_delta_bootstrap_exact_kl_lora_v2",
            "scene_weight_definition": "inverse_log_frequency_clip_0.5_2_unit_mean",
        },
        "records": [{"token": "abc", "selected_mode": 3, "scene_weight": 1.0}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    assert load_paired_residual_manifest(str(path), 8)["abc"]["selected_mode"] == 3
    payload["records"][0]["scene_weight"] = 2.1
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="invalid Stage22"):
        load_paired_residual_manifest(str(path), 8)
    weights = bounded_unit_mean([0.01, 1.0, 100.0] * 10)
    assert weights.min() >= 0.5 and weights.max() <= 2.0
    assert weights.mean() == pytest.approx(1.0)


def test_stage22_formal_config_is_locked():
    config = TransfuserConfig()
    config.grpo_training_mode = "diffgrpo_paired_residual"
    config.generation_policy_algorithm = "diffgrpo_paired_residual"
    config.grpo_decoder_gradient_scope = "all_layers"
    config.inference_selector_source = "reference"
    config.policy_loss_weight = 0.0
    config.kl_loss_weight = 0.0
    config.selector_generation_kl_weight = 0.0
    config.generation_kl_loss_weight = 0.0
    config.weight_decay = 0.0
    config.diffusion_truncation_timestep = 32
    config.diffusion_roll_timesteps = (32, 24, 16, 8, 0)
    validate_formal_grpo_config(config)
    config.diffgrpo_paired_mature_kl_weight = 0.4
    with pytest.raises(ValueError, match="mature_kl_weight"):
        validate_formal_grpo_config(config)
