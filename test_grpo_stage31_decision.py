from pathlib import Path

import pytest
import numpy as np
import torch

from navsim.agents.diffusiondrive import diffusion_grpo
from navsim.agents.diffusiondrive.transfuser_agent import (
    FORMAL_GRPO_MODES,
    STAGE31_GRPO_MODES,
    validate_formal_grpo_config,
)
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_loss import (
    compute_deployed_decision_grpo_objective,
)
from scripts.evaluation import check_grpo_stage31_cv


DP = "stage31_public_deployed_pair"
DPF = "stage31_public_deployed_frontier"
PLAN_SHA = "3ff3d3bcd3120b9d73a017876f942458eb3fa16651517e4e30b27cf2fff7bb0d"


class _Scheduler:
    def add_noise(self, original_samples, noise, timesteps):
        assert timesteps.shape == (original_samples.shape[0],)
        return original_samples + noise


class _Config:
    inference_selector_source = "trajectory_relative_harm_v3"


class _Head:
    def __init__(self):
        self._config = _Config()
        self.ref_policy = object()
        self.diff_decoder = object()
        self.diffusion_scheduler = _Scheduler()
        self.plan_anchor = torch.zeros(20, 8, 3)
        self._truncation_timestep = 32
        self._generation_trust_projection_mode = "none"
        self._stage24_selector_training_updates = torch.tensor(1)
        self._stage25_selector_training_updates = torch.tensor(1)
        self._stage24_calibration_loaded = torch.tensor(True)

    def norm_odo(self, value):
        return value

    def denorm_odo(self, value):
        return value


def _inputs(batch=1, delta=None, base=None):
    if base is None:
        base = torch.linspace(0.40, 0.75, 8).repeat(batch, 1)
    if delta is None:
        delta = torch.tensor([[
            0.004, 0.003, 0.002, 0.001,
            -0.001, -0.002, -0.003, -0.004,
        ]]).repeat(batch, 1)
    current_log_probs = torch.zeros(batch, 8, 5, requires_grad=True)
    bc_log_probs = torch.zeros(batch, 8, 5, requires_grad=True)
    reference_mean_kl = torch.full(
        (batch, 8, 5), 0.01, requires_grad=True
    )
    valid = torch.ones(batch, 8, dtype=torch.bool)
    components = torch.ones(batch, 8, 6)
    return {
        "current_log_probs": current_log_probs,
        "bc_log_probs": bc_log_probs,
        "reference_mean_kl": reference_mean_kl,
        "rewards": base + delta,
        "valid_mask": valid,
        "component_scores": components.clone(),
        "base_rewards": base,
        "base_valid_mask": valid.clone(),
        "base_component_scores": components,
    }


def _formal_config(mode):
    config = TransfuserConfig()
    config.grpo_training_mode = mode
    config.generation_policy_algorithm = "diffgrpo_deployed_selected_set"
    config.grpo_decoder_gradient_scope = "all_layers"
    config.grpo_decoder_layer0_lr_mult = 0.1
    config.inference_selector_source = "trajectory_relative_harm_v3"
    config.stage25_selector_checkpoint_path = "selector.ckpt"
    config.stage25_selector_calibration_path = "calibration.json"
    config.stage31_bucket_manifest_path = "buckets.json"
    config.policy_loss_weight = 0.0
    config.kl_loss_weight = 0.0
    config.selector_generation_kl_weight = 0.0
    config.generation_kl_loss_weight = 0.0
    config.diffusion_truncation_timestep = 32
    config.diffusion_roll_timesteps = (32, 24, 16, 8, 0)
    config.diffgrpo_group_size = 8
    config.weight_decay = 0.0
    return config


def test_stage31_trace_selects_current_and_public_banks_independently(monkeypatch):
    head = _Head()
    selector_calls = []
    replay_state_means = []
    transition = tuple(torch.zeros(1, 160, 8, 3) for _ in range(4))
    final_noise = torch.zeros(1, 160, 8, 3)

    def fake_sample(_head, policy, initial, *args, **kwargs):
        mode_ids = (
            torch.arange(160, dtype=initial.dtype)
            .remainder(20)
            .view(1, 160, 1, 1)
            .expand_as(initial)
        )
        offset = 0.0 if policy is head.diff_decoder else 100.0
        final = mode_ids + offset
        states = tuple(final + index for index in range(5))
        actions = tuple(final + index + 0.5 for index in range(5))
        logits = initial.new_zeros(1, 160)
        if kwargs.get("return_noise_bundle"):
            return states, actions, final, logits, {
                "transition_noises": transition,
                "final_noise": final_noise,
            }
        assert all(
            torch.equal(actual, wanted)
            for actual, wanted in zip(
                kwargs["transition_noises"], transition
            )
        )
        assert torch.equal(kwargs["final_noise"], final_noise)
        return states, actions, final, logits

    def fake_select(_head, candidates, *args):
        selector_calls.append(candidates.detach().clone())
        mode = 1 if len(selector_calls) == 1 else 2
        selected = torch.full((8,), mode, dtype=torch.long)
        return selected, {"switch": torch.zeros(8, dtype=torch.bool)}

    def fake_replay(_head, policy, states, actions, *args):
        replay_state_means.append(states[0].mean().item())
        return states[0].new_zeros(1, 8, 5), states[0].new_zeros(1, 8)

    def fake_kl(_head, current, reference, states, *args):
        return states[0].new_zeros(1, 8, 5)

    monkeypatch.setattr(diffusion_grpo, "_sample_full_chain_actions", fake_sample)
    monkeypatch.setattr(
        diffusion_grpo, "_select_stage31_deployed_modes", fake_select
    )
    monkeypatch.setattr(diffusion_grpo, "_replay_full_chain_actions", fake_replay)
    monkeypatch.setattr(diffusion_grpo, "_replay_reference_mean_kl", fake_kl)

    trace = diffusion_grpo.collect_deployed_selected_set_diffgrpo_trace(
        head,
        8,
        torch.zeros(1, 1, 1),
        torch.zeros(1, 1, 1),
        torch.zeros(1, 1, 1),
        (1, 1),
        torch.zeros(1, 1),
        None,
    )
    assert len(selector_calls) == 2
    assert torch.all(trace["current_selected_modes"] == 1)
    assert torch.all(trace["public_selected_modes"] == 2)
    assert trace["selector_mode_disagreement"].item() == 1.0
    assert torch.all(trace["trajectories"] == 1)
    assert torch.all(trace["base_trajectories"] == 102)
    assert replay_state_means == pytest.approx([1.0, 102.0])
    assert trace["current_sampled_chain_count"].item() == 160
    assert trace["public_sampled_chain_count"].item() == 160
    assert trace["current_replayed_chain_count"].item() == 8
    assert trace["public_replayed_chain_count"].item() == 8


def test_stage31_dp_preserves_absolute_deployment_sign():
    inputs = _inputs()
    delta = inputs["rewards"] - inputs["base_rewards"]
    result = compute_deployed_decision_grpo_objective(
        **inputs, use_frontier_rank=False
    )
    advantages = result["advantages"]
    assert torch.all(advantages[delta > 0] >= 0)
    assert torch.all(advantages[delta < 0] <= 0)


def test_stage31_dpf_bootstraps_exact_public_ties_but_dp_does_not():
    base = torch.linspace(0.40, 0.75, 8).unsqueeze(0)
    inputs = _inputs(delta=torch.zeros(1, 8), base=base)
    dp = compute_deployed_decision_grpo_objective(
        **inputs, use_frontier_rank=False
    )
    dpf = compute_deployed_decision_grpo_objective(
        **inputs, use_frontier_rank=True
    )
    assert torch.all(dp["advantages"] == 0)
    assert torch.any(dpf["advantages"] > 0)
    assert torch.any(dpf["advantages"] < 0)
    assert dpf["tie_fraction"].item() == 1.0


def test_stage31_mature_ties_have_no_frontier_bootstrap():
    base = torch.linspace(0.91, 0.98, 8).unsqueeze(0)
    result = compute_deployed_decision_grpo_objective(
        **_inputs(delta=torch.zeros(1, 8), base=base),
        use_frontier_rank=True,
    )
    assert torch.all(result["advantages"] == 0)
    assert result["headroom_mean"].item() == 0


def test_stage31_safety_regression_overrides_positive_delta_and_raises_kl():
    inputs = _inputs(delta=torch.full((1, 8), 0.01))
    inputs["component_scores"][0, 3, 1] = 0.99
    result = compute_deployed_decision_grpo_objective(
        **inputs, use_frontier_rank=True
    )
    assert result["advantages"][0, 3].item() == -2.0
    assert result["safety_override_fraction"].item() == pytest.approx(0.125)
    assert result["mean_kl_weight"].item() > 0.1


def test_stage31_policy_bc_and_kl_gradients_are_finite():
    inputs = _inputs()
    result = compute_deployed_decision_grpo_objective(
        **inputs, use_frontier_rank=True
    )
    result["loss"].backward()
    for name in ("current_log_probs", "bc_log_probs", "reference_mean_kl"):
        gradient = inputs[name].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum().item() > 0


def test_stage31_fails_closed_on_shape_nonfinite_and_constants():
    bad = _inputs()
    bad["rewards"] = torch.zeros(1, 7)
    with pytest.raises(ValueError, match="rewards"):
        compute_deployed_decision_grpo_objective(
            **bad, use_frontier_rank=True
        )
    bad = _inputs()
    bad["reference_mean_kl"] = -torch.ones(1, 8, 5)
    with pytest.raises(FloatingPointError, match="non-negative"):
        compute_deployed_decision_grpo_objective(
            **bad, use_frontier_rank=True
        )
    with pytest.raises(ValueError, match="constants"):
        compute_deployed_decision_grpo_objective(
            **_inputs(),
            use_frontier_rank=True,
            headroom_low=0.9,
            headroom_high=0.75,
        )


def test_stage31_formal_modes_and_constants_are_locked():
    assert STAGE31_GRPO_MODES <= FORMAL_GRPO_MODES
    validate_formal_grpo_config(_formal_config(DP))
    validate_formal_grpo_config(_formal_config(DPF))
    drifted = _formal_config(DPF)
    drifted.stage31_rank_weight = 0.75
    with pytest.raises(ValueError, match="stage31_rank_weight"):
        validate_formal_grpo_config(drifted)
    missing_bucket = _formal_config(DPF)
    missing_bucket.stage31_bucket_manifest_path = ""
    with pytest.raises(ValueError, match="bucket manifest"):
        validate_formal_grpo_config(missing_bucket)


def test_stage31_plan_sha_and_training_boundaries_are_wired():
    config = TransfuserConfig()
    assert config.stage31_plan_sha256 == PLAN_SHA
    plan = Path(
        "GRPO_STAGE31_SELECTOR_CONSISTENT_DECISION_GRPO_PLAN_20260725.md"
    )
    import hashlib

    assert hashlib.sha256(plan.read_bytes()).hexdigest() == PLAN_SHA
    for path in (
        "navsim/agents/diffusiondrive/transfuser_model_v2.py",
        "navsim/agents/diffusiondrive/transfuser_loss.py",
        "navsim/planning/training/agent_lightning_module.py",
        "navsim/planning/script/run_training.py",
    ):
        text = Path(path).read_text(encoding="utf-8")
        assert DP in text
        assert DPF in text


def test_stage31_oof_gate_is_dpf_only_and_enforces_frozen_thresholds(monkeypatch):
    monkeypatch.setattr(
        check_grpo_stage31_cv,
        "bootstrap_whole_log",
        lambda values, logs: [0.0001, 0.02],
    )
    buckets = np.tile(np.arange(4), 16)
    folds = np.tile(np.repeat(np.arange(4), 4), 4)
    noises = np.tile(np.asarray([20261411, 20261412]), 32)
    selected_by_bucket = np.asarray([0.020, 0.020, 0.004, 0.0001])
    selected = selected_by_bucket[buckets]
    vectors = {
        "selected": selected,
        "candidate_mean": np.zeros_like(selected),
        "oracle": np.zeros_like(selected),
        "fallback": np.full_like(selected, 0.005),
    }
    components = {
        name: np.zeros_like(selected)
        for name in check_grpo_stage31_cv.COMPONENTS
    }
    args = (
        48, vectors, buckets, folds, noises,
        [f"log-{index}" for index in range(len(selected))], components,
    )
    dpf = check_grpo_stage31_cv.summarize("DPF", *args)
    dp = check_grpo_stage31_cv.summarize("DP", *args)
    assert dpf["passes_numeric_gate"]
    assert dpf["eligible"]
    assert dp["passes_numeric_gate"]
    assert not dp["eligible"]

    damaged = {name: value.copy() for name, value in vectors.items()}
    damaged["candidate_mean"][:] = -0.001
    failed = check_grpo_stage31_cv.summarize(
        "DPF", 48, damaged, buckets, folds, noises,
        [f"log-{index}" for index in range(len(selected))], components,
    )
    assert not failed["checks"][
        "candidate_mean_delta_at_least_negative_0.0005"
    ]
    assert not failed["eligible"]
