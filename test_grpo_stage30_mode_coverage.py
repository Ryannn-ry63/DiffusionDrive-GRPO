import pytest
import torch
import numpy as np

from navsim.agents.diffusiondrive import diffusion_grpo
from navsim.agents.diffusiondrive.transfuser_agent import (
    FORMAL_GRPO_MODES,
    STAGE30_GRPO_MODES,
    validate_formal_grpo_config,
)
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_loss import (
    compute_mode_coverage_constrained_objective,
)
from scripts.evaluation import check_grpo_stage30_cv


COV = "stage30_public_mode_coverage"
MCC = "stage30_public_mode_coverage_constrained"


class _Scheduler:
    def add_noise(self, original_samples, noise, timesteps):
        assert timesteps.shape == (original_samples.shape[0],)
        return original_samples + noise


class _Head:
    def __init__(self):
        self.ref_policy = object()
        self.diff_decoder = object()
        self.diffusion_scheduler = _Scheduler()
        self.plan_anchor = torch.zeros(20, 8, 3)
        self._truncation_timestep = 32
        self._generation_trust_projection_mode = "none"
        self._diffgrpo_group_size = 20

    def norm_odo(self, value):
        return value

    def denorm_odo(self, value):
        return value


def _inputs(batch=4, delta=None, buckets=None):
    if buckets is None:
        buckets = torch.arange(batch, dtype=torch.long) % 4
    base = torch.linspace(0.50, 0.88, 20).repeat(batch, 1)
    if delta is None:
        delta = torch.linspace(-0.004, 0.005, 20).repeat(batch, 1)
    current_log_probs = torch.zeros(batch, 20, 5, requires_grad=True)
    bc_log_probs = torch.zeros(batch, 20, 5, requires_grad=True)
    exact_kl = torch.full((batch, 20, 5), 0.01, requires_grad=True)
    valid = torch.ones(batch, 20, dtype=torch.bool)
    components = torch.ones(batch, 20, 6)
    return {
        "current_log_probs": current_log_probs,
        "bc_log_probs": bc_log_probs,
        "reference_mean_kl": exact_kl,
        "rewards": base + delta,
        "valid_mask": valid,
        "component_scores": components.clone(),
        "base_rewards": base,
        "base_valid_mask": valid.clone(),
        "base_component_scores": components,
        "scene_buckets": buckets,
    }


def _config(mode):
    config = TransfuserConfig()
    config.grpo_training_mode = mode
    config.generation_policy_algorithm = "diffgrpo_mode_coverage"
    config.grpo_decoder_gradient_scope = "all_layers"
    config.grpo_decoder_layer0_lr_mult = 0.1
    config.inference_selector_source = "trajectory_relative_harm_v3"
    config.stage25_selector_checkpoint_path = "selector.ckpt"
    config.stage25_selector_calibration_path = "calibration.json"
    config.stage30_bucket_manifest_path = "buckets.json"
    config.diffgrpo_group_size = 20
    config.policy_loss_weight = 0.0
    config.kl_loss_weight = 0.0
    config.selector_generation_kl_weight = 0.0
    config.generation_kl_loss_weight = 0.0
    config.diffusion_truncation_timestep = 32
    config.diffusion_roll_timesteps = (32, 24, 16, 8, 0)
    config.weight_decay = 0.0
    return config


def test_stage30_trace_replays_all_modes_with_common_random_numbers(monkeypatch):
    head = _Head()
    calls = []
    transition = tuple(torch.full((2, 20, 8, 3), index + 1.0) for index in range(4))
    final_noise = torch.full((2, 20, 8, 3), 9.0)

    def fake_sample(_head, policy, initial, *args, **kwargs):
        calls.append((policy, initial.shape, kwargs))
        states = tuple(initial + index for index in range(5))
        actions = tuple(initial + index + 0.5 for index in range(5))
        classification = initial.new_zeros(initial.shape[:2])
        if kwargs.get("return_noise_bundle"):
            return states, actions, initial + 1, classification, {
                "transition_noises": transition,
                "final_noise": final_noise,
            }
        assert all(
            torch.equal(actual, wanted)
            for actual, wanted in zip(kwargs["transition_noises"], transition)
        )
        assert torch.equal(kwargs["final_noise"], final_noise)
        return states, actions, initial + 2, classification

    def fake_replay(_head, policy, states, actions, *args):
        assert states[0].shape[:2] == (2, 20)
        return states[0].new_zeros(2, 20, 5), states[0].new_zeros(2, 20)

    def fake_kl(_head, current, reference, states, *args):
        assert states[0].shape[:2] == (2, 20)
        return states[0].new_zeros(2, 20, 5)

    monkeypatch.setattr(diffusion_grpo, "_sample_full_chain_actions", fake_sample)
    monkeypatch.setattr(diffusion_grpo, "_replay_full_chain_actions", fake_replay)
    monkeypatch.setattr(diffusion_grpo, "_replay_reference_mean_kl", fake_kl)
    trace = diffusion_grpo.collect_mode_coverage_diffgrpo_trace(
        head,
        *(torch.zeros(2, 1, 1) for _ in range(3)),
        (1, 1),
        torch.zeros(2, 1),
        None,
    )
    assert [call[1] for call in calls] == [(2, 20, 8, 3), (2, 20, 8, 3)]
    assert trace["trajectories"].shape == (2, 20, 8, 3)
    assert trace["base_trajectories"].shape == (2, 20, 8, 3)
    assert trace["current_log_probs"].shape == (2, 20, 5)
    assert trace["bc_log_probs"].shape == (2, 20, 5)
    assert trace["sampled_chain_count"].item() == 20
    assert trace["replayed_chain_count"].item() == 20


def test_stage30_coverage_is_sign_preserving_and_top_five_only():
    delta = torch.linspace(-0.004, 0.005, 20).unsqueeze(0)
    result = compute_mode_coverage_constrained_objective(
        **_inputs(batch=1, delta=delta, buckets=torch.tensor([1])),
        constrained=False,
    )
    advantage = result["advantages"][0]
    assert torch.all(advantage[delta[0] > 0] >= 0)
    assert torch.all(advantage[delta[0] < 0] <= 0)
    assert result["top_tail_fraction"].item() == pytest.approx(0.25)


def test_stage30_boundary_margin_and_negative_multiplier():
    delta = torch.zeros(1, 20)
    delta[0, :5] = 0.0005
    delta[0, 5:10] = 0.002
    delta[0, 10:15] = -0.002
    result = compute_mode_coverage_constrained_objective(
        **_inputs(batch=1, delta=delta, buckets=torch.tensor([2])),
        constrained=True,
    )
    advantage = result["advantages"][0]
    assert torch.all(advantage[:5] == 0)
    assert torch.all(advantage[5:10] >= 0)
    assert torch.all(advantage[10:15] <= 0)
    assert result["mean_bc_weight"].item() == pytest.approx(0.2)
    assert result["mean_kl_weight"].item() == pytest.approx(0.5)


def test_stage30_mature_is_neutral_inside_preservation_band():
    delta = torch.full((1, 20), 0.001)
    inputs = _inputs(batch=1, delta=delta, buckets=torch.tensor([3]))
    result = compute_mode_coverage_constrained_objective(
        **inputs, constrained=True,
    )
    assert torch.all(result["advantages"] == 0)
    assert result["mean_bc_weight"].item() == pytest.approx(0.5)
    assert result["mean_kl_weight"].item() == pytest.approx(1.0)
    result["loss"].backward()
    assert inputs["current_log_probs"].grad is not None
    assert torch.all(inputs["current_log_probs"].grad == 0)
    assert inputs["bc_log_probs"].grad.abs().sum().item() > 0
    assert inputs["reference_mean_kl"].grad.abs().sum().item() > 0


def test_stage30_mature_positive_requires_all_components_no_worse():
    inputs = _inputs(
        batch=1,
        delta=torch.full((1, 20), 0.003),
        buckets=torch.tensor([3]),
    )
    inputs["component_scores"][0, 4, 5] = 0.99
    result = compute_mode_coverage_constrained_objective(
        **inputs, constrained=True
    )
    assert result["advantages"][0, 4].item() == 0.0
    allowed = torch.arange(20) != 4
    assert torch.all(result["advantages"][0, allowed] >= 0)


def test_stage30_safety_override_is_always_negative_two():
    inputs = _inputs(
        batch=4,
        delta=torch.full((4, 20), 0.01),
    )
    for scene in range(4):
        inputs["component_scores"][scene, scene, (0, 1, 3)[scene % 3]] = 0.9
    result = compute_mode_coverage_constrained_objective(
        **inputs, constrained=True
    )
    for scene in range(4):
        assert result["advantages"][scene, scene].item() == -2.0


def test_stage30_loss_has_finite_policy_bc_and_kl_gradients():
    inputs = _inputs()
    result = compute_mode_coverage_constrained_objective(
        **inputs, constrained=True
    )
    result["loss"].backward()
    for name in (
        "current_log_probs", "bc_log_probs", "reference_mean_kl"
    ):
        gradient = inputs[name].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()


def test_stage30_fails_closed_on_shapes_nonfinite_and_bucket_ids():
    bad = _inputs()
    bad["rewards"] = torch.zeros(4, 19)
    with pytest.raises(ValueError, match="rewards"):
        compute_mode_coverage_constrained_objective(**bad, constrained=True)
    bad = _inputs()
    bad["scene_buckets"][0] = 4
    with pytest.raises(ValueError, match="bucket"):
        compute_mode_coverage_constrained_objective(**bad, constrained=True)
    bad = _inputs()
    bad["reference_mean_kl"] = -torch.ones(4, 20, 5)
    with pytest.raises(FloatingPointError, match="non-negative"):
        compute_mode_coverage_constrained_objective(**bad, constrained=True)


def test_stage30_formal_modes_and_constants_are_locked():
    assert STAGE30_GRPO_MODES <= FORMAL_GRPO_MODES
    validate_formal_grpo_config(_config(COV))
    validate_formal_grpo_config(_config(MCC))
    drifted = _config(MCC)
    drifted.stage30_mature_positive_margin = 0.003
    with pytest.raises(ValueError, match="mature_positive_margin"):
        validate_formal_grpo_config(drifted)


def _gate_inputs():
    buckets = []
    folds = []
    for repeat in range(4):
        for fold in range(4):
            for bucket in range(4):
                buckets.append(bucket)
                folds.append(fold)
    buckets = np.asarray(buckets, dtype=np.int64)
    selected = np.where(buckets <= 1, 0.020, np.where(buckets == 2, 0.004, 0.001))
    size = selected.size
    vectors = {
        "selected": selected,
        "candidate_mean": np.full(size, 0.001),
        "oracle": np.full(size, 0.001),
        "fallback": np.full(size, 0.005),
    }
    components = {
        name: np.zeros(size)
        for name in check_grpo_stage30_cv.COMPONENTS
    }
    return {
        "vectors": vectors,
        "buckets": buckets,
        "folds": np.asarray(folds, dtype=np.int64),
        "noises": np.resize(np.asarray(check_grpo_stage30_cv.NOISES), size),
        "logs": [f"log-{index}" for index in range(size)],
        "component_values": components,
    }


def test_stage30_frozen_gate_accepts_only_mcc(monkeypatch):
    monkeypatch.setattr(
        check_grpo_stage30_cv,
        "bootstrap_whole_log",
        lambda values, logs: [0.0001, 0.02],
    )
    inputs = _gate_inputs()
    mcc = check_grpo_stage30_cv.summarize("MCC", 96, **inputs)
    cov = check_grpo_stage30_cv.summarize("COV", 96, **inputs)
    assert all(mcc["checks"].values())
    assert mcc["eligible"] is True
    assert cov["passes_numeric_gate"] is True
    assert cov["eligible"] is False


def test_stage30_frozen_gate_rejects_mature_or_component_regression(monkeypatch):
    monkeypatch.setattr(
        check_grpo_stage30_cv,
        "bootstrap_whole_log",
        lambda values, logs: [0.0001, 0.02],
    )
    inputs = _gate_inputs()
    mature = inputs["buckets"] == 3
    inputs["vectors"]["selected"][mature] = -0.001
    inputs["component_values"]["collision"][:] = -0.001
    result = check_grpo_stage30_cv.summarize("MCC", 96, **inputs)
    assert result["checks"]["mature_scene_gain_at_least_negative_0.0001"] is False
    assert result["checks"]["all_components_at_least_negative_0.0005"] is False
    assert result["eligible"] is False
