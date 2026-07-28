from pathlib import Path

import pytest
import torch

from navsim.agents.diffusiondrive.transfuser_agent import (
    FORMAL_GRPO_MODES,
    STAGE29_GRPO_MODES,
    validate_formal_grpo_config,
)
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_loss import (
    compute_reference_anchored_headroom_objective,
)
from scripts.evaluation.audit_grpo_stage29_pilot_checkpoint import (
    AMENDMENT_SHA,
    assess_training_signal_health,
)


H_MODE = "stage29_public_headroom_hybrid"
HC_MODE = "stage29_public_headroom_conditional"


def _inputs(delta=None, base=None, batch=1):
    if base is None:
        base = torch.full((batch, 8), 0.60)
    if delta is None:
        row = torch.tensor(
            [[0.004, 0.003, 0.001, 0.0, -0.001, -0.003, -0.004, 0.005]]
        )
        delta = row.expand(batch, -1).clone()
    current = torch.zeros(batch, 8, 5, requires_grad=True)
    bc = torch.zeros(batch, 8, 5, requires_grad=True)
    kl = torch.full((batch, 8, 5), 0.01, requires_grad=True)
    valid = torch.ones(batch, 8, dtype=torch.bool)
    components = torch.ones(batch, 8, 6)
    base_components = torch.ones(batch, 8, 6)
    return {
        "current_log_probs": current,
        "bc_log_probs": bc,
        "reference_mean_kl": kl,
        "rewards": base + delta,
        "valid_mask": valid,
        "component_scores": components,
        "base_rewards": base,
        "base_valid_mask": valid.clone(),
        "base_component_scores": base_components,
    }


def _config(mode):
    config = TransfuserConfig()
    config.grpo_training_mode = mode
    config.generation_policy_algorithm = "diffgrpo_selected_set"
    config.grpo_decoder_gradient_scope = "all_layers"
    config.grpo_decoder_layer0_lr_mult = 0.1
    config.inference_selector_source = "trajectory_relative_harm_v3"
    config.stage25_selector_checkpoint_path = "selector.ckpt"
    config.stage25_selector_calibration_path = "calibration.json"
    config.policy_loss_weight = 0.0
    config.kl_loss_weight = 0.0
    config.selector_generation_kl_weight = 0.0
    config.generation_kl_loss_weight = 0.0
    config.diffusion_truncation_timestep = 32
    config.diffusion_roll_timesteps = (32, 24, 16, 8, 0)
    config.stage29_conditional_regularization = mode == HC_MODE
    return config


def test_stage29_all_worse_group_cannot_create_positive_advantage():
    inputs = _inputs(delta=torch.tensor([[
        -0.001, -0.002, -0.003, -0.004,
        -0.005, -0.006, -0.007, -0.008,
    ]]))
    result = compute_reference_anchored_headroom_objective(**inputs)
    assert torch.all(result["advantages"] <= 0)
    assert torch.any(result["advantages"] < 0)
    assert result["positive_fraction"].item() == 0


def test_stage29_paired_sign_is_never_reversed():
    delta = torch.tensor([[
        0.004, 0.003, 0.002, 0.001,
        -0.001, -0.002, -0.003, -0.004,
    ]])
    result = compute_reference_anchored_headroom_objective(
        **_inputs(delta=delta)
    )
    advantages = result["advantages"][0]
    assert torch.all(advantages[:4] >= 0)
    assert torch.all(advantages[4:] <= 0)
    assert torch.any(advantages[:4] > 0)
    assert torch.any(advantages[4:] < 0)


def test_stage29_exact_ties_explore_hard_but_not_mature_scenes():
    hard_base = torch.tensor([[
        0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75
    ]])
    hard = compute_reference_anchored_headroom_objective(
        **_inputs(delta=torch.zeros(1, 8), base=hard_base)
    )
    assert torch.any(hard["advantages"] > 0)
    assert torch.any(hard["advantages"] < 0)
    mature_base = torch.tensor([[
        0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98
    ]])
    mature = compute_reference_anchored_headroom_objective(
        **_inputs(delta=torch.zeros(1, 8), base=mature_base)
    )
    assert torch.all(mature["advantages"] == 0)
    assert mature["neutral_fraction"].item() == 1.0


def test_stage29_headroom_and_conditional_regularization_endpoints():
    base = torch.stack((
        torch.full((8,), 0.60),
        torch.full((8,), 0.95),
    ))
    delta = torch.tensor([
        [0.004, 0.003, 0.002, 0.001, -0.001, -0.002, -0.003, -0.004],
        [0.004, 0.003, 0.002, 0.001, -0.001, -0.002, -0.003, -0.004],
    ])
    result = compute_reference_anchored_headroom_objective(
        **_inputs(delta=delta, base=base, batch=2),
        conditional_regularization=True,
    )
    assert result["headroom_mean"].item() == pytest.approx(0.5)
    assert result["hard_scene_fraction"].item() == pytest.approx(0.5)
    assert result["mature_scene_fraction"].item() == pytest.approx(0.5)
    assert result["mean_bc_weight"].item() == pytest.approx(0.075)
    assert result["mean_kl_weight"].item() == pytest.approx(0.275)


def test_stage29_fixed_branch_keeps_fixed_regularization():
    result = compute_reference_anchored_headroom_objective(**_inputs())
    assert result["mean_bc_weight"].item() == pytest.approx(0.1)
    assert result["mean_kl_weight"].item() == pytest.approx(0.1)


def test_stage29_safety_regression_overrides_positive_reward_and_kl():
    inputs = _inputs(delta=torch.full((1, 8), 0.01))
    inputs["component_scores"][0, 3, 1] = 0.99
    result = compute_reference_anchored_headroom_objective(**inputs)
    assert result["advantages"][0, 3].item() == -2.0
    assert result["safety_override_fraction"].item() == pytest.approx(0.125)
    assert result["mean_kl_weight"].item() > 0.1


def test_stage29_policy_bc_and_kl_gradients_are_finite():
    inputs = _inputs()
    result = compute_reference_anchored_headroom_objective(**inputs)
    result["loss"].backward()
    for name in (
        "current_log_probs", "bc_log_probs", "reference_mean_kl"
    ):
        gradient = inputs[name].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum().item() > 0


def test_stage29_fails_closed_on_shape_nonfinite_and_constants():
    bad_shape = _inputs()
    bad_shape["rewards"] = torch.zeros(1, 7)
    with pytest.raises(ValueError, match="rewards"):
        compute_reference_anchored_headroom_objective(**bad_shape)
    nonfinite = _inputs()
    nonfinite["current_log_probs"] = nonfinite["current_log_probs"].clone()
    nonfinite["current_log_probs"][0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="finite"):
        compute_reference_anchored_headroom_objective(**nonfinite)
    negative_kl = _inputs()
    negative_kl["reference_mean_kl"] = -torch.ones(1, 8, 5)
    with pytest.raises(FloatingPointError, match="non-negative"):
        compute_reference_anchored_headroom_objective(**negative_kl)
    with pytest.raises(ValueError, match="constants"):
        compute_reference_anchored_headroom_objective(
            **_inputs(), headroom_low=0.9, headroom_high=0.75
        )


def test_stage29_formal_modes_and_constants_are_locked():
    assert STAGE29_GRPO_MODES <= FORMAL_GRPO_MODES
    validate_formal_grpo_config(_config(H_MODE))
    validate_formal_grpo_config(_config(HC_MODE))
    drifted = _config(HC_MODE)
    drifted.stage29_rank_weight = 0.75
    with pytest.raises(ValueError, match="stage29_rank_weight"):
        validate_formal_grpo_config(drifted)
    wrong_branch = _config(H_MODE)
    wrong_branch.stage29_conditional_regularization = True
    with pytest.raises(ValueError, match="conditional_regularization"):
        validate_formal_grpo_config(wrong_branch)


def test_stage29_modes_are_wired_through_all_training_boundaries():
    paths = (
        "navsim/agents/diffusiondrive/transfuser_model_v2.py",
        "navsim/agents/diffusiondrive/transfuser_loss.py",
        "navsim/planning/training/agent_lightning_module.py",
        "navsim/planning/script/run_training.py",
    )
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        assert H_MODE in text
        assert HC_MODE in text


def test_stage29_amended_dense_scene_health_gate_is_semantically_correct():
    assert AMENDMENT_SHA == (
        "941eef5f370f90ad9e5b9c7ed7a1dbaa12a971f30e79a211c446098f576f3bd9"
    )
    h = assess_training_signal_health(0.984375, 0.342041015625, 0.30755615234375)
    hc = assess_training_signal_health(0.98486328125, 0.4144287109375, 0.27655029296875)
    assert h["passed"] and hc["passed"]
    assert "active_scene_fraction_at_least_0.15" in h["checks"]
    assert not any("0.95" in name for name in h["checks"])
    assert not assess_training_signal_health(0.149, 0.2, 0.4)["passed"]
    assert not assess_training_signal_health(0.8, 0.009, 0.4)["passed"]
    assert not assess_training_signal_health(0.8, 0.1, 0.851)["passed"]
    with pytest.raises(RuntimeError, match="invalid training-signal"):
        assess_training_signal_health(float("nan"), 0.1, 0.4)


def test_stage29_fold4_shell_grid_matches_frozen_namespaces():
    for path in (
        "run_stage29_fold4_8gpu.sh",
        "scripts/evaluation/run_diffusiondrive_grpo_stage29_fold4_cell.sh",
    ):
        text = Path(path).read_text(encoding="utf-8")
        assert "NOISES=(20261211 20261212)" in text
        assert "SYSTEMS=(P A3 H1 H2 H3 H4 HC1 HC2 HC3 HC4)" in text
        assert "NOISES=(20261111 20261112)" not in text
