import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from navsim.agents.diffusiondrive.transfuser_agent import (
    FORMAL_GRPO_MODES,
    STAGE28_GRPO_MODES,
    TransfuserAgent,
    validate_formal_grpo_config,
    validate_stage28_exploration_authorization,
)
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_loss import (
    compute_public_paired_uplift_objective,
)


MULTI_MODE = "stage28_public_paired_uplift_multi"
EXPLORE_MODE = "stage28_public_paired_uplift_explore"
AUTHORIZATION = Path(
    "artifacts/grpo_stage28/authorization/s_public_training_only.json"
)
PUBLIC_CALIBRATION = Path(
    "artifacts/grpo_stage27/selectors/public/calibration.json"
)


def _objective_inputs(delta=None, base=None):
    if base is None:
        base = torch.full((1, 8), 0.6)
    if delta is None:
        delta = torch.tensor(
            [[0.004, 0.003, 0.001, 0.0, -0.001, -0.003, -0.004, 0.005]]
        )
    current = torch.zeros(1, 8, 5, requires_grad=True)
    bc = torch.zeros(1, 8, 5, requires_grad=True)
    kl = torch.full((1, 8, 5), 0.01, requires_grad=True)
    valid = torch.ones(1, 8, dtype=torch.bool)
    components = torch.ones(1, 8, 6)
    base_components = torch.ones(1, 8, 6)
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


def _stage28_config(mode=MULTI_MODE):
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
    config.diffgrpo_paired_positive_margin = 0.002
    config.diffgrpo_paired_negative_margin = 0.002
    config.diffgrpo_paired_mature_negative_margin = 0.0005
    config.diffgrpo_paired_bootstrap_advantage_weight = 0.0
    config.stage28_training_selector_role = (
        "explore" if mode == EXPLORE_MODE else "multi"
    )
    config.stage28_exploration_authorization_path = (
        str(AUTHORIZATION) if mode == EXPLORE_MODE else ""
    )
    return config


def test_stage28_all_worse_group_cannot_create_positive_advantage():
    inputs = _objective_inputs(delta=torch.tensor(
        [[-0.003, -0.004, -0.005, -0.006, -0.007, -0.008, -0.009, -0.010]]
    ))
    result = compute_public_paired_uplift_objective(**inputs)
    assert torch.all(result["advantages"] < 0)
    assert result["positive_fraction"].item() == 0
    assert result["regular_negative_fraction"].item() == 1


def test_stage28_positive_and_neutral_deadzone_are_absolute():
    inputs = _objective_inputs()
    result = compute_public_paired_uplift_objective(**inputs)
    advantages = result["advantages"][0]
    assert torch.all(advantages[[0, 1, 7]] > 0)
    assert torch.all(advantages[[2, 3, 4]] == 0)
    assert torch.all(advantages[[5, 6]] < 0)


def test_stage28_mature_light_regression_is_negative_but_nonmature_is_neutral():
    base = torch.tensor([[0.80, 0.60, 0.80, 0.60, 0.80, 0.60, 0.80, 0.60]])
    inputs = _objective_inputs(delta=torch.full((1, 8), -0.001), base=base)
    result = compute_public_paired_uplift_objective(**inputs)
    advantages = result["advantages"][0]
    assert torch.all(advantages[::2] < 0)
    assert torch.all(advantages[1::2] == 0)
    assert result["mature_negative_fraction"].item() == pytest.approx(0.5)


def test_stage28_safety_regression_overrides_positive_reward():
    inputs = _objective_inputs(delta=torch.full((1, 8), 0.01))
    inputs["component_scores"][0, 3, 1] = 0.99
    result = compute_public_paired_uplift_objective(**inputs)
    assert result["advantages"][0, 3].item() == -2.0
    assert result["safety_override_fraction"].item() == pytest.approx(0.125)


def test_stage28_neutral_policy_zero_but_bc_and_kl_remain_active():
    inputs = _objective_inputs(delta=torch.zeros(1, 8))
    result = compute_public_paired_uplift_objective(**inputs)
    assert result["policy_loss"].item() == 0
    assert result["reference_kl_loss"].item() > 0
    result["loss"].backward()
    assert inputs["current_log_probs"].grad is not None
    assert inputs["current_log_probs"].grad.abs().sum().item() == 0
    assert inputs["bc_log_probs"].grad.abs().sum().item() > 0
    assert inputs["reference_mean_kl"].grad.abs().sum().item() > 0


def test_stage28_objective_fails_closed_on_shape_nonfinite_and_negative_kl():
    inputs = _objective_inputs()
    bad_shape = dict(inputs)
    bad_shape["rewards"] = torch.zeros(1, 7)
    with pytest.raises(ValueError, match="rewards"):
        compute_public_paired_uplift_objective(**bad_shape)
    nonfinite = _objective_inputs()
    nonfinite["current_log_probs"] = nonfinite["current_log_probs"].clone()
    nonfinite["current_log_probs"][0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="finite"):
        compute_public_paired_uplift_objective(**nonfinite)
    negative_kl = _objective_inputs()
    negative_kl["reference_mean_kl"] = -torch.ones(1, 8, 5)
    with pytest.raises(FloatingPointError, match="non-negative"):
        compute_public_paired_uplift_objective(**negative_kl)


def test_stage28_formal_modes_and_constants_are_locked():
    assert STAGE28_GRPO_MODES <= FORMAL_GRPO_MODES
    for mode in (MULTI_MODE, EXPLORE_MODE):
        validate_formal_grpo_config(_stage28_config(mode))
    config = _stage28_config()
    config.diffgrpo_paired_positive_margin = 0.003
    with pytest.raises(ValueError, match="positive_margin"):
        validate_formal_grpo_config(config)


def test_stage28_exploration_authorization_passes_and_hash_drift_fails(tmp_path):
    config = _stage28_config(EXPLORE_MODE)
    payload = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    validate_stage28_exploration_authorization(
        config, payload["selector_checkpoint_sha256"], PUBLIC_CALIBRATION
    )
    payload["deployment_forbidden"] = False
    drifted = tmp_path / "authorization.json"
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    config.stage28_exploration_authorization_path = str(drifted)
    with pytest.raises(RuntimeError, match="deployment_forbidden"):
        validate_stage28_exploration_authorization(
            config, payload["selector_checkpoint_sha256"], PUBLIC_CALIBRATION
        )


def test_stage28_explore_mode_is_impossible_to_run_in_eval():
    agent = TransfuserAgent.__new__(TransfuserAgent)
    nn.Module.__init__(agent)
    agent._config = _stage28_config(EXPLORE_MODE)
    agent.eval()
    with pytest.raises(RuntimeError, match="training-only"):
        agent.forward({})


def test_stage28_modes_are_wired_through_all_training_boundaries():
    paths = (
        "navsim/agents/diffusiondrive/transfuser_model_v2.py",
        "navsim/agents/diffusiondrive/transfuser_loss.py",
        "navsim/planning/training/agent_lightning_module.py",
        "navsim/planning/script/run_training.py",
    )
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        assert MULTI_MODE in text
        assert EXPLORE_MODE in text
