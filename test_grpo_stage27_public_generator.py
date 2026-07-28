from pathlib import Path

import pytest

from navsim.agents.diffusiondrive.transfuser_agent import (
    FORMAL_BASE_SHA256,
    FORMAL_GRPO_MODES,
    STAGE27_PUBLIC_BASE_SHA256,
    validate_formal_grpo_config,
)
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig


STAGE27_MODE = "stage27_public_diffgrpo_selected_set"


def _stage27_config() -> TransfuserConfig:
    config = TransfuserConfig()
    config.grpo_training_mode = STAGE27_MODE
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
    return config


def test_stage27_public_mode_has_independent_frozen_base_identity():
    assert STAGE27_MODE in FORMAL_GRPO_MODES
    assert STAGE27_PUBLIC_BASE_SHA256 == (
        "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
    )
    assert FORMAL_BASE_SHA256 == (
        "59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d"
    )
    assert STAGE27_PUBLIC_BASE_SHA256 != FORMAL_BASE_SHA256


def test_stage27_public_formal_route_is_locked():
    config = _stage27_config()
    validate_formal_grpo_config(config)
    config.inference_selector_source = "trajectory_oof"
    with pytest.raises(ValueError, match="frozen selector sources"):
        validate_formal_grpo_config(config)


def test_stage27_public_formal_route_rejects_hyperparameter_drift():
    config = _stage27_config()
    config.diffgrpo_group_size = 7
    with pytest.raises(ValueError, match="group_size"):
        validate_formal_grpo_config(config)


def test_stage27_mode_is_wired_through_all_training_boundaries():
    paths = (
        "navsim/agents/diffusiondrive/transfuser_model_v2.py",
        "navsim/agents/diffusiondrive/transfuser_loss.py",
        "navsim/planning/training/agent_lightning_module.py",
        "navsim/planning/script/run_training.py",
    )
    for path in paths:
        assert STAGE27_MODE in Path(path).read_text(encoding="utf-8")
