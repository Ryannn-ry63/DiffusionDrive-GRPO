import pytest

from navsim.agents.diffusiondrive.transfuser_agent import (
    FORMAL_GRPO_MODES,
    STAGE34_GRPO_MODES,
    validate_formal_grpo_config,
)
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_model_v2 import (
    resolve_mode_coverage_bucket_manifest,
)


def _config() -> TransfuserConfig:
    config = TransfuserConfig()
    config.grpo_training_mode = "stage34_mode_aligned_frontier_grpo"
    config.generation_policy_algorithm = "diffgrpo_mode_aligned_frontier"
    config.grpo_decoder_gradient_scope = "all_layers"
    config.grpo_decoder_layer0_lr_mult = 0.1
    config.inference_selector_source = "trajectory_relative_harm_v3"
    config.stage25_selector_checkpoint_path = "selector.ckpt"
    config.stage25_selector_calibration_path = "calibration.json"
    config.stage34_bucket_manifest_path = "buckets.json"
    config.policy_loss_weight = 0.0
    config.kl_loss_weight = 0.0
    config.selector_generation_kl_weight = 0.0
    config.generation_policy_loss_weight = 1.0
    config.generation_kl_loss_weight = 0.0
    config.generation_adaptive_kl_enabled = False
    config.diffgrpo_group_size = 20
    config.diffgrpo_bc_weight = 0.1
    config.diffgrpo_base_advantage_clip = 2.0
    config.diffgrpo_safety_regression_tolerance = 1e-6
    config.diffusion_truncation_timestep = 32
    config.diffusion_roll_timesteps = (32, 24, 16, 8, 0)
    config.diffusion_scheduler_num_inference_steps = 125
    config.weight_decay = 0.0
    return config


def test_stage34_formal_contract_accepts_only_frozen_route():
    config = _config()
    assert STAGE34_GRPO_MODES <= FORMAL_GRPO_MODES
    validate_formal_grpo_config(config)
    config.stage34_top1_negative_multiplier = 1.25
    with pytest.raises(ValueError, match="stage34_top1_negative_multiplier"):
        validate_formal_grpo_config(config)


def test_stage34_formal_contract_requires_bucket_and_selector():
    config = _config()
    config.stage34_bucket_manifest_path = ""
    with pytest.raises(ValueError, match="bucket manifest"):
        validate_formal_grpo_config(config)
    config = _config()
    config.stage25_selector_checkpoint_path = ""
    with pytest.raises(ValueError, match="checkpoint and calibration"):
        validate_formal_grpo_config(config)


def test_stage34_model_resolves_its_own_bucket_manifest():
    config = _config()
    config.stage30_bucket_manifest_path = "wrong-stage30-buckets.json"
    stage, path = resolve_mode_coverage_bucket_manifest(
        config, config.grpo_training_mode
    )
    assert stage == "Stage34"
    assert path == "buckets.json"
