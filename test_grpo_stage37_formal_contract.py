import pytest

from navsim.agents.diffusiondrive.stage37_contract import (
    STAGE37_ALLOWED_INITIAL_SHA256,
)
from navsim.agents.diffusiondrive.transfuser_agent import (
    FORMAL_GRPO_MODES,
    STAGE37_GRPO_MODES,
    validate_formal_grpo_config,
)
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_model_v2 import (
    resolve_mode_coverage_bucket_manifest,
)


def _config() -> TransfuserConfig:
    config = TransfuserConfig()
    config.grpo_training_mode = "stage37_bistate_projected_deployment_grpo"
    config.generation_policy_algorithm = "diffgrpo_bistate_projected_deployment"
    config.grpo_decoder_gradient_scope = "all_layers"
    config.grpo_decoder_layer0_lr_mult = 0.1
    config.inference_selector_source = "trajectory_relative_harm_v3"
    config.stage25_selector_checkpoint_path = "selector.ckpt"
    config.stage25_selector_calibration_path = "calibration.json"
    config.stage37_bucket_manifest_path = "buckets.json"
    config.policy_loss_weight = 0.0
    config.kl_loss_weight = 0.0
    config.selector_generation_kl_weight = 0.0
    config.generation_policy_loss_weight = 1.0
    config.generation_kl_loss_weight = 0.0
    config.generation_adaptive_kl_enabled = False
    config.diffgrpo_group_size = 8
    config.diffgrpo_bc_weight = 0.1
    config.diffgrpo_base_advantage_clip = 2.0
    config.diffgrpo_safety_regression_tolerance = 1e-6
    config.diffusion_truncation_timestep = 32
    config.diffusion_roll_timesteps = (32, 24, 16, 8, 0)
    config.diffusion_scheduler_num_inference_steps = 125
    config.weight_decay = 0.0
    return config


def test_stage37_formal_contract_accepts_only_frozen_projected_route():
    config = _config()
    assert STAGE37_GRPO_MODES <= FORMAL_GRPO_MODES
    validate_formal_grpo_config(config)
    config.stage37_projection_recovery_coefficient = 0.5
    with pytest.raises(
        ValueError, match="stage37_projection_recovery_coefficient"
    ):
        validate_formal_grpo_config(config)


def test_stage37_requires_own_bucket_and_stage25_generator_selector():
    config = _config()
    config.stage37_bucket_manifest_path = ""
    with pytest.raises(ValueError, match="bucket manifest"):
        validate_formal_grpo_config(config)
    config = _config()
    config.stage25_selector_checkpoint_path = ""
    with pytest.raises(ValueError, match="checkpoint and calibration"):
        validate_formal_grpo_config(config)


def test_stage37_resolves_own_bucket_and_public_only_initializer():
    config = _config()
    config.stage30_bucket_manifest_path = "wrong-stage30.json"
    config.stage36_bucket_manifest_path = "wrong-stage36.json"
    stage, path = resolve_mode_coverage_bucket_manifest(
        config, config.grpo_training_mode
    )
    assert stage == "Stage37"
    assert path == "buckets.json"
    assert STAGE37_ALLOWED_INITIAL_SHA256 == {
        "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
    }
