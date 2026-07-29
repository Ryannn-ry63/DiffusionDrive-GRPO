from typing import Any, List, Dict, Optional, Union
from pathlib import Path

import copy
import json
import math
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.diffusion_grpo import (
    AdaptiveKLController,
    file_sha256,
)
from navsim.agents.diffusiondrive.paired_residual_lora import (
    attach_stage22_lora,
)
from navsim.agents.diffusiondrive.stage34_contract import (
    STAGE34_LEGACY_STAGE30_KEYS,
    STAGE34_OBJECTIVE_REVISION,
    STAGE34_PLAN_SHA256,
    STAGE34_ROUTE_EXPECTED,
)
from navsim.agents.diffusiondrive.stage35_contract import (
    STAGE35_LEGACY_STAGE31_KEYS,
    STAGE35_ALLOWED_INITIAL_SHA256,
    STAGE35_ROUTE_EXPECTED,
)
from navsim.agents.diffusiondrive.stage36_contract import (
    STAGE36_LEGACY_STAGE31_KEYS,
    STAGE36_ALLOWED_INITIAL_SHA256,
    STAGE36_ROUTE_EXPECTED,
)
from navsim.agents.diffusiondrive.stage37_contract import (
    STAGE37_LEGACY_STAGE31_KEYS,
    STAGE37_ALLOWED_INITIAL_SHA256,
    STAGE37_ROUTE_EXPECTED,
)
from navsim.agents.diffusiondrive.stage38_contract import (
    STAGE38_LEGACY_STAGE31_KEYS,
    STAGE38_ALLOWED_INITIAL_SHA256,
    STAGE38_ROUTE_EXPECTED,
)
from navsim.agents.diffusiondrive.stage39_contract import (
    STAGE39_PUBLIC_SHA256,
    STAGE39_ROUTE_EXPECTED,
    STAGE39_TRAINING_MODES,
)

from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel as TransfuserModel

from navsim.agents.diffusiondrive.transfuser_callback import TransfuserCallback 
from navsim.agents.diffusiondrive.transfuser_loss import transfuser_loss
from navsim.agents.diffusiondrive.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.diffusiondrive.modules.scheduler import WarmupCosLR
from omegaconf import DictConfig, OmegaConf, open_dict
import torch.optim as optim
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig
def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)


STAGE9_GRPO_MODES = {"selector_group", "generation_group"}
STAGE10_GRPO_MODES = {"generation_group_adaptive"}
STAGE16_GRPO_MODES = {"diffgrpo_full_chain"}
STAGE19_GRPO_MODES = {"diffgrpo_selected_anchor"}
STAGE21_GRPO_MODES = {"diffgrpo_selected_anchor_base_preserve"}
STAGE22_GRPO_MODES = {"diffgrpo_paired_residual"}
STAGE23_GRPO_MODES = {"diffgrpo_selected_set"}
STAGE27_GRPO_MODES = {"stage27_public_diffgrpo_selected_set"}
STAGE28_GRPO_MODES = {
    "stage28_public_paired_uplift_multi",
    "stage28_public_paired_uplift_explore",
}
STAGE29_GRPO_MODES = {
    "stage29_public_headroom_hybrid",
    "stage29_public_headroom_conditional",
}
STAGE30_GRPO_MODES = {
    "stage30_public_mode_coverage",
    "stage30_public_mode_coverage_constrained",
}
STAGE31_GRPO_MODES = {
    "stage31_public_deployed_pair",
    "stage31_public_deployed_frontier",
}
STAGE32_GRPO_MODES = {
    "stage32_public_deployed_extended",
    "stage32_selector_aware_frontier",
}
STAGE33_GRPO_MODES = {"stage33_cdc_grpo"}
STAGE34_GRPO_MODES = {"stage34_mode_aligned_frontier_grpo"}
STAGE35_GRPO_MODES = {"stage35_nested_counterfactual_deployment_grpo"}
STAGE36_GRPO_MODES = {"stage36_reference_gated_tail_ncd_grpo"}
STAGE39_GRPO_MODES = STAGE39_TRAINING_MODES
STAGE37_GRPO_MODES = {"stage37_bistate_projected_deployment_grpo"}
STAGE38_GRPO_MODES = {"stage38_elite_set_counterfactual_repair_grpo"}
STAGE32_PLAN_SHA256 = "5c5f3b148c8d03fe3c714558314ebd895a8f143b94fd10fb9bdf8c4ad61cefcd"
STAGE32_SCF_OBJECTIVE_REVISION = "mean_normalized_bc_kl_v2"
STAGE33_PLAN_SHA256 = "0ace8d8959360a15cfac8f45bc2c6e19898c09ad1a04aa7b5e5dcc2fb36283ff"
STAGE33_CDC_OBJECTIVE_REVISION = "cdc_deployment_credit_v1"
SELECTED_SET_GRPO_MODES = (
    STAGE23_GRPO_MODES | STAGE27_GRPO_MODES | STAGE28_GRPO_MODES
    | STAGE29_GRPO_MODES | STAGE31_GRPO_MODES | STAGE32_GRPO_MODES
    | STAGE33_GRPO_MODES | STAGE39_GRPO_MODES
)
FORMAL_GRPO_MODES = (
    STAGE9_GRPO_MODES
    | STAGE10_GRPO_MODES
    | STAGE16_GRPO_MODES
    | STAGE19_GRPO_MODES
    | STAGE21_GRPO_MODES
    | STAGE22_GRPO_MODES
    | STAGE23_GRPO_MODES
    | STAGE27_GRPO_MODES
    | STAGE28_GRPO_MODES
    | STAGE39_GRPO_MODES
    | STAGE29_GRPO_MODES
    | STAGE30_GRPO_MODES
    | STAGE31_GRPO_MODES
    | STAGE32_GRPO_MODES
    | STAGE33_GRPO_MODES
    | STAGE34_GRPO_MODES
    | STAGE35_GRPO_MODES | STAGE36_GRPO_MODES | STAGE37_GRPO_MODES
    | STAGE38_GRPO_MODES
)
FORMAL_BASE_SHA256 = (
    "59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d"
)
STAGE27_PUBLIC_BASE_SHA256 = (
    "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
)
STAGE28_MULTI_SELECTOR_SHA256 = (
    "023d7b6b77bb8fa2dc3e779849f3716688abf0796b019416494c80b29615b691"
)
STAGE28_MULTI_CALIBRATION_SHA256 = (
    "fec2a10573e913e4452f8eea92e6334fcd5e6f1104e6244876770973a1f41990"
)
STAGE28_EXPLORE_SELECTOR_SHA256 = (
    "9a7a202830f0dcd87c5aeac2f91a3b2dc3958584cc965ecf4b7048f56ceca91b"
)
STAGE28_EXPLORE_CALIBRATION_SHA256 = (
    "ab1931fc96c00288c32b8d2fa484c2114dd5cd66ef1ccf05693d8fba8885660a"
)
STAGE28_PLAN_SHA256 = (
    "217ef4b31cdc830e0e8fad7aa6d0f53d05c07fffcc381b00ad57f13912bc287b"
)
STAGE17_GENERATOR_SHA256 = (
    "3a7641d4ac2a9d644eda4cb945d1902d4ac4bebfdad09b156676dc0cbed94e23"
)


def validate_frozen_policy_state(policy: nn.Module, expected_state: Dict[str, torch.Tensor]) -> None:
    """Fail closed if a supposedly frozen policy differs from its base state."""
    actual_state = policy.state_dict()
    if set(actual_state) != set(expected_state):
        raise RuntimeError("Frozen reference policy state keys differ from base")
    for name, expected in expected_state.items():
        actual = actual_state[name].detach().cpu()
        if not torch.equal(actual, expected):
            raise RuntimeError(
                f"Frozen reference policy changed relative to base: {name}"
            )
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("Frozen reference policy has trainable parameters")


def validate_stage28_exploration_authorization(
    config: TransfuserConfig,
    selector_sha256: str,
    calibration_path: Path,
) -> None:
    """Authorize the failed S-public calibration for training exploration only."""
    if str(getattr(config, "grpo_training_mode", "")) != (
        "stage28_public_paired_uplift_explore"
    ):
        raise RuntimeError("Stage28 exploration authorization used outside explore mode")
    authorization_path = Path(str(
        getattr(config, "stage28_exploration_authorization_path", "")
    ))
    if not authorization_path.is_file():
        raise FileNotFoundError("Stage28 exploration authorization is missing")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    calibration_sha256 = file_sha256(calibration_path)
    expected = {
        "schema_version": 1,
        "stage": 28,
        "purpose": "training_selector_exploration",
        "training_only": True,
        "deployment_forbidden": True,
        "public_checkpoint_sha256": STAGE27_PUBLIC_BASE_SHA256,
        "plan_sha256": STAGE28_PLAN_SHA256,
        "selector_checkpoint_sha256": STAGE28_EXPLORE_SELECTOR_SHA256,
        "calibration_sha256": STAGE28_EXPLORE_CALIBRATION_SHA256,
        "calibration_passed": False,
    }
    for name, wanted in expected.items():
        if authorization.get(name) != wanted:
            raise RuntimeError(
                f"Stage28 exploration authorization drifted for {name}"
            )
    plan_path = Path(str(authorization.get("plan", "")))
    selector_path = Path(str(authorization.get("selector_checkpoint", "")))
    authorized_calibration = Path(str(authorization.get("calibration", "")))
    if not plan_path.is_file() or file_sha256(plan_path) != STAGE28_PLAN_SHA256:
        raise RuntimeError("Stage28 exploration authorization plan SHA mismatch")
    if (
        not selector_path.is_file()
        or file_sha256(selector_path) != selector_sha256
        or selector_sha256 != STAGE28_EXPLORE_SELECTOR_SHA256
    ):
        raise RuntimeError("Stage28 exploration authorization selector SHA mismatch")
    if (
        not authorized_calibration.is_file()
        or authorized_calibration.resolve() != calibration_path.resolve()
        or calibration_sha256 != STAGE28_EXPLORE_CALIBRATION_SHA256
    ):
        raise RuntimeError("Stage28 exploration authorization calibration mismatch")


def _validate_stage39_config(config: TransfuserConfig) -> None:
    """Validate Stage39 without inheriting legacy Stage31 keys."""
    expected = {
        "grpo_decoder_gradient_scope": "all_layers",
        "grpo_reward_mode": "pdms",
        "grpo_scene_weight_mode": "uniform",
        "generation_policy_algorithm": "diffgrpo_non_destructive_challenger",
        "generation_advantage_mode": "stage39_branch_objective",
        "generation_mode_weighting": "uniform",
        "generation_trust_projection_mode": "none",
        "grpo_rollouts_per_mode": 1,
        "grpo_old_policy_sync_steps": 32,
        "generation_policy_loss_weight": 1.0,
        "generation_kl_loss_weight": 0.0,
        "generation_adaptive_kl_enabled": False,
        "policy_loss_weight": 0.0,
        "kl_loss_weight": 0.0,
        "selection_entropy_weight": 0.0,
        "selection_exploration_floor": 0.0,
        "selection_rank_loss_weight": 0.0,
        "selector_consistency_kl_weight": 0.0,
        **STAGE39_ROUTE_EXPECTED,
    }
    if not str(getattr(config, "stage39_bucket_manifest_path", "")):
        raise ValueError("formal Stage39 requires its bucket manifest")
    for name, wanted in expected.items():
        actual = getattr(config, name, None)
        if isinstance(wanted, tuple) and actual is not None:
            actual = tuple(actual)
        matches = (
            math.isclose(float(actual), wanted, rel_tol=0.0, abs_tol=1e-12)
            if isinstance(wanted, float) else actual == wanted
        )
        if not matches:
            raise ValueError(
                f"formal Stage39 requires {name}={wanted!r}; got {actual!r}"
            )

def validate_formal_grpo_config(config: TransfuserConfig) -> None:
    """Fail closed if a registered Stage-9/10 objective drifts."""
    mode = str(getattr(config, "grpo_training_mode", ""))
    if mode not in FORMAL_GRPO_MODES:
        return
    if mode in STAGE39_GRPO_MODES:
        return _validate_stage39_config(config)
    expected = {
        "grpo_decoder_gradient_scope": "all_layers",
        "grpo_reward_mode": "pdms",
        "grpo_scene_weight_mode": "uniform",
        "selection_behavior_weighting": "old_policy",
        "generation_advantage_mode": "group_zscore",
        "generation_mode_weighting": "uniform",
        "generation_trust_projection_mode": "none",
        "grpo_rollouts_per_mode": 1,
        "grpo_old_policy_sync_steps": 32,
        "grpo_priority_manifest_path": "",
        "grpo_priority_sample_fraction": 0.0,
        "selection_entropy_weight": 0.0,
        "selection_exploration_floor": 0.0,
        "selection_rank_loss_weight": 0.0,
        "selector_consistency_kl_weight": 0.0,
        "grpo_clip_ratio": 0.2,
    }
    route_expected = {
        "selector_group": {
            "policy_loss_weight": 1.0,
            "kl_loss_weight": 0.01,
            "selector_generation_kl_weight": 0.1,
            "generation_policy_loss_weight": 0.0,
            "generation_kl_loss_weight": 0.0,
            "generation_adaptive_kl_enabled": False,
        },
        "generation_group": {
            "policy_loss_weight": 0.0,
            "kl_loss_weight": 0.0,
            "selector_generation_kl_weight": 0.0,
            "generation_policy_loss_weight": 1.0,
            "generation_kl_loss_weight": 0.1,
            "generation_adaptive_kl_enabled": False,
        },
        "generation_group_adaptive": {
            "policy_loss_weight": 0.0,
            "kl_loss_weight": 0.0,
            "selector_generation_kl_weight": 0.0,
            "generation_policy_loss_weight": 1.0,
            "generation_kl_loss_weight": 0.1,
            "generation_adaptive_kl_enabled": True,
            "generation_kl_initial_coefficient": 0.1,
            "generation_kl_min_coefficient": 0.1,
            "generation_kl_max_coefficient": 100.0,
            "generation_kl_target": 1e-4,
            "generation_kl_hard_limit": 2.5e-4,
            "generation_kl_window": 32,
            "generation_kl_update_interval": 8,
            "generation_kl_adaptation_factor": 2.0,
            "generation_kl_lower_ratio": 2.0 / 3.0,
            "generation_kl_upper_ratio": 1.5,
            "generation_kl_hard_limit_patience": 2,
        },
        "diffgrpo_full_chain": {
            "policy_loss_weight": 0.0,
            "kl_loss_weight": 0.0,
            "selector_generation_kl_weight": 0.0,
            "generation_policy_loss_weight": 1.0,
            "generation_kl_loss_weight": 0.0,
            "generation_adaptive_kl_enabled": False,
            "generation_policy_algorithm": "diffgrpo_full_chain",
            "diffgrpo_bc_weight": 0.1,
            "diffgrpo_step_discount": 0.6,
            "diffgrpo_logprob_reduction": "mean",
            "inference_selector_source": "reference",
            "diffusion_truncation_timestep": 32,
            "diffusion_roll_timesteps": (32, 24, 16, 8, 0),
            "diffusion_scheduler_num_inference_steps": 125,
        },
        "diffgrpo_selected_anchor": {
            "policy_loss_weight": 0.0,
            "kl_loss_weight": 0.0,
            "selector_generation_kl_weight": 0.0,
            "generation_policy_loss_weight": 1.0,
            "generation_kl_loss_weight": 0.0,
            "generation_adaptive_kl_enabled": False,
            "generation_policy_algorithm": "diffgrpo_selected_anchor",
            "diffgrpo_bc_weight": 0.1,
            "diffgrpo_step_discount": 0.6,
            "diffgrpo_logprob_reduction": "mean",
            "diffgrpo_group_size": 8,
            "inference_selector_source": "reference",
            "diffusion_truncation_timestep": 32,
            "diffusion_roll_timesteps": (32, 24, 16, 8, 0),
            "diffusion_scheduler_num_inference_steps": 125,
        },
        "diffgrpo_selected_anchor_base_preserve": {
            "policy_loss_weight": 0.0,
            "kl_loss_weight": 0.0,
            "selector_generation_kl_weight": 0.0,
            "generation_policy_loss_weight": 1.0,
            "generation_kl_loss_weight": 0.0,
            "generation_adaptive_kl_enabled": False,
            "generation_policy_algorithm": "diffgrpo_selected_anchor_base_preserve",
            "diffgrpo_bc_weight": 0.1,
            "diffgrpo_bc_max_weight": 0.3,
            "diffgrpo_base_margin": 0.01,
            "diffgrpo_base_scale": 0.10,
            "diffgrpo_base_advantage_clip": 2.0,
            "diffgrpo_safety_regression_tolerance": 1e-6,
            "diffgrpo_step_discount": 0.6,
            "diffgrpo_logprob_reduction": "mean",
            "diffgrpo_group_size": 8,
            "inference_selector_source": "reference",
            "diffusion_truncation_timestep": 32,
            "diffusion_roll_timesteps": (32, 24, 16, 8, 0),
            "diffusion_scheduler_num_inference_steps": 125,
        },
        "diffgrpo_paired_residual": {
            "policy_loss_weight": 0.0,
            "kl_loss_weight": 0.0,
            "selector_generation_kl_weight": 0.0,
            "generation_policy_loss_weight": 1.0,
            "generation_kl_loss_weight": 0.0,
            "generation_adaptive_kl_enabled": False,
            "generation_policy_algorithm": "diffgrpo_paired_residual",
            "diffgrpo_step_discount": 0.6,
            "diffgrpo_logprob_reduction": "mean",
            "diffgrpo_group_size": 8,
            "diffgrpo_base_scale": 0.10,
            "diffgrpo_base_advantage_clip": 2.0,
            "diffgrpo_safety_regression_tolerance": 1e-6,
            "diffgrpo_paired_positive_margin": 0.01,
            "diffgrpo_paired_negative_margin": 0.01,
            "diffgrpo_paired_mature_negative_margin": 0.002,
            "diffgrpo_paired_mature_reward_threshold": 0.75,
            "diffgrpo_paired_mature_negative_multiplier": 2.0,
            "diffgrpo_paired_regular_kl_weight": 0.1,
            "diffgrpo_paired_mature_kl_weight": 0.5,
            "diffgrpo_paired_bootstrap_advantage_weight": 0.25,
            "diffgrpo_lora_rank": 8,
            "diffgrpo_lora_alpha": 8.0,
            "weight_decay": 0.0,
            "inference_selector_source": "reference",
            "diffusion_truncation_timestep": 32,
            "diffusion_roll_timesteps": (32, 24, 16, 8, 0),
            "diffusion_scheduler_num_inference_steps": 125,
        },
        "diffgrpo_selected_set": {
            "policy_loss_weight": 0.0,
            "kl_loss_weight": 0.0,
            "selector_generation_kl_weight": 0.0,
            "generation_policy_loss_weight": 1.0,
            "generation_kl_loss_weight": 0.0,
            "generation_adaptive_kl_enabled": False,
            "generation_policy_algorithm": "diffgrpo_selected_set",
            "diffgrpo_bc_weight": 0.1,
            "diffgrpo_step_discount": 0.6,
            "diffgrpo_logprob_reduction": "mean",
            "diffgrpo_group_size": 8,
            "diffgrpo_safety_regression_tolerance": 1e-6,
            "diffgrpo_paired_regular_kl_weight": 0.1,
            "diffgrpo_paired_mature_kl_weight": 0.5,
            "inference_selector_source": "trajectory_oof",
            "diffusion_truncation_timestep": 32,
            "diffusion_roll_timesteps": (32, 24, 16, 8, 0),
            "diffusion_scheduler_num_inference_steps": 125,
        },
        "stage27_public_diffgrpo_selected_set": {
            "policy_loss_weight": 0.0,
            "kl_loss_weight": 0.0,
            "selector_generation_kl_weight": 0.0,
            "generation_policy_loss_weight": 1.0,
            "generation_kl_loss_weight": 0.0,
            "generation_adaptive_kl_enabled": False,
            "generation_policy_algorithm": "diffgrpo_selected_set",
            "diffgrpo_bc_weight": 0.1,
            "diffgrpo_step_discount": 0.6,
            "diffgrpo_logprob_reduction": "mean",
            "diffgrpo_group_size": 8,
            "diffgrpo_safety_regression_tolerance": 1e-6,
            "diffgrpo_paired_regular_kl_weight": 0.1,
            "diffgrpo_paired_mature_kl_weight": 0.5,
            "inference_selector_source": "trajectory_relative_harm_v3",
            "diffusion_truncation_timestep": 32,
            "diffusion_roll_timesteps": (32, 24, 16, 8, 0),
            "diffusion_scheduler_num_inference_steps": 125,
        },
        "stage28_public_paired_uplift_multi": {
            "policy_loss_weight": 0.0,
            "kl_loss_weight": 0.0,
            "selector_generation_kl_weight": 0.0,
            "generation_policy_loss_weight": 1.0,
            "generation_kl_loss_weight": 0.0,
            "generation_adaptive_kl_enabled": False,
            "generation_policy_algorithm": "diffgrpo_selected_set",
            "diffgrpo_bc_weight": 0.1,
            "diffgrpo_step_discount": 0.6,
            "diffgrpo_logprob_reduction": "mean",
            "diffgrpo_group_size": 8,
            "diffgrpo_base_scale": 0.1,
            "diffgrpo_base_advantage_clip": 2.0,
            "diffgrpo_safety_regression_tolerance": 1e-6,
            "diffgrpo_paired_positive_margin": 0.002,
            "diffgrpo_paired_negative_margin": 0.002,
            "diffgrpo_paired_mature_negative_margin": 0.0005,
            "diffgrpo_paired_mature_reward_threshold": 0.75,
            "diffgrpo_paired_regular_kl_weight": 0.1,
            "diffgrpo_paired_mature_kl_weight": 0.5,
            "diffgrpo_paired_bootstrap_advantage_weight": 0.0,
            "stage28_training_selector_role": "multi",
            "stage28_exploration_authorization_path": "",
            "inference_selector_source": "trajectory_relative_harm_v3",
            "diffusion_truncation_timestep": 32,
            "diffusion_roll_timesteps": (32, 24, 16, 8, 0),
            "diffusion_scheduler_num_inference_steps": 125,
        },
        "stage28_public_paired_uplift_explore": {
            "policy_loss_weight": 0.0,
            "kl_loss_weight": 0.0,
            "selector_generation_kl_weight": 0.0,
            "generation_policy_loss_weight": 1.0,
            "generation_kl_loss_weight": 0.0,
            "generation_adaptive_kl_enabled": False,
            "generation_policy_algorithm": "diffgrpo_selected_set",
            "diffgrpo_bc_weight": 0.1,
            "diffgrpo_step_discount": 0.6,
            "diffgrpo_logprob_reduction": "mean",
            "diffgrpo_group_size": 8,
            "diffgrpo_base_scale": 0.1,
            "diffgrpo_base_advantage_clip": 2.0,
            "diffgrpo_safety_regression_tolerance": 1e-6,
            "diffgrpo_paired_positive_margin": 0.002,
            "diffgrpo_paired_negative_margin": 0.002,
            "diffgrpo_paired_mature_negative_margin": 0.0005,
            "diffgrpo_paired_mature_reward_threshold": 0.75,
            "diffgrpo_paired_regular_kl_weight": 0.1,
            "diffgrpo_paired_mature_kl_weight": 0.5,
            "diffgrpo_paired_bootstrap_advantage_weight": 0.0,
            "stage28_training_selector_role": "explore",
            "inference_selector_source": "trajectory_relative_harm_v3",
            "diffusion_truncation_timestep": 32,
            "diffusion_roll_timesteps": (32, 24, 16, 8, 0),
            "diffusion_scheduler_num_inference_steps": 125,
        },
        "stage29_public_headroom_hybrid": {
            "policy_loss_weight": 0.0,
            "kl_loss_weight": 0.0,
            "selector_generation_kl_weight": 0.0,
            "generation_policy_loss_weight": 1.0,
            "generation_kl_loss_weight": 0.0,
            "generation_adaptive_kl_enabled": False,
            "generation_policy_algorithm": "diffgrpo_selected_set",
            "diffgrpo_bc_weight": 0.1,
            "diffgrpo_step_discount": 0.6,
            "diffgrpo_logprob_reduction": "mean",
            "diffgrpo_group_size": 8,
            "diffgrpo_base_advantage_clip": 2.0,
            "diffgrpo_safety_regression_tolerance": 1e-6,
            "stage29_headroom_low": 0.75,
            "stage29_headroom_high": 0.90,
            "stage29_delta_scale_floor": 0.002,
            "stage29_rank_weight": 0.5,
            "stage29_conditional_regularization": False,
            "stage29_fixed_bc_weight": 0.1,
            "stage29_fixed_kl_weight": 0.1,
            "stage29_hard_bc_weight": 0.05,
            "stage29_mature_bc_weight": 0.1,
            "stage29_hard_kl_weight": 0.05,
            "stage29_mature_kl_weight": 0.5,
            "stage29_safety_kl_weight": 0.5,
            "inference_selector_source": "trajectory_relative_harm_v3",
            "diffusion_truncation_timestep": 32,
            "diffusion_roll_timesteps": (32, 24, 16, 8, 0),
            "diffusion_scheduler_num_inference_steps": 125,
        },
        "stage29_public_headroom_conditional": {
            "policy_loss_weight": 0.0,
            "kl_loss_weight": 0.0,
            "selector_generation_kl_weight": 0.0,
            "generation_policy_loss_weight": 1.0,
            "generation_kl_loss_weight": 0.0,
            "generation_adaptive_kl_enabled": False,
            "generation_policy_algorithm": "diffgrpo_selected_set",
            "diffgrpo_bc_weight": 0.1,
            "diffgrpo_step_discount": 0.6,
            "diffgrpo_logprob_reduction": "mean",
            "diffgrpo_group_size": 8,
            "diffgrpo_base_advantage_clip": 2.0,
            "diffgrpo_safety_regression_tolerance": 1e-6,
            "stage29_headroom_low": 0.75,
            "stage29_headroom_high": 0.90,
            "stage29_delta_scale_floor": 0.002,
            "stage29_rank_weight": 0.5,
            "stage29_conditional_regularization": True,
            "stage29_fixed_bc_weight": 0.1,
            "stage29_fixed_kl_weight": 0.1,
            "stage29_hard_bc_weight": 0.05,
            "stage29_mature_bc_weight": 0.1,
            "stage29_hard_kl_weight": 0.05,
            "stage29_mature_kl_weight": 0.5,
            "stage29_safety_kl_weight": 0.5,
            "inference_selector_source": "trajectory_relative_harm_v3",
            "diffusion_truncation_timestep": 32,
            "diffusion_roll_timesteps": (32, 24, 16, 8, 0),
            "diffusion_scheduler_num_inference_steps": 125,
        },
        "stage30_public_mode_coverage": {
            "policy_loss_weight": 0.0,
            "kl_loss_weight": 0.0,
            "selector_generation_kl_weight": 0.0,
            "generation_policy_loss_weight": 1.0,
            "generation_kl_loss_weight": 0.0,
            "generation_adaptive_kl_enabled": False,
            "generation_policy_algorithm": "diffgrpo_mode_coverage",
            "diffgrpo_bc_weight": 0.1,
            "diffgrpo_step_discount": 0.6,
            "diffgrpo_logprob_reduction": "mean",
            "diffgrpo_group_size": 20,
            "diffgrpo_base_advantage_clip": 2.0,
            "diffgrpo_safety_regression_tolerance": 1e-6,
            "stage30_top_k": 5,
            "stage30_delta_scale_floor": 0.002,
            "stage30_coverage_top_weight": 0.5,
            "stage30_boundary_top_weight": 0.25,
            "stage30_boundary_positive_margin": 0.001,
            "stage30_boundary_negative_multiplier": 2.0,
            "stage30_mature_negative_floor": -0.0002,
            "stage30_mature_positive_margin": 0.002,
            "stage30_mature_negative_multiplier": 4.0,
            "stage30_component_tolerance": 1e-6,
            "stage30_plan_sha256": "e9867ec48d4806ff97adb0284bae7550305cb1350e4ec66dbfd20d245de435ce",
            "stage30_optimizer_steps_per_epoch": 48,
            "stage30_gradient_accumulation": 8,
            "stage30_global_bucket_composition": (2, 30, 8, 24),
            "inference_selector_source": "trajectory_relative_harm_v3",
            "diffusion_truncation_timestep": 32,
            "diffusion_roll_timesteps": (32, 24, 16, 8, 0),
            "diffusion_scheduler_num_inference_steps": 125,
            "weight_decay": 0.0,
        },
        "stage30_public_mode_coverage_constrained": {
            "policy_loss_weight": 0.0,
            "kl_loss_weight": 0.0,
            "selector_generation_kl_weight": 0.0,
            "generation_policy_loss_weight": 1.0,
            "generation_kl_loss_weight": 0.0,
            "generation_adaptive_kl_enabled": False,
            "generation_policy_algorithm": "diffgrpo_mode_coverage",
            "diffgrpo_bc_weight": 0.1,
            "diffgrpo_step_discount": 0.6,
            "diffgrpo_logprob_reduction": "mean",
            "diffgrpo_group_size": 20,
            "diffgrpo_base_advantage_clip": 2.0,
            "diffgrpo_safety_regression_tolerance": 1e-6,
            "stage30_top_k": 5,
            "stage30_delta_scale_floor": 0.002,
            "stage30_coverage_top_weight": 0.5,
            "stage30_boundary_top_weight": 0.25,
            "stage30_boundary_positive_margin": 0.001,
            "stage30_boundary_negative_multiplier": 2.0,
            "stage30_mature_negative_floor": -0.0002,
            "stage30_mature_positive_margin": 0.002,
            "stage30_mature_negative_multiplier": 4.0,
            "stage30_component_tolerance": 1e-6,
            "stage30_plan_sha256": "e9867ec48d4806ff97adb0284bae7550305cb1350e4ec66dbfd20d245de435ce",
            "stage30_optimizer_steps_per_epoch": 48,
            "stage30_gradient_accumulation": 8,
            "stage30_global_bucket_composition": (2, 30, 8, 24),
            "inference_selector_source": "trajectory_relative_harm_v3",
            "diffusion_truncation_timestep": 32,
            "diffusion_roll_timesteps": (32, 24, 16, 8, 0),
            "diffusion_scheduler_num_inference_steps": 125,
            "weight_decay": 0.0,
        },
        "stage31_public_deployed_pair": {
            "policy_loss_weight": 0.0,
            "kl_loss_weight": 0.0,
            "selector_generation_kl_weight": 0.0,
            "generation_policy_loss_weight": 1.0,
            "generation_kl_loss_weight": 0.0,
            "generation_adaptive_kl_enabled": False,
            "generation_policy_algorithm": "diffgrpo_deployed_selected_set",
            "diffgrpo_bc_weight": 0.1,
            "diffgrpo_step_discount": 0.6,
            "diffgrpo_logprob_reduction": "mean",
            "diffgrpo_group_size": 8,
            "diffgrpo_base_advantage_clip": 2.0,
            "diffgrpo_safety_regression_tolerance": 1e-6,
            "stage31_plan_sha256": "3ff3d3bcd3120b9d73a017876f942458eb3fa16651517e4e30b27cf2fff7bb0d",
            "stage31_headroom_low": 0.75,
            "stage31_headroom_high": 0.90,
            "stage31_delta_scale_floor": 0.002,
            "stage31_rank_weight": 0.5,
            "stage31_bc_weight": 0.1,
            "stage31_kl_weight": 0.1,
            "stage31_safety_kl_weight": 0.5,
            "stage31_optimizer_steps_per_epoch": 48,
            "stage31_gradient_accumulation": 8,
            "stage31_global_bucket_composition": (2, 30, 8, 24),
            "inference_selector_source": "trajectory_relative_harm_v3",
            "diffusion_truncation_timestep": 32,
            "diffusion_roll_timesteps": (32, 24, 16, 8, 0),
            "diffusion_scheduler_num_inference_steps": 125,
            "weight_decay": 0.0,
        },
        "stage31_public_deployed_frontier": {
            "policy_loss_weight": 0.0,
            "kl_loss_weight": 0.0,
            "selector_generation_kl_weight": 0.0,
            "generation_policy_loss_weight": 1.0,
            "generation_kl_loss_weight": 0.0,
            "generation_adaptive_kl_enabled": False,
            "generation_policy_algorithm": "diffgrpo_deployed_selected_set",
            "diffgrpo_bc_weight": 0.1,
            "diffgrpo_step_discount": 0.6,
            "diffgrpo_logprob_reduction": "mean",
            "diffgrpo_group_size": 8,
            "diffgrpo_base_advantage_clip": 2.0,
            "diffgrpo_safety_regression_tolerance": 1e-6,
            "stage31_plan_sha256": "3ff3d3bcd3120b9d73a017876f942458eb3fa16651517e4e30b27cf2fff7bb0d",
            "stage31_headroom_low": 0.75,
            "stage31_headroom_high": 0.90,
            "stage31_delta_scale_floor": 0.002,
            "stage31_rank_weight": 0.5,
            "stage31_bc_weight": 0.1,
            "stage31_kl_weight": 0.1,
            "stage31_safety_kl_weight": 0.5,
            "stage31_optimizer_steps_per_epoch": 48,
            "stage31_gradient_accumulation": 8,
            "stage31_global_bucket_composition": (2, 30, 8, 24),
            "inference_selector_source": "trajectory_relative_harm_v3",
            "diffusion_truncation_timestep": 32,
            "diffusion_roll_timesteps": (32, 24, 16, 8, 0),
            "diffusion_scheduler_num_inference_steps": 125,
            "weight_decay": 0.0,
        },
    }[
        "stage30_public_mode_coverage_constrained"
        if mode in STAGE34_GRPO_MODES
        else "stage31_public_deployed_frontier"
        if mode in (
            STAGE32_GRPO_MODES | STAGE33_GRPO_MODES | STAGE35_GRPO_MODES
            | STAGE36_GRPO_MODES | STAGE37_GRPO_MODES | STAGE38_GRPO_MODES | STAGE39_GRPO_MODES
        )
        else mode
    ]
    if mode in STAGE34_GRPO_MODES:
        for legacy_name in STAGE34_LEGACY_STAGE30_KEYS:
            route_expected.pop(legacy_name)
        route_expected.update(STAGE34_ROUTE_EXPECTED)
    if mode in STAGE35_GRPO_MODES:
        for legacy_name in STAGE35_LEGACY_STAGE31_KEYS:
            route_expected.pop(legacy_name)
        route_expected.update(STAGE35_ROUTE_EXPECTED)
    if mode in STAGE36_GRPO_MODES:
        for legacy_name in STAGE36_LEGACY_STAGE31_KEYS:
            route_expected.pop(legacy_name)
        route_expected.update(STAGE36_ROUTE_EXPECTED)
    if mode in STAGE37_GRPO_MODES:
        for legacy_name in STAGE37_LEGACY_STAGE31_KEYS:
            route_expected.pop(legacy_name)
        route_expected.update(STAGE37_ROUTE_EXPECTED)
    if mode in STAGE38_GRPO_MODES:
        for legacy_name in STAGE38_LEGACY_STAGE31_KEYS:
            route_expected.pop(legacy_name)
        route_expected.update(STAGE38_ROUTE_EXPECTED)
    if mode in (
        SELECTED_SET_GRPO_MODES | STAGE34_GRPO_MODES | STAGE35_GRPO_MODES
        | STAGE36_GRPO_MODES | STAGE37_GRPO_MODES | STAGE38_GRPO_MODES | STAGE39_GRPO_MODES
    ):
        selector_source = str(getattr(config, "inference_selector_source", ""))
        allowed_selector_sources = (
            {"trajectory_relative_harm_v3"}
            if mode in (
                STAGE27_GRPO_MODES | STAGE28_GRPO_MODES | STAGE29_GRPO_MODES
                | STAGE31_GRPO_MODES | STAGE32_GRPO_MODES | STAGE34_GRPO_MODES
                | STAGE35_GRPO_MODES | STAGE36_GRPO_MODES
                | STAGE37_GRPO_MODES | STAGE38_GRPO_MODES | STAGE39_GRPO_MODES
            )
            else {"trajectory_oof", "trajectory_relative_harm_v3"}
        )
        if selector_source not in allowed_selector_sources:
            raise ValueError(
                f"formal {mode} requires one of the frozen selector sources "
                f"{sorted(allowed_selector_sources)}"
            )
        route_expected["inference_selector_source"] = selector_source
        if selector_source == "trajectory_relative_harm_v3" and (
            not str(getattr(config, "stage25_selector_checkpoint_path", ""))
            or not str(getattr(config, "stage25_selector_calibration_path", ""))
        ):
            raise ValueError(
                "formal Stage25 selected-set GRPO requires checkpoint and "
                "calibration paths"
            )
        if mode == "stage28_public_paired_uplift_explore" and not str(
            getattr(config, "stage28_exploration_authorization_path", "")
        ):
            raise ValueError(
                "formal Stage28 explore mode requires its SHA-locked "
                "training-only authorization"
            )
    if mode in STAGE30_GRPO_MODES:
        if (
            str(getattr(config, "inference_selector_source", ""))
            != "trajectory_relative_harm_v3"
            or not str(getattr(config, "stage25_selector_checkpoint_path", ""))
            or not str(getattr(config, "stage25_selector_calibration_path", ""))
        ):
            raise ValueError(
                "formal Stage30 requires the frozen S-multi selector paths"
            )
        if not str(getattr(config, "stage30_bucket_manifest_path", "")):
            raise ValueError("formal Stage30 requires its bucket manifest")
        if tuple(getattr(config, "stage30_global_bucket_composition", ())) != (
            2, 30, 8, 24
        ):
            raise ValueError("formal Stage30 global bucket composition drifted")
        if int(getattr(config, "stage30_gradient_accumulation", -1)) != 8:
            raise ValueError("formal Stage30 requires accumulation=8")
    if mode in STAGE31_GRPO_MODES:
        if (
            str(getattr(config, "inference_selector_source", ""))
            != "trajectory_relative_harm_v3"
            or not str(getattr(config, "stage25_selector_checkpoint_path", ""))
            or not str(getattr(config, "stage25_selector_calibration_path", ""))
        ):
            raise ValueError(
                "formal Stage31 requires the frozen S-multi selector paths"
            )
        if not str(getattr(config, "stage31_bucket_manifest_path", "")):
            raise ValueError("formal Stage31 requires its bucket manifest")
        if tuple(getattr(config, "stage31_global_bucket_composition", ())) != (
            2, 30, 8, 24
        ):
            raise ValueError("formal Stage31 global bucket composition drifted")
        if int(getattr(config, "stage31_gradient_accumulation", -1)) != 8:
            raise ValueError("formal Stage31 requires accumulation=8")
    if mode in STAGE32_GRPO_MODES:
        if (
            str(getattr(config, "inference_selector_source", ""))
            != "trajectory_relative_harm_v3"
            or not str(getattr(config, "stage25_selector_checkpoint_path", ""))
            or not str(getattr(config, "stage25_selector_calibration_path", ""))
        ):
            raise ValueError("formal Stage32 requires the frozen S-multi selector paths")
        if not str(getattr(config, "stage32_bucket_manifest_path", "")):
            raise ValueError("formal Stage32 requires its bucket manifest")
        if tuple(getattr(config, "stage32_global_bucket_composition", ())) != (
            2, 30, 8, 24
        ):
            raise ValueError("formal Stage32 global bucket composition drifted")
        if int(getattr(config, "stage32_gradient_accumulation", -1)) != 8:
            raise ValueError("formal Stage32 requires accumulation=8")
        if int(getattr(config, "stage32_frontier_pool_size", -1)) != 4:
            raise ValueError("formal Stage32 requires frontier pool size=4")
        for name, wanted in (
            ("stage32_frontier_risk_margin", 0.10),
            ("stage32_frontier_owner_margin", 0.001),
            ("stage32_frontier_weight", 0.5),
            ("stage32_frontier_mature_cap", 0.25),
        ):
            if not math.isclose(
                float(getattr(config, name, -1.0)), wanted,
                rel_tol=0.0, abs_tol=1e-12,
            ):
                raise ValueError(f"formal Stage32 requires {name}={wanted}")
        if str(getattr(config, "stage32_plan_sha256", "")) != STAGE32_PLAN_SHA256:
            raise ValueError("formal Stage32 plan SHA drifted")
        if (
            mode == "stage32_selector_aware_frontier"
            and str(getattr(config, "stage32_scf_objective_revision", ""))
            != STAGE32_SCF_OBJECTIVE_REVISION
        ):
            raise ValueError("formal Stage32 SCF objective revision drifted")
    if mode in STAGE33_GRPO_MODES:
        if str(getattr(config, "inference_selector_source", "")) != "trajectory_relative_harm_v3":
            raise ValueError("formal Stage33 requires the frozen Stage25 selector")
        if not str(getattr(config, "stage25_selector_checkpoint_path", "")):
            raise ValueError("formal Stage33 requires the Stage25 selector checkpoint")
        if not str(getattr(config, "stage25_selector_calibration_path", "")):
            raise ValueError("formal Stage33 requires the Stage25 selector calibration")
        if not str(getattr(config, "stage33_bucket_manifest_path", "")):
            raise ValueError("formal Stage33 requires its bucket manifest")
        if tuple(getattr(config, "stage33_global_bucket_composition", ())) != (2, 30, 8, 24):
            raise ValueError("formal Stage33 global bucket composition drifted")
        if int(getattr(config, "stage33_gradient_accumulation", -1)) != 8:
            raise ValueError("formal Stage33 requires accumulation=8")
        if int(getattr(config, "diffgrpo_group_size", -1)) != 8:
            raise ValueError("formal Stage33 requires group_size=8")
        if str(getattr(config, "stage33_plan_sha256", "")) != STAGE33_PLAN_SHA256:
            raise ValueError("formal Stage33 plan SHA drifted")
        if str(getattr(config, "stage33_cdc_objective_revision", "")) != STAGE33_CDC_OBJECTIVE_REVISION:
            raise ValueError("formal Stage33 CDC objective revision drifted")
        for name, wanted in (
            ("stage33_deployment_weight", 0.5),
            ("stage33_headroom_weight", 0.5),
            ("stage33_headroom_margin", 0.001),
            ("stage33_advantage_clip", 2.0),
        ):
            if not math.isclose(float(getattr(config, name, -1.0)), wanted, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"formal Stage33 requires {name}={wanted}")
    if mode in STAGE34_GRPO_MODES and not str(
        getattr(config, "stage34_bucket_manifest_path", "")
    ):
        raise ValueError("formal Stage34 requires its bucket manifest")
    if mode in STAGE35_GRPO_MODES and not str(
        getattr(config, "stage35_bucket_manifest_path", "")
    ):
        raise ValueError("formal Stage35 requires its bucket manifest")
    if mode in STAGE36_GRPO_MODES and not str(
        getattr(config, "stage36_bucket_manifest_path", "")
    ):
        raise ValueError("formal Stage36 requires its bucket manifest")
    if mode in STAGE37_GRPO_MODES and not str(
        getattr(config, "stage37_bucket_manifest_path", "")
    ):
        raise ValueError("formal Stage37 requires its bucket manifest")
    if mode in STAGE38_GRPO_MODES and not str(
        getattr(config, "stage38_bucket_manifest_path", "")
    ):
        raise ValueError("formal Stage38 requires its bucket manifest")
    for name, wanted in {**expected, **route_expected}.items():
        actual = getattr(config, name, None)
        if isinstance(wanted, tuple) and actual is not None:
            actual = tuple(actual)
        matches = (
            math.isclose(float(actual), wanted, rel_tol=0.0, abs_tol=1e-12)
            if isinstance(wanted, float) else actual == wanted
        )
        if not matches:
            raise ValueError(
                f"formal {mode} requires {name}={wanted!r}; got {actual!r}"
            )
    layer0_lr_mult = float(getattr(config, "grpo_decoder_layer0_lr_mult", 1.0))
    if mode in STAGE9_GRPO_MODES and not math.isclose(
        layer0_lr_mult, 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("formal Stage-9 routes require layer-0 LR multiplier 1.0")
    if mode in STAGE10_GRPO_MODES and not any(
        math.isclose(layer0_lr_mult, wanted, rel_tol=0.0, abs_tol=1e-12)
        for wanted in (0.1, 1.0)
    ):
        raise ValueError("formal Stage-10 route requires layer-0 LR multiplier 0.1 or 1.0")
    if mode in (
        STAGE21_GRPO_MODES | STAGE23_GRPO_MODES | STAGE27_GRPO_MODES
        | STAGE28_GRPO_MODES | STAGE29_GRPO_MODES | STAGE30_GRPO_MODES
        | STAGE31_GRPO_MODES | STAGE32_GRPO_MODES | STAGE33_GRPO_MODES
        | STAGE34_GRPO_MODES | STAGE35_GRPO_MODES | STAGE36_GRPO_MODES
        | STAGE37_GRPO_MODES | STAGE38_GRPO_MODES | STAGE39_GRPO_MODES
    ) and not math.isclose(
        layer0_lr_mult, 0.1, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "formal Stage21/23/27/28/29/30/31 requires layer-0 LR multiplier 0.1"
        )


def build_stage10_decoder_param_groups(
    named_parameters, layer0_lr_mult: float
) -> List[Dict[str, Any]]:
    """Split trainable layer-0 parameters without freezing either decoder layer."""
    layer0 = []
    remaining = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if (
            "_trajectory_head.diff_decoder.layers.0." in name
            or "_trajectory_head.stage39_challenger_decoder.layers.0." in name
        ):
            layer0.append(parameter)
        else:
            remaining.append(parameter)
    if not layer0 or not remaining:
        raise RuntimeError("Stage-10 optimizer requires trainable layer-0 and layer-1 parameters")
    return [
        {"params": remaining, "lr_scale": 1.0},
        {"params": layer0, "lr_scale": float(layer0_lr_mult)},
    ]

class TransfuserAgent(AbstractAgent):
    """Agent interface for TransFuser baseline."""

    def __init__(
        self,
        config: TransfuserConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
        reference_checkpoint_path: Optional[str] = None,
    ):
        """
        Initializes TransFuser agent.
        :param config: global config of TransFuser agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        """
        super().__init__()

        self._config = config
        self._lr = lr
        self._evaluation_token: Optional[str] = None

        if not checkpoint_path:
            raise ValueError("A pretrained DiffusionDrive checkpoint is required for GRPO training.")
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"DiffusionDrive checkpoint does not exist: {checkpoint_path}")

        reference_checkpoint_path = reference_checkpoint_path or checkpoint_path
        if not Path(reference_checkpoint_path).is_file():
            raise FileNotFoundError(
                f"DiffusionDrive reference checkpoint does not exist: {reference_checkpoint_path}"
            )
        self._checkpoint_path = checkpoint_path
        checkpoint_file = Path(checkpoint_path)
        reference_file = Path(reference_checkpoint_path)
        self._checkpoint_sha256 = file_sha256(checkpoint_file)
        self._reference_checkpoint_sha256 = (
            self._checkpoint_sha256
            if checkpoint_file.resolve() == reference_file.resolve()
            else file_sha256(reference_file)
        )
        validate_formal_grpo_config(config)
        formal_mode = str(getattr(config, "grpo_training_mode", ""))
        if formal_mode in FORMAL_GRPO_MODES:
            expected_base_sha = (
                STAGE27_PUBLIC_BASE_SHA256
                if formal_mode in (
                    STAGE27_GRPO_MODES | STAGE28_GRPO_MODES
                    | STAGE29_GRPO_MODES | STAGE30_GRPO_MODES
                    | STAGE31_GRPO_MODES | STAGE32_GRPO_MODES | STAGE33_GRPO_MODES
                    | STAGE34_GRPO_MODES | STAGE35_GRPO_MODES
                    | STAGE36_GRPO_MODES | STAGE37_GRPO_MODES
                    | STAGE38_GRPO_MODES | STAGE39_GRPO_MODES
                )
                else FORMAL_BASE_SHA256
            )
            if self._reference_checkpoint_sha256 != expected_base_sha:
                raise RuntimeError(
                    "Formal GRPO reference checkpoint is not the registered "
                    f"base for mode={formal_mode}"
                )
            if formal_mode in STAGE35_GRPO_MODES:
                if self._checkpoint_sha256 not in STAGE35_ALLOWED_INITIAL_SHA256:
                    raise RuntimeError(
                        "Stage35 current policy is not a frozen DPEL192 initializer"
                    )
            elif formal_mode in STAGE36_GRPO_MODES:
                if self._checkpoint_sha256 not in STAGE36_ALLOWED_INITIAL_SHA256:
                    raise RuntimeError(
                        "Stage36 current policy is not a frozen DPEL192 initializer"
                    )
            elif formal_mode in STAGE37_GRPO_MODES:
                if self._checkpoint_sha256 not in STAGE37_ALLOWED_INITIAL_SHA256:
                    raise RuntimeError(
                        "Stage37 must initialize independently from public 88.1"
                    )
            elif formal_mode in STAGE38_GRPO_MODES:
                if self._checkpoint_sha256 not in STAGE38_ALLOWED_INITIAL_SHA256:
                    raise RuntimeError(
                        "Stage38 must initialize from a fold-matched Stage36 RGT192 checkpoint"
                    )
            elif formal_mode in STAGE39_GRPO_MODES:
                if self._checkpoint_sha256 != STAGE39_PUBLIC_SHA256:
                    raise RuntimeError(
                        "Stage39 must initialize its challenger from official public 88.1"
                    )
            elif self._checkpoint_sha256 != self._reference_checkpoint_sha256:
                raise RuntimeError(
                    "Formal GRPO current policy must initialize from frozen base"
                )
            expected_lr = (
                3e-5
                if formal_mode in STAGE22_GRPO_MODES
                else 1e-6
            )
            if not math.isclose(
                float(lr), expected_lr, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"Formal GRPO learning rate must be {expected_lr}"
                )
        stage17_training = (
            str(getattr(config, "grpo_training_mode", ""))
            == "paired_tail_risk_selector"
        )
        stage17_inference = (
            str(getattr(config, "inference_selector_source", ""))
            == "paired_tail_risk"
        )
        if stage17_training or stage17_inference:
            expected_generator = str(
                getattr(config, "paired_risk_expected_generator_sha256", "")
                or STAGE17_GENERATOR_SHA256
            )
            expected_reference = str(
                getattr(config, "paired_risk_expected_reference_sha256", "")
                or FORMAL_BASE_SHA256
            )
            if self._checkpoint_sha256 != expected_generator:
                raise RuntimeError("Stage-17 generator checkpoint SHA mismatch")
            if self._reference_checkpoint_sha256 != expected_reference:
                raise RuntimeError("Stage-17 reference checkpoint SHA mismatch")
            schedule = tuple(
                int(t) for t in getattr(config, "diffusion_roll_timesteps", ())
            )
            if (
                schedule != (32, 24, 16, 8, 0)
                or int(getattr(config, "diffusion_truncation_timestep", -1)) != 32
                or int(getattr(config, "diffusion_scheduler_num_inference_steps", -1)) != 125
                or str(getattr(config, "generation_policy_algorithm", ""))
                not in {
                    "diffgrpo_full_chain",
                    "diffgrpo_selected_anchor",
                    "diffgrpo_selected_anchor_base_preserve",
                }
            ):
                raise ValueError("Stage-17 requires the locked full-chain schedule")
            if stage17_training and not math.isclose(
                float(lr), 1e-4, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("Stage-17 adapter learning rate must be 1e-4")
            if stage17_inference and not 0.0 <= float(
                getattr(config, "paired_risk_threshold", -1.0)
            ) <= 1.0:
                raise ValueError("Stage-17 inference requires a calibrated threshold")
        self._reference_checkpoint_path = reference_checkpoint_path
        self._transfuser_model = TransfuserModel(config)
        self.init_from_pretrained()
        self._value_selector_checkpoint_sha256 = None
        value_selector_checkpoint = str(
            getattr(config, "value_selector_checkpoint_path", "")
        )
        if value_selector_checkpoint:
            self.load_value_selector_checkpoint(value_selector_checkpoint)
        self._stage23_selector_checkpoint_sha256 = None
        stage23_selector_checkpoint = str(
            getattr(config, "stage23_selector_checkpoint_path", "")
        )
        if stage23_selector_checkpoint:
            self.load_stage23_selector_checkpoint(stage23_selector_checkpoint)
        elif str(getattr(config, "inference_selector_source", "")) == "trajectory_oof":
            raise ValueError("trajectory_oof inference requires a Stage23 selector checkpoint")
        if (
            str(getattr(config, "grpo_training_mode", "")) in STAGE23_GRPO_MODES
            and str(getattr(config, "inference_selector_source", ""))
            == "trajectory_oof"
        ):
            calibration_path = Path(
                str(getattr(config, "stage23_selector_calibration_path", ""))
            )
            if not calibration_path.is_file():
                raise FileNotFoundError(
                    "Stage23 generator requires frozen selector calibration"
                )
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            if (
                not calibration.get("passed")
                or calibration.get("selector_checkpoint_sha256")
                != self._stage23_selector_checkpoint_sha256
            ):
                raise RuntimeError("Stage23 selector calibration/checkpoint SHA mismatch")
            expected_margin = float(calibration["residual_margin"])
            expected_threshold = float(calibration["selected"]["threshold"])
            if not math.isclose(
                float(config.stage23_selector_residual_margin), expected_margin,
                rel_tol=0.0, abs_tol=1e-12,
            ) or not math.isclose(
                float(config.stage23_selector_safety_threshold), expected_threshold,
                rel_tol=0.0, abs_tol=1e-12,
            ):
                raise RuntimeError("Stage23 runtime selector calibration values drifted")
        self._stage24_selector_checkpoint_sha256 = None
        self._stage24_calibration_sha256 = None
        stage24_selector_checkpoint = str(
            getattr(config, "stage24_selector_checkpoint_path", "")
        )
        stage24_active = (
            str(getattr(config, "inference_selector_source", ""))
            == "trajectory_safety_value_v2"
        )
        if stage24_selector_checkpoint:
            self.load_stage24_selector_checkpoint(stage24_selector_checkpoint)
        elif stage24_active:
            raise ValueError(
                "trajectory_safety_value_v2 requires a Stage24 selector checkpoint"
            )
        if stage24_active and not bool(
            getattr(config, "stage24_collect_calibration", False)
        ):
            calibration_path = Path(
                str(getattr(config, "stage24_selector_calibration_path", ""))
            )
            if not calibration_path.is_file():
                raise FileNotFoundError(
                    "Stage24 deployment requires frozen selector calibration"
                )
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            if (
                not calibration.get("passed")
                or calibration.get("selector_checkpoint_sha256")
                != self._stage24_selector_checkpoint_sha256
            ):
                raise RuntimeError(
                    "Stage24 selector calibration/checkpoint SHA mismatch"
                )
            expected = {
                "stage24_selector_residual_margin": float(
                    calibration["residual_margin"]
                ),
                "stage24_selector_risk_threshold": float(
                    calibration["risk_threshold"]
                ),
                "stage24_selector_ood_threshold": float(
                    calibration["ood_threshold"]
                ),
            }
            for name, wanted in expected.items():
                if not math.isclose(
                    float(getattr(config, name)), wanted,
                    rel_tol=0.0, abs_tol=1e-12,
                ):
                    raise RuntimeError(
                        f"Stage24 runtime calibration drifted for {name}"
                    )
            ood = calibration.get("ood", {})
            mean = torch.as_tensor(ood.get("mean", ()), dtype=torch.float32)
            variance = torch.as_tensor(
                ood.get("variance", ()), dtype=torch.float32
            )
            head = self._transfuser_model._trajectory_head
            if (
                mean.shape != head._stage24_ood_mean.shape
                or variance.shape != head._stage24_ood_variance.shape
                or not torch.isfinite(mean).all()
                or not torch.isfinite(variance).all()
                or not (variance > 0).all()
            ):
                raise RuntimeError("Stage24 calibration has invalid OOD statistics")
            head._stage24_ood_mean.copy_(mean)
            head._stage24_ood_variance.copy_(variance)
            head._stage24_calibration_loaded.fill_(True)
            self._stage24_calibration_sha256 = file_sha256(calibration_path)
        self._stage25_selector_checkpoint_sha256 = None
        self._stage25_calibration_sha256 = None
        stage25_active = (
            str(getattr(config, "inference_selector_source", ""))
            == "trajectory_relative_harm_v3"
        )
        stage25_selector_checkpoint = str(
            getattr(config, "stage25_selector_checkpoint_path", "")
        )
        if stage25_selector_checkpoint:
            self.load_stage25_selector_checkpoint(stage25_selector_checkpoint)
        elif stage25_active:
            raise ValueError(
                "trajectory_relative_harm_v3 requires a Stage25 checkpoint"
            )
        if stage25_active and not bool(
            getattr(config, "stage25_collect_calibration", False)
        ):
            calibration_path = Path(str(
                getattr(config, "stage25_selector_calibration_path", "")
            ))
            if not calibration_path.is_file():
                raise FileNotFoundError(
                    "Stage25 deployment requires frozen selector calibration"
                )
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            if (
                calibration.get("selector_checkpoint_sha256")
                != self._stage25_selector_checkpoint_sha256
            ):
                raise RuntimeError(
                    "Stage25 selector calibration/checkpoint SHA mismatch"
                )
            if not calibration.get("passed"):
                validate_stage28_exploration_authorization(
                    config,
                    str(self._stage25_selector_checkpoint_sha256),
                    calibration_path,
                )
            stage28_role = str(
                getattr(config, "stage28_training_selector_role", "")
            )
            if formal_mode in STAGE28_GRPO_MODES:
                expected_selector_sha, expected_calibration_sha = (
                    (
                        STAGE28_EXPLORE_SELECTOR_SHA256,
                        STAGE28_EXPLORE_CALIBRATION_SHA256,
                    )
                    if stage28_role == "explore"
                    else (
                        STAGE28_MULTI_SELECTOR_SHA256,
                        STAGE28_MULTI_CALIBRATION_SHA256,
                    )
                )
                if (
                    self._stage25_selector_checkpoint_sha256
                    != expected_selector_sha
                    or file_sha256(calibration_path) != expected_calibration_sha
                ):
                    raise RuntimeError(
                        "Stage28 training selector/calibration SHA drifted"
                    )
            if formal_mode in (
                STAGE29_GRPO_MODES | STAGE30_GRPO_MODES | STAGE31_GRPO_MODES
                | STAGE32_GRPO_MODES | STAGE33_GRPO_MODES | STAGE34_GRPO_MODES
                | STAGE35_GRPO_MODES | STAGE36_GRPO_MODES | STAGE37_GRPO_MODES
                | STAGE38_GRPO_MODES | STAGE39_GRPO_MODES
            ):
                if (
                    self._stage25_selector_checkpoint_sha256
                    != STAGE28_MULTI_SELECTOR_SHA256
                    or file_sha256(calibration_path)
                    != STAGE28_MULTI_CALIBRATION_SHA256
                    or not calibration.get("passed")
                ):
                    raise RuntimeError(
                        "Stage29/30/31 requires the frozen passing S-multi selector "
                        "and calibration"
                    )
            expected = {
                "stage24_selector_residual_margin": calibration[
                    "residual_margin"
                ],
                "stage24_selector_ood_threshold": calibration["ood_threshold"],
                "stage25_selector_risk_threshold": calibration[
                    "risk_threshold"
                ],
            }
            for name, wanted in expected.items():
                if not math.isclose(
                    float(getattr(config, name)), float(wanted),
                    rel_tol=0.0, abs_tol=1e-12,
                ):
                    raise RuntimeError(
                        f"Stage25 runtime calibration drifted for {name}"
                    )
            ood = calibration.get("ood", {})
            mean = torch.as_tensor(ood.get("mean", ()), dtype=torch.float32)
            variance = torch.as_tensor(
                ood.get("variance", ()), dtype=torch.float32
            )
            head = self._transfuser_model._trajectory_head
            if (
                mean.shape != head._stage24_ood_mean.shape
                or variance.shape != head._stage24_ood_variance.shape
                or not torch.isfinite(mean).all()
                or not torch.isfinite(variance).all()
                or not (variance > 0).all()
            ):
                raise RuntimeError("Stage25 calibration has invalid OOD state")
            head._stage24_ood_mean.copy_(mean)
            head._stage24_ood_variance.copy_(variance)
            head._stage24_calibration_loaded.fill_(True)
            self._stage25_calibration_sha256 = file_sha256(calibration_path)
        self._stage37_jfi_checkpoint_sha256 = None
        self._stage37_jfi_calibration_sha256 = None
        stage37_jfi_active = (
            str(getattr(config, "inference_selector_source", ""))
            == "joint_feasible_improvement_v1"
        )
        stage37_jfi_checkpoint = str(getattr(
            config, "stage37_jfi_checkpoint_path", ""
        ))
        if stage37_jfi_checkpoint:
            self.load_stage37_jfi_checkpoint(stage37_jfi_checkpoint)
        elif stage37_jfi_active:
            raise ValueError("JFI deployment requires its frozen checkpoint")
        if stage37_jfi_active and not bool(getattr(
            config, "stage37_jfi_collect_calibration", False
        )):
            calibration_path = Path(str(getattr(
                config, "stage37_jfi_calibration_path", ""
            )))
            if not calibration_path.is_file():
                raise FileNotFoundError("JFI deployment calibration is missing")
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            if (
                not calibration.get("passed")
                or calibration.get("selector_checkpoint_sha256")
                != self._stage37_jfi_checkpoint_sha256
            ):
                raise RuntimeError("JFI calibration/checkpoint SHA mismatch")
            mean_pool = float(calibration.get("mean_candidate_pool_size", -1.0))
            if not 2.0 <= mean_pool <= 4.0:
                raise RuntimeError("JFI calibration pool is outside [2,4]")
            expected = {
                "stage37_jfi_joint_threshold": calibration["joint_threshold"],
                "stage37_jfi_q10_floor": calibration["q10_floor"],
                "stage24_selector_ood_threshold": calibration["ood_threshold"],
            }
            for name, wanted in expected.items():
                if not math.isclose(
                    float(getattr(config, name)), float(wanted),
                    rel_tol=0.0, abs_tol=1e-12,
                ):
                    raise RuntimeError(f"JFI runtime calibration drifted for {name}")
            ood = calibration.get("ood", {})
            mean = torch.as_tensor(ood.get("mean", ()), dtype=torch.float32)
            variance = torch.as_tensor(
                ood.get("variance", ()), dtype=torch.float32
            )
            head = self._transfuser_model._trajectory_head
            if (
                mean.shape != head._stage24_ood_mean.shape
                or variance.shape != head._stage24_ood_variance.shape
                or not torch.isfinite(mean).all()
                or not torch.isfinite(variance).all()
                or not (variance > 0).all()
            ):
                raise RuntimeError("JFI calibration has invalid OOD state")
            head._stage24_ood_mean.copy_(mean)
            head._stage24_ood_variance.copy_(variance)
            head._stage24_calibration_loaded.fill_(True)
            self._stage37_jfi_calibration_sha256 = file_sha256(calibration_path)
        self._paired_risk_checkpoint_sha256 = None
        paired_risk_checkpoint = str(
            getattr(config, "paired_risk_checkpoint_path", "")
        )
        if paired_risk_checkpoint:
            self.load_paired_risk_checkpoint(paired_risk_checkpoint)
        elif stage17_inference:
            raise ValueError("paired_tail_risk inference requires an adapter checkpoint")
        self._adaptive_kl_controller = None
        if bool(getattr(config, "generation_adaptive_kl_enabled", False)):
            self._adaptive_kl_controller = AdaptiveKLController(
                initial_coefficient=config.generation_kl_initial_coefficient,
                minimum_coefficient=config.generation_kl_min_coefficient,
                maximum_coefficient=config.generation_kl_max_coefficient,
                target=config.generation_kl_target,
                hard_limit=config.generation_kl_hard_limit,
                window=config.generation_kl_window,
                update_interval=config.generation_kl_update_interval,
                adaptation_factor=config.generation_kl_adaptation_factor,
                lower_ratio=config.generation_kl_lower_ratio,
                upper_ratio=config.generation_kl_upper_ratio,
                hard_limit_patience=config.generation_kl_hard_limit_patience,
            )

        # 1. 冻结整个模型
        self._transfuser_model.requires_grad_(False)

        # 2. Choose shared-feature training or a fixed-generator baseline.
        training_mode = getattr(config, "grpo_training_mode", "classification_shared")
        trajectory_head = self._transfuser_model._trajectory_head
        if training_mode in {"classification_shared", "selector_group"}:
            trajectory_head.diff_decoder.requires_grad_(True)
        elif training_mode in {"classification_head", "selector"}:
            trajectory_head.diff_decoder.layers[-1].task_decoder.plan_cls_branch.requires_grad_(
                True
            )
        elif training_mode == "value_selector":
            trajectory_head.value_selector.requires_grad_(True)
        elif training_mode == "stage23_selector":
            trajectory_head.stage23_selector.requires_grad_(True)
        elif training_mode == "stage24_selector":
            trajectory_head.stage24_selector.requires_grad_(True)
        elif training_mode == "stage25_relative_harm_selector":
            trajectory_head.stage25_selector.requires_grad_(True)
        elif training_mode == "stage37_jfi_selector":
            trajectory_head.stage37_jfi_selector.requires_grad_(True)
        elif training_mode == "paired_tail_risk_selector":
            trajectory_head.paired_risk_head.requires_grad_(True)
        elif training_mode in STAGE39_GRPO_MODES:
            trajectory_head.stage39_challenger_decoder.requires_grad_(True)
            for layer in trajectory_head.stage39_challenger_decoder.layers:
                layer.task_decoder.plan_cls_branch.requires_grad_(False)
        elif training_mode in {
            "generation", "generation_group", "generation_group_adaptive",
            "diffgrpo_full_chain", "diffgrpo_selected_anchor",
            "diffgrpo_selected_anchor_base_preserve", "joint",
            "diffgrpo_selected_set",
            "stage27_public_diffgrpo_selected_set",
        } | STAGE28_GRPO_MODES | STAGE29_GRPO_MODES | STAGE30_GRPO_MODES \
                | STAGE31_GRPO_MODES | STAGE32_GRPO_MODES | STAGE33_GRPO_MODES \
                | STAGE34_GRPO_MODES | STAGE35_GRPO_MODES | STAGE36_GRPO_MODES \
                | STAGE37_GRPO_MODES | STAGE38_GRPO_MODES | STAGE39_GRPO_MODES:
            trajectory_head.diff_decoder.requires_grad_(True)
            if training_mode in {
                "generation", "generation_group", "generation_group_adaptive",
                "diffgrpo_full_chain", "diffgrpo_selected_anchor",
                "diffgrpo_selected_anchor_base_preserve",
                "diffgrpo_selected_set",
                "stage27_public_diffgrpo_selected_set",
            } | STAGE28_GRPO_MODES | STAGE29_GRPO_MODES | STAGE30_GRPO_MODES \
            | STAGE31_GRPO_MODES | STAGE32_GRPO_MODES | STAGE33_GRPO_MODES \
            | STAGE34_GRPO_MODES | STAGE35_GRPO_MODES | STAGE36_GRPO_MODES \
            | STAGE37_GRPO_MODES | STAGE38_GRPO_MODES | STAGE39_GRPO_MODES:
                for layer in trajectory_head.diff_decoder.layers:
                    layer.task_decoder.plan_cls_branch.requires_grad_(False)
        elif training_mode in STAGE22_GRPO_MODES:
            # The plain base decoder remains frozen until adapters are attached
            # after the frozen reference snapshot has been constructed.
            pass
        else:
            raise ValueError(
                "grpo_training_mode must be one of "
                "{'classification_shared', 'classification_head', 'selector', "
                "'generation', 'joint', 'selector_group', "
                "'generation_group', 'generation_group_adaptive', "
                "'diffgrpo_full_chain', 'diffgrpo_selected_anchor', "
                "'diffgrpo_selected_anchor_base_preserve', "
                "'diffgrpo_paired_residual', "
                "'diffgrpo_selected_set', "
                "'stage27_public_diffgrpo_selected_set', "
                "'stage28_public_paired_uplift_multi', "
                "'stage28_public_paired_uplift_explore', "
                "'stage29_public_headroom_hybrid', "
                "'stage29_public_headroom_conditional', "
                "'stage30_public_mode_coverage', "
                "'stage30_public_mode_coverage_constrained', "
                "'stage31_public_deployed_pair', "
                "'stage31_public_deployed_frontier', "
                "'stage32_public_deployed_extended', "
                "'stage32_selector_aware_frontier', "
                "'stage33_cdc_grpo', "
                "'stage34_mode_aligned_frontier_grpo', "
                "'stage35_nested_counterfactual_deployment_grpo', "
                "'stage36_reference_gated_tail_ncd_grpo', "
                "'stage37_bistate_projected_deployment_grpo', "
                "'stage38_elite_set_counterfactual_repair_grpo', "
                "'value_selector', 'stage23_selector', 'stage24_selector', "
                "'stage25_relative_harm_selector', "
                "'stage37_jfi_selector', "
                "'paired_tail_risk_selector'}; "
                f"got {training_mode!r}"
            )

        ref_policy = copy.deepcopy(self._transfuser_model._trajectory_head.diff_decoder)
        reference_checkpoint = torch.load(self._reference_checkpoint_path, map_location="cpu")
        if "state_dict" not in reference_checkpoint:
            raise KeyError(f"Reference checkpoint has no state_dict: {self._reference_checkpoint_path}")
        normalized_reference_state = {
            (key[len("agent."):] if key.startswith("agent.") else key): value
            for key, value in reference_checkpoint["state_dict"].items()
        }
        decoder_prefix = "_transfuser_model._trajectory_head.diff_decoder."
        reference_decoder_state = {
            key[len(decoder_prefix):]: value
            for key, value in normalized_reference_state.items()
            if key.startswith(decoder_prefix)
        }
        if not reference_decoder_state:
            raise RuntimeError(
                f"Reference checkpoint has no diff_decoder weights: {self._reference_checkpoint_path}"
            )
        self._reference_decoder_state = {
            name: value.detach().cpu().clone()
            for name, value in reference_decoder_state.items()
        }
        ref_policy.load_state_dict(reference_decoder_state, strict=True)
        ref_policy.requires_grad_(False)
        ref_policy.eval()

        old_policy = copy.deepcopy(self._transfuser_model._trajectory_head.diff_decoder)
        old_policy.requires_grad_(False)
        old_policy.eval()

        # 3. 设置到TrajectoryHead中
        self._transfuser_model._trajectory_head.set_ref_policy(ref_policy)
        self._transfuser_model._trajectory_head.set_old_policy(old_policy)
        self._stage22_trainable_names = ()
        if training_mode in STAGE22_GRPO_MODES:
            self._stage22_trainable_names = attach_stage22_lora(
                self._transfuser_model._trajectory_head.diff_decoder,
                rank=int(getattr(config, "diffgrpo_lora_rank", 8)),
                alpha=float(getattr(config, "diffgrpo_lora_alpha", 8.0)),
            )
        self._transfuser_model._trajectory_head.validate_generation_trust_runtime(
            self._reference_checkpoint_path
        )

        trainable_params = sum(
            parameter.numel()
            for parameter in self._transfuser_model.parameters()
            if parameter.requires_grad
        )
        if training_mode in (
            SELECTED_SET_GRPO_MODES | STAGE30_GRPO_MODES | STAGE34_GRPO_MODES
            | STAGE35_GRPO_MODES | STAGE36_GRPO_MODES | STAGE37_GRPO_MODES
            | STAGE38_GRPO_MODES | STAGE39_GRPO_MODES
        ):
            trainable_tensors = [
                name for name, parameter in self._transfuser_model.named_parameters()
                if parameter.requires_grad
            ]
            if len(trainable_tensors) != 64:
                raise RuntimeError(
                    "selected-set/mode-coverage GRPO requires exactly 64 trainable tensors; "
                    f"got {len(trainable_tensors)}"
                )
        print(f"✓ GRPO train mode={training_mode}; trainable parameters={trainable_params:,}")

        print(
            "✓ Reference policy loaded from "
            f"{self._reference_checkpoint_path}; old policy copied from current checkpoint"
        )

    def validate_reference_policy_immutability(self) -> None:
        """Validate the in-memory reference after construction or resume."""
        head = self._transfuser_model._trajectory_head
        if head.ref_policy is None:
            raise RuntimeError("GRPO has no frozen reference policy")
        validate_frozen_policy_state(
            head.ref_policy, self._reference_decoder_state
        )
        
    def init_from_pretrained(self):
        checkpoint = torch.load(self._checkpoint_path, map_location="cpu")
        if "state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint has no state_dict: {self._checkpoint_path}")

        state_dict = {
            (key[len("agent."):] if key.startswith("agent.") else key): value
            for key, value in checkpoint["state_dict"].items()
        }
        stage39_prefix = "_transfuser_model._trajectory_head.stage39_challenger_decoder."
        stage39_loaded = any(key.startswith(stage39_prefix) for key in state_dict)
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        policy_snapshot_prefixes = (
            "_transfuser_model._trajectory_head.ref_policy.",
            "_transfuser_model._trajectory_head.old_policy.",
            "_adaptive_kl_controller.",
        )
        unexpected_keys = [key for key in unexpected_keys if not key.startswith(policy_snapshot_prefixes)]

        critical_prefixes = (
            "_transfuser_model._backbone.",
            "_transfuser_model._trajectory_head.diff_decoder.",
        )
        critical_missing = [key for key in missing_keys if key.startswith(critical_prefixes)]
        if critical_missing:
            raise RuntimeError(
                "Checkpoint is missing critical pretrained weights: "
                + ", ".join(critical_missing[:20])
            )
        optional_missing_prefixes = (
            "_transfuser_model._trajectory_head.value_selector.",
            "_transfuser_model._trajectory_head._value_selector_training_updates",
            "_transfuser_model._trajectory_head.stage23_selector.",
            "_transfuser_model._trajectory_head._stage23_selector_training_updates",
            "_transfuser_model._trajectory_head.stage24_selector.",
            "_transfuser_model._trajectory_head._stage24_selector_training_updates",
            "_transfuser_model._trajectory_head._stage24_ood_",
            "_transfuser_model._trajectory_head._stage24_calibration_loaded",
            "_transfuser_model._trajectory_head._stage24_embedding_",
            "_transfuser_model._trajectory_head.stage25_selector.",
            "_transfuser_model._trajectory_head._stage25_selector_training_updates",
            "_transfuser_model._trajectory_head.stage37_jfi_selector.",
            "_transfuser_model._trajectory_head.stage39_challenger_decoder.",
            "_transfuser_model._trajectory_head._stage37_jfi_training_updates",
            "_transfuser_model._trajectory_head.paired_risk_head.",
            "_transfuser_model._trajectory_head._paired_risk_training_updates",
        )
        missing_keys = [
            key for key in missing_keys
            if not key.startswith(optional_missing_prefixes)
        ]
        if missing_keys:
            print(f"Non-critical missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected checkpoint keys: {unexpected_keys}")
        print(f"✓ Loaded pretrained checkpoint: {self._checkpoint_path}")
        head = self._transfuser_model._trajectory_head
        if head.stage39_challenger_decoder is not None and not stage39_loaded:
            head.stage39_challenger_decoder.load_state_dict(
                head.diff_decoder.state_dict(), strict=True
            )
        self._stage39_challenger_loaded = bool(stage39_loaded)

    def load_value_selector_checkpoint(self, checkpoint_path: str) -> None:
        """Load only the Stage-15 adapter so one selector serves every generator."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Value-selector checkpoint is missing: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        if "state_dict" not in checkpoint:
            raise KeyError(f"Value-selector checkpoint has no state_dict: {path}")
        normalized = {
            (key[len("agent."):] if key.startswith("agent.") else key): value
            for key, value in checkpoint["state_dict"].items()
        }
        prefix = "_transfuser_model._trajectory_head.value_selector."
        selector_state = {
            key[len(prefix):]: value
            for key, value in normalized.items()
            if key.startswith(prefix)
        }
        if not selector_state:
            raise RuntimeError(f"Checkpoint contains no Stage-15 selector: {path}")
        head = self._transfuser_model._trajectory_head
        head.value_selector.load_state_dict(selector_state, strict=True)
        update_key = (
            "_transfuser_model._trajectory_head._value_selector_training_updates"
        )
        if update_key in normalized:
            head._value_selector_training_updates.copy_(normalized[update_key])
        else:
            head._value_selector_training_updates.fill_(1)
        self._value_selector_checkpoint_sha256 = file_sha256(path)
        print(f"✓ Loaded Stage-15 value selector: {path}")

    def load_stage23_selector_checkpoint(self, checkpoint_path: str) -> None:
        """Load only the deployable Stage23 five-member selector."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Stage23 selector checkpoint is missing: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        if "state_dict" not in checkpoint:
            raise KeyError(f"Stage23 selector checkpoint has no state_dict: {path}")
        normalized = {
            (key[len("agent."):] if key.startswith("agent.") else key): value
            for key, value in checkpoint["state_dict"].items()
        }
        prefix = "_transfuser_model._trajectory_head.stage23_selector."
        selector_state = {
            key[len(prefix):]: value
            for key, value in normalized.items()
            if key.startswith(prefix)
        }
        if not selector_state:
            raise RuntimeError(f"Checkpoint contains no Stage23 selector: {path}")
        head = self._transfuser_model._trajectory_head
        head.stage23_selector.load_state_dict(selector_state, strict=True)
        update_key = (
            "_transfuser_model._trajectory_head._stage23_selector_training_updates"
        )
        if update_key not in normalized or int(normalized[update_key].item()) <= 0:
            raise RuntimeError("Stage23 selector checkpoint has no training provenance")
        head._stage23_selector_training_updates.copy_(normalized[update_key])
        self._stage23_selector_checkpoint_sha256 = file_sha256(path)
        print(f"✓ Loaded Stage23 trajectory selector: {path}")

    def load_stage24_selector_checkpoint(self, checkpoint_path: str) -> None:
        """Load only the deployable Stage24 eight-member selector."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Stage24 selector checkpoint is missing: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        if "state_dict" not in checkpoint:
            raise KeyError(f"Stage24 selector checkpoint has no state_dict: {path}")
        normalized = {
            (key[len("agent."):] if key.startswith("agent.") else key): value
            for key, value in checkpoint["state_dict"].items()
        }
        prefix = "_transfuser_model._trajectory_head.stage24_selector."
        selector_state = {
            key[len(prefix):]: value
            for key, value in normalized.items()
            if key.startswith(prefix)
        }
        if not selector_state:
            raise RuntimeError(f"Checkpoint contains no Stage24 selector: {path}")
        head = self._transfuser_model._trajectory_head
        head.stage24_selector.load_state_dict(selector_state, strict=True)
        update_key = (
            "_transfuser_model._trajectory_head._stage24_selector_training_updates"
        )
        if update_key not in normalized or int(normalized[update_key].item()) <= 0:
            raise RuntimeError("Stage24 selector checkpoint has no training provenance")
        head._stage24_selector_training_updates.copy_(normalized[update_key])
        moment_names = (
            "_stage24_embedding_count",
            "_stage24_embedding_sum",
            "_stage24_embedding_sum_sq",
        )
        for name in moment_names:
            key = f"_transfuser_model._trajectory_head.{name}"
            if key not in normalized:
                raise RuntimeError(
                    f"Stage24 selector checkpoint lacks OOD moment: {name}"
                )
            getattr(head, name).copy_(normalized[key])
        if int(head._stage24_embedding_count.item()) <= 0:
            raise RuntimeError("Stage24 selector checkpoint has empty OOD moments")
        self._stage24_selector_checkpoint_sha256 = file_sha256(path)
        print(f"✓ Loaded Stage24 safety/value selector: {path}")

    def load_stage25_selector_checkpoint(self, checkpoint_path: str) -> None:
        """Load frozen Stage24 value state plus the Stage25 harm heads."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Stage25 checkpoint is missing: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        if "state_dict" not in checkpoint:
            raise KeyError(f"Stage25 checkpoint has no state_dict: {path}")
        normalized = {
            (key[len("agent."):] if key.startswith("agent.") else key): value
            for key, value in checkpoint["state_dict"].items()
        }
        head = self._transfuser_model._trajectory_head
        for module_name in ("stage24_selector", "stage25_selector"):
            prefix = f"_transfuser_model._trajectory_head.{module_name}."
            module_state = {
                key[len(prefix):]: value
                for key, value in normalized.items()
                if key.startswith(prefix)
            }
            if not module_state:
                raise RuntimeError(
                    f"Stage25 checkpoint lacks {module_name}: {path}"
                )
            getattr(head, module_name).load_state_dict(
                module_state, strict=True
            )
        for name in (
            "_stage24_selector_training_updates",
            "_stage25_selector_training_updates",
            "_stage24_embedding_count",
            "_stage24_embedding_sum",
            "_stage24_embedding_sum_sq",
        ):
            key = f"_transfuser_model._trajectory_head.{name}"
            if key not in normalized:
                raise RuntimeError(f"Stage25 checkpoint lacks {name}")
            getattr(head, name).copy_(normalized[key])
        if (
            int(head._stage24_selector_training_updates.item()) <= 0
            or int(head._stage25_selector_training_updates.item()) <= 0
            or int(head._stage24_embedding_count.item()) <= 0
        ):
            raise RuntimeError("Stage25 checkpoint has invalid training provenance")
        self._stage25_selector_checkpoint_sha256 = file_sha256(path)
        print(f"✓ Loaded Stage25 relative-harm selector: {path}")

    def load_stage37_jfi_checkpoint(self, checkpoint_path: str) -> None:
        """Load the frozen Stage24 encoder and Stage37 JFI heads only."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Stage37 JFI checkpoint is missing: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        if "state_dict" not in checkpoint:
            raise KeyError(f"Stage37 JFI checkpoint has no state_dict: {path}")
        normalized = {
            (key[len("agent."):] if key.startswith("agent.") else key): value
            for key, value in checkpoint["state_dict"].items()
        }
        head = self._transfuser_model._trajectory_head
        for module_name in ("stage24_selector", "stage37_jfi_selector"):
            prefix = f"_transfuser_model._trajectory_head.{module_name}."
            state = {
                key[len(prefix):]: value for key, value in normalized.items()
                if key.startswith(prefix)
            }
            if not state:
                raise RuntimeError(f"JFI checkpoint lacks {module_name}: {path}")
            getattr(head, module_name).load_state_dict(state, strict=True)
        for name in (
            "_stage24_selector_training_updates",
            "_stage37_jfi_training_updates",
            "_stage24_embedding_count",
            "_stage24_embedding_sum",
            "_stage24_embedding_sum_sq",
        ):
            key = f"_transfuser_model._trajectory_head.{name}"
            if key not in normalized:
                raise RuntimeError(f"JFI checkpoint lacks {name}")
            getattr(head, name).copy_(normalized[key])
        if (
            int(head._stage24_selector_training_updates.item()) <= 0
            or int(head._stage37_jfi_training_updates.item()) <= 0
            or int(head._stage24_embedding_count.item()) <= 0
        ):
            raise RuntimeError("JFI checkpoint has invalid training provenance")
        self._stage37_jfi_checkpoint_sha256 = file_sha256(path)
        print(f"✓ Loaded Stage37 JFI selector: {path}")
    def load_paired_risk_checkpoint(self, checkpoint_path: str) -> None:
        """Load only the Stage-17 adapter and its update provenance."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Paired-risk checkpoint is missing: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        if "state_dict" not in checkpoint:
            raise KeyError(f"Paired-risk checkpoint has no state_dict: {path}")
        normalized = {
            (key[len("agent."):] if key.startswith("agent.") else key): value
            for key, value in checkpoint["state_dict"].items()
        }
        prefix = "_transfuser_model._trajectory_head.paired_risk_head."
        adapter_state = {
            key[len(prefix):]: value
            for key, value in normalized.items()
            if key.startswith(prefix)
        }
        if not adapter_state:
            raise RuntimeError(f"Checkpoint contains no Stage-17 adapter: {path}")
        head = self._transfuser_model._trajectory_head
        head.paired_risk_head.load_state_dict(adapter_state, strict=True)
        update_key = (
            "_transfuser_model._trajectory_head._paired_risk_training_updates"
        )
        if update_key in normalized:
            head._paired_risk_training_updates.copy_(normalized[update_key])
        else:
            head._paired_risk_training_updates.fill_(1)
        self._paired_risk_checkpoint_sha256 = file_sha256(path)
        print(f"✓ Loaded Stage-17 paired-risk adapter: {path}")
    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()},strict=False)


    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=[3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TransfuserTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [TransfuserFeatureBuilder(config=self._config)]

    def _enforce_frozen_eval_mode(self) -> None:
        """Keep frozen perception/planning context deterministic during GRPO."""
        model = self._transfuser_model
        frozen_modules = (
            model._backbone,
            model._tf_decoder,
            model._agent_head,
            model._bev_semantic_head,
            model._bev_downscale,
            model._status_encoding,
            model._keyval_embedding,
            model._query_embedding,
            model.bev_proj,
            model._trajectory_head.plan_anchor_encoder,
            model._trajectory_head.time_mlp,
        )
        for module in frozen_modules:
            module.eval()
        model._trajectory_head.diff_decoder.eval()
        model._trajectory_head.value_selector.eval()
        model._trajectory_head.stage23_selector.eval()
        model._trajectory_head.stage24_selector.eval()
        model._trajectory_head.stage25_selector.eval()
        model._trajectory_head.stage37_jfi_selector.eval()
        model._trajectory_head.paired_risk_head.eval()
        if (
            self.training
            and str(getattr(self._config, "grpo_training_mode", ""))
            == "value_selector"
        ):
            model._trajectory_head.value_selector.train()
        if (
            self.training
            and str(getattr(self._config, "grpo_training_mode", ""))
            == "stage23_selector"
        ):
            model._trajectory_head.stage23_selector.train()
        if (
            self.training
            and str(getattr(self._config, "grpo_training_mode", ""))
            == "stage24_selector"
        ):
            model._trajectory_head.stage24_selector.train()
        if (
            self.training
            and str(getattr(self._config, "grpo_training_mode", ""))
            == "stage37_jfi_selector"
        ):
            model._trajectory_head.stage37_jfi_selector.train()
        if (
            self.training
            and str(getattr(self._config, "grpo_training_mode", ""))
            == "paired_tail_risk_selector"
        ):
            model._trajectory_head.paired_risk_head.train()
        if model._trajectory_head.ref_policy is not None:
            model._trajectory_head.ref_policy.eval()
        if model._trajectory_head.old_policy is not None:
            model._trajectory_head.old_policy.eval()

    def set_evaluation_token(self, token: str) -> None:
        """Provide the scenario identity used for deterministic diffusion noise."""
        self._evaluation_token = str(token)

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None, tokens_list=None) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        if (
            str(getattr(self._config, "grpo_training_mode", ""))
            == "stage28_public_paired_uplift_explore"
            and not self.training
        ):
            raise RuntimeError(
                "Stage28 S-public exploration mode is training-only and "
                "cannot run inference/deployment"
            )
        self._enforce_frozen_eval_mode()
        if tokens_list is None and not self.training and self._evaluation_token is not None:
            tokens_list = (self._evaluation_token,)
        return self._transfuser_model(features, targets=targets, tokens_list=tokens_list)
        
    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        if self._adaptive_kl_controller is not None:
            predictions = dict(predictions)
            predictions["generation_kl_coefficient"] = (
                self._adaptive_kl_controller.coefficient.detach().float()
            )
        return transfuser_loss(targets, predictions, self._config)

    def update_generation_kl_controller(
        self, generation_kl: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if self._adaptive_kl_controller is None:
            return {}
        return self._adaptive_kl_controller.update(generation_kl)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        if str(getattr(self._config, "grpo_training_mode", "")) in (
            SELECTED_SET_GRPO_MODES | STAGE30_GRPO_MODES | STAGE34_GRPO_MODES
            | STAGE35_GRPO_MODES | STAGE36_GRPO_MODES | STAGE37_GRPO_MODES
            | STAGE38_GRPO_MODES
        ):
            groups = build_stage10_decoder_param_groups(
                self._transfuser_model.named_parameters(), 0.1
            )
            groups[0]["lr"] = float(self._lr)
            groups[1]["lr"] = float(self._lr) * 0.1
            optimizer = torch.optim.AdamW(
                groups, lr=float(self._lr), weight_decay=float(self._config.weight_decay)
            )
            actual_lrs = sorted(group["lr"] for group in optimizer.param_groups)
            if actual_lrs != [1e-7, 1e-6]:
                raise RuntimeError(
                    f"selected-set GRPO optimizer LR drift: {actual_lrs}"
                )
            return optimizer
        if str(getattr(self._config, "grpo_training_mode", "")) in STAGE22_GRPO_MODES:
            parameters = [
                parameter for parameter in self._transfuser_model.parameters()
                if parameter.requires_grad
            ]
            if len(parameters) != 4:
                raise RuntimeError(
                    f"Stage22 optimizer expected four LoRA tensors, got {len(parameters)}"
                )
            return torch.optim.AdamW(
                parameters, lr=self._lr, weight_decay=0.0
            )
        return self.get_coslr_optimizers()

    def get_step_lr_optimizers(self):
        optimizer = torch.optim.Adam(
            [p for p in self._transfuser_model.parameters() if p.requires_grad],
            lr=self._lr, weight_decay=self._config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self._config.lr_steps, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_coslr_optimizers(self):
        # import ipdb; ipdb.set_trace()
        optimizer_cfg = dict(type=self._config.optimizer_type, 
                            lr=self._lr, 
                            weight_decay=self._config.weight_decay,
                            paramwise_cfg=self._config.opt_paramwise_cfg
                            )
        scheduler_cfg = dict(type=self._config.scheduler_type,
                            milestones=self._config.lr_steps,
                            gamma=0.1,
        )

        optimizer_cfg = DictConfig(optimizer_cfg)
        scheduler_cfg = DictConfig(scheduler_cfg)
        
        with open_dict(optimizer_cfg):
            paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)

        stage10_param_groups = None
        if str(getattr(self._config, "grpo_training_mode", "")) in (
            STAGE10_GRPO_MODES | STAGE21_GRPO_MODES | STAGE23_GRPO_MODES
            | STAGE27_GRPO_MODES | STAGE28_GRPO_MODES | STAGE29_GRPO_MODES
            | STAGE30_GRPO_MODES | STAGE31_GRPO_MODES | STAGE32_GRPO_MODES
            | STAGE33_GRPO_MODES | STAGE34_GRPO_MODES | STAGE35_GRPO_MODES
            | STAGE36_GRPO_MODES | STAGE37_GRPO_MODES | STAGE38_GRPO_MODES | STAGE39_GRPO_MODES
        ):
            stage10_param_groups = build_stage10_decoder_param_groups(
                self._transfuser_model.named_parameters(),
                self._config.grpo_decoder_layer0_lr_mult,
            )
            paramwise_cfg = None

        if stage10_param_groups is not None:
            params = stage10_param_groups
        elif paramwise_cfg:
            params = []
            pgs = [[] for _ in paramwise_cfg['name']]

            for k, v in self._transfuser_model.named_parameters():
                if not v.requires_grad:
                    continue
                in_param_group = True
                for i, (pattern, pg_cfg) in enumerate(paramwise_cfg['name'].items()):
                    if pattern in k:
                        pgs[i].append(v)
                        in_param_group = False
                if in_param_group:
                    params.append(v)
        else:
            params = [p for p in self._transfuser_model.parameters() if p.requires_grad]
        
        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        # import ipdb; ipdb.set_trace()
        if paramwise_cfg:
            for pg, (_, pg_cfg) in zip(pgs, paramwise_cfg['name'].items()):
                cfg = {}
                if 'lr_mult' in pg_cfg:
                    cfg['lr'] = optimizer_cfg['lr'] * pg_cfg['lr_mult']
                optimizer.add_param_group({'params': pg, **cfg})
        
        # scheduler = build_from_configs(optim.lr_scheduler, scheduler_cfg, optimizer=optimizer)
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=1e-6,
            epochs=100,
            warmup_epochs=3,
        )
        
        if 'interval' in scheduler_cfg:
            scheduler = {'scheduler': scheduler, 'interval': scheduler_cfg['interval']}
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        training_mode = str(getattr(self._config, "grpo_training_mode", ""))
        checkpoint_save_top_k = int(
            getattr(self._config, "grpo_checkpoint_save_top_k", 2)
        )
        if (
            training_mode in FORMAL_GRPO_MODES
            or training_mode in {
                "value_selector", "stage23_selector", "stage24_selector",
                "paired_tail_risk_selector"
            }
            or checkpoint_save_top_k == -1
        ):
            # Stage 9 has no on-policy validation rollout: PDMS selection is
            # performed by the independent fixed-set evaluator. Preserve each
            # completed epoch instead of monitoring an undefined val metric.
            selection_checkpoint = ModelCheckpoint(
                filename="grpo-{epoch:02d}-{step}",
                every_n_epochs=1,
                save_top_k=-1,
                save_last=True,
                auto_insert_metric_name=False,
            )
        else:
            selection_checkpoint = ModelCheckpoint(
                filename="grpo-{epoch:02d}-{step}",
                monitor="val/selected_reward_epoch",
                mode="max",
                save_top_k=checkpoint_save_top_k,
                save_last=True,
                auto_insert_metric_name=False,
            )
        callbacks = [
            TransfuserCallback(self._config),
            selection_checkpoint,
        ]
        checkpoint_every = int(
            getattr(self._config, "grpo_checkpoint_every_n_train_steps", 0)
        )
        if checkpoint_every > 0:
            callbacks.append(
                ModelCheckpoint(
                    filename="grpo-step-{step}",
                    every_n_train_steps=checkpoint_every,
                    save_top_k=-1,
                    save_on_train_epoch_end=False,
                    auto_insert_metric_name=False,
                )
            )
        return callbacks
