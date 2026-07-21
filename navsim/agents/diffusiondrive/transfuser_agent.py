from typing import Any, List, Dict, Optional, Union
from pathlib import Path

import copy
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
FORMAL_GRPO_MODES = (
    STAGE9_GRPO_MODES
    | STAGE10_GRPO_MODES
    | STAGE16_GRPO_MODES
    | STAGE19_GRPO_MODES
)
FORMAL_BASE_SHA256 = (
    "59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d"
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


def validate_formal_grpo_config(config: TransfuserConfig) -> None:
    """Fail closed if a registered Stage-9/10 objective drifts."""
    mode = str(getattr(config, "grpo_training_mode", ""))
    if mode not in FORMAL_GRPO_MODES:
        return
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
    }[mode]
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


def build_stage10_decoder_param_groups(
    named_parameters, layer0_lr_mult: float
) -> List[Dict[str, Any]]:
    """Split trainable layer-0 parameters without freezing either decoder layer."""
    layer0 = []
    remaining = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if "_trajectory_head.diff_decoder.layers.0." in name:
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
        if str(getattr(config, "grpo_training_mode", "")) in FORMAL_GRPO_MODES:
            if self._reference_checkpoint_sha256 != FORMAL_BASE_SHA256:
                raise RuntimeError(
                    "Formal GRPO reference checkpoint is not the registered base"
                )
            if self._checkpoint_sha256 != self._reference_checkpoint_sha256:
                raise RuntimeError(
                    "Formal GRPO current policy must initialize from frozen base"
                )
            if not math.isclose(float(lr), 1e-6, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("Formal GRPO decoder learning rate must be 1e-6")
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
                not in {"diffgrpo_full_chain", "diffgrpo_selected_anchor"}
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
        elif training_mode == "paired_tail_risk_selector":
            trajectory_head.paired_risk_head.requires_grad_(True)
        elif training_mode in {
            "generation", "generation_group", "generation_group_adaptive",
            "diffgrpo_full_chain", "diffgrpo_selected_anchor", "joint"
        }:
            trajectory_head.diff_decoder.requires_grad_(True)
            if training_mode in {
                "generation", "generation_group", "generation_group_adaptive",
                "diffgrpo_full_chain", "diffgrpo_selected_anchor"
            }:
                for layer in trajectory_head.diff_decoder.layers:
                    layer.task_decoder.plan_cls_branch.requires_grad_(False)
        else:
            raise ValueError(
                "grpo_training_mode must be one of "
                "{'classification_shared', 'classification_head', 'selector', "
                "'generation', 'joint', 'selector_group', "
                "'generation_group', 'generation_group_adaptive', "
                "'diffgrpo_full_chain', 'diffgrpo_selected_anchor', "
                "'value_selector', 'paired_tail_risk_selector'}; "
                f"got {training_mode!r}"
            )
        trainable_params = sum(
            parameter.numel()
            for parameter in self._transfuser_model.parameters()
            if parameter.requires_grad
        )
        print(f"✓ GRPO train mode={training_mode}; trainable parameters={trainable_params:,}")

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
        self._transfuser_model._trajectory_head.validate_generation_trust_runtime(
            self._reference_checkpoint_path
        )

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
        if str(getattr(self._config, "grpo_training_mode", "")) in STAGE10_GRPO_MODES:
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
            or training_mode in {"value_selector", "paired_tail_risk_selector"}
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
