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
from navsim.agents.diffusiondrive.diffusion_grpo import file_sha256

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


FORMAL_GRPO_MODES = {"selector_group", "generation_group"}
FORMAL_BASE_SHA256 = (
    "59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d"
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
    """Fail closed if a stage-9 objective drifts from its registration."""
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
        },
        "generation_group": {
            "policy_loss_weight": 0.0,
            "kl_loss_weight": 0.0,
            "selector_generation_kl_weight": 0.0,
            "generation_policy_loss_weight": 1.0,
            "generation_kl_loss_weight": 0.1,
        },
    }[mode]
    for name, wanted in {**expected, **route_expected}.items():
        actual = getattr(config, name, None)
        matches = (
            math.isclose(float(actual), wanted, rel_tol=0.0, abs_tol=1e-12)
            if isinstance(wanted, float) else actual == wanted
        )
        if not matches:
            raise ValueError(
                f"formal {mode} requires {name}={wanted!r}; got {actual!r}"
            )

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
                    "Formal stage-9 reference checkpoint is not the registered base"
                )
            if self._checkpoint_sha256 != self._reference_checkpoint_sha256:
                raise RuntimeError(
                    "Formal stage-9 current policy must initialize from frozen base"
                )
            if not math.isclose(float(lr), 1e-6, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("Formal stage-9 learning rate must be 1e-6")
        self._reference_checkpoint_path = reference_checkpoint_path
        self._transfuser_model = TransfuserModel(config)
        self.init_from_pretrained()

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
        elif training_mode in {"generation", "generation_group", "joint"}:
            trajectory_head.diff_decoder.requires_grad_(True)
            if training_mode in {"generation", "generation_group"}:
                for layer in trajectory_head.diff_decoder.layers:
                    layer.task_decoder.plan_cls_branch.requires_grad_(False)
        else:
            raise ValueError(
                "grpo_training_mode must be one of "
                "{'classification_shared', 'classification_head', 'selector', "
                "'generation', 'joint', 'selector_group', "
                "'generation_group'}; "
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
        if missing_keys:
            print(f"Non-critical missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected checkpoint keys: {unexpected_keys}")
        print(f"✓ Loaded pretrained checkpoint: {self._checkpoint_path}")
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
        return transfuser_loss(targets, predictions, self._config)

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
        
        if paramwise_cfg:
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
        if training_mode in FORMAL_GRPO_MODES:
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
                save_top_k=int(
                    getattr(self._config, "grpo_checkpoint_save_top_k", 2)
                ),
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
