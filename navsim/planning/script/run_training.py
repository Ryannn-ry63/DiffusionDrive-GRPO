from typing import Tuple
from pathlib import Path
import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.grpo_sampler import (
    BalancedPriorityBatchSampler,
    Stage30GlobalBucketBatchSampler,
    load_manifest_tokens,
)
from navsim.agents.diffusiondrive.stage30_mode_coverage import (
    load_stage30_bucket_manifest,
)

from pytorch_lightning.loggers import TensorBoardLogger  #
from pytorch_lightning.strategies import DDPStrategy 
import torch
import torch.nn.utils.rnn as rnn_utils
from typing import List, Dict

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"

def custom_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    features_list, targets_list, tokens_list = zip(*batch)

    camera_feature = torch.stack([features['camera_feature'] for features in features_list], dim=0).cpu()
    lidar_feature = torch.stack([features['lidar_feature'] for features in features_list], dim=0).cpu()
    status_feature = torch.stack([features['status_feature'] for features in features_list], dim=0).cpu()

    trajectory = torch.stack([targets['trajectory'] for targets in targets_list], dim=0).cpu()


    features = {
        'camera_feature': camera_feature,
        'lidar_feature': lidar_feature,
        'status_feature': status_feature,
    }
    targets = {
        'trajectory': trajectory
    }

    return features, targets, tokens_list

def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    agent_config = getattr(agent, "_config", None)
    if getattr(agent_config, "filter_training_by_metric_cache", False):
        metric_cache_path = Path(agent_config.metric_cache_path)
        assert metric_cache_path.is_dir(), f"Metric cache path {metric_cache_path} does not exist!"
        metric_tokens = set(MetricCacheLoader(metric_cache_path).tokens)
        num_train_before, num_val_before = len(train_data), len(val_data)
        train_data.tokens = [token for token in train_data.tokens if token in metric_tokens]
        val_data.tokens = [token for token in val_data.tokens if token in metric_tokens]
        logger.info(
            "Filtered datasets to rewardable metric-cache tokens: train %d/%d, val %d/%d",
            len(train_data), num_train_before, len(val_data), num_val_before,
        )
        assert len(train_data) > 0, "No training token overlaps metric_cache_path!"
        assert len(val_data) > 0, "No validation token overlaps metric_cache_path!"

    value_manifest_path = str(
        getattr(agent_config, "value_selector_train_manifest_path", "")
    )
    if str(getattr(agent_config, "grpo_training_mode", "")) == "value_selector":
        if not value_manifest_path:
            raise ValueError(
                "value_selector training requires value_selector_train_manifest_path"
            )
        value_token_list = load_manifest_tokens(Path(value_manifest_path))
        value_tokens = set(value_token_list)
        available_tokens = set(train_data.tokens)
        missing_tokens = sorted(value_tokens - available_tokens)
        if missing_tokens:
            raise RuntimeError(
                f"Value-selector manifest has {len(missing_tokens)} unavailable tokens; "
                f"first={missing_tokens[0]}"
            )
        train_data.tokens = [
            token for token in train_data.tokens if token in value_tokens
        ]
        if len(train_data.tokens) != len(value_token_list):
            raise RuntimeError("Value-selector manifest filtering changed token count")
        logger.info(
            "Stage-15 selector manifest: %d training tokens from %s",
            len(train_data.tokens), value_manifest_path,
        )
    elif value_manifest_path:
        raise ValueError(
            "value_selector_train_manifest_path is valid only in value_selector mode"
        )

    stage23_manifest_path = str(
        getattr(agent_config, "stage23_selector_train_manifest_path", "")
    )
    if str(getattr(agent_config, "grpo_training_mode", "")) == "stage23_selector":
        if not stage23_manifest_path:
            raise ValueError(
                "stage23_selector training requires stage23_selector_train_manifest_path"
            )
        stage23_token_list = load_manifest_tokens(Path(stage23_manifest_path))
        stage23_tokens = set(stage23_token_list)
        missing_tokens = sorted(stage23_tokens - set(train_data.tokens))
        if missing_tokens:
            raise RuntimeError(
                f"Stage23 manifest has {len(missing_tokens)} unavailable tokens; "
                f"first={missing_tokens[0]}"
            )
        train_data.tokens = [
            token for token in train_data.tokens if token in stage23_tokens
        ]
        if len(train_data.tokens) != len(stage23_token_list):
            raise RuntimeError("Stage23 manifest filtering changed token count")
        logger.info(
            "Stage23 whole-log selector manifest: %d tokens from %s",
            len(train_data.tokens), stage23_manifest_path,
        )
    elif stage23_manifest_path:
        raise ValueError(
            "stage23_selector_train_manifest_path is valid only in stage23_selector mode"
        )

    stage24_manifest_path = str(
        getattr(agent_config, "stage24_selector_train_manifest_path", "")
    )
    stage24_bank_modes = {
        "stage24_selector", "stage25_relative_harm_selector"
    }
    if str(getattr(agent_config, "grpo_training_mode", "")) in stage24_bank_modes:
        if not stage24_manifest_path:
            raise ValueError(
                "Stage24/25 selector training requires "
                "stage24_selector_train_manifest_path"
            )
        stage24_token_list = load_manifest_tokens(Path(stage24_manifest_path))
        stage24_tokens = set(stage24_token_list)
        missing_tokens = sorted(stage24_tokens - set(train_data.tokens))
        if missing_tokens:
            raise RuntimeError(
                f"Stage24 manifest has {len(missing_tokens)} unavailable tokens; "
                f"first={missing_tokens[0]}"
            )
        train_data.tokens = [
            token for token in train_data.tokens if token in stage24_tokens
        ]
        if len(train_data.tokens) != len(stage24_token_list):
            raise RuntimeError("Stage24 manifest filtering changed token count")
        logger.info(
            "Stage24 whole-log selector manifest: %d tokens from %s",
            len(train_data.tokens), stage24_manifest_path,
        )
    elif stage24_manifest_path:
        raise ValueError(
            "stage24_selector_train_manifest_path is valid only in "
            "Stage24/25 selector mode"
        )

    stage23_generator_manifest = str(
        getattr(agent_config, "stage23_generator_train_manifest_path", "")
    )
    if str(getattr(agent_config, "grpo_training_mode", "")) in {
        "diffgrpo_selected_set",
        "stage27_public_diffgrpo_selected_set",
        "stage28_public_paired_uplift_multi",
        "stage28_public_paired_uplift_explore",
        "stage29_public_headroom_hybrid",
        "stage29_public_headroom_conditional",
        "stage30_public_mode_coverage",
        "stage30_public_mode_coverage_constrained",
        "stage31_public_deployed_pair",
        "stage31_public_deployed_frontier",
        "stage32_public_deployed_extended",
        "stage32_selector_aware_frontier",
        # Stage33 uses the same frozen fold token manifest to prevent
        # cross-fold leakage; the field is retained for backwards-compatible
        # Hydra wiring with the existing Stage23/31/32 data path.
        "stage33_cdc_grpo",
        "stage34_mode_aligned_frontier_grpo",
        "stage35_nested_counterfactual_deployment_grpo",
        "stage36_reference_gated_tail_ncd_grpo",
    }:
        if not stage23_generator_manifest:
            raise ValueError("Stage23 generator requires its folds0-3 manifest")
        generator_tokens = load_manifest_tokens(Path(stage23_generator_manifest))
        generator_token_set = set(generator_tokens)
        missing_tokens = sorted(generator_token_set - set(train_data.tokens))
        if missing_tokens:
            raise RuntimeError(
                f"Stage23 generator manifest has unavailable token: {missing_tokens[0]}"
            )
        train_data.tokens = [
            token for token in train_data.tokens if token in generator_token_set
        ]
        if len(train_data.tokens) != len(generator_tokens):
            raise RuntimeError("Stage23 generator manifest token count drifted")
    elif stage23_generator_manifest:
        raise ValueError(
            "stage23_generator_train_manifest_path is Stage23-generator-only"
        )

    diffgrpo_manifest_path = str(
        getattr(agent_config, "diffgrpo_train_manifest_path", "")
    )
    if diffgrpo_manifest_path:
        if str(getattr(agent_config, "grpo_training_mode", "")) != "diffgrpo_full_chain":
            raise ValueError(
                "diffgrpo_train_manifest_path is valid only in diffgrpo_full_chain mode"
            )
        diffgrpo_token_list = load_manifest_tokens(Path(diffgrpo_manifest_path))
        diffgrpo_tokens = set(diffgrpo_token_list)
        available_tokens = set(train_data.tokens)
        missing_tokens = sorted(diffgrpo_tokens - available_tokens)
        if missing_tokens:
            raise RuntimeError(
                f"Stage-16 manifest has {len(missing_tokens)} unavailable tokens; "
                f"first={missing_tokens[0]}"
            )
        train_data.tokens = [
            token for token in train_data.tokens if token in diffgrpo_tokens
        ]
        if len(train_data.tokens) != len(diffgrpo_token_list):
            raise RuntimeError("Stage-16 manifest filtering changed token count")
        logger.info(
            "Stage-16 DiffGRPO manifest: %d training tokens from %s",
            len(train_data.tokens), diffgrpo_manifest_path,
        )

    selected_anchor_manifest_path = str(
        getattr(agent_config, "diffgrpo_selected_mode_manifest_path", "")
    )
    selected_anchor_training = str(
        getattr(agent_config, "grpo_training_mode", "")
    ) in {
        "diffgrpo_selected_anchor",
        "diffgrpo_selected_anchor_base_preserve",
        "diffgrpo_paired_residual",
    }
    if selected_anchor_training:
        if not selected_anchor_manifest_path:
            raise ValueError(
                "selected-anchor training requires its selected-mode manifest"
            )
        selected_tokens = load_manifest_tokens(
            Path(selected_anchor_manifest_path)
        )
        selected_token_set = set(selected_tokens)
        available_tokens = set(train_data.tokens)
        missing_tokens = sorted(selected_token_set - available_tokens)
        if missing_tokens:
            raise RuntimeError(
                f"Selected-anchor manifest has {len(missing_tokens)} unavailable tokens; "
                f"first={missing_tokens[0]}"
            )
        train_data.tokens = [
            token for token in train_data.tokens if token in selected_token_set
        ]
        if len(train_data.tokens) != len(selected_tokens):
            raise RuntimeError("Selected-anchor manifest filtering changed token count")
        logger.info(
            "Selected-anchor manifest: %d training tokens from %s",
            len(train_data.tokens), selected_anchor_manifest_path,
        )
    elif selected_anchor_manifest_path:
        raise ValueError(
            "diffgrpo_selected_mode_manifest_path is valid only in selected-anchor modes"
        )

    paired_risk_manifest_path = str(
        getattr(agent_config, "paired_risk_train_manifest_path", "")
    )
    if paired_risk_manifest_path:
        if str(getattr(agent_config, "grpo_training_mode", "")) != "paired_tail_risk_selector":
            raise ValueError(
                "paired_risk_train_manifest_path is valid only in paired-risk mode"
            )
        paired_token_list = load_manifest_tokens(Path(paired_risk_manifest_path))
        paired_tokens = set(paired_token_list)
        missing_tokens = sorted(paired_tokens - set(train_data.tokens))
        if missing_tokens:
            raise RuntimeError(
                f"Stage-17 manifest has {len(missing_tokens)} unavailable tokens; "
                f"first={missing_tokens[0]}"
            )
        train_data.tokens = [
            token for token in train_data.tokens if token in paired_tokens
        ]
        if len(train_data.tokens) != len(paired_token_list):
            raise RuntimeError("Stage-17 manifest filtering changed token count")
        logger.info(
            "Stage-17 paired-risk manifest: %d training tokens from %s",
            len(train_data.tokens), paired_risk_manifest_path,
        )
    elif str(getattr(agent_config, "grpo_training_mode", "")) == "paired_tail_risk_selector":
        raise ValueError("paired-risk training requires paired_risk_train_manifest_path")

    logger.info("Building Datasets")
    training_mode = str(getattr(agent_config, "grpo_training_mode", ""))
    stage30_mode = training_mode in {
        "stage30_public_mode_coverage",
        "stage30_public_mode_coverage_constrained",
    }
    stage31_mode = training_mode in {
        "stage31_public_deployed_pair",
        "stage31_public_deployed_frontier",
    }
    stage32_mode = training_mode in {
        "stage32_public_deployed_extended",
        "stage32_selector_aware_frontier",
    }
    stage33_mode = training_mode == "stage33_cdc_grpo"
    stage34_mode = training_mode == "stage34_mode_aligned_frontier_grpo"
    stage35_mode = training_mode == "stage35_nested_counterfactual_deployment_grpo"
    stage36_mode = training_mode == "stage36_reference_gated_tail_ncd_grpo"
    priority_manifest_path = str(
        getattr(agent_config, "grpo_priority_manifest_path", "")
    )
    priority_fraction = float(
        getattr(agent_config, "grpo_priority_sample_fraction", 0.0)
    )
    if stage30_mode or stage31_mode or stage32_mode or stage33_mode or stage34_mode or stage35_mode or stage36_mode:
        stage_name = (
            "Stage36" if stage36_mode else (
                "Stage35" if stage35_mode else ("Stage34" if stage34_mode else ("Stage33" if stage33_mode else (
                "Stage32" if stage32_mode else ("Stage31" if stage31_mode else "Stage30")
            )))
            )
        )
        config_prefix = (
            "stage36" if stage36_mode else (
                "stage35" if stage35_mode else ("stage34" if stage34_mode else ("stage33" if stage33_mode else (
                "stage32" if stage32_mode else ("stage31" if stage31_mode else "stage30")
            )))
            )
        )
        if priority_manifest_path or priority_fraction != 0.0:
            raise ValueError(
                f"{stage_name} bucket sampler cannot mix priority sampling"
            )
        bucket_manifest_path = str(
            getattr(
                agent_config, f"{config_prefix}_bucket_manifest_path", ""
            )
        )
        token_buckets = load_stage30_bucket_manifest(
            bucket_manifest_path, require_full=True
        )
        accumulation = int(
            getattr(agent_config, f"{config_prefix}_gradient_accumulation", 8)
        )
        trainer_accumulation = int(
            cfg.trainer.params.get("accumulate_grad_batches", 1)
        )
        if trainer_accumulation != accumulation:
            raise ValueError(
                f"{stage_name} trainer accumulate_grad_batches must equal 8"
            )
        if bool(cfg.trainer.params.get("use_distributed_sampler", True)):
            raise ValueError(
                f"{stage_name} requires "
                "trainer.params.use_distributed_sampler=false"
            )
        train_loader_params = dict(cfg.dataloader.params)
        configured_batch = int(train_loader_params.pop("batch_size"))
        if configured_batch != 1:
            raise ValueError(
                f"{stage_name} DataLoader batch_size override must be 1"
            )
        configured_devices = int(cfg.trainer.params.get("devices", 1))
        if configured_devices != 8:
            raise ValueError(
                f"formal {stage_name} requires exactly eight DDP devices"
            )
        batch_sampler = Stage30GlobalBucketBatchSampler(
            train_data.tokens,
            token_buckets,
            optimizer_steps_per_epoch=int(getattr(
                agent_config,
                f"{config_prefix}_optimizer_steps_per_epoch",
                48,
            )),
            composition=tuple(getattr(
                agent_config,
                f"{config_prefix}_global_bucket_composition",
                (2, 30, 8, 24),
            )),
            accumulation_steps=accumulation,
            world_size=configured_devices,
            seed=int(cfg.seed),
        )
        train_dataloader = DataLoader(
            train_data,
            collate_fn=custom_collate_fn,
            batch_sampler=batch_sampler,
            **train_loader_params,
        )
        logger.info(
            "%s global bucket sampler: rank=%d/%d composition=%s "
            "optimizer_steps=%d microbatches=%d",
            stage_name,
            batch_sampler.rank,
            batch_sampler.world_size,
            batch_sampler.composition,
            batch_sampler.optimizer_steps_per_epoch,
            len(batch_sampler),
        )
    elif priority_manifest_path:
        if not 0.0 < priority_fraction < 1.0:
            raise ValueError(
                "grpo_priority_sample_fraction must be between 0 and 1 "
                "when a priority manifest is configured"
            )
        priority_tokens = load_manifest_tokens(Path(priority_manifest_path))
        train_loader_params = dict(cfg.dataloader.params)
        batch_size = int(train_loader_params.pop("batch_size"))
        batch_sampler = BalancedPriorityBatchSampler(
            train_data.tokens,
            priority_tokens,
            batch_size=batch_size,
            priority_fraction=priority_fraction,
            seed=int(cfg.seed),
        )
        train_dataloader = DataLoader(
            train_data,
            collate_fn=custom_collate_fn,
            batch_sampler=batch_sampler,
            **train_loader_params,
        )
        logger.info(
            "GRPO balanced sampler: priority=%d common=%d fraction=%.3f batches=%d",
            len(batch_sampler.priority_indices),
            len(batch_sampler.common_indices),
            priority_fraction,
            len(batch_sampler),
        )
    else:
        if priority_fraction != 0.0:
            raise ValueError(
                "grpo_priority_sample_fraction requires a priority manifest"
            )
        train_dataloader = DataLoader(
            train_data,
            collate_fn=custom_collate_fn,
            **cfg.dataloader.params,
            shuffle=True,
        )
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, collate_fn=custom_collate_fn, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    trainer = pl.Trainer(
                        **cfg.trainer.params,
                        callbacks=agent.get_training_callbacks()
                        )
    
    # 创建TensorBoard Logger - 添加详细日志
    tensorboard_dir = "/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/tensorboard_logs"
    logger.info(f"Creating TensorBoard directory: {tensorboard_dir}")
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    tensorboard_logger = TensorBoardLogger(
        save_dir=tensorboard_dir,
        name="diffusiondrive",
        version=f"experiment_seed_{cfg.seed}",
        default_hp_metric=False
    )
    
    logger.info(f"=== TensorBoard配置确认 ===")
    logger.info(f"Save dir: {tensorboard_logger.save_dir}")
    logger.info(f"Name: {tensorboard_logger.name}") 
    logger.info(f"Version: {tensorboard_logger.version}")
    logger.info(f"Final log_dir: {tensorboard_logger.log_dir}")
    logger.info(f"=== TensorBoard配置结束 ===")

    # 确保日志目录存在
    final_log_dir = tensorboard_logger.log_dir
    os.makedirs(final_log_dir, exist_ok=True)
    logger.info(f"确保日志目录存在: {final_log_dir} - {os.path.exists(final_log_dir)}")
    
    logger.info("Starting Training")
    resume_checkpoint = cfg.get("resume_checkpoint_path", None)
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=str(resume_checkpoint) if resume_checkpoint else None,
    )
    

if __name__ == "__main__":
    main()
