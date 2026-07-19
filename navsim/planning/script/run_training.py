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
    load_manifest_tokens,
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

    logger.info("Building Datasets")
    priority_manifest_path = str(
        getattr(agent_config, "grpo_priority_manifest_path", "")
    )
    priority_fraction = float(
        getattr(agent_config, "grpo_priority_sample_fraction", 0.0)
    )
    if priority_manifest_path:
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
