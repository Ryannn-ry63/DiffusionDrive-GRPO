from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import logging
import uuid
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl

from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.planning.training.dataset import Dataset
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.agents.abstract_agent import AbstractAgent

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def load_requested_manifest(path: Path) -> Tuple[List[str], Optional[List[str]]]:
    """Load tokens and optional log names from a JSON manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        tokens = payload
        log_names = None
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        tokens = [record["token"] for record in payload["records"]]
        if all("log_name" in record for record in payload["records"]):
            log_names = [str(record["log_name"]) for record in payload["records"]]
        else:
            log_names = None
    else:
        raise ValueError(f"Unsupported token manifest schema: {path}")
    tokens = [str(token) for token in tokens]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Token manifest contains duplicates: {path}")
    return tokens, log_names


def cache_features(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Optional[Any]]:
    """
    Helper function to cache features and targets of learnable agent.
    :param args: arguments for caching
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    agent: AbstractAgent = instantiate(cfg.agent)

    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    logger.info(f"Extracted {len(scene_loader.tokens)} scenarios for thread_id={thread_id}, node_id={node_id}.")

    dataset = Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )
    return []


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for dataset caching script.
    :param cfg: omegaconf dictionary
    """
    logger.info("Global Seed set to 0")
    pl.seed_everything(0, workers=True)

    logger.info("Building Worker")
    worker: WorkerPool = instantiate(cfg.worker)

    logger.info("Building SceneLoader")
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    tokens_file = cfg.get("tokens_file", None)
    requested_tokens = None
    if tokens_file is not None:
        tokens_path = Path(tokens_file)
        if not tokens_path.is_file():
            raise FileNotFoundError(tokens_path)
        requested_tokens, requested_log_names = load_requested_manifest(tokens_path)
        tokens_limit = cfg.get("tokens_limit", None)
        if tokens_limit is not None:
            tokens_limit = int(tokens_limit)
            if tokens_limit <= 0:
                raise ValueError("tokens_limit must be positive")
            requested_tokens = requested_tokens[:tokens_limit]
            if requested_log_names is not None:
                requested_log_names = requested_log_names[:tokens_limit]
        scene_filter.tokens = requested_tokens
        if requested_log_names is not None:
            scene_filter.log_names = sorted(set(requested_log_names))
        logger.info("Restricting dataset caching to %d manifest tokens", len(requested_tokens))
    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    if requested_tokens is not None:
        missing_tokens = set(requested_tokens) - set(scene_loader.tokens)
        if missing_tokens:
            first_missing = sorted(missing_tokens)[0]
            raise RuntimeError(
                f"SceneLoader could not resolve {len(missing_tokens)} manifest tokens; "
                f"first={first_missing}"
            )
    logger.info(f"Extracted {len(scene_loader)} scenarios for training/validation dataset")

    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]

    _ = worker_map(worker, cache_features, data_points)
    logger.info(f"Finished caching {len(scene_loader)} scenarios for training/validation dataset")


if __name__ == "__main__":
    main()
