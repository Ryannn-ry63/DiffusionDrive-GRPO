#!/usr/bin/env python3
"""Paired NAVSIM holdout evaluation for truncated-diffusion schedules."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.common.dataloader import MetricCacheLoader
from navsim.planning.training.dataset import CacheOnlyDataset


BASE_CHECKPOINT = Path(
    "/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/"
    "training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/"
    "version_0/checkpoints/eval_model"
)
DEFAULT_CACHE = Path(
    "/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache"
)
DEFAULT_METRIC_CACHE = Path(
    "/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/metric_cache_trainval"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=BASE_CHECKPOINT)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--metric-cache-path", type=Path, default=DEFAULT_METRIC_CACHE)
    parser.add_argument("--limit", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--truncation-timestep", type=int, required=True)
    parser.add_argument("--roll-timesteps", type=int, nargs="+", required=True)
    parser.add_argument("--scheduler-num-inference-steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def collate(batch):
    features_list, targets_list, tokens = zip(*batch)
    features = {
        key: torch.stack([sample[key] for sample in features_list], dim=0)
        for key in features_list[0]
    }
    targets = {
        key: torch.stack([sample[key] for sample in targets_list], dim=0)
        for key in targets_list[0]
    }
    return features, targets, tokens


def to_device(values, device):
    return {key: value.to(device, non_blocking=True) for key, value in values.items()}


def pairwise_diversity(trajectories: torch.Tensor) -> torch.Tensor:
    """Mean pairwise ADE across candidate modes, returned once per scene."""
    xy = trajectories[..., :2]
    distances = torch.linalg.vector_norm(
        xy[:, :, None] - xy[:, None, :], dim=-1
    ).mean(dim=-1)
    modes = trajectories.shape[1]
    upper = torch.triu(
        torch.ones((modes, modes), dtype=torch.bool, device=trajectories.device),
        diagonal=1,
    )
    return distances[:, upper].mean(dim=-1)


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    config = replace(
        TransfuserConfig(),
        metric_cache_path=str(args.metric_cache_path),
        diffusion_truncation_timestep=args.truncation_timestep,
        diffusion_roll_timesteps=tuple(args.roll_timesteps),
        diffusion_scheduler_num_inference_steps=args.scheduler_num_inference_steps,
    )
    agent = TransfuserAgent(
        config=config,
        lr=0.0,
        checkpoint_path=str(args.checkpoint),
        reference_checkpoint_path=str(args.checkpoint),
    )
    # No reference forward is needed for schedule diagnostics.
    agent._transfuser_model._trajectory_head.ref_policy = None
    agent.eval().to(args.device)

    split_path = (
        Path(__file__).resolve().parents[2]
        / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
    )
    val_logs = list(OmegaConf.load(split_path).val_logs)
    dataset = CacheOnlyDataset(
        cache_path=str(args.cache_path),
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        log_names=val_logs,
    )
    rewardable = set(MetricCacheLoader(args.metric_cache_path).tokens)
    dataset.tokens = sorted(set(dataset.tokens).intersection(rewardable))[: args.limit]
    if len(dataset.tokens) != args.limit:
        raise RuntimeError(
            f"Requested {args.limit} tokens, found only {len(dataset.tokens)} rewardable validation tokens"
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    records = []
    selected_values = []
    oracle_values = []
    candidate_values = []
    hit_values = []
    entropy_values = []
    diversity_values = []
    rank_values = []

    with torch.inference_mode():
        for batch_idx, (features, targets, tokens) in enumerate(loader):
            predictions = agent.forward(
                to_device(features, args.device),
                to_device(targets, args.device),
                tokens,
            )
            rewards = predictions["rewards"].float()
            valid = predictions["reward_valid_mask"].bool()
            logits = predictions["final_poses_cls"].float()
            trajectories = predictions["final_poses_reg"].float()

            selected_idx = logits.argmax(dim=-1)
            selected = rewards.gather(1, selected_idx[:, None]).squeeze(1)
            oracle_idx = rewards.masked_fill(~valid, float("-inf")).argmax(dim=-1)
            oracle = rewards.gather(1, oracle_idx[:, None]).squeeze(1)
            entropy = torch.distributions.Categorical(logits=logits).entropy()
            diversity = pairwise_diversity(trajectories)

            for row, token in enumerate(tokens):
                mask = valid[row]
                reward_np = rewards[row, mask].cpu().numpy()
                logit_np = logits[row, mask].cpu().numpy()
                if np.ptp(reward_np) == 0 or np.ptp(logit_np) == 0:
                    correlation = 0.0
                else:
                    correlation = float(spearmanr(reward_np, logit_np).statistic)
                    if not np.isfinite(correlation):
                        correlation = 0.0
                selected_value = float(selected[row].item())
                oracle_value = float(oracle[row].item())
                candidate_value = float(rewards[row, mask].mean().item())
                hit_value = float(selected_idx[row] == oracle_idx[row])
                entropy_value = float(entropy[row].item())
                diversity_value = float(diversity[row].item())

                selected_values.append(selected_value)
                oracle_values.append(oracle_value)
                candidate_values.append(candidate_value)
                hit_values.append(hit_value)
                entropy_values.append(entropy_value)
                diversity_values.append(diversity_value)
                rank_values.append(correlation)
                records.append(
                    {
                        "token": token,
                        "selected_reward": selected_value,
                        "oracle_reward": oracle_value,
                        "regret": oracle_value - selected_value,
                        "candidate_reward": candidate_value,
                        "oracle_hit": hit_value,
                        "entropy": entropy_value,
                        "diversity": diversity_value,
                        "reward_logit_spearman": correlation,
                    }
                )

            if (batch_idx + 1) % 32 == 0:
                print(f"evaluated={len(records)}/{len(dataset)}", flush=True)

    summary = {
        "checkpoint": str(args.checkpoint),
        "num_tokens": len(records),
        "schedule": agent._transfuser_model._trajectory_head.get_roll_schedule(),
        "selected_reward": float(np.mean(selected_values)),
        "oracle_reward": float(np.mean(oracle_values)),
        "selection_regret": float(np.mean(oracle_values) - np.mean(selected_values)),
        "candidate_reward": float(np.mean(candidate_values)),
        "oracle_hit_rate": float(np.mean(hit_values)),
        "classification_entropy": float(np.mean(entropy_values)),
        "trajectory_diversity": float(np.mean(diversity_values)),
        "reward_logit_spearman": float(np.mean(rank_values)),
    }
    payload = {"summary": summary, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
