#!/usr/bin/env python3
"""Paired NAVSIM holdout evaluation for truncated-diffusion schedules."""

import argparse
import hashlib
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
COMPONENT_NAMES = (
    "collision",
    "drivable",
    "progress",
    "ttc",
    "comfort",
    "direction",
)
SAFETY_COMPONENTS = ("collision", "drivable", "ttc")
TRUST_METRICS = (
    "pre_distance",
    "post_distance",
    "alpha",
    "projected",
    "reference_coverage",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=BASE_CHECKPOINT)
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        default=BASE_CHECKPOINT,
        help="Frozen base checkpoint used by hard trust projection",
    )
    parser.add_argument(
        "--generation-trust-projection-mode",
        choices=("none", "reference_mean_ball"),
        default="none",
    )
    parser.add_argument("--generation-trust-calibration-path", type=Path)
    parser.add_argument(
        "--collect-trust-calibration",
        action="store_true",
        help="Record raw candidate/base displacements without applying projection",
    )
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--metric-cache-path", type=Path, default=DEFAULT_METRIC_CACHE)
    parser.add_argument("--limit", type=int, default=1024)
    parser.add_argument(
        "--log-split",
        choices=("train", "val", "test"),
        default="val",
        help="Select cached train, val, or navtest test logs",
    )
    parser.add_argument(
        "--tokens-file",
        type=Path,
        help="JSON token list or prior evaluation artifact whose record order is reused",
    )
    parser.add_argument(
        "--baseline-artifact",
        type=Path,
        help="Prior evaluation artifact used for paired deltas and bootstrap CI",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--truncation-timestep", type=int, required=True)
    parser.add_argument("--roll-timesteps", type=int, nargs="+", required=True)
    parser.add_argument("--scheduler-num-inference-steps", type=int, required=True)
    parser.add_argument(
        "--generation-policy-algorithm",
        choices=(
            "legacy_ppo",
            "diffgrpo_full_chain",
            "diffgrpo_selected_anchor",
        ),
        default="legacy_ppo",
        help="Record and validate the generation algorithm associated with the schedule",
    )
    parser.add_argument(
        "--evaluation-noise-namespace",
        type=int,
        default=-1,
        help="-1 uses the default token hash; non-negative values add a namespace",
    )
    parser.add_argument(
        "--selector-logits-source",
        choices=("current", "reference", "value_top2", "paired_tail_risk"),
        default="current",
        help="Choose current candidates with current, frozen-reference, or Stage-15 logits",
    )
    parser.add_argument("--value-selector-checkpoint", type=Path)
    parser.add_argument("--value-selector-calibration-margin", type=float, default=-1.0)
    parser.add_argument("--paired-risk-checkpoint", type=Path)
    parser.add_argument("--paired-risk-threshold", type=float, default=-1.0)
    parser.add_argument("--paired-risk-calibration-path", type=Path)
    parser.add_argument(
        "--paired-risk-expected-generator-sha256",
        default="",
        help="Fail closed unless the evaluated generator has this SHA256",
    )
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


def extract_trust_batch(predictions, batch_size, num_modes):
    trust_batch = {}
    expected_shape = (batch_size, num_modes, 2)
    for metric in TRUST_METRICS:
        key = f"generation_trust_{metric}"
        if key not in predictions:
            raise RuntimeError(f"Missing trust diagnostic: {key}")
        value = predictions[key]
        if tuple(value.shape) != expected_shape:
            raise RuntimeError(
                f"{key} has shape {tuple(value.shape)}, "
                f"expected {expected_shape}"
            )
        if metric in {"pre_distance", "post_distance", "alpha"}:
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Non-finite trust diagnostic: {key}")
        trust_batch[metric] = value.detach().cpu()
    return trust_batch




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


def load_ordered_tokens(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        tokens = payload
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        tokens = [record["token"] for record in payload["records"]]
    else:
        raise ValueError(
            "tokens file must be a JSON list or an evaluation artifact with records"
        )
    tokens = [str(token) for token in tokens]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"tokens file contains duplicate tokens: {path}")
    return tokens


def paired_bootstrap_ci(
    differences: np.ndarray, samples: int, seed: int
) -> list[float]:
    if samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        size = min(1000, samples - start)
        indices = rng.integers(
            0, differences.size, size=(size, differences.size)
        )
        means[start : start + size] = differences[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    trust_active = (
        args.collect_trust_calibration
        or args.generation_trust_projection_mode == "reference_mean_ball"
    )
    if trust_active and not args.reference_checkpoint.is_file():
        raise FileNotFoundError(args.reference_checkpoint)
    if args.collect_trust_calibration and args.generation_trust_projection_mode != "none":
        raise ValueError("raw calibration collection requires projection mode=none")
    if (
        args.generation_trust_projection_mode == "reference_mean_ball"
        and args.generation_trust_calibration_path is None
    ):
        raise ValueError("reference_mean_ball requires a calibration artifact")
    if args.collect_trust_calibration and args.baseline_artifact is not None:
        raise ValueError("calibration collection cannot read a reward baseline")
    if args.selector_logits_source == "value_top2":
        if args.value_selector_checkpoint is None:
            raise ValueError("value_top2 requires --value-selector-checkpoint")
        if not args.value_selector_checkpoint.is_file():
            raise FileNotFoundError(args.value_selector_checkpoint)
        if args.value_selector_calibration_margin < 0:
            raise ValueError("value_top2 requires a non-negative calibration margin")
    elif args.value_selector_checkpoint is not None:
        raise ValueError(
            "--value-selector-checkpoint is valid only with value_top2"
        )
    if args.selector_logits_source == "paired_tail_risk":
        if args.paired_risk_checkpoint is None or not args.paired_risk_checkpoint.is_file():
            raise ValueError("paired_tail_risk requires --paired-risk-checkpoint")
        if not 0.0 <= args.paired_risk_threshold <= 1.0:
            raise ValueError("paired_tail_risk requires a threshold in [0,1]")
    elif args.paired_risk_checkpoint is not None:
        raise ValueError(
            "--paired-risk-checkpoint is valid only with paired_tail_risk"
        )

    config = replace(
        TransfuserConfig(),
        metric_cache_path=(
            "" if args.collect_trust_calibration else str(args.metric_cache_path)
        ),
        diffusion_truncation_timestep=args.truncation_timestep,
        diffusion_roll_timesteps=tuple(args.roll_timesteps),
        diffusion_scheduler_num_inference_steps=args.scheduler_num_inference_steps,
        generation_policy_algorithm=args.generation_policy_algorithm,
        evaluation_noise_namespace=args.evaluation_noise_namespace,
        inference_selector_source=args.selector_logits_source,
        value_selector_checkpoint_path=str(args.value_selector_checkpoint or ""),
        value_selector_calibration_margin=args.value_selector_calibration_margin,
        paired_risk_checkpoint_path=str(args.paired_risk_checkpoint or ""),
        paired_risk_threshold=args.paired_risk_threshold,
        paired_risk_calibration_path=str(args.paired_risk_calibration_path or ""),
        paired_risk_expected_generator_sha256=(
            args.paired_risk_expected_generator_sha256
        ),
        generation_trust_projection_mode=args.generation_trust_projection_mode,
        generation_trust_calibration_path=str(
            args.generation_trust_calibration_path or ""
        ),
        generation_trust_collect_calibration=args.collect_trust_calibration,
        generation_trust_calibration_seed=20260719,
    )
    agent = TransfuserAgent(
        config=config,
        lr=0.0,
        checkpoint_path=str(args.checkpoint),
        reference_checkpoint_path=str(args.reference_checkpoint),
    )
    agent.eval().to(args.device)

    split_path = (
        Path(__file__).resolve().parents[2]
        / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
    )
    split_config = OmegaConf.load(split_path)
    selected_logs = list(getattr(split_config, f"{args.log_split}_logs"))
    dataset = CacheOnlyDataset(
        cache_path=str(args.cache_path),
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        log_names=selected_logs,
    )
    if args.collect_trust_calibration:
        available = set(dataset.tokens)
    else:
        rewardable = set(MetricCacheLoader(args.metric_cache_path).tokens)
        available = set(dataset.tokens).intersection(rewardable)
    if args.tokens_file is not None:
        requested_tokens = load_ordered_tokens(args.tokens_file)[: args.limit]
        missing_tokens = [token for token in requested_tokens if token not in available]
        if missing_tokens:
            raise RuntimeError(
                f"{len(missing_tokens)} fixed tokens are unavailable; "
                f"first missing token={missing_tokens[0]}"
            )
        dataset.tokens = requested_tokens
    else:
        dataset.tokens = sorted(available)[: args.limit]
    if len(dataset.tokens) != args.limit:
        raise RuntimeError(
            f"Requested {args.limit} tokens, found only {len(dataset.tokens)} "
            f"rewardable {args.log_split} tokens"
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
    component_values = {name: [] for name in COMPONENT_NAMES}
    current_reference_agreement_values = []
    value_switch_values = []
    value_switch_gain_values = []

    with torch.inference_mode():
        for batch_idx, (features, targets, tokens) in enumerate(loader):
            predictions = agent.forward(
                to_device(features, args.device),
                to_device(targets, args.device),
                tokens,
            )
            if args.collect_trust_calibration:
                logits = predictions["final_poses_cls"]
                trust_batch = extract_trust_batch(
                    predictions,
                    len(tokens),
                    logits.shape[1],
                )
                for row, token in enumerate(tokens):
                    records.append(
                        {
                            "token": token,
                            "generation_trust": {
                                metric: value[row].tolist()
                                for metric, value in trust_batch.items()
                            },
                        }
                    )
                if (batch_idx + 1) % 32 == 0:
                    print(
                        f"collected={len(records)}/{len(dataset)}",
                        flush=True,
                    )
                continue
            rewards = predictions["rewards"].float()
            valid = predictions["reward_valid_mask"].bool()
            current_logits = predictions["final_poses_cls"].float()
            reference_logits = predictions["final_ref_poses_cls"].float()
            logits = predictions["inference_selector_logits"].float()
            trajectories = predictions["final_poses_reg"].float()
            components = predictions["reward_component_scores"].float()
            value_active = args.selector_logits_source == "value_top2"
            paired_active = args.selector_logits_source == "paired_tail_risk"
            value_batch = {}
            if value_active:
                for name in (
                    "fallback_mode",
                    "challenger_mode",
                    "switch",
                    "predicted_advantage",
                    "fallback_score",
                    "challenger_score",
                    "fallback_safety_lower",
                    "challenger_safety_lower",
                    "score_mean",
                    "score_std",
                    "component_mean",
                    "component_std",
                    "component_predictions",
                    "score_predictions",
                ):
                    key = f"value_selector_{name}"
                    if key not in predictions:
                        raise RuntimeError(f"Missing Stage-15 diagnostic: {key}")
                    value_batch[name] = predictions[key]
            paired_batch = {}
            if paired_active:
                for name in (
                    "fallback", "selected_score", "event_probabilities",
                    "fallback_score", "delta_prediction", "current_trajectory",
                    "base_trajectory",
                ):
                    key = f"paired_risk_{name}"
                    if key not in predictions:
                        raise RuntimeError(f"Missing Stage-17 diagnostic: {key}")
                    paired_batch[name] = predictions[key]
                paired_batch["base_rewards"] = predictions["paired_base_rewards"]
                paired_batch["base_valid"] = predictions["paired_base_valid"]
                paired_batch["base_components"] = predictions["paired_base_components"]
            trust_batch = {}
            if trust_active:
                trust_batch = extract_trust_batch(
                    predictions, rewards.shape[0], rewards.shape[1]
                )
            masked_logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
            selector_probs = torch.softmax(masked_logits, dim=-1) * valid.float()

            selected_idx = predictions["mode_idx"]
            if not torch.equal(selected_idx, logits.argmax(dim=-1)):
                raise RuntimeError("model trajectory mode and evaluator selector disagree")
            current_idx = current_logits.argmax(dim=-1)
            reference_idx = reference_logits.argmax(dim=-1)
            selected = rewards.gather(1, selected_idx[:, None]).squeeze(1)
            selected_components = components.gather(
                1,
                selected_idx[:, None, None].expand(-1, 1, len(COMPONENT_NAMES)),
            ).squeeze(1)
            if paired_active:
                base_selected = paired_batch["base_rewards"].gather(
                    1, selected_idx[:, None]
                ).squeeze(1)
                base_selected_components = paired_batch["base_components"].gather(
                    1,
                    selected_idx[:, None, None].expand(-1, 1, len(COMPONENT_NAMES)),
                ).squeeze(1)
                fallback = paired_batch["fallback"].bool()
                selected = torch.where(fallback, base_selected, selected)
                selected_components = torch.where(
                    fallback[:, None], base_selected_components, selected_components
                )
            oracle_idx = rewards.masked_fill(~valid, float("-inf")).argmax(dim=-1)
            oracle = rewards.gather(1, oracle_idx[:, None]).squeeze(1)
            entropy = torch.distributions.Categorical(logits=logits).entropy()
            diversity = pairwise_diversity(trajectories)

            for row, token in enumerate(tokens):
                mask = valid[row]
                reward_np = rewards[row, mask].cpu().numpy()
                diagnostic_logits = (
                    value_batch["score_mean"] if value_active else logits
                )
                logit_np = diagnostic_logits[row, mask].cpu().numpy()
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
                selected_mode = int(selected_idx[row].item())
                current_mode = int(current_idx[row].item())
                reference_mode = int(reference_idx[row].item())
                oracle_mode = int(oracle_idx[row].item())
                valid_logits = logits[row, mask]
                if valid_logits.numel() >= 2:
                    top_logits = torch.topk(valid_logits, k=2).values
                    selector_margin = float((top_logits[0] - top_logits[1]).item())
                else:
                    selector_margin = 0.0

                selected_values.append(selected_value)
                oracle_values.append(oracle_value)
                candidate_values.append(candidate_value)
                hit_values.append(hit_value)
                entropy_values.append(entropy_value)
                diversity_values.append(diversity_value)
                rank_values.append(correlation)
                current_reference_agreement_values.append(
                    float(current_mode == reference_mode)
                )
                component_record = {
                    name: float(selected_components[row, index].item())
                    for index, name in enumerate(COMPONENT_NAMES)
                }
                for name, value in component_record.items():
                    component_values[name].append(value)
                value_record = {}
                if value_active:
                    fallback_mode = int(value_batch["fallback_mode"][row].item())
                    challenger_mode = int(value_batch["challenger_mode"][row].item())
                    switched = bool(value_batch["switch"][row].item())
                    switch_gain = float(
                        rewards[row, challenger_mode].item()
                        - rewards[row, fallback_mode].item()
                    )
                    value_switch_values.append(float(switched))
                    value_switch_gain_values.append(switch_gain if switched else 0.0)
                    value_record = {
                        "value_selector": {
                            "fallback_mode": fallback_mode,
                            "challenger_mode": challenger_mode,
                            "switched": switched,
                            "true_challenger_minus_fallback": switch_gain,
                            "fallback_components": components[
                                row, fallback_mode
                            ].detach().cpu().tolist(),
                            "challenger_components": components[
                                row, challenger_mode
                            ].detach().cpu().tolist(),
                            "predicted_advantage": float(
                                value_batch["predicted_advantage"][row].item()
                            ),
                            "fallback_score": float(
                                value_batch["fallback_score"][row].item()
                            ),
                            "challenger_score": float(
                                value_batch["challenger_score"][row].item()
                            ),
                            "fallback_safety_lower": value_batch[
                                "fallback_safety_lower"
                            ][row].detach().cpu().tolist(),
                            "challenger_safety_lower": value_batch[
                                "challenger_safety_lower"
                            ][row].detach().cpu().tolist(),
                            "score_mean": value_batch["score_mean"][row]
                            .detach().cpu().tolist(),
                            "score_std": value_batch["score_std"][row]
                            .detach().cpu().tolist(),
                            "component_mean": value_batch["component_mean"][row]
                            .detach().cpu().tolist(),
                            "component_std": value_batch["component_std"][row]
                            .detach().cpu().tolist(),
                            "component_predictions": value_batch[
                                "component_predictions"
                            ][row].detach().cpu().tolist(),
                            "score_predictions": value_batch[
                                "score_predictions"
                            ][row].detach().cpu().tolist(),
                            "reference_logits": reference_logits[row]
                            .detach().cpu().tolist(),
                            "candidate_components": components[row]
                            .detach().cpu().tolist(),
                        }
                    }
                paired_record = {}
                if paired_active:
                    fallback_used = bool(paired_batch["fallback"][row].item())
                    base_reward = float(base_selected[row].item())
                    current_reward = float(rewards[row, selected_mode].item())
                    paired_record = {
                        "paired_risk": {
                            "fallback": fallback_used,
                            "threshold": float(args.paired_risk_threshold),
                            "fallback_score": float(
                                paired_batch["selected_score"][row].item()
                            ),
                            "event_probabilities": paired_batch[
                                "event_probabilities"
                            ][row, selected_mode].detach().cpu().tolist(),
                            "event_probabilities_all": paired_batch[
                                "event_probabilities"
                            ][row].detach().cpu().tolist(),
                            "delta_prediction": float(
                                paired_batch["delta_prediction"][row, selected_mode].item()
                            ),
                            "current_reward": current_reward,
                            "base_reward": base_reward,
                            "true_current_minus_base": current_reward - base_reward,
                            "current_components": components[
                                row, selected_mode
                            ].detach().cpu().tolist(),
                            "base_components": paired_batch["base_components"][
                                row, selected_mode
                            ].detach().cpu().tolist(),
                            "current_candidate_components": components[row]
                            .detach().cpu().tolist(),
                            "base_candidate_components": paired_batch[
                                "base_components"
                            ][row].detach().cpu().tolist(),
                            "base_candidate_rewards": [
                                float(paired_batch["base_rewards"][row, mode].item())
                                if bool(paired_batch["base_valid"][row, mode])
                                else None
                                for mode in range(rewards.shape[1])
                            ],
                            "current_trajectory": paired_batch["current_trajectory"][
                                row
                            ].detach().cpu().tolist(),
                            "base_trajectory": paired_batch["base_trajectory"][
                                row
                            ].detach().cpu().tolist(),
                        }
                    }
                records.append(
                    {
                        "token": token,
                        "selector_logits_source": args.selector_logits_source,
                        "selected_reward": selected_value,
                        "oracle_reward": oracle_value,
                        "regret": oracle_value - selected_value,
                        "candidate_reward": candidate_value,
                        "oracle_hit": hit_value,
                        "selected_mode": selected_mode,
                        "oracle_mode": oracle_mode,
                        "selector_margin": selector_margin,
                        "current_mode": current_mode,
                        "reference_mode": reference_mode,
                        "selected_trajectory": predictions["trajectory"][row]
                        .detach().cpu().tolist(),
                        "selected_probability": float(
                            selector_probs[row, selected_mode].item()
                        ),
                        "oracle_probability": float(
                            selector_probs[row, oracle_mode].item()
                        ),
                        "candidate_rewards": [
                            float(rewards[row, mode].item())
                            if bool(valid[row, mode]) else None
                            for mode in range(rewards.shape[1])
                        ],
                        "selector_probabilities": [
                            float(selector_probs[row, mode].item())
                            for mode in range(rewards.shape[1])
                        ],
                        "entropy": entropy_value,
                        "diversity": diversity_value,
                        "reward_logit_spearman": correlation,
                        "selected_components": component_record,
                        **value_record,
                        **paired_record,
                        **(
                            {
                                "generation_trust": {
                                    metric: value[row].tolist()
                                    for metric, value in trust_batch.items()
                                }
                            }
                            if trust_active else {}
                        ),
                    }
                )

            if (batch_idx + 1) % 32 == 0:
                print(f"evaluated={len(records)}/{len(dataset)}", flush=True)

    if args.collect_trust_calibration:
        selected_values = oracle_values = candidate_values = [0.0]
        hit_values = entropy_values = diversity_values = rank_values = [0.0]
        current_reference_agreement_values = [0.0]
        component_values = {
            name: [0.0] for name in COMPONENT_NAMES
        }
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": agent._checkpoint_sha256,
        "reference_checkpoint": str(args.reference_checkpoint),
        "reference_checkpoint_sha256": agent._reference_checkpoint_sha256,
        "generation_policy_algorithm": args.generation_policy_algorithm,
        "evaluation_noise_namespace": args.evaluation_noise_namespace,
        "cache_path": str(args.cache_path),
        "metric_cache_path": str(args.metric_cache_path),
        "tokens_file": str(args.tokens_file),
        "requested_limit": int(args.limit),
        "num_tokens": len(records),
        "num_failures": 0,
        "completed": len(records) == min(int(args.limit), len(dataset)),
        "log_split": args.log_split,
        "selector_logits_source": args.selector_logits_source,
        "schedule": agent._transfuser_model._trajectory_head.get_roll_schedule(),
        "selected_reward": float(np.mean(selected_values)),
        "oracle_reward": float(np.mean(oracle_values)),
        "selection_regret": float(np.mean(oracle_values) - np.mean(selected_values)),
        "candidate_reward": float(np.mean(candidate_values)),
        "oracle_hit_rate": float(np.mean(hit_values)),
        "classification_entropy": float(np.mean(entropy_values)),
        "trajectory_diversity": float(np.mean(diversity_values)),
        "reward_logit_spearman": float(np.mean(rank_values)),
        "current_reference_mode_agreement": float(
            np.mean(current_reference_agreement_values)
        ),
        "selected_component_means": {
            name: float(np.mean(values))
            for name, values in component_values.items()
        },
        "token_set_sha256": hashlib.sha256(
            "\n".join(record["token"] for record in records).encode("utf-8")
        ).hexdigest(),
    }
    if args.selector_logits_source == "value_top2":
        switch_array = np.asarray(value_switch_values, dtype=bool)
        gain_array = np.asarray(value_switch_gain_values, dtype=np.float64)
        switched_gain = gain_array[switch_array]
        summary["value_selector"] = {
            "checkpoint": str(args.value_selector_checkpoint),
            "checkpoint_sha256": agent._value_selector_checkpoint_sha256,
            "calibration_margin": args.value_selector_calibration_margin,
            "switch_count": int(switch_array.sum()),
            "switch_rate": float(switch_array.mean()),
            "switched_gain_mean": (
                float(switched_gain.mean()) if switched_gain.size else 0.0
            ),
            "beneficial_switches": int((switched_gain > 0).sum()),
            "neutral_switches": int((switched_gain == 0).sum()),
            "harmful_switches": int((switched_gain < 0).sum()),
            "beneficial_gain_sum": float(switched_gain[switched_gain > 0].sum()),
            "harmful_loss_sum": float(-switched_gain[switched_gain < 0].sum()),
            "worst_switch_gain": (
                float(switched_gain.min()) if switched_gain.size else 0.0
            ),
        }
    if args.selector_logits_source == "paired_tail_risk":
        paired_records = [record["paired_risk"] for record in records]
        fallback_array = np.asarray(
            [record["fallback"] for record in paired_records], dtype=bool
        )
        delta_array = np.asarray(
            [record["true_current_minus_base"] for record in paired_records],
            dtype=np.float64,
        )
        tail_count = max(1, int(np.ceil(0.01 * delta_array.size)))
        summary["paired_risk"] = {
            "checkpoint": str(args.paired_risk_checkpoint),
            "checkpoint_sha256": agent._paired_risk_checkpoint_sha256,
            "calibration_path": str(args.paired_risk_calibration_path),
            "threshold": float(args.paired_risk_threshold),
            "fallback_count": int(fallback_array.sum()),
            "fallback_rate": float(fallback_array.mean()),
            "current_minus_base_mean": float(delta_array.mean()),
            "current_minus_base_minimum": float(delta_array.min()),
            "current_minus_base_bottom_1pct_cvar": float(
                np.sort(delta_array)[:tail_count].mean()
            ),
            "catastrophic_current_count": int((delta_array <= -0.5).sum()),
            "admitted_catastrophic_count": int(
                ((~fallback_array) & (delta_array <= -0.5)).sum()
            ),
        }
    if args.collect_trust_calibration:
        for key in (
            "selected_reward",
            "oracle_reward",
            "selection_regret",
            "candidate_reward",
            "oracle_hit_rate",
            "classification_entropy",
            "trajectory_diversity",
            "reward_logit_spearman",
            "current_reference_mode_agreement",
            "selected_component_means",
        ):
            summary.pop(key)
    if trust_active:
        trust_arrays = {
            metric: np.asarray(
                [
                    record["generation_trust"][metric]
                    for record in records
                ]
            )
            for metric in (
                "pre_distance",
                "post_distance",
                "alpha",
                "projected",
                "reference_coverage",
            )
        }
        trust_steps = {}
        radii = agent._transfuser_model._trajectory_head._generation_trust_radii
        sigmas = (
            agent._transfuser_model._trajectory_head.get_generation_trust_sigmas()
        )
        for step_index, step_name in enumerate(("transition", "final")):
            pre = trust_arrays["pre_distance"][..., step_index].reshape(-1)
            post = trust_arrays["post_distance"][..., step_index].reshape(-1)
            alpha = trust_arrays["alpha"][..., step_index].reshape(-1)
            projected = trust_arrays["projected"][..., step_index].reshape(-1)
            coverage = trust_arrays["reference_coverage"][..., step_index].reshape(-1)
            trust_steps[step_name] = {
                "sigma": float(sigmas[step_name]),
                "radius": (
                    float(radii[step_name])
                    if step_name in radii else None
                ),
                "pre_distance_mean": float(pre.mean()),
                "pre_distance_p90": float(np.quantile(pre, 0.90)),
                "pre_distance_p99": float(np.quantile(pre, 0.99)),
                "pre_distance_max": float(pre.max()),
                "post_distance_mean": float(post.mean()),
                "post_distance_p90": float(np.quantile(post, 0.90)),
                "post_distance_p99": float(np.quantile(post, 0.99)),
                "post_distance_max": float(post.max()),
                "projection_fraction": float(projected.mean()),
                "alpha_mean": float(alpha.mean()),
                "alpha_min": float(alpha.min()),
                "reference_coverage": float(coverage.mean()),
                "count": int(pre.size),
            }
        summary["generation_trust"] = {
            "mode": args.generation_trust_projection_mode,
            "collect_calibration": args.collect_trust_calibration,
            "calibration_seed": (
                20260719 if args.collect_trust_calibration else None
            ),
            "reference_checkpoint": str(args.reference_checkpoint),
            "steps": trust_steps,
        }

    if args.baseline_artifact is not None:
        baseline_payload = json.loads(
            args.baseline_artifact.read_text(encoding="utf-8")
        )
        baseline_records = {
            str(record["token"]): record
            for record in baseline_payload["records"]
        }
        missing_baseline = [
            record["token"] for record in records
            if record["token"] not in baseline_records
        ]
        if missing_baseline:
            raise RuntimeError(
                f"baseline artifact is missing {len(missing_baseline)} tokens"
            )
        selected_differences = np.asarray(
            [
                record["selected_reward"]
                - float(baseline_records[record["token"]]["selected_reward"])
                for record in records
            ],
            dtype=np.float64,
        )
        oracle_differences = np.asarray(
            [
                record["oracle_reward"]
                - float(baseline_records[record["token"]]["oracle_reward"])
                for record in records
            ],
            dtype=np.float64,
        )
        summary["paired_baseline"] = {
            "artifact": str(args.baseline_artifact),
            "selected_difference": float(selected_differences.mean()),
            "selected_bootstrap_ci95": paired_bootstrap_ci(
                selected_differences,
                args.bootstrap_samples,
                args.bootstrap_seed,
            ),
            "oracle_difference": float(oracle_differences.mean()),
            "wins": int((selected_differences > 0).sum()),
            "ties": int((selected_differences == 0).sum()),
            "losses": int((selected_differences < 0).sum()),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        }
        baseline_has_selector_diagnostics = all(
            "selected_mode" in baseline_records[record["token"]]
            for record in records
        )
        if baseline_has_selector_diagnostics:
            switches = np.asarray(
                [
                    record["selected_mode"]
                    != int(baseline_records[record["token"]]["selected_mode"])
                    for record in records
                ],
                dtype=bool,
            )
            switched_differences = selected_differences[switches]
            margin_differences = np.asarray(
                [
                    record["selector_margin"]
                    - float(baseline_records[record["token"]]["selector_margin"])
                    for record in records
                ],
                dtype=np.float64,
            )
            oracle_probability_differences = np.asarray(
                [
                    record["oracle_probability"]
                    - float(
                        baseline_records[record["token"]]["oracle_probability"]
                    )
                    for record in records
                ],
                dtype=np.float64,
            )
            summary["paired_baseline"]["selector_diagnostics"] = {
                "mode_switches": int(switches.sum()),
                "mode_switch_rate": float(switches.mean()),
                "switched_selected_difference": (
                    float(switched_differences.mean())
                    if switched_differences.size else 0.0
                ),
                "beneficial_switches": int((switched_differences > 0).sum()),
                "neutral_switches": int((switched_differences == 0).sum()),
                "harmful_switches": int((switched_differences < 0).sum()),
                "selector_margin_difference": float(margin_differences.mean()),
                "oracle_probability_difference": float(
                    oracle_probability_differences.mean()
                ),
            }
        baseline_has_components = all(
            "selected_components" in baseline_records[record["token"]]
            for record in records
        )
        if baseline_has_components:
            component_differences = {}
            for name in COMPONENT_NAMES:
                differences = np.asarray(
                    [
                        record["selected_components"][name]
                        - float(
                            baseline_records[record["token"]][
                                "selected_components"
                            ][name]
                        )
                        for record in records
                    ],
                    dtype=np.float64,
                )
                component_differences[name] = float(differences.mean())

            baseline_safety_pass = np.asarray(
                [
                    all(
                        float(
                            baseline_records[record["token"]][
                                "selected_components"
                            ][name]
                        )
                        >= 1.0 - 1e-6
                        for name in SAFETY_COMPONENTS
                    )
                    for record in records
                ],
                dtype=bool,
            )
            safety_differences = selected_differences[baseline_safety_pass]
            summary["paired_baseline"]["selected_component_differences"] = (
                component_differences
            )
            summary["paired_baseline"]["baseline_safety_pass_tokens"] = int(
                baseline_safety_pass.sum()
            )
            summary["paired_baseline"]["safety_pass_selected_difference"] = (
                float(safety_differences.mean())
                if safety_differences.size
                else None
            )
    payload = {"summary": summary, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
