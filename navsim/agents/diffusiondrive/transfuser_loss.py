from typing import Dict
import math
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_features import BoundingBox2DIndex
from navsim.agents.diffusiondrive.trajectory_value_selector import (
    compute_value_selector_loss,
)
from navsim.agents.diffusiondrive.paired_advantage_risk import (
    compute_paired_advantage_risk_loss,
)


def compute_group_relative_advantages(
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-3,
):
    """Normalize rewards within each scene while excluding invalid modes."""
    if rewards.ndim != 2 or valid_mask.shape != rewards.shape:
        raise ValueError("rewards and valid_mask must both have shape [batch, num_modes]")

    rewards = rewards.float()
    valid_mask = valid_mask.bool() & torch.isfinite(rewards)
    clean_rewards = torch.where(valid_mask, rewards, torch.zeros_like(rewards))
    counts = valid_mask.sum(dim=-1, keepdim=True)
    safe_counts = counts.clamp_min(1)
    means = clean_rewards.sum(dim=-1, keepdim=True) / safe_counts
    centered = torch.where(valid_mask, clean_rewards - means, torch.zeros_like(rewards))
    variances = centered.square().sum(dim=-1, keepdim=True) / safe_counts
    stds = variances.sqrt()
    group_valid = (counts.squeeze(-1) >= 2) & (stds.squeeze(-1) >= eps)

    advantages = centered / stds.clamp_min(eps)
    advantages = torch.where(
        valid_mask & group_valid.unsqueeze(-1), advantages, torch.zeros_like(advantages)
    ).detach()
    return advantages, valid_mask, group_valid, means.squeeze(-1), stds.squeeze(-1)


def compute_anchor_rloo_advantages(
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    rollouts_per_mode: int = 2,
    margin: float = 0.01,
    scale: float = 0.20,
    clip: float = 1.0,
):
    """Return detached magnitude-aware leave-one-out advantages per anchor.

    Candidates must be in anchor-major order. A rollout has an RLOO signal
    only when its anchor contains at least two valid rollouts; invalid rewards
    never contribute to the leave-one-out baseline.
    """
    if rewards.ndim != 2 or valid_mask.shape != rewards.shape:
        raise ValueError("rewards and valid_mask must have shape [batch, candidate]")
    if rollouts_per_mode != 2:
        raise ValueError("anchor_rloo requires rollouts_per_mode=2")
    if rewards.shape[1] % rollouts_per_mode != 0:
        raise ValueError(
            "generation candidate count must be divisible by rollouts_per_mode"
        )
    if margin < 0:
        raise ValueError("RLOO advantage margin must be non-negative")
    if scale <= 0:
        raise ValueError("RLOO advantage scale must be positive")
    if clip <= 0:
        raise ValueError("RLOO advantage clip must be positive")

    detached_rewards = rewards.detach().float()
    valid_mask = valid_mask.bool() & torch.isfinite(detached_rewards)
    num_anchors = rewards.shape[1] // rollouts_per_mode
    anchor_rewards = detached_rewards.reshape(
        rewards.shape[0], num_anchors, rollouts_per_mode
    )
    anchor_valid = valid_mask.reshape(
        rewards.shape[0], num_anchors, rollouts_per_mode
    )
    clean_rewards = torch.where(
        anchor_valid, anchor_rewards, torch.zeros_like(anchor_rewards)
    )
    valid_counts = anchor_valid.sum(dim=-1, keepdim=True)
    rloo_valid = anchor_valid & (valid_counts >= 2)
    loo_baseline = (
        (clean_rewards.sum(dim=-1, keepdim=True) - clean_rewards)
        / (valid_counts - 1).clamp_min(1)
    )
    gaps = torch.where(
        rloo_valid, anchor_rewards - loo_baseline, torch.zeros_like(anchor_rewards)
    )
    magnitudes = ((gaps.abs() - margin).clamp_min(0.0) / scale).clamp_max(clip)
    advantages = torch.where(
        rloo_valid, gaps.sign() * magnitudes, torch.zeros_like(gaps)
    )
    group_valid = valid_counts.squeeze(-1) >= 2
    return (
        advantages.reshape_as(rewards).detach(),
        gaps.reshape_as(rewards).detach(),
        rloo_valid.reshape_as(valid_mask),
        group_valid,
        valid_mask,
    )


def compute_smoothed_selector_policy(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float = 1.0,
    exploration_floor: float = 0.0,
):
    """Return a valid-mode categorical policy with optional uniform support."""
    if logits.ndim != 2 or valid_mask.shape != logits.shape:
        raise ValueError("logits and valid_mask must have shape [batch, num_modes]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= exploration_floor < 1.0:
        raise ValueError("exploration_floor must satisfy 0 <= epsilon < 1")

    valid_mask = valid_mask.bool()
    valid_counts = valid_mask.sum(dim=-1, keepdim=True)
    has_valid = valid_counts.squeeze(-1) > 0
    masked_logits = (logits.float() / temperature).masked_fill(
        ~valid_mask, torch.finfo(torch.float32).min
    )
    raw_probs = F.softmax(masked_logits, dim=-1)
    raw_probs = torch.where(
        has_valid.unsqueeze(-1),
        raw_probs * valid_mask.to(raw_probs.dtype),
        torch.zeros_like(raw_probs),
    )
    uniform_probs = valid_mask.to(raw_probs.dtype) / valid_counts.clamp_min(1)
    probs = (
        (1.0 - exploration_floor) * raw_probs
        + exploration_floor * uniform_probs
    )
    probs = torch.where(valid_mask, probs, torch.zeros_like(probs))
    log_probs = torch.where(
        valid_mask,
        probs.clamp_min(torch.finfo(probs.dtype).tiny).log(),
        torch.zeros_like(probs),
    )
    return probs, log_probs, has_valid


def _prepare_scene_weights(
    scene_weights: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if scene_weights is None:
        return torch.ones(batch_size, device=device, dtype=dtype)
    if scene_weights.shape != (batch_size,):
        raise ValueError("scene_weights must have shape [batch]")
    if not torch.isfinite(scene_weights).all() or torch.any(scene_weights < 0):
        raise ValueError("scene_weights must be finite and non-negative")
    return scene_weights.detach().to(device=device, dtype=dtype)


def _weighted_scene_mean(
    values: torch.Tensor,
    valid_scenes: torch.Tensor,
    scene_weights: torch.Tensor,
) -> torch.Tensor:
    weights = scene_weights * valid_scenes.to(scene_weights.dtype)
    weight_sum = weights.sum()
    return (
        (values * weights).sum() / weight_sum
        if weight_sum > 0
        else values.sum() * 0.0
    )


def compute_pairwise_reward_ranking_objective(
    current_logits: torch.Tensor,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    reward_gap: float = 0.01,
    reward_scale: float = 0.10,
    logit_margin: float = 0.20,
    scene_weights: torch.Tensor = None,
):
    """Rank valid modes using detached raw-reward preferences within each scene."""
    if current_logits.ndim != 2 or rewards.shape != current_logits.shape:
        raise ValueError("current_logits and rewards must have shape [batch, num_modes]")
    if valid_mask.shape != current_logits.shape:
        raise ValueError("valid_mask must match current_logits")
    if reward_gap < 0:
        raise ValueError("reward_gap must be non-negative")
    if reward_scale <= 0:
        raise ValueError("reward_scale must be positive")
    if logit_margin < 0:
        raise ValueError("logit_margin must be non-negative")

    rewards = rewards.detach().float()
    valid_mask = valid_mask.bool() & torch.isfinite(rewards)
    clean_rewards = torch.where(valid_mask, rewards, torch.zeros_like(rewards))
    reward_differences = clean_rewards.unsqueeze(2) - clean_rewards.unsqueeze(1)
    pair_valid = (
        valid_mask.unsqueeze(2)
        & valid_mask.unsqueeze(1)
        & (reward_differences > reward_gap)
    )
    pair_weights = ((reward_differences - reward_gap) / reward_scale).clamp(0.0, 1.0)
    pair_weights = torch.where(pair_valid, pair_weights, torch.zeros_like(pair_weights))

    logit_differences = current_logits.float().unsqueeze(2) - current_logits.float().unsqueeze(1)
    pair_losses = F.softplus(logit_margin - logit_differences)
    pair_weight_sums = pair_weights.sum(dim=(1, 2))
    active_scenes = pair_weight_sums > 0
    per_scene_loss = (
        (pair_weights * pair_losses).sum(dim=(1, 2))
        / pair_weight_sums.clamp_min(torch.finfo(pair_weights.dtype).eps)
    )
    scene_weights = _prepare_scene_weights(
        scene_weights,
        current_logits.shape[0],
        current_logits.device,
        current_logits.dtype,
    )
    rank_loss = _weighted_scene_mean(per_scene_loss, active_scenes, scene_weights)

    pair_counts = pair_valid.sum(dim=(1, 2)).float()
    active_pair_weights = pair_weights[pair_valid]
    weighted_gap_sum = (pair_weights * reward_differences).sum(dim=(1, 2))
    per_scene_reward_gap = weighted_gap_sum / pair_weight_sums.clamp_min(
        torch.finfo(pair_weights.dtype).eps
    )
    zero = current_logits.detach().float().new_zeros(())
    return {
        "rank_loss": rank_loss,
        "rank_active_scene_fraction": active_scenes.float().mean().detach(),
        "rank_pair_count": (
            pair_counts[active_scenes].mean().detach() if active_scenes.any() else zero
        ),
        "rank_pair_weight": (
            active_pair_weights.mean().detach() if active_pair_weights.numel() else zero
        ),
        "rank_pair_reward_gap": (
            per_scene_reward_gap[active_scenes].mean().detach()
            if active_scenes.any() else zero
        ),
    }


def compute_grpo_objective(
    current_logits: torch.Tensor,
    old_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    clip_ratio: float = 0.2,
    advantage_eps: float = 1e-3,
    temperature: float = 1.0,
    exploration_floor: float = 0.0,
    behavior_weighting: str = "old_policy",
    scene_weights: torch.Tensor = None,
):
    """Exact categorical, group-relative clipped policy objective over all modes."""
    if not (current_logits.shape == old_logits.shape == reference_logits.shape == rewards.shape):
        raise ValueError("policy logits and rewards must have identical [batch, num_modes] shapes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    advantages, valid_mask, group_valid, reward_mean, reward_std = (
        compute_group_relative_advantages(rewards, valid_mask, advantage_eps)
    )
    current_probs, current_log_probs, has_valid = compute_smoothed_selector_policy(
        current_logits,
        valid_mask,
        temperature=temperature,
        exploration_floor=exploration_floor,
    )
    old_probs, old_log_probs, _ = compute_smoothed_selector_policy(
        old_logits.detach(),
        valid_mask,
        temperature=temperature,
        exploration_floor=exploration_floor,
    )
    if behavior_weighting == "old_policy":
        behavior_weights = old_probs
    elif behavior_weighting == "uniform_valid":
        valid_counts = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        behavior_weights = valid_mask.to(old_probs.dtype) / valid_counts
    else:
        raise ValueError(
            "behavior_weighting must be one of "
            "{'old_policy', 'uniform_valid'}"
        )
    _, reference_log_probs, _ = compute_smoothed_selector_policy(
        reference_logits.detach(),
        valid_mask,
        temperature=temperature,
        exploration_floor=exploration_floor,
    )
    scene_weights = _prepare_scene_weights(
        scene_weights,
        current_logits.shape[0],
        current_logits.device,
        current_logits.dtype,
    )

    log_ratio = (current_log_probs - old_log_probs).clamp(min=-20.0, max=20.0)
    ratio = log_ratio.exp()
    surrogate = torch.minimum(
        ratio * advantages,
        ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages,
    )
    per_scene_policy_loss = -(behavior_weights * surrogate).sum(dim=-1)
    policy_loss = _weighted_scene_mean(
        per_scene_policy_loss, group_valid, scene_weights
    )

    per_scene_kl = (
        current_probs * (current_log_probs - reference_log_probs)
    ).sum(dim=-1)
    kl_loss = (
        per_scene_kl[has_valid].mean()
        if has_valid.any()
        else current_logits.sum() * 0.0
    )
    per_scene_entropy = -(current_probs * current_log_probs).sum(dim=-1)
    entropy = (
        per_scene_entropy[has_valid].mean()
        if has_valid.any()
        else current_logits.sum() * 0.0
    )
    entropy_for_loss = _weighted_scene_mean(
        per_scene_entropy, has_valid, scene_weights
    )

    selected_idx = current_logits.float().argmax(dim=-1)
    clean_rewards = torch.nan_to_num(rewards.float(), nan=0.0, posinf=0.0, neginf=0.0)
    selected_reward_by_scene = clean_rewards.gather(1, selected_idx.unsqueeze(-1)).squeeze(-1)
    oracle_reward_by_scene = clean_rewards.masked_fill(~valid_mask, float("-inf")).max(dim=-1).values
    has_valid = valid_mask.any(dim=-1)
    selected_is_valid = valid_mask.gather(1, selected_idx.unsqueeze(-1)).squeeze(-1)
    metric_valid = has_valid & selected_is_valid
    selected_reward = (
        selected_reward_by_scene[metric_valid].mean()
        if metric_valid.any() else rewards.new_zeros(())
    )
    oracle_reward = (
        oracle_reward_by_scene[metric_valid].mean()
        if metric_valid.any() else rewards.new_zeros(())
    )

    clipped = (ratio < 1.0 - clip_ratio) | (ratio > 1.0 + clip_ratio)
    oracle_idx = clean_rewards.masked_fill(~valid_mask, float("-inf")).argmax(dim=-1)
    oracle_hit_rate = (
        (selected_idx[metric_valid] == oracle_idx[metric_valid]).float().mean()
        if metric_valid.any() else rewards.new_zeros(())
    )
    clip_fraction_per_scene = (behavior_weights * clipped.float()).sum(dim=-1)
    clip_fraction = (
        clip_fraction_per_scene[group_valid].mean()
        if group_valid.any()
        else rewards.new_zeros(())
    )

    return {
        "policy_loss": policy_loss,
        "kl_loss": kl_loss,
        "entropy_for_loss": entropy_for_loss,
        "entropy": entropy.detach(),
        "reward_mean": reward_mean.mean().detach(),
        "reward_std": reward_std.mean().detach(),
        "selected_reward": selected_reward.detach(),
        "oracle_reward": oracle_reward.detach(),
        "selection_regret": (oracle_reward - selected_reward).detach(),
        "oracle_hit_rate": oracle_hit_rate.detach(),
        "valid_mode_fraction": valid_mask.float().mean().detach(),
        "valid_group_fraction": group_valid.float().mean().detach(),
        "zero_advantage_fraction": (1.0 - group_valid.float().mean()).detach(),
        "ratio_mean": (
            (behavior_weights * ratio).sum(dim=-1)[has_valid].mean()
            if has_valid.any() else rewards.new_zeros(())
        ).detach(),
        "clip_fraction": clip_fraction.detach(),
    }


def compute_generation_grpo_objective(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    generation_kl: torch.Tensor,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    mode_weights: torch.Tensor = None,
    clip_ratio: float = 0.2,
    advantage_eps: float = 1e-3,
    scene_weights: torch.Tensor = None,
    advantage_mode: str = "group_zscore",
    reference_selected_reward: torch.Tensor = None,
    reference_valid_mask: torch.Tensor = None,
    reference_anchor_rewards: torch.Tensor = None,
    reference_anchor_valid_mask: torch.Tensor = None,
    reference_margin: float = 0.01,
    reference_scale: float = 0.10,
    reference_clip: float = 2.0,
    rloo_margin: float = 0.01,
    rloo_scale: float = 0.20,
    rloo_clip: float = 1.0,
    rollouts_per_mode: int = 1,
    no_collision_scores: torch.Tensor = None,
):
    """Clipped policy objective over sampled denoising actions.

    group_zscore preserves the established generation-GRPO behavior.
    reference_centered uses the fixed reference policy deployable raw reward
    as an absolute baseline. within_anchor and hierarchical use K=2 samples
    per anchor; hierarchical mixes equal-weight within-anchor and reference
    advantages. anchor_hierarchical uses a fixed-reference raw-PDMS baseline
    for the matching anchor and supports K=2 or K=4. anchor_rloo replaces
    within-anchor standardization with a magnitude-aware K2 leave-one-out
    signal while retaining the matching fixed-reference anchor term.
    collision_truncated_intra_anchor keeps only positive K2 within-anchor
    advantages for safe candidates and assigns -1 to every valid candidate
    whose PDM no-collision component is below one.
    """
    if current_log_probs.shape != old_log_probs.shape:
        raise ValueError("current and old generation log-probs must have identical shapes")
    if current_log_probs.ndim != 3 or current_log_probs.shape[:2] != rewards.shape:
        raise ValueError("generation log-probs must have shape [batch, mode, step]")
    if generation_kl.shape != current_log_probs.shape:
        raise ValueError("generation KL must match generation log-prob shape")
    if mode_weights is None:
        mode_weights = torch.ones_like(rewards)
    if mode_weights.shape != rewards.shape:
        raise ValueError("generation mode weights must match reward shape")
    if not torch.isfinite(mode_weights).all() or torch.any(mode_weights < 0):
        raise ValueError("generation mode weights must be finite and non-negative")
    mode_weights = mode_weights.detach().to(
        device=current_log_probs.device, dtype=current_log_probs.dtype
    )
    scene_weights = _prepare_scene_weights(
        scene_weights,
        current_log_probs.shape[0],
        current_log_probs.device,
        current_log_probs.dtype,
    )

    rewards = rewards.float()
    if advantage_mode == "anchor_rloo":
        rewards = rewards.detach()
    valid_mask = valid_mask.bool() & torch.isfinite(rewards)
    reference_delta = None
    reference_valid_mask_for_metrics = None
    within_anchor_metrics = {}
    if advantage_mode == "group_zscore":
        advantages, valid_mask, group_valid, _, _ = compute_group_relative_advantages(
            rewards, valid_mask, advantage_eps
        )
        optimize_mode_mask = valid_mask & group_valid.unsqueeze(-1)
    elif advantage_mode in {
        "reference_centered",
        "within_anchor",
        "hierarchical",
        "anchor_hierarchical",
        "anchor_rloo",
        "collision_truncated_intra_anchor",
    }:
        within_anchor_mode = advantage_mode in {
            "within_anchor",
            "hierarchical",
            "anchor_hierarchical",
            "collision_truncated_intra_anchor",
        }
        rloo_mode = advantage_mode == "anchor_rloo"
        if rloo_mode:
            (
                within_advantages,
                rloo_gaps,
                rloo_valid_mask,
                anchor_group_valid,
                valid_mask,
            ) = compute_anchor_rloo_advantages(
                rewards,
                valid_mask,
                rollouts_per_mode=rollouts_per_mode,
                margin=rloo_margin,
                scale=rloo_scale,
                clip=rloo_clip,
            )
            num_anchors = rewards.shape[1] // rollouts_per_mode
            within_valid_mask = rloo_valid_mask
            absolute_gaps = rloo_gaps[rloo_valid_mask].abs()
            active_rloo = rloo_valid_mask & (within_advantages != 0)
            clipped_rloo = rloo_valid_mask & (
                within_advantages.abs() >= rloo_clip
            )
            zero = rewards.detach().new_zeros(())
            within_anchor_metrics = {
                "within_anchor_pair_fraction": (
                    anchor_group_valid.float().mean().detach()
                ),
                "within_anchor_reward_gap_mean": (
                    absolute_gaps.mean().detach()
                    if absolute_gaps.numel()
                    else zero
                ),
                "rloo_absolute_gap_mean": (
                    absolute_gaps.mean().detach()
                    if absolute_gaps.numel()
                    else zero
                ),
                "rloo_absolute_gap_p50": (
                    torch.quantile(absolute_gaps, 0.50).detach()
                    if absolute_gaps.numel()
                    else zero
                ),
                "rloo_absolute_gap_p90": (
                    torch.quantile(absolute_gaps, 0.90).detach()
                    if absolute_gaps.numel()
                    else zero
                ),
                "rloo_dead_zone_fraction": (
                    (absolute_gaps <= rloo_margin).float().mean().detach()
                    if absolute_gaps.numel()
                    else zero
                ),
                "rloo_clip_fraction": (
                    clipped_rloo[rloo_valid_mask].float().mean().detach()
                    if rloo_valid_mask.any()
                    else zero
                ),
                "rloo_active_fraction": (
                    active_rloo[valid_mask].float().mean().detach()
                    if valid_mask.any()
                    else zero
                ),
            }
        if within_anchor_mode:
            if (
                advantage_mode
                in {
                    "within_anchor",
                    "hierarchical",
                    "collision_truncated_intra_anchor",
                }
                and rollouts_per_mode != 2
            ):
                raise ValueError(
                    f"{advantage_mode} generation advantage requires "
                    "rollouts_per_mode=2"
                )
            if (
                advantage_mode == "anchor_hierarchical"
                and rollouts_per_mode not in {2, 4}
            ):
                raise ValueError(
                    "anchor_hierarchical generation advantage requires "
                    "rollouts_per_mode=2 or 4"
                )
            if rewards.shape[1] % rollouts_per_mode != 0:
                raise ValueError(
                    "generation candidate count must be divisible by "
                    "rollouts_per_mode"
                )
            num_anchors = rewards.shape[1] // rollouts_per_mode
            anchor_rewards = rewards.reshape(
                rewards.shape[0] * num_anchors, rollouts_per_mode
            )
            anchor_valid_mask = valid_mask.reshape(
                rewards.shape[0] * num_anchors, rollouts_per_mode
            )
            (
                within_advantages,
                _,
                anchor_group_valid,
                _,
                _,
            ) = compute_group_relative_advantages(
                anchor_rewards,
                anchor_valid_mask,
                advantage_eps,
            )
            within_advantages = within_advantages.reshape_as(rewards)
            within_valid_mask = (
                anchor_group_valid.reshape(rewards.shape[0], num_anchors, 1)
                .expand(-1, -1, rollouts_per_mode)
                .reshape_as(valid_mask)
            )
            anchor_reward_range = (
                anchor_rewards.masked_fill(~anchor_valid_mask, float("-inf")).max(dim=-1).values
                - anchor_rewards.masked_fill(~anchor_valid_mask, float("inf")).min(dim=-1).values
            )
            within_anchor_metrics = {
                "within_anchor_pair_fraction": (
                    anchor_group_valid.float().mean().detach()
                ),
                "within_anchor_reward_gap_mean": (
                    anchor_reward_range[anchor_group_valid].mean().detach()
                    if anchor_group_valid.any()
                    else rewards.new_zeros(())
                ),
            }
        elif not rloo_mode:
            within_advantages = None
            within_valid_mask = None

        if advantage_mode in {"reference_centered", "hierarchical"}:
            if reference_selected_reward is None:
                raise ValueError(
                    f"{advantage_mode} generation advantage requires "
                    "reference_selected_reward"
                )
            if reference_selected_reward.shape != (rewards.shape[0],):
                raise ValueError("reference_selected_reward must have shape [batch]")
            if reference_margin < 0:
                raise ValueError("reference advantage margin must be non-negative")
            if reference_scale <= 0:
                raise ValueError("reference advantage scale must be positive")
            if reference_clip <= 0:
                raise ValueError("reference advantage clip must be positive")
            if reference_valid_mask is None:
                reference_valid_mask = torch.isfinite(reference_selected_reward)
            elif reference_valid_mask.shape != (rewards.shape[0],):
                raise ValueError("reference_valid_mask must have shape [batch]")
            reference_valid_mask = (
                reference_valid_mask.bool()
                & torch.isfinite(reference_selected_reward)
            )
            reference_valid_mask_for_metrics = reference_valid_mask
            reference_delta = (
                rewards - reference_selected_reward.detach().float().unsqueeze(-1)
            )
            magnitude = (
                (reference_delta.abs() - reference_margin).clamp_min(0.0)
                / reference_scale
            )
            reference_advantages = (
                reference_delta.sign() * magnitude
            ).clamp(min=-reference_clip, max=reference_clip)
            reference_signal_mask = (
                valid_mask & reference_valid_mask.unsqueeze(-1)
            )
            reference_advantages = torch.where(
                reference_signal_mask,
                reference_advantages,
                torch.zeros_like(reference_advantages),
            ).detach()
        else:
            reference_advantages = None
            reference_signal_mask = torch.zeros_like(valid_mask)

        if advantage_mode in {"anchor_hierarchical", "anchor_rloo"}:
            if reference_margin < 0:
                raise ValueError("reference advantage margin must be non-negative")
            if reference_scale <= 0:
                raise ValueError("reference advantage scale must be positive")
            if reference_clip <= 0:
                raise ValueError("reference advantage clip must be positive")
            num_anchors = rewards.shape[1] // rollouts_per_mode
            expected_shape = (rewards.shape[0], num_anchors)
            if reference_anchor_rewards is None:
                raise ValueError(
                    f"{advantage_mode} generation advantage requires "
                    "reference_anchor_rewards"
                )
            if reference_anchor_rewards.shape != expected_shape:
                raise ValueError(
                    "reference_anchor_rewards must have shape "
                    f"{expected_shape}"
                )
            if reference_anchor_valid_mask is None:
                reference_anchor_valid_mask = torch.isfinite(
                    reference_anchor_rewards
                )
            elif reference_anchor_valid_mask.shape != expected_shape:
                raise ValueError(
                    "reference_anchor_valid_mask must match "
                    "reference_anchor_rewards"
                )
            reference_anchor_valid_mask = (
                reference_anchor_valid_mask.bool()
                & torch.isfinite(reference_anchor_rewards)
            )
            expanded_anchor_rewards = (
                reference_anchor_rewards.detach().float()
                .repeat_interleave(rollouts_per_mode, dim=-1)
            )
            expanded_anchor_valid = reference_anchor_valid_mask.repeat_interleave(
                rollouts_per_mode, dim=-1
            )
            reference_delta = rewards - expanded_anchor_rewards
            magnitude = (
                (reference_delta.abs() - reference_margin).clamp_min(0.0)
                / reference_scale
            )
            reference_advantages = (
                reference_delta.sign() * magnitude
            ).clamp(min=-reference_clip, max=reference_clip)
            reference_signal_mask = valid_mask & expanded_anchor_valid
            reference_advantages = torch.where(
                reference_signal_mask,
                reference_advantages,
                torch.zeros_like(reference_advantages),
            ).detach()
            reference_valid_mask_for_metrics = expanded_anchor_valid
            within_anchor_metrics.update(
                {
                    "reference_anchor_reward_mean": (
                        reference_anchor_rewards[
                            reference_anchor_valid_mask
                        ].float().mean().detach()
                        if reference_anchor_valid_mask.any()
                        else rewards.new_zeros(())
                    ),
                    "reference_anchor_valid_fraction": (
                        reference_anchor_valid_mask.float().mean().detach()
                    ),
                }
            )

        if advantage_mode == "reference_centered":
            advantages = reference_advantages
            optimize_mode_mask = advantages != 0
        elif advantage_mode == "within_anchor":
            advantages = within_advantages.clamp(-2.0, 2.0).detach()
            optimize_mode_mask = within_valid_mask & (advantages != 0)
        elif advantage_mode == "collision_truncated_intra_anchor":
            if no_collision_scores is None:
                raise ValueError(
                    "collision_truncated_intra_anchor requires no_collision_scores"
                )
            if no_collision_scores.shape != rewards.shape:
                raise ValueError("no_collision_scores must match generation rewards")
            no_collision_scores = no_collision_scores.detach().float()
            if not torch.isfinite(no_collision_scores[valid_mask]).all():
                raise ValueError("valid no_collision_scores must all be finite")
            collision_mask = valid_mask & (no_collision_scores < 1.0)
            safe_positive_mask = (
                valid_mask
                & ~collision_mask
                & within_valid_mask
                & (within_advantages > 0)
            )
            advantages = torch.where(
                safe_positive_mask,
                within_advantages.clamp(max=2.0),
                torch.zeros_like(within_advantages),
            )
            advantages = torch.where(
                collision_mask, -torch.ones_like(advantages), advantages
            ).detach()
            optimize_mode_mask = collision_mask | safe_positive_mask
            valid_count_for_truncation = valid_mask.float().sum().clamp_min(1.0)
            safe_zero_mask = valid_mask & ~collision_mask & ~safe_positive_mask
            within_anchor_metrics.update(
                {
                    "truncated_positive_fraction": (
                        safe_positive_mask.float().sum() / valid_count_for_truncation
                    ).detach(),
                    "truncated_safe_zero_fraction": (
                        safe_zero_mask.float().sum() / valid_count_for_truncation
                    ).detach(),
                    "collision_penalty_fraction": (
                        collision_mask.float().sum() / valid_count_for_truncation
                    ).detach(),
                    "collision_candidate_count": collision_mask.float().sum().detach(),
                    "optimized_candidate_count": (
                        optimize_mode_mask.float().sum().detach()
                    ),
                    "valid_candidate_count": valid_mask.float().sum().detach(),
                }
            )
        elif advantage_mode == "anchor_rloo":
            advantages = (
                0.5 * within_advantages
                + 0.5 * reference_advantages
            ).detach()
            optimize_mode_mask = (
                valid_mask
                & (within_valid_mask | reference_signal_mask)
                & (advantages != 0)
            )
        else:
            advantages = (
                0.5 * within_advantages.clamp(-2.0, 2.0)
                + 0.5 * reference_advantages
            ).detach()
            optimize_mode_mask = (
                valid_mask
                & (within_valid_mask | reference_signal_mask)
                & (advantages != 0)
            )
        group_valid = optimize_mode_mask.any(dim=-1)
    else:
        raise ValueError(
            "generation advantage mode must be one of "
            "{'group_zscore', 'reference_centered', "
            "'within_anchor', 'hierarchical', 'anchor_hierarchical', "
            "'anchor_rloo', 'collision_truncated_intra_anchor'}"
        )
    log_ratio = (current_log_probs - old_log_probs.detach()).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    step_advantages = advantages.unsqueeze(-1)
    surrogate = torch.minimum(
        ratio * step_advantages,
        ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * step_advantages,
    )
    optimize_mask = optimize_mode_mask.unsqueeze(-1).expand_as(surrogate)
    step_weights = mode_weights.unsqueeze(-1).expand_as(surrogate)
    optimize_weights = (
        step_weights
        * optimize_mask.to(step_weights.dtype)
        * scene_weights[:, None, None]
    )
    optimize_weight_sum = optimize_weights.sum()
    policy_loss = (
        -(surrogate * optimize_weights).sum() / optimize_weight_sum
        if optimize_weight_sum > 0
        else current_log_probs.sum() * 0.0
    )
    kl_mask = valid_mask.unsqueeze(-1).expand_as(generation_kl)
    kl_loss = (
        generation_kl[kl_mask].mean()
        if kl_mask.any()
        else generation_kl.sum() * 0.0
    )
    clipped = (ratio < 1.0 - clip_ratio) | (ratio > 1.0 + clip_ratio)
    metric_weights = step_weights * kl_mask.to(step_weights.dtype)
    metric_weight_sum = metric_weights.sum()
    ratio_mean = (
        (ratio * metric_weights).sum() / metric_weight_sum
        if metric_weight_sum > 0
        else ratio.sum() * 0.0
    )
    clip_fraction = (
        (clipped.float() * metric_weights).sum() / metric_weight_sum
        if metric_weight_sum > 0
        else clipped.float().sum() * 0.0
    )
    valid_count = valid_mask.float().sum().clamp_min(1.0)
    result = {
        "policy_loss": policy_loss,
        "kl_loss": kl_loss,
        "ratio_mean": ratio_mean.detach(),
        "clip_fraction": clip_fraction.detach(),
        "policy_active_scene_fraction": group_valid.float().mean().detach(),
        "positive_advantage_fraction": (
            ((advantages > 0) & valid_mask).float().sum() / valid_count
        ).detach(),
        "negative_advantage_fraction": (
            ((advantages < 0) & valid_mask).float().sum() / valid_count
        ).detach(),
        **within_anchor_metrics,
    }
    if reference_delta is not None:
        reference_metric_mask = reference_signal_mask
        if reference_metric_mask.any():
            valid_delta = reference_delta[reference_metric_mask]
            result.update(
                {
                    "reference_delta_mean": valid_delta.mean().detach(),
                    "reference_delta_std": valid_delta.std(unbiased=False).detach(),
                    "within_margin_fraction": (
                        (valid_delta.abs() <= reference_margin).float().mean().detach()
                    ),
                }
            )
        else:
            zero = rewards.new_zeros(())
            result.update(
                {
                    "reference_delta_mean": zero,
                    "reference_delta_std": zero,
                    "within_margin_fraction": zero,
                }
            )
    return result


def compute_full_chain_diffgrpo_objective(
    current_log_probs: torch.Tensor,
    bc_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    advantage_eps: float = 1e-3,
    bc_weight: float = 0.1,
    step_discount: float = 0.6,
) -> Dict[str, torch.Tensor]:
    """ReCogDrive-style on-policy objective for a complete denoising chain."""
    if current_log_probs.ndim != 3:
        raise ValueError("full-chain log probabilities must have shape [B,M,S]")
    if (
        bc_log_probs.ndim != 3
        or bc_log_probs.shape[0] != current_log_probs.shape[0]
        or bc_log_probs.shape[-1] != current_log_probs.shape[-1]
        or bc_log_probs.shape[1] not in {1, current_log_probs.shape[1]}
    ):
        raise ValueError(
            "full-chain current/BC steps must match; BC shape is [B,1,S] or [B,M,S]"
        )
    if rewards.shape != current_log_probs.shape[:2] or valid_mask.shape != rewards.shape:
        raise ValueError("full-chain rewards/valid mask must have shape [B,M]")
    if not math.isfinite(float(bc_weight)) or bc_weight < 0:
        raise ValueError("diffgrpo BC weight must be finite and non-negative")
    if not math.isfinite(float(step_discount)) or not 0 < step_discount <= 1:
        raise ValueError("diffgrpo step discount must satisfy 0 < gamma <= 1")
    if not torch.isfinite(current_log_probs).all() or not torch.isfinite(bc_log_probs).all():
        raise FloatingPointError("full-chain log probabilities must be finite")

    advantages, finite_valid, group_valid, reward_mean, reward_std = (
        compute_group_relative_advantages(rewards, valid_mask, advantage_eps)
    )
    optimize_mode = finite_valid & group_valid.unsqueeze(-1)
    num_steps = current_log_probs.shape[-1]
    indices = torch.arange(
        num_steps, device=current_log_probs.device, dtype=current_log_probs.dtype
    )
    discounts = step_discount ** (num_steps - indices - 1)
    weighted_advantage = advantages.unsqueeze(-1) * discounts.view(1, 1, -1)
    optimize_mask = optimize_mode.unsqueeze(-1).expand_as(current_log_probs)
    policy_terms = -(current_log_probs * weighted_advantage)
    policy_loss = (
        policy_terms[optimize_mask].mean()
        if optimize_mask.any()
        else current_log_probs.sum() * 0.0
    )
    bc_mask = torch.isfinite(bc_log_probs)
    bc_loss = (
        -bc_log_probs[bc_mask].mean()
        if bc_mask.any()
        else bc_log_probs.sum() * 0.0
    )
    total_loss = policy_loss + float(bc_weight) * bc_loss
    valid_count = finite_valid.float().sum().clamp_min(1.0)
    return {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "bc_loss": bc_loss,
        "reward_mean": (
            rewards.float()[finite_valid].mean().detach()
            if finite_valid.any() else rewards.sum() * 0.0
        ),
        "reward_std": (
            reward_std[group_valid].mean().detach()
            if group_valid.any() else rewards.sum() * 0.0
        ),
        "policy_active_scene_fraction": group_valid.float().mean().detach(),
        "positive_advantage_fraction": (
            ((advantages > 0) & finite_valid).float().sum() / valid_count
        ).detach(),
        "negative_advantage_fraction": (
            ((advantages < 0) & finite_valid).float().sum() / valid_count
        ).detach(),
        "discount_first": discounts[0].detach(),
        "discount_last": discounts[-1].detach(),
        "mean_current_log_prob": current_log_probs.mean().detach(),
        "mean_bc_log_prob": bc_log_probs.mean().detach(),
    }


def compute_reference_headroom_scene_weights(
    raw_rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    reference_selected_reward: torch.Tensor,
    reference_valid_mask: torch.Tensor = None,
    margin: float = 0.01,
    scale: float = 0.05,
):
    """Compute detached per-scene update weights from deployable reference reward."""
    if raw_rewards.ndim != 2 or valid_mask.shape != raw_rewards.shape:
        raise ValueError("raw_rewards and valid_mask must have shape [batch, num_modes]")
    if reference_selected_reward.shape != (raw_rewards.shape[0],):
        raise ValueError("reference_selected_reward must have shape [batch]")
    if margin < 0:
        raise ValueError("reference gate margin must be non-negative")
    if scale <= 0:
        raise ValueError("reference gate scale must be positive")

    valid_mask = valid_mask.bool() & torch.isfinite(raw_rewards)
    has_candidate = valid_mask.any(dim=-1)
    oracle_reward = raw_rewards.float().masked_fill(
        ~valid_mask, float("-inf")
    ).max(dim=-1).values
    if reference_valid_mask is None:
        reference_valid_mask = torch.isfinite(reference_selected_reward)
    else:
        if reference_valid_mask.shape != (raw_rewards.shape[0],):
            raise ValueError("reference_valid_mask must have shape [batch]")
        reference_valid_mask = (
            reference_valid_mask.bool() & torch.isfinite(reference_selected_reward)
        )
    valid_scene = has_candidate & reference_valid_mask
    headroom = oracle_reward - reference_selected_reward.float()
    headroom = torch.where(valid_scene, headroom, torch.zeros_like(headroom))
    weights = ((headroom - margin) / scale).clamp(0.0, 1.0)
    weights = torch.where(valid_scene, weights, torch.zeros_like(weights))
    return weights.detach(), headroom.detach(), valid_scene.detach()


def compute_selector_consistency_kl(
    current_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return KL(reference || current) on the same candidate trace.

    The reference logits are detached so the regularizer can constrain shared
    current-policy features while the fixed reference policy stays frozen.
    """
    if current_logits.shape != reference_logits.shape or current_logits.ndim != 2:
        raise ValueError(
            "current and reference selector logits must have identical "
            "[batch, num_modes] shapes"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    current_log_probs = F.log_softmax(current_logits.float() / temperature, dim=-1)
    reference_probs = F.softmax(
        reference_logits.detach().float() / temperature, dim=-1
    )
    return F.kl_div(
        current_log_probs,
        reference_probs,
        reduction="batchmean",
    )


def transfuser_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfig
):
    """Pure GRPO objective for selection, generation, or their joint policy."""
    current_logits = predictions["final_poses_cls"]
    training_mode = getattr(config, "grpo_training_mode", "classification_shared")
    if training_mode == "paired_tail_risk_selector":
        if not predictions.get("grpo_training_rollout", True):
            zero = current_logits.sum() * 0.0
            return {"loss": zero, "paired_risk_loss": zero.detach()}
        result = compute_paired_advantage_risk_loss(
            predictions,
            positive_weights=getattr(
                config, "paired_risk_positive_weights", (1.0,) * 6
            ),
            selected_mode_weight=float(
                getattr(config, "paired_risk_selected_mode_weight", 4.0)
            ),
            delta_loss_weight=float(
                getattr(config, "paired_risk_delta_loss_weight", 0.25)
            ),
        )
        result["paired_risk_loss"] = result["loss"].detach()
        return result
    if training_mode == "value_selector":
        if not predictions.get("grpo_training_rollout", True):
            zero = current_logits.sum() * 0.0
            return {"loss": zero, "value_selector_loss": zero.detach()}
        result = compute_value_selector_loss(
            predictions,
            reward_gap=float(
                getattr(config, "value_selector_pair_reward_gap", 0.01)
            ),
        )
        result["value_selector_loss"] = result["loss"].detach()
        return result
    grpo_weight = getattr(config, "policy_loss_weight", 1.0)
    kl_weight = getattr(config, "kl_loss_weight", 0.01)
    entropy_weight = getattr(config, "selection_entropy_weight", 0.0)
    valid_mask = predictions.get(
        "reward_valid_mask", torch.ones_like(predictions["rewards"], dtype=torch.bool)
    ) if predictions.get("rewards") is not None else None

    # Validation/inference does not sample from the current and old behavior
    # policies, so a PPO objective is undefined there. Keep the validation
    # loop side-effect free and reserve all GRPO tensors for training forwards.
    if not predictions.get("grpo_training_rollout", True):
        zero = current_logits.sum() * 0.0
        return {
            "loss": zero,
            "grpo_loss": zero.detach(),
            "kl_loss": zero.detach(),
        }

    if "rewards" not in predictions or predictions["rewards"] is None:
        zero = current_logits.sum() * 0.0
        return {
            "loss": zero,
            "grpo_loss": zero.detach(),
            "kl_loss": zero.detach(),
        }

    if training_mode in {"diffgrpo_full_chain", "diffgrpo_selected_anchor"}:
        required = (
            "diffgrpo_current_log_probs",
            "diffgrpo_bc_log_probs",
            "raw_rewards",
            "reward_valid_mask",
        )
        missing = [name for name in required if predictions.get(name) is None]
        if missing:
            raise ValueError(f"Missing full-chain DiffGRPO tensors: {missing}")
        objective = compute_full_chain_diffgrpo_objective(
            current_log_probs=predictions["diffgrpo_current_log_probs"],
            bc_log_probs=predictions["diffgrpo_bc_log_probs"],
            rewards=predictions["raw_rewards"],
            valid_mask=predictions["reward_valid_mask"],
            advantage_eps=float(getattr(config, "grpo_advantage_eps", 1e-3)),
            bc_weight=float(getattr(config, "diffgrpo_bc_weight", 0.1)),
            step_discount=float(getattr(config, "diffgrpo_step_discount", 0.6)),
        )
        return {
            "loss": objective["loss"],
            "generation_grpo_loss": objective["policy_loss"],
            "diffgrpo_bc_loss": objective["bc_loss"],
            "raw_reward_mean": objective["reward_mean"],
            "raw_reward_std": objective["reward_std"],
            "generation_policy_active_scene_fraction": objective[
                "policy_active_scene_fraction"
            ],
            "generation_positive_advantage_fraction": objective[
                "positive_advantage_fraction"
            ],
            "generation_negative_advantage_fraction": objective[
                "negative_advantage_fraction"
            ],
            "diffgrpo_discount_first": objective["discount_first"],
            "diffgrpo_discount_last": objective["discount_last"],
            "diffgrpo_mean_current_log_prob": objective["mean_current_log_prob"],
            "diffgrpo_mean_bc_log_prob": objective["mean_bc_log_prob"],
        }

    if predictions.get("final_old_poses_cls") is None:
        raise ValueError("GRPO training requires final_old_poses_cls")
    if predictions.get("final_ref_poses_cls") is None:
        raise ValueError("GRPO training requires final_ref_poses_cls")

    scene_weight_mode = getattr(config, "grpo_scene_weight_mode", "uniform")
    if scene_weight_mode == "uniform":
        scene_weights = torch.ones(
            current_logits.shape[0],
            device=current_logits.device,
            dtype=current_logits.dtype,
        )
        headroom = None
        reference_scene_valid = None
    elif scene_weight_mode == "reference_headroom":
        raw_for_gate = predictions.get("raw_rewards")
        reference_reward = predictions.get("reference_selected_reward")
        if raw_for_gate is None or reference_reward is None:
            if predictions.get("grpo_training_rollout", True):
                raise ValueError(
                    "reference_headroom gating requires raw_rewards and "
                    "reference_selected_reward during training"
                )
            scene_weights = torch.ones(
                current_logits.shape[0],
                device=current_logits.device,
                dtype=current_logits.dtype,
            )
            headroom = None
            reference_scene_valid = None
        else:
            scene_weights, headroom, reference_scene_valid = (
                compute_reference_headroom_scene_weights(
                    raw_rewards=raw_for_gate,
                    valid_mask=valid_mask,
                    reference_selected_reward=reference_reward,
                    reference_valid_mask=predictions.get("reference_reward_valid_mask"),
                    margin=getattr(config, "grpo_reference_gate_margin", 0.01),
                    scale=getattr(config, "grpo_reference_gate_scale", 0.05),
                )
            )
    else:
        raise ValueError(
            "grpo_scene_weight_mode must be one of "
            "{'uniform', 'reference_headroom'}"
        )

    exploration_floor = getattr(config, "selection_exploration_floor", 0.0)
    behavior_weighting = getattr(
        config, "selection_behavior_weighting", "old_policy"
    )
    objective = compute_grpo_objective(
        current_logits=current_logits,
        old_logits=predictions["final_old_poses_cls"],
        reference_logits=predictions["final_ref_poses_cls"],
        rewards=(
            predictions.get("raw_rewards")
            if getattr(config, "grpo_training_mode", "") == "selector_group"
            else predictions["rewards"]
        ),
        valid_mask=valid_mask,
        clip_ratio=getattr(config, "grpo_clip_ratio", 0.2),
        advantage_eps=getattr(config, "grpo_advantage_eps", 1e-3),
        temperature=getattr(config, "selection_temperature", 1.0),
        exploration_floor=exploration_floor,
        behavior_weighting=behavior_weighting,
        scene_weights=scene_weights,
    )
    raw_rewards = predictions.get("raw_rewards")
    raw_objective = None
    if raw_rewards is not None:
        raw_objective = compute_grpo_objective(
            current_logits=current_logits,
            old_logits=predictions["final_old_poses_cls"],
            reference_logits=predictions["final_ref_poses_cls"],
            rewards=raw_rewards,
            valid_mask=predictions.get(
                "reward_valid_mask",
                torch.ones_like(raw_rewards, dtype=torch.bool),
            ),
            clip_ratio=getattr(config, "grpo_clip_ratio", 0.2),
            advantage_eps=getattr(config, "grpo_advantage_eps", 1e-3),
            temperature=getattr(config, "selection_temperature", 1.0),
            exploration_floor=exploration_floor,
            behavior_weighting=behavior_weighting,
        )

    generation_only = training_mode in {
        "generation", "generation_group", "generation_group_adaptive"
    }
    selection_weight = 0.0 if generation_only else grpo_weight
    selection_regularization_enabled = not generation_only
    rank_weight = (
        getattr(config, "selection_rank_loss_weight", 0.0)
        if selection_regularization_enabled else 0.0
    )
    if rank_weight < 0:
        raise ValueError("selection_rank_loss_weight must be non-negative")
    rank_objective = None
    if rank_weight > 0:
        ranking_rewards = raw_rewards
        if (
            ranking_rewards is None
            and not predictions.get("grpo_training_rollout", True)
            and getattr(config, "grpo_reward_mode", "pdms") == "pdms"
        ):
            ranking_rewards = predictions["rewards"]
        if ranking_rewards is None:
            raise ValueError("selector ranking requires raw_rewards during training")
        rank_objective = compute_pairwise_reward_ranking_objective(
            current_logits=current_logits,
            rewards=ranking_rewards,
            valid_mask=predictions.get(
                "reward_valid_mask", torch.ones_like(ranking_rewards, dtype=torch.bool)
            ),
            reward_gap=getattr(config, "selection_rank_reward_gap", 0.01),
            reward_scale=getattr(config, "selection_rank_reward_scale", 0.10),
            logit_margin=getattr(config, "selection_rank_logit_margin", 0.20),
            scene_weights=scene_weights,
        )
    total_loss = (
        selection_weight * objective["policy_loss"]
        + (rank_weight * rank_objective["rank_loss"] if rank_objective is not None else 0.0)
        + (kl_weight if selection_regularization_enabled else 0.0)
        * objective["kl_loss"]
        - (entropy_weight if selection_regularization_enabled else 0.0)
        * objective["entropy_for_loss"]
    )

    selector_consistency_kl = compute_selector_consistency_kl(
        current_logits=current_logits,
        reference_logits=predictions["final_ref_poses_cls"],
        temperature=getattr(config, "selection_temperature", 1.0),
    )
    selector_consistency_weight = (
        getattr(config, "selector_consistency_kl_weight", 0.0)
        if training_mode == "generation"
        else 0.0
    )
    total_loss = total_loss + selector_consistency_weight * selector_consistency_kl

    selector_generation_kl = current_logits.sum() * 0.0
    if (
        training_mode == "selector_group"
        and predictions.get("grpo_training_rollout", True)
    ):
        selector_generation_kl_tensor = predictions.get("selector_generation_kl")
        if selector_generation_kl_tensor is None:
            raise ValueError("selector_group requires selector_generation_kl")
        if selector_generation_kl_tensor.shape != (*current_logits.shape, 2):
            raise ValueError(
                "selector_generation_kl must have shape [batch, mode, 2]"
            )
        selector_kl_mask = valid_mask.unsqueeze(-1).expand_as(
            selector_generation_kl_tensor
        )
        selector_generation_kl = (
            selector_generation_kl_tensor[selector_kl_mask].mean()
            if selector_kl_mask.any()
            else selector_generation_kl_tensor.sum() * 0.0
        )
        total_loss = total_loss + getattr(
            config, "selector_generation_kl_weight", 0.0
        ) * selector_generation_kl

    generation_objective = None
    if (
        training_mode in {
            "generation", "generation_group", "generation_group_adaptive", "joint"
        }
        and predictions.get("generation_current_log_probs") is not None
    ):
        required = (
            "generation_current_log_probs",
            "generation_old_log_probs",
            "generation_kl",
        )
        missing = [key for key in required if predictions.get(key) is None]
        if missing:
            raise ValueError(f"Missing generation GRPO tensors: {missing}")
        generation_mode_weighting = getattr(
            config, "generation_mode_weighting", "uniform"
        )
        generation_mode_temperature = float(
            getattr(config, "generation_mode_temperature", 1.0)
        )
        if generation_mode_temperature <= 0:
            raise ValueError("generation_mode_temperature must be positive")
        if generation_mode_weighting == "uniform":
            generation_mode_weights = None
        elif generation_mode_weighting == "selector_softmax":
            generation_mode_weights = F.softmax(
                current_logits.detach() / generation_mode_temperature,
                dim=-1,
            )
        elif generation_mode_weighting == "selector_top1":
            generation_mode_weights = F.one_hot(
                current_logits.detach().argmax(dim=-1),
                num_classes=current_logits.shape[-1],
            ).to(current_logits.dtype)
        else:
            raise ValueError(
                "generation_mode_weighting must be one of "
                "{'uniform', 'selector_softmax', 'selector_top1'}"
            )
        generation_advantage_mode = getattr(
            config, "generation_advantage_mode", "group_zscore"
        )
        generation_rewards = predictions["rewards"]
        if generation_advantage_mode in {
            "reference_centered",
            "anchor_hierarchical",
            "anchor_rloo",
        }:
            generation_rewards = predictions.get("raw_rewards")
            if generation_rewards is None:
                raise ValueError(
                    f"{generation_advantage_mode} generation advantage "
                    "requires raw_rewards"
                )
        generation_objective = compute_generation_grpo_objective(
            current_log_probs=predictions["generation_current_log_probs"],
            old_log_probs=predictions["generation_old_log_probs"],
            generation_kl=predictions["generation_kl"],
            rewards=generation_rewards,
            valid_mask=predictions.get(
                "reward_valid_mask",
                torch.ones_like(generation_rewards, dtype=torch.bool),
            ),
            mode_weights=generation_mode_weights,
            clip_ratio=getattr(config, "grpo_clip_ratio", 0.2),
            advantage_eps=getattr(config, "grpo_advantage_eps", 1e-3),
            scene_weights=scene_weights,
            advantage_mode=generation_advantage_mode,
            reference_selected_reward=predictions.get(
                "reference_selected_reward"
            ),
            reference_valid_mask=predictions.get(
                "reference_reward_valid_mask"
            ),
            reference_anchor_rewards=predictions.get(
                "reference_anchor_rewards"
            ),
            reference_anchor_valid_mask=predictions.get(
                "reference_anchor_valid_mask"
            ),
            reference_margin=getattr(
                config, "reference_advantage_margin", 0.01
            ),
            reference_scale=getattr(
                config, "reference_advantage_scale", 0.10
            ),
            reference_clip=getattr(
                config, "reference_advantage_clip", 2.0
            ),
            rloo_margin=getattr(
                config, "rloo_advantage_margin", 0.01
            ),
            rloo_scale=getattr(
                config, "rloo_advantage_scale", 0.20
            ),
            rloo_clip=getattr(
                config, "rloo_advantage_clip", 1.0
            ),
            rollouts_per_mode=getattr(
                config, "grpo_rollouts_per_mode", 1
            ),
            no_collision_scores=(
                predictions.get("component_scores")[..., 0]
                if predictions.get("component_scores") is not None
                else None
            ),
        )
        generation_kl_coefficient = predictions.get("generation_kl_coefficient")
        if generation_kl_coefficient is None:
            generation_kl_coefficient = generation_objective["kl_loss"].new_tensor(
                float(getattr(config, "generation_kl_loss_weight", 0.1))
            )
        else:
            generation_kl_coefficient = generation_kl_coefficient.detach().to(
                generation_objective["kl_loss"]
            )
            if generation_kl_coefficient.numel() != 1:
                raise ValueError("generation_kl_coefficient must be scalar")
            if not torch.isfinite(generation_kl_coefficient).all() or torch.any(
                generation_kl_coefficient < 0
            ):
                raise ValueError("generation_kl_coefficient must be finite and non-negative")
        total_loss = (
            total_loss
            + getattr(config, "generation_policy_loss_weight", 1.0)
            * generation_objective["policy_loss"]
            + generation_kl_coefficient * generation_objective["kl_loss"]
        )

    result = {
        "loss": total_loss,
        "grpo_loss": selection_weight * objective["policy_loss"],
        "kl_loss": objective["kl_loss"],
        "selector_consistency_kl_loss": selector_consistency_kl,
        **{
            key: value for key, value in objective.items()
            if key not in {"policy_loss", "kl_loss", "entropy_for_loss"}
        },
        "selector_generation_kl_loss": selector_generation_kl.detach(),
        "grpo_scene_weight": scene_weights.float().mean().detach(),
        "active_scene_fraction": (scene_weights > 0).float().mean().detach(),
    }
    if rank_objective is not None:
        result.update(rank_objective)
    else:
        result["rank_loss"] = current_logits.detach().new_zeros(())
    if headroom is not None:
        result["oracle_headroom"] = (
            headroom[reference_scene_valid].mean().detach()
            if reference_scene_valid.any()
            else headroom.new_zeros(())
        )
        reference_reward = predictions["reference_selected_reward"].float()
        result["reference_selected_reward"] = (
            reference_reward[reference_scene_valid].mean().detach()
            if reference_scene_valid.any()
            else reference_reward.new_zeros(())
        )
    elif predictions.get("reference_selected_reward") is not None:
        reference_reward = predictions["reference_selected_reward"].float()
        reference_valid = predictions.get("reference_reward_valid_mask")
        if reference_valid is None:
            reference_valid = torch.isfinite(reference_reward)
        else:
            reference_valid = reference_valid.bool() & torch.isfinite(
                reference_reward
            )
        result["reference_selected_reward"] = (
            reference_reward[reference_valid].mean().detach()
            if reference_valid.any()
            else reference_reward.new_zeros(())
        )
    if generation_objective is not None:
        result.update(
            {
                "generation_grpo_loss": generation_objective["policy_loss"],
                "generation_kl_loss": generation_objective["kl_loss"],
                "generation_kl_coefficient": generation_kl_coefficient,
                "generation_ratio_mean": generation_objective["ratio_mean"],
                "generation_clip_fraction": generation_objective["clip_fraction"],
                "generation_policy_active_scene_fraction": generation_objective[
                    "policy_active_scene_fraction"
                ],
                "generation_positive_advantage_fraction": generation_objective[
                    "positive_advantage_fraction"
                ],
                "generation_negative_advantage_fraction": generation_objective[
                    "negative_advantage_fraction"
                ],
            }
        )
        for key in (
            "reference_delta_mean",
            "reference_delta_std",
            "within_margin_fraction",
            "within_anchor_pair_fraction",
            "within_anchor_reward_gap_mean",
            "reference_anchor_reward_mean",
            "reference_anchor_valid_fraction",
            "rloo_absolute_gap_mean",
            "rloo_absolute_gap_p50",
            "rloo_absolute_gap_p90",
            "rloo_dead_zone_fraction",
            "rloo_clip_fraction",
            "rloo_active_fraction",
            "truncated_positive_fraction",
            "truncated_safe_zero_fraction",
            "collision_penalty_fraction",
            "collision_candidate_count",
            "optimized_candidate_count",
            "valid_candidate_count",
        ):
            if key in generation_objective:
                result[f"generation_{key}"] = generation_objective[key]
        for key, value in predictions.items():
            if key.startswith("generation_trust_"):
                if not torch.is_tensor(value) or value.numel() != 1:
                    raise ValueError(f"{key} must be one scalar tensor")
                result[key] = value.detach()
    if raw_objective is not None:
        result.update(
            {
                "raw_reward_mean": raw_objective["reward_mean"],
                "raw_reward_std": raw_objective["reward_std"],
                "raw_selected_reward": raw_objective["selected_reward"],
                "raw_oracle_reward": raw_objective["oracle_reward"],
                "raw_selection_regret": raw_objective["selection_regret"],
            }
        )
    tie_epsilon = predictions.get("reward_tiebreak_epsilon")
    if tie_epsilon is not None:
        result["reward_tiebreak_epsilon"] = tie_epsilon.float().mean().detach()
    return result


def _agent_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfig
):
    """
    Hungarian matching loss for agent detection
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: detection loss
    """

    gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
    pred_states, pred_logits = predictions["agent_states"], predictions["agent_labels"]

    if config.latent:
        rad_to_ego = torch.arctan2(
            gt_states[..., BoundingBox2DIndex.Y],
            gt_states[..., BoundingBox2DIndex.X],
        )

        in_latent_rad_thresh = torch.logical_and(
            -config.latent_rad_thresh <= rad_to_ego,
            rad_to_ego <= config.latent_rad_thresh,
        )
        gt_valid = torch.logical_and(in_latent_rad_thresh, gt_valid)

    # save constants
    batch_dim, num_instances = pred_states.shape[:2]
    num_gt_instances = gt_valid.sum()
    num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1

    ce_cost = _get_ce_cost(gt_valid, pred_logits)
    l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)

    cost = config.agent_class_weight * ce_cost + config.agent_box_weight * l1_cost
    cost = cost.cpu()

    indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
    matching = [
        (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        for i, j in indices
    ]
    idx = _get_src_permutation_idx(matching)

    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

    pred_valid_idx = pred_logits[idx]
    gt_valid_idx = torch.cat([t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0).float()

    l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
    l1_loss = l1_loss.sum(-1) * gt_valid_idx
    l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt_instances

    ce_loss = F.binary_cross_entropy_with_logits(pred_valid_idx, gt_valid_idx, reduction="none")
    ce_loss = ce_loss.view(batch_dim, -1).mean()

    return ce_loss, l1_loss


@torch.no_grad()
def _get_ce_cost(gt_valid: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Function to calculate cross-entropy cost for cost matrix.
    :param gt_valid: tensor of binary ground-truth labels
    :param pred_logits: tensor of predicted logits of neural net
    :return: bce cost matrix as tensor
    """

    # NOTE: numerically stable BCE with logits
    # https://github.com/pytorch/pytorch/blob/c64e006fc399d528bb812ae589789d0365f3daf4/aten/src/ATen/native/Loss.cpp#L214
    gt_valid_expanded = gt_valid[:, :, None].detach().float()  # (b, n, 1)
    pred_logits_expanded = pred_logits[:, None, :].detach()  # (b, 1, n)

    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(
        torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val)
    )
    ce_cost = (1 - gt_valid_expanded) * pred_logits_expanded + helper_term  # (b, n, n)
    ce_cost = ce_cost.permute(0, 2, 1)

    return ce_cost


@torch.no_grad()
def _get_l1_cost(
    gt_states: torch.Tensor, pred_states: torch.Tensor, gt_valid: torch.Tensor
) -> torch.Tensor:
    """
    Function to calculate L1 cost for cost matrix.
    :param gt_states: tensor of ground-truth bounding boxes
    :param pred_states: tensor of predicted bounding boxes
    :param gt_valid: mask of binary ground-truth labels
    :return: l1 cost matrix as tensor
    """

    gt_states_expanded = gt_states[:, :, None, :2].detach()  # (b, n, 1, 2)
    pred_states_expanded = pred_states[:, None, :, :2].detach()  # (b, 1, n, 2)
    l1_cost = gt_valid[..., None].float() * (gt_states_expanded - pred_states_expanded).abs().sum(
        dim=-1
    )
    l1_cost = l1_cost.permute(0, 2, 1)
    return l1_cost


def _get_src_permutation_idx(indices):
    """
    Helper function to align indices after matching
    :param indices: matched indices
    :return: permuted indices
    """
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

def compute_grpo_loss(
    current_poses_cls: torch.Tensor,
    ref_poses_cls: torch.Tensor,
    rewards: torch.Tensor,
    num_modes: int,
    clip_advantage_lower_quantile: float = 0.0,
    clip_advantage_upper_quantile: float = 1.0,
    clip_ratio: float = 0.2
) -> torch.Tensor:
    """
    计算GRPO损失
    
    Args:
        current_poses_cls: 当前策略的分类logits [bs, num_modes]
        ref_poses_cls: 参考策略的分类logits [bs, num_modes]
        rewards: PDM奖励 [bs, num_modes]
        num_modes: 轨迹模式数量
        clip_advantage_lower_quantile: 优势函数裁剪下分位数
        clip_advantage_upper_quantile: 优势函数裁剪上分位数
        clip_ratio: PPO裁剪比例
    """
    
    batch_size = current_poses_cls.shape[0]
    
    # 1. 计算优势函数
    mean_r = rewards.mean(dim=1, keepdim=True)  # [bs, 1]
    std_r = rewards.std(dim=1, keepdim=True) + 1e-8  # [bs, 1]
    #advantages = ((rewards - mean_r) / std_r)  # [bs, num_modes]
    
    # 确保std_r不为零，且没有NaN
    std_r = torch.where(std_r < 1e-6, torch.ones_like(std_r) * 1e-6, std_r)
    std_r = torch.clamp(std_r, min=1e-6, max=1e6)  # 防止过大过小
    
    advantages = (rewards - mean_r) / (std_r + 1e-8)  # [bs, num_modes]
    
    # 扁平化并裁剪优势函数
    advantages_flat = advantages.view(-1).detach()  # [bs * num_modes]
    advantages_flat = advantages_flat.float()
    adv_min = torch.quantile(advantages_flat, clip_advantage_lower_quantile)
    adv_max = torch.quantile(advantages_flat, clip_advantage_upper_quantile)
    advantages_clipped = advantages.clamp(min=adv_min, max=adv_max)
    
    # 2. 计算对数概率比
    current_probs = F.softmax(current_poses_cls, dim=-1)  # [bs, num_modes]
    ref_probs = F.softmax(ref_poses_cls, dim=-1)  # [bs, num_modes]
    
    # 避免数值问题
    current_probs = current_probs.clamp(min=1e-7, max=1.0)
    ref_probs = ref_probs.clamp(min=1e-7, max=1.0)
    
    log_ratios = torch.log(current_probs) - torch.log(ref_probs)
    
    policy_loss = -torch.mean(log_ratios * advantages)
    
    return policy_loss

def compute_grpo_loss2(
    current_poses_cls: torch.Tensor,
    ref_poses_cls: torch.Tensor,
    rewards: torch.Tensor,
    num_modes: int,
    clip_advantage_lower_quantile: float = 0.0,
    clip_advantage_upper_quantile: float = 1.0,
    clip_ratio: float = 0.2
) -> torch.Tensor:
    """
    计算GRPO损失
    """
    batch_size = current_poses_cls.shape[0]
    
    # ===== 1. 输入清理（保持原样） =====
    rewards = torch.nan_to_num(rewards, nan=0.0, posinf=10.0, neginf=-10.0)
    current_poses_cls = torch.nan_to_num(current_poses_cls, nan=0.0)
    ref_poses_cls = torch.nan_to_num(ref_poses_cls, nan=0.0)
    
    # ===== 2. 安全计算优势函数（更合理的范围） =====
    # PDM奖励通常在[-1, 1]或[0, 1]范围
    # 使用更宽松的裁剪，只处理极端值
    rewards = torch.clamp(rewards, -2.0, 2.0)  # 宽松裁剪
    
    mean_r = rewards.mean(dim=1, keepdim=True)
    std_r = rewards.std(dim=1, keepdim=True)
    
    # 更合理的std范围
    min_std = 0.1  # 避免除零，同时保持数值稳定
    max_std = 2.0  # 允许一定的奖励方差
    
    std_r = torch.where(std_r < min_std, torch.ones_like(std_r) * min_std, std_r)
    std_r = torch.clamp(std_r, min=min_std, max=max_std)
    
    advantages = (rewards - mean_r) / (std_r + 1e-8)
    
    # advantages的合理范围：根据经验，3-5个标准差是合理的
    advantages = torch.clamp(advantages, -4.0, 4.0)
    
    # ===== 3. 使用原分位数裁剪（更稳定） =====
    advantages_flat = advantages.view(-1).detach()
    
    advantages_flat = advantages_flat.float()
    # 2. 清理可能的NaN/inf（虽然前面清理过，再加一层保险）
    advantages_flat = torch.nan_to_num(advantages_flat, nan=0.0)
    
    # 检查是否有足够的数据计算分位数
    if advantages_flat.numel() > 10:  # 至少有10个值
        try:
            adv_min = torch.quantile(advantages_flat, clip_advantage_lower_quantile)
            adv_max = torch.quantile(advantages_flat, clip_advantage_upper_quantile)
            advantages_clipped = advantages.clamp(min=adv_min, max=adv_max)
        except:
            # 分位数计算失败，使用原始advantages
            advantages_clipped = advantages
    else:
        advantages_clipped = advantages
    
    # ===== 4. 安全计算对数概率比 =====
    current_probs = F.softmax(current_poses_cls, dim=-1)
    ref_probs = F.softmax(ref_poses_cls, dim=-1)
    
    # 使用更合理的eps，避免影响正常训练
    eps = 1e-6  # 1e-6对softmax输出是安全的
    current_probs = current_probs.clamp(min=eps, max=1.0)
    ref_probs = ref_probs.clamp(min=eps, max=1.0)
    
    log_ratios = torch.log(current_probs) - torch.log(ref_probs)
    
    # PPO裁剪：保持你的clip_ratio参数
    if clip_ratio > 0:
        log_ratios = torch.clamp(log_ratios, -clip_ratio, clip_ratio)
    
    # ===== 5. 计算损失 =====
    # 使用advantages_clipped而不是原始advantages
    policy_loss = -torch.mean(log_ratios * advantages_clipped)
    
    # 检查loss是否合理
    max_loss = 10.0  # 合理的最大损失值
    if torch.abs(policy_loss) > max_loss:
        policy_loss = torch.clamp(policy_loss, -max_loss, max_loss)
    
    if torch.isnan(policy_loss) or torch.isinf(policy_loss):
        policy_loss = torch.tensor(0.0, device=current_poses_cls.device)
    
    return policy_loss

def compute_grpo_loss3(
    current_poses_cls: torch.Tensor,
    ref_poses_cls: torch.Tensor,
    rewards: torch.Tensor,
    num_modes: int,
    clip_advantage_lower_quantile: float = 0.0,
    clip_advantage_upper_quantile: float = 1.0,
    clip_ratio: float = 0.2
) -> torch.Tensor:
    """
    GRPO loss with PPO-style probability ratio clipping.
    """
    rewards = torch.nan_to_num(rewards, nan=0.0).float()
    current_poses_cls = torch.nan_to_num(current_poses_cls, nan=0.0).float()
    ref_poses_cls = torch.nan_to_num(ref_poses_cls, nan=0.0).float()

    mean_r = rewards.mean(dim=1, keepdim=True)
    std_r = rewards.std(dim=1, keepdim=True).clamp(min=0.01)
    advantages = ((rewards - mean_r) / std_r).detach()

    current_log_probs = F.log_softmax(current_poses_cls, dim=-1)
    ref_log_probs = F.log_softmax(ref_poses_cls.detach(), dim=-1)

    log_ratios = current_log_probs - ref_log_probs
    ratios = torch.exp(log_ratios)

    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -torch.mean(torch.min(surr1, surr2))

    return policy_loss
