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
from navsim.agents.diffusiondrive.stage23_trajectory_selector import (
    compute_stage23_selector_loss,
)
from navsim.agents.diffusiondrive.stage24_safety_value_selector import (
    compute_stage24_selector_loss,
)
from navsim.agents.diffusiondrive.stage25_relative_harm_selector import (
    compute_stage25_selector_loss,
)
from navsim.agents.diffusiondrive.paired_advantage_risk import (
    compute_paired_advantage_risk_loss,
)
from navsim.agents.diffusiondrive.stage34_loss_bridge import (
    compute_stage34_transfuser_loss,
)
from navsim.agents.diffusiondrive.stage35_loss_bridge import (
    compute_stage35_transfuser_loss,
)
from navsim.agents.diffusiondrive.stage36_loss_bridge import (
    compute_stage36_transfuser_loss,
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


def compute_selected_set_diffgrpo_objective(
    current_log_probs: torch.Tensor,
    bc_log_probs: torch.Tensor,
    reference_mean_kl: torch.Tensor,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    component_scores: torch.Tensor,
    base_rewards: torch.Tensor,
    base_valid_mask: torch.Tensor,
    base_component_scores: torch.Tensor,
    advantage_eps: float = 1e-3,
    safety_tolerance: float = 1e-6,
    step_discount: float = 0.6,
    bc_weight: float = 0.1,
    regular_kl_weight: float = 0.1,
    mature_kl_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Deployment-aligned Stage23 objective on eight selector-chosen chains."""
    if current_log_probs.ndim != 3:
        raise ValueError("Stage23 log probabilities must have shape [B,8,S]")
    batch, group, steps = current_log_probs.shape
    if group != 8:
        raise ValueError("Stage23 requires exactly eight selected sets")
    expected = (batch, group)
    if rewards.shape != expected or valid_mask.shape != expected:
        raise ValueError("Stage23 selected rewards must have shape [B,8]")
    if base_rewards.shape != expected or base_valid_mask.shape != expected:
        raise ValueError("Stage23 base guard rewards must have shape [B,8]")
    if component_scores.shape != (batch, group, 6):
        raise ValueError("Stage23 selected components must have shape [B,8,6]")
    if base_component_scores.shape != component_scores.shape:
        raise ValueError("Stage23 current/base component shapes differ")
    if bc_log_probs.shape != current_log_probs.shape:
        raise ValueError("Stage23 BC must contain one base chain per selected set")
    if reference_mean_kl.shape != current_log_probs.shape:
        raise ValueError("Stage23 exact KL shape must match selected log probabilities")
    constants = (
        advantage_eps, safety_tolerance, step_discount, bc_weight,
        regular_kl_weight, mature_kl_weight,
    )
    if not all(math.isfinite(float(value)) for value in constants):
        raise ValueError("Stage23 objective constants must be finite")
    if (
        advantage_eps <= 0 or safety_tolerance < 0
        or not 0 < step_discount <= 1 or bc_weight != 0.1
        or regular_kl_weight != 0.1 or mature_kl_weight != 0.5
    ):
        raise ValueError("Stage23 objective constants drifted from preregistration")
    if not (
        torch.isfinite(current_log_probs).all()
        and torch.isfinite(bc_log_probs).all()
        and torch.isfinite(reference_mean_kl).all()
    ):
        raise FloatingPointError("Stage23 policy/BC/KL tensors must be finite")

    pair_valid = (
        valid_mask.bool() & base_valid_mask.bool()
        & torch.isfinite(rewards) & torch.isfinite(base_rewards)
        & torch.isfinite(component_scores).all(dim=-1)
        & torch.isfinite(base_component_scores).all(dim=-1)
    )
    advantages, _, group_valid, reward_mean, reward_std = (
        compute_group_relative_advantages(rewards, pair_valid, advantage_eps)
    )
    delta = rewards.float() - base_rewards.float()
    safety_indices = torch.tensor((0, 1, 3), device=rewards.device)
    safety_regression = (
        component_scores.float().index_select(-1, safety_indices)
        < base_component_scores.float().index_select(-1, safety_indices)
        - float(safety_tolerance)
    ).any(dim=-1) & pair_valid
    base_negative = (delta < -0.01) & pair_valid
    mature = (base_rewards.float() >= 0.75) & pair_valid
    mature_negative = mature & (delta < -0.002)
    hard_guard = safety_regression | base_negative | mature_negative
    advantages = torch.where(hard_guard, torch.full_like(advantages, -2.0), advantages)
    optimize = pair_valid & group_valid.unsqueeze(-1)
    advantages = torch.where(optimize, advantages.clamp(-2.0, 2.0), 0.0).detach()
    if not torch.isfinite(advantages).all():
        raise FloatingPointError("Stage23 group advantages must be finite")

    indices = torch.arange(steps, device=rewards.device, dtype=current_log_probs.dtype)
    discounts = step_discount ** (steps - indices - 1)
    mask = optimize.unsqueeze(-1).expand_as(current_log_probs)
    policy_terms = -current_log_probs * advantages.unsqueeze(-1) * discounts.view(1, 1, -1)
    policy_loss = policy_terms[mask].mean() if mask.any() else current_log_probs.sum() * 0.0
    bc_loss = -bc_weight * (
        bc_log_probs * discounts.view(1, 1, -1)
    )[mask].mean() if mask.any() else bc_log_probs.sum() * 0.0
    kl_coefficients = torch.where(
        mature_negative,
        reference_mean_kl.new_tensor(mature_kl_weight),
        reference_mean_kl.new_tensor(regular_kl_weight),
    )
    kl_terms = reference_mean_kl * kl_coefficients.unsqueeze(-1)
    kl_loss = kl_terms[mask].mean() if mask.any() else reference_mean_kl.sum() * 0.0
    total = policy_loss + bc_loss + kl_loss
    valid_total = pair_valid.float().sum().clamp_min(1.0)
    return {
        "loss": total,
        "policy_loss": policy_loss,
        "bc_loss": bc_loss,
        "reference_kl_loss": kl_loss,
        "advantages": advantages,
        "advantage_mean": (
            advantages[optimize].mean().detach()
            if optimize.any() else advantages.sum() * 0.0
        ),
        "mean_current_log_prob": (
            current_log_probs[mask].mean().detach()
            if mask.any() else current_log_probs.sum() * 0.0
        ),
        "reward_mean": reward_mean.mean().detach(),
        "reward_std": reward_std.mean().detach(),
        "base_delta_mean": delta[pair_valid].mean().detach() if pair_valid.any() else delta.sum() * 0.0,
        "safety_override_fraction": (safety_regression.float().sum() / valid_total).detach(),
        "base_negative_fraction": (base_negative.float().sum() / valid_total).detach(),
        "mature_negative_fraction": (mature_negative.float().sum() / valid_total).detach(),
        "catastrophic_fraction": (((delta <= -0.5) & pair_valid).float().sum() / valid_total).detach(),
        "mean_exact_kl": reference_mean_kl[mask].mean().detach() if mask.any() else reference_mean_kl.sum() * 0.0,
        "discount_first": discounts[0].detach(),
        "discount_last": discounts[-1].detach(),
    }


def compute_public_paired_uplift_objective(
    current_log_probs: torch.Tensor,
    bc_log_probs: torch.Tensor,
    reference_mean_kl: torch.Tensor,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    component_scores: torch.Tensor,
    base_rewards: torch.Tensor,
    base_valid_mask: torch.Tensor,
    base_component_scores: torch.Tensor,
    advantage_eps: float = 1e-3,
    positive_margin: float = 0.002,
    negative_margin: float = 0.002,
    mature_negative_margin: float = 0.0005,
    mature_reward_threshold: float = 0.75,
    evidence_scale: float = 0.1,
    advantage_clip: float = 2.0,
    safety_tolerance: float = 1e-6,
    step_discount: float = 0.6,
    bc_weight: float = 0.1,
    regular_kl_weight: float = 0.1,
    mature_kl_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Absolute paired-uplift objective for a strong public reference.

    A rollout receives positive policy advantage only when it beats its
    common-random reference rollout by the preregistered absolute margin.
    Relative ranking inside an all-worse group can never create a positive
    policy update.
    """
    if current_log_probs.ndim != 3:
        raise ValueError("Stage28 log probabilities must have shape [B,8,S]")
    batch, group, steps = current_log_probs.shape
    if group != 8:
        raise ValueError("Stage28 requires exactly eight selected sets")
    expected = (batch, group)
    for name, tensor in (
        ("rewards", rewards),
        ("valid_mask", valid_mask),
        ("base_rewards", base_rewards),
        ("base_valid_mask", base_valid_mask),
    ):
        if tensor.shape != expected:
            raise ValueError(f"Stage28 {name} must have shape [B,8]")
    if component_scores.shape != (batch, group, 6):
        raise ValueError("Stage28 current components must have shape [B,8,6]")
    if base_component_scores.shape != component_scores.shape:
        raise ValueError("Stage28 current/base component shapes differ")
    if bc_log_probs.shape != current_log_probs.shape:
        raise ValueError("Stage28 BC log probabilities must match [B,8,S]")
    if reference_mean_kl.shape != current_log_probs.shape:
        raise ValueError("Stage28 exact KL must match [B,8,S]")

    constants = (
        advantage_eps,
        positive_margin,
        negative_margin,
        mature_negative_margin,
        mature_reward_threshold,
        evidence_scale,
        advantage_clip,
        safety_tolerance,
        step_discount,
        bc_weight,
        regular_kl_weight,
        mature_kl_weight,
    )
    if not all(math.isfinite(float(value)) for value in constants):
        raise ValueError("Stage28 objective constants must be finite")
    if (
        advantage_eps <= 0
        or min(positive_margin, negative_margin, mature_negative_margin) <= 0
        or not 0 <= mature_reward_threshold <= 1
        or evidence_scale <= 0
        or advantage_clip <= 0
        or safety_tolerance < 0
        or not 0 < step_discount <= 1
        or bc_weight < 0
        or regular_kl_weight < 0
        or mature_kl_weight < regular_kl_weight
    ):
        raise ValueError("invalid Stage28 objective constants")
    if not (
        torch.isfinite(current_log_probs).all()
        and torch.isfinite(bc_log_probs).all()
        and torch.isfinite(reference_mean_kl).all()
    ):
        raise FloatingPointError("Stage28 policy/BC/KL tensors must be finite")
    if (reference_mean_kl < 0).any():
        raise FloatingPointError("Stage28 exact KL must be non-negative")

    pair_valid = (
        valid_mask.bool()
        & base_valid_mask.bool()
        & torch.isfinite(rewards)
        & torch.isfinite(base_rewards)
        & torch.isfinite(component_scores).all(dim=-1)
        & torch.isfinite(base_component_scores).all(dim=-1)
    )
    valid_count = pair_valid.sum(dim=-1)
    group_valid = valid_count >= 2
    delta = torch.where(
        pair_valid,
        rewards.float() - base_rewards.float(),
        torch.zeros_like(rewards.float()),
    )
    delta_mean = delta.sum(dim=-1) / valid_count.clamp_min(1).float()
    centered = torch.where(
        pair_valid,
        delta - delta_mean.unsqueeze(-1),
        torch.zeros_like(delta),
    )
    delta_std = torch.sqrt(
        centered.square().sum(dim=-1) / valid_count.clamp_min(1).float()
    )
    delta_z = torch.where(
        pair_valid & (delta_std > advantage_eps).unsqueeze(-1),
        centered / delta_std.clamp_min(advantage_eps).unsqueeze(-1),
        torch.zeros_like(centered),
    )
    evidence = (delta.abs() / float(evidence_scale)).clamp(
        max=float(advantage_clip)
    )

    safety_indices = torch.tensor((0, 1, 3), device=rewards.device)
    safety_regression = pair_valid & (
        component_scores.float().index_select(-1, safety_indices)
        < base_component_scores.float().index_select(-1, safety_indices)
        - float(safety_tolerance)
    ).any(dim=-1)
    mature = pair_valid & (
        base_rewards.float() >= float(mature_reward_threshold)
    )
    positive = pair_valid & (delta > float(positive_margin))
    regular_negative = pair_valid & (delta < -float(negative_margin))
    mature_negative = mature & (delta < -float(mature_negative_margin))
    negative = regular_negative | mature_negative

    positive_value = (
        0.5 * delta_z.clamp_min(0.0) + 0.5 * evidence
    ).clamp(max=float(advantage_clip))
    negative_value = -(
        0.5 * (-delta_z).clamp_min(0.0) + 0.5 * evidence
    ).clamp(max=float(advantage_clip))
    advantages = torch.zeros_like(delta)
    advantages = torch.where(positive, positive_value, advantages)
    advantages = torch.where(negative, negative_value, advantages)
    advantages = torch.where(
        safety_regression,
        torch.full_like(advantages, -float(advantage_clip)),
        advantages,
    )
    optimize = pair_valid & group_valid.unsqueeze(-1)
    advantages = torch.where(optimize, advantages, 0.0).detach()
    if not torch.isfinite(advantages).all():
        raise FloatingPointError("Stage28 advantages must be finite")

    indices = torch.arange(
        steps, device=rewards.device, dtype=current_log_probs.dtype
    )
    discounts = step_discount ** (steps - indices - 1)
    all_mask = optimize.unsqueeze(-1).expand_as(current_log_probs)
    active = optimize & (advantages != 0)
    policy_mask = active.unsqueeze(-1).expand_as(current_log_probs)
    policy_terms = (
        -current_log_probs
        * advantages.unsqueeze(-1)
        * discounts.view(1, 1, -1)
    )
    policy_loss = (
        policy_terms[policy_mask].mean()
        if policy_mask.any()
        else current_log_probs.sum() * 0.0
    )
    bc_loss = (
        -float(bc_weight)
        * (bc_log_probs * discounts.view(1, 1, -1))[all_mask].mean()
        if all_mask.any()
        else bc_log_probs.sum() * 0.0
    )
    kl_coefficients = torch.where(
        mature_negative | safety_regression,
        reference_mean_kl.new_tensor(float(mature_kl_weight)),
        reference_mean_kl.new_tensor(float(regular_kl_weight)),
    )
    kl_terms = reference_mean_kl * kl_coefficients.unsqueeze(-1)
    kl_loss = (
        kl_terms[all_mask].mean()
        if all_mask.any()
        else reference_mean_kl.sum() * 0.0
    )
    total = policy_loss + bc_loss + kl_loss
    valid_total = pair_valid.float().sum().clamp_min(1.0)
    neutral = pair_valid & ~positive & ~negative & ~safety_regression
    current_reward_mean = (
        torch.where(
            pair_valid, rewards.float(), torch.zeros_like(rewards.float())
        ).sum(dim=-1)
        / valid_count.clamp_min(1).float()
    )
    current_centered = torch.where(
        pair_valid,
        rewards.float() - current_reward_mean.unsqueeze(-1),
        torch.zeros_like(rewards.float()),
    )
    current_reward_std = torch.sqrt(
        current_centered.square().sum(dim=-1)
        / valid_count.clamp_min(1).float()
    )
    return {
        "loss": total,
        "policy_loss": policy_loss,
        "bc_loss": bc_loss,
        "reference_kl_loss": kl_loss,
        "advantages": advantages,
        "advantage_mean": (
            advantages[optimize].mean().detach()
            if optimize.any()
            else advantages.sum() * 0.0
        ),
        "reward_mean": (
            current_reward_mean[group_valid].mean().detach()
            if group_valid.any()
            else rewards.sum() * 0.0
        ),
        "reward_std": (
            current_reward_std[group_valid].mean().detach()
            if group_valid.any()
            else rewards.sum() * 0.0
        ),
        "base_delta_mean": (
            delta[pair_valid].mean().detach()
            if pair_valid.any()
            else delta.sum() * 0.0
        ),
        "positive_fraction": (positive.float().sum() / valid_total).detach(),
        "regular_negative_fraction": (
            regular_negative.float().sum() / valid_total
        ).detach(),
        "mature_negative_fraction": (
            mature_negative.float().sum() / valid_total
        ).detach(),
        "safety_override_fraction": (
            safety_regression.float().sum() / valid_total
        ).detach(),
        "neutral_fraction": (neutral.float().sum() / valid_total).detach(),
        "policy_active_scene_fraction": (
            active.any(dim=-1).float().mean()
        ).detach(),
        "catastrophic_fraction": (
            ((delta <= -0.5) & pair_valid).float().sum() / valid_total
        ).detach(),
        "mean_exact_kl": (
            reference_mean_kl[all_mask].mean().detach()
            if all_mask.any()
            else reference_mean_kl.sum() * 0.0
        ),
        "mean_current_log_prob": (
            current_log_probs[all_mask].mean().detach()
            if all_mask.any()
            else current_log_probs.sum() * 0.0
        ),
        "discount_first": discounts[0].detach(),
        "discount_last": discounts[-1].detach(),
    }


def compute_reference_anchored_headroom_objective(
    current_log_probs: torch.Tensor,
    bc_log_probs: torch.Tensor,
    reference_mean_kl: torch.Tensor,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    component_scores: torch.Tensor,
    base_rewards: torch.Tensor,
    base_valid_mask: torch.Tensor,
    base_component_scores: torch.Tensor,
    advantage_eps: float = 1e-3,
    headroom_low: float = 0.75,
    headroom_high: float = 0.90,
    delta_scale_floor: float = 0.002,
    rank_weight: float = 0.5,
    advantage_clip: float = 2.0,
    safety_tolerance: float = 1e-6,
    step_discount: float = 0.6,
    conditional_regularization: bool = False,
    fixed_bc_weight: float = 0.1,
    fixed_kl_weight: float = 0.1,
    hard_bc_weight: float = 0.05,
    mature_bc_weight: float = 0.1,
    hard_kl_weight: float = 0.05,
    mature_kl_weight: float = 0.5,
    safety_kl_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Stage29 dense GRPO with public-reference sign and headroom anchoring."""
    if current_log_probs.ndim != 3:
        raise ValueError("Stage29 log probabilities must have shape [B,8,S]")
    batch, group, steps = current_log_probs.shape
    if group != 8:
        raise ValueError("Stage29 requires exactly eight selected sets")
    expected = (batch, group)
    for name, tensor in (
        ("rewards", rewards),
        ("valid_mask", valid_mask),
        ("base_rewards", base_rewards),
        ("base_valid_mask", base_valid_mask),
    ):
        if tensor.shape != expected:
            raise ValueError(f"Stage29 {name} must have shape [B,8]")
    if component_scores.shape != (batch, group, 6):
        raise ValueError("Stage29 current components must have shape [B,8,6]")
    if base_component_scores.shape != component_scores.shape:
        raise ValueError("Stage29 current/base component shapes differ")
    if bc_log_probs.shape != current_log_probs.shape:
        raise ValueError("Stage29 BC log probabilities must match [B,8,S]")
    if reference_mean_kl.shape != current_log_probs.shape:
        raise ValueError("Stage29 exact KL must match [B,8,S]")

    constants = (
        advantage_eps,
        headroom_low,
        headroom_high,
        delta_scale_floor,
        rank_weight,
        advantage_clip,
        safety_tolerance,
        step_discount,
        fixed_bc_weight,
        fixed_kl_weight,
        hard_bc_weight,
        mature_bc_weight,
        hard_kl_weight,
        mature_kl_weight,
        safety_kl_weight,
    )
    if not all(math.isfinite(float(value)) for value in constants):
        raise ValueError("Stage29 objective constants must be finite")
    if (
        advantage_eps <= 0
        or not 0 <= headroom_low < headroom_high <= 1
        or delta_scale_floor <= 0
        or rank_weight < 0
        or advantage_clip <= 0
        or safety_tolerance < 0
        or not 0 < step_discount <= 1
        or min(
            fixed_bc_weight,
            fixed_kl_weight,
            hard_bc_weight,
            mature_bc_weight,
            hard_kl_weight,
            mature_kl_weight,
            safety_kl_weight,
        ) < 0
        or hard_bc_weight > mature_bc_weight
        or hard_kl_weight > mature_kl_weight
        or safety_kl_weight < mature_kl_weight
    ):
        raise ValueError("invalid Stage29 objective constants")
    if not (
        torch.isfinite(current_log_probs).all()
        and torch.isfinite(bc_log_probs).all()
        and torch.isfinite(reference_mean_kl).all()
    ):
        raise FloatingPointError("Stage29 policy/BC/KL tensors must be finite")
    if (reference_mean_kl < 0).any():
        raise FloatingPointError("Stage29 exact KL must be non-negative")

    pair_valid = (
        valid_mask.bool()
        & base_valid_mask.bool()
        & torch.isfinite(rewards)
        & torch.isfinite(base_rewards)
        & torch.isfinite(component_scores).all(dim=-1)
        & torch.isfinite(base_component_scores).all(dim=-1)
    )
    valid_count = pair_valid.sum(dim=-1)
    group_valid = valid_count >= 2
    valid_denominator = valid_count.clamp_min(1).float()

    current = torch.where(
        pair_valid, rewards.float(), torch.zeros_like(rewards.float())
    )
    reference = torch.where(
        pair_valid, base_rewards.float(), torch.zeros_like(base_rewards.float())
    )
    reward_mean = current.sum(dim=-1) / valid_denominator
    reference_quality = reference.sum(dim=-1) / valid_denominator
    reward_centered = torch.where(
        pair_valid,
        rewards.float() - reward_mean.unsqueeze(-1),
        torch.zeros_like(rewards.float()),
    )
    reward_std = torch.sqrt(
        reward_centered.square().sum(dim=-1) / valid_denominator
    )
    reward_z = torch.where(
        pair_valid & (reward_std > advantage_eps).unsqueeze(-1),
        reward_centered / reward_std.clamp_min(advantage_eps).unsqueeze(-1),
        torch.zeros_like(reward_centered),
    ).clamp(min=-float(advantage_clip), max=float(advantage_clip))

    delta = torch.where(
        pair_valid,
        rewards.float() - base_rewards.float(),
        torch.zeros_like(rewards.float()),
    )
    delta_mean = delta.sum(dim=-1) / valid_denominator
    delta_centered = torch.where(
        pair_valid,
        delta - delta_mean.unsqueeze(-1),
        torch.zeros_like(delta),
    )
    delta_std = torch.sqrt(
        delta_centered.square().sum(dim=-1) / valid_denominator
    )
    paired_scale = delta_std.clamp_min(float(delta_scale_floor))
    paired = (
        delta / paired_scale.unsqueeze(-1)
    ).clamp(min=-float(advantage_clip), max=float(advantage_clip))

    headroom = (
        (float(headroom_high) - reference_quality)
        / float(headroom_high - headroom_low)
    ).clamp(min=0.0, max=1.0)
    rank_term = float(rank_weight) * headroom.unsqueeze(-1) * reward_z
    raw_advantages = paired + rank_term
    advantages = torch.where(
        delta > 0,
        raw_advantages.clamp_min(0.0),
        torch.where(
            delta < 0,
            raw_advantages.clamp_max(0.0),
            rank_term,
        ),
    )

    safety_indices = torch.tensor((0, 1, 3), device=rewards.device)
    safety_regression = pair_valid & (
        component_scores.float().index_select(-1, safety_indices)
        < base_component_scores.float().index_select(-1, safety_indices)
        - float(safety_tolerance)
    ).any(dim=-1)
    advantages = torch.where(
        safety_regression,
        torch.full_like(advantages, -float(advantage_clip)),
        advantages,
    )
    optimize = pair_valid & group_valid.unsqueeze(-1)
    advantages = torch.where(
        optimize,
        advantages.clamp(min=-float(advantage_clip), max=float(advantage_clip)),
        torch.zeros_like(advantages),
    ).detach()
    if not torch.isfinite(advantages).all():
        raise FloatingPointError("Stage29 advantages must be finite")

    indices = torch.arange(
        steps, device=rewards.device, dtype=current_log_probs.dtype
    )
    discounts = step_discount ** (steps - indices - 1)
    active = optimize & (advantages != 0)
    policy_mask = active.unsqueeze(-1).expand_as(current_log_probs)
    policy_terms = (
        -current_log_probs
        * advantages.unsqueeze(-1)
        * discounts.view(1, 1, -1)
    )
    policy_count = policy_mask.sum(dim=(1, 2)).clamp_min(1).float()
    policy_per_scene = torch.where(
        active.any(dim=-1),
        (policy_terms * policy_mask).sum(dim=(1, 2)) / policy_count,
        torch.zeros_like(policy_count),
    )
    active_scenes = active.any(dim=-1)
    policy_loss = (
        policy_per_scene[active_scenes].mean()
        if active_scenes.any()
        else current_log_probs.sum() * 0.0
    )

    if conditional_regularization:
        bc_coefficients = float(hard_bc_weight) + (
            float(mature_bc_weight - hard_bc_weight) * (1.0 - headroom)
        )
        scene_kl_coefficients = float(hard_kl_weight) + (
            float(mature_kl_weight - hard_kl_weight) * (1.0 - headroom)
        )
    else:
        bc_coefficients = headroom.new_full(
            headroom.shape, float(fixed_bc_weight)
        )
        scene_kl_coefficients = headroom.new_full(
            headroom.shape, float(fixed_kl_weight)
        )
    all_mask = optimize.unsqueeze(-1).expand_as(current_log_probs)
    all_count = all_mask.sum(dim=(1, 2)).clamp_min(1).float()
    bc_terms = (
        -bc_log_probs
        * discounts.view(1, 1, -1)
        * bc_coefficients.view(-1, 1, 1)
    )
    bc_per_scene = (bc_terms * all_mask).sum(dim=(1, 2)) / all_count
    bc_loss = (
        bc_per_scene[group_valid].mean()
        if group_valid.any()
        else bc_log_probs.sum() * 0.0
    )
    kl_coefficients = scene_kl_coefficients.unsqueeze(-1).expand_as(delta)
    kl_coefficients = torch.where(
        safety_regression,
        kl_coefficients.new_full(kl_coefficients.shape, float(safety_kl_weight)),
        kl_coefficients,
    )
    kl_terms = reference_mean_kl * kl_coefficients.unsqueeze(-1)
    kl_per_scene = (kl_terms * all_mask).sum(dim=(1, 2)) / all_count
    kl_loss = (
        kl_per_scene[group_valid].mean()
        if group_valid.any()
        else reference_mean_kl.sum() * 0.0
    )
    total = policy_loss + bc_loss + kl_loss

    hard_scene = group_valid & (reference_quality <= float(headroom_low))
    mature_scene = group_valid & (reference_quality >= float(headroom_high))
    transition_scene = group_valid & ~hard_scene & ~mature_scene

    def bucket_delta_mean(scene_mask: torch.Tensor) -> torch.Tensor:
        mask = pair_valid & scene_mask.unsqueeze(-1)
        return delta[mask].mean().detach() if mask.any() else delta.sum() * 0.0

    valid_total = pair_valid.float().sum().clamp_min(1.0)
    neutral = optimize & (advantages == 0)
    return {
        "loss": total,
        "policy_loss": policy_loss,
        "bc_loss": bc_loss,
        "reference_kl_loss": kl_loss,
        "advantages": advantages,
        "advantage_mean": (
            advantages[optimize].mean().detach()
            if optimize.any() else advantages.sum() * 0.0
        ),
        "reward_mean": (
            reward_mean[group_valid].mean().detach()
            if group_valid.any() else rewards.sum() * 0.0
        ),
        "reward_std": (
            reward_std[group_valid].mean().detach()
            if group_valid.any() else rewards.sum() * 0.0
        ),
        "base_delta_mean": (
            delta[pair_valid].mean().detach()
            if pair_valid.any() else delta.sum() * 0.0
        ),
        "hard_delta_mean": bucket_delta_mean(hard_scene),
        "transition_delta_mean": bucket_delta_mean(transition_scene),
        "mature_delta_mean": bucket_delta_mean(mature_scene),
        "headroom_mean": (
            headroom[group_valid].mean().detach()
            if group_valid.any() else headroom.sum() * 0.0
        ),
        "hard_scene_fraction": hard_scene.float().mean().detach(),
        "transition_scene_fraction": transition_scene.float().mean().detach(),
        "mature_scene_fraction": mature_scene.float().mean().detach(),
        "positive_fraction": (
            ((advantages > 0) & optimize).float().sum() / valid_total
        ).detach(),
        "negative_fraction": (
            ((advantages < 0) & optimize).float().sum() / valid_total
        ).detach(),
        "neutral_fraction": (neutral.float().sum() / valid_total).detach(),
        "policy_active_scene_fraction": active_scenes.float().mean().detach(),
        "safety_override_fraction": (
            safety_regression.float().sum() / valid_total
        ).detach(),
        "catastrophic_fraction": (
            ((delta <= -0.5) & pair_valid).float().sum() / valid_total
        ).detach(),
        "mean_bc_weight": (
            bc_coefficients[group_valid].mean().detach()
            if group_valid.any() else bc_coefficients.sum() * 0.0
        ),
        "mean_kl_weight": (
            kl_coefficients[optimize].mean().detach()
            if optimize.any() else kl_coefficients.sum() * 0.0
        ),
        "mean_exact_kl": (
            reference_mean_kl[all_mask].mean().detach()
            if all_mask.any() else reference_mean_kl.sum() * 0.0
        ),
        "mean_current_log_prob": (
            current_log_probs[all_mask].mean().detach()
            if all_mask.any() else current_log_probs.sum() * 0.0
        ),
        "paired_scale_mean": (
            paired_scale[group_valid].mean().detach()
            if group_valid.any() else paired_scale.sum() * 0.0
        ),
        "discount_first": discounts[0].detach(),
        "discount_last": discounts[-1].detach(),
    }


def compute_deployed_decision_grpo_objective(
    current_log_probs: torch.Tensor,
    bc_log_probs: torch.Tensor,
    reference_mean_kl: torch.Tensor,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    component_scores: torch.Tensor,
    base_rewards: torch.Tensor,
    base_valid_mask: torch.Tensor,
    base_component_scores: torch.Tensor,
    use_frontier_rank: bool,
    advantage_eps: float = 1e-3,
    headroom_low: float = 0.75,
    headroom_high: float = 0.90,
    delta_scale_floor: float = 0.002,
    rank_weight: float = 0.5,
    advantage_clip: float = 2.0,
    safety_tolerance: float = 1e-6,
    step_discount: float = 0.6,
    bc_weight: float = 0.1,
    kl_weight: float = 0.1,
    safety_kl_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Stage31 GRPO on independently selected current/public deployments."""
    if current_log_probs.ndim != 3:
        raise ValueError("Stage31 log probabilities must have shape [B,8,S]")
    batch, group, steps = current_log_probs.shape
    if group != 8:
        raise ValueError("Stage31 requires exactly eight deployed decisions")
    expected = (batch, group)
    for name, tensor in (
        ("rewards", rewards),
        ("valid_mask", valid_mask),
        ("base_rewards", base_rewards),
        ("base_valid_mask", base_valid_mask),
    ):
        if tensor.shape != expected:
            raise ValueError(f"Stage31 {name} must have shape [B,8]")
    if component_scores.shape != (batch, group, 6):
        raise ValueError("Stage31 current components must have shape [B,8,6]")
    if base_component_scores.shape != component_scores.shape:
        raise ValueError("Stage31 current/public component shapes differ")
    if bc_log_probs.shape != current_log_probs.shape:
        raise ValueError("Stage31 BC log probabilities must match [B,8,S]")
    if reference_mean_kl.shape != current_log_probs.shape:
        raise ValueError("Stage31 exact KL must match [B,8,S]")

    constants = (
        advantage_eps,
        headroom_low,
        headroom_high,
        delta_scale_floor,
        rank_weight,
        advantage_clip,
        safety_tolerance,
        step_discount,
        bc_weight,
        kl_weight,
        safety_kl_weight,
    )
    if not all(math.isfinite(float(value)) for value in constants):
        raise ValueError("Stage31 objective constants must be finite")
    if (
        advantage_eps <= 0
        or not 0 <= headroom_low < headroom_high <= 1
        or delta_scale_floor <= 0
        or rank_weight < 0
        or advantage_clip <= 0
        or safety_tolerance < 0
        or not 0 < step_discount <= 1
        or min(bc_weight, kl_weight, safety_kl_weight) < 0
        or safety_kl_weight < kl_weight
    ):
        raise ValueError("invalid Stage31 objective constants")
    if not (
        torch.isfinite(current_log_probs).all()
        and torch.isfinite(bc_log_probs).all()
        and torch.isfinite(reference_mean_kl).all()
    ):
        raise FloatingPointError("Stage31 policy/BC/KL tensors must be finite")
    if (reference_mean_kl < 0).any():
        raise FloatingPointError("Stage31 exact KL must be non-negative")

    pair_valid = (
        valid_mask.bool()
        & base_valid_mask.bool()
        & torch.isfinite(rewards)
        & torch.isfinite(base_rewards)
        & torch.isfinite(component_scores).all(dim=-1)
        & torch.isfinite(base_component_scores).all(dim=-1)
    )
    valid_count = pair_valid.sum(dim=-1)
    group_valid = valid_count >= 2
    denominator = valid_count.clamp_min(1).float()

    current = torch.where(
        pair_valid, rewards.float(), torch.zeros_like(rewards.float())
    )
    public = torch.where(
        pair_valid, base_rewards.float(), torch.zeros_like(base_rewards.float())
    )
    reward_mean = current.sum(dim=-1) / denominator
    public_quality = public.sum(dim=-1) / denominator
    reward_centered = torch.where(
        pair_valid,
        rewards.float() - reward_mean.unsqueeze(-1),
        torch.zeros_like(rewards.float()),
    )
    reward_std = torch.sqrt(
        reward_centered.square().sum(dim=-1) / denominator
    )
    reward_z = torch.where(
        pair_valid & (reward_std > advantage_eps).unsqueeze(-1),
        reward_centered / reward_std.clamp_min(advantage_eps).unsqueeze(-1),
        torch.zeros_like(reward_centered),
    ).clamp(min=-float(advantage_clip), max=float(advantage_clip))

    delta = torch.where(
        pair_valid,
        rewards.float() - base_rewards.float(),
        torch.zeros_like(rewards.float()),
    )
    delta_mean = delta.sum(dim=-1) / denominator
    delta_centered = torch.where(
        pair_valid,
        delta - delta_mean.unsqueeze(-1),
        torch.zeros_like(delta),
    )
    delta_std = torch.sqrt(
        delta_centered.square().sum(dim=-1) / denominator
    )
    delta_scale = delta_std.clamp_min(float(delta_scale_floor))
    delta_z = (
        delta_centered / delta_scale.unsqueeze(-1)
    ).clamp(min=-float(advantage_clip), max=float(advantage_clip))

    headroom = (
        (float(headroom_high) - public_quality)
        / float(headroom_high - headroom_low)
    ).clamp(min=0.0, max=1.0)
    if use_frontier_rank:
        rank_term = float(rank_weight) * headroom.unsqueeze(-1) * reward_z
    else:
        rank_term = torch.zeros_like(reward_z)
    raw_advantages = delta_z + rank_term
    advantages = torch.where(
        delta > 0,
        raw_advantages.clamp_min(0.0),
        torch.where(
            delta < 0,
            raw_advantages.clamp_max(0.0),
            rank_term,
        ),
    )

    safety_indices = torch.tensor((0, 1, 3), device=rewards.device)
    safety_regression = pair_valid & (
        component_scores.float().index_select(-1, safety_indices)
        < base_component_scores.float().index_select(-1, safety_indices)
        - float(safety_tolerance)
    ).any(dim=-1)
    advantages = torch.where(
        safety_regression,
        torch.full_like(advantages, -float(advantage_clip)),
        advantages,
    )
    optimize = pair_valid & group_valid.unsqueeze(-1)
    advantages = torch.where(
        optimize,
        advantages.clamp(min=-float(advantage_clip), max=float(advantage_clip)),
        torch.zeros_like(advantages),
    ).detach()
    if not torch.isfinite(advantages).all():
        raise FloatingPointError("Stage31 advantages must be finite")

    indices = torch.arange(
        steps, device=rewards.device, dtype=current_log_probs.dtype
    )
    discounts = step_discount ** (steps - indices - 1)
    active = optimize & (advantages != 0)
    policy_mask = active.unsqueeze(-1).expand_as(current_log_probs)
    policy_terms = (
        -current_log_probs
        * advantages.unsqueeze(-1)
        * discounts.view(1, 1, -1)
    )
    policy_count = policy_mask.sum(dim=(1, 2)).clamp_min(1).float()
    policy_per_scene = torch.where(
        active.any(dim=-1),
        (policy_terms * policy_mask).sum(dim=(1, 2)) / policy_count,
        torch.zeros_like(policy_count),
    )
    active_scenes = active.any(dim=-1)
    policy_loss = (
        policy_per_scene[active_scenes].mean()
        if active_scenes.any()
        else current_log_probs.sum() * 0.0
    )

    all_mask = optimize.unsqueeze(-1).expand_as(current_log_probs)
    all_count = all_mask.sum(dim=(1, 2)).clamp_min(1).float()
    bc_terms = (
        -bc_log_probs
        * discounts.view(1, 1, -1)
        * float(bc_weight)
    )
    bc_per_scene = (bc_terms * all_mask).sum(dim=(1, 2)) / all_count
    bc_loss = (
        bc_per_scene[group_valid].mean()
        if group_valid.any()
        else bc_log_probs.sum() * 0.0
    )
    kl_coefficients = torch.full_like(delta, float(kl_weight))
    kl_coefficients = torch.where(
        safety_regression,
        torch.full_like(kl_coefficients, float(safety_kl_weight)),
        kl_coefficients,
    )
    kl_terms = reference_mean_kl * kl_coefficients.unsqueeze(-1)
    kl_per_scene = (kl_terms * all_mask).sum(dim=(1, 2)) / all_count
    kl_loss = (
        kl_per_scene[group_valid].mean()
        if group_valid.any()
        else reference_mean_kl.sum() * 0.0
    )
    total = policy_loss + bc_loss + kl_loss

    hard_scene = group_valid & (public_quality <= float(headroom_low))
    mature_scene = group_valid & (public_quality >= float(headroom_high))
    transition_scene = group_valid & ~hard_scene & ~mature_scene

    def bucket_delta_mean(scene_mask: torch.Tensor) -> torch.Tensor:
        mask = pair_valid & scene_mask.unsqueeze(-1)
        return delta[mask].mean().detach() if mask.any() else delta.sum() * 0.0

    valid_total = pair_valid.float().sum().clamp_min(1.0)
    neutral = optimize & (advantages == 0)
    tie = optimize & (delta == 0)
    return {
        "loss": total,
        "policy_loss": policy_loss,
        "bc_loss": bc_loss,
        "reference_kl_loss": kl_loss,
        "advantages": advantages,
        "advantage_mean": (
            advantages[optimize].mean().detach()
            if optimize.any() else advantages.sum() * 0.0
        ),
        "reward_mean": (
            reward_mean[group_valid].mean().detach()
            if group_valid.any() else rewards.sum() * 0.0
        ),
        "reward_std": (
            reward_std[group_valid].mean().detach()
            if group_valid.any() else rewards.sum() * 0.0
        ),
        "deployment_delta_mean": (
            delta[pair_valid].mean().detach()
            if pair_valid.any() else delta.sum() * 0.0
        ),
        "hard_delta_mean": bucket_delta_mean(hard_scene),
        "transition_delta_mean": bucket_delta_mean(transition_scene),
        "mature_delta_mean": bucket_delta_mean(mature_scene),
        "headroom_mean": (
            headroom[group_valid].mean().detach()
            if group_valid.any() else headroom.sum() * 0.0
        ),
        "hard_scene_fraction": hard_scene.float().mean().detach(),
        "transition_scene_fraction": transition_scene.float().mean().detach(),
        "mature_scene_fraction": mature_scene.float().mean().detach(),
        "positive_fraction": (
            ((advantages > 0) & optimize).float().sum() / valid_total
        ).detach(),
        "negative_fraction": (
            ((advantages < 0) & optimize).float().sum() / valid_total
        ).detach(),
        "neutral_fraction": (neutral.float().sum() / valid_total).detach(),
        "tie_fraction": (tie.float().sum() / valid_total).detach(),
        "policy_active_scene_fraction": active_scenes.float().mean().detach(),
        "safety_override_fraction": (
            safety_regression.float().sum() / valid_total
        ).detach(),
        "catastrophic_fraction": (
            ((delta <= -0.5) & pair_valid).float().sum() / valid_total
        ).detach(),
        "mean_bc_weight": rewards.new_tensor(float(bc_weight)),
        "mean_kl_weight": (
            kl_coefficients[optimize].mean().detach()
            if optimize.any() else kl_coefficients.sum() * 0.0
        ),
        "mean_exact_kl": (
            reference_mean_kl[all_mask].mean().detach()
            if all_mask.any() else reference_mean_kl.sum() * 0.0
        ),
        "mean_current_log_prob": (
            current_log_probs[all_mask].mean().detach()
            if all_mask.any() else current_log_probs.sum() * 0.0
        ),
        "delta_scale_mean": (
            delta_scale[group_valid].mean().detach()
            if group_valid.any() else delta_scale.sum() * 0.0
        ),
        "frontier_rank_enabled": rewards.new_tensor(float(use_frontier_rank)),
        "discount_first": discounts[0].detach(),
        "discount_last": discounts[-1].detach(),
    }


def compute_stage33_cdc_smoke_objective(
    *,
    current_log_probs: torch.Tensor,
    bc_log_probs: torch.Tensor,
    reference_mean_kl: torch.Tensor,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    component_scores: torch.Tensor,
    base_rewards: torch.Tensor,
    base_valid_mask: torch.Tensor,
    base_component_scores: torch.Tensor,
    deployment_weight: float = 0.5,
    headroom_weight: float = 0.5,
    headroom_margin: float = 0.001,
    advantage_clip: float = 2.0,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Stage33 candidate-credit-compatible smoke objective.

    The full candidate-level trace is enabled only when its manifest is
    available.  Until then this fail-closed pilot uses the selected common-
    noise pair and adds detached headroom credit to the deployed reward.  It
    is intentionally a separate wrapper so Stage31/32 semantics are unchanged.
    """
    if not all(torch.isfinite(t).all() for t in (
        current_log_probs, bc_log_probs, reference_mean_kl, rewards,
        base_rewards, component_scores, base_component_scores,
    )):
        raise FloatingPointError("Stage33 CDC tensors must be finite")
    if rewards.shape != base_rewards.shape:
        raise ValueError("Stage33 current/public rewards must have equal shape")
    deployment_delta = rewards - base_rewards
    headroom = (deployment_delta - float(headroom_margin)).clamp_min(0.0)
    effective_rewards = base_rewards + float(deployment_weight) * deployment_delta
    effective_rewards = effective_rewards + float(headroom_weight) * headroom.detach()
    objective = compute_deployed_decision_grpo_objective(
        current_log_probs=current_log_probs,
        bc_log_probs=bc_log_probs,
        reference_mean_kl=reference_mean_kl,
        rewards=effective_rewards,
        valid_mask=valid_mask,
        component_scores=component_scores,
        base_rewards=base_rewards,
        base_valid_mask=base_valid_mask,
        base_component_scores=base_component_scores,
        use_frontier_rank=False,
        advantage_clip=float(advantage_clip),
        **kwargs,
    )
    objective["stage33_deployment_credit_mean"] = deployment_delta.mean().detach()
    objective["stage33_headroom_credit_mean"] = headroom.mean().detach()
    objective["stage33_candidate_level_trace"] = rewards.new_tensor(0.0)
    return objective


def compute_selector_aware_frontier_objective(
    current_log_probs: torch.Tensor,
    bc_log_probs: torch.Tensor,
    reference_mean_kl: torch.Tensor,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    component_scores: torch.Tensor,
    base_rewards: torch.Tensor,
    base_valid_mask: torch.Tensor,
    base_component_scores: torch.Tensor,
    current_mode_valid: torch.Tensor,
    public_mode_valid: torch.Tensor,
    group_size: int = 8,
    pool_size: int = 4,
    advantage_eps: float = 1e-3,
    headroom_low: float = 0.75,
    headroom_high: float = 0.90,
    delta_scale_floor: float = 0.002,
    owner_margin: float = 0.001,
    frontier_weight: float = 0.5,
    mature_cap: float = 0.25,
    advantage_clip: float = 2.0,
    safety_tolerance: float = 1e-6,
    step_discount: float = 0.6,
    bc_weight: float = 0.1,
    kl_weight: float = 0.1,
    safety_kl_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Stage32 selector-aware frontier credit assignment."""
    if current_log_probs.ndim != 3:
        raise ValueError("Stage32 log probabilities must have shape [B,N,S]")
    batch, total, steps = current_log_probs.shape
    expected_total = int(group_size) * (int(pool_size) + 1)
    if total != expected_total:
        raise ValueError("Stage32 replay width does not match group/pool size")
    for name, tensor in (
        ("bc_log_probs", bc_log_probs),
        ("reference_mean_kl", reference_mean_kl),
    ):
        if tensor.shape != current_log_probs.shape:
            raise ValueError(f"Stage32 {name} shape differs from current log probs")
    expected = (batch, total)
    for name, tensor in (
        ("rewards", rewards), ("valid_mask", valid_mask),
        ("base_rewards", base_rewards), ("base_valid_mask", base_valid_mask),
        ("current_mode_valid", current_mode_valid),
        ("public_mode_valid", public_mode_valid),
    ):
        if tensor.shape != expected:
            raise ValueError(f"Stage32 {name} must have shape [B,G*(K+1)]")
    if component_scores.shape != (batch, total, 6):
        raise ValueError("Stage32 current components must have shape [B,N,6]")
    if base_component_scores.shape != component_scores.shape:
        raise ValueError("Stage32 current/public component shapes differ")
    constants = (
        advantage_eps, headroom_low, headroom_high, delta_scale_floor,
        owner_margin, frontier_weight, mature_cap, advantage_clip,
        safety_tolerance, step_discount, bc_weight, kl_weight,
        safety_kl_weight,
    )
    if not all(math.isfinite(float(value)) for value in constants):
        raise ValueError("Stage32 objective constants must be finite")
    if (
        advantage_eps <= 0 or not 0 <= headroom_low < headroom_high <= 1
        or delta_scale_floor <= 0 or owner_margin < 0 or frontier_weight < 0
        or not 0 <= mature_cap <= 1 or advantage_clip <= 0
        or safety_tolerance < 0 or not 0 < step_discount <= 1
        or min(bc_weight, kl_weight, safety_kl_weight) < 0
        or safety_kl_weight < kl_weight
    ):
        raise ValueError("invalid Stage32 objective constants")
    if not (
        torch.isfinite(current_log_probs).all()
        and torch.isfinite(bc_log_probs).all()
        and torch.isfinite(reference_mean_kl).all()
    ):
        raise FloatingPointError("Stage32 policy/BC/KL tensors must be finite")
    if (reference_mean_kl < 0).any():
        raise FloatingPointError("Stage32 exact KL must be non-negative")

    groups = int(group_size)
    width = int(pool_size) + 1
    rewards3 = rewards.float().reshape(batch, groups, width)
    base_rewards3 = base_rewards.float().reshape(batch, groups, width)
    valid3 = (
        valid_mask.bool().reshape(batch, groups, width)
        & base_valid_mask.bool().reshape(batch, groups, width)
        & current_mode_valid.bool().reshape(batch, groups, width)
        & public_mode_valid.bool().reshape(batch, groups, width)
        & torch.isfinite(rewards3)
        & torch.isfinite(base_rewards3)
        & torch.isfinite(component_scores).all(dim=-1).reshape(batch, groups, width)
        & torch.isfinite(base_component_scores).all(dim=-1).reshape(batch, groups, width)
    )
    selected_valid = valid3[..., 0]
    group_valid = selected_valid
    scene_valid = group_valid.any(dim=-1)
    selected_delta = torch.where(
        selected_valid, rewards3[..., 0] - base_rewards3[..., 0],
        torch.zeros_like(rewards3[..., 0]),
    )
    selected_current = torch.where(
        selected_valid, rewards3[..., 0], torch.zeros_like(rewards3[..., 0])
    )
    selected_public = torch.where(
        selected_valid, base_rewards3[..., 0], torch.zeros_like(base_rewards3[..., 0])
    )
    denominator = selected_valid.sum(dim=-1).clamp_min(1).float()
    public_quality = selected_public.sum(dim=-1) / denominator
    selected_mean = selected_delta.sum(dim=-1) / denominator
    selected_centered = torch.where(
        selected_valid, selected_delta - selected_mean.unsqueeze(-1),
        torch.zeros_like(selected_delta),
    )
    selected_std = torch.sqrt(
        selected_centered.square().sum(dim=-1) / denominator
    )
    selected_scale = selected_std.clamp_min(float(delta_scale_floor))
    selected_z = selected_centered / selected_scale.unsqueeze(-1)
    selected_reward_mean = selected_current.sum(dim=-1) / denominator
    selected_reward_centered = torch.where(
        selected_valid, selected_current - selected_reward_mean.unsqueeze(-1),
        torch.zeros_like(selected_current),
    )
    selected_reward_std = torch.sqrt(
        selected_reward_centered.square().sum(dim=-1) / denominator
    )
    selected_reward_z = torch.where(
        selected_valid & (selected_reward_std > advantage_eps).unsqueeze(-1),
        selected_reward_centered
        / selected_reward_std.clamp_min(advantage_eps).unsqueeze(-1),
        torch.zeros_like(selected_reward_centered),
    ).clamp(-float(advantage_clip), float(advantage_clip))
    headroom = (
        (float(headroom_high) - public_quality)
        / float(headroom_high - headroom_low)
    ).clamp(0.0, 1.0)
    selected_raw = selected_z + 0.5 * headroom.unsqueeze(-1) * selected_reward_z
    selected_adv = torch.where(
        selected_delta > 0, selected_raw.clamp_min(0.0),
        torch.where(selected_delta < 0, selected_raw.clamp_max(0.0), selected_raw),
    )

    frontier_valid = valid3
    current_frontier, current_owner = rewards3.masked_fill(
        ~frontier_valid, -torch.inf
    ).max(dim=-1)
    public_frontier = base_rewards3.masked_fill(
        ~frontier_valid, -torch.inf
    ).max(dim=-1).values
    frontier_ok = torch.isfinite(current_frontier) & torch.isfinite(public_frontier)
    frontier_delta = torch.where(
        frontier_ok, current_frontier - public_frontier,
        torch.zeros_like(current_frontier),
    )
    frontier_centered = torch.where(
        frontier_ok, frontier_delta - frontier_delta.mean(dim=-1, keepdim=True),
        torch.zeros_like(frontier_delta),
    )
    frontier_std = torch.sqrt(
        frontier_centered.square().sum(dim=-1)
        / frontier_ok.sum(dim=-1).clamp_min(1).float()
    )
    frontier_scale = frontier_std.clamp_min(float(delta_scale_floor))
    frontier_z = frontier_centered / frontier_scale.unsqueeze(-1)
    selected_frontier = current_frontier <= selected_current + float(owner_margin)
    candidate_owner = (
        frontier_ok & ~selected_frontier & (current_owner > 0)
    )
    owner_mask = torch.zeros_like(valid3)
    owner_mask[..., 0] = selected_frontier & frontier_ok
    owner_mask.scatter_(2, current_owner.unsqueeze(-1), frontier_ok.unsqueeze(-1))
    frontier_credit = (
        frontier_z * float(frontier_weight)
        * torch.where(
            (public_quality >= float(headroom_high)).unsqueeze(-1),
            torch.full_like(frontier_z, float(mature_cap)),
            torch.ones_like(frontier_z),
        )
    )
    advantages3 = torch.zeros_like(rewards3)
    advantages3[..., 0] = selected_adv
    advantages3 = advantages3 + owner_mask.float() * frontier_credit.unsqueeze(-1)

    safety_indices = torch.tensor((0, 1, 3), device=rewards.device)
    comp3 = component_scores.reshape(batch, groups, width, 6).float()
    base_comp3 = base_component_scores.reshape(batch, groups, width, 6).float()
    safety_regression = valid3 & (
        comp3.index_select(-1, safety_indices)
        < base_comp3.index_select(-1, safety_indices) - float(safety_tolerance)
    ).any(dim=-1)
    advantages3 = torch.where(
        safety_regression,
        torch.full_like(advantages3, -float(advantage_clip)),
        advantages3,
    )
    optimize3 = valid3 & group_valid.unsqueeze(-1)
    advantages3 = torch.where(
        optimize3,
        advantages3.clamp(-float(advantage_clip), float(advantage_clip)),
        torch.zeros_like(advantages3),
    ).detach()
    if not torch.isfinite(advantages3).all():
        raise FloatingPointError("Stage32 advantages must be finite")

    advantages = advantages3.reshape(batch, total)
    optimize = optimize3.reshape(batch, total)
    indices = torch.arange(
        steps, device=rewards.device, dtype=current_log_probs.dtype
    )
    discounts = step_discount ** (steps - indices - 1)
    active = optimize & (advantages != 0)
    policy_terms = -current_log_probs * advantages.unsqueeze(-1) * discounts.view(1, 1, -1)
    policy_mask = active.unsqueeze(-1).expand_as(current_log_probs)
    policy_count = policy_mask.sum(dim=(1, 2)).clamp_min(1).float()
    policy_per_scene = torch.where(
        active.any(dim=-1),
        (policy_terms * policy_mask).sum(dim=(1, 2)) / policy_count,
        torch.zeros_like(policy_count),
    )
    active_scenes = active.any(dim=-1)
    policy_loss = (
        policy_per_scene[active_scenes].mean()
        if active_scenes.any() else current_log_probs.sum() * 0.0
    )
    all_mask = optimize.unsqueeze(-1).expand_as(current_log_probs)
    all_count = all_mask.sum(dim=(1, 2)).clamp_min(1).float()
    bc_terms = -bc_log_probs * discounts.view(1, 1, -1) * float(bc_weight)
    bc_per_scene = (bc_terms * all_mask).sum(dim=(1, 2)) / all_count
    bc_loss = (
        bc_per_scene[scene_valid].mean()
        if scene_valid.any() else bc_log_probs.sum() * 0.0
    )
    kl_coeff = torch.full_like(advantages, float(kl_weight))
    kl_coeff = torch.where(
        safety_regression.reshape(batch, total),
        torch.full_like(kl_coeff, float(safety_kl_weight)), kl_coeff,
    )
    kl_terms = reference_mean_kl * kl_coeff.unsqueeze(-1)
    kl_per_scene = (kl_terms * all_mask).sum(dim=(1, 2)) / all_count
    kl_loss = (
        kl_per_scene[scene_valid].mean()
        if scene_valid.any() else reference_mean_kl.sum() * 0.0
    )
    pair_valid_count = valid3.float().sum().clamp_min(1.0)
    candidate_owner_fraction = (
        candidate_owner.float().sum() / frontier_ok.float().sum().clamp_min(1.0)
    ).detach()
    return {
        "loss": policy_loss + bc_loss + kl_loss,
        "policy_loss": policy_loss,
        "bc_loss": bc_loss,
        "reference_kl_loss": kl_loss,
        "advantages": advantages,
        "advantage_mean": advantages[optimize].mean().detach() if optimize.any() else advantages.sum() * 0.0,
        "reward_mean": rewards[valid_mask.bool()].float().mean().detach() if valid_mask.any() else rewards.sum() * 0.0,
        "reward_std": rewards[valid_mask.bool()].float().std(unbiased=False).detach() if valid_mask.any() else rewards.sum() * 0.0,
        "deployment_delta_mean": selected_delta[selected_valid].mean().detach() if selected_valid.any() else selected_delta.sum() * 0.0,
        "hard_delta_mean": selected_delta[(selected_valid) & (public_quality <= float(headroom_low)).unsqueeze(-1)].mean().detach() if ((selected_valid) & (public_quality <= float(headroom_low)).unsqueeze(-1)).any() else selected_delta.sum() * 0.0,
        "transition_delta_mean": selected_delta[(selected_valid) & (public_quality > float(headroom_low)).unsqueeze(-1) & (public_quality < float(headroom_high)).unsqueeze(-1)].mean().detach() if ((selected_valid) & (public_quality > float(headroom_low)).unsqueeze(-1) & (public_quality < float(headroom_high)).unsqueeze(-1)).any() else selected_delta.sum() * 0.0,
        "mature_delta_mean": selected_delta[(selected_valid) & (public_quality >= float(headroom_high)).unsqueeze(-1)].mean().detach() if ((selected_valid) & (public_quality >= float(headroom_high)).unsqueeze(-1)).any() else selected_delta.sum() * 0.0,
        "frontier_delta_mean": frontier_delta[frontier_ok].mean().detach() if frontier_ok.any() else frontier_delta.sum() * 0.0,
        "frontier_candidate_gain": frontier_delta[candidate_owner].mean().detach() if candidate_owner.any() else frontier_delta.sum() * 0.0,
        "frontier_owner_fraction": owner_mask.float().sum().detach() / pair_valid_count,
        "frontier_candidate_owner_fraction": candidate_owner_fraction,
        "frontier_pool_valid_fraction": valid3[..., 1:].float().mean().detach(),
        "positive_fraction": ((advantages > 0) & optimize).float().sum().detach() / pair_valid_count,
        "negative_fraction": ((advantages < 0) & optimize).float().sum().detach() / pair_valid_count,
        "neutral_fraction": ((advantages == 0) & optimize).float().sum().detach() / pair_valid_count,
        "policy_active_scene_fraction": active_scenes.float().mean().detach(),
        "safety_override_fraction": safety_regression.float().sum().detach() / pair_valid_count,
        "catastrophic_fraction": ((rewards.float() - base_rewards.float() <= -0.5) & valid3.reshape(batch, total)).float().sum().detach() / pair_valid_count,
        "mean_bc_weight": rewards.new_tensor(float(bc_weight)),
        "mean_kl_weight": kl_coeff[optimize].mean().detach() if optimize.any() else kl_coeff.sum() * 0.0,
        "mean_exact_kl": reference_mean_kl[all_mask].mean().detach() if all_mask.any() else reference_mean_kl.sum() * 0.0,
        "mean_current_log_prob": current_log_probs[all_mask].mean().detach() if all_mask.any() else current_log_probs.sum() * 0.0,
        "delta_scale_mean": selected_scale[scene_valid].mean().detach() if scene_valid.any() else selected_scale.sum() * 0.0,
        "frontier_rank_enabled": rewards.new_tensor(1.0),
        "discount_first": discounts[0].detach(),
        "discount_last": discounts[-1].detach(),
    }


def compute_mode_coverage_constrained_objective(
    current_log_probs: torch.Tensor,
    bc_log_probs: torch.Tensor,
    reference_mean_kl: torch.Tensor,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    component_scores: torch.Tensor,
    base_rewards: torch.Tensor,
    base_valid_mask: torch.Tensor,
    base_component_scores: torch.Tensor,
    scene_buckets: torch.Tensor,
    constrained: bool,
    advantage_eps: float = 1e-3,
    delta_scale_floor: float = 0.002,
    advantage_clip: float = 2.0,
    top_k: int = 5,
    coverage_top_weight: float = 0.5,
    boundary_top_weight: float = 0.25,
    boundary_positive_margin: float = 0.001,
    boundary_negative_multiplier: float = 2.0,
    mature_negative_floor: float = -0.0002,
    mature_positive_margin: float = 0.002,
    mature_negative_multiplier: float = 4.0,
    safety_tolerance: float = 1e-6,
    component_tolerance: float = 1e-6,
    step_discount: float = 0.6,
) -> Dict[str, torch.Tensor]:
    """Stage30 all-mode paired objective with bucket-specific preservation."""
    if current_log_probs.ndim != 3:
        raise ValueError("Stage30 log probabilities must have shape [B,20,S]")
    batch, group, steps = current_log_probs.shape
    if group != 20:
        raise ValueError("Stage30 requires exactly 20 modes")
    expected = (batch, group)
    for name, tensor in (
        ("rewards", rewards), ("valid_mask", valid_mask),
        ("base_rewards", base_rewards), ("base_valid_mask", base_valid_mask),
    ):
        if tensor.shape != expected:
            raise ValueError(f"Stage30 {name} must have shape [B,20]")
    if component_scores.shape != (batch, group, 6):
        raise ValueError("Stage30 current components must have shape [B,20,6]")
    if base_component_scores.shape != component_scores.shape:
        raise ValueError("Stage30 current/base component shapes differ")
    if bc_log_probs.shape != current_log_probs.shape:
        raise ValueError("Stage30 BC log probabilities must match [B,20,S]")
    if reference_mean_kl.shape != current_log_probs.shape:
        raise ValueError("Stage30 exact KL must match [B,20,S]")
    if scene_buckets.shape != (batch,):
        raise ValueError("Stage30 scene buckets must have shape [B]")
    if not torch.all((scene_buckets >= 0) & (scene_buckets <= 3)):
        raise ValueError("Stage30 scene bucket ids must be in [0,3]")
    constants = (
        advantage_eps, delta_scale_floor, advantage_clip,
        coverage_top_weight, boundary_top_weight, boundary_positive_margin,
        boundary_negative_multiplier, mature_negative_floor,
        mature_positive_margin, mature_negative_multiplier, safety_tolerance,
        component_tolerance, step_discount,
    )
    if not all(math.isfinite(float(value)) for value in constants):
        raise ValueError("Stage30 objective constants must be finite")
    if (
        advantage_eps <= 0 or delta_scale_floor <= 0 or advantage_clip <= 0
        or top_k <= 0 or top_k > group or coverage_top_weight < 0
        or boundary_top_weight < 0 or boundary_positive_margin < 0
        or boundary_negative_multiplier < 1 or mature_negative_floor > 0
        or mature_positive_margin <= 0 or mature_negative_multiplier < 1
        or safety_tolerance < 0 or component_tolerance < 0
        or not 0 < step_discount <= 1
    ):
        raise ValueError("invalid Stage30 objective constants")
    if not (
        torch.isfinite(current_log_probs).all()
        and torch.isfinite(bc_log_probs).all()
        and torch.isfinite(reference_mean_kl).all()
    ):
        raise FloatingPointError("Stage30 policy/BC/KL tensors must be finite")
    if (reference_mean_kl < 0).any():
        raise FloatingPointError("Stage30 exact KL must be non-negative")

    pair_valid = (
        valid_mask.bool() & base_valid_mask.bool()
        & torch.isfinite(rewards) & torch.isfinite(base_rewards)
        & torch.isfinite(component_scores).all(dim=-1)
        & torch.isfinite(base_component_scores).all(dim=-1)
    )
    valid_count = pair_valid.sum(dim=-1)
    group_valid = valid_count >= top_k
    denominator = valid_count.clamp_min(1).float()
    current = torch.where(pair_valid, rewards.float(), torch.zeros_like(rewards.float()))
    reward_mean = current.sum(dim=-1) / denominator
    centered = torch.where(
        pair_valid, rewards.float() - reward_mean.unsqueeze(-1),
        torch.zeros_like(rewards.float()),
    )
    reward_std = torch.sqrt(centered.square().sum(dim=-1) / denominator)
    reward_z = torch.where(
        pair_valid & (reward_std > advantage_eps).unsqueeze(-1),
        centered / reward_std.clamp_min(advantage_eps).unsqueeze(-1),
        torch.zeros_like(centered),
    ).clamp(-float(advantage_clip), float(advantage_clip))

    delta = torch.where(
        pair_valid, rewards.float() - base_rewards.float(),
        torch.zeros_like(rewards.float()),
    )
    delta_mean = delta.sum(dim=-1) / denominator
    delta_centered = torch.where(
        pair_valid, delta - delta_mean.unsqueeze(-1), torch.zeros_like(delta),
    )
    delta_std = torch.sqrt(delta_centered.square().sum(dim=-1) / denominator)
    paired_scale = delta_std.clamp_min(float(delta_scale_floor))
    paired = (delta / paired_scale.unsqueeze(-1)).clamp(
        -float(advantage_clip), float(advantage_clip)
    )
    masked_rewards = rewards.float().masked_fill(~pair_valid, float("-inf"))
    top_indices = masked_rewards.topk(int(top_k), dim=-1).indices
    top_tail = torch.zeros_like(pair_valid)
    top_tail.scatter_(1, top_indices, True)
    top_tail &= pair_valid & group_valid.unsqueeze(-1)
    top_bonus = torch.relu(reward_z) * top_tail.float()

    bucket = scene_buckets.to(device=rewards.device, dtype=torch.long)
    top_weight = torch.full_like(delta, float(coverage_top_weight))
    if constrained:
        top_weight = torch.where(
            (bucket == 2).unsqueeze(-1),
            top_weight.new_full(top_weight.shape, float(boundary_top_weight)),
            top_weight,
        )
    raw_advantage = paired + top_weight * top_bonus
    advantages = torch.where(
        delta > 0, raw_advantage.clamp_min(0.0),
        torch.where(delta < 0, raw_advantage.clamp_max(0.0), torch.zeros_like(delta)),
    )

    component_no_worse = (
        component_scores.float()
        >= base_component_scores.float() - float(component_tolerance)
    ).all(dim=-1)
    if constrained:
        boundary = (bucket == 2).unsqueeze(-1)
        mature = (bucket == 3).unsqueeze(-1)
        boundary_advantage = torch.where(
            delta < 0,
            advantages * float(boundary_negative_multiplier),
            torch.where(
                delta >= float(boundary_positive_margin),
                advantages,
                torch.zeros_like(advantages),
            ),
        )
        mature_advantage = torch.where(
            delta < float(mature_negative_floor),
            advantages * float(mature_negative_multiplier),
            torch.where(
                (delta >= float(mature_positive_margin)) & component_no_worse,
                advantages.clamp_min(0.0),
                torch.zeros_like(advantages),
            ),
        )
        advantages = torch.where(
            mature, mature_advantage,
            torch.where(boundary, boundary_advantage, advantages),
        )

    safety_indices = torch.tensor((0, 1, 3), device=rewards.device)
    safety_regression = pair_valid & (
        component_scores.float().index_select(-1, safety_indices)
        < base_component_scores.float().index_select(-1, safety_indices)
        - float(safety_tolerance)
    ).any(dim=-1)
    advantages = torch.where(
        safety_regression, torch.full_like(advantages, -float(advantage_clip)),
        advantages,
    )
    optimize = pair_valid & group_valid.unsqueeze(-1)
    advantages = torch.where(
        optimize, advantages.clamp(-float(advantage_clip), float(advantage_clip)),
        torch.zeros_like(advantages),
    ).detach()
    if not torch.isfinite(advantages).all():
        raise FloatingPointError("Stage30 advantages must be finite")

    indices = torch.arange(steps, device=rewards.device, dtype=current_log_probs.dtype)
    discounts = step_discount ** (steps - indices - 1)
    active = optimize & (advantages != 0)
    policy_terms = -current_log_probs * advantages.unsqueeze(-1) * discounts.view(1, 1, -1)
    policy_mask = active.unsqueeze(-1).expand_as(current_log_probs)
    policy_count = policy_mask.sum(dim=(1, 2)).clamp_min(1).float()
    policy_per_scene = torch.where(
        active.any(dim=-1),
        (policy_terms * policy_mask).sum(dim=(1, 2)) / policy_count,
        torch.zeros_like(policy_count),
    )
    active_scene = active.any(dim=-1)
    policy_parts = []
    for scene_mask in (bucket <= 1, bucket == 2, bucket == 3):
        selected = group_valid & active_scene & scene_mask
        if selected.any():
            policy_parts.append(policy_per_scene[selected].mean())
    policy_loss = (
        torch.stack(policy_parts).sum()
        if policy_parts else current_log_probs.sum() * 0.0
    )

    if constrained:
        bc_coeff = torch.where(
            bucket <= 1, rewards.new_tensor(0.05),
            torch.where(bucket == 2, rewards.new_tensor(0.2), rewards.new_tensor(0.5)),
        ).float()
        kl_coeff = torch.where(
            bucket <= 1, rewards.new_tensor(0.05),
            torch.where(bucket == 2, rewards.new_tensor(0.5), rewards.new_tensor(1.0)),
        ).float()
    else:
        bc_coeff = rewards.new_full((batch,), 0.1).float()
        kl_coeff = rewards.new_full((batch,), 0.1).float()
    all_mask = optimize.unsqueeze(-1).expand_as(current_log_probs)
    all_count = all_mask.sum(dim=(1, 2)).clamp_min(1).float()
    bc_terms = -bc_log_probs * discounts.view(1, 1, -1) * bc_coeff.view(-1, 1, 1)
    bc_per_scene = (bc_terms * all_mask).sum(dim=(1, 2)) / all_count
    kl_terms = reference_mean_kl * kl_coeff.view(-1, 1, 1)
    kl_per_scene = (kl_terms * all_mask).sum(dim=(1, 2)) / all_count
    bc_loss = bc_per_scene[group_valid].mean() if group_valid.any() else bc_log_probs.sum() * 0.0
    kl_loss = kl_per_scene[group_valid].mean() if group_valid.any() else reference_mean_kl.sum() * 0.0
    valid_total = optimize.float().sum().clamp_min(1.0)

    def bucket_delta(scene_mask: torch.Tensor) -> torch.Tensor:
        selected = pair_valid & scene_mask.unsqueeze(-1)
        return delta[selected].mean().detach() if selected.any() else delta.sum() * 0.0

    return {
        "loss": policy_loss + bc_loss + kl_loss,
        "policy_loss": policy_loss,
        "bc_loss": bc_loss,
        "reference_kl_loss": kl_loss,
        "advantages": advantages,
        "advantage_mean": advantages[optimize].mean().detach() if optimize.any() else advantages.sum() * 0.0,
        "reward_mean": reward_mean[group_valid].mean().detach() if group_valid.any() else rewards.sum() * 0.0,
        "reward_std": reward_std[group_valid].mean().detach() if group_valid.any() else rewards.sum() * 0.0,
        "base_delta_mean": delta[pair_valid].mean().detach() if pair_valid.any() else delta.sum() * 0.0,
        "severe_delta_mean": bucket_delta(bucket == 0),
        "recoverable_delta_mean": bucket_delta(bucket == 1),
        "boundary_delta_mean": bucket_delta(bucket == 2),
        "mature_delta_mean": bucket_delta(bucket == 3),
        "positive_fraction": ((advantages > 0) & optimize).float().sum().detach() / valid_total,
        "negative_fraction": ((advantages < 0) & optimize).float().sum().detach() / valid_total,
        "neutral_fraction": ((advantages == 0) & optimize).float().sum().detach() / valid_total,
        "policy_active_scene_fraction": active_scene.float().mean().detach(),
        "top_tail_fraction": top_tail.float().sum().detach() / valid_total,
        "safety_override_fraction": safety_regression.float().sum().detach() / valid_total,
        "catastrophic_fraction": ((delta <= -0.5) & pair_valid).float().sum().detach() / valid_total,
        "mean_bc_weight": bc_coeff[group_valid].mean().detach() if group_valid.any() else bc_coeff.sum() * 0.0,
        "mean_kl_weight": kl_coeff[group_valid].mean().detach() if group_valid.any() else kl_coeff.sum() * 0.0,
        "mean_exact_kl": reference_mean_kl[all_mask].mean().detach() if all_mask.any() else reference_mean_kl.sum() * 0.0,
        "mean_current_log_prob": current_log_probs[all_mask].mean().detach() if all_mask.any() else current_log_probs.sum() * 0.0,
        "paired_scale_mean": paired_scale[group_valid].mean().detach() if group_valid.any() else paired_scale.sum() * 0.0,
        "discount_first": discounts[0].detach(),
        "discount_last": discounts[-1].detach(),
    }


def compute_base_preserve_diffgrpo_objective(
    current_log_probs: torch.Tensor,
    bc_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    component_scores: torch.Tensor,
    base_rewards: torch.Tensor,
    base_safety: torch.Tensor,
    scene_weights: torch.Tensor,
    bc_weights: torch.Tensor,
    advantage_eps: float = 1e-3,
    margin: float = 0.01,
    scale: float = 0.10,
    advantage_clip: float = 2.0,
    safety_tolerance: float = 1e-6,
    step_discount: float = 0.6,
) -> Dict[str, torch.Tensor]:
    """Stage21 group-relative policy loss anchored to the frozen-base score."""
    if current_log_probs.ndim != 3:
        raise ValueError("Stage21 log probabilities must have shape [B,G,S]")
    batch, group, num_steps = current_log_probs.shape
    if group != 8:
        raise ValueError("Stage21 requires exactly G=8 rollouts")
    if rewards.shape != (batch, group) or valid_mask.shape != rewards.shape:
        raise ValueError("Stage21 rewards/valid mask must have shape [B,G]")
    if bc_log_probs.ndim != 3 or bc_log_probs.shape[0] != batch or bc_log_probs.shape[2] != num_steps:
        raise ValueError("Stage21 BC log probabilities must have shape [B,1|G,S]")
    if bc_log_probs.shape[1] not in {1, group}:
        raise ValueError("Stage21 BC group dimension must be one or G")
    if component_scores.ndim != 3 or component_scores.shape[:2] != (batch, group) or component_scores.shape[2] < 4:
        raise ValueError("Stage21 component scores must have shape [B,G,>=4]")
    if base_rewards.shape != (batch,) or base_safety.shape != (batch, 3):
        raise ValueError("Stage21 base reward/safety metadata shape mismatch")
    for name, tensor in (("scene_weights", scene_weights), ("bc_weights", bc_weights)):
        if tensor.shape != (batch,):
            raise ValueError(f"Stage21 {name} must have shape [B]")
    scalars = (advantage_eps, margin, scale, advantage_clip, safety_tolerance, step_discount)
    if not all(math.isfinite(float(value)) for value in scalars):
        raise ValueError("Stage21 objective constants must be finite")
    if advantage_eps <= 0 or margin < 0 or scale <= 0 or advantage_clip <= 0:
        raise ValueError("invalid Stage21 advantage constants")
    if safety_tolerance < 0 or not 0 < step_discount <= 1:
        raise ValueError("invalid Stage21 safety tolerance or step discount")
    metadata = torch.cat(
        (base_rewards[:, None], base_safety, scene_weights[:, None], bc_weights[:, None]),
        dim=1,
    )
    if not torch.isfinite(current_log_probs).all() or not torch.isfinite(bc_log_probs).all() or not torch.isfinite(metadata).all():
        raise FloatingPointError("Stage21 log probabilities and metadata must be finite")
    if (scene_weights < 0.25).any() or (scene_weights > 4.0).any():
        raise ValueError("Stage21 scene weights must lie in [0.25,4]")
    if (bc_weights < 0.1).any() or (bc_weights > 0.3).any():
        raise ValueError("Stage21 BC weights must lie in [0.1,0.3]")

    finite_valid = valid_mask.bool() & torch.isfinite(rewards)
    valid_count = finite_valid.sum(dim=-1)
    group_valid = valid_count >= 2
    safe_rewards = torch.where(finite_valid, rewards.float(), torch.zeros_like(rewards.float()))
    reward_mean = safe_rewards.sum(dim=-1) / valid_count.clamp_min(1).float()
    centered = torch.where(
        finite_valid, rewards.float() - reward_mean.unsqueeze(-1), torch.zeros_like(rewards.float())
    )
    reward_std = torch.sqrt(
        centered.square().sum(dim=-1) / valid_count.clamp_min(1).float()
    )
    z = torch.where(
        finite_valid & (reward_std > advantage_eps).unsqueeze(-1),
        centered / reward_std.clamp_min(advantage_eps).unsqueeze(-1),
        torch.zeros_like(centered),
    )
    delta = rewards.float() - base_rewards.float().unsqueeze(-1)
    absolute_evidence = (delta.abs() / float(scale)).clamp(max=float(advantage_clip))
    positive = 0.5 * z.clamp_min(0.0) + 0.5 * absolute_evidence
    negative = 0.5 * (-z).clamp_min(0.0) + 0.5 * absolute_evidence
    advantages = torch.zeros_like(rewards.float())
    advantages = torch.where(delta > margin, positive.clamp(max=advantage_clip), advantages)
    advantages = torch.where(delta < -margin, -negative.clamp(max=advantage_clip), advantages)

    rollout_safety = component_scores[..., (0, 1, 3)].float()
    safety_finite = torch.isfinite(rollout_safety).all(dim=-1)
    safety_regression = (
        rollout_safety
        < base_safety.float().unsqueeze(1) - float(safety_tolerance)
    ).any(dim=-1)
    safety_regression &= safety_finite & finite_valid
    advantages = torch.where(
        safety_regression,
        torch.full_like(advantages, -float(advantage_clip)),
        advantages,
    )
    optimize_mode = finite_valid & group_valid.unsqueeze(-1) & (advantages != 0)
    advantages = torch.where(optimize_mode, advantages, torch.zeros_like(advantages)).detach()

    indices = torch.arange(num_steps, device=current_log_probs.device, dtype=current_log_probs.dtype)
    discounts = step_discount ** (num_steps - indices - 1)
    rollout_weights = scene_weights.float().unsqueeze(-1).expand(-1, group)
    rollout_weights = torch.where(
        safety_regression, rollout_weights.clamp_min(1.0), rollout_weights
    ).detach()
    optimize_mask = optimize_mode.unsqueeze(-1).expand_as(current_log_probs)
    weighted_terms = -current_log_probs * advantages.unsqueeze(-1) * discounts.view(1, 1, -1)
    weighted_terms = weighted_terms * rollout_weights.unsqueeze(-1)
    policy_denominator = (
        rollout_weights.unsqueeze(-1).expand_as(current_log_probs)[optimize_mask].sum().clamp_min(1.0)
    )
    policy_loss = (
        weighted_terms[optimize_mask].sum() / policy_denominator
        if optimize_mask.any()
        else current_log_probs.sum() * 0.0
    )

    bc_finite = torch.isfinite(bc_log_probs)
    bc_count = bc_finite.sum(dim=(1, 2)).clamp_min(1)
    bc_per_scene = -torch.where(bc_finite, bc_log_probs, torch.zeros_like(bc_log_probs)).sum(dim=(1, 2)) / bc_count
    bc_loss = (bc_per_scene * bc_weights.detach()).mean()
    total_loss = policy_loss + bc_loss
    valid_total = finite_valid.float().sum().clamp_min(1.0)
    return {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "bc_loss": bc_loss,
        "advantages": advantages,
        "reward_mean": safe_rewards[finite_valid].mean().detach() if finite_valid.any() else rewards.sum() * 0.0,
        "reward_std": reward_std[group_valid].mean().detach() if group_valid.any() else rewards.sum() * 0.0,
        "policy_active_scene_fraction": optimize_mode.any(dim=-1).float().mean().detach(),
        "positive_advantage_fraction": (((advantages > 0) & finite_valid).float().sum() / valid_total).detach(),
        "negative_advantage_fraction": (((advantages < 0) & finite_valid).float().sum() / valid_total).detach(),
        "safety_override_fraction": (safety_regression.float().sum() / valid_total).detach(),
        "base_delta_mean": delta[finite_valid].mean().detach() if finite_valid.any() else rewards.sum() * 0.0,
        "mean_scene_weight": scene_weights.mean().detach(),
        "mean_bc_weight": bc_weights.mean().detach(),
        "discount_first": discounts[0].detach(),
        "discount_last": discounts[-1].detach(),
        "mean_current_log_prob": current_log_probs.mean().detach(),
        "mean_bc_log_prob": bc_log_probs.mean().detach(),
    }


def compute_paired_residual_diffgrpo_objective(
    current_log_probs: torch.Tensor,
    reference_mean_kl: torch.Tensor,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    component_scores: torch.Tensor,
    base_rewards: torch.Tensor,
    base_valid_mask: torch.Tensor,
    base_component_scores: torch.Tensor,
    scene_weights: torch.Tensor,
    advantage_eps: float = 1e-3,
    positive_margin: float = 0.01,
    negative_margin: float = 0.01,
    mature_negative_margin: float = 0.002,
    mature_reward_threshold: float = 0.75,
    scale: float = 0.10,
    advantage_clip: float = 2.0,
    mature_negative_multiplier: float = 2.0,
    safety_tolerance: float = 1e-6,
    regular_kl_weight: float = 0.1,
    mature_kl_weight: float = 0.5,
    bootstrap_advantage_weight: float = 0.25,
    step_discount: float = 0.6,
) -> Dict[str, torch.Tensor]:
    """Stage22 common-random paired-delta policy loss plus exact mean KL."""
    if current_log_probs.ndim != 3 or current_log_probs.shape[1] != 8:
        raise ValueError("Stage22 log probabilities must have shape [B,8,S]")
    batch, group, num_steps = current_log_probs.shape
    if reference_mean_kl.shape != current_log_probs.shape:
        raise ValueError("Stage22 exact KL must match [B,8,S] log probabilities")
    pair_shape = (batch, group)
    for name, tensor in (
        ("rewards", rewards),
        ("valid_mask", valid_mask),
        ("base_rewards", base_rewards),
        ("base_valid_mask", base_valid_mask),
    ):
        if tensor.shape != pair_shape:
            raise ValueError(f"Stage22 {name} must have shape [B,8]")
    if (
        component_scores.shape != (batch, group, 6)
        or base_component_scores.shape != (batch, group, 6)
    ):
        raise ValueError("Stage22 component tensors must have shape [B,8,6]")
    if scene_weights.shape != (batch,):
        raise ValueError("Stage22 scene weights must have shape [B]")
    constants = (
        advantage_eps, positive_margin, negative_margin,
        mature_negative_margin, mature_reward_threshold, scale,
        advantage_clip, mature_negative_multiplier, safety_tolerance,
        regular_kl_weight, mature_kl_weight, bootstrap_advantage_weight,
        step_discount,
    )
    if not all(math.isfinite(float(value)) for value in constants):
        raise ValueError("Stage22 objective constants must be finite")
    if (
        advantage_eps <= 0
        or min(positive_margin, negative_margin, mature_negative_margin) < 0
        or not 0 <= mature_reward_threshold <= 1
        or scale <= 0
        or advantage_clip <= 0
        or mature_negative_multiplier <= 0
        or safety_tolerance < 0
        or regular_kl_weight < 0
        or mature_kl_weight < regular_kl_weight
        or bootstrap_advantage_weight <= 0
        or not 0 < step_discount <= 1
    ):
        raise ValueError("invalid Stage22 objective constants")
    if not torch.isfinite(current_log_probs).all():
        raise FloatingPointError("Stage22 current log probabilities must be finite")
    if not torch.isfinite(reference_mean_kl).all() or (reference_mean_kl < 0).any():
        raise FloatingPointError("Stage22 exact KL must be finite and non-negative")
    if (
        not torch.isfinite(scene_weights).all()
        or (scene_weights < 0.5).any()
        or (scene_weights > 2.0).any()
    ):
        raise ValueError("Stage22 scene weights must lie in [0.5,2]")

    pair_valid = (
        valid_mask.bool()
        & base_valid_mask.bool()
        & torch.isfinite(rewards)
        & torch.isfinite(base_rewards)
        & torch.isfinite(component_scores).all(dim=-1)
        & torch.isfinite(base_component_scores).all(dim=-1)
    )
    valid_count = pair_valid.sum(dim=-1)
    group_valid = valid_count >= 2
    delta = torch.where(
        pair_valid,
        rewards.float() - base_rewards.float(),
        torch.zeros_like(rewards.float()),
    )
    delta_mean = delta.sum(dim=-1) / valid_count.clamp_min(1).float()
    centered = torch.where(
        pair_valid,
        delta - delta_mean.unsqueeze(-1),
        torch.zeros_like(delta),
    )
    delta_std = torch.sqrt(
        centered.square().sum(dim=-1) / valid_count.clamp_min(1).float()
    )
    z = torch.where(
        pair_valid & (delta_std > advantage_eps).unsqueeze(-1),
        centered / delta_std.clamp_min(advantage_eps).unsqueeze(-1),
        torch.zeros_like(delta),
    )
    evidence = (delta.abs() / float(scale)).clamp(max=float(advantage_clip))
    positive_value = 0.5 * z.clamp_min(0.0) + 0.5 * evidence
    negative_value = 0.5 * (-z).clamp_min(0.0) + 0.5 * evidence
    mature = pair_valid & (base_rewards.float() >= mature_reward_threshold)
    positive_mask = pair_valid & (delta > positive_margin)
    normal_negative = pair_valid & (delta < -negative_margin)
    mature_negative = mature & (delta < -mature_negative_margin)
    negative_mask = normal_negative | mature_negative

    current_reward_mean = (
        torch.where(pair_valid, rewards.float(), torch.zeros_like(rewards.float()))
        .sum(dim=-1)
        / valid_count.clamp_min(1).float()
    )
    current_reward_centered = torch.where(
        pair_valid,
        rewards.float() - current_reward_mean.unsqueeze(-1),
        torch.zeros_like(rewards.float()),
    )
    current_reward_std = torch.sqrt(
        current_reward_centered.square().sum(dim=-1)
        / valid_count.clamp_min(1).float()
    )
    current_reward_z = torch.where(
        pair_valid & (current_reward_std > advantage_eps).unsqueeze(-1),
        current_reward_centered
        / current_reward_std.clamp_min(advantage_eps).unsqueeze(-1),
        torch.zeros_like(current_reward_centered),
    )
    bootstrap_advantage = (
        current_reward_z.clamp(min=-1.0, max=1.0)
        * float(bootstrap_advantage_weight)
    )

    advantages = torch.zeros_like(delta)
    advantages = torch.where(
        positive_mask,
        positive_value.clamp(max=advantage_clip),
        advantages,
    )
    negative_advantage = -negative_value
    negative_advantage = torch.where(
        mature_negative,
        negative_advantage * mature_negative_multiplier,
        negative_advantage,
    ).clamp(min=-advantage_clip, max=0.0)
    advantages = torch.where(negative_mask, negative_advantage, advantages)

    safety_indices = (0, 1, 3)
    current_safety = component_scores[..., safety_indices].float()
    base_safety = base_component_scores[..., safety_indices].float()
    safety_regression = pair_valid & (
        current_safety < base_safety - float(safety_tolerance)
    ).any(dim=-1)
    advantages = torch.where(
        safety_regression,
        torch.full_like(advantages, -float(advantage_clip)),
        advantages,
    )
    paired_active = positive_mask | negative_mask | safety_regression
    bootstrap_mask = (
        pair_valid
        & group_valid.unsqueeze(-1)
        & ~paired_active
        & (bootstrap_advantage != 0)
    )
    advantages = torch.where(
        bootstrap_mask, bootstrap_advantage, advantages
    )
    optimize_mode = (
        pair_valid & group_valid.unsqueeze(-1) & (advantages != 0)
    )
    advantages = torch.where(
        optimize_mode, advantages, torch.zeros_like(advantages)
    ).detach()

    rollout_weights = scene_weights.float().unsqueeze(-1).expand(-1, group)
    protected_negative = safety_regression | mature_negative
    rollout_weights = torch.where(
        protected_negative, rollout_weights.clamp_min(1.0), rollout_weights
    ).detach()
    indices = torch.arange(
        num_steps, device=current_log_probs.device, dtype=current_log_probs.dtype
    )
    discounts = step_discount ** (num_steps - indices - 1)
    optimize_mask = optimize_mode.unsqueeze(-1).expand_as(current_log_probs)
    policy_terms = (
        -current_log_probs
        * advantages.unsqueeze(-1)
        * discounts.view(1, 1, -1)
        * rollout_weights.unsqueeze(-1)
    )
    denominator = (
        rollout_weights.unsqueeze(-1)
        .expand_as(current_log_probs)[optimize_mask]
        .sum()
        .clamp_min(1.0)
    )
    policy_loss = (
        policy_terms[optimize_mask].sum() / denominator
        if optimize_mask.any()
        else current_log_probs.sum() * 0.0
    )

    kl_coefficients = torch.where(
        mature,
        reference_mean_kl.new_tensor(mature_kl_weight),
        reference_mean_kl.new_tensor(regular_kl_weight),
    )
    kl_mask = pair_valid.unsqueeze(-1).expand_as(reference_mean_kl)
    weighted_kl = reference_mean_kl * kl_coefficients.unsqueeze(-1)
    reference_kl_loss = (
        weighted_kl[kl_mask].mean()
        if kl_mask.any()
        else reference_mean_kl.sum() * 0.0
    )
    total_loss = policy_loss + reference_kl_loss
    valid_total = pair_valid.float().sum().clamp_min(1.0)
    mature_valid = mature & pair_valid
    return {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "reference_kl_loss": reference_kl_loss,
        "advantages": advantages,
        "paired_delta_mean": (
            delta[pair_valid].mean().detach()
            if pair_valid.any() else delta.sum() * 0.0
        ),
        "paired_delta_std": (
            delta_std[group_valid].mean().detach()
            if group_valid.any() else delta.sum() * 0.0
        ),
        "mature_delta_mean": (
            delta[mature_valid].mean().detach()
            if mature_valid.any() else delta.sum() * 0.0
        ),
        "policy_active_scene_fraction": optimize_mode.any(dim=-1).float().mean().detach(),
        "positive_advantage_fraction": (
            ((advantages > 0) & pair_valid).float().sum() / valid_total
        ).detach(),
        "negative_advantage_fraction": (
            ((advantages < 0) & pair_valid).float().sum() / valid_total
        ).detach(),
        "safety_override_fraction": (
            safety_regression.float().sum() / valid_total
        ).detach(),
        "mature_negative_fraction": (
            mature_negative.float().sum() / valid_total
        ).detach(),
        "bootstrap_advantage_fraction": (
            bootstrap_mask.float().sum() / valid_total
        ).detach(),
        "catastrophic_fraction": (
            ((delta <= -0.5) & pair_valid).float().sum() / valid_total
        ).detach(),
        "mean_scene_weight": scene_weights.mean().detach(),
        "mean_exact_kl": (
            reference_mean_kl[kl_mask].mean().detach()
            if kl_mask.any() else reference_mean_kl.sum() * 0.0
        ),
        "discount_first": discounts[0].detach(),
        "discount_last": discounts[-1].detach(),
        "mean_current_log_prob": current_log_probs.mean().detach(),
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
    if training_mode == "stage25_relative_harm_selector":
        if not predictions.get("grpo_training_rollout", True):
            zero = current_logits.sum() * 0.0
            return {"loss": zero, "stage25_selector_loss": zero.detach()}
        result = compute_stage25_selector_loss(
            predictions,
            focal_gamma=float(getattr(
                config, "stage25_selector_focal_gamma", 2.0
            )),
            hard_negative_count=int(getattr(
                config, "stage25_selector_hard_negative_count", 4
            )),
            hard_risk_margin=float(getattr(
                config, "stage25_selector_hard_risk_margin", 0.9
            )),
        )
        result["stage25_selector_loss"] = result["loss"].detach()
        return result
    if training_mode == "stage24_selector":
        if not predictions.get("grpo_training_rollout", True):
            zero = current_logits.sum() * 0.0
            return {"loss": zero, "stage24_selector_loss": zero.detach()}
        result = compute_stage24_selector_loss(
            predictions,
            reward_gap=float(getattr(
                config, "stage24_selector_pair_reward_gap", 0.005
            )),
            focal_gamma=float(getattr(
                config, "stage24_selector_focal_gamma", 2.0
            )),
            hard_negative_count=int(getattr(
                config, "stage24_selector_hard_negative_count", 4
            )),
        )
        result["stage24_selector_loss"] = result["loss"].detach()
        return result
    if training_mode == "stage23_selector":
        if not predictions.get("grpo_training_rollout", True):
            zero = current_logits.sum() * 0.0
            return {"loss": zero, "stage23_selector_loss": zero.detach()}
        result = compute_stage23_selector_loss(
            predictions,
            reward_gap=float(
                getattr(config, "stage23_selector_pair_reward_gap", 0.01)
            ),
            focal_gamma=float(
                getattr(config, "stage23_selector_focal_gamma", 2.0)
            ),
            unsafe_positive_weight=float(
                getattr(config, "stage23_selector_unsafe_positive_weight", 50.0)
            ),
        )
        result["stage23_selector_loss"] = result["loss"].detach()
        return result
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

    if training_mode == "stage36_reference_gated_tail_ncd_grpo":
        return compute_stage36_transfuser_loss(predictions, config)
    if training_mode == "stage35_nested_counterfactual_deployment_grpo":
        return compute_stage35_transfuser_loss(predictions, config)

    if training_mode == "stage34_mode_aligned_frontier_grpo":
        return compute_stage34_transfuser_loss(predictions, config)

    if training_mode in {
        "stage31_public_deployed_pair",
        "stage31_public_deployed_frontier",
        "stage32_public_deployed_extended",
        "stage33_cdc_grpo",
    }:
        required = (
            "diffgrpo_current_log_probs", "diffgrpo_bc_log_probs",
            "diffgrpo_reference_mean_kl", "raw_rewards", "reward_valid_mask",
            "component_scores", "diffgrpo_base_rewards",
            "diffgrpo_base_valid_mask", "diffgrpo_base_component_scores",
            "stage31_current_selected_modes", "stage31_public_selected_modes",
        )
        missing = [name for name in required if predictions.get(name) is None]
        if missing:
            raise ValueError(f"Missing Stage31 deployed-GRPO tensors: {missing}")
        objective_kwargs = dict(
            current_log_probs=predictions["diffgrpo_current_log_probs"],
            bc_log_probs=predictions["diffgrpo_bc_log_probs"],
            reference_mean_kl=predictions["diffgrpo_reference_mean_kl"],
            rewards=predictions["raw_rewards"],
            valid_mask=predictions["reward_valid_mask"],
            component_scores=predictions["component_scores"],
            base_rewards=predictions["diffgrpo_base_rewards"],
            base_valid_mask=predictions["diffgrpo_base_valid_mask"],
            base_component_scores=predictions[
                "diffgrpo_base_component_scores"
            ],
            use_frontier_rank=(
                training_mode in {
                    "stage31_public_deployed_frontier",
                    "stage32_public_deployed_extended",
                }
            ),
            advantage_eps=float(getattr(config, "grpo_advantage_eps", 1e-3)),
            headroom_low=float(getattr(config, "stage31_headroom_low", 0.75)),
            headroom_high=float(getattr(config, "stage31_headroom_high", 0.90)),
            delta_scale_floor=float(getattr(
                config, "stage31_delta_scale_floor", 0.002
            )),
            rank_weight=float(getattr(config, "stage31_rank_weight", 0.5)),
            advantage_clip=float(getattr(
                config, "diffgrpo_base_advantage_clip", 2.0
            )),
            safety_tolerance=float(getattr(
                config, "diffgrpo_safety_regression_tolerance", 1e-6
            )),
            step_discount=float(getattr(config, "diffgrpo_step_discount", 0.6)),
            bc_weight=float(getattr(config, "stage31_bc_weight", 0.1)),
            kl_weight=float(getattr(config, "stage31_kl_weight", 0.1)),
            safety_kl_weight=float(getattr(
                config, "stage31_safety_kl_weight", 0.5
            )),
        )
        if training_mode == "stage33_cdc_grpo":
            objective_kwargs.pop("use_frontier_rank")
            objective_kwargs.pop("advantage_clip")
            objective = compute_stage33_cdc_smoke_objective(
                **objective_kwargs,
                deployment_weight=float(getattr(config, "stage33_deployment_weight", 0.5)),
                headroom_weight=float(getattr(config, "stage33_headroom_weight", 0.5)),
                headroom_margin=float(getattr(config, "stage33_headroom_margin", 0.001)),
                advantage_clip=float(getattr(config, "stage33_advantage_clip", 2.0)),
            )
        else:
            objective = compute_deployed_decision_grpo_objective(**objective_kwargs)
        result = {
            "loss": objective["loss"],
            "generation_grpo_loss": objective["policy_loss"],
            "diffgrpo_bc_loss": objective["bc_loss"],
            "generation_reference_kl_loss": objective["reference_kl_loss"],
            "diffgrpo_mean_current_log_prob": objective[
                "mean_current_log_prob"
            ],
            "raw_reward_mean": objective["reward_mean"],
            "raw_reward_std": objective["reward_std"],
        }
        for name in (
            "advantage_mean", "deployment_delta_mean", "hard_delta_mean",
            "transition_delta_mean", "mature_delta_mean", "headroom_mean",
            "hard_scene_fraction", "transition_scene_fraction",
            "mature_scene_fraction", "positive_fraction", "negative_fraction",
            "neutral_fraction", "tie_fraction",
            "policy_active_scene_fraction", "safety_override_fraction",
            "catastrophic_fraction", "mean_bc_weight", "mean_kl_weight",
            "mean_exact_kl", "delta_scale_mean", "frontier_rank_enabled",
        ):
            result[f"stage31_{name}"] = objective[name]
        result["stage31_selector_mode_disagreement"] = predictions[
            "stage31_selector_mode_disagreement"
        ]
        result["stage31_current_selector_switch_rate"] = predictions[
            "stage31_current_selector_switch_rate"
        ]
        result["stage31_public_selector_switch_rate"] = predictions[
            "stage31_public_selector_switch_rate"
        ]
        if training_mode == "stage33_cdc_grpo":
            result["stage33_deployment_credit_mean"] = objective[
                "stage33_deployment_credit_mean"
            ]
            result["stage33_headroom_credit_mean"] = objective[
                "stage33_headroom_credit_mean"
            ]
            result["stage33_candidate_level_trace"] = objective[
                "stage33_candidate_level_trace"
            ]
        result["diffgrpo_discount_first"] = objective["discount_first"]
        result["diffgrpo_discount_last"] = objective["discount_last"]
        return result

    if training_mode == "stage32_selector_aware_frontier":
        required = (
            "diffgrpo_current_log_probs", "diffgrpo_bc_log_probs",
            "diffgrpo_reference_mean_kl", "raw_rewards", "reward_valid_mask",
            "component_scores", "diffgrpo_base_rewards",
            "diffgrpo_base_valid_mask", "diffgrpo_base_component_scores",
            "stage32_current_mode_valid", "stage32_public_mode_valid",
            "stage32_pool_size", "diffgrpo_group_size",
        )
        missing = [name for name in required if predictions.get(name) is None]
        if missing:
            raise ValueError(f"Missing Stage32 frontier tensors: {missing}")
        objective = compute_selector_aware_frontier_objective(
            current_log_probs=predictions["diffgrpo_current_log_probs"],
            bc_log_probs=predictions["diffgrpo_bc_log_probs"],
            reference_mean_kl=predictions["diffgrpo_reference_mean_kl"],
            rewards=predictions["raw_rewards"],
            valid_mask=predictions["reward_valid_mask"],
            component_scores=predictions["component_scores"],
            base_rewards=predictions["diffgrpo_base_rewards"],
            base_valid_mask=predictions["diffgrpo_base_valid_mask"],
            base_component_scores=predictions["diffgrpo_base_component_scores"],
            current_mode_valid=predictions["stage32_current_mode_valid"],
            public_mode_valid=predictions["stage32_public_mode_valid"],
            group_size=int(predictions["diffgrpo_group_size"].item()),
            pool_size=int(predictions["stage32_pool_size"].item()),
            advantage_eps=float(getattr(config, "grpo_advantage_eps", 1e-3)),
            headroom_low=float(getattr(config, "stage31_headroom_low", 0.75)),
            headroom_high=float(getattr(config, "stage31_headroom_high", 0.90)),
            delta_scale_floor=float(getattr(config, "stage31_delta_scale_floor", 0.002)),
            owner_margin=float(getattr(config, "stage32_frontier_owner_margin", 0.001)),
            frontier_weight=float(getattr(config, "stage32_frontier_weight", 0.5)),
            mature_cap=float(getattr(config, "stage32_frontier_mature_cap", 0.25)),
            advantage_clip=float(getattr(config, "diffgrpo_base_advantage_clip", 2.0)),
            safety_tolerance=float(getattr(config, "diffgrpo_safety_regression_tolerance", 1e-6)),
            step_discount=float(getattr(config, "diffgrpo_step_discount", 0.6)),
            bc_weight=float(getattr(config, "stage31_bc_weight", 0.1)),
            kl_weight=float(getattr(config, "stage31_kl_weight", 0.1)),
            safety_kl_weight=float(getattr(config, "stage31_safety_kl_weight", 0.5)),
        )
        result = {
            "loss": objective["loss"],
            "generation_grpo_loss": objective["policy_loss"],
            "diffgrpo_bc_loss": objective["bc_loss"],
            "generation_reference_kl_loss": objective["reference_kl_loss"],
            "diffgrpo_mean_current_log_prob": objective["mean_current_log_prob"],
            "raw_reward_mean": objective["reward_mean"],
            "raw_reward_std": objective["reward_std"],
        }
        for name in (
            "advantage_mean", "deployment_delta_mean", "hard_delta_mean",
            "transition_delta_mean", "mature_delta_mean", "frontier_delta_mean",
            "frontier_candidate_gain", "frontier_owner_fraction",
            "frontier_candidate_owner_fraction", "frontier_pool_valid_fraction",
            "positive_fraction", "negative_fraction", "neutral_fraction",
            "policy_active_scene_fraction", "safety_override_fraction",
            "catastrophic_fraction", "mean_bc_weight", "mean_kl_weight",
            "mean_exact_kl", "delta_scale_mean", "frontier_rank_enabled",
        ):
            result[f"stage32_{name}"] = objective[name]
        result["stage32_selector_mode_disagreement"] = predictions[
            "stage32_selector_mode_disagreement"
        ]
        result["stage32_current_selector_switch_rate"] = predictions[
            "stage32_current_selector_switch_rate"
        ]
        result["stage32_public_selector_switch_rate"] = predictions[
            "stage32_public_selector_switch_rate"
        ]
        result["diffgrpo_discount_first"] = objective["discount_first"]
        result["diffgrpo_discount_last"] = objective["discount_last"]
        return result

    if training_mode in {
        "stage30_public_mode_coverage",
        "stage30_public_mode_coverage_constrained",
    }:
        required = (
            "diffgrpo_current_log_probs", "diffgrpo_bc_log_probs",
            "diffgrpo_reference_mean_kl", "raw_rewards", "reward_valid_mask",
            "component_scores", "diffgrpo_base_rewards",
            "diffgrpo_base_valid_mask", "diffgrpo_base_component_scores",
            "stage30_scene_buckets",
        )
        missing = [name for name in required if predictions.get(name) is None]
        if missing:
            raise ValueError(f"Missing Stage30 mode-coverage tensors: {missing}")
        objective = compute_mode_coverage_constrained_objective(
            current_log_probs=predictions["diffgrpo_current_log_probs"],
            bc_log_probs=predictions["diffgrpo_bc_log_probs"],
            reference_mean_kl=predictions["diffgrpo_reference_mean_kl"],
            rewards=predictions["raw_rewards"],
            valid_mask=predictions["reward_valid_mask"],
            component_scores=predictions["component_scores"],
            base_rewards=predictions["diffgrpo_base_rewards"],
            base_valid_mask=predictions["diffgrpo_base_valid_mask"],
            base_component_scores=predictions["diffgrpo_base_component_scores"],
            scene_buckets=predictions["stage30_scene_buckets"],
            constrained=(
                training_mode == "stage30_public_mode_coverage_constrained"
            ),
            advantage_eps=float(getattr(config, "grpo_advantage_eps", 1e-3)),
            delta_scale_floor=float(getattr(config, "stage30_delta_scale_floor", 0.002)),
            advantage_clip=float(getattr(config, "diffgrpo_base_advantage_clip", 2.0)),
            top_k=int(getattr(config, "stage30_top_k", 5)),
            coverage_top_weight=float(getattr(config, "stage30_coverage_top_weight", 0.5)),
            boundary_top_weight=float(getattr(config, "stage30_boundary_top_weight", 0.25)),
            boundary_positive_margin=float(getattr(config, "stage30_boundary_positive_margin", 0.001)),
            boundary_negative_multiplier=float(getattr(config, "stage30_boundary_negative_multiplier", 2.0)),
            mature_negative_floor=float(getattr(config, "stage30_mature_negative_floor", -0.0002)),
            mature_positive_margin=float(getattr(config, "stage30_mature_positive_margin", 0.002)),
            mature_negative_multiplier=float(getattr(config, "stage30_mature_negative_multiplier", 4.0)),
            safety_tolerance=float(getattr(config, "diffgrpo_safety_regression_tolerance", 1e-6)),
            component_tolerance=float(getattr(config, "stage30_component_tolerance", 1e-6)),
            step_discount=float(getattr(config, "diffgrpo_step_discount", 0.6)),
        )
        result = {
            "loss": objective["loss"],
            "generation_grpo_loss": objective["policy_loss"],
            "diffgrpo_bc_loss": objective["bc_loss"],
            "generation_reference_kl_loss": objective["reference_kl_loss"],
            "diffgrpo_mean_current_log_prob": objective["mean_current_log_prob"],
            "raw_reward_mean": objective["reward_mean"],
            "raw_reward_std": objective["reward_std"],
        }
        for name in (
            "advantage_mean", "base_delta_mean", "severe_delta_mean",
            "recoverable_delta_mean", "boundary_delta_mean",
            "mature_delta_mean", "positive_fraction", "negative_fraction",
            "neutral_fraction", "policy_active_scene_fraction",
            "top_tail_fraction", "safety_override_fraction",
            "catastrophic_fraction", "mean_bc_weight", "mean_kl_weight",
            "mean_exact_kl", "paired_scale_mean",
        ):
            result[f"stage30_{name}"] = objective[name]
        result["diffgrpo_discount_first"] = objective["discount_first"]
        result["diffgrpo_discount_last"] = objective["discount_last"]
        return result

    if training_mode in {
        "stage29_public_headroom_hybrid",
        "stage29_public_headroom_conditional",
    }:
        required = (
            "diffgrpo_current_log_probs", "diffgrpo_bc_log_probs",
            "diffgrpo_reference_mean_kl", "raw_rewards", "reward_valid_mask",
            "component_scores", "diffgrpo_base_rewards",
            "diffgrpo_base_valid_mask", "diffgrpo_base_component_scores",
        )
        missing = [name for name in required if predictions.get(name) is None]
        if missing:
            raise ValueError(f"Missing Stage29 headroom-GRPO tensors: {missing}")
        objective = compute_reference_anchored_headroom_objective(
            current_log_probs=predictions["diffgrpo_current_log_probs"],
            bc_log_probs=predictions["diffgrpo_bc_log_probs"],
            reference_mean_kl=predictions["diffgrpo_reference_mean_kl"],
            rewards=predictions["raw_rewards"],
            valid_mask=predictions["reward_valid_mask"],
            component_scores=predictions["component_scores"],
            base_rewards=predictions["diffgrpo_base_rewards"],
            base_valid_mask=predictions["diffgrpo_base_valid_mask"],
            base_component_scores=predictions[
                "diffgrpo_base_component_scores"
            ],
            advantage_eps=float(getattr(config, "grpo_advantage_eps", 1e-3)),
            headroom_low=float(getattr(config, "stage29_headroom_low", 0.75)),
            headroom_high=float(getattr(config, "stage29_headroom_high", 0.90)),
            delta_scale_floor=float(getattr(
                config, "stage29_delta_scale_floor", 0.002
            )),
            rank_weight=float(getattr(config, "stage29_rank_weight", 0.5)),
            advantage_clip=float(getattr(
                config, "diffgrpo_base_advantage_clip", 2.0
            )),
            safety_tolerance=float(getattr(
                config, "diffgrpo_safety_regression_tolerance", 1e-6
            )),
            step_discount=float(getattr(config, "diffgrpo_step_discount", 0.6)),
            conditional_regularization=bool(getattr(
                config, "stage29_conditional_regularization", False
            )),
            fixed_bc_weight=float(getattr(
                config, "stage29_fixed_bc_weight", 0.1
            )),
            fixed_kl_weight=float(getattr(
                config, "stage29_fixed_kl_weight", 0.1
            )),
            hard_bc_weight=float(getattr(
                config, "stage29_hard_bc_weight", 0.05
            )),
            mature_bc_weight=float(getattr(
                config, "stage29_mature_bc_weight", 0.1
            )),
            hard_kl_weight=float(getattr(
                config, "stage29_hard_kl_weight", 0.05
            )),
            mature_kl_weight=float(getattr(
                config, "stage29_mature_kl_weight", 0.5
            )),
            safety_kl_weight=float(getattr(
                config, "stage29_safety_kl_weight", 0.5
            )),
        )
        return {
            "loss": objective["loss"],
            "generation_grpo_loss": objective["policy_loss"],
            "diffgrpo_bc_loss": objective["bc_loss"],
            "generation_reference_kl_loss": objective["reference_kl_loss"],
            "stage29_advantage_mean": objective["advantage_mean"],
            "diffgrpo_mean_current_log_prob": objective["mean_current_log_prob"],
            "raw_reward_mean": objective["reward_mean"],
            "raw_reward_std": objective["reward_std"],
            "stage29_base_delta_mean": objective["base_delta_mean"],
            "stage29_hard_delta_mean": objective["hard_delta_mean"],
            "stage29_transition_delta_mean": objective["transition_delta_mean"],
            "stage29_mature_delta_mean": objective["mature_delta_mean"],
            "stage29_headroom_mean": objective["headroom_mean"],
            "stage29_hard_scene_fraction": objective["hard_scene_fraction"],
            "stage29_transition_scene_fraction": objective[
                "transition_scene_fraction"
            ],
            "stage29_mature_scene_fraction": objective["mature_scene_fraction"],
            "stage29_positive_fraction": objective["positive_fraction"],
            "stage29_negative_fraction": objective["negative_fraction"],
            "stage29_neutral_fraction": objective["neutral_fraction"],
            "stage29_policy_active_scene_fraction": objective[
                "policy_active_scene_fraction"
            ],
            "stage29_safety_override_fraction": objective[
                "safety_override_fraction"
            ],
            "stage29_catastrophic_fraction": objective["catastrophic_fraction"],
            "stage29_mean_bc_weight": objective["mean_bc_weight"],
            "stage29_mean_kl_weight": objective["mean_kl_weight"],
            "stage29_mean_exact_kl": objective["mean_exact_kl"],
            "stage29_paired_scale_mean": objective["paired_scale_mean"],
            "diffgrpo_discount_first": objective["discount_first"],
            "diffgrpo_discount_last": objective["discount_last"],
        }

    if training_mode in {
        "stage28_public_paired_uplift_multi",
        "stage28_public_paired_uplift_explore",
    }:
        required = (
            "diffgrpo_current_log_probs", "diffgrpo_bc_log_probs",
            "diffgrpo_reference_mean_kl", "raw_rewards", "reward_valid_mask",
            "component_scores", "diffgrpo_base_rewards",
            "diffgrpo_base_valid_mask", "diffgrpo_base_component_scores",
        )
        missing = [name for name in required if predictions.get(name) is None]
        if missing:
            raise ValueError(f"Missing Stage28 paired-uplift tensors: {missing}")
        objective = compute_public_paired_uplift_objective(
            current_log_probs=predictions["diffgrpo_current_log_probs"],
            bc_log_probs=predictions["diffgrpo_bc_log_probs"],
            reference_mean_kl=predictions["diffgrpo_reference_mean_kl"],
            rewards=predictions["raw_rewards"],
            valid_mask=predictions["reward_valid_mask"],
            component_scores=predictions["component_scores"],
            base_rewards=predictions["diffgrpo_base_rewards"],
            base_valid_mask=predictions["diffgrpo_base_valid_mask"],
            base_component_scores=predictions[
                "diffgrpo_base_component_scores"
            ],
            advantage_eps=float(getattr(config, "grpo_advantage_eps", 1e-3)),
            positive_margin=float(getattr(
                config, "diffgrpo_paired_positive_margin", 0.002
            )),
            negative_margin=float(getattr(
                config, "diffgrpo_paired_negative_margin", 0.002
            )),
            mature_negative_margin=float(getattr(
                config, "diffgrpo_paired_mature_negative_margin", 0.0005
            )),
            mature_reward_threshold=float(getattr(
                config, "diffgrpo_paired_mature_reward_threshold", 0.75
            )),
            evidence_scale=float(getattr(config, "diffgrpo_base_scale", 0.1)),
            advantage_clip=float(getattr(
                config, "diffgrpo_base_advantage_clip", 2.0
            )),
            safety_tolerance=float(getattr(
                config, "diffgrpo_safety_regression_tolerance", 1e-6
            )),
            step_discount=float(getattr(config, "diffgrpo_step_discount", 0.6)),
            bc_weight=float(getattr(config, "diffgrpo_bc_weight", 0.1)),
            regular_kl_weight=float(getattr(
                config, "diffgrpo_paired_regular_kl_weight", 0.1
            )),
            mature_kl_weight=float(getattr(
                config, "diffgrpo_paired_mature_kl_weight", 0.5
            )),
        )
        return {
            "loss": objective["loss"],
            "generation_grpo_loss": objective["policy_loss"],
            "diffgrpo_bc_loss": objective["bc_loss"],
            "generation_reference_kl_loss": objective["reference_kl_loss"],
            "stage28_advantage_mean": objective["advantage_mean"],
            "diffgrpo_mean_current_log_prob": objective[
                "mean_current_log_prob"
            ],
            "raw_reward_mean": objective["reward_mean"],
            "raw_reward_std": objective["reward_std"],
            "stage28_base_delta_mean": objective["base_delta_mean"],
            "stage28_positive_fraction": objective["positive_fraction"],
            "stage28_regular_negative_fraction": objective[
                "regular_negative_fraction"
            ],
            "stage28_mature_negative_fraction": objective[
                "mature_negative_fraction"
            ],
            "stage28_safety_override_fraction": objective[
                "safety_override_fraction"
            ],
            "stage28_neutral_fraction": objective["neutral_fraction"],
            "stage28_policy_active_scene_fraction": objective[
                "policy_active_scene_fraction"
            ],
            "stage28_catastrophic_fraction": objective[
                "catastrophic_fraction"
            ],
            "stage28_mean_exact_kl": objective["mean_exact_kl"],
            "diffgrpo_discount_first": objective["discount_first"],
            "diffgrpo_discount_last": objective["discount_last"],
        }

    if training_mode in {
        "diffgrpo_selected_set",
        "stage27_public_diffgrpo_selected_set",
    }:
        required = (
            "diffgrpo_current_log_probs", "diffgrpo_bc_log_probs",
            "diffgrpo_reference_mean_kl", "raw_rewards", "reward_valid_mask",
            "component_scores", "diffgrpo_base_rewards",
            "diffgrpo_base_valid_mask", "diffgrpo_base_component_scores",
        )
        missing = [name for name in required if predictions.get(name) is None]
        if missing:
            raise ValueError(f"Missing Stage23 selected-set tensors: {missing}")
        objective = compute_selected_set_diffgrpo_objective(
            current_log_probs=predictions["diffgrpo_current_log_probs"],
            bc_log_probs=predictions["diffgrpo_bc_log_probs"],
            reference_mean_kl=predictions["diffgrpo_reference_mean_kl"],
            rewards=predictions["raw_rewards"],
            valid_mask=predictions["reward_valid_mask"],
            component_scores=predictions["component_scores"],
            base_rewards=predictions["diffgrpo_base_rewards"],
            base_valid_mask=predictions["diffgrpo_base_valid_mask"],
            base_component_scores=predictions["diffgrpo_base_component_scores"],
            advantage_eps=float(getattr(config, "grpo_advantage_eps", 1e-3)),
            safety_tolerance=float(
                getattr(config, "diffgrpo_safety_regression_tolerance", 1e-6)
            ),
            step_discount=float(getattr(config, "diffgrpo_step_discount", 0.6)),
            bc_weight=float(getattr(config, "diffgrpo_bc_weight", 0.1)),
            regular_kl_weight=float(
                getattr(config, "diffgrpo_paired_regular_kl_weight", 0.1)
            ),
            mature_kl_weight=float(
                getattr(config, "diffgrpo_paired_mature_kl_weight", 0.5)
            ),
        )
        return {
            "loss": objective["loss"],
            "generation_grpo_loss": objective["policy_loss"],
            "diffgrpo_bc_loss": objective["bc_loss"],
            "generation_reference_kl_loss": objective["reference_kl_loss"],
            "stage23_advantage_mean": objective["advantage_mean"],
            "diffgrpo_mean_current_log_prob": objective[
                "mean_current_log_prob"
            ],
            "raw_reward_mean": objective["reward_mean"],
            "raw_reward_std": objective["reward_std"],
            "stage23_base_delta_mean": objective["base_delta_mean"],
            "stage23_safety_override_fraction": objective["safety_override_fraction"],
            "stage23_base_negative_fraction": objective["base_negative_fraction"],
            "stage23_mature_negative_fraction": objective["mature_negative_fraction"],
            "stage23_catastrophic_fraction": objective["catastrophic_fraction"],
            "stage23_mean_exact_kl": objective["mean_exact_kl"],
            "diffgrpo_discount_first": objective["discount_first"],
            "diffgrpo_discount_last": objective["discount_last"],
        }

    if training_mode == "diffgrpo_paired_residual":
        required = (
            "diffgrpo_current_log_probs",
            "diffgrpo_reference_mean_kl",
            "raw_rewards",
            "reward_valid_mask",
            "component_scores",
            "diffgrpo_base_rewards",
            "diffgrpo_base_valid_mask",
            "diffgrpo_base_component_scores",
            "diffgrpo_scene_weights",
        )
        missing = [name for name in required if predictions.get(name) is None]
        if missing:
            raise ValueError(f"Missing Stage22 DiffGRPO tensors: {missing}")
        objective = compute_paired_residual_diffgrpo_objective(
            current_log_probs=predictions["diffgrpo_current_log_probs"],
            reference_mean_kl=predictions["diffgrpo_reference_mean_kl"],
            rewards=predictions["raw_rewards"],
            valid_mask=predictions["reward_valid_mask"],
            component_scores=predictions["component_scores"],
            base_rewards=predictions["diffgrpo_base_rewards"],
            base_valid_mask=predictions["diffgrpo_base_valid_mask"],
            base_component_scores=predictions[
                "diffgrpo_base_component_scores"
            ],
            scene_weights=predictions["diffgrpo_scene_weights"],
            advantage_eps=float(getattr(config, "grpo_advantage_eps", 1e-3)),
            positive_margin=float(
                getattr(config, "diffgrpo_paired_positive_margin", 0.01)
            ),
            negative_margin=float(
                getattr(config, "diffgrpo_paired_negative_margin", 0.01)
            ),
            mature_negative_margin=float(
                getattr(config, "diffgrpo_paired_mature_negative_margin", 0.002)
            ),
            mature_reward_threshold=float(
                getattr(config, "diffgrpo_paired_mature_reward_threshold", 0.75)
            ),
            scale=float(getattr(config, "diffgrpo_base_scale", 0.10)),
            advantage_clip=float(
                getattr(config, "diffgrpo_base_advantage_clip", 2.0)
            ),
            mature_negative_multiplier=float(
                getattr(config, "diffgrpo_paired_mature_negative_multiplier", 2.0)
            ),
            safety_tolerance=float(
                getattr(config, "diffgrpo_safety_regression_tolerance", 1e-6)
            ),
            regular_kl_weight=float(
                getattr(config, "diffgrpo_paired_regular_kl_weight", 0.1)
            ),
            mature_kl_weight=float(
                getattr(config, "diffgrpo_paired_mature_kl_weight", 0.5)
            ),
            bootstrap_advantage_weight=float(
                getattr(
                    config,
                    "diffgrpo_paired_bootstrap_advantage_weight",
                    0.25,
                )
            ),
            step_discount=float(getattr(config, "diffgrpo_step_discount", 0.6)),
        )
        return {
            "loss": objective["loss"],
            "generation_grpo_loss": objective["policy_loss"],
            "stage22_exact_kl_loss": objective["reference_kl_loss"],
            "stage22_paired_delta_mean": objective["paired_delta_mean"],
            "stage22_paired_delta_std": objective["paired_delta_std"],
            "stage22_mature_delta_mean": objective["mature_delta_mean"],
            "generation_policy_active_scene_fraction": objective[
                "policy_active_scene_fraction"
            ],
            "generation_positive_advantage_fraction": objective[
                "positive_advantage_fraction"
            ],
            "generation_negative_advantage_fraction": objective[
                "negative_advantage_fraction"
            ],
            "stage22_safety_override_fraction": objective[
                "safety_override_fraction"
            ],
            "stage22_mature_negative_fraction": objective[
                "mature_negative_fraction"
            ],
            "stage22_bootstrap_advantage_fraction": objective[
                "bootstrap_advantage_fraction"
            ],
            "stage22_catastrophic_fraction": objective[
                "catastrophic_fraction"
            ],
            "stage22_mean_scene_weight": objective["mean_scene_weight"],
            "stage22_mean_exact_kl": objective["mean_exact_kl"],
            "diffgrpo_discount_first": objective["discount_first"],
            "diffgrpo_discount_last": objective["discount_last"],
            "diffgrpo_mean_current_log_prob": objective[
                "mean_current_log_prob"
            ],
        }

    if training_mode == "diffgrpo_selected_anchor_base_preserve":
        required = (
            "diffgrpo_current_log_probs",
            "diffgrpo_bc_log_probs",
            "raw_rewards",
            "reward_valid_mask",
            "component_scores",
            "diffgrpo_base_rewards",
            "diffgrpo_base_safety",
            "diffgrpo_scene_weights",
            "diffgrpo_bc_weights",
        )
        missing = [name for name in required if predictions.get(name) is None]
        if missing:
            raise ValueError(f"Missing Stage21 DiffGRPO tensors: {missing}")
        objective = compute_base_preserve_diffgrpo_objective(
            current_log_probs=predictions["diffgrpo_current_log_probs"],
            bc_log_probs=predictions["diffgrpo_bc_log_probs"],
            rewards=predictions["raw_rewards"],
            valid_mask=predictions["reward_valid_mask"],
            component_scores=predictions["component_scores"],
            base_rewards=predictions["diffgrpo_base_rewards"],
            base_safety=predictions["diffgrpo_base_safety"],
            scene_weights=predictions["diffgrpo_scene_weights"],
            bc_weights=predictions["diffgrpo_bc_weights"],
            advantage_eps=float(getattr(config, "grpo_advantage_eps", 1e-3)),
            margin=float(getattr(config, "diffgrpo_base_margin", 0.01)),
            scale=float(getattr(config, "diffgrpo_base_scale", 0.10)),
            advantage_clip=float(getattr(config, "diffgrpo_base_advantage_clip", 2.0)),
            safety_tolerance=float(
                getattr(config, "diffgrpo_safety_regression_tolerance", 1e-6)
            ),
            step_discount=float(getattr(config, "diffgrpo_step_discount", 0.6)),
        )
        return {
            "loss": objective["loss"],
            "generation_grpo_loss": objective["policy_loss"],
            "diffgrpo_bc_loss": objective["bc_loss"],
            "raw_reward_mean": objective["reward_mean"],
            "raw_reward_std": objective["reward_std"],
            "generation_policy_active_scene_fraction": objective["policy_active_scene_fraction"],
            "generation_positive_advantage_fraction": objective["positive_advantage_fraction"],
            "generation_negative_advantage_fraction": objective["negative_advantage_fraction"],
            "stage21_safety_override_fraction": objective["safety_override_fraction"],
            "stage21_base_delta_mean": objective["base_delta_mean"],
            "stage21_mean_scene_weight": objective["mean_scene_weight"],
            "stage21_mean_bc_weight": objective["mean_bc_weight"],
            "diffgrpo_discount_first": objective["discount_first"],
            "diffgrpo_discount_last": objective["discount_last"],
            "diffgrpo_mean_current_log_prob": objective["mean_current_log_prob"],
            "diffgrpo_mean_bc_log_prob": objective["mean_bc_log_prob"],
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
