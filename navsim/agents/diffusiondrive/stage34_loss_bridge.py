"""Loss bridge for the Stage34 mode-aligned frontier objective."""

from __future__ import annotations

from typing import Any, Dict

import torch

from navsim.agents.diffusiondrive.stage34_mode_aligned_frontier import (
    compute_mode_aligned_frontier_objective,
)


def compute_stage34_transfuser_loss(
    predictions: Dict[str, torch.Tensor],
    config: Any,
) -> Dict[str, torch.Tensor]:
    """Validate the Stage34 trace and expose stable Lightning metric names."""
    required = (
        "diffgrpo_current_log_probs",
        "diffgrpo_bc_log_probs",
        "diffgrpo_reference_mean_kl",
        "raw_rewards",
        "reward_valid_mask",
        "component_scores",
        "diffgrpo_base_rewards",
        "diffgrpo_base_valid_mask",
        "diffgrpo_base_component_scores",
        "stage34_current_selected_modes",
        "stage34_public_selected_modes",
        "stage34_current_eligible_mask",
        "stage34_public_eligible_mask",
        "stage34_scene_buckets",
        "stage34_sampled_chain_count",
        "stage34_replayed_chain_count",
        "stage34_mode_index_alignment",
        "stage34_candidate_level_trace",
        "diffgrpo_group_size",
    )
    missing = [name for name in required if predictions.get(name) is None]
    if missing:
        raise ValueError(f"Missing Stage34 mode-aligned tensors: {missing}")

    group_size = int(predictions["diffgrpo_group_size"].detach().item())
    sampled = int(predictions["stage34_sampled_chain_count"].detach().item())
    replayed = int(predictions["stage34_replayed_chain_count"].detach().item())
    if group_size != 20 or sampled != 20 or replayed != 20:
        raise ValueError(
            "Stage34 requires exactly 20 sampled and 20 replayed aligned chains"
        )
    for name in ("stage34_mode_index_alignment", "stage34_candidate_level_trace"):
        value = float(predictions[name].detach().item())
        if value != 1.0:
            raise ValueError(f"{name} must be exactly 1")

    objective = compute_mode_aligned_frontier_objective(
        current_log_probs=predictions["diffgrpo_current_log_probs"],
        bc_log_probs=predictions["diffgrpo_bc_log_probs"],
        reference_mean_kl=predictions["diffgrpo_reference_mean_kl"],
        rewards=predictions["raw_rewards"],
        valid_mask=predictions["reward_valid_mask"],
        component_scores=predictions["component_scores"],
        base_rewards=predictions["diffgrpo_base_rewards"],
        base_valid_mask=predictions["diffgrpo_base_valid_mask"],
        base_component_scores=predictions["diffgrpo_base_component_scores"],
        current_selected_modes=predictions["stage34_current_selected_modes"],
        public_selected_modes=predictions["stage34_public_selected_modes"],
        current_eligible_mask=predictions["stage34_current_eligible_mask"],
        public_eligible_mask=predictions["stage34_public_eligible_mask"],
        scene_buckets=predictions["stage34_scene_buckets"],
        top_k=int(getattr(config, "stage34_top_k", 5)),
        headroom_k=int(getattr(config, "stage34_headroom_k", 2)),
        delta_scale_floor=float(
            getattr(config, "stage34_delta_scale_floor", 0.002)
        ),
        deployment_weight=float(
            getattr(config, "stage34_deployment_weight", 0.5)
        ),
        headroom_weight=float(getattr(config, "stage34_headroom_weight", 0.25)),
        headroom_margin=float(getattr(config, "stage34_headroom_margin", 0.001)),
        top5_negative_multiplier=float(
            getattr(config, "stage34_top5_negative_multiplier", 1.5)
        ),
        top1_negative_multiplier=float(
            getattr(config, "stage34_top1_negative_multiplier", 2.0)
        ),
        mature_positive_multiplier=float(
            getattr(config, "stage34_mature_positive_multiplier", 0.25)
        ),
        advantage_clip=float(getattr(config, "stage34_advantage_clip", 2.0)),
        safety_tolerance=float(
            getattr(config, "diffgrpo_safety_regression_tolerance", 1e-6)
        ),
        step_discount=float(getattr(config, "diffgrpo_step_discount", 0.6)),
        bc_weight=float(getattr(config, "stage34_bc_weight", 0.1)),
        kl_weight=float(getattr(config, "stage34_kl_weight", 0.1)),
        safety_kl_weight=float(
            getattr(config, "stage34_safety_kl_weight", 0.5)
        ),
    )

    reward_mask = (
        predictions["reward_valid_mask"].bool()
        & torch.isfinite(predictions["raw_rewards"])
    )
    valid_rewards = predictions["raw_rewards"][reward_mask].float()
    reward_mean = (
        valid_rewards.mean().detach()
        if valid_rewards.numel()
        else objective["loss"].detach() * 0.0
    )
    reward_std = (
        valid_rewards.std(unbiased=False).detach()
        if valid_rewards.numel()
        else objective["loss"].detach() * 0.0
    )
    result = {
        "loss": objective["loss"],
        "generation_grpo_loss": objective["policy_loss"],
        "diffgrpo_bc_loss": objective["bc_loss"],
        "generation_reference_kl_loss": objective["reference_kl_loss"],
        "diffgrpo_mean_current_log_prob": objective["mean_current_log_prob"],
        "raw_reward_mean": reward_mean,
        "raw_reward_std": reward_std,
    }
    for name in (
        "candidate_level_trace",
        "same_mode_delta_mean",
        "candidate_mean_delta",
        "deployment_delta_mean",
        "hard_selected_delta_mean",
        "mature_selected_delta_mean",
        "public_top1_delta_mean",
        "public_top5_delta_mean",
        "raw_oracle_delta_mean",
        "safe_deployable_oracle_delta_mean",
        "headroom_credit_mean",
        "active_mode_fraction",
        "positive_fraction",
        "negative_fraction",
        "safety_override_fraction",
        "catastrophic_fraction",
        "mean_bc_weight",
        "mean_kl_weight",
        "mean_exact_kl",
        "delta_scale_mean",
    ):
        result[f"stage34_{name}"] = objective[name]
    for name in (
        "stage34_selector_mode_disagreement",
        "stage34_current_selector_switch_rate",
        "stage34_public_selector_switch_rate",
        "stage34_mode_index_alignment",
        "stage34_sampled_chain_count",
        "stage34_replayed_chain_count",
    ):
        if name in predictions:
            result[name] = predictions[name]
    result["diffgrpo_discount_first"] = objective["discount_first"]
    result["diffgrpo_discount_last"] = objective["discount_last"]
    return result
