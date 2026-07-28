"""Loss bridge for Stage35 Nested Counterfactual Deployment GRPO."""

from __future__ import annotations

from typing import Any, Dict

import torch

from navsim.agents.diffusiondrive.stage35_counterfactual import (
    compute_nested_counterfactual_deployment_objective,
)


def _reshape_bank(
    value: torch.Tensor,
    *,
    batch: int,
    groups: int,
    modes: int,
    trailing: tuple[int, ...] = (),
) -> torch.Tensor:
    expected = (batch, groups * modes, *trailing)
    if value.shape != expected:
        raise ValueError(
            f"Stage35 bank tensor must have shape {expected}, got {value.shape}"
        )
    return value.reshape(batch, groups, modes, *trailing)


def compute_stage35_transfuser_loss(
    predictions: Dict[str, torch.Tensor],
    config: Any,
) -> Dict[str, torch.Tensor]:
    """Validate the candidate-level trace and expose stable metric names."""
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
        "stage35_current_selected_modes",
        "stage35_public_selected_modes",
        "stage35_active_modes",
        "stage35_active_valid",
        "stage35_hybrid_selected_modes",
        "stage35_donor_indices",
        "stage35_scene_buckets",
        "stage35_sampled_chain_count",
        "stage35_replayed_chain_count",
        "stage35_counterfactual_selector_count",
        "diffgrpo_group_size",
    )
    missing = [name for name in required if predictions.get(name) is None]
    if missing:
        raise ValueError(f"Missing Stage35 NCD tensors: {missing}")

    current_log_probs = predictions["diffgrpo_current_log_probs"]
    if current_log_probs.ndim != 4:
        raise ValueError("Stage35 current log probabilities must be [B,8,4,S]")
    batch, groups, active_width, _ = current_log_probs.shape
    modes = 20
    if (groups, active_width) != (8, 4):
        raise ValueError("Stage35 freezes G=8 and active width=4")
    group_size = int(predictions["diffgrpo_group_size"].detach().item())
    sampled = int(predictions["stage35_sampled_chain_count"].detach().item())
    replayed = int(predictions["stage35_replayed_chain_count"].detach().item())
    counterfactuals = int(
        predictions["stage35_counterfactual_selector_count"].detach().item()
    )
    if (
        group_size != groups
        or sampled != 2 * groups * modes
        or replayed != 2 * groups * active_width
        or counterfactuals != groups * active_width * (groups - 1)
    ):
        raise ValueError("Stage35 trace accounting constants have drifted")

    rewards = _reshape_bank(
        predictions["raw_rewards"],
        batch=batch,
        groups=groups,
        modes=modes,
    )
    valid_mask = _reshape_bank(
        predictions["reward_valid_mask"],
        batch=batch,
        groups=groups,
        modes=modes,
    )
    components = _reshape_bank(
        predictions["component_scores"],
        batch=batch,
        groups=groups,
        modes=modes,
        trailing=(6,),
    )
    base_rewards = _reshape_bank(
        predictions["diffgrpo_base_rewards"],
        batch=batch,
        groups=groups,
        modes=modes,
    )
    base_valid = _reshape_bank(
        predictions["diffgrpo_base_valid_mask"],
        batch=batch,
        groups=groups,
        modes=modes,
    )
    base_components = _reshape_bank(
        predictions["diffgrpo_base_component_scores"],
        batch=batch,
        groups=groups,
        modes=modes,
        trailing=(6,),
    )
    objective = compute_nested_counterfactual_deployment_objective(
        current_log_probs=current_log_probs,
        bc_log_probs=predictions["diffgrpo_bc_log_probs"],
        reference_mean_kl=predictions["diffgrpo_reference_mean_kl"],
        rewards=rewards,
        valid_mask=valid_mask,
        component_scores=components,
        base_rewards=base_rewards,
        base_valid_mask=base_valid,
        base_component_scores=base_components,
        current_selected_modes=predictions["stage35_current_selected_modes"],
        public_selected_modes=predictions["stage35_public_selected_modes"],
        active_modes=predictions["stage35_active_modes"],
        active_valid=predictions["stage35_active_valid"],
        hybrid_selected_modes=predictions["stage35_hybrid_selected_modes"],
        donor_indices=predictions["stage35_donor_indices"],
        scene_buckets=predictions["stage35_scene_buckets"],
        counterfactual_weight=float(
            getattr(config, "stage35_counterfactual_weight", 0.25)
        ),
        mature_positive_multiplier=float(
            getattr(config, "stage35_mature_positive_multiplier", 0.25)
        ),
        advantage_eps=float(
            getattr(config, "stage35_advantage_eps", 1e-3)
        ),
        delta_scale_floor=float(
            getattr(config, "stage35_delta_scale_floor", 0.002)
        ),
        headroom_low=float(getattr(config, "stage35_headroom_low", 0.75)),
        headroom_high=float(getattr(config, "stage35_headroom_high", 0.90)),
        rank_weight=float(getattr(config, "stage35_rank_weight", 0.5)),
        advantage_clip=float(
            getattr(config, "stage35_advantage_clip", 2.0)
        ),
        safety_tolerance=float(
            getattr(config, "diffgrpo_safety_regression_tolerance", 1e-6)
        ),
        step_discount=float(getattr(config, "diffgrpo_step_discount", 0.6)),
        bc_weight=float(getattr(config, "stage35_bc_weight", 0.1)),
        kl_weight=float(getattr(config, "stage35_kl_weight", 0.1)),
        safety_kl_weight=float(
            getattr(config, "stage35_safety_kl_weight", 0.5)
        ),
    )

    reward_mask = valid_mask.bool() & torch.isfinite(rewards)
    valid_rewards = rewards[reward_mask].float()
    zero = objective["loss"].detach() * 0.0
    result = {
        "loss": objective["loss"],
        "generation_grpo_loss": objective["policy_loss"],
        "diffgrpo_bc_loss": objective["bc_loss"],
        "generation_reference_kl_loss": objective["reference_kl_loss"],
        "diffgrpo_mean_current_log_prob": objective[
            "mean_current_log_prob"
        ],
        "raw_reward_mean": (
            valid_rewards.mean().detach() if valid_rewards.numel() else zero
        ),
        "raw_reward_std": (
            valid_rewards.std(unbiased=False).detach()
            if valid_rewards.numel() else zero
        ),
    }
    for name in (
        "candidate_level_trace",
        "same_anchor_only",
        "deployment_delta_mean",
        "counterfactual_credit_mean",
        "counterfactual_nonzero_fraction",
        "nonowner_influence_scene_fraction",
        "counterfactual_group_valid_fraction",
        "deployment_influence_fraction",
        "outer_positive_fraction",
        "inner_positive_fraction",
        "safety_override_fraction",
        "mean_exact_kl",
        "mean_bc_weight",
        "mean_kl_weight",
    ):
        result[f"stage35_{name}"] = objective[name]
    for name in (
        "stage35_selector_mode_disagreement",
        "stage35_current_selector_switch_rate",
        "stage35_public_selector_switch_rate",
        "stage35_sampled_chain_count",
        "stage35_replayed_chain_count",
        "stage35_counterfactual_selector_count",
    ):
        result[name] = predictions[name]
    result["diffgrpo_discount_first"] = objective["discount_first"]
    result["diffgrpo_discount_last"] = objective["discount_last"]
    return result
