"""Loss bridge for Stage36 Reference-Gated Tail NCD-GRPO."""

from __future__ import annotations

from typing import Any, Dict

import torch

from navsim.agents.diffusiondrive.stage36_reference_gated_tail import (
    compute_reference_gated_tail_ncd_objective,
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
            f"Stage36 bank tensor must have shape {expected}, got {value.shape}"
        )
    return value.reshape(batch, groups, modes, *trailing)


def compute_stage36_transfuser_loss(
    predictions: Dict[str, torch.Tensor],
    config: Any,
) -> Dict[str, torch.Tensor]:
    """Validate the deferred replay trace and expose stable metric names."""
    required = (
        "diffgrpo_current_log_probs",
        "diffgrpo_bc_log_probs",
        "diffgrpo_reference_mean_kl",
        "stage36_tail_current_log_probs",
        "stage36_tail_reference_mean_kl",
        "stage36_frontier_bc_log_probs",
        "raw_rewards",
        "reward_valid_mask",
        "component_scores",
        "diffgrpo_base_rewards",
        "diffgrpo_base_valid_mask",
        "diffgrpo_base_component_scores",
        "stage36_current_selected_modes",
        "stage36_public_selected_modes",
        "stage36_active_modes",
        "stage36_active_valid",
        "stage36_hybrid_selected_modes",
        "stage36_donor_indices",
        "stage36_scene_buckets",
        "stage36_public_frontier_modes",
        "stage36_public_frontier_valid",
        "stage36_retention_coefficients",
        "stage36_tail_group_indices",
        "stage36_tail_valid",
        "stage36_current_unique_replay_count",
        "stage36_public_unique_replay_count",
        "stage36_current_physical_replay_count",
        "stage36_public_physical_replay_count",
        "stage36_reward_dependent_replay",
        "stage36_replay_deduplicated",
        "stage36_selector_mode_disagreement",
        "stage36_current_selector_switch_rate",
        "stage36_public_selector_switch_rate",
        "stage36_sampled_chain_count",
        "stage36_counterfactual_selector_count",
        "diffgrpo_group_size",
    )
    missing = [name for name in required if predictions.get(name) is None]
    if missing:
        raise ValueError(f"Missing Stage36 RGT-NCD tensors: {missing}")
    current_log_probs = predictions["diffgrpo_current_log_probs"]
    if current_log_probs.ndim != 4:
        raise ValueError("Stage36 NCD log probabilities must be [B,8,4,S]")
    batch, groups, active_width, _ = current_log_probs.shape
    modes = 20
    if (groups, active_width) != (8, 4):
        raise ValueError("Stage36 freezes G=8 and NCD active width=4")
    scalar_expectations = {
        "diffgrpo_group_size": 8,
        "stage36_sampled_chain_count": 320,
        "stage36_counterfactual_selector_count": 224,
        "stage36_reward_dependent_replay": 1,
        "stage36_replay_deduplicated": 1,
    }
    for name, expected in scalar_expectations.items():
        value = int(predictions[name].detach().item())
        if value != expected:
            raise ValueError(
                f"Stage36 trace accounting drifted: {name}={value}"
            )

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
    objective = compute_reference_gated_tail_ncd_objective(
        current_log_probs=current_log_probs,
        bc_log_probs=predictions["diffgrpo_bc_log_probs"],
        reference_mean_kl=predictions["diffgrpo_reference_mean_kl"],
        tail_current_log_probs=predictions[
            "stage36_tail_current_log_probs"
        ],
        tail_reference_mean_kl=predictions[
            "stage36_tail_reference_mean_kl"
        ],
        frontier_bc_log_probs=predictions[
            "stage36_frontier_bc_log_probs"
        ],
        rewards=rewards,
        valid_mask=valid_mask,
        component_scores=components,
        base_rewards=base_rewards,
        base_valid_mask=base_valid,
        base_component_scores=base_components,
        current_selected_modes=predictions[
            "stage36_current_selected_modes"
        ],
        public_selected_modes=predictions[
            "stage36_public_selected_modes"
        ],
        active_modes=predictions["stage36_active_modes"],
        active_valid=predictions["stage36_active_valid"],
        hybrid_selected_modes=predictions[
            "stage36_hybrid_selected_modes"
        ],
        donor_indices=predictions["stage36_donor_indices"],
        scene_buckets=predictions["stage36_scene_buckets"],
        public_frontier_modes=predictions[
            "stage36_public_frontier_modes"
        ],
        public_frontier_valid=predictions[
            "stage36_public_frontier_valid"
        ],
        retention_coefficients=predictions[
            "stage36_retention_coefficients"
        ],
        tail_group_indices=predictions["stage36_tail_group_indices"],
        tail_valid=predictions["stage36_tail_valid"],
        counterfactual_weight=float(getattr(
            config, "stage36_counterfactual_weight", 0.25
        )),
        mature_positive_multiplier=float(getattr(
            config, "stage36_mature_positive_multiplier", 0.25
        )),
        advantage_eps=float(getattr(
            config, "stage36_advantage_eps", 1e-3
        )),
        delta_scale_floor=float(getattr(
            config, "stage36_delta_scale_floor", 0.002
        )),
        headroom_low=float(getattr(config, "stage36_headroom_low", 0.75)),
        headroom_high=float(getattr(config, "stage36_headroom_high", 0.90)),
        rank_weight=float(getattr(config, "stage36_rank_weight", 0.5)),
        advantage_clip=float(getattr(
            config, "stage36_advantage_clip", 2.0
        )),
        retention_tolerance=float(getattr(
            config, "stage36_retention_tolerance", 1e-4
        )),
        tail_margin=float(getattr(config, "stage36_tail_margin", 0.001)),
        tail_scale=float(getattr(config, "stage36_tail_scale", 0.002)),
        tail_weight=float(getattr(config, "stage36_tail_weight", 0.25)),
        retention_weight=float(getattr(
            config, "stage36_retention_weight", 0.25
        )),
        safety_tolerance=float(getattr(
            config, "diffgrpo_safety_regression_tolerance", 1e-6
        )),
        step_discount=float(getattr(
            config, "diffgrpo_step_discount", 0.6
        )),
        bc_weight=float(getattr(config, "stage36_bc_weight", 0.1)),
        kl_weight=float(getattr(config, "stage36_kl_weight", 0.1)),
        safety_kl_weight=float(getattr(
            config, "stage36_safety_kl_weight", 0.5
        )),
    )

    reward_mask = valid_mask.bool() & torch.isfinite(rewards)
    valid_rewards = rewards[reward_mask].float()
    zero = objective["loss"].detach() * 0.0
    result = {
        "loss": objective["loss"],
        "generation_grpo_loss": (
            objective["policy_loss"] + objective["tail_policy_loss"]
        ),
        "diffgrpo_bc_loss": (
            objective["bc_loss"] + objective["frontier_retention_loss"]
        ),
        "generation_reference_kl_loss": (
            objective["reference_kl_loss"]
            + objective["tail_reference_kl_loss"]
        ),
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
    metric_names = (
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
        "ncd_loss",
        "tail_policy_loss",
        "frontier_retention_loss",
        "tail_reference_kl_loss",
        "frontier_retention_active_fraction",
        "tail_expansion_active_fraction",
        "tail_safety_regression_fraction",
        "public_frontier_delta_mean",
        "anchor_tail_delta_mean",
        "candidate_mean_delta",
        "raw_oracle_delta_mean",
        "frontier_exact_width",
        "tail_exact_width",
        "retention_target_consistency",
        "tail_target_consistency",
    )
    for name in metric_names:
        result[f"stage36_{name}"] = objective[name]
    passthrough = (
        "stage36_selector_mode_disagreement",
        "stage36_current_selector_switch_rate",
        "stage36_public_selector_switch_rate",
        "stage36_sampled_chain_count",
        "stage36_counterfactual_selector_count",
        "stage36_current_unique_replay_count",
        "stage36_public_unique_replay_count",
        "stage36_current_physical_replay_count",
        "stage36_public_physical_replay_count",
        "stage36_reward_dependent_replay",
        "stage36_replay_deduplicated",
    )
    for name in passthrough:
        result[name] = predictions[name]
    result["diffgrpo_discount_first"] = objective["discount_first"]
    result["diffgrpo_discount_last"] = objective["discount_last"]
    return result

