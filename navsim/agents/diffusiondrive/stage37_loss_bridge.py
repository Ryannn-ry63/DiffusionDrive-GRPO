"""Loss bridge for Stage37 bi-state projected deployment GRPO."""

from __future__ import annotations

from typing import Any, Dict

import torch

from navsim.agents.diffusiondrive.stage37_bistate_projected import (
    compute_bistate_projected_deployment_objective,
)


def _bank(
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
            f"Stage37 bank tensor must have shape {expected}, got {value.shape}"
        )
    return value.reshape(batch, groups, modes, *trailing)


def compute_stage37_transfuser_loss(
    predictions: Dict[str, torch.Tensor], config: Any
) -> Dict[str, torch.Tensor]:
    """Validate Stage37 trace accounting and expose split objectives."""
    required = (
        "diffgrpo_current_log_probs",
        "diffgrpo_bc_log_probs",
        "diffgrpo_reference_mean_kl",
        "stage37_tail_current_log_probs",
        "stage37_current_frontier_mean_kl",
        "stage37_public_frontier_mean_kl",
        "raw_rewards",
        "reward_valid_mask",
        "component_scores",
        "diffgrpo_base_rewards",
        "diffgrpo_base_valid_mask",
        "diffgrpo_base_component_scores",
        "stage37_current_selected_modes",
        "stage37_public_selected_modes",
        "stage37_active_modes",
        "stage37_active_valid",
        "stage37_hybrid_selected_modes",
        "stage37_donor_indices",
        "stage37_scene_buckets",
        "stage37_public_frontier_modes",
        "stage37_public_frontier_valid",
        "stage37_tail_group_indices",
        "stage37_tail_valid",
        "stage37_all_frontier_replay",
        "stage37_replay_deduplicated",
        "stage37_sampled_chain_count",
        "stage37_counterfactual_selector_count",
        "diffgrpo_group_size",
    )
    missing = [name for name in required if predictions.get(name) is None]
    if missing:
        raise ValueError(f"Missing Stage37 BPD-GRPO tensors: {missing}")
    current_log_probs = predictions["diffgrpo_current_log_probs"]
    if current_log_probs.ndim != 4:
        raise ValueError("Stage37 NCD log probabilities must be [B,8,4,S]")
    batch, groups, active_width, _ = current_log_probs.shape
    if (groups, active_width) != (8, 4):
        raise ValueError("Stage37 freezes G=8 and active width=4")
    for name, expected in {
        "diffgrpo_group_size": 8,
        "stage37_sampled_chain_count": 320,
        "stage37_counterfactual_selector_count": 224,
        "stage37_all_frontier_replay": 1,
        "stage37_replay_deduplicated": 1,
    }.items():
        actual = int(predictions[name].detach().item())
        if actual != expected:
            raise ValueError(f"Stage37 trace accounting drifted: {name}={actual}")
    modes = 20
    rewards = _bank(
        predictions["raw_rewards"], batch=batch, groups=groups, modes=modes
    )
    valid = _bank(
        predictions["reward_valid_mask"],
        batch=batch,
        groups=groups,
        modes=modes,
    )
    components = _bank(
        predictions["component_scores"],
        batch=batch,
        groups=groups,
        modes=modes,
        trailing=(6,),
    )
    base_rewards = _bank(
        predictions["diffgrpo_base_rewards"],
        batch=batch,
        groups=groups,
        modes=modes,
    )
    base_valid = _bank(
        predictions["diffgrpo_base_valid_mask"],
        batch=batch,
        groups=groups,
        modes=modes,
    )
    base_components = _bank(
        predictions["diffgrpo_base_component_scores"],
        batch=batch,
        groups=groups,
        modes=modes,
        trailing=(6,),
    )
    objective = compute_bistate_projected_deployment_objective(
        current_log_probs=current_log_probs,
        bc_log_probs=predictions["diffgrpo_bc_log_probs"],
        reference_mean_kl=predictions["diffgrpo_reference_mean_kl"],
        tail_current_log_probs=predictions[
            "stage37_tail_current_log_probs"
        ],
        current_frontier_mean_kl=predictions[
            "stage37_current_frontier_mean_kl"
        ],
        public_frontier_mean_kl=predictions[
            "stage37_public_frontier_mean_kl"
        ],
        rewards=rewards,
        valid_mask=valid,
        component_scores=components,
        base_rewards=base_rewards,
        base_valid_mask=base_valid,
        base_component_scores=base_components,
        current_selected_modes=predictions[
            "stage37_current_selected_modes"
        ],
        public_selected_modes=predictions[
            "stage37_public_selected_modes"
        ],
        active_modes=predictions["stage37_active_modes"],
        active_valid=predictions["stage37_active_valid"],
        hybrid_selected_modes=predictions[
            "stage37_hybrid_selected_modes"
        ],
        donor_indices=predictions["stage37_donor_indices"],
        scene_buckets=predictions["stage37_scene_buckets"],
        public_frontier_modes=predictions[
            "stage37_public_frontier_modes"
        ],
        public_frontier_valid=predictions[
            "stage37_public_frontier_valid"
        ],
        tail_group_indices=predictions["stage37_tail_group_indices"],
        tail_valid=predictions["stage37_tail_valid"],
        counterfactual_weight=float(getattr(
            config, "stage37_counterfactual_weight", 0.25
        )),
        tail_weight=float(getattr(config, "stage37_tail_weight", 0.25)),
        frontier_weight=float(getattr(
            config, "stage37_frontier_weight", 0.25
        )),
        advantage_scale=float(getattr(
            config, "stage37_advantage_scale", 0.002
        )),
        advantage_clip=float(getattr(
            config, "stage37_advantage_clip", 2.0
        )),
        tail_margin=float(getattr(config, "stage37_tail_margin", 0.001)),
        safety_tolerance=float(getattr(
            config, "diffgrpo_safety_regression_tolerance", 1e-6
        )),
        step_discount=float(getattr(config, "diffgrpo_step_discount", 0.6)),
        bc_weight=float(getattr(config, "stage37_bc_weight", 0.1)),
        kl_weight=float(getattr(config, "stage37_kl_weight", 0.1)),
        safety_kl_weight=float(getattr(
            config, "stage37_safety_kl_weight", 0.5
        )),
    )
    reward_mask = valid.bool() & torch.isfinite(rewards)
    valid_rewards = rewards[reward_mask].float()
    zero = objective["loss"].detach() * 0.0
    result = {
        "loss": objective["loss"],
        # These two non-detached values are consumed by the optimizer bridge.
        "stage37_reward_loss": objective["reward_loss"],
        "stage37_preservation_loss": objective["preservation_loss"],
        "generation_grpo_loss": objective["reward_loss"].detach(),
        "generation_kl_loss": objective["preservation_loss"].detach(),
        "stage37_public_frontier_delta_mean": objective[
            "public_frontier_delta_mean"
        ],
        "stage37_candidate_mean_delta": objective["candidate_mean_delta"],
        "stage37_raw_oracle_delta_mean": objective["raw_oracle_delta_mean"],
        "stage37_tail_policy_loss": objective["tail_policy_loss"],
        "stage37_current_state_frontier_kl_loss": objective[
            "current_state_frontier_kl_loss"
        ],
        "stage37_public_state_frontier_kl_loss": objective[
            "public_state_frontier_kl_loss"
        ],
        "stage37_frontier_exact_width": objective["frontier_exact_width"],
        "stage37_tail_exact_width": objective["tail_exact_width"],
        "stage37_mature_positive_exploration_zero": objective[
            "mature_positive_exploration_zero"
        ],
        "stage37_bistate_constraint_active": objective[
            "bistate_constraint_active"
        ],
        "stage37_reward_mean": (
            valid_rewards.mean() if valid_rewards.numel() else zero
        ).detach(),
        "stage37_reward_std": (
            valid_rewards.std(unbiased=False)
            if valid_rewards.numel() else zero
        ).detach(),
    }
    for name in (
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
        "mean_current_log_prob",
    ):
        result[f"stage37_{name}"] = objective[name]
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("Stage37 loss bridge produced non-finite output")
    return result
