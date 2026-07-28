"""Mode-aligned deployable-frontier objective for Stage34.

The objective consumes PDM rewards computed outside this module.  Inference
never calls this code and never depends on PDM.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch.nn import functional as F


STAGE34_OBJECTIVE_REVISION = "mode_aligned_frontier_v1"


def compute_mode_aligned_frontier_objective(
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
    current_selected_modes: torch.Tensor,
    public_selected_modes: torch.Tensor,
    current_eligible_mask: torch.Tensor,
    public_eligible_mask: torch.Tensor,
    scene_buckets: torch.Tensor,
    top_k: int = 5,
    headroom_k: int = 2,
    delta_scale_floor: float = 0.002,
    deployment_weight: float = 0.5,
    headroom_weight: float = 0.25,
    headroom_margin: float = 0.001,
    top5_negative_multiplier: float = 1.5,
    top1_negative_multiplier: float = 2.0,
    mature_positive_multiplier: float = 0.25,
    advantage_clip: float = 2.0,
    safety_tolerance: float = 1e-6,
    step_discount: float = 0.6,
    bc_weight: float = 0.1,
    kl_weight: float = 0.1,
    safety_kl_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Compute Stage34 advantages from same-anchor current/public pairs."""
    if current_log_probs.ndim != 3:
        raise ValueError("Stage34 log probabilities must have shape [B,20,S]")
    batch, modes, steps = current_log_probs.shape
    if modes != 20:
        raise ValueError("Stage34 requires exactly 20 mode-aligned chains")
    expected = (batch, modes)
    for name, tensor in (
        ("rewards", rewards),
        ("valid_mask", valid_mask),
        ("base_rewards", base_rewards),
        ("base_valid_mask", base_valid_mask),
        ("current_eligible_mask", current_eligible_mask),
        ("public_eligible_mask", public_eligible_mask),
    ):
        if tensor.shape != expected:
            raise ValueError(f"Stage34 {name} must have shape [B,20]")
    if component_scores.shape != (batch, modes, 6):
        raise ValueError("Stage34 current components must have shape [B,20,6]")
    if base_component_scores.shape != component_scores.shape:
        raise ValueError("Stage34 current/public component shapes differ")
    if bc_log_probs.shape != current_log_probs.shape:
        raise ValueError("Stage34 BC log probabilities must match [B,20,S]")
    if reference_mean_kl.shape != current_log_probs.shape:
        raise ValueError("Stage34 exact KL must match [B,20,S]")
    if current_selected_modes.shape != (batch,):
        raise ValueError("Stage34 current selected modes must have shape [B]")
    if public_selected_modes.shape != (batch,):
        raise ValueError("Stage34 public selected modes must have shape [B]")
    if scene_buckets.shape != (batch,):
        raise ValueError("Stage34 scene buckets must have shape [B]")
    if (
        (current_selected_modes < 0).any()
        or (current_selected_modes >= modes).any()
        or (public_selected_modes < 0).any()
        or (public_selected_modes >= modes).any()
    ):
        raise ValueError("Stage34 selected mode index is out of range")
    if ((scene_buckets < 0) | (scene_buckets > 3)).any():
        raise ValueError("Stage34 scene bucket IDs must lie in [0,3]")

    constants = (
        delta_scale_floor,
        deployment_weight,
        headroom_weight,
        headroom_margin,
        top5_negative_multiplier,
        top1_negative_multiplier,
        mature_positive_multiplier,
        advantage_clip,
        safety_tolerance,
        step_discount,
        bc_weight,
        kl_weight,
        safety_kl_weight,
    )
    if not all(math.isfinite(float(value)) for value in constants):
        raise ValueError("Stage34 objective constants must be finite")
    if (
        not 1 <= int(top_k) <= modes
        or not 1 <= int(headroom_k) < modes
        or delta_scale_floor <= 0
        or min(deployment_weight, headroom_weight, headroom_margin) < 0
        or top5_negative_multiplier < 1
        or top1_negative_multiplier < top5_negative_multiplier
        or not 0 <= mature_positive_multiplier <= 1
        or advantage_clip <= 0
        or safety_tolerance < 0
        or not 0 < step_discount <= 1
        or min(bc_weight, kl_weight, safety_kl_weight) < 0
        or safety_kl_weight < kl_weight
    ):
        raise ValueError("invalid Stage34 objective constants")
    if not (
        torch.isfinite(current_log_probs).all()
        and torch.isfinite(bc_log_probs).all()
        and torch.isfinite(reference_mean_kl).all()
    ):
        raise FloatingPointError("Stage34 policy/BC/KL tensors must be finite")
    if (reference_mean_kl < 0).any():
        raise FloatingPointError("Stage34 exact KL must be non-negative")

    pair_valid = (
        valid_mask.bool()
        & base_valid_mask.bool()
        & torch.isfinite(rewards)
        & torch.isfinite(base_rewards)
        & torch.isfinite(component_scores).all(dim=-1)
        & torch.isfinite(base_component_scores).all(dim=-1)
    )
    batch_index = torch.arange(batch, device=rewards.device)
    current_selected_modes = current_selected_modes.long()
    public_selected_modes = public_selected_modes.long()
    scene_valid = (
        pair_valid[batch_index, current_selected_modes]
        & pair_valid[batch_index, public_selected_modes]
    )
    current_owner = F.one_hot(current_selected_modes, modes).bool()
    public_owner = F.one_hot(public_selected_modes, modes).bool()

    public_rank_scores = base_rewards.float().masked_fill(~pair_valid, -torch.inf)
    public_top_indices = public_rank_scores.topk(int(top_k), dim=-1).indices
    public_top_mask = torch.zeros_like(pair_valid)
    public_top_mask.scatter_(1, public_top_indices, True)
    public_top_mask &= pair_valid
    public_top1_mask = torch.zeros_like(pair_valid)
    public_top1_mask.scatter_(1, public_top_indices[:, :1], True)
    public_top1_mask &= pair_valid

    current_selected_reward = rewards.float()[batch_index, current_selected_modes]
    public_selected_reward = base_rewards.float()[batch_index, public_selected_modes]
    deployment_delta = current_selected_reward - public_selected_reward
    headroom = (
        rewards.float()
        - current_selected_reward.unsqueeze(-1)
        - float(headroom_margin)
    ).clamp_min(0.0)
    headroom_allowed = pair_valid & current_eligible_mask.bool() & (headroom > 0)
    headroom_scores = headroom.masked_fill(~headroom_allowed, -torch.inf)
    headroom_values, headroom_indices = headroom_scores.topk(
        int(headroom_k), dim=-1
    )
    headroom_mask = torch.zeros_like(pair_valid)
    headroom_mask.scatter_(1, headroom_indices, torch.isfinite(headroom_values))
    headroom_mask &= headroom_allowed

    active = (
        current_owner | public_owner | public_top_mask | headroom_mask
    ) & pair_valid & scene_valid.unsqueeze(-1)
    active_count = active.sum(dim=-1)
    group_valid = active_count >= 2
    delta = torch.where(
        pair_valid,
        rewards.float() - base_rewards.float(),
        torch.zeros_like(rewards.float()),
    )
    scale = (
        (delta.abs() * active).sum(dim=-1)
        / active_count.clamp_min(1).float()
    ).clamp_min(float(delta_scale_floor))
    pair_advantage = delta / scale.unsqueeze(-1)

    negative_multiplier = torch.ones_like(pair_advantage)
    negative_multiplier = torch.where(
        public_top_mask & (delta < 0),
        negative_multiplier.new_full(
            negative_multiplier.shape, float(top5_negative_multiplier)
        ),
        negative_multiplier,
    )
    negative_multiplier = torch.where(
        public_top1_mask & (delta < 0),
        negative_multiplier.new_full(
            negative_multiplier.shape, float(top1_negative_multiplier)
        ),
        negative_multiplier,
    )
    advantages = pair_advantage * negative_multiplier
    advantages = advantages + (
        float(deployment_weight)
        * current_owner.to(advantages)
        * (deployment_delta / scale).unsqueeze(-1)
    )
    headroom_credit = (
        float(headroom_weight)
        * headroom_mask.to(advantages)
        * headroom
        / scale.unsqueeze(-1)
    )
    advantages = advantages + headroom_credit

    # Public-frontier regressions may not be erased by deployment/headroom
    # credit. This fail-closed rule directly addresses the Stage30/33 result.
    frontier_negative_ceiling = pair_advantage * negative_multiplier
    advantages = torch.where(
        public_top_mask & (delta < 0),
        torch.minimum(advantages, frontier_negative_ceiling),
        advantages,
    )
    mature = scene_buckets.long() == 3
    advantages = torch.where(
        mature.unsqueeze(-1) & (advantages > 0),
        advantages * float(mature_positive_multiplier),
        advantages,
    )

    safety_indices = torch.tensor((0, 1, 3), device=rewards.device)
    safety_regression = pair_valid & (
        component_scores.float().index_select(-1, safety_indices)
        < base_component_scores.float().index_select(-1, safety_indices)
        - float(safety_tolerance)
    ).any(dim=-1)
    advantages = torch.where(
        safety_regression & active,
        torch.full_like(advantages, -float(advantage_clip)),
        advantages,
    )
    optimize = active & group_valid.unsqueeze(-1)
    advantages = torch.where(
        optimize,
        advantages.clamp(-float(advantage_clip), float(advantage_clip)),
        torch.zeros_like(advantages),
    ).detach()
    if not torch.isfinite(advantages).all():
        raise FloatingPointError("Stage34 advantages must be finite")

    indices = torch.arange(
        steps, device=rewards.device, dtype=current_log_probs.dtype
    )
    discounts = step_discount ** (steps - indices - 1)
    policy_mask = (optimize & (advantages != 0)).unsqueeze(-1).expand_as(
        current_log_probs
    )
    policy_terms = (
        -current_log_probs
        * advantages.unsqueeze(-1)
        * discounts.view(1, 1, -1)
    )
    policy_count = policy_mask.sum(dim=(1, 2)).clamp_min(1).float()
    policy_per_scene = (
        (policy_terms * policy_mask).sum(dim=(1, 2)) / policy_count
    )
    policy_scene_valid = policy_mask.flatten(1).any(dim=1)
    policy_loss = (
        policy_per_scene[policy_scene_valid].mean()
        if policy_scene_valid.any()
        else current_log_probs.sum() * 0.0
    )

    # Inactive low-ranked modes get no policy credit but remain anchored.
    all_mask = pair_valid.unsqueeze(-1).expand_as(current_log_probs)
    all_count = all_mask.sum(dim=(1, 2)).clamp_min(1).float()
    bc_terms = (
        -bc_log_probs
        * discounts.view(1, 1, -1)
        * float(bc_weight)
    )
    bc_per_scene = (bc_terms * all_mask).sum(dim=(1, 2)) / all_count
    regularization_valid = pair_valid.any(dim=-1)
    bc_loss = (
        bc_per_scene[regularization_valid].mean()
        if regularization_valid.any()
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
        kl_per_scene[regularization_valid].mean()
        if regularization_valid.any()
        else reference_mean_kl.sum() * 0.0
    )
    total = policy_loss + bc_loss + kl_loss

    current_deployable = (
        current_eligible_mask.bool() | current_owner
    ) & pair_valid
    public_deployable = (
        public_eligible_mask.bool() | public_owner
    ) & pair_valid
    current_safe_oracle = rewards.float().masked_fill(
        ~current_deployable, -torch.inf
    ).amax(dim=-1)
    public_safe_oracle = base_rewards.float().masked_fill(
        ~public_deployable, -torch.inf
    ).amax(dim=-1)
    raw_oracle_delta = (
        rewards.float().masked_fill(~pair_valid, -torch.inf).amax(dim=-1)
        - base_rewards.float().masked_fill(~pair_valid, -torch.inf).amax(dim=-1)
    )
    hard = scene_buckets.long() <= 1
    valid_total = pair_valid.float().sum().clamp_min(1.0)

    def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return value[mask].mean().detach() if mask.any() else value.sum() * 0.0

    return {
        "loss": total,
        "policy_loss": policy_loss,
        "bc_loss": bc_loss,
        "reference_kl_loss": kl_loss,
        "advantages": advantages,
        "candidate_level_trace": rewards.new_tensor(1.0),
        "same_mode_delta_mean": masked_mean(delta, pair_valid),
        "candidate_mean_delta": masked_mean(delta, pair_valid),
        "deployment_delta_mean": masked_mean(deployment_delta, scene_valid),
        "hard_selected_delta_mean": masked_mean(
            deployment_delta, scene_valid & hard
        ),
        "mature_selected_delta_mean": masked_mean(
            deployment_delta, scene_valid & mature
        ),
        "public_top1_delta_mean": masked_mean(delta, public_top1_mask),
        "public_top5_delta_mean": masked_mean(delta, public_top_mask),
        "raw_oracle_delta_mean": masked_mean(raw_oracle_delta, scene_valid),
        "safe_deployable_oracle_delta_mean": masked_mean(
            current_safe_oracle - public_safe_oracle, scene_valid
        ),
        "headroom_credit_mean": masked_mean(headroom_credit, headroom_mask),
        "active_mode_fraction": optimize.float().sum().detach() / valid_total,
        "positive_fraction": (
            ((advantages > 0) & optimize).float().sum() / valid_total
        ).detach(),
        "negative_fraction": (
            ((advantages < 0) & optimize).float().sum() / valid_total
        ).detach(),
        "safety_override_fraction": (
            (safety_regression & optimize).float().sum() / valid_total
        ).detach(),
        "catastrophic_fraction": (
            ((deployment_delta <= -0.5) & scene_valid).float().sum()
            / scene_valid.float().sum().clamp_min(1.0)
        ).detach(),
        "mean_bc_weight": rewards.new_tensor(float(bc_weight)),
        "mean_kl_weight": masked_mean(kl_coefficients, pair_valid),
        "mean_exact_kl": masked_mean(reference_mean_kl, all_mask),
        "mean_current_log_prob": masked_mean(current_log_probs, all_mask),
        "delta_scale_mean": masked_mean(scale, group_valid),
        "discount_first": discounts[0].detach(),
        "discount_last": discounts[-1].detach(),
    }

