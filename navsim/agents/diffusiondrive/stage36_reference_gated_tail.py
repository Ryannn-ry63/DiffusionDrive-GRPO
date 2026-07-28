"""Reference-gated upper-tail objective for Stage36 RGT-NCD-GRPO.

All PDM rewards and component scores consumed here are detached training
signals.  The differentiable policy terms are diffusion-chain log
probabilities computed after the reward-dependent replay set has been frozen.
"""

from __future__ import annotations

import math
from typing import Dict

import torch

from navsim.agents.diffusiondrive.stage35_counterfactual import (
    compute_nested_counterfactual_deployment_objective,
)


STAGE36_OBJECTIVE_REVISION = "reference_gated_tail_ncd_v1"
STAGE36_PLAN_SHA256 = (
    "858e6688faa9dff6d9027254e6a0262c67e47582d51a5829cc12a9723a25e521"
)

SAFETY_COMPONENT_INDICES = (0, 1, 3)


def _gather_bank_modes(
    values: torch.Tensor, modes: torch.Tensor
) -> torch.Tensor:
    """Gather ``[B,G,K,...]`` entries from a ``[B,G,M,...]`` bank."""
    if values.ndim < 3 or modes.ndim != 3:
        raise ValueError("Stage36 bank gather requires [B,G,M,...] and [B,G,K]")
    if values.shape[:2] != modes.shape[:2]:
        raise ValueError("Stage36 bank gather batch/group dimensions differ")
    index = modes.view(
        *modes.shape, *([1] * (values.ndim - 3))
    ).expand(*modes.shape, *values.shape[3:])
    return values.gather(2, index)


def _gather_anchor_groups(
    values: torch.Tensor, group_indices: torch.Tensor
) -> torch.Tensor:
    """Gather groups for each anchor.

    ``values`` is ``[B,G,M,...]`` and ``group_indices`` is ``[B,M,K]``.
    The result is ``[B,M,K,...]``.
    """
    if values.ndim < 3 or group_indices.ndim != 3:
        raise ValueError("Stage36 anchor gather shape mismatch")
    batch, groups, modes = values.shape[:3]
    if group_indices.shape[:2] != (batch, modes):
        raise ValueError("Stage36 anchor gather batch/mode dimensions differ")
    if ((group_indices < 0) | (group_indices >= groups)).any():
        raise ValueError("Stage36 anchor group index is out of range")
    anchors = torch.arange(modes, device=values.device).view(1, modes, 1)
    anchors = anchors.expand_as(group_indices)
    batches = torch.arange(batch, device=values.device).view(batch, 1, 1)
    batches = batches.expand_as(group_indices)
    return values[batches, group_indices, anchors]


def build_reference_gated_tail_targets(
    *,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    component_scores: torch.Tensor,
    base_rewards: torch.Tensor,
    base_valid_mask: torch.Tensor,
    base_component_scores: torch.Tensor,
    scene_buckets: torch.Tensor,
    frontier_width: int = 5,
    elite_count: int = 2,
    retention_tolerance: float = 1e-4,
    tail_margin: float = 0.001,
    tail_scale: float = 0.002,
    tail_weight: float = 0.25,
    retention_weight: float = 0.25,
    mature_positive_multiplier: float = 0.25,
    advantage_clip: float = 2.0,
    safety_tolerance: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Build the detached public-frontier and same-anchor tail targets."""
    if rewards.ndim != 3 or rewards.shape[1:] != (8, 20):
        raise ValueError("Stage36 rewards must have shape [B,8,20]")
    if valid_mask.shape != rewards.shape or base_rewards.shape != rewards.shape:
        raise ValueError("Stage36 reward/valid/base shapes differ")
    if base_valid_mask.shape != rewards.shape:
        raise ValueError("Stage36 base validity shape differs")
    if component_scores.shape != (*rewards.shape, 6):
        raise ValueError("Stage36 component scores must be [B,8,20,6]")
    if base_component_scores.shape != component_scores.shape:
        raise ValueError("Stage36 current/public component shapes differ")
    if scene_buckets.shape != (rewards.shape[0],):
        raise ValueError("Stage36 scene buckets must have shape [B]")
    constants = (
        retention_tolerance,
        tail_margin,
        tail_scale,
        tail_weight,
        retention_weight,
        mature_positive_multiplier,
        advantage_clip,
        safety_tolerance,
    )
    if not all(math.isfinite(float(value)) for value in constants):
        raise ValueError("Stage36 target constants must be finite")
    if (
        int(frontier_width) != 5
        or int(elite_count) != 2
        or retention_tolerance < 0
        or tail_margin < 0
        or tail_scale <= 0
        or min(tail_weight, retention_weight) < 0
        or not 0 <= mature_positive_multiplier <= 1
        or advantage_clip <= 0
        or safety_tolerance < 0
    ):
        raise ValueError("Stage36 target constants drifted")

    pair_valid = (
        valid_mask.bool()
        & base_valid_mask.bool()
        & torch.isfinite(rewards)
        & torch.isfinite(base_rewards)
        & torch.isfinite(component_scores).all(dim=-1)
        & torch.isfinite(base_component_scores).all(dim=-1)
    )
    public_valid = (
        base_valid_mask.bool()
        & torch.isfinite(base_rewards)
        & torch.isfinite(base_component_scores).all(dim=-1)
    )
    # Freeze the frontier from the public bank alone; current validity cannot
    # silently change which public top-5 modes the retention term protects.
    public_scores = base_rewards.float().masked_fill(~public_valid, -torch.inf)
    frontier_values, frontier_modes = public_scores.topk(
        int(frontier_width), dim=-1
    )
    frontier_valid = torch.isfinite(frontier_values)
    current_frontier_rewards = _gather_bank_modes(
        rewards.float(), frontier_modes
    )
    public_frontier_rewards = _gather_bank_modes(
        base_rewards.float(), frontier_modes
    )
    frontier_pair_valid = (
        frontier_valid
        & _gather_bank_modes(pair_valid, frontier_modes)
    )
    frontier_delta = torch.where(
        frontier_pair_valid,
        current_frontier_rewards - public_frontier_rewards,
        torch.zeros_like(current_frontier_rewards),
    )
    retention_gap = (
        public_frontier_rewards
        - current_frontier_rewards
        - float(retention_tolerance)
    )
    retention_active = frontier_pair_valid & (retention_gap > 0)
    retention_coefficients = torch.where(
        retention_active,
        float(retention_weight)
        * (retention_gap / float(tail_scale)).clamp(
            0.0, float(advantage_clip)
        ),
        torch.zeros_like(retention_gap),
    ).detach()

    # Rank rollouts only across stochastic replicas of the same anchor.
    current_by_anchor = rewards.float().permute(0, 2, 1)
    public_by_anchor = base_rewards.float().permute(0, 2, 1)
    valid_by_anchor = pair_valid.permute(0, 2, 1)
    current_tail_values, tail_group_indices = (
        current_by_anchor.masked_fill(~valid_by_anchor, -torch.inf).topk(
            int(elite_count), dim=-1
        )
    )
    public_tail_values = (
        public_by_anchor.masked_fill(~valid_by_anchor, -torch.inf).topk(
            int(elite_count), dim=-1
        ).values
    )
    current_tail_valid = torch.isfinite(current_tail_values)
    public_tail_valid = torch.isfinite(public_tail_values).all(dim=-1)
    tail_valid = current_tail_valid & public_tail_valid.unsqueeze(-1)
    public_tail_mean = torch.where(
        torch.isfinite(public_tail_values),
        public_tail_values,
        torch.zeros_like(public_tail_values),
    ).mean(dim=-1)
    current_tail_mean = torch.where(
        current_tail_valid,
        current_tail_values,
        torch.zeros_like(current_tail_values),
    ).sum(dim=-1) / current_tail_valid.sum(dim=-1).clamp_min(1)

    current_tail_components = _gather_anchor_groups(
        component_scores.float(), tail_group_indices
    )
    public_tail_components = _gather_anchor_groups(
        base_component_scores.float(), tail_group_indices
    )
    safety_indices = torch.tensor(
        SAFETY_COMPONENT_INDICES, device=rewards.device
    )
    tail_safety_regression = tail_valid & (
        current_tail_components.index_select(-1, safety_indices)
        < public_tail_components.index_select(-1, safety_indices)
        - float(safety_tolerance)
    ).any(dim=-1)
    tail_improvement = (
        current_tail_values
        - public_tail_mean.unsqueeze(-1)
        - float(tail_margin)
    )
    tail_positive = (
        tail_valid & (tail_improvement > 0) & ~tail_safety_regression
    )
    tail_advantages = torch.where(
        tail_positive,
        float(tail_weight)
        * (tail_improvement / float(tail_scale)).clamp(
            0.0, float(advantage_clip)
        ),
        torch.zeros_like(tail_improvement),
    )
    tail_advantages = torch.where(
        tail_safety_regression,
        torch.full_like(tail_advantages, -float(advantage_clip)),
        tail_advantages,
    )
    mature = scene_buckets.long() == 3
    tail_advantages = torch.where(
        mature[:, None, None] & (tail_advantages > 0),
        tail_advantages * float(mature_positive_multiplier),
        tail_advantages,
    )
    tail_advantages = torch.where(
        tail_valid, tail_advantages, torch.zeros_like(tail_advantages)
    ).detach()

    return {
        "pair_valid": pair_valid.detach(),
        "public_frontier_modes": frontier_modes.detach(),
        "public_frontier_valid": frontier_pair_valid.detach(),
        "frontier_delta": frontier_delta.detach(),
        "retention_active": retention_active.detach(),
        "retention_coefficients": retention_coefficients,
        "tail_group_indices": tail_group_indices.detach(),
        "tail_valid": tail_valid.detach(),
        "tail_positive": tail_positive.detach(),
        "tail_improvement": tail_improvement.detach(),
        "tail_safety_regression": tail_safety_regression.detach(),
        "tail_advantages": tail_advantages,
        "public_tail_mean": public_tail_mean.detach(),
        "current_tail_mean": current_tail_mean.detach(),
        "anchor_tail_delta": (
            current_tail_mean - public_tail_mean
        ).detach(),
    }


def _masked_scene_mean(
    terms: torch.Tensor, mask: torch.Tensor, differentiable_zero: torch.Tensor
) -> torch.Tensor:
    """Normalize one objective branch per scene before taking a batch mean."""
    if mask.shape != terms.shape:
        raise ValueError("Stage36 objective term/mask shapes differ")
    flat_terms = terms.flatten(1)
    flat_mask = mask.flatten(1)
    count = flat_mask.sum(dim=-1).clamp_min(1).float()
    per_scene = (flat_terms * flat_mask).sum(dim=-1) / count
    scene_valid = flat_mask.any(dim=-1)
    return (
        per_scene[scene_valid].mean()
        if scene_valid.any()
        else differentiable_zero.sum() * 0.0
    )


def compute_reference_gated_tail_ncd_objective(
    *,
    current_log_probs: torch.Tensor,
    bc_log_probs: torch.Tensor,
    reference_mean_kl: torch.Tensor,
    tail_current_log_probs: torch.Tensor,
    tail_reference_mean_kl: torch.Tensor,
    frontier_bc_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    component_scores: torch.Tensor,
    base_rewards: torch.Tensor,
    base_valid_mask: torch.Tensor,
    base_component_scores: torch.Tensor,
    current_selected_modes: torch.Tensor,
    public_selected_modes: torch.Tensor,
    active_modes: torch.Tensor,
    active_valid: torch.Tensor,
    hybrid_selected_modes: torch.Tensor,
    donor_indices: torch.Tensor,
    scene_buckets: torch.Tensor,
    public_frontier_modes: torch.Tensor,
    public_frontier_valid: torch.Tensor,
    retention_coefficients: torch.Tensor,
    tail_group_indices: torch.Tensor,
    tail_valid: torch.Tensor,
    counterfactual_weight: float = 0.25,
    mature_positive_multiplier: float = 0.25,
    advantage_eps: float = 1e-3,
    delta_scale_floor: float = 0.002,
    headroom_low: float = 0.75,
    headroom_high: float = 0.90,
    rank_weight: float = 0.5,
    advantage_clip: float = 2.0,
    retention_tolerance: float = 1e-4,
    tail_margin: float = 0.001,
    tail_scale: float = 0.002,
    tail_weight: float = 0.25,
    retention_weight: float = 0.25,
    safety_tolerance: float = 1e-6,
    step_discount: float = 0.6,
    bc_weight: float = 0.1,
    kl_weight: float = 0.1,
    safety_kl_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Combine Stage35 NCD with reference retention and safe tail expansion."""
    ncd = compute_nested_counterfactual_deployment_objective(
        current_log_probs=current_log_probs,
        bc_log_probs=bc_log_probs,
        reference_mean_kl=reference_mean_kl,
        rewards=rewards,
        valid_mask=valid_mask,
        component_scores=component_scores,
        base_rewards=base_rewards,
        base_valid_mask=base_valid_mask,
        base_component_scores=base_component_scores,
        current_selected_modes=current_selected_modes,
        public_selected_modes=public_selected_modes,
        active_modes=active_modes,
        active_valid=active_valid,
        hybrid_selected_modes=hybrid_selected_modes,
        donor_indices=donor_indices,
        scene_buckets=scene_buckets,
        counterfactual_weight=counterfactual_weight,
        mature_positive_multiplier=mature_positive_multiplier,
        advantage_eps=advantage_eps,
        delta_scale_floor=delta_scale_floor,
        headroom_low=headroom_low,
        headroom_high=headroom_high,
        rank_weight=rank_weight,
        advantage_clip=advantage_clip,
        safety_tolerance=safety_tolerance,
        step_discount=step_discount,
        bc_weight=bc_weight,
        kl_weight=kl_weight,
        safety_kl_weight=safety_kl_weight,
    )
    batch, groups, modes = rewards.shape
    steps = current_log_probs.shape[-1]
    expected_tail = (batch, modes, 2, steps)
    expected_frontier = (batch, groups, 5, steps)
    if (
        tail_current_log_probs.shape != expected_tail
        or tail_reference_mean_kl.shape != expected_tail
        or frontier_bc_log_probs.shape != expected_frontier
    ):
        raise ValueError("Stage36 compact replay probability shapes drifted")
    probability_tensors = (
        tail_current_log_probs,
        tail_reference_mean_kl,
        frontier_bc_log_probs,
    )
    if not all(torch.isfinite(value).all() for value in probability_tensors):
        raise FloatingPointError("Stage36 compact replay contains non-finite values")

    targets = build_reference_gated_tail_targets(
        rewards=rewards,
        valid_mask=valid_mask,
        component_scores=component_scores,
        base_rewards=base_rewards,
        base_valid_mask=base_valid_mask,
        base_component_scores=base_component_scores,
        scene_buckets=scene_buckets,
        frontier_width=5,
        elite_count=2,
        retention_tolerance=retention_tolerance,
        tail_margin=tail_margin,
        tail_scale=tail_scale,
        tail_weight=tail_weight,
        retention_weight=retention_weight,
        mature_positive_multiplier=mature_positive_multiplier,
        advantage_clip=advantage_clip,
        safety_tolerance=safety_tolerance,
    )
    exact_pairs = (
        (public_frontier_modes, targets["public_frontier_modes"]),
        (public_frontier_valid.bool(), targets["public_frontier_valid"]),
        (tail_group_indices, targets["tail_group_indices"]),
        (tail_valid.bool(), targets["tail_valid"]),
    )
    if not all(torch.equal(actual, expected) for actual, expected in exact_pairs):
        raise RuntimeError("Stage36 replay selection drifted from reward targets")
    if not torch.allclose(
        retention_coefficients.float(),
        targets["retention_coefficients"].float(),
        rtol=0.0,
        atol=1e-7,
    ):
        raise RuntimeError("Stage36 retention coefficients drifted")

    indices = torch.arange(
        steps, device=rewards.device, dtype=tail_current_log_probs.dtype
    )
    discounts = step_discount ** (steps - indices - 1)
    discount_shape = (1, 1, 1, steps)

    tail_advantages = targets["tail_advantages"]
    tail_mask = (
        targets["tail_valid"] & (tail_advantages != 0)
    ).unsqueeze(-1).expand_as(tail_current_log_probs)
    tail_terms = (
        -tail_current_log_probs
        * tail_advantages.unsqueeze(-1)
        * discounts.view(discount_shape)
    )
    tail_policy_loss = _masked_scene_mean(
        tail_terms,
        tail_mask,
        tail_current_log_probs,
    )

    retention = targets["retention_coefficients"]
    retention_mask = (
        targets["retention_active"].unsqueeze(-1)
        .expand_as(frontier_bc_log_probs)
    )
    retention_terms = (
        -frontier_bc_log_probs
        * retention.unsqueeze(-1)
        * discounts.view(discount_shape)
    )
    retention_loss = _masked_scene_mean(
        retention_terms,
        retention_mask,
        frontier_bc_log_probs,
    )

    tail_kl_coeff = torch.full_like(
        tail_advantages, float(kl_weight)
    )
    tail_kl_coeff = torch.where(
        targets["tail_safety_regression"],
        torch.full_like(tail_kl_coeff, float(safety_kl_weight)),
        tail_kl_coeff,
    )
    tail_kl_mask = targets["tail_valid"].unsqueeze(-1).expand_as(
        tail_reference_mean_kl
    )
    tail_kl_terms = (
        tail_reference_mean_kl
        * tail_kl_coeff.unsqueeze(-1)
    )
    tail_kl_loss = _masked_scene_mean(
        tail_kl_terms,
        tail_kl_mask,
        tail_reference_mean_kl,
    )

    total = ncd["loss"] + tail_policy_loss + retention_loss + tail_kl_loss
    if not torch.isfinite(total):
        raise FloatingPointError("Stage36 total loss is non-finite")

    frontier_valid_mask = targets["public_frontier_valid"]
    tail_valid_mask = targets["tail_valid"]
    pair_valid = targets["pair_valid"]
    current_oracle = rewards.float().masked_fill(
        ~pair_valid, -torch.inf
    ).amax(dim=(1, 2))
    public_oracle = base_rewards.float().masked_fill(
        ~pair_valid, -torch.inf
    ).amax(dim=(1, 2))
    valid_rewards = pair_valid.float().sum().clamp_min(1.0)
    result = dict(ncd)
    result.update(
        {
            "loss": total,
            "ncd_loss": ncd["loss"].detach(),
            "tail_policy_loss": tail_policy_loss.detach(),
            "frontier_retention_loss": retention_loss.detach(),
            "tail_reference_kl_loss": tail_kl_loss.detach(),
            "frontier_retention_active_fraction": (
                targets["retention_active"].float().sum()
                / frontier_valid_mask.float().sum().clamp_min(1.0)
            ).detach(),
            "tail_expansion_active_fraction": (
                targets["tail_positive"].float().sum()
                / tail_valid_mask.float().sum().clamp_min(1.0)
            ).detach(),
            "tail_safety_regression_fraction": (
                targets["tail_safety_regression"].float().sum()
                / tail_valid_mask.float().sum().clamp_min(1.0)
            ).detach(),
            "public_frontier_delta_mean": (
                targets["frontier_delta"][frontier_valid_mask].mean()
                if frontier_valid_mask.any()
                else rewards.new_zeros(())
            ).detach(),
            "anchor_tail_delta_mean": (
                targets["anchor_tail_delta"][tail_valid_mask.all(dim=-1)].mean()
                if tail_valid_mask.all(dim=-1).any()
                else rewards.new_zeros(())
            ).detach(),
            "candidate_mean_delta": (
                (
                    torch.where(
                        pair_valid, rewards.float(), torch.zeros_like(rewards.float())
                    )
                    - torch.where(
                        pair_valid,
                        base_rewards.float(),
                        torch.zeros_like(base_rewards.float()),
                    )
                ).sum()
                / valid_rewards
            ).detach(),
            "raw_oracle_delta_mean": (
                current_oracle - public_oracle
            ).mean().detach(),
            "frontier_exact_width": frontier_valid_mask.sum(dim=-1)
            .eq(5).all().to(rewards.dtype).detach(),
            "tail_exact_width": tail_valid_mask.sum(dim=-1)
            .eq(2).all().to(rewards.dtype).detach(),
            "retention_target_consistency": (
                ~targets["retention_active"]
                | (targets["frontier_delta"] < -float(retention_tolerance))
            ).all().to(rewards.dtype).detach(),
            "tail_target_consistency": (
                ~targets["tail_positive"]
                | ((targets["tail_improvement"] > 0)
                   & ~targets["tail_safety_regression"])
            ).all().to(rewards.dtype).detach(),
        }
    )
    return result

