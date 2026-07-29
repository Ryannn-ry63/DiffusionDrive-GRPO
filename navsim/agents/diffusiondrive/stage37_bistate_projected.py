"""Stage37 bi-state projected deployment GRPO objective and projection.

PDM rewards are detached signals.  This module deliberately returns separate
reward and preservation objectives: Lightning back-propagates the reward
objective normally, captures the preservation gradient with ``autograd.grad``,
and combines them only at the optimizer boundary.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence, Tuple

import torch

from navsim.agents.diffusiondrive.stage35_counterfactual import (
    compute_nested_counterfactual_deployment_objective,
)
from navsim.agents.diffusiondrive.stage36_reference_gated_tail import (
    _masked_scene_mean,
    build_reference_gated_tail_targets,
)


STAGE37_OBJECTIVE_REVISION = "bistate_projected_deployment_v1"
STAGE37_PLAN_SHA256 = (
    "4dc469dd0be4d2579796ff70ae7a09ed9fd8e4c92449cd057e42463d268bb5d8"
)


def _masked_mean(
    values: torch.Tensor, mask: torch.Tensor, differentiable_zero: torch.Tensor
) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("Stage37 value/mask shapes differ")
    if mask.any():
        return values[mask].mean()
    return differentiable_zero.sum() * 0.0


def compute_bistate_projected_deployment_objective(
    *,
    current_log_probs: torch.Tensor,
    bc_log_probs: torch.Tensor,
    reference_mean_kl: torch.Tensor,
    tail_current_log_probs: torch.Tensor,
    current_frontier_mean_kl: torch.Tensor,
    public_frontier_mean_kl: torch.Tensor,
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
    tail_group_indices: torch.Tensor,
    tail_valid: torch.Tensor,
    counterfactual_weight: float = 0.25,
    tail_weight: float = 0.25,
    frontier_weight: float = 0.25,
    advantage_scale: float = 0.002,
    advantage_clip: float = 2.0,
    tail_margin: float = 0.001,
    safety_tolerance: float = 1e-6,
    step_discount: float = 0.6,
    bc_weight: float = 0.1,
    kl_weight: float = 0.1,
    safety_kl_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Build independent reward and function-preservation objectives."""
    constants = (
        counterfactual_weight,
        tail_weight,
        frontier_weight,
        advantage_scale,
        advantage_clip,
        tail_margin,
        safety_tolerance,
        step_discount,
        bc_weight,
        kl_weight,
        safety_kl_weight,
    )
    if not all(math.isfinite(float(value)) for value in constants):
        raise ValueError("Stage37 objective constants must be finite")
    if (
        min(counterfactual_weight, tail_weight, frontier_weight) < 0
        or advantage_scale <= 0
        or advantage_clip <= 0
        or tail_margin < 0
        or safety_tolerance < 0
        or not 0 < step_discount <= 1
        or min(bc_weight, kl_weight, safety_kl_weight) < 0
        or safety_kl_weight < kl_weight
    ):
        raise ValueError("Stage37 objective constants are invalid")

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
        # Mature scenes receive protection but no positive exploration.
        mature_positive_multiplier=0.0,
        advantage_eps=1e-3,
        delta_scale_floor=advantage_scale,
        headroom_low=0.75,
        headroom_high=0.90,
        rank_weight=0.5,
        advantage_clip=advantage_clip,
        safety_tolerance=safety_tolerance,
        step_discount=step_discount,
        bc_weight=bc_weight,
        kl_weight=kl_weight,
        safety_kl_weight=safety_kl_weight,
    )
    batch, groups, modes = rewards.shape
    steps = current_log_probs.shape[-1]
    if (groups, modes) != (8, 20):
        raise ValueError("Stage37 freezes G=8 and M=20")
    expected_tail = (batch, modes, 2, steps)
    expected_frontier = (batch, groups, 5, steps)
    if tail_current_log_probs.shape != expected_tail:
        raise ValueError("Stage37 tail probability shape drifted")
    if (
        current_frontier_mean_kl.shape != expected_frontier
        or public_frontier_mean_kl.shape != expected_frontier
    ):
        raise ValueError("Stage37 bi-state frontier KL shape drifted")
    if not all(torch.isfinite(value).all() for value in (
        tail_current_log_probs,
        current_frontier_mean_kl,
        public_frontier_mean_kl,
    )):
        raise FloatingPointError("Stage37 replay probabilities are non-finite")

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
        retention_tolerance=1e-4,
        tail_margin=tail_margin,
        tail_scale=advantage_scale,
        tail_weight=tail_weight,
        retention_weight=frontier_weight,
        mature_positive_multiplier=0.0,
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
        raise RuntimeError("Stage37 replay selection drifted from reward targets")

    indices = torch.arange(
        steps, device=rewards.device, dtype=tail_current_log_probs.dtype
    )
    discounts = step_discount ** (steps - indices - 1)
    tail_advantages = targets["tail_advantages"]
    tail_mask = (
        targets["tail_valid"] & (tail_advantages != 0)
    ).unsqueeze(-1).expand_as(tail_current_log_probs)
    tail_terms = (
        -tail_current_log_probs
        * tail_advantages.unsqueeze(-1)
        * discounts.view(1, 1, 1, steps)
    )
    tail_policy_loss = _masked_scene_mean(
        tail_terms, tail_mask, tail_current_log_probs
    )

    frontier_mask = targets["public_frontier_valid"].unsqueeze(-1).expand_as(
        current_frontier_mean_kl
    )
    current_state_loss = _masked_mean(
        current_frontier_mean_kl, frontier_mask, current_frontier_mean_kl
    )
    public_state_loss = _masked_mean(
        public_frontier_mean_kl, frontier_mask, public_frontier_mean_kl
    )
    reward_loss = ncd["policy_loss"] + tail_policy_loss
    preservation_loss = (
        ncd["bc_loss"]
        + ncd["reference_kl_loss"]
        + float(frontier_weight) * (current_state_loss + public_state_loss)
    )
    if not torch.isfinite(reward_loss) or not torch.isfinite(preservation_loss):
        raise FloatingPointError("Stage37 split objective is non-finite")

    pair_valid = targets["pair_valid"]
    valid_pairs = pair_valid.float().sum().clamp_min(1.0)
    current_oracle = rewards.float().masked_fill(~pair_valid, -torch.inf).amax(
        dim=(1, 2)
    )
    public_oracle = base_rewards.float().masked_fill(
        ~pair_valid, -torch.inf
    ).amax(dim=(1, 2))
    result = dict(ncd)
    result.update({
        # Automatic backward must see reward only. Preservation is captured
        # explicitly before backward and applied at the optimizer boundary.
        "loss": reward_loss,
        "reward_loss": reward_loss,
        "preservation_loss": preservation_loss,
        "tail_policy_loss": tail_policy_loss.detach(),
        "current_state_frontier_kl_loss": current_state_loss.detach(),
        "public_state_frontier_kl_loss": public_state_loss.detach(),
        "public_frontier_delta_mean": (
            targets["frontier_delta"][targets["public_frontier_valid"]].mean()
            if targets["public_frontier_valid"].any()
            else rewards.new_zeros(())
        ).detach(),
        "candidate_mean_delta": (
            (
                torch.where(pair_valid, rewards.float(), torch.zeros_like(rewards))
                - torch.where(
                    pair_valid, base_rewards.float(), torch.zeros_like(base_rewards)
                )
            ).sum() / valid_pairs
        ).detach(),
        "raw_oracle_delta_mean": (current_oracle - public_oracle).mean().detach(),
        "frontier_exact_width": targets["public_frontier_valid"].sum(dim=-1)
        .eq(5).all().to(rewards.dtype).detach(),
        "tail_exact_width": targets["tail_valid"].sum(dim=-1)
        .eq(2).all().to(rewards.dtype).detach(),
        "mature_positive_exploration_zero": (
            ~(
                (scene_buckets.long() == 3)[:, None, None]
                & (targets["tail_advantages"] > 0)
            )
        ).all().to(rewards.dtype).detach(),
        "bistate_constraint_active": rewards.new_tensor(1.0),
    })
    return result


def project_reward_gradient(
    reward_gradients: Sequence[torch.Tensor],
    preservation_gradients: Sequence[torch.Tensor],
    *,
    frontier_delta: torch.Tensor | float,
    regression_tolerance: float = 1e-4,
    recovery_coefficient: float = 0.25,
    epsilon: float = 1e-12,
) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
    """Project conflicting reward gradients and conditionally add recovery."""
    if len(reward_gradients) != len(preservation_gradients) or not reward_gradients:
        raise ValueError("Stage37 gradient lists must be non-empty and aligned")
    if (
        regression_tolerance < 0
        or recovery_coefficient < 0
        or epsilon <= 0
        or not all(math.isfinite(float(value)) for value in (
            regression_tolerance, recovery_coefficient, epsilon
        ))
    ):
        raise ValueError("Stage37 projection constants are invalid")
    for reward, preserve in zip(reward_gradients, preservation_gradients):
        if reward.shape != preserve.shape:
            raise ValueError("Stage37 reward/preservation gradient shapes differ")
        if not torch.isfinite(reward).all() or not torch.isfinite(preserve).all():
            raise FloatingPointError("Stage37 gradient contains non-finite values")
    device = reward_gradients[0].device
    dot = torch.zeros((), device=device, dtype=torch.float64)
    reward_sq = torch.zeros_like(dot)
    preserve_sq = torch.zeros_like(dot)
    for reward, preserve in zip(reward_gradients, preservation_gradients):
        reward64 = reward.detach().double()
        preserve64 = preserve.detach().double()
        dot += (reward64 * preserve64).sum()
        reward_sq += reward64.square().sum()
        preserve_sq += preserve64.square().sum()
    conflict = dot < 0
    denominator = preserve_sq + float(epsilon)
    coefficient = torch.where(conflict, dot / denominator, torch.zeros_like(dot))
    delta = torch.as_tensor(frontier_delta, device=device, dtype=torch.float64)
    recovery = delta < -float(regression_tolerance)
    projected = tuple(
        reward
        - coefficient.to(reward.dtype) * preserve
        + (
            float(recovery_coefficient) * preserve
            if bool(recovery.item()) else torch.zeros_like(preserve)
        )
        for reward, preserve in zip(reward_gradients, preservation_gradients)
    )
    projection_sq = torch.zeros_like(dot)
    for reward, value in zip(reward_gradients, projected):
        projection_sq += (value.detach().double() - reward.detach().double()).square().sum()
    cosine = dot / (reward_sq.sqrt() * preserve_sq.sqrt() + float(epsilon))
    return projected, {
        "dot": dot.float(),
        "cosine": cosine.float(),
        "conflict": conflict.to(torch.float32),
        "reward_grad_norm": reward_sq.sqrt().float(),
        "preservation_grad_norm": preserve_sq.sqrt().float(),
        "projection_norm": projection_sq.sqrt().float(),
        "recovery_active": recovery.to(torch.float32),
        "frontier_delta": delta.float(),
        "finite": torch.stack([
            torch.isfinite(value).all().to(torch.float32) for value in projected
        ]).amin(),
    }
