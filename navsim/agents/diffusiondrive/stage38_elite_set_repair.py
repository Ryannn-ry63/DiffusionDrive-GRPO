"""Stage38 Elite-Set Counterfactual Repair (ESCR) objective.

The public 88.1 policy defines an immutable top-five trajectory set for every
common-noise group.  A current-policy trajectory receives credit only for the
change in that set's utility when it replaces its mode-aligned public member
or enters as a new challenger.  Rewards and set membership are detached; only
the replayed diffusion-chain log probabilities carry gradients.
"""

from __future__ import annotations

import math
from typing import Dict

import torch

from navsim.agents.diffusiondrive.stage36_reference_gated_tail import (
    SAFETY_COMPONENT_INDICES,
    _masked_scene_mean,
)


STAGE38_OBJECTIVE_REVISION = "elite_set_counterfactual_repair_v1"
STAGE38_PLAN_SHA256 = (
    "8cd826b6ab8a4376c6e4a4fdffc97e96fe00fba00a25a1029d5fa5bcf4a8b027"
)


def _gather_modes(values: torch.Tensor, modes: torch.Tensor) -> torch.Tensor:
    """Gather ``[B,G,K,...]`` values from a ``[B,G,M,...]`` bank."""
    if values.ndim < 3 or modes.ndim != 3:
        raise ValueError("Stage38 gather expects [B,G,M,...] and [B,G,K]")
    if values.shape[:2] != modes.shape[:2]:
        raise ValueError("Stage38 gather batch/group dimensions differ")
    index = modes.view(
        *modes.shape, *([1] * (values.ndim - 3))
    ).expand(*modes.shape, *values.shape[3:])
    return values.gather(2, index)


def _stable_descending(values: torch.Tensor) -> torch.Tensor:
    """Sort descending while keeping the lower mode index on exact ties."""
    try:
        return torch.argsort(values, dim=-1, descending=True, stable=True)
    except TypeError as error:  # pragma: no cover - supported training builds
        raise RuntimeError("Stage38 requires stable torch.argsort") from error


def _masked_detached_mean(
    values: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    valid = mask.bool() & torch.isfinite(values)
    return (
        values[valid].float().mean().detach()
        if valid.any()
        else values.new_zeros(())
    )


def build_elite_set_repair_targets(
    *,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    component_scores: torch.Tensor,
    base_rewards: torch.Tensor,
    base_valid_mask: torch.Tensor,
    base_component_scores: torch.Tensor,
    current_selected_modes: torch.Tensor,
    scene_buckets: torch.Tensor,
    elite_width: int = 5,
    positive_challenger_width: int = 2,
    active_pool_width: int = 8,
    positive_margin: float = 0.001,
    negative_tolerance: float = 0.0001,
    advantage_scale: float = 0.002,
    advantage_clip: float = 2.0,
    mature_positive_multiplier: float = 0.25,
    safety_tolerance: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Build deterministic public-set substitution targets and replay pools."""
    if rewards.ndim != 3 or rewards.shape[1:] != (8, 20):
        raise ValueError("Stage38 rewards must have shape [B,8,20]")
    batch, groups, modes = rewards.shape
    if valid_mask.shape != rewards.shape or base_rewards.shape != rewards.shape:
        raise ValueError("Stage38 reward/valid/base shapes differ")
    if base_valid_mask.shape != rewards.shape:
        raise ValueError("Stage38 base validity shape differs")
    if component_scores.shape != (*rewards.shape, 6):
        raise ValueError("Stage38 component scores must be [B,8,20,6]")
    if base_component_scores.shape != component_scores.shape:
        raise ValueError("Stage38 current/public component shapes differ")
    if current_selected_modes.shape != (batch, groups):
        raise ValueError("Stage38 current selected modes must be [B,8]")
    if scene_buckets.shape != (batch,):
        raise ValueError("Stage38 scene buckets must have shape [B]")
    constants = (
        positive_margin,
        negative_tolerance,
        advantage_scale,
        advantage_clip,
        mature_positive_multiplier,
        safety_tolerance,
    )
    if not all(math.isfinite(float(value)) for value in constants):
        raise ValueError("Stage38 target constants must be finite")
    if (
        int(elite_width) != 5
        or int(positive_challenger_width) != 2
        or int(active_pool_width) != 8
        or positive_margin < 0
        or negative_tolerance < 0
        or advantage_scale <= 0
        or advantage_clip <= 0
        or not 0 <= mature_positive_multiplier <= 1
        or safety_tolerance < 0
    ):
        raise ValueError("Stage38 target constants drifted")
    if (
        (current_selected_modes < 0) | (current_selected_modes >= modes)
    ).any():
        raise ValueError("Stage38 selected mode is out of range")

    current = rewards.detach().float()
    public = base_rewards.detach().float()
    current_components = component_scores.detach().float()
    public_components = base_component_scores.detach().float()
    current_valid = (
        valid_mask.bool()
        & torch.isfinite(current)
        & torch.isfinite(current_components).all(dim=-1)
    )
    public_valid = (
        base_valid_mask.bool()
        & torch.isfinite(public)
        & torch.isfinite(public_components).all(dim=-1)
    )
    pair_valid = current_valid & public_valid

    public_scores = public.masked_fill(~public_valid, -torch.inf)
    public_order = _stable_descending(public_scores)
    public_elite_modes = public_order[..., :elite_width]
    public_elite_values = _gather_modes(public, public_elite_modes)
    public_elite_valid = _gather_modes(
        public_valid, public_elite_modes
    )
    public_group_valid = public_elite_valid.all(dim=-1)
    public_elite_valid = (
        public_elite_valid & public_group_valid.unsqueeze(-1)
    )
    public_utility = torch.where(
        public_elite_valid,
        public_elite_values,
        torch.zeros_like(public_elite_values),
    ).sum(dim=-1)

    mode_ids = torch.arange(modes, device=rewards.device)
    elite_matches = (
        public_elite_modes.unsqueeze(-1)
        == mode_ids.view(1, 1, 1, modes)
    )
    is_public_elite = elite_matches.any(dim=2)
    elite_base_by_mode = (
        public_elite_values.unsqueeze(-1)
        * elite_matches.to(public_elite_values.dtype)
    ).sum(dim=2)

    aligned_utility = (
        public_utility.unsqueeze(-1) - elite_base_by_mode + current
    )
    expanded_elite = public_elite_values.unsqueeze(2).expand(
        batch, groups, modes, elite_width
    )
    challenger_pool = torch.cat(
        (expanded_elite, current.unsqueeze(-1)), dim=-1
    )
    challenger_utility = torch.topk(
        challenger_pool, elite_width, dim=-1
    ).values.sum(dim=-1)
    substituted_utility = torch.where(
        is_public_elite, aligned_utility, challenger_utility
    )
    # A missing current trajectory at a public-elite mode is itself a severe
    # retention failure.  Keep it in the differentiable replay and assign the
    # maximum negative credit.  Invalid non-elite challengers remain excluded.
    invalid_elite = (
        is_public_elite
        & public_group_valid.unsqueeze(-1)
        & ~current_valid
    )
    target_valid = (
        (pair_valid | invalid_elite) & public_group_valid.unsqueeze(-1)
    )
    finite_delta = substituted_utility - public_utility.unsqueeze(-1)
    invalid_credit = -(
        float(advantage_clip) * float(advantage_scale)
        + float(negative_tolerance)
    )
    delta_set = torch.where(
        invalid_elite,
        torch.full_like(current, invalid_credit),
        torch.where(target_valid, finite_delta, torch.zeros_like(current)),
    )

    safety_indices = torch.tensor(
        SAFETY_COMPONENT_INDICES, device=rewards.device
    )
    safety_regression = target_valid & (
        invalid_elite
        | (
            current_components.index_select(-1, safety_indices)
            < public_components.index_select(-1, safety_indices)
            - float(safety_tolerance)
        ).any(dim=-1)
    )
    positive = (
        target_valid
        & ~safety_regression
        & (delta_set > float(positive_margin))
    )
    negative = target_valid & (
        delta_set < -float(negative_tolerance)
    )
    advantages = torch.zeros_like(delta_set)
    advantages = torch.where(
        positive,
        (delta_set - float(positive_margin)) / float(advantage_scale),
        advantages,
    )
    advantages = torch.where(
        negative,
        (delta_set + float(negative_tolerance)) / float(advantage_scale),
        advantages,
    )
    advantages = advantages.clamp(
        -float(advantage_clip), float(advantage_clip)
    )
    advantages = torch.where(
        safety_regression,
        torch.full_like(advantages, -float(advantage_clip)),
        advantages,
    )
    mature = scene_buckets.long() == 3
    advantages = torch.where(
        mature[:, None, None] & (advantages > 0),
        advantages * float(mature_positive_multiplier),
        advantages,
    )
    advantages = torch.where(
        target_valid, advantages, torch.zeros_like(advantages)
    ).detach()

    # Active replay is deterministic and bounded: five public elites, the two
    # best safe positive non-elites, then the frozen selector's chosen mode.
    active_modes = torch.zeros(
        batch,
        groups,
        active_pool_width,
        device=rewards.device,
        dtype=torch.long,
    )
    active_valid = torch.zeros_like(active_modes, dtype=torch.bool)
    active_modes[..., :elite_width] = public_elite_modes
    active_valid[..., :elite_width] = (
        _gather_modes(target_valid, public_elite_modes)
        & public_elite_valid
    )

    challenger_mask = positive & ~is_public_elite
    challenger_scores = delta_set.masked_fill(
        ~challenger_mask, -torch.inf
    )
    challenger_order = _stable_descending(challenger_scores)
    challenger_modes = challenger_order[..., :positive_challenger_width]
    challenger_valid = torch.isfinite(
        _gather_modes(challenger_scores, challenger_modes)
    )
    challenger_start = elite_width
    challenger_end = elite_width + positive_challenger_width
    active_modes[..., challenger_start:challenger_end] = challenger_modes
    active_valid[..., challenger_start:challenger_end] = challenger_valid

    selected = current_selected_modes.long()
    selected_pair_valid = pair_valid.gather(
        -1, selected.unsqueeze(-1)
    ).squeeze(-1)
    duplicate_selected = (
        (active_modes[..., :challenger_end] == selected.unsqueeze(-1))
        & active_valid[..., :challenger_end]
    ).any(dim=-1)
    active_modes[..., -1] = selected
    active_valid[..., -1] = (
        selected_pair_valid
        & public_group_valid
        & ~duplicate_selected
    )
    duplicate_matrix = (
        active_modes.unsqueeze(-1) == active_modes.unsqueeze(-2)
    )
    duplicate_matrix = torch.triu(duplicate_matrix, diagonal=1)
    if (
        duplicate_matrix
        & active_valid.unsqueeze(-1)
        & active_valid.unsqueeze(-2)
    ).any():
        raise RuntimeError("Stage38 active replay contains duplicate modes")

    # Ceiling diagnostic: for every mode keep the better of the public elite
    # fallback and a safety-approved current trajectory, then take the top five.
    safe_current_valid = target_valid & ~safety_regression
    union_mode_values = current.masked_fill(
        ~safe_current_valid, -torch.inf
    ).clone()
    for elite_slot in range(elite_width):
        elite_mode = public_elite_modes[..., elite_slot]
        public_value = public_elite_values[..., elite_slot]
        existing = union_mode_values.gather(
            -1, elite_mode.unsqueeze(-1)
        ).squeeze(-1)
        union_mode_values.scatter_(
            -1,
            elite_mode.unsqueeze(-1),
            torch.maximum(existing, public_value).unsqueeze(-1),
        )
    union_values = torch.topk(
        union_mode_values, elite_width, dim=-1
    ).values
    union_valid = torch.isfinite(union_values).all(dim=-1)
    union_utility = torch.where(
        torch.isfinite(union_values),
        union_values,
        torch.zeros_like(union_values),
    ).sum(dim=-1)
    union_safe_oracle_gain = torch.where(
        public_group_valid & union_valid,
        union_utility - public_utility,
        torch.zeros_like(public_utility),
    )
    selected_delta = delta_set.gather(
        -1, selected.unsqueeze(-1)
    ).squeeze(-1)
    selected_delta_valid = target_valid.gather(
        -1, selected.unsqueeze(-1)
    ).squeeze(-1)
    current_elite_values = _gather_modes(current, public_elite_modes)
    current_elite_pair_valid = (
        _gather_modes(pair_valid, public_elite_modes)
        & public_elite_valid
    )
    current_elite_delta = torch.where(
        current_elite_pair_valid.all(dim=-1),
        current_elite_values.sum(dim=-1) - public_utility,
        torch.zeros_like(public_utility),
    )

    return {
        "pair_valid": pair_valid.detach(),
        "current_valid": current_valid.detach(),
        "target_valid": target_valid.detach(),
        "invalid_elite": invalid_elite.detach(),
        "public_group_valid": public_group_valid.detach(),
        "public_elite_modes": public_elite_modes.detach(),
        "public_elite_valid": public_elite_valid.detach(),
        "public_utility": public_utility.detach(),
        "substituted_utility": substituted_utility.detach(),
        "delta_set": delta_set.detach(),
        "is_public_elite": is_public_elite.detach(),
        "safety_regression": safety_regression.detach(),
        "positive": positive.detach(),
        "negative": negative.detach(),
        "advantages": advantages,
        "active_modes": active_modes.detach(),
        "active_valid": active_valid.detach(),
        "union_safe_oracle_gain": union_safe_oracle_gain.detach(),
        "selected_delta": selected_delta.detach(),
        "selected_delta_valid": selected_delta_valid.detach(),
        "current_public_elite_delta": current_elite_delta.detach(),
    }


def compute_elite_set_repair_objective(
    *,
    current_log_probs: torch.Tensor,
    reference_mean_kl: torch.Tensor,
    public_elite_bc_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    component_scores: torch.Tensor,
    base_rewards: torch.Tensor,
    base_valid_mask: torch.Tensor,
    base_component_scores: torch.Tensor,
    current_selected_modes: torch.Tensor,
    scene_buckets: torch.Tensor,
    active_modes: torch.Tensor,
    active_valid: torch.Tensor,
    public_elite_modes: torch.Tensor,
    public_elite_valid: torch.Tensor,
    elite_width: int = 5,
    positive_challenger_width: int = 2,
    active_pool_width: int = 8,
    positive_margin: float = 0.001,
    negative_tolerance: float = 0.0001,
    advantage_scale: float = 0.002,
    advantage_clip: float = 2.0,
    mature_positive_multiplier: float = 0.25,
    safety_tolerance: float = 1e-6,
    step_discount: float = 0.6,
    bc_weight: float = 0.1,
    kl_weight: float = 0.1,
    safety_kl_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Compute the differentiable ESCR policy, elite-BC, and active-KL loss."""
    constants = (
        step_discount,
        bc_weight,
        kl_weight,
        safety_kl_weight,
    )
    if not all(math.isfinite(float(value)) for value in constants):
        raise ValueError("Stage38 objective constants must be finite")
    if (
        not 0 < step_discount <= 1
        or min(bc_weight, kl_weight, safety_kl_weight) < 0
        or safety_kl_weight < kl_weight
    ):
        raise ValueError("Stage38 objective constants drifted")
    batch = rewards.shape[0]
    steps = current_log_probs.shape[-1]
    expected_current = (batch, 8, active_pool_width, steps)
    expected_elite = (batch, 8, elite_width, steps)
    if (
        current_log_probs.shape != expected_current
        or reference_mean_kl.shape != expected_current
        or public_elite_bc_log_probs.shape != expected_elite
    ):
        raise ValueError("Stage38 compact replay probability shapes drifted")
    if not all(torch.isfinite(value).all() for value in (
        current_log_probs,
        reference_mean_kl,
        public_elite_bc_log_probs,
    )):
        raise FloatingPointError("Stage38 compact replay is non-finite")

    targets = build_elite_set_repair_targets(
        rewards=rewards,
        valid_mask=valid_mask,
        component_scores=component_scores,
        base_rewards=base_rewards,
        base_valid_mask=base_valid_mask,
        base_component_scores=base_component_scores,
        current_selected_modes=current_selected_modes,
        scene_buckets=scene_buckets,
        elite_width=elite_width,
        positive_challenger_width=positive_challenger_width,
        active_pool_width=active_pool_width,
        positive_margin=positive_margin,
        negative_tolerance=negative_tolerance,
        advantage_scale=advantage_scale,
        advantage_clip=advantage_clip,
        mature_positive_multiplier=mature_positive_multiplier,
        safety_tolerance=safety_tolerance,
    )
    exact_pairs = (
        (active_modes, targets["active_modes"]),
        (active_valid.bool(), targets["active_valid"]),
        (public_elite_modes, targets["public_elite_modes"]),
        (public_elite_valid.bool(), targets["public_elite_valid"]),
    )
    if not all(torch.equal(actual, expected) for actual, expected in exact_pairs):
        raise RuntimeError("Stage38 replay selection drifted from ESCR targets")

    active_advantages = _gather_modes(
        targets["advantages"], active_modes
    )
    active_safety = _gather_modes(
        targets["safety_regression"], active_modes
    )
    indices = torch.arange(
        steps, device=rewards.device, dtype=current_log_probs.dtype
    )
    discounts = step_discount ** (steps - indices - 1)
    discounts = discounts.view(1, 1, 1, steps)

    policy_mask = (
        active_valid.bool() & (active_advantages != 0)
    ).unsqueeze(-1).expand_as(current_log_probs)
    policy_terms = (
        -current_log_probs
        * active_advantages.unsqueeze(-1)
        * discounts
    )
    policy_loss = _masked_scene_mean(
        policy_terms, policy_mask, current_log_probs
    )

    elite_mask = public_elite_valid.bool().unsqueeze(-1).expand_as(
        public_elite_bc_log_probs
    )
    elite_bc_terms = (
        -public_elite_bc_log_probs * discounts * float(bc_weight)
    )
    elite_bc_loss = _masked_scene_mean(
        elite_bc_terms, elite_mask, public_elite_bc_log_probs
    )

    kl_coefficients = torch.full_like(
        active_advantages, float(kl_weight)
    )
    kl_coefficients = torch.where(
        active_safety,
        torch.full_like(kl_coefficients, float(safety_kl_weight)),
        kl_coefficients,
    )
    kl_mask = active_valid.bool().unsqueeze(-1).expand_as(
        reference_mean_kl
    )
    active_kl_loss = _masked_scene_mean(
        reference_mean_kl * kl_coefficients.unsqueeze(-1),
        kl_mask,
        reference_mean_kl,
    )
    total = policy_loss + elite_bc_loss + active_kl_loss
    if not torch.isfinite(total):
        raise FloatingPointError("Stage38 total loss is non-finite")

    target_valid = targets["target_valid"]
    public_group_valid = targets["public_group_valid"]
    hard = scene_buckets.long() <= 1
    mature = scene_buckets.long() == 3
    return {
        "loss": total,
        "_target_delta_set": targets["delta_set"],
        "_target_advantages": targets["advantages"],
        "_target_union_safe_oracle_gain": targets[
            "union_safe_oracle_gain"
        ],
        "policy_loss": policy_loss.detach(),
        "elite_bc_loss": elite_bc_loss.detach(),
        "active_kl_loss": active_kl_loss.detach(),
        "delta_set_mean": _masked_detached_mean(
            targets["delta_set"], target_valid
        ),
        "positive_fraction": (
            targets["positive"].float().sum()
            / target_valid.float().sum().clamp_min(1.0)
        ).detach(),
        "negative_fraction": (
            targets["negative"].float().sum()
            / target_valid.float().sum().clamp_min(1.0)
        ).detach(),
        "positive_challenger_fraction": (
            (
                targets["positive"] & ~targets["is_public_elite"]
            ).float().sum()
            / target_valid.float().sum().clamp_min(1.0)
        ).detach(),
        "negative_elite_fraction": (
            (
                targets["negative"] & targets["is_public_elite"]
            ).float().sum()
            / target_valid.float().sum().clamp_min(1.0)
        ).detach(),
        "invalid_elite_fraction": (
            targets["invalid_elite"].float().sum()
            / target_valid.float().sum().clamp_min(1.0)
        ).detach(),
        "safety_veto_fraction": (
            targets["safety_regression"].float().sum()
            / target_valid.float().sum().clamp_min(1.0)
        ).detach(),
        "active_fraction": active_valid.float().mean().detach(),
        "union_safe_oracle_gain_mean": _masked_detached_mean(
            targets["union_safe_oracle_gain"], public_group_valid
        ),
        "selected_set_gain_mean": _masked_detached_mean(
            targets["selected_delta"], targets["selected_delta_valid"]
        ),
        "public_elite_current_delta_mean": _masked_detached_mean(
            targets["current_public_elite_delta"], public_group_valid
        ),
        "hard_union_safe_oracle_gain_mean": _masked_detached_mean(
            targets["union_safe_oracle_gain"],
            public_group_valid & hard[:, None],
        ),
        "mature_union_safe_oracle_gain_mean": _masked_detached_mean(
            targets["union_safe_oracle_gain"],
            public_group_valid & mature[:, None],
        ),
        "public_elite_exact_width": (
            public_elite_valid.sum(dim=-1).eq(elite_width).all()
            .to(rewards.dtype).detach()
        ),
        "active_pool_bounded": (
            active_valid.sum(dim=-1).le(active_pool_width).all()
            .to(rewards.dtype).detach()
        ),
        "advantage_abs_mean": _masked_detached_mean(
            targets["advantages"].abs(), target_valid
        ),
        "mean_current_log_prob": _masked_detached_mean(
            current_log_probs,
            active_valid.bool().unsqueeze(-1).expand_as(current_log_probs),
        ),
        "mean_exact_kl": _masked_detached_mean(
            reference_mean_kl,
            active_valid.bool().unsqueeze(-1).expand_as(reference_mean_kl),
        ),
    }
