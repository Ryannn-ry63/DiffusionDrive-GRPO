"""Stage39 non-destructive challenger targets and objectives.

The public bank is an immutable fallback.  Only the physically independent
challenger policy carries gradients.  This module deliberately contains no
model code so the reward contract can be audited offline and unit tested.
"""

from __future__ import annotations

import math
from typing import Dict

import torch


STAGE39_PLAN_SHA256 = (
    "7e1e6142412778524714a0b919f971edf1d2bf14138b0efcf21fc9dd7c4d6f1b"
)
STAGE39_OBJECTIVE_REVISION = "non_destructive_challenger_set_grpo_v1"
STAGE39_BRANCHES = ("BC", "STD", "SET")
SAFETY_COMPONENT_INDICES = (0, 1, 3)


def _stable_descending(values: torch.Tensor) -> torch.Tensor:
    try:
        return torch.argsort(values, dim=-1, descending=True, stable=True)
    except TypeError as error:  # pragma: no cover - supported training builds
        raise RuntimeError("Stage39 requires stable torch.argsort") from error


def _gather_modes(values: torch.Tensor, modes: torch.Tensor) -> torch.Tensor:
    if values.ndim < 3 or modes.ndim != 3:
        raise ValueError("Stage39 gather expects [B,G,M,...] and [B,G,K]")
    index = modes.view(
        *modes.shape, *([1] * (values.ndim - 3))
    ).expand(*modes.shape, *values.shape[3:])
    return values.gather(2, index)


def _scene_normalized_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    anchor: torch.Tensor,
) -> torch.Tensor:
    """Average valid terms per scene before averaging scenes."""
    mask = mask.bool() & torch.isfinite(values)
    if values.shape != mask.shape or values.shape[0] != anchor.shape[0]:
        raise ValueError("Stage39 scene-normalized tensors have incompatible shapes")
    flattened_values = torch.where(mask, values, torch.zeros_like(values)).flatten(1)
    flattened_mask = mask.flatten(1)
    counts = flattened_mask.sum(dim=1)
    valid_scenes = counts > 0
    if not valid_scenes.any():
        return anchor.sum() * 0.0
    per_scene = flattened_values.sum(dim=1) / counts.clamp_min(1)
    return per_scene[valid_scenes].mean()


def _masked_detached_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool() & torch.isfinite(values)
    return (
        values[valid].float().mean().detach()
        if valid.any()
        else values.new_zeros(())
    )


def build_stage39_challenger_targets(
    *,
    challenger_rewards: torch.Tensor,
    challenger_valid_mask: torch.Tensor,
    challenger_component_scores: torch.Tensor,
    public_rewards: torch.Tensor,
    public_valid_mask: torch.Tensor,
    public_component_scores: torch.Tensor,
    elite_width: int = 5,
    positive_margin: float = 0.001,
    advantage_scale: float = 0.02,
    advantage_clip: float = 2.0,
    standard_deviation_floor: float = 1e-4,
    safety_tolerance: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Build standard-GRPO and public-frontier set-marginal targets."""
    if challenger_rewards.ndim != 3 or challenger_rewards.shape[1:] != (8, 20):
        raise ValueError("Stage39 rewards must have shape [B,8,20]")
    if (
        public_rewards.shape != challenger_rewards.shape
        or challenger_valid_mask.shape != challenger_rewards.shape
        or public_valid_mask.shape != challenger_rewards.shape
    ):
        raise ValueError("Stage39 reward/valid bank shapes differ")
    expected_components = (*challenger_rewards.shape, 6)
    if (
        challenger_component_scores.shape != expected_components
        or public_component_scores.shape != expected_components
    ):
        raise ValueError("Stage39 component banks must have shape [B,8,20,6]")
    constants = (
        positive_margin,
        advantage_scale,
        advantage_clip,
        standard_deviation_floor,
        safety_tolerance,
    )
    if not all(math.isfinite(float(value)) for value in constants):
        raise ValueError("Stage39 target constants must be finite")
    if (
        int(elite_width) != 5
        or positive_margin < 0
        or advantage_scale <= 0
        or advantage_clip <= 0
        or standard_deviation_floor <= 0
        or safety_tolerance < 0
    ):
        raise ValueError("Stage39 target constants drifted")

    challenger = challenger_rewards.detach().float()
    public = public_rewards.detach().float()
    challenger_components = challenger_component_scores.detach().float()
    public_components = public_component_scores.detach().float()
    challenger_valid = (
        challenger_valid_mask.bool()
        & torch.isfinite(challenger)
        & torch.isfinite(challenger_components).all(dim=-1)
    )
    public_valid = (
        public_valid_mask.bool()
        & torch.isfinite(public)
        & torch.isfinite(public_components).all(dim=-1)
    )

    public_order = _stable_descending(
        public.masked_fill(~public_valid, -torch.inf)
    )
    public_elite_modes = public_order[..., :elite_width]
    public_elite_rewards = _gather_modes(public, public_elite_modes)
    public_elite_valid = _gather_modes(public_valid, public_elite_modes)
    public_group_valid = public_elite_valid.all(dim=-1)
    public_frontier = public_elite_rewards.mean(dim=-1)

    safety_index = torch.tensor(
        SAFETY_COMPONENT_INDICES,
        device=challenger.device,
        dtype=torch.long,
    )
    public_elite_components = _gather_modes(
        public_components, public_elite_modes
    ).index_select(-1, safety_index)
    # A challenger must not be worse than the weakest member of the immutable
    # public elite set on any multiplicative safety component.
    public_safety_floor = public_elite_components.amin(dim=2)
    challenger_safety = challenger_components.index_select(-1, safety_index)
    safety_regression = (
        challenger_safety
        < public_safety_floor.unsqueeze(2) - float(safety_tolerance)
    ).any(dim=-1)
    target_valid = challenger_valid & public_group_valid.unsqueeze(-1)
    strict_safe = target_valid & ~safety_regression

    expanded_public = public_elite_rewards.unsqueeze(2).expand(
        *challenger.shape, elite_width
    )
    marginal_pool = torch.cat(
        (expanded_public, challenger.unsqueeze(-1)), dim=-1
    )
    marginal_utility = torch.topk(
        marginal_pool, elite_width, dim=-1
    ).values.mean(dim=-1)
    raw_set_delta = (marginal_utility - public_frontier.unsqueeze(-1)).clamp_min(0.0)
    set_delta = torch.where(
        strict_safe, raw_set_delta, torch.zeros_like(raw_set_delta)
    )

    set_advantages = torch.zeros_like(set_delta)
    positive = strict_safe & (set_delta > float(positive_margin))
    set_advantages = torch.where(
        positive,
        ((set_delta - float(positive_margin)) / float(advantage_scale)).clamp(
            max=float(advantage_clip)
        ),
        set_advantages,
    )
    veto = ~target_valid | safety_regression
    set_advantages = torch.where(
        veto,
        torch.full_like(set_advantages, -float(advantage_clip)),
        set_advantages,
    ).detach()

    # Standard GRPO is per anchor across the eight stochastic rollouts.
    standard_values = challenger.masked_fill(~challenger_valid, torch.nan)
    standard_mean = torch.nanmean(standard_values, dim=1, keepdim=True)
    centered = standard_values - standard_mean
    finite = torch.isfinite(centered)
    squared = torch.where(finite, centered.square(), torch.zeros_like(centered))
    counts = finite.sum(dim=1, keepdim=True)
    standard_std = (
        squared.sum(dim=1, keepdim=True)
        / counts.clamp_min(1)
    ).sqrt().clamp_min(float(standard_deviation_floor))
    standard_advantages = (centered / standard_std).clamp(
        -float(advantage_clip), float(advantage_clip)
    )
    standard_advantages = torch.where(
        challenger_valid,
        standard_advantages,
        torch.full_like(standard_advantages, -float(advantage_clip)),
    )
    standard_advantages = torch.nan_to_num(
        standard_advantages,
        nan=0.0,
        posinf=float(advantage_clip),
        neginf=-float(advantage_clip),
    ).detach()

    safe_challenger = challenger.masked_fill(~strict_safe, -torch.inf)
    union = torch.cat((public_elite_rewards, safe_challenger), dim=-1)
    union_top5 = torch.topk(union, elite_width, dim=-1).values
    union_valid = torch.isfinite(union_top5).all(dim=-1)
    union_safe_oracle_gain = torch.where(
        public_group_valid & union_valid,
        union_top5.mean(dim=-1) - public_frontier,
        torch.zeros_like(public_frontier),
    )
    return {
        "challenger_valid": challenger_valid.detach(),
        "public_group_valid": public_group_valid.detach(),
        "public_elite_modes": public_elite_modes.detach(),
        "public_elite_valid": public_elite_valid.detach(),
        "public_frontier": public_frontier.detach(),
        "public_safety_floor": public_safety_floor.detach(),
        "strict_safe": strict_safe.detach(),
        "safety_regression": safety_regression.detach(),
        "set_delta": set_delta.detach(),
        "set_positive": positive.detach(),
        "set_advantages": set_advantages,
        "standard_advantages": standard_advantages,
        "union_safe_oracle_gain": union_safe_oracle_gain.detach(),
    }


def compute_stage39_challenger_objective(
    *,
    branch: str,
    challenger_log_probs: torch.Tensor,
    challenger_reference_kl: torch.Tensor,
    public_bc_log_probs: torch.Tensor,
    challenger_rewards: torch.Tensor,
    challenger_valid_mask: torch.Tensor,
    challenger_component_scores: torch.Tensor,
    public_rewards: torch.Tensor,
    public_valid_mask: torch.Tensor,
    public_component_scores: torch.Tensor,
    elite_width: int = 5,
    positive_margin: float = 0.001,
    advantage_scale: float = 0.02,
    advantage_clip: float = 2.0,
    standard_deviation_floor: float = 1e-4,
    safety_tolerance: float = 1e-6,
    step_discount: float = 0.6,
    kl_weight: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Compute one of the three preregistered Stage39 training objectives."""
    branch = str(branch).upper()
    if branch not in STAGE39_BRANCHES:
        raise ValueError(f"Unknown Stage39 branch: {branch}")
    if challenger_log_probs.ndim != 4:
        raise ValueError("Stage39 log probabilities must be [B,8,20,S]")
    batch, groups, modes, steps = challenger_log_probs.shape
    expected = (batch, 8, 20, steps)
    if (groups, modes) != (8, 20) or any(
        value.shape != expected
        for value in (
            challenger_reference_kl,
            public_bc_log_probs,
        )
    ):
        raise ValueError("Stage39 replay probability shapes drifted")
    if not all(torch.isfinite(value).all() for value in (
        challenger_log_probs,
        challenger_reference_kl,
        public_bc_log_probs,
    )):
        raise FloatingPointError("Stage39 replay contains non-finite values")
    if (
        not math.isfinite(float(step_discount))
        or not 0 < step_discount <= 1
        or not math.isfinite(float(kl_weight))
        or kl_weight < 0
    ):
        raise ValueError("Stage39 loss constants drifted")

    targets = build_stage39_challenger_targets(
        challenger_rewards=challenger_rewards,
        challenger_valid_mask=challenger_valid_mask,
        challenger_component_scores=challenger_component_scores,
        public_rewards=public_rewards,
        public_valid_mask=public_valid_mask,
        public_component_scores=public_component_scores,
        elite_width=elite_width,
        positive_margin=positive_margin,
        advantage_scale=advantage_scale,
        advantage_clip=advantage_clip,
        standard_deviation_floor=standard_deviation_floor,
        safety_tolerance=safety_tolerance,
    )
    indices = torch.arange(
        steps,
        device=challenger_log_probs.device,
        dtype=challenger_log_probs.dtype,
    )
    discounts = (
        float(step_discount) ** (steps - indices - 1)
    ).view(1, 1, 1, steps)
    valid = targets["challenger_valid"].unsqueeze(-1).expand_as(
        challenger_log_probs
    )
    bc_mask = (
        public_valid_mask.bool()
        & torch.isfinite(public_rewards)
    ).unsqueeze(-1).expand_as(public_bc_log_probs)
    bc_loss = _scene_normalized_mean(
        -public_bc_log_probs * discounts,
        bc_mask,
        public_bc_log_probs,
    )
    if branch == "BC":
        policy_loss = challenger_log_probs.sum() * 0.0
        kl_loss = challenger_reference_kl.sum() * 0.0
        loss = bc_loss
        active_advantages = torch.zeros_like(targets["set_advantages"])
    else:
        active_advantages = (
            targets["standard_advantages"]
            if branch == "STD"
            else targets["set_advantages"]
        )
        policy_mask = valid & (active_advantages != 0).unsqueeze(-1)
        policy_loss = _scene_normalized_mean(
            -challenger_log_probs
            * active_advantages.unsqueeze(-1)
            * discounts,
            policy_mask,
            challenger_log_probs,
        )
        kl_loss = _scene_normalized_mean(
            challenger_reference_kl * float(kl_weight),
            valid,
            challenger_reference_kl,
        )
        loss = policy_loss + kl_loss

    zero = loss.detach() * 0.0
    return {
        "loss": loss,
        "policy_loss": policy_loss.detach(),
        "bc_loss": bc_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "advantage_abs_mean": _masked_detached_mean(
            active_advantages.abs(), targets["challenger_valid"]
        ),
        "standard_positive_fraction": _masked_detached_mean(
            (targets["standard_advantages"] > 0).float(),
            targets["challenger_valid"],
        ),
        "set_positive_fraction": _masked_detached_mean(
            targets["set_positive"].float(), targets["challenger_valid"]
        ),
        "safety_veto_fraction": _masked_detached_mean(
            targets["safety_regression"].float(),
            targets["challenger_valid"],
        ),
        "set_delta_mean": _masked_detached_mean(
            targets["set_delta"], targets["strict_safe"]
        ),
        "union_safe_oracle_gain_mean": _masked_detached_mean(
            targets["union_safe_oracle_gain"],
            targets["public_group_valid"],
        ),
        "public_group_valid_fraction": (
            targets["public_group_valid"].float().mean().detach()
            if targets["public_group_valid"].numel()
            else zero
        ),
        "_target_standard_advantages": targets["standard_advantages"],
        "_target_set_advantages": targets["set_advantages"],
        "_target_set_delta": targets["set_delta"],
    }

