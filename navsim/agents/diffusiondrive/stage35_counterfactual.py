"""Nested counterfactual deployment credit for Stage35 NCD-GRPO."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch


STAGE35_OBJECTIVE_REVISION = "nested_counterfactual_deployment_v1"
STAGE35_PLAN_SHA256 = (
    "301e5fb37066c34f6a3e224be08fd1ca435cc9a8d961121869cf6dcd21d2fae7"
)


def union_selector_top2(
    current_modes: torch.Tensor,
    current_valid: torch.Tensor,
    public_modes: torch.Tensor,
    public_valid: torch.Tensor,
    *,
    max_modes: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the stable unique union of two selected+alternative pools."""
    if (
        current_modes.ndim != 3
        or public_modes.shape != current_modes.shape
        or current_valid.shape != current_modes.shape
        or public_valid.shape != current_modes.shape
        or current_modes.shape[-1] != 2
    ):
        raise ValueError("Stage35 selector pools must have shape [B,G,2]")
    if max_modes != 4:
        raise ValueError("Stage35 freezes the active pool width at four")
    batch, groups, _ = current_modes.shape
    result = torch.zeros(
        batch, groups, max_modes,
        device=current_modes.device, dtype=torch.long,
    )
    valid = torch.zeros_like(result, dtype=torch.bool)
    for batch_index in range(batch):
        for group_index in range(groups):
            write = 0
            for values, masks in (
                (current_modes, current_valid),
                (public_modes, public_valid),
            ):
                for slot in range(2):
                    if not bool(masks[batch_index, group_index, slot]):
                        continue
                    mode = int(values[batch_index, group_index, slot])
                    if any(
                        bool(valid[batch_index, group_index, prior])
                        and int(result[batch_index, group_index, prior]) == mode
                        for prior in range(write)
                    ):
                        continue
                    if write >= max_modes:
                        raise RuntimeError("Stage35 active selector union overflow")
                    result[batch_index, group_index, write] = mode
                    valid[batch_index, group_index, write] = True
                    write += 1
    return result, valid


def counterfactual_donor_indices(
    groups: int, *, device: torch.device | None = None
) -> torch.Tensor:
    """Return every other bank as a deterministic same-scene donor."""
    if int(groups) < 2:
        raise ValueError("counterfactual donor groups must be at least two")
    rows = [
        [donor for donor in range(int(groups)) if donor != target]
        for target in range(int(groups))
    ]
    return torch.tensor(rows, device=device, dtype=torch.long)


def compute_counterfactual_contributions(
    *,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    selected_modes: torch.Tensor,
    active_modes: torch.Tensor,
    active_valid: torch.Tensor,
    hybrid_selected_modes: torch.Tensor,
    donor_indices: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute hard leave-one-bank-out deployment contributions.

    ``hybrid_selected_modes[b,g,a,d]`` is the selector output after replacing
    active mode ``a`` in target bank ``g`` by the same anchor from donor ``d``.
    """
    if rewards.ndim != 3 or valid_mask.shape != rewards.shape:
        raise ValueError("Stage35 candidate rewards must have shape [B,G,M]")
    batch, groups, modes = rewards.shape
    if selected_modes.shape != (batch, groups):
        raise ValueError("Stage35 selected modes must have shape [B,G]")
    if active_modes.ndim != 3 or active_modes.shape[:2] != (batch, groups):
        raise ValueError("Stage35 active modes must have shape [B,G,A]")
    if active_valid.shape != active_modes.shape:
        raise ValueError("Stage35 active validity shape mismatch")
    active_width = active_modes.shape[-1]
    if donor_indices.shape != (groups, groups - 1):
        raise ValueError("Stage35 donor table must have shape [G,G-1]")
    if hybrid_selected_modes.shape != (
        batch, groups, active_width, groups - 1
    ):
        raise ValueError("Stage35 hybrid selected modes shape mismatch")
    if (
        (selected_modes < 0).any()
        or (selected_modes >= modes).any()
        or (active_modes < 0).any()
        or (active_modes >= modes).any()
        or (hybrid_selected_modes < 0).any()
        or (hybrid_selected_modes >= modes).any()
    ):
        raise ValueError("Stage35 mode index lies outside the candidate bank")

    valid_mask = valid_mask.bool() & torch.isfinite(rewards)
    batch_grid = torch.arange(batch, device=rewards.device)[:, None]
    group_grid = torch.arange(groups, device=rewards.device)[None, :]
    original_rewards = rewards[batch_grid, group_grid, selected_modes]
    original_valid = valid_mask[batch_grid, group_grid, selected_modes]

    contributions = torch.zeros(
        batch, groups, active_width,
        device=rewards.device, dtype=rewards.dtype,
    )
    contribution_valid = torch.zeros_like(contributions, dtype=torch.bool)
    hybrid_reward_mean = torch.zeros_like(contributions)
    nonzero_selection_change = torch.zeros_like(contributions, dtype=torch.bool)
    for group_index in range(groups):
        donors = donor_indices[group_index]
        for slot in range(active_width):
            mode = active_modes[:, group_index, slot]
            slot_valid = active_valid[:, group_index, slot]
            selected = hybrid_selected_modes[:, group_index, slot]
            donor_selected = selected == mode[:, None]
            target_bank = torch.full_like(selected, group_index)
            selected_bank = torch.where(
                donor_selected,
                donors.view(1, -1).expand(batch, -1),
                target_bank,
            )
            selected_rewards = rewards[
                torch.arange(batch, device=rewards.device)[:, None],
                selected_bank,
                selected,
            ]
            selected_valid = valid_mask[
                torch.arange(batch, device=rewards.device)[:, None],
                selected_bank,
                selected,
            ]
            donor_valid = selected_valid.all(dim=-1)
            valid = slot_valid & original_valid[:, group_index] & donor_valid
            mean_reward = selected_rewards.mean(dim=-1)
            value = original_rewards[:, group_index] - mean_reward
            contributions[:, group_index, slot] = torch.where(
                valid, value, torch.zeros_like(value)
            )
            hybrid_reward_mean[:, group_index, slot] = torch.where(
                valid, mean_reward, torch.zeros_like(mean_reward)
            )
            contribution_valid[:, group_index, slot] = valid
            original_mode = selected_modes[:, group_index].unsqueeze(-1)
            same_source = (
                (selected == original_mode)
                & ~(donor_selected & (mode[:, None] == original_mode))
            )
            nonzero_selection_change[:, group_index, slot] = (
                valid & ~same_source.all(dim=-1)
            )

    return {
        "contributions": contributions,
        "valid_mask": contribution_valid,
        "hybrid_reward_mean": hybrid_reward_mean,
        "deployment_influence": nonzero_selection_change,
    }


def compute_same_anchor_advantages(
    *,
    contributions: torch.Tensor,
    valid_mask: torch.Tensor,
    active_modes: torch.Tensor,
    num_modes: int,
    eps: float = 1e-3,
    clip: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """Normalize counterfactual credit only among replicas of one anchor."""
    if (
        contributions.ndim != 3
        or valid_mask.shape != contributions.shape
        or active_modes.shape != contributions.shape
    ):
        raise ValueError("Stage35 counterfactual tensors must share [B,G,A]")
    if eps <= 0 or clip <= 0 or int(num_modes) <= 0:
        raise ValueError("Stage35 advantage constants are invalid")
    batch, _, _ = contributions.shape
    advantages = torch.zeros_like(contributions)
    group_valid = torch.zeros_like(contributions, dtype=torch.bool)
    group_std = torch.zeros(
        batch, int(num_modes),
        device=contributions.device, dtype=contributions.dtype,
    )
    group_count = torch.zeros_like(group_std, dtype=torch.long)
    finite = valid_mask.bool() & torch.isfinite(contributions)
    for batch_index in range(batch):
        for mode in range(int(num_modes)):
            mask = finite[batch_index] & (
                active_modes[batch_index] == mode
            )
            values = contributions[batch_index][mask]
            count = int(values.numel())
            group_count[batch_index, mode] = count
            if count < 2:
                continue
            mean = values.mean()
            std = torch.sqrt((values - mean).square().mean())
            group_std[batch_index, mode] = std
            if float(std.detach()) <= eps:
                continue
            advantages[batch_index][mask] = (
                (values - mean) / std.clamp_min(float(eps))
            ).clamp(-float(clip), float(clip))
            group_valid[batch_index][mask] = True
    return {
        "advantages": advantages.detach(),
        "group_valid_mask": group_valid,
        "group_std": group_std.detach(),
        "group_count": group_count.detach(),
    }


def _gather_modes(values: torch.Tensor, modes: torch.Tensor) -> torch.Tensor:
    """Gather [B,G,...] values from a [B,G,M,...] candidate bank."""
    if values.ndim < 3 or modes.shape != values.shape[:2]:
        raise ValueError("Stage35 gather shape mismatch")
    index = modes.view(*modes.shape, 1, *([1] * (values.ndim - 3)))
    index = index.expand(*modes.shape, 1, *values.shape[3:])
    return values.gather(2, index).squeeze(2)


def _selected_active_slots(
    active_modes: torch.Tensor,
    active_valid: torch.Tensor,
    selected_modes: torch.Tensor,
) -> torch.Tensor:
    matches = active_valid & (active_modes == selected_modes.unsqueeze(-1))
    if not matches.any(dim=-1).all() or (matches.sum(dim=-1) != 1).any():
        raise RuntimeError("Stage35 active pool must contain each selected mode once")
    return matches.long().argmax(dim=-1)


def compute_nested_counterfactual_deployment_objective(
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
    active_modes: torch.Tensor,
    active_valid: torch.Tensor,
    hybrid_selected_modes: torch.Tensor,
    donor_indices: torch.Tensor,
    scene_buckets: torch.Tensor,
    counterfactual_weight: float = 0.25,
    mature_positive_multiplier: float = 0.25,
    advantage_eps: float = 1e-3,
    delta_scale_floor: float = 0.002,
    headroom_low: float = 0.75,
    headroom_high: float = 0.90,
    rank_weight: float = 0.5,
    advantage_clip: float = 2.0,
    safety_tolerance: float = 1e-6,
    step_discount: float = 0.6,
    bc_weight: float = 0.1,
    kl_weight: float = 0.1,
    safety_kl_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Combine DPEL bank credit with same-anchor counterfactual credit."""
    if current_log_probs.ndim != 4:
        raise ValueError("Stage35 log probabilities must have shape [B,8,4,S]")
    batch, groups, active_width, steps = current_log_probs.shape
    modes = rewards.shape[-1] if rewards.ndim == 3 else -1
    if (groups, active_width, modes) != (8, 4, 20):
        raise ValueError("Stage35 freezes G=8, active width=4, and 20 modes")
    if (
        bc_log_probs.shape != current_log_probs.shape
        or reference_mean_kl.shape != current_log_probs.shape
        or active_modes.shape != (batch, groups, active_width)
        or active_valid.shape != active_modes.shape
        or rewards.shape != (batch, groups, modes)
        or valid_mask.shape != rewards.shape
        or base_rewards.shape != rewards.shape
        or base_valid_mask.shape != rewards.shape
        or component_scores.shape != (batch, groups, modes, 6)
        or base_component_scores.shape != component_scores.shape
        or current_selected_modes.shape != (batch, groups)
        or public_selected_modes.shape != (batch, groups)
        or scene_buckets.shape != (batch,)
    ):
        raise ValueError("Stage35 objective tensor shape mismatch")
    constants = (
        counterfactual_weight, mature_positive_multiplier, advantage_eps,
        delta_scale_floor, headroom_low, headroom_high, rank_weight,
        advantage_clip, safety_tolerance, step_discount, bc_weight, kl_weight,
        safety_kl_weight,
    )
    if not all(math.isfinite(float(value)) for value in constants):
        raise ValueError("Stage35 objective constants must be finite")
    if (
        counterfactual_weight < 0
        or not 0 <= mature_positive_multiplier <= 1
        or advantage_eps <= 0
        or delta_scale_floor <= 0
        or not 0 <= headroom_low < headroom_high <= 1
        or rank_weight < 0
        or advantage_clip <= 0
        or safety_tolerance < 0
        or not 0 < step_discount <= 1
        or min(bc_weight, kl_weight, safety_kl_weight) < 0
        or safety_kl_weight < kl_weight
    ):
        raise ValueError("Stage35 objective constants are invalid")
    if not (
        torch.isfinite(current_log_probs).all()
        and torch.isfinite(bc_log_probs).all()
        and torch.isfinite(reference_mean_kl).all()
    ):
        raise FloatingPointError("Stage35 probability tensors must be finite")

    active_valid = active_valid.bool()
    current_slot = _selected_active_slots(
        active_modes, active_valid, current_selected_modes
    )
    public_slot = _selected_active_slots(
        active_modes, active_valid, public_selected_modes
    )
    current_selected_logp = _gather_modes(
        current_log_probs, current_slot
    )
    public_selected_bc = _gather_modes(bc_log_probs, public_slot)
    selected_kl = _gather_modes(reference_mean_kl, current_slot)

    current_reward = _gather_modes(rewards, current_selected_modes).float()
    public_reward = _gather_modes(base_rewards, public_selected_modes).float()
    current_valid = _gather_modes(valid_mask, current_selected_modes).bool()
    public_valid = _gather_modes(base_valid_mask, public_selected_modes).bool()
    current_components = _gather_modes(
        component_scores, current_selected_modes
    ).float()
    public_components = _gather_modes(
        base_component_scores, public_selected_modes
    ).float()
    pair_valid = (
        current_valid & public_valid
        & torch.isfinite(current_reward) & torch.isfinite(public_reward)
        & torch.isfinite(current_components).all(dim=-1)
        & torch.isfinite(public_components).all(dim=-1)
    )
    count = pair_valid.sum(dim=-1)
    group_valid = count >= 2
    denom = count.clamp_min(1).float()
    current_zero = torch.where(
        pair_valid, current_reward, torch.zeros_like(current_reward)
    )
    reward_mean = current_zero.sum(dim=-1) / denom
    centered = torch.where(
        pair_valid,
        current_reward - reward_mean.unsqueeze(-1),
        torch.zeros_like(current_reward),
    )
    reward_std = torch.sqrt(centered.square().sum(dim=-1) / denom)
    reward_z = torch.where(
        pair_valid & (reward_std > advantage_eps).unsqueeze(-1),
        centered / reward_std.clamp_min(advantage_eps).unsqueeze(-1),
        torch.zeros_like(centered),
    ).clamp(-advantage_clip, advantage_clip)
    delta = torch.where(
        pair_valid,
        current_reward - public_reward,
        torch.zeros_like(current_reward),
    )
    delta_mean = delta.sum(dim=-1) / denom
    delta_centered = torch.where(
        pair_valid,
        delta - delta_mean.unsqueeze(-1),
        torch.zeros_like(delta),
    )
    delta_std = torch.sqrt(delta_centered.square().sum(dim=-1) / denom)
    delta_z = (
        delta_centered
        / delta_std.clamp_min(delta_scale_floor).unsqueeze(-1)
    ).clamp(-advantage_clip, advantage_clip)
    public_quality = torch.where(
        pair_valid, public_reward, torch.zeros_like(public_reward)
    ).sum(dim=-1) / denom
    headroom = (
        (headroom_high - public_quality) / (headroom_high - headroom_low)
    ).clamp(0.0, 1.0)
    raw_outer = delta_z + rank_weight * headroom.unsqueeze(-1) * reward_z
    outer = torch.where(
        delta > 0,
        raw_outer.clamp_min(0.0),
        torch.where(delta < 0, raw_outer.clamp_max(0.0), raw_outer),
    )
    safety_indices = torch.tensor((0, 1, 3), device=rewards.device)
    outer_safety = pair_valid & (
        current_components.index_select(-1, safety_indices)
        < public_components.index_select(-1, safety_indices)
        - safety_tolerance
    ).any(dim=-1)
    outer = torch.where(
        outer_safety, torch.full_like(outer, -advantage_clip), outer
    )
    outer = torch.where(
        pair_valid & group_valid.unsqueeze(-1),
        outer.clamp(-advantage_clip, advantage_clip),
        torch.zeros_like(outer),
    ).detach()

    cf = compute_counterfactual_contributions(
        rewards=rewards.float(),
        valid_mask=valid_mask,
        selected_modes=current_selected_modes,
        active_modes=active_modes,
        active_valid=active_valid,
        hybrid_selected_modes=hybrid_selected_modes,
        donor_indices=donor_indices,
    )
    inner = compute_same_anchor_advantages(
        contributions=cf["contributions"],
        valid_mask=cf["valid_mask"],
        active_modes=active_modes,
        num_modes=modes,
        eps=advantage_eps,
        clip=advantage_clip,
    )
    inner_advantage = inner["advantages"]
    component_index = active_modes[..., None].expand(
        batch, groups, active_width, 6
    )
    active_current_components = component_scores.gather(
        2, component_index
    ).float()
    active_base_components = base_component_scores.gather(
        2, component_index
    ).float()
    inner_safety = active_valid & (
        active_current_components.index_select(-1, safety_indices)
        < active_base_components.index_select(-1, safety_indices)
        - safety_tolerance
    ).any(dim=-1)
    inner_advantage = torch.where(
        inner_safety,
        torch.full_like(inner_advantage, -advantage_clip),
        inner_advantage,
    )
    mature = scene_buckets.long() == 3
    inner_advantage = torch.where(
        mature[:, None, None] & (inner_advantage > 0),
        inner_advantage * mature_positive_multiplier,
        inner_advantage,
    )
    outer_slots = torch.zeros_like(inner_advantage)
    outer_slots.scatter_(2, current_slot.unsqueeze(-1), outer.unsqueeze(-1))
    total_advantage = (
        outer_slots + counterfactual_weight * inner_advantage
    ).clamp(-advantage_clip, advantage_clip).detach()
    policy_mask = (
        active_valid & (
            (outer_slots != 0)
            | inner["group_valid_mask"]
            | inner_safety
        )
    ).unsqueeze(-1)
    indices = torch.arange(
        steps, device=rewards.device, dtype=current_log_probs.dtype
    )
    discounts = step_discount ** (steps - indices - 1)
    terms = (
        -current_log_probs
        * total_advantage.unsqueeze(-1)
        * discounts.view(1, 1, 1, -1)
    )
    policy_mask_full = policy_mask.expand_as(current_log_probs)
    policy_count = policy_mask_full.sum(dim=(1, 2, 3)).clamp_min(1).float()
    policy_per_scene = (
        terms * policy_mask_full
    ).sum(dim=(1, 2, 3)) / policy_count
    policy_scene_valid = policy_mask_full.flatten(1).any(dim=-1)
    policy_loss = (
        policy_per_scene[policy_scene_valid].mean()
        if policy_scene_valid.any()
        else current_log_probs.sum() * 0.0
    )

    selected_mask = pair_valid.unsqueeze(-1).expand_as(public_selected_bc)
    selected_count = selected_mask.sum(dim=(1, 2)).clamp_min(1).float()
    bc_terms = (
        -public_selected_bc
        * discounts.view(1, 1, -1)
        * bc_weight
    )
    bc_per_scene = (
        bc_terms * selected_mask
    ).sum(dim=(1, 2)) / selected_count
    bc_loss = (
        bc_per_scene[group_valid].mean()
        if group_valid.any()
        else bc_log_probs.sum() * 0.0
    )
    kl_coeff = torch.full_like(delta, kl_weight)
    kl_coeff = torch.where(
        outer_safety, torch.full_like(kl_coeff, safety_kl_weight), kl_coeff
    )
    kl_terms = selected_kl * kl_coeff.unsqueeze(-1)
    kl_per_scene = (
        kl_terms * selected_mask
    ).sum(dim=(1, 2)) / selected_count
    kl_loss = (
        kl_per_scene[group_valid].mean()
        if group_valid.any()
        else reference_mean_kl.sum() * 0.0
    )
    total = policy_loss + bc_loss + kl_loss
    valid_active = active_valid.float().sum().clamp_min(1.0)
    non_owner = active_valid & (
        active_modes != current_selected_modes.unsqueeze(-1)
    )
    nonzero_cf = cf["valid_mask"] & (
        cf["contributions"].abs() > 1e-4
    )
    scene_nonowner_influence = (
        nonzero_cf & non_owner
    ).flatten(1).any(dim=-1)

    return {
        "loss": total,
        "policy_loss": policy_loss,
        "bc_loss": bc_loss,
        "reference_kl_loss": kl_loss,
        "advantages": total_advantage,
        "outer_advantages": outer,
        "counterfactual_advantages": inner_advantage.detach(),
        "deployment_delta_mean": delta[pair_valid].mean().detach(),
        "counterfactual_credit_mean": (
            cf["contributions"][cf["valid_mask"]].mean().detach()
            if cf["valid_mask"].any() else rewards.new_zeros(())
        ),
        "counterfactual_nonzero_fraction": (
            nonzero_cf.float().sum().detach() / valid_active
        ),
        "nonowner_influence_scene_fraction": (
            scene_nonowner_influence.float().mean().detach()
        ),
        "counterfactual_group_valid_fraction": (
            inner["group_valid_mask"].float().sum().detach() / valid_active
        ),
        "deployment_influence_fraction": (
            cf["deployment_influence"].float().sum().detach() / valid_active
        ),
        "outer_positive_fraction": (
            (outer > 0).float().sum().detach()
            / pair_valid.float().sum().clamp_min(1.0)
        ),
        "inner_positive_fraction": (
            ((inner_advantage > 0) & active_valid).float().sum().detach()
            / valid_active
        ),
        "safety_override_fraction": (
            inner_safety.float().sum().detach() / valid_active
        ),
        "mean_exact_kl": (
            selected_kl[selected_mask].mean().detach()
            if selected_mask.any() else selected_kl.sum() * 0.0
        ),
        "mean_current_log_prob": (
            current_log_probs[policy_mask_full].mean().detach()
            if policy_mask_full.any() else current_log_probs.sum() * 0.0
        ),
        "mean_bc_weight": rewards.new_tensor(float(bc_weight)),
        "mean_kl_weight": kl_coeff[pair_valid].mean().detach(),
        "discount_first": discounts[0].detach(),
        "discount_last": discounts[-1].detach(),
        "candidate_level_trace": rewards.new_tensor(1.0),
        "same_anchor_only": rewards.new_tensor(1.0),
    }
