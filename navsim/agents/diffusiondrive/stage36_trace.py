"""Deferred reward-dependent replay for Stage36 RGT-NCD-GRPO."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from navsim.agents.diffusiondrive.diffusion_grpo import (
    _gather_set_mode_pool,
    _replay_full_chain_actions,
    _replay_reference_mean_kl,
    _sample_full_chain_actions,
    _select_stage31_deployed_modes,
)
from navsim.agents.diffusiondrive.stage35_counterfactual import (
    counterfactual_donor_indices,
    union_selector_top2,
)
from navsim.agents.diffusiondrive.stage35_nested_trace import (
    _select_counterfactual_hybrids,
    _selector_pool,
)
from navsim.agents.diffusiondrive.stage36_reference_gated_tail import (
    build_reference_gated_tail_targets,
)


def _validate_stage36_head(head, group_size: int) -> None:
    if head.ref_policy is None:
        raise RuntimeError("Stage36 RGT-NCD requires a frozen public reference")
    if int(group_size) != 8:
        raise ValueError("Stage36 RGT-NCD requires exactly eight banks")
    if str(getattr(head._config, "inference_selector_source", "")) != (
        "trajectory_relative_harm_v3"
    ):
        raise ValueError("Stage36 requires the frozen Stage25 S-multi selector")
    if (
        int(head._stage24_selector_training_updates.item()) <= 0
        or int(head._stage25_selector_training_updates.item()) <= 0
        or not bool(head._stage24_calibration_loaded.item())
    ):
        raise RuntimeError("Stage36 requires a trained and calibrated selector")
    if head._generation_trust_projection_mode != "none":
        raise ValueError("Stage36 does not permit inference trust projection")


def collect_stage36_sampling_trace(
    head,
    group_size: int,
    ego_query: torch.Tensor,
    agents_query: torch.Tensor,
    bev_feature: torch.Tensor,
    bev_spatial_shape,
    status_encoding: torch.Tensor,
    global_img: torch.Tensor | None,
    selector_micro_batch_size: int | None = None,
) -> Dict[str, object]:
    """Sample paired banks and freeze selector-counterfactual decisions.

    Chain log-probability replay is deliberately deferred until PDM rewards
    have selected the public frontier and same-anchor upper tail.
    """
    _validate_stage36_head(head, group_size)
    batch = ego_query.shape[0]
    groups = int(group_size)
    modes = int(head.plan_anchor.shape[0])
    if modes != 20:
        raise RuntimeError("Stage36 requires complete 20-mode banks")
    clean = head.norm_odo(
        head.plan_anchor.unsqueeze(0).unsqueeze(1)
        .expand(batch, groups, -1, -1, -1)
        .reshape(batch, groups * modes, *head.plan_anchor.shape[1:])
    )
    timesteps = torch.full(
        (batch,),
        int(head._truncation_timestep),
        device=clean.device,
        dtype=torch.long,
    )
    initial = head.diffusion_scheduler.add_noise(
        original_samples=clean,
        noise=torch.randn_like(clean),
        timesteps=timesteps,
    ).detach()
    (
        current_states,
        current_actions,
        current_final,
        current_bank_logits,
        noise_bundle,
    ) = _sample_full_chain_actions(
        head,
        head.diff_decoder,
        initial,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img,
        return_noise_bundle=True,
    )
    with torch.no_grad():
        (
            public_states,
            public_actions,
            public_final,
            public_bank_logits,
        ) = _sample_full_chain_actions(
            head,
            head.ref_policy,
            initial,
            ego_query,
            agents_query,
            bev_feature,
            bev_spatial_shape,
            status_encoding,
            global_img,
            transition_noises=noise_bundle["transition_noises"],
            final_noise=noise_bundle["final_noise"],
        )
        current_candidates = head.denorm_odo(current_final).reshape(
            batch * groups, modes, 8, 3
        )
        public_candidates = head.denorm_odo(public_final).reshape(
            batch * groups, modes, 8, 3
        )
        current_logits = current_bank_logits.reshape(batch * groups, modes)
        public_logits = public_bank_logits.reshape(batch * groups, modes)
        repeated_bev = bev_feature.repeat_interleave(groups, dim=0)
        repeated_agents = agents_query.repeat_interleave(groups, dim=0)
        repeated_ego = ego_query.repeat_interleave(groups, dim=0)
        repeated_status = status_encoding.repeat_interleave(groups, dim=0)
        current_flat_modes, current_diagnostics = _select_stage31_deployed_modes(
            head,
            current_candidates,
            current_logits,
            repeated_bev,
            bev_spatial_shape,
            repeated_agents,
            repeated_ego,
            repeated_status,
        )
        public_flat_modes, public_diagnostics = _select_stage31_deployed_modes(
            head,
            public_candidates,
            public_logits,
            repeated_bev,
            bev_spatial_shape,
            repeated_agents,
            repeated_ego,
            repeated_status,
        )
        current_selected_modes = current_flat_modes.reshape(batch, groups)
        public_selected_modes = public_flat_modes.reshape(batch, groups)
        current_pool, current_pool_valid = _selector_pool(
            head,
            current_diagnostics,
            current_flat_modes,
            groups,
            modes,
        )
        public_pool, public_pool_valid = _selector_pool(
            head,
            public_diagnostics,
            public_flat_modes,
            groups,
            modes,
        )
        active_modes, active_valid = union_selector_top2(
            current_pool,
            current_pool_valid,
            public_pool,
            public_pool_valid,
        )
        donors = counterfactual_donor_indices(
            groups, device=current_candidates.device
        )
        hybrid_selected_modes = _select_counterfactual_hybrids(
            head,
            current_candidates.reshape(batch, groups, modes, 8, 3),
            current_logits.reshape(batch, groups, modes),
            active_modes,
            donors,
            ego_query,
            agents_query,
            bev_feature,
            bev_spatial_shape,
            status_encoding,
            selector_micro_batch_size=int(getattr(
                head._config, "stage36_selector_micro_batch_size", 16
            )) if selector_micro_batch_size is None else int(
                selector_micro_batch_size
            ),
        )
    return {
        "current_states": current_states,
        "current_actions": current_actions,
        "public_states": public_states,
        "public_actions": public_actions,
        "current_final": current_final,
        "public_final": public_final,
        "current_cls": current_bank_logits,
        "reference_cls": public_bank_logits,
        "current_selected_modes": current_selected_modes.detach(),
        "public_selected_modes": public_selected_modes.detach(),
        "active_modes": active_modes.detach(),
        "active_valid": active_valid.detach(),
        "hybrid_selected_modes": hybrid_selected_modes.detach(),
        "donor_indices": donors.detach(),
        "selector_mode_disagreement": (
            current_selected_modes != public_selected_modes
        ).float().mean().detach(),
        "current_selector_switch_rate": current_diagnostics[
            "switch"
        ].float().mean().detach(),
        "public_selector_switch_rate": public_diagnostics[
            "switch"
        ].float().mean().detach(),
        "sampled_chain_count": current_final.new_tensor(
            float(2 * groups * modes)
        ),
        "counterfactual_selector_count": current_final.new_tensor(
            float(groups * active_modes.shape[-1] * (groups - 1))
        ),
        "group_size": current_final.new_tensor(float(groups)),
    }


def _scatter_mode_mask(
    mask: torch.Tensor,
    modes: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    """Set a ``[B,G,M]`` mask from ``[B,G,K]`` mode indices."""
    if modes.shape != valid.shape or modes.shape[:2] != mask.shape[:2]:
        raise ValueError("Stage36 mode-mask scatter shape mismatch")
    batch = torch.arange(
        mask.shape[0], device=mask.device
    ).view(-1, 1, 1).expand_as(modes)
    groups = torch.arange(
        mask.shape[1], device=mask.device
    ).view(1, -1, 1).expand_as(modes)
    selected = valid.bool()
    mask[
        batch[selected], groups[selected], modes.long()[selected]
    ] = True


def _scatter_tail_mask(
    mask: torch.Tensor,
    group_indices: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    """Set a ``[B,G,M]`` mask from same-anchor ``[B,M,K]`` groups."""
    if group_indices.shape != valid.shape:
        raise ValueError("Stage36 tail-mask scatter shape mismatch")
    batch, modes, elite = group_indices.shape
    if mask.shape[0] != batch or mask.shape[2] != modes:
        raise ValueError("Stage36 tail-mask bank shape mismatch")
    batch_index = torch.arange(batch, device=mask.device).view(batch, 1, 1)
    batch_index = batch_index.expand(batch, modes, elite)
    mode_index = torch.arange(modes, device=mask.device).view(1, modes, 1)
    mode_index = mode_index.expand(batch, modes, elite)
    selected = valid.bool()
    mask[
        batch_index[selected],
        group_indices.long()[selected],
        mode_index[selected],
    ] = True


def _pack_mode_mask(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack a boolean ``[B,G,M]`` mask into stable ascending mode pools."""
    if mask.ndim != 3:
        raise ValueError("Stage36 replay mask must have shape [B,G,M]")
    batch, groups, modes = mask.shape
    counts = mask.sum(dim=-1)
    width = int(counts.max().item())
    if width <= 0 or width > modes:
        raise RuntimeError("Stage36 replay union has invalid width")
    packed = torch.zeros(
        batch, groups, width, device=mask.device, dtype=torch.long
    )
    valid = torch.zeros_like(packed, dtype=torch.bool)
    for batch_index in range(batch):
        for group_index in range(groups):
            indices = torch.nonzero(
                mask[batch_index, group_index], as_tuple=False
            ).flatten()
            count = int(indices.numel())
            if count:
                packed[batch_index, group_index, :count] = indices
                valid[batch_index, group_index, :count] = True
    return packed, valid


def _mode_to_slot(
    packed_modes: torch.Tensor,
    packed_valid: torch.Tensor,
    modes: int,
) -> torch.Tensor:
    result = torch.full(
        (*packed_modes.shape[:2], int(modes)),
        -1,
        device=packed_modes.device,
        dtype=torch.long,
    )
    slots = torch.arange(
        packed_modes.shape[-1], device=packed_modes.device
    ).view(1, 1, -1).expand_as(packed_modes)
    batch = torch.arange(
        packed_modes.shape[0], device=packed_modes.device
    ).view(-1, 1, 1).expand_as(packed_modes)
    groups = torch.arange(
        packed_modes.shape[1], device=packed_modes.device
    ).view(1, -1, 1).expand_as(packed_modes)
    selected = packed_valid.bool()
    result[
        batch[selected], groups[selected], packed_modes[selected]
    ] = slots[selected]
    return result


def _gather_active_replay(
    values: torch.Tensor,
    mode_to_slot: torch.Tensor,
    active_modes: torch.Tensor,
    active_valid: torch.Tensor,
) -> torch.Tensor:
    slots = mode_to_slot.gather(2, active_modes.long())
    if ((slots < 0) & active_valid.bool()).any():
        raise RuntimeError("Stage36 active NCD chain is absent from replay union")
    slots = slots.clamp_min(0)
    index = slots.unsqueeze(-1).expand(*slots.shape, values.shape[-1])
    return values.gather(2, index)


def _gather_tail_replay(
    values: torch.Tensor,
    mode_to_slot: torch.Tensor,
    group_indices: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    batch, modes, elite = group_indices.shape
    batch_index = torch.arange(
        batch, device=values.device
    ).view(batch, 1, 1).expand(batch, modes, elite)
    mode_index = torch.arange(
        modes, device=values.device
    ).view(1, modes, 1).expand(batch, modes, elite)
    slots = mode_to_slot[
        batch_index, group_indices.long(), mode_index
    ]
    if ((slots < 0) & valid.bool()).any():
        raise RuntimeError("Stage36 tail chain is absent from current replay union")
    slots = slots.clamp_min(0)
    return values[
        batch_index, group_indices.long(), slots
    ]


def _gather_frontier_replay(
    values: torch.Tensor,
    mode_to_slot: torch.Tensor,
    frontier_modes: torch.Tensor,
    retention_active: torch.Tensor,
) -> torch.Tensor:
    slots = mode_to_slot.gather(2, frontier_modes.long())
    if ((slots < 0) & retention_active.bool()).any():
        raise RuntimeError(
            "Stage36 retention teacher is absent from public replay union"
        )
    slots = slots.clamp_min(0)
    index = slots.unsqueeze(-1).expand(*slots.shape, values.shape[-1])
    return values.gather(2, index)


def finalize_stage36_reward_dependent_replay(
    head,
    trace: Dict[str, object],
    *,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    component_scores: torch.Tensor,
    base_rewards: torch.Tensor,
    base_valid_mask: torch.Tensor,
    base_component_scores: torch.Tensor,
    scene_buckets: torch.Tensor,
    ego_query: torch.Tensor,
    agents_query: torch.Tensor,
    bev_feature: torch.Tensor,
    bev_spatial_shape,
    status_encoding: torch.Tensor,
    global_img: torch.Tensor | None,
) -> Dict[str, torch.Tensor]:
    """Freeze reward-dependent replay sets and evaluate each chain once."""
    batch = rewards.shape[0]
    groups, modes = 8, 20
    rewards_bank = rewards.reshape(batch, groups, modes)
    valid_bank = valid_mask.reshape(batch, groups, modes)
    components_bank = component_scores.reshape(batch, groups, modes, 6)
    base_rewards_bank = base_rewards.reshape(batch, groups, modes)
    base_valid_bank = base_valid_mask.reshape(batch, groups, modes)
    base_components_bank = base_component_scores.reshape(
        batch, groups, modes, 6
    )
    targets = build_reference_gated_tail_targets(
        rewards=rewards_bank,
        valid_mask=valid_bank,
        component_scores=components_bank,
        base_rewards=base_rewards_bank,
        base_valid_mask=base_valid_bank,
        base_component_scores=base_components_bank,
        scene_buckets=scene_buckets,
        frontier_width=int(getattr(head._config, "stage36_frontier_width", 5)),
        elite_count=int(getattr(head._config, "stage36_tail_elite_count", 2)),
        retention_tolerance=float(getattr(
            head._config, "stage36_retention_tolerance", 1e-4
        )),
        tail_margin=float(getattr(head._config, "stage36_tail_margin", 0.001)),
        tail_scale=float(getattr(head._config, "stage36_tail_scale", 0.002)),
        tail_weight=float(getattr(head._config, "stage36_tail_weight", 0.25)),
        retention_weight=float(getattr(
            head._config, "stage36_retention_weight", 0.25
        )),
        mature_positive_multiplier=float(getattr(
            head._config, "stage36_mature_positive_multiplier", 0.25
        )),
        advantage_clip=float(getattr(
            head._config, "stage36_advantage_clip", 2.0
        )),
        safety_tolerance=float(getattr(
            head._config, "diffgrpo_safety_regression_tolerance", 1e-6
        )),
    )
    active_modes = trace["active_modes"]
    active_valid = trace["active_valid"]
    current_mask = torch.zeros(
        batch, groups, modes, device=rewards.device, dtype=torch.bool
    )
    public_mask = torch.zeros_like(current_mask)
    _scatter_mode_mask(current_mask, active_modes, active_valid)
    _scatter_mode_mask(public_mask, active_modes, active_valid)
    _scatter_tail_mask(
        current_mask,
        targets["tail_group_indices"],
        targets["tail_valid"],
    )
    _scatter_mode_mask(
        public_mask,
        targets["public_frontier_modes"],
        targets["retention_active"],
    )
    current_modes, current_union_valid = _pack_mode_mask(current_mask)
    public_modes, public_union_valid = _pack_mode_mask(public_mask)

    current_states = tuple(
        _gather_set_mode_pool(value, current_modes, modes)
        for value in trace["current_states"]
    )
    current_actions = tuple(
        _gather_set_mode_pool(value, current_modes, modes)
        for value in trace["current_actions"]
    )
    current_log_probs, _ = _replay_full_chain_actions(
        head,
        head.diff_decoder,
        current_states,
        current_actions,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img,
    )
    current_kl = _replay_reference_mean_kl(
        head,
        head.diff_decoder,
        head.ref_policy,
        current_states,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img,
    )
    public_states = tuple(
        _gather_set_mode_pool(value, public_modes, modes)
        for value in trace["public_states"]
    )
    public_actions = tuple(
        _gather_set_mode_pool(value, public_modes, modes)
        for value in trace["public_actions"]
    )
    public_log_probs, _ = _replay_full_chain_actions(
        head,
        head.diff_decoder,
        public_states,
        public_actions,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img,
    )
    current_width = current_modes.shape[-1]
    public_width = public_modes.shape[-1]
    steps = current_log_probs.shape[-1]
    current_values = current_log_probs.reshape(
        batch, groups, current_width, steps
    )
    current_kl_values = current_kl.reshape(
        batch, groups, current_width, steps
    )
    public_values = public_log_probs.reshape(
        batch, groups, public_width, steps
    )
    current_slots = _mode_to_slot(
        current_modes, current_union_valid, modes
    )
    public_slots = _mode_to_slot(
        public_modes, public_union_valid, modes
    )
    ncd_current = _gather_active_replay(
        current_values, current_slots, active_modes, active_valid
    )
    ncd_kl = _gather_active_replay(
        current_kl_values, current_slots, active_modes, active_valid
    )
    ncd_bc = _gather_active_replay(
        public_values, public_slots, active_modes, active_valid
    )
    tail_log_probs = _gather_tail_replay(
        current_values,
        current_slots,
        targets["tail_group_indices"],
        targets["tail_valid"],
    )
    tail_kl = _gather_tail_replay(
        current_kl_values,
        current_slots,
        targets["tail_group_indices"],
        targets["tail_valid"],
    )
    frontier_bc = _gather_frontier_replay(
        public_values,
        public_slots,
        targets["public_frontier_modes"],
        targets["retention_active"],
    )

    # Padded packed entries are evaluated for shape regularity but never enter
    # a loss. Report both physical and unique valid replay counts.
    current_unique = current_mask.sum().to(current_log_probs.dtype)
    public_unique = public_mask.sum().to(current_log_probs.dtype)
    return {
        "diffgrpo_current_log_probs": ncd_current,
        "diffgrpo_bc_log_probs": ncd_bc,
        "diffgrpo_reference_mean_kl": ncd_kl,
        "stage36_tail_current_log_probs": tail_log_probs,
        "stage36_tail_reference_mean_kl": tail_kl,
        "stage36_frontier_bc_log_probs": frontier_bc,
        "stage36_public_frontier_modes": targets[
            "public_frontier_modes"
        ],
        "stage36_public_frontier_valid": targets[
            "public_frontier_valid"
        ],
        "stage36_retention_coefficients": targets[
            "retention_coefficients"
        ],
        "stage36_tail_group_indices": targets["tail_group_indices"],
        "stage36_tail_valid": targets["tail_valid"],
        "stage36_current_replay_modes": current_modes.detach(),
        "stage36_current_replay_valid": current_union_valid.detach(),
        "stage36_public_replay_modes": public_modes.detach(),
        "stage36_public_replay_valid": public_union_valid.detach(),
        "stage36_current_unique_replay_count": current_unique.detach(),
        "stage36_public_unique_replay_count": public_unique.detach(),
        "stage36_current_physical_replay_count": current_log_probs.new_tensor(
            float(groups * current_width)
        ),
        "stage36_public_physical_replay_count": current_log_probs.new_tensor(
            float(groups * public_width)
        ),
        "stage36_reward_dependent_replay": current_log_probs.new_tensor(1.0),
        "stage36_replay_deduplicated": current_log_probs.new_tensor(1.0),
        "diffgrpo_num_denoising_steps": current_log_probs.new_tensor(
            float(steps)
        ),
    }
