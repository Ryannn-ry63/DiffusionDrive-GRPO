"""Deferred all-frontier replay for Stage37 BPD-GRPO."""

from __future__ import annotations

from typing import Dict

import torch

from navsim.agents.diffusiondrive.diffusion_grpo import (
    _gather_set_mode_pool,
    _replay_full_chain_actions,
    _replay_reference_mean_kl,
)
from navsim.agents.diffusiondrive.stage36_reference_gated_tail import (
    build_reference_gated_tail_targets,
)
from navsim.agents.diffusiondrive.stage36_trace import (
    _gather_active_replay,
    _gather_frontier_replay,
    _gather_tail_replay,
    _mode_to_slot,
    _pack_mode_mask,
    _scatter_mode_mask,
    _scatter_tail_mask,
    collect_stage36_sampling_trace,
)


def collect_stage37_sampling_trace(
    head,
    group_size: int,
    ego_query: torch.Tensor,
    agents_query: torch.Tensor,
    bev_feature: torch.Tensor,
    bev_spatial_shape,
    status_encoding: torch.Tensor,
    global_img: torch.Tensor | None,
) -> Dict[str, object]:
    """Reuse the audited Stage36 paired sampler and counterfactual selector."""
    return collect_stage36_sampling_trace(
        head,
        group_size,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img,
        selector_micro_batch_size=int(getattr(
            head._config, "stage37_selector_micro_batch_size", 16
        )),
    )


def finalize_stage37_reward_dependent_replay(
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
    """Replay active, tail, and every public-top-five chain exactly once."""
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
        frontier_width=int(getattr(head._config, "stage37_frontier_width", 5)),
        elite_count=int(getattr(head._config, "stage37_tail_elite_count", 2)),
        retention_tolerance=float(getattr(
            head._config, "stage37_frontier_regression_tolerance", 1e-4
        )),
        tail_margin=float(getattr(head._config, "stage37_tail_margin", 0.001)),
        tail_scale=float(getattr(
            head._config, "stage37_advantage_scale", 0.002
        )),
        tail_weight=float(getattr(head._config, "stage37_tail_weight", 0.25)),
        retention_weight=float(getattr(
            head._config, "stage37_frontier_weight", 0.25
        )),
        mature_positive_multiplier=0.0,
        advantage_clip=float(getattr(
            head._config, "stage37_advantage_clip", 2.0
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
    # Stage37 protects all public-top-five modes at both denoising state
    # distributions, not only already-regressed entries.
    _scatter_mode_mask(
        current_mask,
        targets["public_frontier_modes"],
        targets["public_frontier_valid"],
    )
    _scatter_mode_mask(
        public_mask,
        targets["public_frontier_modes"],
        targets["public_frontier_valid"],
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
    public_kl = _replay_reference_mean_kl(
        head,
        head.diff_decoder,
        head.ref_policy,
        public_states,
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
    public_kl_values = public_kl.reshape(
        batch, groups, public_width, steps
    )
    current_slots = _mode_to_slot(current_modes, current_union_valid, modes)
    public_slots = _mode_to_slot(public_modes, public_union_valid, modes)
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
    current_frontier_kl = _gather_frontier_replay(
        current_kl_values,
        current_slots,
        targets["public_frontier_modes"],
        targets["public_frontier_valid"],
    )
    public_frontier_kl = _gather_frontier_replay(
        public_kl_values,
        public_slots,
        targets["public_frontier_modes"],
        targets["public_frontier_valid"],
    )

    return {
        "diffgrpo_current_log_probs": ncd_current,
        "diffgrpo_bc_log_probs": ncd_bc,
        "diffgrpo_reference_mean_kl": ncd_kl,
        "stage37_tail_current_log_probs": tail_log_probs,
        "stage37_current_frontier_mean_kl": current_frontier_kl,
        "stage37_public_frontier_mean_kl": public_frontier_kl,
        "stage37_public_frontier_modes": targets["public_frontier_modes"],
        "stage37_public_frontier_valid": targets["public_frontier_valid"],
        "stage37_tail_group_indices": targets["tail_group_indices"],
        "stage37_tail_valid": targets["tail_valid"],
        "stage37_current_replay_modes": current_modes.detach(),
        "stage37_current_replay_valid": current_union_valid.detach(),
        "stage37_public_replay_modes": public_modes.detach(),
        "stage37_public_replay_valid": public_union_valid.detach(),
        "stage37_current_unique_replay_count": current_mask.sum().detach(),
        "stage37_public_unique_replay_count": public_mask.sum().detach(),
        "stage37_current_physical_replay_count": current_log_probs.new_tensor(
            float(groups * current_width)
        ),
        "stage37_public_physical_replay_count": current_log_probs.new_tensor(
            float(groups * public_width)
        ),
        "stage37_all_frontier_replay": current_log_probs.new_tensor(1.0),
        "stage37_replay_deduplicated": current_log_probs.new_tensor(1.0),
        "diffgrpo_num_denoising_steps": current_log_probs.new_tensor(
            float(steps)
        ),
    }
