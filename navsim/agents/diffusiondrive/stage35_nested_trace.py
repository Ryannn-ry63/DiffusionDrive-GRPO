"""Trace collector for Stage35 Nested Counterfactual Deployment GRPO."""

from __future__ import annotations

from typing import Dict

import torch

from navsim.agents.diffusiondrive.diffusion_grpo import (
    _gather_set_mode_pool,
    _replay_full_chain_actions,
    _replay_reference_mean_kl,
    _sample_full_chain_actions,
    _select_stage31_deployed_modes,
    _stage32_frontier_pool,
)
from navsim.agents.diffusiondrive.stage35_counterfactual import (
    counterfactual_donor_indices,
    union_selector_top2,
)


def _selector_pool(
    head,
    diagnostics: Dict[str, torch.Tensor],
    selected_modes: torch.Tensor,
    group_size: int,
    modes_per_set: int,
):
    """Return selected plus the best safe alternative for each bank."""
    return _stage32_frontier_pool(
        diagnostics,
        selected_modes,
        group_size,
        modes_per_set,
        float(getattr(head._config, "stage25_selector_risk_threshold", 1.0)),
        float(getattr(head._config, "stage35_frontier_risk_margin", 0.10)),
        float(getattr(head._config, "stage24_selector_ood_threshold", 1.0)),
        1,
    )


@torch.no_grad()
def _select_counterfactual_hybrids(
    head,
    candidate_banks: torch.Tensor,
    candidate_logits: torch.Tensor,
    active_modes: torch.Tensor,
    donor_indices: torch.Tensor,
    ego_query: torch.Tensor,
    agents_query: torch.Tensor,
    bev_feature: torch.Tensor,
    bev_spatial_shape,
    status_encoding: torch.Tensor,
    selector_micro_batch_size: int | None = None,
) -> torch.Tensor:
    """Replace one same-anchor candidate and rerun the frozen selector."""
    if candidate_banks.ndim != 5:
        raise ValueError("Stage35 candidate banks must be [B,G,M,8,3]")
    batch, groups, modes, horizon, state_dim = candidate_banks.shape
    if (groups, modes, horizon, state_dim) != (8, 20, 8, 3):
        raise ValueError("Stage35 freezes candidate banks at [B,8,20,8,3]")
    if candidate_logits.shape != (batch, groups, modes):
        raise ValueError("Stage35 candidate logits do not match candidate banks")
    if active_modes.shape != (batch, groups, 4):
        raise ValueError("Stage35 active modes must have shape [B,8,4]")
    if donor_indices.shape != (groups, groups - 1):
        raise ValueError("Stage35 donor table must have shape [8,7]")

    active_width = active_modes.shape[-1]
    donors_per_bank = donor_indices.shape[-1]
    hybrid = candidate_banks[:, :, None, None].expand(
        batch,
        groups,
        active_width,
        donors_per_bank,
        modes,
        horizon,
        state_dim,
    ).clone()
    hybrid_logits = candidate_logits[:, :, None, None].expand(
        batch,
        groups,
        active_width,
        donors_per_bank,
        modes,
    ).clone()

    shape = (batch, groups, active_width, donors_per_bank)
    batch_index = torch.arange(
        batch, device=candidate_banks.device
    ).view(batch, 1, 1, 1).expand(shape)
    group_index = torch.arange(
        groups, device=candidate_banks.device
    ).view(1, groups, 1, 1).expand(shape)
    slot_index = torch.arange(
        active_width, device=candidate_banks.device
    ).view(1, 1, active_width, 1).expand(shape)
    donor_slot = torch.arange(
        donors_per_bank, device=candidate_banks.device
    ).view(1, 1, 1, donors_per_bank).expand(shape)
    donor_bank = donor_indices.view(
        1, groups, 1, donors_per_bank
    ).expand(shape)
    mode_index = active_modes[..., None].expand(shape)
    replacement = candidate_banks[
        batch_index, donor_bank, mode_index
    ]
    replacement_logits = candidate_logits[
        batch_index, donor_bank, mode_index
    ]
    hybrid[
        batch_index,
        group_index,
        slot_index,
        donor_slot,
        mode_index,
    ] = replacement
    hybrid_logits[
        batch_index,
        group_index,
        slot_index,
        donor_slot,
        mode_index,
    ] = replacement_logits

    hybrids_per_scene = groups * active_width * donors_per_bank
    flat_candidates = hybrid.reshape(
        batch * hybrids_per_scene, modes, horizon, state_dim
    )
    flat_logits = hybrid_logits.reshape(batch * hybrids_per_scene, modes)
    scene_indices = torch.arange(
        batch, device=candidate_banks.device
    ).repeat_interleave(hybrids_per_scene)
    micro_batch = int(
        selector_micro_batch_size
        if selector_micro_batch_size is not None
        else getattr(head._config, "stage35_selector_micro_batch_size", 16)
    )
    if micro_batch <= 0:
        raise ValueError("Stage35 selector micro-batch size must be positive")

    selections = []
    for start in range(0, flat_candidates.shape[0], micro_batch):
        stop = min(start + micro_batch, flat_candidates.shape[0])
        scene = scene_indices[start:stop]
        selected, _ = _select_stage31_deployed_modes(
            head,
            flat_candidates[start:stop],
            flat_logits[start:stop],
            bev_feature.index_select(0, scene),
            bev_spatial_shape,
            agents_query.index_select(0, scene),
            ego_query.index_select(0, scene),
            status_encoding.index_select(0, scene),
        )
        selections.append(selected)
    return torch.cat(selections).reshape(
        batch, groups, active_width, donors_per_bank
    )


def collect_nested_counterfactual_deployment_trace(
    head,
    group_size: int,
    ego_query: torch.Tensor,
    agents_query: torch.Tensor,
    bev_feature: torch.Tensor,
    bev_spatial_shape,
    status_encoding: torch.Tensor,
    global_img: torch.Tensor | None,
) -> Dict[str, torch.Tensor]:
    """Sample G full banks and replay selector-relevant same-anchor chains."""
    if head.ref_policy is None:
        raise RuntimeError("Stage35 NCD-GRPO requires a frozen public reference")
    if int(group_size) != 8:
        raise ValueError("Stage35 NCD-GRPO requires exactly eight banks")
    if str(getattr(head._config, "inference_selector_source", "")) != (
        "trajectory_relative_harm_v3"
    ):
        raise ValueError("Stage35 requires the frozen Stage25 S-multi selector")
    if (
        int(head._stage24_selector_training_updates.item()) <= 0
        or int(head._stage25_selector_training_updates.item()) <= 0
        or not bool(head._stage24_calibration_loaded.item())
    ):
        raise RuntimeError("Stage35 requires a trained and calibrated selector")
    if head._generation_trust_projection_mode != "none":
        raise ValueError("Stage35 does not permit trust projection")

    batch = ego_query.shape[0]
    groups = int(group_size)
    modes = int(head.plan_anchor.shape[0])
    if modes != 20:
        raise RuntimeError("Stage35 requires complete 20-mode candidate banks")
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
        )

    current_pool_states = tuple(
        _gather_set_mode_pool(value, active_modes, modes)
        for value in current_states
    )
    current_pool_actions = tuple(
        _gather_set_mode_pool(value, active_modes, modes)
        for value in current_actions
    )
    current_log_probs, _ = _replay_full_chain_actions(
        head,
        head.diff_decoder,
        current_pool_states,
        current_pool_actions,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img,
    )
    reference_mean_kl = _replay_reference_mean_kl(
        head,
        head.diff_decoder,
        head.ref_policy,
        current_pool_states,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img,
    )
    public_pool_states = tuple(
        _gather_set_mode_pool(value, active_modes, modes)
        for value in public_states
    )
    public_pool_actions = tuple(
        _gather_set_mode_pool(value, active_modes, modes)
        for value in public_actions
    )
    bc_log_probs, _ = _replay_full_chain_actions(
        head,
        head.diff_decoder,
        public_pool_states,
        public_pool_actions,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img,
    )
    active_width = active_modes.shape[-1]
    expected = groups * active_width
    if current_log_probs.shape[:2] != (batch, expected):
        raise RuntimeError("Stage35 must replay exactly 8x4 current chains")
    if (
        bc_log_probs.shape != current_log_probs.shape
        or reference_mean_kl.shape != current_log_probs.shape
    ):
        raise RuntimeError("Stage35 probability replay shapes disagree")
    steps = current_log_probs.shape[-1]
    return {
        "trajectories": head.denorm_odo(current_final),
        "base_trajectories": head.denorm_odo(public_final),
        "current_cls": current_bank_logits,
        "reference_cls": public_bank_logits,
        "current_log_probs": current_log_probs.reshape(
            batch, groups, active_width, steps
        ),
        "bc_log_probs": bc_log_probs.reshape(
            batch, groups, active_width, steps
        ),
        "reference_mean_kl": reference_mean_kl.reshape(
            batch, groups, active_width, steps
        ),
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
        "sampled_chain_count": current_log_probs.new_tensor(
            float(2 * groups * modes)
        ),
        "replayed_chain_count": current_log_probs.new_tensor(
            float(2 * expected)
        ),
        "counterfactual_selector_count": current_log_probs.new_tensor(
            float(groups * active_width * (groups - 1))
        ),
        "group_size": current_log_probs.new_tensor(float(groups)),
        "num_denoising_steps": current_log_probs.new_tensor(float(steps)),
    }
