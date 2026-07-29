"""Reward-dependent compact replay for Stage38 ESCR."""

from __future__ import annotations

from typing import Dict

import torch

from navsim.agents.diffusiondrive.diffusion_grpo import (
    _gather_set_mode_pool,
    _replay_full_chain_actions,
    _replay_reference_mean_kl,
)
from navsim.agents.diffusiondrive.stage36_trace import (
    collect_stage36_sampling_trace,
)
from navsim.agents.diffusiondrive.stage38_elite_set_repair import (
    build_elite_set_repair_targets,
)


def collect_stage38_sampling_trace(
    head,
    group_size: int,
    ego_query: torch.Tensor,
    agents_query: torch.Tensor,
    bev_feature: torch.Tensor,
    bev_spatial_shape,
    status_encoding: torch.Tensor,
    global_img: torch.Tensor | None,
) -> Dict[str, object]:
    """Reuse the audited paired common-noise sampler from Stage36."""
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
            head._config, "stage38_selector_micro_batch_size", 16
        )),
    )


def finalize_stage38_reward_dependent_replay(
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
    """Freeze ESCR targets, then replay only eight current and five public modes."""
    batch = rewards.shape[0]
    groups, modes = 8, 20
    targets = build_elite_set_repair_targets(
        rewards=rewards.reshape(batch, groups, modes),
        valid_mask=valid_mask.reshape(batch, groups, modes),
        component_scores=component_scores.reshape(batch, groups, modes, 6),
        base_rewards=base_rewards.reshape(batch, groups, modes),
        base_valid_mask=base_valid_mask.reshape(batch, groups, modes),
        base_component_scores=base_component_scores.reshape(
            batch, groups, modes, 6
        ),
        current_selected_modes=trace["current_selected_modes"],
        scene_buckets=scene_buckets,
        elite_width=int(getattr(head._config, "stage38_elite_width", 5)),
        positive_challenger_width=int(getattr(
            head._config, "stage38_positive_challenger_width", 2
        )),
        active_pool_width=int(getattr(
            head._config, "stage38_active_pool_width", 8
        )),
        positive_margin=float(getattr(
            head._config, "stage38_positive_margin", 0.001
        )),
        negative_tolerance=float(getattr(
            head._config, "stage38_negative_tolerance", 0.0001
        )),
        advantage_scale=float(getattr(
            head._config, "stage38_advantage_scale", 0.002
        )),
        advantage_clip=float(getattr(
            head._config, "stage38_advantage_clip", 2.0
        )),
        mature_positive_multiplier=float(getattr(
            head._config, "stage38_mature_positive_multiplier", 0.25
        )),
        safety_tolerance=float(getattr(
            head._config, "diffgrpo_safety_regression_tolerance", 1e-6
        )),
    )
    current_modes = targets["active_modes"]
    current_valid = targets["active_valid"]
    public_modes = targets["public_elite_modes"]
    public_valid = targets["public_elite_valid"]

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
    current_reference_kl = _replay_reference_mean_kl(
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
    public_bc_log_probs, _ = _replay_full_chain_actions(
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

    active_width = current_modes.shape[-1]
    elite_width = public_modes.shape[-1]
    steps = current_log_probs.shape[-1]
    current_log_probs = current_log_probs.reshape(
        batch, groups, active_width, steps
    )
    current_reference_kl = current_reference_kl.reshape(
        batch, groups, active_width, steps
    )
    public_bc_log_probs = public_bc_log_probs.reshape(
        batch, groups, elite_width, steps
    )
    return {
        "diffgrpo_current_log_probs": current_log_probs,
        "diffgrpo_reference_mean_kl": current_reference_kl,
        "stage38_public_elite_bc_log_probs": public_bc_log_probs,
        "stage38_active_modes": current_modes,
        "stage38_active_valid": current_valid,
        "stage38_public_elite_modes": public_modes,
        "stage38_public_elite_valid": public_valid,
        "stage38_delta_set": targets["delta_set"],
        "stage38_escr_advantages": targets["advantages"],
        "stage38_union_safe_oracle_gain": targets[
            "union_safe_oracle_gain"
        ],
        "stage38_current_replay_modes": current_modes,
        "stage38_current_replay_valid": current_valid,
        "stage38_public_replay_modes": public_modes,
        "stage38_public_replay_valid": public_valid,
        "stage38_current_unique_replay_count": current_valid.sum().detach(),
        "stage38_public_unique_replay_count": public_valid.sum().detach(),
        "stage38_current_physical_replay_count": current_log_probs.new_tensor(
            float(groups * active_width)
        ),
        "stage38_public_physical_replay_count": current_log_probs.new_tensor(
            float(groups * elite_width)
        ),
        "stage38_reward_dependent_replay": current_log_probs.new_tensor(1.0),
        "stage38_replay_deduplicated": current_log_probs.new_tensor(1.0),
        "diffgrpo_num_denoising_steps": current_log_probs.new_tensor(
            float(steps)
        ),
    }
