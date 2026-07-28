"""Stage34 selector diagnostics over a mode-aligned Stage30-style trace."""

from __future__ import annotations

import torch

from navsim.agents.diffusiondrive.diffusion_grpo import (
    _select_stage31_deployed_modes,
)


@torch.no_grad()
def build_stage34_trace_diagnostics(
    head,
    trace,
    ego_query,
    agents_query,
    bev_feature,
    bev_spatial_shape,
    status_encoding,
):
    """Apply the frozen Stage25 selector to both aligned candidate banks."""
    current = trace["trajectories"].detach()
    public = trace["base_trajectories"].detach()
    current_logits = trace["current_cls"].detach()
    public_logits = trace["reference_cls"].detach()
    current_selected, current_diagnostics = _select_stage31_deployed_modes(
        head,
        current,
        current_logits,
        bev_feature,
        bev_spatial_shape,
        agents_query,
        ego_query,
        status_encoding,
    )
    public_selected, public_diagnostics = _select_stage31_deployed_modes(
        head,
        public,
        public_logits,
        bev_feature,
        bev_spatial_shape,
        agents_query,
        ego_query,
        status_encoding,
    )
    return {
        "stage34_current_selected_modes": current_selected.detach(),
        "stage34_public_selected_modes": public_selected.detach(),
        "stage34_current_eligible_mask": current_diagnostics[
            "eligible"
        ].detach(),
        "stage34_public_eligible_mask": public_diagnostics[
            "eligible"
        ].detach(),
        "stage34_current_risk_ucb": current_diagnostics["risk_ucb"].detach(),
        "stage34_public_risk_ucb": public_diagnostics["risk_ucb"].detach(),
        "stage34_current_delta_lcb": current_diagnostics[
            "delta_lcb"
        ].detach(),
        "stage34_public_delta_lcb": public_diagnostics["delta_lcb"].detach(),
        "stage34_current_ood_distance": current_diagnostics[
            "ood_distance"
        ].detach(),
        "stage34_public_ood_distance": public_diagnostics[
            "ood_distance"
        ].detach(),
        "stage34_selector_mode_disagreement": (
            (current_selected != public_selected).float().mean().detach()
        ),
        "stage34_current_selector_switch_rate": current_diagnostics[
            "switch"
        ].float().mean().detach(),
        "stage34_public_selector_switch_rate": public_diagnostics[
            "switch"
        ].float().mean().detach(),
        "stage34_mode_index_alignment": current_logits.new_tensor(1.0),
        "stage34_candidate_level_trace": current_logits.new_tensor(1.0),
    }

