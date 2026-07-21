"""Trajectory-conditioned value prediction and conservative top-2 selection."""

from __future__ import annotations

import hashlib
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.diffusiondrive.modules.blocks import (
    GridSampleCrossBEVAttention,
    gen_sineembed_for_position,
)


PDM_COMPONENT_COUNT = 6
PDM_SAFETY_INDICES = (0, 1, 3)
PDM_WEIGHTED_INDICES = (2, 3, 4, 5)
PDM_WEIGHTED_WEIGHTS = (10.0, 5.0, 2.0, 0.0)


def compose_pdm_score(component_scores: torch.Tensor) -> torch.Tensor:
    """Compose PDM components using the scorer's registered metric weights."""
    if component_scores.shape[-1] != PDM_COMPONENT_COUNT:
        raise ValueError("PDM component tensor must end in six components")
    multipliers = component_scores[..., 0] * component_scores[..., 1]
    weights = component_scores.new_tensor(PDM_WEIGHTED_WEIGHTS)
    weighted = component_scores[..., 2:6]
    weighted_score = (weighted * weights).sum(dim=-1) / weights.sum()
    return multipliers * weighted_score


def deterministic_bootstrap_mask(
    tokens: Sequence[str], num_heads: int, device: torch.device, keep_fraction: float = 0.8
) -> torch.Tensor:
    """Return a reproducible token-level bootstrap mask with one live head minimum."""
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must satisfy 0 < fraction <= 1")
    rows = []
    cutoff = int(round(keep_fraction * 10_000))
    for token in tokens:
        row = []
        for head in range(num_heads):
            digest = hashlib.sha256(f"stage15:{token}:{head}".encode()).digest()
            row.append(int.from_bytes(digest[:4], "little") % 10_000 < cutoff)
        if not any(row):
            digest = hashlib.sha256(f"stage15:fallback:{token}".encode()).digest()
            row[int.from_bytes(digest[:4], "little") % num_heads] = True
        rows.append(row)
    return torch.tensor(rows, dtype=torch.bool, device=device)


class TrajectoryValueSelector(nn.Module):
    """Predict PDM components from final trajectories and frozen scene context."""

    def __init__(self, config, num_heads: int = 3):
        super().__init__()
        if num_heads <= 0:
            raise ValueError("value selector requires at least one head")
        self.num_heads = int(num_heads)
        d_model = int(config.tf_d_model)
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(8 * 64, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )
        self.cross_bev = GridSampleCrossBEVAttention(
            embed_dims=d_model,
            num_heads=int(config.tf_num_head),
            num_points=8,
            config=config,
            in_bev_dims=d_model,
        )
        self.cross_agents = nn.MultiheadAttention(
            d_model, int(config.tf_num_head), dropout=0.0, batch_first=True
        )
        self.cross_ego = nn.MultiheadAttention(
            d_model, int(config.tf_num_head), dropout=0.0, batch_first=True
        )
        self.status_projection = nn.Linear(d_model, d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, PDM_COMPONENT_COUNT),
                )
                for _ in range(self.num_heads)
            ]
        )

    def forward(
        self,
        trajectories: torch.Tensor,
        bev_feature: torch.Tensor,
        bev_spatial_shape,
        agents_query: torch.Tensor,
        ego_query: torch.Tensor,
        status_encoding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if trajectories.ndim != 4 or trajectories.shape[-2:] != (8, 3):
            raise ValueError("value selector trajectories must have shape [B,M,8,3]")
        position_embedding = gen_sineembed_for_position(
            trajectories[..., :2], hidden_dim=64
        ).flatten(-2)
        query = self.trajectory_encoder(position_embedding)
        query = self.cross_bev(
            query, trajectories[..., :2], bev_feature, bev_spatial_shape
        )
        query = query + self.cross_agents(query, agents_query, agents_query)[0]
        query = query + self.cross_ego(query, ego_query, ego_query)[0]
        status = status_encoding
        if status.ndim == 2:
            status = status.unsqueeze(1)
        query = self.context_norm(query + self.status_projection(status))

        component_predictions = torch.stack(
            [torch.sigmoid(head(query)) for head in self.heads], dim=2
        )
        score_predictions = compose_pdm_score(component_predictions)
        component_mean = component_predictions.mean(dim=2)
        component_std = component_predictions.std(dim=2, unbiased=False)
        score_mean = score_predictions.mean(dim=2)
        score_std = score_predictions.std(dim=2, unbiased=False)
        return {
            "component_predictions": component_predictions,
            "score_predictions": score_predictions,
            "component_mean": component_mean,
            "component_std": component_std,
            "score_mean": score_mean,
            "score_std": score_std,
        }


def select_conservative_top2(
    reference_logits: torch.Tensor,
    component_mean: torch.Tensor,
    component_std: torch.Tensor,
    score_mean: torch.Tensor,
    margin: float,
    safety_threshold: float = 0.9,
    confidence_z: float = 1.64,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Select the reference top-2 challenger only when safety/value gates pass."""
    if reference_logits.ndim != 2 or reference_logits.shape != score_mean.shape:
        raise ValueError("reference logits and value scores must have shape [B,M]")
    if component_mean.shape != (*reference_logits.shape, PDM_COMPONENT_COUNT):
        raise ValueError("component mean must have shape [B,M,6]")
    if component_std.shape != component_mean.shape:
        raise ValueError("component mean/std shapes differ")
    if reference_logits.shape[1] < 2:
        raise ValueError("value top-2 selection requires at least two modes")
    if margin < 0:
        raise ValueError("value selector calibration margin must be non-negative")
    if not 0.0 <= safety_threshold <= 1.0:
        raise ValueError("value selector safety threshold must be within [0,1]")
    if confidence_z < 0:
        raise ValueError("value selector confidence z must be non-negative")

    top2 = torch.topk(reference_logits, k=2, dim=-1).indices
    fallback = top2[:, 0]
    challenger = top2[:, 1]
    gather_components = top2.unsqueeze(-1).expand(-1, -1, PDM_COMPONENT_COUNT)
    top_components = component_mean.gather(1, gather_components)
    top_component_std = component_std.gather(1, gather_components)
    lower = top_components - confidence_z * top_component_std
    safety_index = torch.tensor(
        PDM_SAFETY_INDICES, device=reference_logits.device, dtype=torch.long
    )
    fallback_safety = lower[:, 0].index_select(-1, safety_index)
    challenger_safety = lower[:, 1].index_select(-1, safety_index)
    safe_absolute = (challenger_safety >= safety_threshold).all(dim=-1)
    safe_relative = (challenger_safety >= fallback_safety).all(dim=-1)
    top_scores = score_mean.gather(1, top2)
    predicted_advantage = top_scores[:, 1] - top_scores[:, 0]
    switch = safe_absolute & safe_relative & (predicted_advantage > margin)
    selected = torch.where(switch, challenger, fallback)
    return selected, {
        "fallback_mode": fallback,
        "challenger_mode": challenger,
        "switch": switch,
        "predicted_advantage": predicted_advantage,
        "fallback_score": top_scores[:, 0],
        "challenger_score": top_scores[:, 1],
        "fallback_safety_lower": fallback_safety,
        "challenger_safety_lower": challenger_safety,
    }


def compute_value_selector_loss(
    predictions: Dict[str, torch.Tensor], reward_gap: float = 0.01
) -> Dict[str, torch.Tensor]:
    """Train component/value heads and a focused frozen-reference top-2 ranker."""
    components = predictions["value_component_predictions"]
    scores = predictions["value_score_predictions"]
    targets = predictions["component_scores"].detach().float()
    rewards = predictions["raw_rewards"].detach().float()
    valid = predictions["reward_valid_mask"].bool()
    bootstrap = predictions["value_bootstrap_mask"].bool()
    reference_logits = predictions["value_reference_logits"].detach().float()
    group_size = int(predictions["value_group_size"])
    if components.ndim != 4 or components.shape[-1] != PDM_COMPONENT_COUNT:
        raise ValueError("value component predictions must have shape [B,M,H,6]")
    batch, modes, heads, _ = components.shape
    if scores.shape != (batch, modes, heads):
        raise ValueError("value score prediction shape mismatch")
    if targets.shape != (batch, modes, PDM_COMPONENT_COUNT):
        raise ValueError("value component target shape mismatch")
    if rewards.shape != valid.shape or rewards.shape != (batch, modes):
        raise ValueError("value reward/valid shapes mismatch")
    if bootstrap.shape != (batch, heads):
        raise ValueError("value bootstrap mask shape mismatch")
    if reference_logits.shape != (batch, modes):
        raise ValueError("value reference-logit shape mismatch")
    if modes % group_size != 0 or group_size < 2:
        raise ValueError("invalid value selector group size")

    finite = valid & torch.isfinite(rewards) & torch.isfinite(targets).all(dim=-1)
    train_mask = finite.unsqueeze(-1) & bootstrap.unsqueeze(1)
    expanded_target = targets.unsqueeze(2).expand_as(components)
    safety_prediction = components[..., :2].clamp(1e-6, 1 - 1e-6)
    safety_target = expanded_target[..., :2].clamp(0.0, 1.0)
    safety_mask = train_mask.unsqueeze(-1).expand_as(safety_prediction)
    safety_loss = (
        F.binary_cross_entropy(safety_prediction[safety_mask], safety_target[safety_mask])
        if safety_mask.any()
        else components.sum() * 0.0
    )
    continuous_prediction = components[..., 2:]
    continuous_target = expanded_target[..., 2:]
    continuous_mask = train_mask.unsqueeze(-1).expand_as(continuous_prediction)
    component_loss = (
        F.smooth_l1_loss(
            continuous_prediction[continuous_mask], continuous_target[continuous_mask]
        )
        if continuous_mask.any()
        else components.sum() * 0.0
    )
    score_mask = train_mask
    expanded_rewards = rewards.unsqueeze(-1).expand_as(scores)
    pdms_loss = (
        F.smooth_l1_loss(scores[score_mask], expanded_rewards[score_mask])
        if score_mask.any()
        else scores.sum() * 0.0
    )

    pair_losses = []
    active_pairs = 0
    for start in range(0, modes, group_size):
        group_logits = reference_logits[:, start : start + group_size]
        top2_local = torch.topk(group_logits, k=2, dim=-1).indices
        top2 = top2_local + start
        pair_rewards = rewards.gather(1, top2)
        pair_valid = finite.gather(1, top2).all(dim=-1)
        reward_delta = pair_rewards[:, 1] - pair_rewards[:, 0]
        active_scene = pair_valid & (reward_delta.abs() >= reward_gap)
        if not active_scene.any():
            continue
        pair_scores = scores.gather(1, top2.unsqueeze(-1).expand(-1, -1, heads))
        score_delta = pair_scores[:, 1] - pair_scores[:, 0]
        signs = reward_delta.sign().unsqueeze(-1)
        pair_mask = active_scene.unsqueeze(-1) & bootstrap
        if pair_mask.any():
            pair_losses.append(F.softplus(-(signs * score_delta))[pair_mask])
            active_pairs += int(pair_mask.sum().item())
    rank_loss = (
        torch.cat(pair_losses).mean() if pair_losses else scores.sum() * 0.0
    )
    total = safety_loss + component_loss + pdms_loss + rank_loss
    ensemble_score = scores.mean(dim=-1)
    selected = ensemble_score.gather(
        1, reference_logits.argmax(dim=-1, keepdim=True)
    ).squeeze(-1)
    return {
        "loss": total,
        "value_safety_loss": safety_loss.detach(),
        "value_component_loss": component_loss.detach(),
        "value_pdms_loss": pdms_loss.detach(),
        "value_rank_loss": rank_loss.detach(),
        "value_valid_candidate_fraction": finite.float().mean().detach(),
        "value_bootstrap_fraction": bootstrap.float().mean().detach(),
        "value_active_pair_count": total.detach().new_tensor(float(active_pairs)),
        "value_predicted_fallback_score": selected.mean().detach(),
    }
