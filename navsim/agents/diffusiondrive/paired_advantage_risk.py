"""Stage-17 same-mode advantage/risk prediction and exact base fallback."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.diffusiondrive.modules.blocks import (
    GridSampleCrossBEVAttention,
    gen_sineembed_for_position,
)


PAIR_EVENT_NAMES: Tuple[str, ...] = (
    "base_better",
    "loss_0p1",
    "loss_0p5",
    "collision_regression",
    "drivable_regression",
    "ttc_regression",
)
PAIR_EVENT_COUNT = len(PAIR_EVENT_NAMES)
PAIR_SAFETY_COMPONENT_INDICES = (0, 1, 3)


class PairedAdvantageRiskHead(nn.Module):
    """Predict whether a GRPO trajectory should fall back to its base-mode pair."""

    def __init__(self, config):
        super().__init__()
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
        self.pair_encoder = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )
        self.event_head = nn.Linear(d_model, PAIR_EVENT_COUNT)
        self.delta_head = nn.Linear(d_model, 1)

    def _encode(
        self,
        trajectories: torch.Tensor,
        bev_feature: torch.Tensor,
        bev_spatial_shape,
        agents_query: torch.Tensor,
        ego_query: torch.Tensor,
        status_encoding: torch.Tensor,
    ) -> torch.Tensor:
        position_embedding = gen_sineembed_for_position(
            trajectories[..., :2], hidden_dim=64
        ).flatten(-2)
        query = self.trajectory_encoder(position_embedding)
        query = self.cross_bev(
            query, trajectories[..., :2], bev_feature, bev_spatial_shape
        )
        query = query + self.cross_agents(query, agents_query, agents_query)[0]
        query = query + self.cross_ego(query, ego_query, ego_query)[0]
        status = status_encoding.unsqueeze(1) if status_encoding.ndim == 2 else status_encoding
        return self.context_norm(query + self.status_projection(status))

    def forward(
        self,
        current_trajectories: torch.Tensor,
        base_trajectories: torch.Tensor,
        bev_feature: torch.Tensor,
        bev_spatial_shape,
        agents_query: torch.Tensor,
        ego_query: torch.Tensor,
        status_encoding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        expected_tail = (8, 3)
        if current_trajectories.ndim != 4 or tuple(current_trajectories.shape[-2:]) != expected_tail:
            raise ValueError("paired current trajectories must have shape [B,M,8,3]")
        if base_trajectories.shape != current_trajectories.shape:
            raise ValueError("paired current/base trajectory shapes differ")
        current = self._encode(
            current_trajectories, bev_feature, bev_spatial_shape,
            agents_query, ego_query, status_encoding,
        )
        base = self._encode(
            base_trajectories, bev_feature, bev_spatial_shape,
            agents_query, ego_query, status_encoding,
        )
        pair = self.pair_encoder(
            torch.cat((current, base, current - base, (current - base).abs()), dim=-1)
        )
        event_logits = self.event_head(pair)
        delta_prediction = self.delta_head(pair).squeeze(-1)
        event_probabilities = event_logits.sigmoid()
        fallback_score = event_probabilities.amax(dim=-1)
        return {
            "event_logits": event_logits,
            "event_probabilities": event_probabilities,
            "fallback_score": fallback_score,
            "delta_prediction": delta_prediction,
        }


def build_paired_advantage_risk_targets(
    current_rewards: torch.Tensor,
    base_rewards: torch.Tensor,
    current_components: torch.Tensor,
    base_components: torch.Tensor,
    current_valid: torch.Tensor,
    base_valid: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Build six detached Stage-17 labels for every same-mode pair."""
    if current_rewards.ndim != 2 or base_rewards.shape != current_rewards.shape:
        raise ValueError("paired rewards must have shape [B,M]")
    if current_valid.shape != current_rewards.shape or base_valid.shape != current_rewards.shape:
        raise ValueError("paired valid masks must match rewards")
    expected_components = (*current_rewards.shape, 6)
    if current_components.shape != expected_components or base_components.shape != expected_components:
        raise ValueError("paired components must have shape [B,M,6]")
    valid = (
        current_valid.bool()
        & base_valid.bool()
        & torch.isfinite(current_rewards)
        & torch.isfinite(base_rewards)
        & torch.isfinite(current_components).all(dim=-1)
        & torch.isfinite(base_components).all(dim=-1)
    )
    delta = (current_rewards - base_rewards).detach().float()
    safety_labels = [
        current_components[..., index] < base_components[..., index] - 1e-6
        for index in PAIR_SAFETY_COMPONENT_INDICES
    ]
    labels = torch.stack(
        (
            delta < 0.0,
            delta <= -0.1,
            delta <= -0.5,
            *safety_labels,
        ),
        dim=-1,
    ).detach().float()
    return {"labels": labels, "delta": delta, "valid": valid.detach()}


def compute_paired_advantage_risk_loss(
    predictions: Dict[str, torch.Tensor],
    positive_weights: Sequence[float],
    selected_mode_weight: float = 4.0,
    delta_loss_weight: float = 0.25,
) -> Dict[str, torch.Tensor]:
    """Return the locked weighted classification and auxiliary delta objective."""
    if len(tuple(positive_weights)) != PAIR_EVENT_COUNT:
        raise ValueError(f"paired risk requires {PAIR_EVENT_COUNT} positive weights")
    if selected_mode_weight < 1.0 or delta_loss_weight < 0.0:
        raise ValueError("invalid paired risk loss weights")
    logits = predictions["paired_risk_event_logits"].float()
    delta_prediction = predictions["paired_risk_delta_prediction"].float()
    targets = build_paired_advantage_risk_targets(
        predictions["paired_current_rewards"],
        predictions["paired_base_rewards"],
        predictions["paired_current_components"],
        predictions["paired_base_components"],
        predictions["paired_current_valid"],
        predictions["paired_base_valid"],
    )
    labels, delta, valid = targets["labels"], targets["delta"], targets["valid"]
    if logits.shape != labels.shape or delta_prediction.shape != delta.shape:
        raise ValueError("paired risk prediction/target shapes differ")
    if not valid.any():
        raise RuntimeError("paired risk batch has no valid current/base pair")
    weights = logits.new_tensor(tuple(float(value) for value in positive_weights))
    if not torch.isfinite(weights).all() or (weights <= 0).any() or (weights > 100).any():
        raise ValueError("paired positive weights must be finite in (0,100]")
    element_loss = F.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=weights, reduction="none"
    ).mean(dim=-1)
    reference_logits = predictions["paired_reference_logits"].detach().float()
    if reference_logits.shape != valid.shape:
        raise ValueError("paired reference logits must have shape [B,M]")
    mode_weights = torch.ones_like(delta_prediction)
    selected = reference_logits.argmax(dim=-1)
    mode_weights.scatter_(1, selected.unsqueeze(-1), float(selected_mode_weight))
    valid_weights = mode_weights * valid.float()
    classification_loss = (element_loss * valid_weights).sum() / valid_weights.sum()
    delta_loss = F.smooth_l1_loss(
        delta_prediction[valid], delta[valid], reduction="none"
    )
    delta_loss = (delta_loss * mode_weights[valid]).sum() / mode_weights[valid].sum()
    total = classification_loss + float(delta_loss_weight) * delta_loss
    probabilities = logits.sigmoid()
    fallback_score = probabilities.amax(dim=-1)
    selected_score = fallback_score.gather(1, selected.unsqueeze(-1)).squeeze(-1)
    output = {
        "loss": total,
        "paired_risk_classification_loss": classification_loss.detach(),
        "paired_risk_delta_loss": delta_loss.detach(),
        "paired_risk_valid_pair_fraction": valid.float().mean().detach(),
        "paired_risk_selected_fallback_score": selected_score.mean().detach(),
        "paired_risk_delta_mae": (delta_prediction[valid] - delta[valid]).abs().mean().detach(),
    }
    for index, name in enumerate(PAIR_EVENT_NAMES):
        output[f"paired_risk_{name}_rate"] = labels[..., index][valid].mean().detach()
    return output


def select_same_mode_with_base_fallback(
    current_trajectories: torch.Tensor,
    base_trajectories: torch.Tensor,
    reference_logits: torch.Tensor,
    fallback_scores: torch.Tensor,
    threshold: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Select one reference mode, then return current or exact base trajectory."""
    if current_trajectories.shape != base_trajectories.shape:
        raise ValueError("paired fallback trajectory shapes differ")
    if reference_logits.shape != fallback_scores.shape:
        raise ValueError("reference logits and fallback scores must have shape [B,M]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("paired fallback threshold must be in [0,1]")
    mode = reference_logits.argmax(dim=-1)
    batch = torch.arange(mode.shape[0], device=mode.device)
    selected_score = fallback_scores[batch, mode]
    fallback = selected_score > float(threshold)
    current = current_trajectories[batch, mode]
    base = base_trajectories[batch, mode]
    selected = torch.where(fallback[:, None, None], base, current)
    return selected, {
        "mode": mode,
        "fallback": fallback,
        "selected_score": selected_score,
        "current_trajectory": current,
        "base_trajectory": base,
    }
