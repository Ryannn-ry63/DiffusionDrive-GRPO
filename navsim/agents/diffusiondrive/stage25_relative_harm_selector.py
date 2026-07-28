"""Stage25 relative-to-fallback harm heads for the frozen Stage24 selector."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.diffusiondrive.stage24_safety_value_selector import (
    PDM_SAFETY_INDICES,
    STAGE24_SELECTOR_MEMBERS,
    stage24_ood_distance,
)


STAGE25_HARM_TOLERANCE = 0.0005
STAGE25_HARM_EPSILON = 1e-7


class Stage25RelativeHarmSelector(nn.Module):
    """Independent member-wise harm MLPs over frozen Stage24 embeddings."""

    def __init__(self, config):
        super().__init__()
        self.num_members = int(getattr(config, "stage24_selector_num_members", 8))
        self.selector_dim = int(getattr(config, "stage24_selector_dim", 128))
        if self.num_members != STAGE24_SELECTOR_MEMBERS:
            raise ValueError("formal Stage25 requires exactly eight members")
        with torch.random.fork_rng(devices=[]):
            self.members = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.selector_dim, self.selector_dim),
                    nn.GELU(),
                    nn.LayerNorm(self.selector_dim),
                    nn.Linear(self.selector_dim, 4),
                )
                for _ in range(self.num_members)
            ])

    def forward(self, member_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        if (
            member_embeddings.ndim != 4
            or member_embeddings.shape[2:] != (
                self.num_members, self.selector_dim
            )
        ):
            raise ValueError(
                "Stage25 embeddings must have shape [B,M,8,selector_dim]"
            )
        logits = torch.stack([
            member(member_embeddings[:, :, index])
            for index, member in enumerate(self.members)
        ], dim=2)
        return {
            "harm_logits": logits[..., :3],
            "catastrophe_logits": logits[..., 3],
            "harm_probabilities": logits[..., :3].sigmoid(),
            "catastrophe_probabilities": logits[..., 3].sigmoid(),
        }


def compute_stage25_positive_weights(candidate_bank) -> tuple[float, ...]:
    """Compute exact train-bank relative-harm imbalance weights."""
    positive = np.zeros(4, dtype=np.float64)
    negative = np.zeros(4, dtype=np.float64)
    for records in candidate_bank.values():
        for record in records:
            components = np.asarray(
                record["candidate_components"], dtype=np.float64
            )
            rewards = np.asarray(record["candidate_rewards"], dtype=np.float64)
            logits = np.asarray(
                record["candidate_reference_logits"], dtype=np.float64
            )
            fallback = int(np.argmax(logits))
            component_delta = components[:, list(PDM_SAFETY_INDICES)] - components[
                fallback, list(PDM_SAFETY_INDICES)
            ]
            harm = component_delta < -(
                STAGE25_HARM_TOLERANCE + STAGE25_HARM_EPSILON
            )
            catastrophe = harm.any(axis=-1) | (
                rewards - rewards[fallback] <= -0.5
            )
            labels = np.concatenate(
                (harm, catastrophe[:, None]), axis=-1
            )
            valid = (
                np.isfinite(components).all(axis=-1)
                & np.isfinite(rewards)
            )
            valid[fallback] = False
            positive += labels[valid].sum(axis=0)
            negative += (~labels[valid]).sum(axis=0)
    if np.any(positive <= 0):
        raise RuntimeError("Stage25 train bank lacks a positive harm label")
    return tuple(np.clip(negative / positive, 1.0, 200.0).tolist())


def _focal_loss(
    logits: torch.Tensor, targets: torch.Tensor,
    positive_weights: torch.Tensor, gamma: float,
) -> torch.Tensor:
    probabilities = logits.sigmoid()
    pt = torch.where(targets.bool(), probabilities, 1.0 - probabilities)
    weights = torch.where(
        targets.bool(), positive_weights, torch.ones_like(targets)
    )
    return -weights * (1.0 - pt).pow(gamma) * pt.clamp_min(1e-6).log()


def compute_stage25_selector_loss(
    predictions: Dict[str, torch.Tensor], focal_gamma: float = 2.0,
    hard_negative_count: int = 4, hard_risk_margin: float = 0.9,
) -> Dict[str, torch.Tensor]:
    """Train direct relative-harm and catastrophe classification heads."""
    harm_logits = predictions["stage25_harm_logits"]
    catastrophe_logits = predictions["stage25_catastrophe_logits"]
    delta_predictions = predictions["stage24_delta_predictions"].detach()
    components = predictions["component_scores"].detach().float()
    rewards = predictions["raw_rewards"].detach().float()
    valid = predictions["reward_valid_mask"].bool()
    fallback = predictions["stage24_fallback_mode"].long()
    member_mask = predictions["stage24_member_training_mask"].bool()
    positive_weights = predictions["stage25_positive_weights"].float()
    if harm_logits.ndim != 4 or harm_logits.shape[-2:] != (8, 3):
        raise ValueError("Stage25 harm logits must have shape [B,20,8,3]")
    batch, modes, members, _ = harm_logits.shape
    if (
        modes != 20
        or catastrophe_logits.shape != (batch, modes, members)
        or delta_predictions.shape != (batch, modes, members)
        or components.shape != (batch, modes, 6)
        or rewards.shape != (batch, modes)
        or positive_weights.shape != (4,)
    ):
        raise ValueError("Stage25 training tensor shape mismatch")
    batch_index = torch.arange(batch, device=rewards.device)
    fallback_components = components[batch_index, fallback][
        :, list(PDM_SAFETY_INDICES)
    ]
    fallback_rewards = rewards[batch_index, fallback]
    relative_components = (
        components[..., list(PDM_SAFETY_INDICES)]
        - fallback_components.unsqueeze(1)
    )
    harm = relative_components < -(
        STAGE25_HARM_TOLERANCE + STAGE25_HARM_EPSILON
    )
    catastrophe = harm.any(dim=-1) | (
        rewards - fallback_rewards.unsqueeze(1) <= -0.5
    )
    finite = (
        valid & torch.isfinite(rewards)
        & torch.isfinite(components).all(dim=-1)
    )
    nonfallback = torch.ones(
        batch, modes, dtype=torch.bool, device=rewards.device
    )
    nonfallback[batch_index, fallback] = False
    base_mask = (
        finite.unsqueeze(-1) & nonfallback.unsqueeze(-1)
        & member_mask.unsqueeze(1)
    )

    harm_targets = harm.unsqueeze(2).expand_as(harm_logits).float()
    harm_values = _focal_loss(
        harm_logits, harm_targets,
        positive_weights[:3].view(1, 1, 1, 3), focal_gamma,
    )
    harm_mask = base_mask.unsqueeze(-1).expand_as(harm_values)
    harm_loss = harm_values[harm_mask].mean()

    catastrophe_targets = catastrophe.unsqueeze(-1).expand_as(
        catastrophe_logits
    ).float()
    catastrophe_values = _focal_loss(
        catastrophe_logits, catastrophe_targets,
        positive_weights[3].view(1, 1, 1), focal_gamma,
    )
    catastrophe_loss = catastrophe_values[base_mask].mean()

    risk = torch.maximum(
        harm_logits.sigmoid().amax(dim=-1),
        catastrophe_logits.sigmoid(),
    )
    harmful = catastrophe & nonfallback & finite
    mining_score = delta_predictions.masked_fill(
        ~harmful.unsqueeze(-1), -torch.inf
    )
    topk = min(max(int(hard_negative_count), 1), modes)
    hard_index = mining_score.topk(topk, dim=1).indices
    hard_risk = risk.gather(1, hard_index)
    hard_valid = harmful.unsqueeze(-1).expand_as(risk).gather(1, hard_index)
    hard_mask = hard_valid & member_mask.unsqueeze(1)
    hard_loss = (
        F.relu(float(hard_risk_margin) - hard_risk[hard_mask]).mean()
        if hard_mask.any() else risk.sum() * 0.0
    )
    total = 3.0 * harm_loss + 4.0 * catastrophe_loss + 4.0 * hard_loss
    return {
        "loss": total,
        "stage25_relative_harm_loss": harm_loss.detach(),
        "stage25_catastrophe_loss": catastrophe_loss.detach(),
        "stage25_high_value_false_safe_loss": hard_loss.detach(),
        "stage25_harm_rate": harm.float().mean().detach(),
        "stage25_catastrophe_rate": catastrophe.float().mean().detach(),
    }


def select_stage25_trajectory(
    reference_logits: torch.Tensor,
    harm_probabilities: torch.Tensor,
    catastrophe_probabilities: torch.Tensor,
    delta_predictions: torch.Tensor,
    embedding_mean: torch.Tensor,
    residual_margin: float,
    risk_threshold: float,
    ood_mean: torch.Tensor,
    ood_variance: torch.Tensor,
    ood_threshold: float,
    confidence_z: float = 1.96,
):
    """Select by frozen Stage24 value subject to Stage25 relative-harm risk."""
    batch, modes = reference_logits.shape
    if harm_probabilities.shape != (batch, modes, 8, 3):
        raise ValueError("Stage25 harm ensemble shape mismatch")
    if catastrophe_probabilities.shape != (batch, modes, 8):
        raise ValueError("Stage25 catastrophe ensemble shape mismatch")
    if delta_predictions.shape != (batch, modes, 8):
        raise ValueError("Stage25 value ensemble shape mismatch")
    if residual_margin < 0 or ood_threshold < 0:
        raise ValueError("Stage25 calibration margins must be non-negative")
    if not 0.0 <= risk_threshold <= 1.0:
        raise ValueError("Stage25 risk threshold must lie in [0,1]")
    risk_members = torch.maximum(
        harm_probabilities.amax(dim=-1), catastrophe_probabilities
    )
    risk_ucb = risk_members.mean(dim=-1) + confidence_z * risk_members.std(
        dim=-1, unbiased=False
    )
    delta_mean = delta_predictions.mean(dim=-1)
    delta_std = delta_predictions.std(dim=-1, unbiased=False)
    delta_lcb = delta_mean - confidence_z * delta_std - residual_margin
    ood_distance = stage24_ood_distance(
        embedding_mean, ood_mean.to(embedding_mean),
        ood_variance.to(embedding_mean),
    )
    eligible = (
        (risk_ucb <= risk_threshold)
        & (delta_lcb > 0)
        & (ood_distance <= ood_threshold)
    )
    fallback = reference_logits.argmax(dim=-1)
    batch_index = torch.arange(batch, device=reference_logits.device)
    eligible[batch_index, fallback] = False
    challenger = delta_lcb.masked_fill(~eligible, -torch.inf).argmax(dim=-1)
    switch = eligible.any(dim=-1)
    selected = torch.where(switch, challenger, fallback)
    return selected, {
        "fallback_mode": fallback,
        "challenger_mode": challenger,
        "switch": switch,
        "eligible": eligible,
        "risk_ucb": risk_ucb,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "delta_lcb": delta_lcb,
        "ood_distance": ood_distance,
    }
