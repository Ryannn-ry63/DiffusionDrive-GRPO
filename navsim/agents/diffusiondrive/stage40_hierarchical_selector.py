"""Candidate-count-agnostic Stage40 selector primitives.

Stage40 is intentionally independent from the Stage39 generator objective.  The
selector sees a candidate relative to the frozen public fallback, not a branch
label, so an ablation can change candidate count or generator source without
changing the selector input contract.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


STAGE40_OBJECTIVE_REVISION = "hierarchical_relative_risk_selector_v1"
COMPONENT_COUNT = 6
SAFETY_COMPONENT_INDICES = (0, 1, 3)


def _gather_fallback(candidates: torch.Tensor, fallback_index: torch.Tensor) -> torch.Tensor:
    if candidates.ndim != 4 or candidates.shape[-1] < 2:
        raise ValueError("candidates must have shape [B,N,T,D>=2]")
    if fallback_index.ndim != 1 or fallback_index.shape[0] != candidates.shape[0]:
        raise ValueError("fallback_index must have shape [B]")
    if (fallback_index < 0).any() or (fallback_index >= candidates.shape[1]).any():
        raise ValueError("fallback_index is outside candidate bank")
    batch = torch.arange(candidates.shape[0], device=candidates.device)
    return candidates[batch, fallback_index]


def build_stage40_relative_features(
    candidates: torch.Tensor,
    fallback_index: torch.Tensor,
    candidate_logits: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build relative trajectory features without a fixed candidate count.

    The feature width is independent of N.  It contains relative XY/velocity
    summaries, curvature and length deltas, normalized rank, and the candidate
    logit margin to the fallback.  No candidate-origin or branch ID is used.
    """
    fallback = _gather_fallback(candidates, fallback_index)
    xy = candidates[..., :2]
    ref_xy = fallback[:, None, :, :2]
    delta = xy - ref_xy
    step = xy[..., 1:, :] - xy[..., :-1, :]
    ref_step = ref_xy[..., 1:, :] - ref_xy[..., :-1, :]
    speed = torch.linalg.vector_norm(step, dim=-1)
    ref_speed = torch.linalg.vector_norm(ref_step, dim=-1)
    curvature = (step[..., 1:, :] - step[..., :-1, :]).abs().mean(dim=(-1, -2))
    ref_curvature = (ref_step[..., 1:, :] - ref_step[..., :-1, :]).abs().mean(dim=(-1, -2))
    displacement = delta.abs().mean(dim=(-1, -2))
    endpoint = delta[..., -1, :].abs()
    speed_delta = (speed - ref_speed).abs().mean(dim=-1)
    length_delta = (speed.sum(dim=-1) - ref_speed.sum(dim=-1)).abs()
    if candidate_logits is None:
        logit_margin = torch.zeros(
            candidates.shape[:2], dtype=candidates.dtype, device=candidates.device
        )
        normalized_rank = torch.zeros_like(logit_margin)
    else:
        if candidate_logits.shape != candidates.shape[:2]:
            raise ValueError("candidate_logits must have shape [B,N]")
        fallback_logits = candidate_logits.gather(1, fallback_index[:, None]).squeeze(1)
        logit_margin = candidate_logits - fallback_logits[:, None]
        order = candidate_logits.argsort(dim=1, descending=True).argsort(dim=1)
        normalized_rank = order.to(candidates.dtype) / max(candidates.shape[1] - 1, 1)
    features = torch.cat(
        (
            delta.mean(dim=-2),
            delta.std(dim=-2, unbiased=False),
            endpoint,
            displacement.unsqueeze(-1),
            speed_delta.unsqueeze(-1),
            length_delta.unsqueeze(-1),
            (curvature - ref_curvature).abs().unsqueeze(-1),
            logit_margin.unsqueeze(-1),
            normalized_rank.unsqueeze(-1),
        ),
        dim=-1,
    )
    return torch.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)


class Stage40HierarchicalSelector(nn.Module):
    """Ensemble risk head for hierarchical public-fallback selection."""

    def __init__(self, feature_dim: int = 12, hidden_dim: int = 128, ensemble_size: int = 4):
        super().__init__()
        if ensemble_size < 2:
            raise ValueError("Stage40 requires an ensemble of at least two heads")
        self.feature_dim = int(feature_dim)
        self.ensemble_size = int(ensemble_size)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1 + COMPONENT_COUNT + 2),
                )
                for _ in range(ensemble_size)
            ]
        )

    def forward(
        self,
        candidates: torch.Tensor,
        fallback_index: torch.Tensor,
        candidate_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        features = build_stage40_relative_features(
            candidates, fallback_index, candidate_logits
        )
        predictions = torch.stack([head(features) for head in self.heads], dim=0)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0, unbiased=False)
        return {
            "features": features,
            "ensemble_predictions": predictions,
            "delta_mean": mean[..., 0],
            "component_delta_mean": mean[..., 1 : 1 + COMPONENT_COUNT],
            "risk_logits": mean[..., 1 + COMPONENT_COUNT :],
            "delta_std": std[..., 0],
            "component_delta_std": std[..., 1 : 1 + COMPONENT_COUNT],
            "risk_std": std[..., 1 + COMPONENT_COUNT :],
        }


def select_stage40_hierarchical(
    fallback_index: torch.Tensor,
    candidate_logits: torch.Tensor,
    outputs: Dict[str, torch.Tensor],
    candidate_valid: Optional[torch.Tensor] = None,
    candidate_components: Optional[torch.Tensor] = None,
    margin: float = 0.001,
    safety_floor: float = -0.0005,
    harm_threshold: float = 0.35,
    catastrophe_threshold: float = 0.05,
    confidence_z: float = 1.64,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Select a challenger only when a conservative lower bound beats fallback."""
    if candidate_logits.ndim != 2:
        raise ValueError("candidate_logits must have shape [B,N]")
    batch, modes = candidate_logits.shape
    for key in ("delta_mean", "component_delta_mean", "risk_logits"):
        if key not in outputs:
            raise KeyError(f"missing Stage40 output {key}")
    delta_lcb = outputs["delta_mean"] - confidence_z * outputs["delta_std"]
    component_lcb = outputs["component_delta_mean"] - confidence_z * outputs["component_delta_std"]
    risk_prob = torch.sigmoid(outputs["risk_logits"] + confidence_z * outputs["risk_std"])
    eligible = delta_lcb >= float(margin)
    eligible &= component_lcb[..., list(SAFETY_COMPONENT_INDICES)].min(dim=-1).values >= float(safety_floor)
    eligible &= risk_prob[..., 0] <= float(harm_threshold)
    eligible &= risk_prob[..., 1] <= float(catastrophe_threshold)
    if candidate_valid is not None:
        if candidate_valid.shape != (batch, modes):
            raise ValueError("candidate_valid must have shape [B,N]")
        eligible &= candidate_valid.bool()
    if candidate_components is not None:
        if candidate_components.shape[:2] != (batch, modes) or candidate_components.shape[-1] != COMPONENT_COUNT:
            raise ValueError("candidate_components must have shape [B,N,6]")
        eligible &= candidate_components[..., list(SAFETY_COMPONENT_INDICES)].min(dim=-1).values >= float(safety_floor)
    eligible.scatter_(1, fallback_index[:, None], torch.ones((batch, 1), dtype=torch.bool, device=candidate_logits.device))
    utility = torch.where(eligible, delta_lcb + 0.05 * candidate_logits, torch.full_like(delta_lcb, -torch.inf))
    selected_index = utility.argmax(dim=-1)
    selected_index = torch.where(
        torch.isfinite(utility.gather(1, selected_index[:, None]).squeeze(1)),
        selected_index, fallback_index,
    )
    diagnostics = {
        "delta_lcb": delta_lcb,
        "component_lcb": component_lcb,
        "risk_probabilities": risk_prob,
        "eligible": eligible,
        "switch": selected_index != fallback_index,
        "selected_index": selected_index,
    }
    return selected_index, diagnostics


def stage40_pairwise_loss(
    outputs: Dict[str, torch.Tensor],
    target_delta: torch.Tensor,
    target_components: torch.Tensor,
    target_harm: torch.Tensor,
    target_catastrophe: torch.Tensor,
) -> torch.Tensor:
    """Calibrate selector outputs against OOF paired deltas/risk labels."""
    delta = F.smooth_l1_loss(outputs["delta_mean"], target_delta)
    components = F.smooth_l1_loss(outputs["component_delta_mean"], target_components)
    risk_target = torch.stack((target_harm, target_catastrophe), dim=-1).float()
    risk = F.binary_cross_entropy_with_logits(outputs["risk_logits"], risk_target)
    return delta + components + risk
