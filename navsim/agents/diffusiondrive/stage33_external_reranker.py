"""Stage33 external selector reranker.

This module is deliberately independent from the diffusion trajectory head.  It
is trained offline with PDM labels, but inference only consumes candidate
trajectories and selector-side diagnostics.  The separation makes the selector
an ablation that can be swapped without changing the GRPO generator.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


STAGE33_RERANKER_VERSION = "stage33_relative_harm_reranker_v1"
NUM_CANDIDATES = 20
TRAJECTORY_FEATURES = 8 * 3
OPTIONAL_FEATURES = 10
FEATURE_DIM = TRAJECTORY_FEATURES + 2 + OPTIONAL_FEATURES


def _finite_tensor(value: Any, shape: Optional[Tuple[int, ...]] = None) -> Optional[Tensor]:
    if value is None:
        return None
    try:
        result = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError):
        return None
    if shape is not None and tuple(result.shape) != tuple(shape):
        return None
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _find_first(mapping: Mapping[str, Any], names: Iterable[str]) -> Any:
    wanted = set(names)
    for key, value in mapping.items():
        if key in wanted:
            return value
        if isinstance(value, Mapping):
            found = _find_first(value, wanted)
            if found is not None:
                return found
    return None


def _mode_rows(value: Any, *, default: float = 0.0) -> Tensor:
    """Convert an optional per-mode value to a finite [20] tensor."""
    raw = _finite_tensor(value)
    if raw is None or raw.numel() == 0:
        return torch.full((NUM_CANDIDATES,), default, dtype=torch.float32)
    if raw.ndim == 0:
        return raw.repeat(NUM_CANDIDATES)
    if raw.shape[0] != NUM_CANDIDATES:
        return torch.full((NUM_CANDIDATES,), default, dtype=torch.float32)
    raw = raw.reshape(NUM_CANDIDATES, -1)
    return raw.mean(dim=1)


def _mode_summary(value: Any) -> Tuple[Tensor, Tensor]:
    raw = _finite_tensor(value)
    if raw is None or raw.numel() == 0 or raw.ndim == 0 or raw.shape[0] != NUM_CANDIDATES:
        zero = torch.zeros(NUM_CANDIDATES, dtype=torch.float32)
        return zero, zero
    raw = raw.reshape(NUM_CANDIDATES, -1)
    return raw.mean(dim=1), raw.std(dim=1, unbiased=False)


def build_candidate_feature_matrix(
    record: Mapping[str, Any],
    *,
    include_optional: bool = True,
) -> Tensor:
    """Build the fixed [20, FEATURE_DIM] selector input without PDM labels."""
    trajectories = _finite_tensor(record.get("candidate_trajectories"))
    if trajectories is None or tuple(trajectories.shape) != (NUM_CANDIDATES, 8, 3):
        raise ValueError("candidate_trajectories must have shape [20, 8, 3]")
    # Keep a fixed physical scale so a scene cannot change the feature scale.
    scale = torch.tensor([50.0, 50.0, 3.2], dtype=torch.float32)
    trajectory_features = (trajectories / scale).clamp(-4.0, 4.0).reshape(NUM_CANDIDATES, -1)

    reference_logits = _mode_rows(record.get("candidate_reference_logits"))
    reference_logits = (reference_logits - reference_logits.mean()) / (
        reference_logits.std(unbiased=False).clamp_min(1e-3)
    )
    selector_probabilities = _mode_rows(record.get("selector_probabilities"))
    selector_probabilities = selector_probabilities.clamp(0.0, 1.0)

    optional = torch.zeros(NUM_CANDIDATES, OPTIONAL_FEATURES, dtype=torch.float32)
    if include_optional:
        delta_mean, delta_std = _mode_summary(
            _find_first(record, ("delta_predictions", "delta_prediction"))
        )
        risk = _mode_rows(_find_first(record, ("risk_ucb", "risk_upper_confidence_bound")))
        delta_lcb = _mode_rows(_find_first(record, ("delta_lcb", "delta_lower_confidence_bound")))
        ood = _mode_rows(_find_first(record, ("ood_distance", "ood")))
        harm_mean, harm_std = _mode_summary(
            _find_first(record, ("harm_probabilities", "harm_probability"))
        )
        catastrophe_mean, catastrophe_std = _mode_summary(
            _find_first(record, ("catastrophe_probabilities", "catastrophe_probability"))
        )
        optional[:, 0] = delta_mean
        optional[:, 1] = delta_std
        optional[:, 2] = risk
        optional[:, 3] = delta_lcb
        optional[:, 4] = ood
        optional[:, 5] = harm_mean
        optional[:, 6] = harm_std
        optional[:, 7] = catastrophe_mean
        optional[:, 8] = catastrophe_std
        # One shared availability bit prevents zero-filled diagnostics from
        # being mistaken for a calibrated zero risk.
        optional[:, 9] = float(
            any(
                _find_first(record, (name,)) is not None
                for name in (
                    "delta_predictions",
                    "risk_ucb",
                    "delta_lcb",
                    "ood_distance",
                    "harm_probabilities",
                    "catastrophe_probabilities",
                )
            )
        )
    features = torch.cat(
        (trajectory_features, reference_logits[:, None], selector_probabilities[:, None], optional),
        dim=1,
    )
    if features.shape != (NUM_CANDIDATES, FEATURE_DIM) or not torch.isfinite(features).all():
        raise RuntimeError("Stage33 feature construction produced an invalid matrix")
    return features


class Stage33ExternalReranker(nn.Module):
    """Listwise candidate scorer with an explicit uncertainty/catastrophe head."""

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if hidden_dim % nhead:
            raise ValueError("hidden_dim must be divisible by nhead")
        self.feature_dim = int(feature_dim)
        self.input = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.score_head = nn.Linear(hidden_dim, 1)
        self.log_std_head = nn.Linear(hidden_dim, 1)
        self.catastrophe_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features must be [batch, {NUM_CANDIDATES}, {self.feature_dim}]"
            )
        hidden = self.context(self.input(features))
        return {
            "score_mean": self.score_head(hidden).squeeze(-1),
            "score_log_std": self.log_std_head(hidden).squeeze(-1).clamp(-6.0, 3.0),
            "catastrophe_logit": self.catastrophe_head(hidden).squeeze(-1),
        }


def compute_stage33_reranker_loss(
    outputs: Mapping[str, Tensor],
    candidate_rewards: Tensor,
    valid_mask: Optional[Tensor] = None,
    *,
    temperature: float = 0.15,
    uncertainty_weight: float = 0.02,
    pairwise_weight: float = 0.15,
    catastrophe_targets: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Listwise PDM ranking loss; labels never enter the inference path."""
    rewards = torch.as_tensor(candidate_rewards, dtype=torch.float32)
    if rewards.ndim != 2 or rewards.shape[1] != NUM_CANDIDATES:
        raise ValueError("candidate_rewards must be [batch, 20]")
    mask = (
        torch.ones_like(rewards, dtype=torch.bool)
        if valid_mask is None
        else torch.as_tensor(valid_mask, dtype=torch.bool, device=rewards.device)
    )
    if mask.shape != rewards.shape:
        raise ValueError("valid_mask must match candidate_rewards")
    mean = outputs["score_mean"]
    log_std = outputs["score_log_std"]
    cat_logit = outputs["catastrophe_logit"]
    safe_rewards = rewards.masked_fill(~mask, -1e4)
    target = F.softmax((safe_rewards - safe_rewards.max(dim=1, keepdim=True).values) / temperature, dim=1)
    target = target * mask
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
    log_probs = F.log_softmax(mean.masked_fill(~mask, -1e4), dim=1)
    listwise = -(target * log_probs).sum(dim=1).mean()

    # Sample a dense pairwise term from the reward ordering without introducing
    # another model head or changing the diffusion policy log-prob definition.
    pair_delta = rewards[:, :, None] - rewards[:, None, :]
    pair_mask = mask[:, :, None] & mask[:, None, :] & (pair_delta.abs() > 1e-6)
    pair_target = (pair_delta > 0).float()
    pair_logit = mean[:, :, None] - mean[:, None, :]
    pair_loss = F.binary_cross_entropy_with_logits(
        pair_logit[pair_mask], pair_target[pair_mask]
    ) if pair_mask.any() else mean.new_zeros(())
    uncertainty = torch.exp(log_std).masked_select(mask).mean() if mask.any() else mean.new_zeros(())

    catastrophe_loss = mean.new_zeros(())
    if catastrophe_targets is not None:
        target_cat = torch.as_tensor(
            catastrophe_targets, dtype=torch.float32, device=mean.device
        )
        if target_cat.shape != rewards.shape:
            raise ValueError("catastrophe_targets must match candidate_rewards")
        catastrophe_loss = F.binary_cross_entropy_with_logits(
            cat_logit[mask], target_cat[mask]
        ) if mask.any() else mean.new_zeros(())
    total = listwise + pairwise_weight * pair_loss + uncertainty_weight * uncertainty + catastrophe_loss
    return {
        "loss": total,
        "listwise_loss": listwise,
        "pairwise_loss": pair_loss,
        "uncertainty": uncertainty,
        "catastrophe_loss": catastrophe_loss,
    }


@torch.no_grad()
def select_stage33(
    outputs: Mapping[str, Tensor],
    *,
    eligible_mask: Optional[Tensor] = None,
    fallback_mode: int = 0,
    fallback_score: Optional[Tensor] = None,
    risk_z: float = 1.0,
    fallback_margin: float = 0.0,
) -> Dict[str, Tensor]:
    """Select a candidate with uncertainty/risk and a fail-safe fallback."""
    mean = outputs["score_mean"]
    std = torch.exp(outputs["score_log_std"].clamp(-6.0, 3.0))
    cat = torch.sigmoid(outputs["catastrophe_logit"])
    if mean.ndim != 2 or mean.shape[1] != NUM_CANDIDATES:
        raise ValueError("reranker outputs must be [batch, 20]")
    mask = (
        torch.ones_like(mean, dtype=torch.bool)
        if eligible_mask is None
        else torch.as_tensor(eligible_mask, dtype=torch.bool, device=mean.device)
    )
    if mask.shape != mean.shape:
        raise ValueError("eligible_mask must match reranker outputs")
    safe_score = mean - float(risk_z) * std - cat * 0.25
    safe_score = safe_score.masked_fill(~mask, -torch.inf)
    selected = safe_score.argmax(dim=1)
    fallback = torch.full_like(selected, int(fallback_mode))
    if fallback_score is None:
        base = safe_score.new_full((mean.shape[0],), -torch.inf)
    else:
        base = torch.as_tensor(fallback_score, dtype=mean.dtype, device=mean.device).reshape(-1)
        if base.numel() == 1:
            base = base.expand(mean.shape[0])
    best = safe_score.gather(1, selected[:, None]).squeeze(1)
    use_fallback = (~mask.any(dim=1)) | (~torch.isfinite(best)) | (best < base + fallback_margin)
    selected = torch.where(use_fallback, fallback, selected)
    return {
        "selected_mode": selected,
        "fallback_used": use_fallback,
        "safe_score": safe_score,
        "uncertainty": std,
        "catastrophe_probability": cat,
    }

