"""Counterfactual Deployment-Credit GRPO utilities for Stage33.

The functions here are tensor-only and intentionally do not evaluate PDM.
PDM values are supplied by a frozen/common-noise candidate bank or training
trace.  This keeps the generator objective distinct from a selector
classification loss and makes the credit assignment unit-testable.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.nn import functional as F


STAGE33_CDC_OBJECTIVE_REVISION = "cdc_deployment_credit_v1"


def _as_scene_mode(value: Tensor, name: str) -> tuple[Tensor, bool]:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 2:
        return tensor[:, None, :], True
    if tensor.ndim == 3:
        return tensor, False
    raise ValueError(f"{name} must have shape [B, M] or [B, G, M]")


def _mode_index(value: Any, batch: int, groups: int, modes: int, device: torch.device) -> Tensor:
    index = torch.as_tensor(value, dtype=torch.long, device=device)
    if index.ndim == 1:
        index = index[:, None]
    if index.shape != (batch, groups):
        raise ValueError("selected modes must have shape [B] or [B, G]")
    return index.clamp(0, modes - 1)


def _broadcast_mask(mask: Optional[Tensor], shape: tuple[int, int, int], device: torch.device) -> Tensor:
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    result = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if result.ndim == 2:
        result = result[:, None, :]
    if result.shape != shape:
        raise ValueError("valid mask must match reward shape")
    return result


def _normalize_credit(credit: Tensor, valid: Tensor, eps: float) -> Tensor:
    masked = credit.masked_fill(~valid, 0.0)
    mean = masked.sum(dim=-1, keepdim=True) / valid.sum(dim=-1, keepdim=True).clamp_min(1)
    centered = (masked - mean).masked_fill(~valid, 0.0)
    scale = centered.abs().sum(dim=-1, keepdim=True) / valid.sum(dim=-1, keepdim=True).clamp_min(1)
    return (centered / scale.clamp_min(eps)).clamp(-2.0, 2.0)


def compute_cdc_advantages(
    current_rewards: Tensor,
    public_rewards: Tensor,
    current_selected_modes: Tensor,
    public_selected_modes: Tensor,
    selector_probabilities: Optional[Tensor] = None,
    valid_mask: Optional[Tensor] = None,
    *,
    catastrophe_mask: Optional[Tensor] = None,
    deployment_weight: float = 0.5,
    headroom_weight: float = 0.5,
    headroom_margin: float = 0.001,
    catastrophe_penalty: float = 1.0,
    advantage_eps: float = 1e-3,
    advantage_clip: float = 2.0,
) -> Dict[str, Tensor]:
    """Return candidate-level CDC credits and normalized advantages.

    The selected current mode receives the paired deployment delta against the
    public generator.  Unselected candidates receive only a detached
    selector-probability-weighted headroom term, which prevents oracle chasing.
    """
    current, current_was_2d = _as_scene_mode(current_rewards, "current_rewards")
    public, public_was_2d = _as_scene_mode(public_rewards, "public_rewards")
    if current.shape != public.shape:
        raise ValueError("current_rewards and public_rewards must have equal shape")
    batch, groups, modes = current.shape
    device = current.device
    current_mode = _mode_index(current_selected_modes, batch, groups, modes, device)
    public_mode = _mode_index(public_selected_modes, batch, groups, modes, device)
    valid = _broadcast_mask(valid_mask, current.shape, device)

    current_selected = current.gather(-1, current_mode[..., None]).squeeze(-1)
    public_selected = public.gather(-1, public_mode[..., None]).squeeze(-1)
    deployment_delta = current_selected - public_selected
    selected_one_hot = F.one_hot(current_mode, modes).to(current.dtype)
    selected_credit = selected_one_hot * deployment_delta[..., None]

    if selector_probabilities is None:
        probabilities = selected_one_hot
    else:
        probabilities, _ = _as_scene_mode(selector_probabilities, "selector_probabilities")
        if probabilities.shape != current.shape:
            raise ValueError("selector_probabilities must match reward shape")
        probabilities = probabilities.clamp_min(0.0) * valid
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    headroom = (current - current_selected[..., None] - float(headroom_margin)).clamp_min(0.0)
    soft_headroom = probabilities.detach() * headroom

    credit = float(deployment_weight) * selected_credit + float(headroom_weight) * soft_headroom
    catastrophe = torch.zeros_like(credit)
    if catastrophe_mask is not None:
        catastrophe = torch.as_tensor(catastrophe_mask, dtype=torch.float32, device=device)
        if catastrophe.ndim == 2:
            catastrophe = catastrophe[:, None, :]
        if catastrophe.shape != current.shape:
            raise ValueError("catastrophe_mask must match reward shape")
        credit = credit - float(catastrophe_penalty) * catastrophe
    advantages = _normalize_credit(credit, valid, float(advantage_eps))
    advantages = advantages.clamp(-float(advantage_clip), float(advantage_clip))

    if current_was_2d or public_was_2d:
        # Preserve the convenient [B, M] shape for single-group callers.
        advantages = advantages[:, 0]
        credit = credit[:, 0]
        valid = valid[:, 0]
        soft_headroom = soft_headroom[:, 0]
    return {
        "advantages": advantages,
        "credit": credit,
        "deployment_delta": deployment_delta,
        "headroom": soft_headroom,
        "selected_current_reward": current_selected,
        "selected_public_reward": public_selected,
        "valid_mask": valid,
        "catastrophe_penalty": catastrophe.mean(),
        "deployment_credit_mean": credit.mean(),
        "headroom_credit_mean": soft_headroom.mean(),
        "positive_fraction": (advantages > 0).float().mean(),
        "negative_fraction": (advantages < 0).float().mean(),
    }

