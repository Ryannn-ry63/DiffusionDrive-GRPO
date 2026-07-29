"""Stage37 Joint Feasible Improvement (JFI) selector heads."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.diffusiondrive.stage24_safety_value_selector import (
    PDM_SAFETY_INDICES,
    STAGE24_SELECTOR_DIM,
    STAGE24_SELECTOR_MEMBERS,
    stage24_ood_distance,
)


JFI_REWARD_MARGIN = 0.005
JFI_SAFETY_TOLERANCE = 0.0005
JFI_CATASTROPHIC_DELTA = -0.5
JFI_QUANTILES = (0.10, 0.50, 0.90)


class Stage37JointFeasibleSelector(nn.Module):
    """Eight independent JFI heads over frozen Stage24 embeddings."""

    def __init__(self, config):
        super().__init__()
        self.num_members = int(getattr(
            config, "stage37_jfi_num_members", STAGE24_SELECTOR_MEMBERS
        ))
        self.selector_dim = int(getattr(
            config, "stage24_selector_dim", STAGE24_SELECTOR_DIM
        ))
        if self.num_members != STAGE24_SELECTOR_MEMBERS:
            raise ValueError("formal Stage37 JFI requires exactly eight members")
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
                "Stage37 JFI embeddings must be [B,M,8,selector_dim]"
            )
        raw = torch.stack([
            member(member_embeddings[:, :, index])
            for index, member in enumerate(self.members)
        ], dim=2)
        q10 = raw[..., 1]
        q50 = q10 + F.softplus(raw[..., 2])
        q90 = q50 + F.softplus(raw[..., 3])
        quantiles = torch.stack((q10, q50, q90), dim=-1)
        return {
            "joint_logits": raw[..., 0],
            "joint_probabilities": raw[..., 0].sigmoid(),
            "delta_quantiles": quantiles,
        }


def build_joint_feasible_labels(
    *,
    rewards: torch.Tensor,
    components: torch.Tensor,
    valid_mask: torch.Tensor,
    fallback_modes: torch.Tensor,
    reward_margin: float = JFI_REWARD_MARGIN,
    safety_tolerance: float = JFI_SAFETY_TOLERANCE,
) -> Dict[str, torch.Tensor]:
    """Construct the preregistered joint event relative to fallback."""
    if rewards.ndim != 2 or components.shape != (*rewards.shape, 6):
        raise ValueError("Stage37 JFI labels require [B,M] and [B,M,6]")
    batch, modes = rewards.shape
    if valid_mask.shape != rewards.shape or fallback_modes.shape != (batch,):
        raise ValueError("Stage37 JFI label shapes differ")
    if modes != 20 or reward_margin < 0 or safety_tolerance < 0:
        raise ValueError("Stage37 JFI label constants drifted")
    batch_index = torch.arange(batch, device=rewards.device)
    fallback_rewards = rewards[batch_index, fallback_modes]
    fallback_components = components[batch_index, fallback_modes]
    reward_delta = rewards.float() - fallback_rewards.float().unsqueeze(-1)
    safety_delta = (
        components[..., list(PDM_SAFETY_INDICES)].float()
        - fallback_components[:, None, list(PDM_SAFETY_INDICES)].float()
    )
    finite = (
        valid_mask.bool()
        & torch.isfinite(rewards)
        & torch.isfinite(components).all(dim=-1)
        & torch.isfinite(fallback_rewards).unsqueeze(-1)
        & torch.isfinite(fallback_components).all(dim=-1).unsqueeze(-1)
    )
    safe = (safety_delta >= -float(safety_tolerance)).all(dim=-1)
    catastrophic = reward_delta <= float(JFI_CATASTROPHIC_DELTA)
    jointly_feasible = (
        finite
        & safe
        & ~catastrophic
        & (reward_delta >= float(reward_margin))
    )
    nonfallback = torch.ones_like(finite)
    nonfallback[batch_index, fallback_modes] = False
    return {
        "joint": jointly_feasible & nonfallback,
        "safe": safe & finite & nonfallback,
        "catastrophic": catastrophic & finite & nonfallback,
        "reward_delta": reward_delta,
        "safety_delta": safety_delta,
        "valid_nonfallback": finite & nonfallback,
    }


def _focal_binary(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positive_weight: float,
    gamma: float,
) -> torch.Tensor:
    probability = logits.sigmoid()
    pt = torch.where(targets.bool(), probability, 1.0 - probability)
    weight = torch.where(
        targets.bool(),
        torch.full_like(pt, float(positive_weight)),
        torch.ones_like(pt),
    )
    return -weight * (1.0 - pt).pow(float(gamma)) * pt.clamp_min(1e-6).log()


def compute_stage37_jfi_loss(
    predictions: Dict[str, torch.Tensor],
    *,
    focal_gamma: float = 2.0,
    positive_weight: float = 1.0,
    joint_weight: float = 4.0,
    quantile_weight: float = 1.0,
    rank_weight: float = 1.0,
    rank_margin: float = JFI_REWARD_MARGIN,
) -> Dict[str, torch.Tensor]:
    """Train joint feasibility, safe delta quantiles, and pair ordering."""
    logits = predictions["stage37_jfi_joint_logits"]
    quantiles = predictions["stage37_jfi_delta_quantiles"]
    rewards = predictions["raw_rewards"].detach().float()
    components = predictions["component_scores"].detach().float()
    valid = predictions["reward_valid_mask"].bool()
    fallback = predictions["stage24_fallback_mode"].long()
    member_mask = predictions["stage24_member_training_mask"].bool()
    if logits.ndim != 3 or logits.shape[-1] != 8:
        raise ValueError("Stage37 JFI logits must be [B,20,8]")
    batch, modes, members = logits.shape
    if (
        modes != 20
        or quantiles.shape != (batch, modes, members, 3)
        or member_mask.shape != (batch, members)
    ):
        raise ValueError("Stage37 JFI training shapes drifted")
    labels = build_joint_feasible_labels(
        rewards=rewards,
        components=components,
        valid_mask=valid,
        fallback_modes=fallback,
    )
    train_mask = (
        labels["valid_nonfallback"].unsqueeze(-1)
        & member_mask.unsqueeze(1)
    )
    targets = labels["joint"].unsqueeze(-1).expand_as(logits).float()
    focal = _focal_binary(logits, targets, positive_weight, focal_gamma)
    joint_loss = (
        focal[train_mask].mean() if train_mask.any() else logits.sum() * 0.0
    )

    delta_target = labels["reward_delta"].unsqueeze(-1).unsqueeze(-1)
    error = delta_target - quantiles
    taus = quantiles.new_tensor(JFI_QUANTILES).view(1, 1, 1, 3)
    pinball = torch.maximum(taus * error, (taus - 1.0) * error)
    safe_mask = (
        labels["safe"].unsqueeze(-1) & member_mask.unsqueeze(1)
    ).unsqueeze(-1).expand_as(pinball)
    quantile_loss = (
        pinball[safe_mask].mean()
        if safe_mask.any() else quantiles.sum() * 0.0
    )

    # Pair every valid candidate with every strictly better candidate in its
    # own scene; no cross-scene label may enter the rank objective.
    true_delta = labels["reward_delta"]
    pair_valid = (
        labels["safe"][:, :, None]
        & labels["safe"][:, None, :]
        & (
            true_delta[:, :, None]
            >= true_delta[:, None, :] + float(rank_margin)
        )
    )
    predicted_q10 = quantiles[..., 0]
    pair_score = (
        predicted_q10[:, :, None, :] - predicted_q10[:, None, :, :]
    )
    pair_member_mask = pair_valid.unsqueeze(-1) & member_mask[:, None, None, :]
    rank_values = F.softplus(-pair_score)
    rank_loss_value = (
        rank_values[pair_member_mask].mean()
        if pair_member_mask.any() else predicted_q10.sum() * 0.0
    )
    total = (
        float(joint_weight) * joint_loss
        + float(quantile_weight) * quantile_loss
        + float(rank_weight) * rank_loss_value
    )
    return {
        "loss": total,
        "stage37_jfi_joint_focal_loss": joint_loss.detach(),
        "stage37_jfi_safe_quantile_loss": quantile_loss.detach(),
        "stage37_jfi_pair_rank_loss": rank_loss_value.detach(),
        "stage37_jfi_positive_fraction": labels["joint"].float().mean().detach(),
        "stage37_jfi_safe_fraction": labels["safe"].float().mean().detach(),
        "stage37_jfi_catastrophic_fraction": labels[
            "catastrophic"
        ].float().mean().detach(),
        "stage37_jfi_quantile_monotonic": (
            (quantiles[..., 0] <= quantiles[..., 1])
            & (quantiles[..., 1] <= quantiles[..., 2])
        ).all().to(logits.dtype).detach(),
    }


def select_stage37_jfi_trajectory(
    *,
    reference_logits: torch.Tensor,
    joint_probabilities: torch.Tensor,
    delta_quantiles: torch.Tensor,
    embedding_mean: torch.Tensor,
    joint_threshold: float,
    q10_floor: float,
    ood_mean: torch.Tensor,
    ood_variance: torch.Tensor,
    ood_threshold: float,
    confidence_z: float = 1.96,
    max_candidates: int = 4,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Fail-closed JFI deployment ranked by ensemble lower quantile."""
    batch, modes = reference_logits.shape
    if (
        modes != 20
        or joint_probabilities.shape != (batch, modes, 8)
        or delta_quantiles.shape != (batch, modes, 8, 3)
    ):
        raise ValueError("Stage37 JFI deployment shapes drifted")
    if (
        not 0 <= joint_threshold <= 1
        or ood_threshold < 0
        or max_candidates != 4
    ):
        raise ValueError("Stage37 JFI calibration constants are invalid")
    joint_mean = joint_probabilities.mean(dim=-1)
    joint_std = joint_probabilities.std(dim=-1, unbiased=False)
    joint_lcb = joint_mean - float(confidence_z) * joint_std
    q10 = delta_quantiles[..., 0].mean(dim=-1)
    ood_distance = stage24_ood_distance(
        embedding_mean,
        ood_mean.to(embedding_mean),
        ood_variance.to(embedding_mean),
    )
    fallback = reference_logits.argmax(dim=-1)
    batch_index = torch.arange(batch, device=reference_logits.device)
    eligible = (
        (joint_lcb >= float(joint_threshold))
        & (q10 >= float(q10_floor))
        & (ood_distance <= float(ood_threshold))
    )
    eligible[batch_index, fallback] = False
    # ``max_candidates`` includes the always-present fallback.
    challenger_limit = int(max_candidates) - 1
    top_index = q10.masked_fill(~eligible, -torch.inf).topk(
        challenger_limit, dim=-1
    ).indices
    top_valid = eligible.gather(1, top_index)
    capped = torch.zeros_like(eligible)
    capped.scatter_(1, top_index, top_valid)
    challenger = q10.masked_fill(~capped, -torch.inf).argmax(dim=-1)
    switch = capped.any(dim=-1)
    selected = torch.where(switch, challenger, fallback)
    return selected, {
        "fallback_mode": fallback,
        "challenger_mode": challenger,
        "switch": switch,
        "eligible": capped,
        "joint_mean": joint_mean,
        "joint_std": joint_std,
        "joint_lcb": joint_lcb,
        "q10": q10,
        "ood_distance": ood_distance,
        "candidate_pool_size": capped.sum(dim=-1) + 1,
    }
