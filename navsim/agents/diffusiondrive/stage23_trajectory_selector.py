"""Deployment-aligned, whole-log OOF trajectory selector for Stage23.

This module is intentionally separate from the Stage15 top-2 selector.  A
Stage23 member owns its complete feature extractor, is trained on four of five
deterministic *log* buckets, and scores every trajectory in the 20-mode set.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.diffusiondrive.modules.blocks import GridSampleCrossBEVAttention
from navsim.agents.diffusiondrive.trajectory_value_selector import (
    PDM_COMPONENT_COUNT,
    PDM_SAFETY_INDICES,
)


STAGE23_SELECTOR_MEMBERS = 5
STAGE23_SELECTOR_MODES = 20


def load_stage23_token_log_map(path: str) -> Dict[str, str]:
    """Load token/log provenance and fail closed on ambiguous membership."""
    manifest = Path(path)
    if not manifest.is_file():
        raise FileNotFoundError(f"Stage23 selector manifest missing: {path}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Stage23 selector manifest has no records")
    mapping: Dict[str, str] = {}
    for record in records:
        token = str(record.get("token", ""))
        log_id = str(record.get("log_name", record.get("log_id", "")))
        if not token or not log_id:
            raise RuntimeError("Stage23 selector manifest record lacks token/log_name")
        if token in mapping and mapping[token] != log_id:
            raise RuntimeError(f"Stage23 token belongs to multiple logs: {token}")
        mapping[token] = log_id
    return mapping


def load_stage23_candidate_banks(
    paths: Sequence[str], expected_checkpoint_sha256: str,
    expected_namespaces: Sequence[int] = (-1, 20260811, 20260812),
) -> Dict[str, Tuple[dict, ...]]:
    """Load the locked all-20 banks and verify cross-namespace alignment."""
    if len(paths) != len(expected_namespaces):
        raise ValueError("Stage23 requires exactly three candidate-bank artifacts")
    per_namespace = []
    canonical_tokens = None
    for path_text, expected_namespace in zip(paths, expected_namespaces):
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(f"Stage23 candidate bank missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        if summary.get("checkpoint_sha256") != expected_checkpoint_sha256:
            raise RuntimeError("Stage23 candidate-bank generator SHA mismatch")
        if int(summary.get("evaluation_noise_namespace", -999)) != int(expected_namespace):
            raise RuntimeError("Stage23 candidate-bank namespace/order mismatch")
        if not summary.get("stores_candidate_trajectories"):
            raise RuntimeError("Stage23 bank does not store all candidate trajectories")
        records = payload.get("records", [])
        tokens = [str(record.get("token", "")) for record in records]
        if not tokens or len(tokens) != len(set(tokens)):
            raise RuntimeError("Stage23 candidate bank has empty/duplicate tokens")
        if canonical_tokens is None:
            canonical_tokens = tokens
        elif canonical_tokens != tokens:
            raise RuntimeError("Stage23 candidate-bank token order differs by namespace")
        indexed = {}
        for record in records:
            trajectories = record.get("candidate_trajectories")
            logits = record.get("candidate_reference_logits")
            rewards = record.get("candidate_rewards")
            components = record.get("candidate_components")
            if (
                not isinstance(trajectories, list) or len(trajectories) != 20
                or not isinstance(logits, list) or len(logits) != 20
                or not isinstance(rewards, list) or len(rewards) != 20
                or not isinstance(components, list) or len(components) != 20
                or not str(record.get("log_name", ""))
            ):
                raise RuntimeError("Stage23 bank record lacks all-20 data/provenance")
            indexed[str(record["token"])] = record
        per_namespace.append(indexed)
    return {
        token: tuple(namespace[token] for namespace in per_namespace)
        for token in canonical_tokens
    }


def deterministic_log_bucket(log_ids: Sequence[str], num_members: int = 5) -> torch.Tensor:
    """Assign complete logs to a stable OOF bucket.

    All frames from the same log necessarily receive the same bucket.  The
    explicit namespace prevents earlier token-bootstrap assignments from being
    silently reused as Stage23 folds.
    """
    if num_members < 2:
        raise ValueError("Stage23 OOF requires at least two members")
    buckets = []
    for log_id in log_ids:
        if not str(log_id):
            raise ValueError("Stage23 OOF log identifiers must be non-empty")
        digest = hashlib.sha256(f"stage23:whole-log:{log_id}".encode()).digest()
        buckets.append(int.from_bytes(digest[:8], "little") % num_members)
    return torch.tensor(buckets, dtype=torch.long)


def member_training_mask(
    log_ids: Sequence[str], num_members: int, device: torch.device
) -> torch.Tensor:
    """Return [B,H], false for the one member holding out each sample's log."""
    buckets = deterministic_log_bucket(log_ids, num_members).to(device)
    members = torch.arange(num_members, device=device)
    return buckets.unsqueeze(1) != members.unsqueeze(0)


def gather_oof_predictions(predictions: torch.Tensor, log_ids: Sequence[str]) -> torch.Tensor:
    """Select the sole prediction from the member that held out each log.

    ``predictions`` is [B,M,H,...].  The returned tensor is [B,M,...].
    """
    if predictions.ndim < 3:
        raise ValueError("OOF predictions must have shape [B,M,H,...]")
    batch, _, members = predictions.shape[:3]
    if len(log_ids) != batch:
        raise ValueError("OOF log count must equal prediction batch size")
    buckets = deterministic_log_bucket(log_ids, members).to(predictions.device)
    index_shape = (batch, predictions.shape[1], 1) + predictions.shape[3:]
    index = buckets.view(batch, 1, 1, *([1] * (predictions.ndim - 3)))
    index = index.expand(index_shape)
    return predictions.gather(2, index).squeeze(2)


def _trajectory_kinematics(trajectories: torch.Tensor) -> torch.Tensor:
    """Create local geometry/kinematics without coupling candidate order."""
    xy = trajectories[..., :2]
    yaw = trajectories[..., 2:3]
    delta = torch.diff(xy, dim=-2, prepend=torch.zeros_like(xy[..., :1, :]))
    speed = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    accel = torch.diff(speed, dim=-2, prepend=torch.zeros_like(speed[..., :1, :]))
    dyaw = torch.diff(yaw, dim=-2, prepend=torch.zeros_like(yaw[..., :1, :]))
    curvature = dyaw / speed.clamp_min(1e-3)
    return torch.cat((xy, yaw, delta, speed, accel, curvature), dim=-1)


class _SelectorMember(nn.Module):
    """One independently initialized scene/trajectory value model."""

    def __init__(self, config, selector_dim: int, max_modes: int):
        super().__init__()
        scene_dim = int(config.tf_d_model)
        heads = int(config.tf_num_head)
        if selector_dim % heads:
            raise ValueError("stage23_selector_dim must be divisible by tf_num_head")
        self.max_modes = int(max_modes)
        self.step_encoder = nn.Sequential(
            nn.Linear(8, selector_dim), nn.GELU(), nn.LayerNorm(selector_dim),
            nn.Linear(selector_dim, selector_dim), nn.GELU(),
        )
        self.trajectory_norm = nn.LayerNorm(selector_dim)
        self.mode_embedding = nn.Embedding(self.max_modes, selector_dim)
        self.reference_projection = nn.Sequential(
            nn.Linear(3, selector_dim), nn.GELU(), nn.Linear(selector_dim, selector_dim)
        )
        self.cross_bev = GridSampleCrossBEVAttention(
            embed_dims=selector_dim,
            num_heads=heads,
            num_points=8,
            config=config,
            in_bev_dims=scene_dim,
        )
        self.cross_agents = nn.MultiheadAttention(
            selector_dim, heads, dropout=0.0, batch_first=True,
            kdim=scene_dim, vdim=scene_dim,
        )
        self.cross_ego = nn.MultiheadAttention(
            selector_dim, heads, dropout=0.0, batch_first=True,
            kdim=scene_dim, vdim=scene_dim,
        )
        self.status_projection = nn.Linear(scene_dim, selector_dim)
        self.context_norm = nn.LayerNorm(selector_dim)
        self.component_head = nn.Sequential(
            nn.Linear(selector_dim, selector_dim), nn.GELU(),
            nn.Linear(selector_dim, PDM_COMPONENT_COUNT),
        )
        self.pdms_head = nn.Sequential(
            nn.Linear(selector_dim, selector_dim), nn.GELU(), nn.Linear(selector_dim, 1)
        )

    def forward(
        self,
        trajectories: torch.Tensor,
        reference_logits: torch.Tensor,
        mode_ids: torch.Tensor,
        bev_feature: torch.Tensor,
        bev_spatial_shape,
        agents_query: torch.Tensor,
        ego_query: torch.Tensor,
        status_encoding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kinematics = _trajectory_kinematics(trajectories)
        query = self.trajectory_norm(self.step_encoder(kinematics).mean(dim=-2))
        probabilities = reference_logits.softmax(dim=-1)
        centered = reference_logits - reference_logits.mean(dim=-1, keepdim=True)
        scaled = centered / centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
        ref_features = torch.stack((reference_logits, probabilities, scaled), dim=-1)
        query = query + self.reference_projection(ref_features)
        query = query + self.mode_embedding(mode_ids)
        query = self.cross_bev(
            query, trajectories[..., :2], bev_feature, bev_spatial_shape
        )
        query = query + self.cross_agents(query, agents_query, agents_query)[0]
        query = query + self.cross_ego(query, ego_query, ego_query)[0]
        status = status_encoding.unsqueeze(1) if status_encoding.ndim == 2 else status_encoding
        query = self.context_norm(query + self.status_projection(status))
        component_logits = self.component_head(query)
        components = component_logits.sigmoid()
        pdms = self.pdms_head(query).squeeze(-1).sigmoid()
        return component_logits, components, pdms


class Stage23TrajectorySelector(nn.Module):
    """Five complete members for log-OOF calibration and deployment ensemble."""

    def __init__(self, config):
        super().__init__()
        self.num_members = int(
            getattr(config, "stage23_selector_num_members", STAGE23_SELECTOR_MEMBERS)
        )
        self.max_modes = int(
            getattr(config, "stage23_selector_num_modes", STAGE23_SELECTOR_MODES)
        )
        selector_dim = int(getattr(config, "stage23_selector_dim", 128))
        if self.num_members != STAGE23_SELECTOR_MEMBERS:
            raise ValueError("formal Stage23 requires exactly five selector members")
        if self.max_modes != STAGE23_SELECTOR_MODES:
            raise ValueError("formal Stage23 requires exactly twenty modes")
        # fork_rng avoids perturbing the legacy decoder initialization stream.
        with torch.random.fork_rng(devices=[]):
            self.members = nn.ModuleList(
                [_SelectorMember(config, selector_dim, self.max_modes) for _ in range(self.num_members)]
            )

    def forward(
        self,
        trajectories: torch.Tensor,
        reference_logits: torch.Tensor,
        bev_feature: torch.Tensor,
        bev_spatial_shape,
        agents_query: torch.Tensor,
        ego_query: torch.Tensor,
        status_encoding: torch.Tensor,
        mode_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if trajectories.ndim != 4 or trajectories.shape[-2:] != (8, 3):
            raise ValueError("Stage23 trajectories must have shape [B,M,8,3]")
        batch, modes = trajectories.shape[:2]
        if modes != self.max_modes or reference_logits.shape != (batch, modes):
            raise ValueError("Stage23 selector requires matching [B,20] reference logits")
        if mode_ids is None:
            mode_ids = torch.arange(modes, device=trajectories.device).expand(batch, -1)
        if mode_ids.shape != (batch, modes):
            raise ValueError("Stage23 mode_ids must have shape [B,20]")
        outputs = [
            member(
                trajectories, reference_logits, mode_ids, bev_feature,
                bev_spatial_shape, agents_query, ego_query, status_encoding,
            )
            for member in self.members
        ]
        component_logits = torch.stack([item[0] for item in outputs], dim=2)
        components = torch.stack([item[1] for item in outputs], dim=2)
        scores = torch.stack([item[2] for item in outputs], dim=2)
        return {
            "component_logits": component_logits,
            "component_predictions": components,
            "score_predictions": scores,
            "component_mean": components.mean(dim=2),
            "component_std": components.std(dim=2, unbiased=False),
            "score_mean": scores.mean(dim=2),
            "score_std": scores.std(dim=2, unbiased=False),
        }


def compute_stage23_selector_loss(
    predictions: Dict[str, torch.Tensor], reward_gap: float = 0.01,
    focal_gamma: float = 2.0, unsafe_positive_weight: float = 50.0,
) -> Dict[str, torch.Tensor]:
    """PDM component supervision plus gap-weighted all-pairs ranking."""
    logits = predictions["stage23_component_logits"]
    components = predictions["stage23_component_predictions"]
    scores = predictions["stage23_score_predictions"]
    targets = predictions["component_scores"].detach().float()
    rewards = predictions["raw_rewards"].detach().float()
    valid = predictions["reward_valid_mask"].bool()
    train_member = predictions["stage23_member_training_mask"].bool()
    if logits.shape != components.shape or logits.ndim != 4 or logits.shape[-1] != 6:
        raise ValueError("Stage23 component predictions must have shape [B,20,5,6]")
    batch, modes, members, _ = logits.shape
    if modes != STAGE23_SELECTOR_MODES or members != STAGE23_SELECTOR_MEMBERS:
        raise ValueError("Stage23 loss requires exactly 20 modes and five members")
    if scores.shape != (batch, modes, members):
        raise ValueError("Stage23 score prediction shape mismatch")
    if targets.shape != (batch, modes, 6) or rewards.shape != (batch, modes):
        raise ValueError("Stage23 PDM target shape mismatch")
    if valid.shape != rewards.shape or train_member.shape != (batch, members):
        raise ValueError("Stage23 valid/member mask shape mismatch")

    finite = valid & torch.isfinite(rewards) & torch.isfinite(targets).all(dim=-1)
    mask = finite.unsqueeze(-1) & train_member.unsqueeze(1)
    expanded_targets = targets.unsqueeze(2).expand_as(components)

    safety_index = torch.tensor(PDM_SAFETY_INDICES, device=logits.device)
    # Safety components are compliance scores; train rare unsafe events as positives.
    unsafe_logits = -logits.index_select(-1, safety_index)
    unsafe_targets = (expanded_targets.index_select(-1, safety_index) < 0.999).float()
    unsafe_probability = unsafe_logits.sigmoid()
    pt = torch.where(unsafe_targets.bool(), unsafe_probability, 1.0 - unsafe_probability)
    class_weight = torch.where(
        unsafe_targets.bool(),
        unsafe_targets.new_tensor(unsafe_positive_weight),
        unsafe_targets.new_tensor(1.0),
    )
    focal = -class_weight * (1.0 - pt).pow(focal_gamma) * pt.clamp_min(1e-6).log()
    safety_mask = mask.unsqueeze(-1).expand_as(focal)
    safety_loss = focal[safety_mask].mean() if safety_mask.any() else logits.sum() * 0.0

    component_mask = mask.unsqueeze(-1).expand_as(components)
    component_loss = (
        F.smooth_l1_loss(components[component_mask], expanded_targets[component_mask])
        if component_mask.any() else components.sum() * 0.0
    )
    expanded_rewards = rewards.unsqueeze(-1).expand_as(scores)
    pdms_loss = (
        F.smooth_l1_loss(scores[mask], expanded_rewards[mask])
        if mask.any() else scores.sum() * 0.0
    )

    reward_delta = rewards.unsqueeze(2) - rewards.unsqueeze(1)  # [B,M,M]
    score_delta = scores.unsqueeze(2) - scores.unsqueeze(1)  # [B,M,M,H]
    pair_valid = finite.unsqueeze(2) & finite.unsqueeze(1)
    upper = torch.triu(torch.ones(modes, modes, dtype=torch.bool, device=scores.device), diagonal=1)
    active = pair_valid & upper.unsqueeze(0) & (reward_delta.abs() >= reward_gap)
    pair_mask = active.unsqueeze(-1) & train_member[:, None, None, :]
    rank_values = F.softplus(-reward_delta.sign().unsqueeze(-1) * score_delta)
    rank_values = rank_values * reward_delta.abs().unsqueeze(-1)
    rank_loss = rank_values[pair_mask].mean() if pair_mask.any() else scores.sum() * 0.0

    total = safety_loss + component_loss + pdms_loss + rank_loss
    return {
        "loss": total,
        "stage23_safety_focal_loss": safety_loss.detach(),
        "stage23_component_loss": component_loss.detach(),
        "stage23_pdms_loss": pdms_loss.detach(),
        "stage23_rank_loss": rank_loss.detach(),
        "stage23_valid_candidate_fraction": finite.float().mean().detach(),
        "stage23_training_member_fraction": train_member.float().mean().detach(),
        "stage23_active_pair_count": total.new_tensor(float(pair_mask.sum().item())),
    }


def select_stage23_trajectory(
    reference_logits: torch.Tensor,
    component_predictions: torch.Tensor,
    score_predictions: torch.Tensor,
    residual_margin: float,
    safety_threshold: float = 0.95,
    confidence_z: float = 1.96,
    safety_relative_tolerance: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Conservatively rerank all modes, falling back to reference top-1."""
    if reference_logits.ndim != 2:
        raise ValueError("Stage23 reference logits must have shape [B,M]")
    batch, modes = reference_logits.shape
    if component_predictions.shape != (batch, modes, STAGE23_SELECTOR_MEMBERS, 6):
        raise ValueError("Stage23 component ensemble shape mismatch")
    if score_predictions.shape != (batch, modes, STAGE23_SELECTOR_MEMBERS):
        raise ValueError("Stage23 score ensemble shape mismatch")
    if residual_margin < 0 or confidence_z < 0:
        raise ValueError("Stage23 calibration margin/z must be non-negative")
    if not 0 <= safety_threshold <= 1:
        raise ValueError("Stage23 safety threshold must be in [0,1]")

    fallback = reference_logits.argmax(dim=-1)
    batch_index = torch.arange(batch, device=reference_logits.device)
    fallback_member_scores = score_predictions[batch_index, fallback]
    paired_delta = score_predictions - fallback_member_scores.unsqueeze(1)
    advantage_mean = paired_delta.mean(dim=-1)
    advantage_std = paired_delta.std(dim=-1, unbiased=False)
    advantage_lcb = advantage_mean - confidence_z * advantage_std - residual_margin

    safety_index = torch.tensor(PDM_SAFETY_INDICES, device=reference_logits.device)
    safety = component_predictions.index_select(-1, safety_index)
    safety_lower = safety.mean(dim=2) - confidence_z * safety.std(dim=2, unbiased=False)
    fallback_safety = safety_lower[batch_index, fallback]
    absolute_safe = (safety_lower >= safety_threshold).all(dim=-1)
    relative_safe = (
        safety_lower >= fallback_safety.unsqueeze(1) - safety_relative_tolerance
    ).all(dim=-1)
    eligible = absolute_safe & relative_safe & (advantage_lcb > 0)
    eligible[batch_index, fallback] = False
    conservative_value = advantage_lcb.masked_fill(~eligible, -torch.inf)
    challenger = conservative_value.argmax(dim=-1)
    switch = eligible.any(dim=-1)
    selected = torch.where(switch, challenger, fallback)
    return selected, {
        "fallback_mode": fallback,
        "challenger_mode": challenger,
        "switch": switch,
        "eligible": eligible,
        "advantage_mean": advantage_mean,
        "advantage_std": advantage_std,
        "advantage_lcb": advantage_lcb,
        "safety_lower": safety_lower,
    }
