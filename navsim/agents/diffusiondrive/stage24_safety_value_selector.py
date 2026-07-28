"""Cross-generator, safety-first trajectory selector for Stage24.

Stage24 is intentionally separate from the Stage23 OOF selector.  It is trained
on several frozen generator domains, calibrated as the complete deployment
ensemble on a disjoint whole-log fold, and predicts every challenger relative
to the current generator's own reference-logit top-1 trajectory.
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


STAGE24_SELECTOR_MEMBERS = 8
STAGE24_SELECTOR_MODES = 20
STAGE24_SELECTOR_DIM = 128
STAGE24_SUBBAG_FRACTION = 0.8


def load_stage24_token_log_map(path: str) -> Dict[str, str]:
    """Load a whole-log manifest and fail closed on ambiguous membership."""
    manifest = Path(path)
    if not manifest.is_file():
        raise FileNotFoundError(f"Stage24 selector manifest missing: {path}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Stage24 selector manifest has no records")
    mapping: Dict[str, str] = {}
    for record in records:
        token = str(record.get("token", ""))
        log_id = str(record.get("log_name", record.get("log_id", "")))
        if not token or not log_id:
            raise RuntimeError("Stage24 manifest record lacks token/log_name")
        if token in mapping and mapping[token] != log_id:
            raise RuntimeError(f"Stage24 token belongs to multiple logs: {token}")
        mapping[token] = log_id
    return mapping


def load_stage24_candidate_banks(
    paths: Sequence[str], allowed_tokens: Sequence[str],
) -> Tuple[Dict[str, Tuple[dict, ...]], Tuple[Tuple[str, int, str], ...]]:
    """Load an arbitrary, aligned generator-domain/namespace bank grid.

    Each artifact must contain the same token set, or a strict superset of the
    requested manifest tokens.  Artifact order is retained as the deterministic
    training cycle and returned as ``(domain, namespace, checkpoint_sha)``.
    """
    if not paths:
        raise ValueError("Stage24 requires candidate-bank artifacts")
    requested = tuple(str(token) for token in allowed_tokens)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("Stage24 allowed tokens must be non-empty and unique")
    requested_set = set(requested)
    indexed_banks = []
    combinations = []
    seen_combinations = set()
    for path_text in paths:
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(f"Stage24 candidate bank missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        records = payload.get("records", [])
        source_text = str(summary.get("source_artifact", ""))
        if source_text:
            source = Path(source_text)
            if not source.is_file():
                raise FileNotFoundError(f"Stage24 source bank missing: {source}")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest != str(summary.get("source_artifact_sha256", "")):
                raise RuntimeError("Stage24 source-bank SHA mismatch")
            source_payload = json.loads(source.read_text(encoding="utf-8"))
            source_summary = source_payload.get("summary", {})
            records = source_payload.get("records", [])
            summary = {**source_summary, **summary}
        domain = str(summary.get("generator_domain", ""))
        namespace = int(summary.get("evaluation_noise_namespace", -999))
        checkpoint_sha = str(summary.get("checkpoint_sha256", ""))
        if not domain or not checkpoint_sha:
            raise RuntimeError("Stage24 bank lacks generator domain/checkpoint SHA")
        combination = (domain, namespace)
        if combination in seen_combinations:
            raise RuntimeError(f"duplicate Stage24 bank combination: {combination}")
        seen_combinations.add(combination)
        if not summary.get("stores_candidate_trajectories"):
            raise RuntimeError("Stage24 bank does not store all trajectories")
        indexed = {}
        for record in records:
            token = str(record.get("token", ""))
            if token not in requested_set:
                continue
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
                raise RuntimeError("Stage24 bank record lacks all-20 data/provenance")
            if token in indexed:
                raise RuntimeError(f"duplicate token in Stage24 bank: {token}")
            # Do not retain evaluator-only diagnostics from twelve large JSON
            # artifacts.  Selector training needs only the frozen all-20 bank
            # fields and provenance below; copying entire records needlessly
            # multiplies host memory consumption.
            normalized = {
                "token": token,
                "log_name": str(record["log_name"]),
                "candidate_trajectories": trajectories,
                "candidate_reference_logits": logits,
                "candidate_rewards": rewards,
                "candidate_components": components,
                "generator_domain": domain,
                "generator_checkpoint_sha256": checkpoint_sha,
                "evaluation_noise_namespace": namespace,
            }
            indexed[token] = normalized
        missing = requested_set - set(indexed)
        if missing:
            raise RuntimeError(
                f"Stage24 bank lacks {len(missing)} manifest tokens; first={min(missing)}"
            )
        indexed_banks.append(indexed)
        combinations.append((domain, namespace, checkpoint_sha))
    return (
        {token: tuple(bank[token] for bank in indexed_banks) for token in requested},
        tuple(combinations),
    )


def stage24_member_training_mask(
    log_ids: Sequence[str], num_members: int = STAGE24_SELECTOR_MEMBERS,
    fraction: float = STAGE24_SUBBAG_FRACTION,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Deterministic independent whole-log sub-bags for deep-ensemble members."""
    if num_members < 2 or not 0.0 < fraction < 1.0:
        raise ValueError("Stage24 sub-bag configuration is invalid")
    threshold = int(fraction * (1 << 64))
    rows = []
    for log_id in log_ids:
        if not str(log_id):
            raise ValueError("Stage24 log identifiers must be non-empty")
        row = []
        for member in range(num_members):
            digest = hashlib.sha256(
                f"stage24:subbag:{member}:{log_id}".encode("utf-8")
            ).digest()
            row.append(int.from_bytes(digest[:8], "little") < threshold)
        if not any(row):
            # This is astronomically rare, but training must never silently
            # discard a scene from every ensemble member.
            digest = hashlib.sha256(f"stage24:fallback:{log_id}".encode()).digest()
            row[int.from_bytes(digest[:8], "little") % num_members] = True
        rows.append(row)
    return torch.tensor(rows, dtype=torch.bool, device=device)


def compute_stage24_positive_weights(
    candidate_bank: Dict[str, Tuple[dict, ...]],
) -> Tuple[Tuple[float, float, float], float]:
    """Compute the locked clipped negative/positive unsafe-event weights."""
    positives = torch.zeros(3, dtype=torch.float64)
    total = 0
    any_positive = 0
    for records in candidate_bank.values():
        for record in records:
            components = torch.as_tensor(record["candidate_components"], dtype=torch.float64)
            unsafe = components[:, list(PDM_SAFETY_INDICES)] < 0.999
            positives += unsafe.sum(dim=0)
            total += int(unsafe.shape[0])
            any_positive += int(unsafe.any(dim=-1).sum())
    if total <= 0:
        raise RuntimeError("Stage24 candidate bank has no safety labels")
    negatives = total - positives
    weights = (negatives / positives.clamp_min(1.0)).clamp(1.0, 50.0)
    any_negative = total - any_positive
    any_weight = max(1.0, min(50.0, any_negative / max(any_positive, 1)))
    return tuple(float(value) for value in weights), float(any_weight)


def _trajectory_kinematics(trajectories: torch.Tensor) -> torch.Tensor:
    xy = trajectories[..., :2]
    yaw = trajectories[..., 2:3]
    delta = torch.diff(xy, dim=-2, prepend=torch.zeros_like(xy[..., :1, :]))
    speed = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    accel = torch.diff(speed, dim=-2, prepend=torch.zeros_like(speed[..., :1, :]))
    dyaw = torch.diff(yaw, dim=-2, prepend=torch.zeros_like(yaw[..., :1, :]))
    curvature = dyaw / speed.clamp_min(1e-3)
    return torch.cat((xy, yaw, delta, speed, accel, curvature), dim=-1)


class _Stage24Member(nn.Module):
    def __init__(self, config, selector_dim: int):
        super().__init__()
        scene_dim = int(config.tf_d_model)
        heads = int(config.tf_num_head)
        if selector_dim % heads:
            raise ValueError("stage24_selector_dim must be divisible by tf_num_head")
        self.step_encoder = nn.Sequential(
            nn.Linear(8, selector_dim), nn.GELU(), nn.LayerNorm(selector_dim),
            nn.Linear(selector_dim, selector_dim), nn.GELU(),
        )
        self.trajectory_norm = nn.LayerNorm(selector_dim)
        self.reference_projection = nn.Sequential(
            nn.Linear(3, selector_dim), nn.GELU(), nn.Linear(selector_dim, selector_dim)
        )
        self.cross_bev = GridSampleCrossBEVAttention(
            embed_dims=selector_dim, num_heads=heads, num_points=8,
            config=config, in_bev_dims=scene_dim,
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
        layer = nn.TransformerEncoderLayer(
            d_model=selector_dim, nhead=heads, dim_feedforward=4 * selector_dim,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.relative_fusion = nn.Sequential(
            nn.Linear(4 * selector_dim, 2 * selector_dim), nn.GELU(),
            nn.LayerNorm(2 * selector_dim),
            nn.Linear(2 * selector_dim, selector_dim), nn.GELU(),
        )
        self.safety_head = nn.Linear(selector_dim, 3)
        self.any_unsafe_head = nn.Linear(selector_dim, 1)
        self.component_delta_head = nn.Linear(selector_dim, PDM_COMPONENT_COUNT)
        self.pdms_delta_head = nn.Linear(selector_dim, 1)

    def forward(
        self, trajectories: torch.Tensor, reference_logits: torch.Tensor,
        fallback: torch.Tensor, bev_feature: torch.Tensor, bev_spatial_shape,
        agents_query: torch.Tensor, ego_query: torch.Tensor,
        status_encoding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.trajectory_norm(
            self.step_encoder(_trajectory_kinematics(trajectories)).mean(dim=-2)
        )
        probabilities = reference_logits.softmax(dim=-1)
        centered = reference_logits - reference_logits.mean(dim=-1, keepdim=True)
        scaled = centered / centered.std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(1e-5)
        query = query + self.reference_projection(
            torch.stack((reference_logits, probabilities, scaled), dim=-1)
        )
        query = self.cross_bev(
            query, trajectories[..., :2], bev_feature, bev_spatial_shape
        )
        query = query + self.cross_agents(query, agents_query, agents_query)[0]
        query = query + self.cross_ego(query, ego_query, ego_query)[0]
        status = status_encoding.unsqueeze(1) if status_encoding.ndim == 2 else status_encoding
        query = self.context_norm(query + self.status_projection(status))
        query = self.set_encoder(query)
        batch_index = torch.arange(query.shape[0], device=query.device)
        fallback_query = query[batch_index, fallback].unsqueeze(1).expand_as(query)
        relative = self.relative_fusion(torch.cat(
            (query, fallback_query, query - fallback_query, query * fallback_query),
            dim=-1,
        ))
        embedding = F.normalize(relative, dim=-1)
        return (
            self.safety_head(relative),
            self.any_unsafe_head(relative).squeeze(-1),
            self.component_delta_head(relative),
            self.pdms_delta_head(relative).squeeze(-1),
            embedding,
        )


class Stage24SafetyValueSelector(nn.Module):
    """Eight-member cross-generator selector calibrated as a full ensemble."""

    def __init__(self, config):
        super().__init__()
        self.num_members = int(getattr(
            config, "stage24_selector_num_members", STAGE24_SELECTOR_MEMBERS
        ))
        self.num_modes = int(getattr(
            config, "stage24_selector_num_modes", STAGE24_SELECTOR_MODES
        ))
        self.selector_dim = int(getattr(
            config, "stage24_selector_dim", STAGE24_SELECTOR_DIM
        ))
        if self.num_members != STAGE24_SELECTOR_MEMBERS:
            raise ValueError("formal Stage24 requires exactly eight members")
        if self.num_modes != STAGE24_SELECTOR_MODES:
            raise ValueError("formal Stage24 requires exactly twenty modes")
        with torch.random.fork_rng(devices=[]):
            self.members = nn.ModuleList([
                _Stage24Member(config, self.selector_dim)
                for _ in range(self.num_members)
            ])

    def forward(
        self, trajectories: torch.Tensor, reference_logits: torch.Tensor,
        bev_feature: torch.Tensor, bev_spatial_shape, agents_query: torch.Tensor,
        ego_query: torch.Tensor, status_encoding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if trajectories.ndim != 4 or trajectories.shape[-2:] != (8, 3):
            raise ValueError("Stage24 trajectories must have shape [B,M,8,3]")
        batch, modes = trajectories.shape[:2]
        if modes != self.num_modes or reference_logits.shape != (batch, modes):
            raise ValueError("Stage24 selector requires matching [B,20] logits")
        fallback = reference_logits.argmax(dim=-1)
        outputs = [
            member(
                trajectories, reference_logits, fallback, bev_feature,
                bev_spatial_shape, agents_query, ego_query, status_encoding,
            )
            for member in self.members
        ]
        safety_logits = torch.stack([item[0] for item in outputs], dim=2)
        any_logits = torch.stack([item[1] for item in outputs], dim=2)
        component_delta = torch.stack([item[2] for item in outputs], dim=2)
        delta = torch.stack([item[3] for item in outputs], dim=2)
        embedding = torch.stack([item[4] for item in outputs], dim=2)
        return {
            "fallback_mode": fallback,
            "safety_logits": safety_logits,
            "safety_probabilities": safety_logits.sigmoid(),
            "any_unsafe_logits": any_logits,
            "any_unsafe_probabilities": any_logits.sigmoid(),
            "component_delta_predictions": component_delta,
            "delta_predictions": delta,
            "embedding_predictions": embedding,
            "embedding_mean": embedding.mean(dim=2),
        }


def _focal_binary_loss(
    logits: torch.Tensor, targets: torch.Tensor, positive_weight: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    probabilities = logits.sigmoid()
    pt = torch.where(targets.bool(), probabilities, 1.0 - probabilities)
    weight = torch.where(targets.bool(), positive_weight, torch.ones_like(targets))
    return -weight * (1.0 - pt).pow(gamma) * pt.clamp_min(1e-6).log()


def compute_stage24_selector_loss(
    predictions: Dict[str, torch.Tensor], reward_gap: float = 0.005,
    focal_gamma: float = 2.0, hard_negative_count: int = 4,
) -> Dict[str, torch.Tensor]:
    """Safety-first relative-value loss with online hard false-safe mining."""
    safety_logits = predictions["stage24_safety_logits"]
    any_logits = predictions["stage24_any_unsafe_logits"]
    component_delta = predictions["stage24_component_delta_predictions"]
    delta = predictions["stage24_delta_predictions"]
    targets = predictions["component_scores"].detach().float()
    rewards = predictions["raw_rewards"].detach().float()
    valid = predictions["reward_valid_mask"].bool()
    member_mask = predictions["stage24_member_training_mask"].bool()
    fallback = predictions["stage24_fallback_mode"].long()
    safety_weights = predictions["stage24_safety_positive_weights"].float()
    any_weight = predictions["stage24_any_unsafe_positive_weight"].float()
    if safety_logits.ndim != 4 or safety_logits.shape[-2:] != (8, 3):
        raise ValueError("Stage24 safety logits must have shape [B,20,8,3]")
    batch, modes, members, _ = safety_logits.shape
    if modes != 20 or any_logits.shape != (batch, modes, members):
        raise ValueError("Stage24 selector prediction shape mismatch")
    if component_delta.shape != (batch, modes, members, 6):
        raise ValueError("Stage24 component-delta shape mismatch")
    if delta.shape != (batch, modes, members):
        raise ValueError("Stage24 PDMS-delta shape mismatch")
    if targets.shape != (batch, modes, 6) or rewards.shape != (batch, modes):
        raise ValueError("Stage24 target shape mismatch")
    if member_mask.shape != (batch, members) or fallback.shape != (batch,):
        raise ValueError("Stage24 member/fallback shape mismatch")
    if safety_weights.numel() != 3 or any_weight.numel() != 1:
        raise ValueError("Stage24 unsafe positive weights have invalid shape")

    finite = valid & torch.isfinite(rewards) & torch.isfinite(targets).all(dim=-1)
    batch_index = torch.arange(batch, device=rewards.device)
    fallback_reward = rewards[batch_index, fallback]
    fallback_components = targets[batch_index, fallback]
    target_delta = rewards - fallback_reward.unsqueeze(1)
    target_component_delta = targets - fallback_components.unsqueeze(1)
    unsafe = targets[..., list(PDM_SAFETY_INDICES)] < 0.999
    any_unsafe = unsafe.any(dim=-1)
    base_mask = finite.unsqueeze(-1) & member_mask.unsqueeze(1)

    expanded_unsafe = unsafe.unsqueeze(2).expand_as(safety_logits).float()
    safety_loss_values = _focal_binary_loss(
        safety_logits, expanded_unsafe,
        safety_weights.to(safety_logits.device).view(1, 1, 1, 3), focal_gamma,
    )
    safety_mask = base_mask.unsqueeze(-1).expand_as(safety_loss_values)
    safety_loss = safety_loss_values[safety_mask].mean()

    expanded_any = any_unsafe.unsqueeze(-1).expand_as(any_logits).float()
    any_loss_values = _focal_binary_loss(
        any_logits, expanded_any,
        any_weight.to(any_logits.device).view(1, 1, 1), focal_gamma,
    )
    any_loss = any_loss_values[base_mask].mean()

    nonfallback = torch.ones(batch, modes, dtype=torch.bool, device=rewards.device)
    nonfallback[batch_index, fallback] = False
    safe = ~any_unsafe
    value_mask = base_mask & safe.unsqueeze(-1) & nonfallback.unsqueeze(-1)
    expanded_delta = target_delta.unsqueeze(-1).expand_as(delta)
    pdms_loss = (
        F.smooth_l1_loss(delta[value_mask], expanded_delta[value_mask])
        if value_mask.any() else delta.sum() * 0.0
    )
    expanded_component_delta = target_component_delta.unsqueeze(2).expand_as(component_delta)
    component_mask = value_mask.unsqueeze(-1).expand_as(component_delta)
    component_loss = (
        F.smooth_l1_loss(
            component_delta[component_mask], expanded_component_delta[component_mask]
        ) if component_mask.any() else component_delta.sum() * 0.0
    )

    reward_pair_delta = rewards.unsqueeze(2) - rewards.unsqueeze(1)
    predicted_pair_delta = delta.unsqueeze(2) - delta.unsqueeze(1)
    upper = torch.triu(torch.ones(modes, modes, dtype=torch.bool, device=rewards.device), 1)
    pair_valid = (
        finite.unsqueeze(2) & finite.unsqueeze(1)
        & safe.unsqueeze(2) & safe.unsqueeze(1)
        & upper.unsqueeze(0) & (reward_pair_delta.abs() >= reward_gap)
    )
    pair_mask = pair_valid.unsqueeze(-1) & member_mask[:, None, None, :]
    rank_values = F.softplus(
        -reward_pair_delta.sign().unsqueeze(-1) * predicted_pair_delta
    ) * reward_pair_delta.abs().unsqueeze(-1)
    rank_loss = (
        rank_values[pair_mask].mean() if pair_mask.any() else delta.sum() * 0.0
    )

    risk = torch.maximum(
        any_logits.sigmoid(), safety_logits.sigmoid().amax(dim=-1)
    )
    hard_candidate = any_unsafe & nonfallback
    mining_score = delta.detach().masked_fill(
        ~hard_candidate.unsqueeze(-1), -torch.inf
    )
    topk = min(max(int(hard_negative_count), 1), modes)
    hard_index = mining_score.topk(topk, dim=1).indices
    hard_risk = risk.gather(1, hard_index)
    hard_valid = hard_candidate.unsqueeze(-1).expand_as(risk).gather(
        1, hard_index
    )
    hard_mask = hard_valid & member_mask.unsqueeze(1)
    hard_loss = (
        F.relu(0.5 - hard_risk[hard_mask]).mean()
        if hard_mask.any() else risk.sum() * 0.0
    )

    total = (
        2.0 * any_loss + safety_loss + 2.0 * hard_loss
        + pdms_loss + 0.5 * component_loss + rank_loss
    )
    return {
        "loss": total,
        "stage24_any_unsafe_loss": any_loss.detach(),
        "stage24_safety_loss": safety_loss.detach(),
        "stage24_false_safe_hinge_loss": hard_loss.detach(),
        "stage24_pdms_delta_loss": pdms_loss.detach(),
        "stage24_component_delta_loss": component_loss.detach(),
        "stage24_safe_pair_rank_loss": rank_loss.detach(),
        "stage24_active_pair_count": total.new_tensor(float(pair_mask.sum().item())),
        "stage24_member_fraction": member_mask.float().mean().detach(),
    }


def stage24_ood_distance(
    embedding: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor,
) -> torch.Tensor:
    """Dimension-normalized diagonal Mahalanobis distance."""
    if embedding.shape[-1] != mean.numel() or mean.shape != variance.shape:
        raise ValueError("Stage24 OOD calibration dimension mismatch")
    return ((embedding - mean) ** 2 / variance.clamp_min(1e-6)).mean(dim=-1)


def select_stage24_trajectory(
    reference_logits: torch.Tensor, safety_probabilities: torch.Tensor,
    any_unsafe_probabilities: torch.Tensor,
    component_delta_predictions: torch.Tensor,
    delta_predictions: torch.Tensor, embedding_mean: torch.Tensor,
    residual_margin: float, risk_threshold: float,
    ood_mean: torch.Tensor, ood_variance: torch.Tensor, ood_threshold: float,
    confidence_z: float = 1.96, safety_tolerance: float = 0.001,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Select the best eligible challenger or the same-generator fallback."""
    if reference_logits.ndim != 2:
        raise ValueError("Stage24 reference logits must have shape [B,M]")
    batch, modes = reference_logits.shape
    expected_members = STAGE24_SELECTOR_MEMBERS
    if safety_probabilities.shape != (batch, modes, expected_members, 3):
        raise ValueError("Stage24 safety ensemble shape mismatch")
    if any_unsafe_probabilities.shape != (batch, modes, expected_members):
        raise ValueError("Stage24 any-unsafe ensemble shape mismatch")
    if component_delta_predictions.shape != (batch, modes, expected_members, 6):
        raise ValueError("Stage24 component-delta ensemble shape mismatch")
    if delta_predictions.shape != (batch, modes, expected_members):
        raise ValueError("Stage24 value ensemble shape mismatch")
    if embedding_mean.shape[:2] != (batch, modes):
        raise ValueError("Stage24 embedding shape mismatch")
    if residual_margin < 0 or confidence_z < 0 or ood_threshold < 0:
        raise ValueError("Stage24 calibration margins must be non-negative")
    if not 0.0 <= risk_threshold <= 1.0:
        raise ValueError("Stage24 risk threshold must lie in [0,1]")

    fallback = reference_logits.argmax(dim=-1)
    risk_members = torch.maximum(
        any_unsafe_probabilities, safety_probabilities.amax(dim=-1)
    )
    risk_ucb = risk_members.mean(dim=-1) + confidence_z * risk_members.std(
        dim=-1, unbiased=False
    )
    delta_mean = delta_predictions.mean(dim=-1)
    delta_std = delta_predictions.std(dim=-1, unbiased=False)
    delta_lcb = delta_mean - confidence_z * delta_std - residual_margin
    safety_delta = component_delta_predictions[..., list(PDM_SAFETY_INDICES)]
    safety_delta_lcb = safety_delta.mean(dim=2) - confidence_z * safety_delta.std(
        dim=2, unbiased=False
    )
    ood_distance = stage24_ood_distance(
        embedding_mean, ood_mean.to(embedding_mean), ood_variance.to(embedding_mean)
    )
    eligible = (
        (risk_ucb <= risk_threshold)
        & (safety_delta_lcb >= -safety_tolerance).all(dim=-1)
        & (delta_lcb > 0)
        & (ood_distance <= ood_threshold)
    )
    batch_index = torch.arange(batch, device=reference_logits.device)
    eligible[batch_index, fallback] = False
    conservative_value = delta_lcb.masked_fill(~eligible, -torch.inf)
    challenger = conservative_value.argmax(dim=-1)
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
        "safety_delta_lcb": safety_delta_lcb,
        "ood_distance": ood_distance,
        "ood_rejected": ood_distance > ood_threshold,
    }
