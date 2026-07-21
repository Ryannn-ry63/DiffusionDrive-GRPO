"""Probability helpers for reward-guided truncated diffusion."""

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn


TRUST_PROJECTION_FORMULA_VERSION = "reference_mean_ball_v1"
TRUST_PROJECTION_POST_TOLERANCE = 1e-6
INFERENCE_SELECTOR_SOURCES = (
    "current", "reference", "value_top2", "paired_tail_risk"
)


def validate_inference_selector_source(source: str) -> str:
    """Validate and normalize the selector used only for deployed inference."""
    source = str(source)
    if source not in INFERENCE_SELECTOR_SOURCES:
        raise ValueError(
            "inference_selector_source must be current, reference, value_top2, "
            "or paired_tail_risk; "
            f"got {source!r}"
        )
    return source


def select_inference_mode(
    current_logits: torch.Tensor,
    reference_logits: Optional[torch.Tensor],
    source: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the deployed mode and logits without touching candidate trajectories."""
    source = validate_inference_selector_source(source)
    if source in {"value_top2", "paired_tail_risk"}:
        raise ValueError(
            f"{source} requires trajectory-conditioned predictions and must be "
            "resolved by TrajectoryHead"
        )
    if current_logits.ndim != 2:
        raise ValueError("selector logits must have shape [batch, modes]")
    if source == "reference":
        if reference_logits is None:
            raise RuntimeError(
                "reference inference selector requires frozen reference logits"
            )
        if reference_logits.shape != current_logits.shape:
            raise ValueError(
                "current and reference selector logits must have identical shapes"
            )
        selector_logits = reference_logits.detach()
    else:
        selector_logits = current_logits
    return selector_logits.argmax(dim=-1), selector_logits


class AdaptiveKLController(nn.Module):
    """Checkpointable one-sided controller for a generation KL penalty."""

    def __init__(
        self,
        initial_coefficient: float,
        minimum_coefficient: float,
        maximum_coefficient: float,
        target: float,
        hard_limit: float,
        window: int,
        update_interval: int,
        adaptation_factor: float,
        lower_ratio: float,
        upper_ratio: float,
        hard_limit_patience: int,
    ) -> None:
        super().__init__()
        values = (
            initial_coefficient,
            minimum_coefficient,
            maximum_coefficient,
            target,
            hard_limit,
            adaptation_factor,
            lower_ratio,
            upper_ratio,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("adaptive KL parameters must be finite")
        if not 0 < minimum_coefficient <= initial_coefficient <= maximum_coefficient:
            raise ValueError("adaptive KL coefficient bounds are invalid")
        if target <= 0 or hard_limit <= target:
            raise ValueError("adaptive KL requires 0 < target < hard_limit")
        if window <= 0 or update_interval <= 0 or hard_limit_patience <= 0:
            raise ValueError("adaptive KL integer parameters must be positive")
        if adaptation_factor <= 1:
            raise ValueError("adaptive KL adaptation factor must be > 1")
        if not 0 < lower_ratio < 1 < upper_ratio:
            raise ValueError("adaptive KL ratios must straddle 1")

        self.minimum_coefficient = float(minimum_coefficient)
        self.maximum_coefficient = float(maximum_coefficient)
        self.target = float(target)
        self.hard_limit = float(hard_limit)
        self.window = int(window)
        self.update_interval = int(update_interval)
        self.adaptation_factor = float(adaptation_factor)
        self.lower_ratio = float(lower_ratio)
        self.upper_ratio = float(upper_ratio)
        self.hard_limit_patience = int(hard_limit_patience)
        self.register_buffer(
            "coefficient", torch.tensor(float(initial_coefficient), dtype=torch.float64)
        )
        self.register_buffer(
            "rolling_values", torch.zeros(self.window, dtype=torch.float64)
        )
        self.register_buffer("rolling_index", torch.zeros((), dtype=torch.long))
        self.register_buffer("rolling_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("observation_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("consecutive_hard_violations", torch.zeros((), dtype=torch.long))
        self.register_buffer("rolling_mean", torch.zeros((), dtype=torch.float64))
        self.register_buffer("should_stop", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def update(self, kl_value: torch.Tensor) -> Dict[str, torch.Tensor]:
        value = torch.as_tensor(
            kl_value, device=self.rolling_values.device, dtype=torch.float64
        ).detach()
        if value.numel() != 1 or not torch.isfinite(value):
            raise FloatingPointError("adaptive KL observation must be finite and scalar")
        index = int(self.rolling_index.item())
        self.rolling_values[index] = value
        self.rolling_index.fill_((index + 1) % self.window)
        self.rolling_count.add_(1).clamp_(max=self.window)
        self.observation_count.add_(1)
        count = int(self.rolling_count.item())
        self.rolling_mean.copy_(self.rolling_values[:count].mean())

        checked = (
            count == self.window
            and int(self.observation_count.item()) % self.update_interval == 0
        )
        if checked:
            rolling = float(self.rolling_mean.item())
            coefficient = float(self.coefficient.item())
            if rolling > self.target * self.upper_ratio:
                coefficient = min(
                    self.maximum_coefficient,
                    coefficient * self.adaptation_factor,
                )
            elif rolling < self.target * self.lower_ratio:
                coefficient = max(
                    self.minimum_coefficient,
                    coefficient / self.adaptation_factor,
                )
            self.coefficient.fill_(coefficient)
            if rolling > self.hard_limit:
                self.consecutive_hard_violations.add_(1)
            else:
                self.consecutive_hard_violations.zero_()
            if int(self.consecutive_hard_violations.item()) >= self.hard_limit_patience:
                self.should_stop.fill_(True)

        return {
            "generation_kl_coefficient": self.coefficient.detach().float().clone(),
            "generation_kl_rolling_mean": self.rolling_mean.detach().float().clone(),
            "generation_kl_controller_checked": torch.tensor(
                float(checked), device=self.coefficient.device
            ),
            "generation_kl_hard_violation_count": (
                self.consecutive_hard_violations.detach().float().clone()
            ),
            "generation_kl_should_stop": self.should_stop.detach().float().clone(),
        }


def file_sha256(path: Path) -> str:
    """Return the SHA256 of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_generation_trust_calibration(path: str) -> Dict:
    """Load and structurally validate an immutable Phase-5 calibration artifact."""
    calibration_path = Path(path)
    if not calibration_path.is_file():
        raise FileNotFoundError(
            f"Generation trust calibration does not exist: {calibration_path}"
        )
    if calibration_path.stat().st_mode & 0o222:
        raise PermissionError(
            "Generation trust calibration must be read-only (mode 0444)"
        )
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("generation trust calibration schema_version must be 1")
    if payload.get("formula_version") != TRUST_PROJECTION_FORMULA_VERSION:
        raise ValueError(
            "generation trust calibration formula_version mismatch: "
            f"{payload.get('formula_version')!r}"
        )
    if payload.get("mode") != "reference_mean_ball":
        raise ValueError("generation trust calibration mode must be reference_mean_ball")
    if payload.get("seed") != 20260719:
        raise ValueError("generation trust calibration seed must be 20260719")
    if payload.get("percentile") != 0.99:
        raise ValueError("generation trust calibration percentile must be 0.99")
    steps = payload.get("steps")
    if not isinstance(steps, dict) or set(steps) != {"transition", "final"}:
        raise ValueError(
            "generation trust calibration must contain transition and final steps"
        )
    for step_name, step in steps.items():
        if not isinstance(step, dict):
            raise ValueError(f"invalid {step_name} calibration entry")
        radius = step.get("radius")
        sigma = step.get("sigma")
        if (
            not isinstance(radius, (int, float))
            or not math.isfinite(radius)
            or radius < 0
        ):
            raise ValueError(f"invalid {step_name} trust radius")
        if (
            not isinstance(sigma, (int, float))
            or not math.isfinite(sigma)
            or sigma <= 0
        ):
            raise ValueError(f"invalid {step_name} calibration sigma")
        if int(step.get("count", 0)) <= 0:
            raise ValueError(f"invalid {step_name} calibration count")
        quantiles = step.get("quantiles")
        if not isinstance(quantiles, dict) or "p99" not in quantiles:
            raise ValueError(f"missing {step_name} calibration quantiles")
        if not math.isclose(float(radius), float(quantiles["p99"]), abs_tol=1e-12):
            raise ValueError(f"{step_name} radius is not the registered P99")
    return payload


def validate_generation_trust_provenance(
    calibration: Dict,
    reference_checkpoint_path: str,
    roll_timesteps: Tuple[int, ...],
    scheduler_num_inference_steps: int,
    transition_sigma: float,
    final_sigma: float,
) -> None:
    """Fail closed when runtime reference/schedule differs from calibration."""
    reference_path = Path(reference_checkpoint_path)
    base = calibration.get("base_checkpoint", {})
    expected_sha = base.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("calibration is missing base checkpoint SHA256")
    actual_sha = file_sha256(reference_path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            "Frozen base checkpoint SHA256 does not match trust calibration: "
            f"expected={expected_sha}, actual={actual_sha}"
        )
    schedule = calibration.get("schedule", {})
    if tuple(schedule.get("roll_timesteps", ())) != tuple(roll_timesteps):
        raise RuntimeError("Trust calibration roll_timesteps mismatch")
    if int(schedule.get("scheduler_num_inference_steps", -1)) != int(
        scheduler_num_inference_steps
    ):
        raise RuntimeError("Trust calibration scheduler inference-step mismatch")
    runtime_sigmas = {"transition": transition_sigma, "final": final_sigma}
    for step_name, runtime_sigma in runtime_sigmas.items():
        calibrated_sigma = float(calibration["steps"][step_name]["sigma"])
        if not math.isclose(
            float(runtime_sigma), calibrated_sigma, rel_tol=1e-6, abs_tol=1e-8
        ):
            raise RuntimeError(
                f"Trust calibration {step_name} sigma mismatch: "
                f"calibrated={calibrated_sigma}, runtime={runtime_sigma}"
            )


def normalized_rms_displacement(
    mean: torch.Tensor,
    reference_mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Return one normalized RMS displacement for each [batch, anchor]."""
    if mean.shape != reference_mean.shape:
        raise ValueError(
            "current and reference means must have identical shapes; "
            f"got {tuple(mean.shape)} and {tuple(reference_mean.shape)}"
        )
    if mean.ndim < 3:
        raise ValueError("denoising means must have shape [batch, anchor, ...]")
    if not torch.isfinite(mean).all() or not torch.isfinite(reference_mean).all():
        raise FloatingPointError("non-finite current/reference denoising mean")
    std = torch.as_tensor(std, device=mean.device, dtype=mean.dtype).detach()
    if std.numel() != 1 or not torch.isfinite(std).all() or torch.any(std <= 0):
        raise ValueError("trust projection sigma must be one finite positive scalar")
    reference_mean = reference_mean.detach()
    normalized_delta = ((mean - reference_mean) / std).flatten(start_dim=2)
    # This is exactly sqrt(mean(z^2)), but unlike an explicit sqrt at zero,
    # vector_norm has a finite (zero) gradient when current == reference.
    return torch.linalg.vector_norm(normalized_delta, dim=-1) / math.sqrt(
        normalized_delta.shape[-1]
    )


def project_reference_mean_ball(
    mean: torch.Tensor,
    reference_mean: torch.Tensor,
    std: torch.Tensor,
    radius: float,
    eps: float = 1e-8,
    post_tolerance: float = TRUST_PROJECTION_POST_TOLERANCE,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Project each batch/anchor mean onto its frozen-reference RMS ball."""
    if not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius < 0:
        raise ValueError("trust projection radius must be finite and non-negative")
    reference_mean = reference_mean.detach()
    pre_distance = normalized_rms_displacement(mean, reference_mean, std)
    radius_tensor = pre_distance.new_tensor(float(radius)).detach()
    alpha = torch.minimum(
        torch.ones_like(pre_distance), radius_tensor / (pre_distance + eps)
    )
    alpha_view = alpha.view(*alpha.shape, *([1] * (mean.ndim - 2)))
    projected = reference_mean + alpha_view * (mean - reference_mean)
    post_distance = normalized_rms_displacement(projected, reference_mean, std)
    if not torch.isfinite(projected).all() or not torch.isfinite(post_distance).all():
        raise FloatingPointError("non-finite post-projection denoising mean")
    excess = post_distance.detach() - radius_tensor
    if torch.any(excess > post_tolerance):
        raise RuntimeError(
            "Hard trust projection post-check failed: "
            f"max_distance={float(post_distance.detach().max())}, radius={radius}"
        )
    diagnostics = {
        "pre_distance": pre_distance.detach(),
        "post_distance": post_distance.detach(),
        "alpha": alpha.detach(),
        "projected": (pre_distance.detach() > radius_tensor),
        "reference_coverage": torch.ones_like(pre_distance, dtype=torch.bool),
    }
    return projected, diagnostics


def summarize_trust_projection(
    pre_distance: torch.Tensor,
    post_distance: torch.Tensor,
    alpha: torch.Tensor,
    projected: torch.Tensor,
    reference_coverage: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Build detached per-step logging scalars from [batch, anchor, step] data."""
    tensors = (post_distance, alpha, projected, reference_coverage)
    if pre_distance.ndim != 3 or any(value.shape != pre_distance.shape for value in tensors):
        raise ValueError("trust diagnostics must share shape [batch, anchor, step]")
    result = {}
    for step_index, step_name in enumerate(("transition", "final")):
        pre = pre_distance[..., step_index].detach().float().reshape(-1)
        post = post_distance[..., step_index].detach().float().reshape(-1)
        step_alpha = alpha[..., step_index].detach().float().reshape(-1)
        step_projected = projected[..., step_index].detach().float().reshape(-1)
        coverage = reference_coverage[..., step_index].detach().float().reshape(-1)
        for metric_name, value in (
            ("pre_distance_mean", pre.mean()),
            ("pre_distance_p90", torch.quantile(pre, 0.90)),
            ("pre_distance_p99", torch.quantile(pre, 0.99)),
            ("pre_distance_max", pre.max()),
            ("post_distance_mean", post.mean()),
            ("post_distance_p90", torch.quantile(post, 0.90)),
            ("post_distance_p99", torch.quantile(post, 0.99)),
            ("post_distance_max", post.max()),
            ("projection_fraction", step_projected.mean()),
            ("alpha_mean", step_alpha.mean()),
            ("alpha_min", step_alpha.min()),
            ("reference_coverage", coverage.mean()),
        ):
            result[f"generation_trust_{step_name}_{metric_name}"] = value.detach()
    return result


def flatten_generation_rollouts(
    tensor: torch.Tensor,
    batch_size: int,
    rollouts_per_mode: int,
) -> torch.Tensor:
    """Flatten [batch * rollout, mode, ...] in anchor-major order.

    The output order is mode0/rollout0, mode0/rollout1, and so on, so the
    generation objective can recover within-anchor groups by a reshape.
    """
    if batch_size <= 0 or rollouts_per_mode <= 0:
        raise ValueError("batch_size and rollouts_per_mode must be positive")
    if tensor.ndim < 2 or tensor.shape[0] != batch_size * rollouts_per_mode:
        raise ValueError(
            "rollout tensor must have leading shape "
            "[batch_size * rollouts_per_mode, mode]"
        )
    if rollouts_per_mode == 1:
        return tensor
    num_modes = tensor.shape[1]
    trailing_shape = tensor.shape[2:]
    return (
        tensor.reshape(batch_size, rollouts_per_mode, num_modes, *trailing_shape)
        .transpose(1, 2)
        .reshape(batch_size, num_modes * rollouts_per_mode, *trailing_shape)
    )


def _compute_pdm_secondary_score(
    aggregate_rewards: torch.Tensor,
    component_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    weighted_metric_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if aggregate_rewards.ndim != 2 or valid_mask.shape != aggregate_rewards.shape:
        raise ValueError(
            "aggregate_rewards and valid_mask must have shape [batch, num_modes]"
        )
    if component_scores.shape != (*aggregate_rewards.shape, 6):
        raise ValueError(
            "component_scores must have shape [batch, num_modes, 6]"
        )
    if weighted_metric_weights.numel() != 4:
        raise ValueError("weighted_metric_weights must contain four values")

    valid_mask = (
        valid_mask.bool()
        & torch.isfinite(aggregate_rewards)
        & torch.isfinite(component_scores).all(dim=-1)
    )
    components = component_scores.float().clamp(0.0, 1.0)
    safety_score = components[..., :2].mean(dim=-1)
    metric_weights = weighted_metric_weights.to(
        device=components.device, dtype=components.dtype
    ).clamp_min(0.0)
    weight_sum = metric_weights.sum()
    if weight_sum <= 0:
        raise ValueError("weighted_metric_weights must have a positive sum")
    weighted_score = (
        components[..., 2:] * metric_weights.view(1, 1, -1)
    ).sum(dim=-1) / weight_sum
    secondary_score = 0.5 * safety_score + 0.5 * weighted_score
    return valid_mask, components, secondary_score


def compute_pdm_tiebreak_rewards(
    aggregate_rewards: torch.Tensor,
    component_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    weighted_metric_weights: torch.Tensor,
    max_epsilon: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add an order-preserving PDM-component tie-break to aggregate rewards.

    Component scores store no-collision, drivable-area, progress, TTC,
    comfort, and driving-direction values. The epsilon for each scene is at
    most one quarter of the smallest positive aggregate-reward gap, so the
    secondary score cannot reverse the aggregate PDMS ordering.
    """
    if max_epsilon <= 0:
        raise ValueError("max_epsilon must be positive")

    valid_mask, _, secondary_score = _compute_pdm_secondary_score(
        aggregate_rewards,
        component_scores,
        valid_mask,
        weighted_metric_weights,
    )

    epsilons = aggregate_rewards.new_zeros((aggregate_rewards.shape[0],))
    for batch_idx in range(aggregate_rewards.shape[0]):
        values = aggregate_rewards[batch_idx][valid_mask[batch_idx]].float()
        if values.numel() < 2:
            epsilon = max_epsilon
        else:
            distinct = torch.unique(values).sort().values
            gaps = distinct[1:] - distinct[:-1]
            positive_gaps = gaps[gaps > 0]
            epsilon = (
                min(max_epsilon, float(positive_gaps.min()) / 4.0)
                if positive_gaps.numel() > 0
                else max_epsilon
            )
        epsilons[batch_idx] = epsilon

    shaped_rewards = aggregate_rewards + epsilons.unsqueeze(-1) * secondary_score
    shaped_rewards = torch.where(valid_mask, shaped_rewards, aggregate_rewards)
    return shaped_rewards, secondary_score, epsilons


def compute_pdm_dense_rewards(
    aggregate_rewards: torch.Tensor,
    component_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    weighted_metric_weights: torch.Tensor,
    dense_weight: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add safety-gated PDM component shaping to aggregate PDMS.

    Unlike the conservative tie-break mode, this reward may reorder valid
    candidates. The collision and drivable-area product gates the dense term,
    so a candidate that fails either multiplicative safety gate receives no
    component bonus.
    """
    if dense_weight < 0:
        raise ValueError("dense_weight must be non-negative")
    valid_mask, components, secondary_score = _compute_pdm_secondary_score(
        aggregate_rewards,
        component_scores,
        valid_mask,
        weighted_metric_weights,
    )
    safety_gate = components[..., 0] * components[..., 1]
    shaped_rewards = (
        aggregate_rewards
        + float(dense_weight) * safety_gate * secondary_score
    )
    shaped_rewards = torch.where(valid_mask, shaped_rewards, aggregate_rewards)
    return shaped_rewards, secondary_score, safety_gate


def diagonal_gaussian_log_prob(
    value: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Return one log-probability per [batch, mode]."""
    std = torch.as_tensor(std, device=mean.device, dtype=mean.dtype)
    if torch.any(std <= 0):
        raise ValueError("Gaussian standard deviation must be positive")
    log_prob = -0.5 * (
        ((value - mean) / std).square() + 2.0 * std.log() + math.log(2.0 * math.pi)
    )
    return log_prob.flatten(start_dim=2).sum(dim=-1)


def diagonal_gaussian_log_prob_mean(
    value: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Return a dimension-normalized log probability per [batch, mode]."""
    std = torch.as_tensor(std, device=mean.device, dtype=mean.dtype)
    if torch.any(std <= 0):
        raise ValueError("Gaussian standard deviation must be positive")
    log_prob = -0.5 * (
        ((value - mean) / std).square() + 2.0 * std.log() + math.log(2.0 * math.pi)
    )
    return log_prob.flatten(start_dim=2).mean(dim=-1)


def diagonal_gaussian_kl_same_std(
    mean: torch.Tensor,
    reference_mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Dimension-normalized exact KL for Gaussians with the same variance."""
    std = torch.as_tensor(std, device=mean.device, dtype=mean.dtype)
    if torch.any(std <= 0):
        raise ValueError("Gaussian standard deviation must be positive")
    kl = 0.5 * ((mean - reference_mean) / std).square()
    return kl.flatten(start_dim=2).mean(dim=-1)


def ddim_transition_with_log_prob(
    scheduler,
    model_output: torch.Tensor,
    timestep: int,
    sample: torch.Tensor,
    eta: float,
    action: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
    sigma_min: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample/evaluate one stochastic DDIM transition.

    Returns `(action, log_prob, mean, std)`. The log-probability is summed over
    trajectory coordinates and retained per batch/mode.
    """
    if eta <= 0:
        raise ValueError("eta must be positive for a stochastic GRPO transition")
    stride = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    prev_timestep = int(timestep) - stride
    variance = scheduler._get_variance(int(timestep), prev_timestep)
    std = (eta * variance.sqrt()).clamp_min(sigma_min).to(
        device=model_output.device, dtype=model_output.dtype
    )
    mean = scheduler.step(
        model_output=model_output,
        timestep=int(timestep),
        sample=sample,
        eta=eta,
        variance_noise=torch.zeros_like(model_output),
    ).prev_sample
    if action is None:
        if noise is None:
            noise = torch.randn_like(mean)
        action = mean + std * noise
    log_prob = diagonal_gaussian_log_prob(action, mean, std)
    return action, log_prob, mean, std


def _decode_policy_step(
    head,
    policy,
    sample,
    timestep,
    ego_query,
    agents_query,
    bev_feature,
    bev_spatial_shape,
    status_encoding,
    global_img,
):
    from navsim.agents.diffusiondrive.modules.blocks import gen_sineembed_for_position

    batch_size = sample.shape[0]
    noisy_points = head.denorm_odo(sample.clamp(min=-1, max=1))
    position = gen_sineembed_for_position(noisy_points, hidden_dim=64).flatten(-2)
    trajectory_feature = head.plan_anchor_encoder(position)
    trajectory_feature = trajectory_feature.view(batch_size, noisy_points.shape[1], -1)
    time_embedding = head.time_mlp(
        torch.full((batch_size,), int(timestep), device=sample.device, dtype=torch.long)
    ).view(batch_size, 1, -1)
    regression, classification = policy(
        trajectory_feature,
        noisy_points,
        bev_feature,
        bev_spatial_shape,
        agents_query,
        ego_query,
        time_embedding,
        status_encoding,
        global_img,
    )
    return regression[-1], classification[-1]


@torch.no_grad()
def _sample_full_chain_actions(
    head,
    policy,
    initial_sample,
    ego_query,
    agents_query,
    bev_feature,
    bev_spatial_shape,
    status_encoding,
    global_img,
):
    """Sample and detach one complete Stage-16 denoising action chain."""
    timesteps = tuple(int(t) for t in head._roll_timesteps)
    if len(timesteps) < 2 or timesteps[-1] != 0:
        raise ValueError("full-chain DiffGRPO requires at least two steps ending at 0")
    scheduler = head.diffusion_scheduler
    scheduler.set_timesteps(head._scheduler_num_inference_steps, initial_sample.device)
    states = []
    actions = []
    sample = initial_sample.detach()
    for timestep in timesteps[:-1]:
        states.append(sample)
        regression, _ = _decode_policy_step(
            head, policy, sample, timestep, ego_query, agents_query,
            bev_feature, bev_spatial_shape, status_encoding, global_img,
        )
        action, _, _, _ = ddim_transition_with_log_prob(
            scheduler,
            head.norm_odo(regression[..., :2]),
            timestep,
            sample,
            eta=head._generation_ddim_eta,
            sigma_min=head._generation_sigma_min,
        )
        sample = action.detach()
        actions.append(sample)

    states.append(sample)
    final_regression, final_classification = _decode_policy_step(
        head, policy, sample, timesteps[-1], ego_query, agents_query,
        bev_feature, bev_spatial_shape, status_encoding, global_img,
    )
    final_mean = head.norm_odo(final_regression)
    final_std = final_mean.new_tensor(head._generation_final_std)
    final_action = (final_mean + final_std * torch.randn_like(final_mean)).detach()
    actions.append(final_action)
    return tuple(states), tuple(actions), final_action, final_classification.detach()


def _replay_full_chain_actions(
    head,
    policy,
    states,
    actions,
    ego_query,
    agents_query,
    bev_feature,
    bev_spatial_shape,
    status_encoding,
    global_img,
):
    """Evaluate a detached full chain under one policy with mean log-probability."""
    timesteps = tuple(int(t) for t in head._roll_timesteps)
    if len(states) != len(timesteps) or len(actions) != len(timesteps):
        raise ValueError("full-chain replay state/action count does not match schedule")
    scheduler = head.diffusion_scheduler
    scheduler.set_timesteps(head._scheduler_num_inference_steps, states[0].device)
    log_probs = []
    final_classification = None
    for index, timestep in enumerate(timesteps):
        regression, classification = _decode_policy_step(
            head, policy, states[index], timestep, ego_query, agents_query,
            bev_feature, bev_spatial_shape, status_encoding, global_img,
        )
        if index < len(timesteps) - 1:
            _, _, mean, std = ddim_transition_with_log_prob(
                scheduler,
                head.norm_odo(regression[..., :2]),
                timestep,
                states[index],
                eta=head._generation_ddim_eta,
                action=actions[index],
                sigma_min=head._generation_sigma_min,
            )
            log_prob = diagonal_gaussian_log_prob_mean(actions[index], mean, std)
        else:
            mean = head.norm_odo(regression)
            std = mean.new_tensor(head._generation_final_std)
            log_prob = diagonal_gaussian_log_prob_mean(actions[index], mean, std)
            final_classification = classification
        log_probs.append(log_prob)
    if final_classification is None:
        raise RuntimeError("full-chain replay produced no final classification logits")
    return torch.stack(log_probs, dim=-1), final_classification


def collect_full_chain_diffgrpo_trace(
    head,
    initial_sample,
    ego_query,
    agents_query,
    bev_feature,
    bev_spatial_shape,
    status_encoding,
    global_img,
):
    """Collect on-policy actions and frozen-reference teacher actions for Stage 16."""
    if head.ref_policy is None:
        raise RuntimeError("full-chain DiffGRPO requires a frozen reference policy")
    if head._generation_trust_projection_mode != "none":
        raise ValueError("full-chain DiffGRPO does not support trust projection")

    behavior_states, behavior_actions, final_action, _ = _sample_full_chain_actions(
        head, head.diff_decoder, initial_sample, ego_query, agents_query,
        bev_feature, bev_spatial_shape, status_encoding, global_img,
    )
    current_log_probs, current_cls = _replay_full_chain_actions(
        head, head.diff_decoder, behavior_states, behavior_actions,
        ego_query, agents_query, bev_feature, bev_spatial_shape,
        status_encoding, global_img,
    )
    with torch.no_grad():
        _, reference_cls = _replay_full_chain_actions(
            head, head.ref_policy, behavior_states, behavior_actions,
            ego_query, agents_query, bev_feature, bev_spatial_shape,
            status_encoding, global_img,
        )
        teacher_states, teacher_actions, _, _ = _sample_full_chain_actions(
            head, head.ref_policy, initial_sample, ego_query, agents_query,
            bev_feature, bev_spatial_shape, status_encoding, global_img,
        )
    bc_log_probs, _ = _replay_full_chain_actions(
        head, head.diff_decoder, teacher_states, teacher_actions,
        ego_query, agents_query, bev_feature, bev_spatial_shape,
        status_encoding, global_img,
    )
    if not torch.isfinite(current_log_probs).all() or not torch.isfinite(bc_log_probs).all():
        raise FloatingPointError("full-chain DiffGRPO produced non-finite log probabilities")
    return {
        "trajectories": head.denorm_odo(final_action),
        "current_log_probs": current_log_probs,
        "bc_log_probs": bc_log_probs,
        "current_cls": current_cls,
        "reference_cls": reference_cls,
        "num_denoising_steps": current_log_probs.new_tensor(
            float(current_log_probs.shape[-1])
        ),
    }


def collect_selected_anchor_diffgrpo_trace(
    head,
    selected_clean_sample,
    selected_modes,
    group_size,
    ego_query,
    agents_query,
    bev_feature,
    bev_spatial_shape,
    status_encoding,
    global_img,
):
    """Collect G same-anchor rollouts and one frozen-base BC teacher chain."""
    if head.ref_policy is None:
        raise RuntimeError("selected-anchor DiffGRPO requires a frozen reference policy")
    if selected_clean_sample.ndim != 4 or selected_clean_sample.shape[1] != 1:
        raise ValueError("selected clean sample must have shape [B,1,T,D]")
    if selected_modes.shape != (selected_clean_sample.shape[0],):
        raise ValueError("selected modes must have shape [B]")
    if int(group_size) != 8:
        raise ValueError("Stage19 selected-anchor group size must be 8")
    if head._generation_trust_projection_mode != "none":
        raise ValueError("selected-anchor DiffGRPO does not support trust projection")

    batch_size = selected_clean_sample.shape[0]
    scheduler = head.diffusion_scheduler
    timesteps = torch.full(
        (batch_size,),
        int(head._truncation_timestep),
        device=selected_clean_sample.device,
        dtype=torch.long,
    )
    clean_group = selected_clean_sample.expand(
        -1, int(group_size), -1, -1
    ).contiguous()
    behavior_initial = scheduler.add_noise(
        original_samples=clean_group,
        noise=torch.randn_like(clean_group),
        timesteps=timesteps,
    ).detach()
    behavior_states, behavior_actions, final_action, _ = (
        _sample_full_chain_actions(
            head, head.diff_decoder, behavior_initial, ego_query, agents_query,
            bev_feature, bev_spatial_shape, status_encoding, global_img,
        )
    )
    current_log_probs, current_cls = _replay_full_chain_actions(
        head, head.diff_decoder, behavior_states, behavior_actions,
        ego_query, agents_query, bev_feature, bev_spatial_shape,
        status_encoding, global_img,
    )
    with torch.no_grad():
        _, reference_cls = _replay_full_chain_actions(
            head, head.ref_policy, behavior_states, behavior_actions,
            ego_query, agents_query, bev_feature, bev_spatial_shape,
            status_encoding, global_img,
        )
        teacher_initial = scheduler.add_noise(
            original_samples=selected_clean_sample,
            noise=torch.randn_like(selected_clean_sample),
            timesteps=timesteps,
        ).detach()
        teacher_states, teacher_actions, _, _ = _sample_full_chain_actions(
            head, head.ref_policy, teacher_initial, ego_query, agents_query,
            bev_feature, bev_spatial_shape, status_encoding, global_img,
        )
    bc_log_probs, _ = _replay_full_chain_actions(
        head, head.diff_decoder, teacher_states, teacher_actions,
        ego_query, agents_query, bev_feature, bev_spatial_shape,
        status_encoding, global_img,
    )
    if current_log_probs.shape[:2] != (batch_size, int(group_size)):
        raise RuntimeError("selected-anchor current rollout group shape drifted")
    if bc_log_probs.shape[:2] != (batch_size, 1):
        raise RuntimeError("selected-anchor BC teacher must have one chain per scene")
    if not torch.isfinite(current_log_probs).all() or not torch.isfinite(
        bc_log_probs
    ).all():
        raise FloatingPointError(
            "selected-anchor DiffGRPO produced non-finite log probabilities"
        )
    return {
        "trajectories": head.denorm_odo(final_action),
        "current_log_probs": current_log_probs,
        "bc_log_probs": bc_log_probs,
        "current_cls": current_cls,
        "reference_cls": reference_cls,
        "selected_anchor_modes": selected_modes.detach(),
        "group_size": current_log_probs.new_tensor(float(group_size)),
        "num_denoising_steps": current_log_probs.new_tensor(
            float(current_log_probs.shape[-1])
        ),
    }


def _collect_generation_trace_legacy(
    head,
    initial_sample,
    ego_query,
    agents_query,
    bev_feature,
    bev_spatial_shape,
    status_encoding,
    global_img,
):
    """Collect old-policy actions and replay them under current/reference policies."""
    if len(head._roll_timesteps) != 2 or head._roll_timesteps[-1] != 0:
        raise ValueError("generation GRPO currently requires exactly two timesteps ending at 0")
    first_timestep, final_timestep = head._roll_timesteps
    scheduler = head.diffusion_scheduler
    scheduler.set_timesteps(head._scheduler_num_inference_steps, initial_sample.device)

    with torch.no_grad():
        old_first_reg, _ = _decode_policy_step(
            head, head.old_policy, initial_sample, first_timestep, ego_query, agents_query,
            bev_feature, bev_spatial_shape, status_encoding, global_img,
        )
        transition_action, old_transition_log_prob, _, transition_std = (
            ddim_transition_with_log_prob(
                scheduler,
                head.norm_odo(old_first_reg[..., :2]),
                first_timestep,
                initial_sample,
                eta=head._generation_ddim_eta,
                sigma_min=head._generation_sigma_min,
            )
        )
        old_final_reg, old_final_cls = _decode_policy_step(
            head, head.old_policy, transition_action, final_timestep, ego_query,
            agents_query, bev_feature, bev_spatial_shape, status_encoding, global_img,
        )
        old_final_mean = head.norm_odo(old_final_reg)
        final_std = torch.as_tensor(
            head._generation_final_std,
            device=old_final_mean.device,
            dtype=old_final_mean.dtype,
        )
        final_action = old_final_mean + final_std * torch.randn_like(old_final_mean)
        old_final_log_prob = diagonal_gaussian_log_prob(
            final_action, old_final_mean, final_std
        )

    def replay(policy):
        first_reg, _ = _decode_policy_step(
            head, policy, initial_sample, first_timestep, ego_query, agents_query,
            bev_feature, bev_spatial_shape, status_encoding, global_img,
        )
        _, transition_log_prob, transition_mean, _ = ddim_transition_with_log_prob(
            scheduler,
            head.norm_odo(first_reg[..., :2]),
            first_timestep,
            initial_sample,
            eta=head._generation_ddim_eta,
            action=transition_action,
            sigma_min=head._generation_sigma_min,
        )
        final_reg, final_cls = _decode_policy_step(
            head, policy, transition_action, final_timestep, ego_query, agents_query,
            bev_feature, bev_spatial_shape, status_encoding, global_img,
        )
        final_mean = head.norm_odo(final_reg)
        final_log_prob = diagonal_gaussian_log_prob(final_action, final_mean, final_std)
        return (
            torch.stack((transition_log_prob, final_log_prob), dim=-1),
            transition_mean,
            final_mean,
            final_cls,
        )

    current_log_probs, current_transition_mean, current_final_mean, current_cls = replay(
        head.diff_decoder
    )
    with torch.no_grad():
        reference_log_probs, reference_transition_mean, reference_final_mean, reference_cls = replay(
            head.ref_policy
        )
    old_log_probs = torch.stack((old_transition_log_prob, old_final_log_prob), dim=-1)
    generation_kl = torch.stack(
        (
            diagonal_gaussian_kl_same_std(
                current_transition_mean, reference_transition_mean, transition_std
            ),
            diagonal_gaussian_kl_same_std(
                current_final_mean, reference_final_mean, final_std
            ),
        ),
        dim=-1,
    )
    return {
        "trajectories": head.denorm_odo(final_action),
        "current_log_probs": current_log_probs,
        "old_log_probs": old_log_probs,
        "reference_log_probs": reference_log_probs,
        "generation_kl": generation_kl,
        "current_cls": current_cls,
        "old_cls": old_final_cls,
        "reference_cls": reference_cls,
    }

def collect_generation_trace(
    head,
    initial_sample,
    ego_query,
    agents_query,
    bev_feature,
    bev_spatial_shape,
    status_encoding,
    global_img,
):
    """Collect behavior actions under the actual hard-projected policy."""
    if head._generation_trust_projection_mode == "none":
        return _collect_generation_trace_legacy(
            head,
            initial_sample,
            ego_query,
            agents_query,
            bev_feature,
            bev_spatial_shape,
            status_encoding,
            global_img,
        )
    if head._generation_trust_projection_mode != "reference_mean_ball":
        raise RuntimeError("unsupported generation trust projection mode")
    if head.ref_policy is None or head.old_policy is None:
        raise RuntimeError("Hard trust projection requires frozen base and old policies")
    if len(head._roll_timesteps) != 2 or head._roll_timesteps[-1] != 0:
        raise ValueError("generation GRPO requires transition and final timesteps")

    first_timestep, final_timestep = head._roll_timesteps
    scheduler = head.diffusion_scheduler
    scheduler.set_timesteps(head._scheduler_num_inference_steps, initial_sample.device)
    transition_radius = head._generation_trust_radii["transition"]
    final_radius = head._generation_trust_radii["final"]

    with torch.no_grad():
        reference_first_reg, _ = _decode_policy_step(
            head, head.ref_policy, initial_sample, first_timestep, ego_query,
            agents_query, bev_feature, bev_spatial_shape, status_encoding, global_img,
        )
        _, _, reference_transition_mean, transition_std = ddim_transition_with_log_prob(
            scheduler,
            head.norm_odo(reference_first_reg[..., :2]),
            first_timestep,
            initial_sample,
            eta=head._generation_ddim_eta,
            noise=torch.zeros_like(initial_sample),
            sigma_min=head._generation_sigma_min,
        )
        old_first_reg, _ = _decode_policy_step(
            head, head.old_policy, initial_sample, first_timestep, ego_query,
            agents_query, bev_feature, bev_spatial_shape, status_encoding, global_img,
        )
        _, _, old_transition_mean, _ = ddim_transition_with_log_prob(
            scheduler,
            head.norm_odo(old_first_reg[..., :2]),
            first_timestep,
            initial_sample,
            eta=head._generation_ddim_eta,
            noise=torch.zeros_like(initial_sample),
            sigma_min=head._generation_sigma_min,
        )
        projected_old_transition_mean, _ = project_reference_mean_ball(
            old_transition_mean,
            reference_transition_mean,
            transition_std,
            transition_radius,
        )
        transition_noise = torch.randn_like(projected_old_transition_mean)
        transition_action = (
            projected_old_transition_mean + transition_std * transition_noise
        ).detach()
        old_transition_log_prob = diagonal_gaussian_log_prob(
            transition_action, projected_old_transition_mean, transition_std
        )

        reference_final_reg, reference_cls = _decode_policy_step(
            head, head.ref_policy, transition_action, final_timestep, ego_query,
            agents_query, bev_feature, bev_spatial_shape, status_encoding, global_img,
        )
        reference_final_mean = head.norm_odo(reference_final_reg).detach()
        old_final_reg, old_final_cls = _decode_policy_step(
            head, head.old_policy, transition_action, final_timestep, ego_query,
            agents_query, bev_feature, bev_spatial_shape, status_encoding, global_img,
        )
        old_final_mean = head.norm_odo(old_final_reg)
        final_std = old_final_mean.new_tensor(head._generation_final_std).detach()
        projected_old_final_mean, _ = project_reference_mean_ball(
            old_final_mean, reference_final_mean, final_std, final_radius
        )
        final_noise = torch.randn_like(projected_old_final_mean)
        final_action = (
            projected_old_final_mean + final_std * final_noise
        ).detach()
        old_final_log_prob = diagonal_gaussian_log_prob(
            final_action, projected_old_final_mean, final_std
        )
        reference_log_probs = torch.stack(
            (
                diagonal_gaussian_log_prob(
                    transition_action, reference_transition_mean, transition_std
                ),
                diagonal_gaussian_log_prob(
                    final_action, reference_final_mean, final_std
                ),
            ),
            dim=-1,
        )

    current_first_reg, _ = _decode_policy_step(
        head, head.diff_decoder, initial_sample, first_timestep, ego_query,
        agents_query, bev_feature, bev_spatial_shape, status_encoding, global_img,
    )
    _, _, current_transition_mean, _ = ddim_transition_with_log_prob(
        scheduler,
        head.norm_odo(current_first_reg[..., :2]),
        first_timestep,
        initial_sample,
        eta=head._generation_ddim_eta,
        noise=torch.zeros_like(initial_sample),
        sigma_min=head._generation_sigma_min,
    )
    projected_current_transition_mean, transition_diagnostics = (
        project_reference_mean_ball(
            current_transition_mean,
            reference_transition_mean,
            transition_std,
            transition_radius,
        )
    )
    current_transition_log_prob = diagonal_gaussian_log_prob(
        transition_action, projected_current_transition_mean, transition_std
    )
    current_final_reg, current_cls = _decode_policy_step(
        head, head.diff_decoder, transition_action, final_timestep, ego_query,
        agents_query, bev_feature, bev_spatial_shape, status_encoding, global_img,
    )
    current_final_mean = head.norm_odo(current_final_reg)
    projected_current_final_mean, final_diagnostics = project_reference_mean_ball(
        current_final_mean, reference_final_mean, final_std, final_radius
    )
    current_final_log_prob = diagonal_gaussian_log_prob(
        final_action, projected_current_final_mean, final_std
    )

    current_log_probs = torch.stack(
        (current_transition_log_prob, current_final_log_prob), dim=-1
    )
    old_log_probs = torch.stack(
        (old_transition_log_prob, old_final_log_prob), dim=-1
    )
    generation_kl = torch.stack(
        (
            diagonal_gaussian_kl_same_std(
                projected_current_transition_mean,
                reference_transition_mean,
                transition_std,
            ),
            diagonal_gaussian_kl_same_std(
                projected_current_final_mean,
                reference_final_mean,
                final_std,
            ),
        ),
        dim=-1,
    )
    diagnostics = {}
    for key in (
        "pre_distance",
        "post_distance",
        "alpha",
        "projected",
        "reference_coverage",
    ):
        diagnostics[f"trust_{key}"] = torch.stack(
            (transition_diagnostics[key], final_diagnostics[key]), dim=-1
        )
    return {
        "trajectories": head.denorm_odo(final_action),
        "current_log_probs": current_log_probs,
        "old_log_probs": old_log_probs.detach(),
        "reference_log_probs": reference_log_probs.detach(),
        "generation_kl": generation_kl,
        "current_cls": current_cls,
        "old_cls": old_final_cls.detach(),
        "reference_cls": reference_cls.detach(),
        **diagnostics,
    }
