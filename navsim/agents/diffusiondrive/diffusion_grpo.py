"""Probability helpers for reward-guided truncated diffusion."""

import math
from typing import Optional, Tuple

import torch


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
