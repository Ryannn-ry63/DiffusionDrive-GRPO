"""Probability helpers for reward-guided truncated diffusion."""

import math
from typing import Optional, Tuple

import torch


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
