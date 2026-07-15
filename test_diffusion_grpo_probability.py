"""Numerical tests for stochastic diffusion-policy probabilities."""

import torch
from diffusers import DDIMScheduler

from navsim.agents.diffusiondrive.diffusion_grpo import (
    ddim_transition_with_log_prob,
    diagonal_gaussian_kl_same_std,
    diagonal_gaussian_log_prob,
)


def test_gaussian_log_prob_is_largest_at_mean():
    mean = torch.zeros(1, 2, 3, 2)
    std = torch.tensor(0.2)
    at_mean = diagonal_gaussian_log_prob(mean, mean, std)
    away = diagonal_gaussian_log_prob(torch.ones_like(mean), mean, std)
    assert torch.all(at_mean > away)


def test_gaussian_kl_is_zero_for_identical_means():
    mean = torch.randn(2, 3, 4, 2)
    kl = diagonal_gaussian_kl_same_std(mean, mean, torch.tensor(0.1))
    torch.testing.assert_close(kl, torch.zeros_like(kl))


def test_ddim_transition_can_replay_the_same_action():
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="sample",
    )
    scheduler.set_timesteps(125)
    sample = torch.randn(1, 2, 4, 2)
    model_output = torch.randn_like(sample)
    action, old_log_prob, _, _ = ddim_transition_with_log_prob(
        scheduler, model_output, timestep=8, sample=sample, eta=1.0
    )
    replay_action, replay_log_prob, _, _ = ddim_transition_with_log_prob(
        scheduler,
        model_output,
        timestep=8,
        sample=sample,
        eta=1.0,
        action=action,
    )
    torch.testing.assert_close(replay_action, action)
    torch.testing.assert_close(replay_log_prob, old_log_prob)
