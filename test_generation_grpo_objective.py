"""Checks for the denoising-action GRPO objective."""

import torch

from navsim.agents.diffusiondrive.transfuser_loss import (
    compute_generation_grpo_objective,
)
from navsim.agents.diffusiondrive.diffusion_grpo import (
    flatten_generation_rollouts,
)


def test_generation_objective_backpropagates_to_log_probs():
    current = torch.zeros(1, 3, 2, requires_grad=True)
    old = torch.zeros_like(current)
    generation_kl = torch.zeros_like(current)
    rewards = torch.tensor([[0.1, 0.5, 0.9]])
    valid = torch.ones_like(rewards, dtype=torch.bool)

    objective = compute_generation_grpo_objective(
        current, old, generation_kl, rewards, valid
    )
    objective["policy_loss"].backward()

    assert torch.isfinite(objective["policy_loss"])
    assert current.grad is not None
    assert current.grad.abs().sum() > 0


def test_generation_objective_rejects_wrong_shape():
    current = torch.zeros(1, 3)
    rewards = torch.zeros(1, 3)
    valid = torch.ones_like(rewards, dtype=torch.bool)
    try:
        compute_generation_grpo_objective(current, current, current, rewards, valid)
    except ValueError as error:
        assert "shape [batch, mode, step]" in str(error)
    else:
        raise AssertionError("invalid generation log-prob shape was accepted")


def test_flatten_generation_rollouts_is_anchor_major():
    tensor = torch.tensor(
        [
            [[0.0], [10.0]],
            [[1.0], [11.0]],
            [[100.0], [110.0]],
            [[101.0], [111.0]],
        ]
    )
    flattened = flatten_generation_rollouts(
        tensor, batch_size=2, rollouts_per_mode=2
    )
    torch.testing.assert_close(
        flattened[..., 0],
        torch.tensor([[0.0, 1.0, 10.0, 11.0], [100.0, 101.0, 110.0, 111.0]]),
    )


def test_within_anchor_k2_has_opposite_rollout_gradients():
    current = torch.zeros(1, 4, 2, requires_grad=True)
    rewards = torch.tensor([[0.2, 0.8, 0.7, 0.4]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    objective = compute_generation_grpo_objective(
        current,
        torch.zeros_like(current),
        torch.zeros_like(current),
        rewards,
        valid,
        advantage_mode="within_anchor",
        rollouts_per_mode=2,
    )
    objective["policy_loss"].backward()

    assert current.grad[0, 0].mean() > 0
    assert current.grad[0, 1].mean() < 0
    assert current.grad[0, 2].mean() < 0
    assert current.grad[0, 3].mean() > 0
    torch.testing.assert_close(
        objective["within_anchor_pair_fraction"], torch.ones(())
    )


def test_hierarchical_k2_combines_within_and_reference_signals():
    current = torch.zeros(1, 4, 2, requires_grad=True)
    rewards = torch.tensor([[0.4, 0.8, 0.7, 0.6]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    objective = compute_generation_grpo_objective(
        current,
        torch.zeros_like(current),
        torch.zeros_like(current),
        rewards,
        valid,
        advantage_mode="hierarchical",
        reference_selected_reward=torch.tensor([0.65]),
        reference_margin=0.0,
        reference_scale=0.10,
        reference_clip=2.0,
        rollouts_per_mode=2,
    )
    objective["policy_loss"].backward()

    assert current.grad[0, 0].mean() > 0
    assert current.grad[0, 1].mean() < 0
    assert current.grad[0, 2].mean() < 0
    assert current.grad[0, 3].mean() > 0
    torch.testing.assert_close(
        objective["reference_delta_mean"], torch.tensor(-0.025)
    )


def test_within_anchor_rejects_k1():
    current = torch.zeros(1, 2, 2)
    rewards = torch.tensor([[0.2, 0.8]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    try:
        compute_generation_grpo_objective(
            current,
            current,
            current,
            rewards,
            valid,
            advantage_mode="within_anchor",
            rollouts_per_mode=1,
        )
    except ValueError as error:
        assert "rollouts_per_mode=2" in str(error)
    else:
        raise AssertionError("within_anchor accepted K=1")


def test_generation_mode_weights_focus_policy_gradient():
    current = torch.zeros(1, 3, 2, requires_grad=True)
    old = torch.zeros_like(current)
    generation_kl = torch.zeros_like(current)
    rewards = torch.tensor([[0.1, 0.5, 0.9]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    mode_weights = torch.tensor([[0.0, 0.0, 1.0]])

    objective = compute_generation_grpo_objective(
        current,
        old,
        generation_kl,
        rewards,
        valid,
        mode_weights=mode_weights,
    )
    objective["policy_loss"].backward()

    torch.testing.assert_close(current.grad[:, :2], torch.zeros_like(current.grad[:, :2]))
    assert current.grad[:, 2].abs().sum() > 0


def test_generation_mode_weights_reject_negative_values():
    current = torch.zeros(1, 2, 2)
    rewards = torch.tensor([[0.0, 1.0]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    try:
        compute_generation_grpo_objective(
            current,
            current,
            current,
            rewards,
            valid,
            mode_weights=torch.tensor([[1.0, -1.0]]),
        )
    except ValueError as error:
        assert "finite and non-negative" in str(error)
    else:
        raise AssertionError("negative generation mode weights were accepted")


def test_group_zscore_explicit_mode_preserves_default_objective():
    rewards = torch.tensor([[0.1, 0.5, 0.9]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    current_default = torch.tensor(
        [[[0.1, -0.2], [0.0, 0.3], [-0.1, 0.2]]], requires_grad=True
    )
    current_explicit = current_default.detach().clone().requires_grad_(True)
    old = torch.zeros_like(current_default)
    generation_kl = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2) / 10

    default = compute_generation_grpo_objective(
        current_default, old, generation_kl, rewards, valid
    )
    explicit = compute_generation_grpo_objective(
        current_explicit,
        old,
        generation_kl,
        rewards,
        valid,
        advantage_mode="group_zscore",
    )
    default["policy_loss"].backward()
    explicit["policy_loss"].backward()

    torch.testing.assert_close(default["policy_loss"], explicit["policy_loss"])
    torch.testing.assert_close(default["kl_loss"], explicit["kl_loss"])
    torch.testing.assert_close(current_default.grad, current_explicit.grad)


def test_reference_centered_advantage_has_absolute_gradient_direction():
    current = torch.zeros(1, 3, 2, requires_grad=True)
    old = torch.zeros_like(current)
    generation_kl = torch.zeros_like(current)
    rewards = torch.tensor([[0.4, 0.5, 0.8]])
    valid = torch.ones_like(rewards, dtype=torch.bool)

    objective = compute_generation_grpo_objective(
        current,
        old,
        generation_kl,
        rewards,
        valid,
        advantage_mode="reference_centered",
        reference_selected_reward=torch.tensor([0.5]),
        reference_margin=0.01,
        reference_scale=0.10,
        reference_clip=2.0,
    )
    objective["policy_loss"].backward()

    assert current.grad[:, 0].mean() > 0
    torch.testing.assert_close(current.grad[:, 1], torch.zeros_like(current.grad[:, 1]))
    assert current.grad[:, 2].mean() < 0
    torch.testing.assert_close(
        objective["positive_advantage_fraction"], torch.tensor(1.0 / 3.0)
    )
    torch.testing.assert_close(
        objective["negative_advantage_fraction"], torch.tensor(1.0 / 3.0)
    )
    torch.testing.assert_close(
        objective["within_margin_fraction"], torch.tensor(1.0 / 3.0)
    )


def test_reference_centered_all_worse_has_no_positive_reinforcement():
    current = torch.zeros(1, 3, 2, requires_grad=True)
    rewards = torch.tensor([[0.1, 0.2, 0.3]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    objective = compute_generation_grpo_objective(
        current,
        torch.zeros_like(current),
        torch.zeros_like(current),
        rewards,
        valid,
        advantage_mode="reference_centered",
        reference_selected_reward=torch.tensor([0.8]),
    )
    objective["policy_loss"].backward()

    torch.testing.assert_close(
        objective["positive_advantage_fraction"], torch.zeros(())
    )
    torch.testing.assert_close(
        objective["negative_advantage_fraction"], torch.ones(())
    )
    assert torch.all(current.grad > 0)


def test_reference_centered_margin_zeroes_policy_but_keeps_kl():
    current = torch.zeros(1, 3, 2, requires_grad=True)
    generation_kl = torch.ones_like(current, requires_grad=True)
    rewards = torch.tensor([[0.495, 0.5, 0.505]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    objective = compute_generation_grpo_objective(
        current,
        torch.zeros_like(current),
        generation_kl,
        rewards,
        valid,
        advantage_mode="reference_centered",
        reference_selected_reward=torch.tensor([0.5]),
        reference_margin=0.01,
    )
    (objective["policy_loss"] + objective["kl_loss"]).backward()

    torch.testing.assert_close(current.grad, torch.zeros_like(current.grad))
    assert generation_kl.grad is not None
    assert generation_kl.grad.abs().sum() > 0
    torch.testing.assert_close(
        objective["policy_active_scene_fraction"], torch.zeros(())
    )


def test_reference_centered_detaches_reward_and_reference():
    current = torch.zeros(1, 2, 1, requires_grad=True)
    rewards = torch.tensor([[0.2, 0.8]], requires_grad=True)
    reference = torch.tensor([0.5], requires_grad=True)
    valid = torch.ones_like(rewards, dtype=torch.bool)
    objective = compute_generation_grpo_objective(
        current,
        torch.zeros_like(current),
        torch.zeros_like(current),
        rewards,
        valid,
        advantage_mode="reference_centered",
        reference_selected_reward=reference,
    )
    objective["policy_loss"].backward()

    assert current.grad is not None and current.grad.abs().sum() > 0
    assert rewards.grad is None
    assert reference.grad is None
