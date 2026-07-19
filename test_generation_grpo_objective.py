"""Checks for the denoising-action GRPO objective."""

import torch

from navsim.agents.diffusiondrive.transfuser_loss import (
    compute_anchor_rloo_advantages,
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
    torch.testing.assert_close(objective["policy_loss"], torch.tensor(0.0625))


def test_anchor_hierarchical_k2_uses_matching_anchor_baselines():
    current = torch.zeros(1, 4, 2, requires_grad=True)
    rewards = torch.tensor([[0.4, 0.8, 0.7, 0.6]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    reference_anchors = torch.tensor([[0.5, 0.65]], requires_grad=True)
    objective = compute_generation_grpo_objective(
        current,
        torch.zeros_like(current),
        torch.zeros_like(current),
        rewards,
        valid,
        advantage_mode="anchor_hierarchical",
        reference_anchor_rewards=reference_anchors,
        reference_margin=0.0,
        reference_scale=0.10,
        reference_clip=2.0,
        rollouts_per_mode=2,
    )
    objective["policy_loss"].backward()

    # Anchor 0 compares both rollouts with 0.5; anchor 1 compares with 0.65.
    torch.testing.assert_close(
        objective["reference_delta_mean"], torch.tensor(0.05)
    )
    torch.testing.assert_close(
        objective["reference_anchor_reward_mean"], torch.tensor(0.575)
    )
    torch.testing.assert_close(
        objective["reference_anchor_valid_fraction"], torch.ones(())
    )
    torch.testing.assert_close(objective["policy_loss"], torch.tensor(-0.125))
    assert current.grad is not None and current.grad.abs().sum() > 0
    assert reference_anchors.grad is None


def test_anchor_hierarchical_k4_supports_invalid_reference_anchor():
    current = torch.zeros(1, 8, 2, requires_grad=True)
    rewards = torch.tensor([[0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    reference_anchors = torch.tensor([[0.5, 0.4]])
    reference_valid = torch.tensor([[True, False]])
    objective = compute_generation_grpo_objective(
        current,
        torch.zeros_like(current),
        torch.zeros_like(current),
        rewards,
        valid,
        advantage_mode="anchor_hierarchical",
        reference_anchor_rewards=reference_anchors,
        reference_anchor_valid_mask=reference_valid,
        reference_margin=0.0,
        reference_scale=0.10,
        reference_clip=2.0,
        rollouts_per_mode=4,
    )
    objective["policy_loss"].backward()

    torch.testing.assert_close(
        objective["reference_delta_mean"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        objective["reference_anchor_valid_fraction"], torch.tensor(0.5)
    )
    torch.testing.assert_close(
        objective["within_anchor_pair_fraction"], torch.ones(())
    )
    torch.testing.assert_close(
        objective["within_anchor_reward_gap_mean"], torch.tensor(0.6)
    )
    # The invalid reference anchor still receives its within-anchor GRPO signal.
    assert current.grad[0, 4:].abs().sum() > 0


def test_anchor_hierarchical_rejects_wrong_reference_shape_and_k():
    current = torch.zeros(1, 4, 2)
    rewards = torch.tensor([[0.2, 0.4, 0.6, 0.8]])
    valid = torch.ones_like(rewards, dtype=torch.bool)

    try:
        compute_generation_grpo_objective(
            current,
            current,
            current,
            rewards,
            valid,
            advantage_mode="anchor_hierarchical",
            reference_anchor_rewards=torch.zeros(1, 4),
            rollouts_per_mode=2,
        )
    except ValueError as error:
        assert "reference_anchor_rewards must have shape" in str(error)
    else:
        raise AssertionError("wrong anchor-reference shape was accepted")

    try:
        compute_generation_grpo_objective(
            current,
            current,
            current,
            rewards,
            valid,
            advantage_mode="anchor_hierarchical",
            reference_anchor_rewards=torch.zeros(1, 1),
            rollouts_per_mode=3,
        )
    except ValueError as error:
        assert "rollouts_per_mode=2 or 4" in str(error)
    else:
        raise AssertionError("anchor_hierarchical accepted K=3")


def test_flatten_generation_rollouts_k4_remains_anchor_major():
    tensor = torch.tensor(
        [
            [[0.0], [10.0]],
            [[1.0], [11.0]],
            [[2.0], [12.0]],
            [[3.0], [13.0]],
        ]
    )
    flattened = flatten_generation_rollouts(
        tensor, batch_size=1, rollouts_per_mode=4
    )
    torch.testing.assert_close(
        flattened[..., 0],
        torch.tensor([[0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]]),
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


def test_anchor_rloo_k2_matches_manual_gaps_and_detaches():
    rewards = torch.tensor([[0.20, 0.50, 0.80, 0.805]], requires_grad=True)
    valid = torch.ones_like(rewards, dtype=torch.bool)
    advantages, gaps, rloo_valid, group_valid, clean_valid = (
        compute_anchor_rloo_advantages(rewards, valid)
    )

    torch.testing.assert_close(
        gaps,
        torch.tensor([[-0.30, 0.30, -0.005, 0.005]]),
    )
    torch.testing.assert_close(
        advantages,
        torch.tensor([[-1.0, 1.0, 0.0, 0.0]]),
    )
    assert rloo_valid.all()
    assert group_valid.all()
    assert clean_valid.all()
    torch.testing.assert_close(group_valid, torch.tensor([[True, True]]))
    torch.testing.assert_close(clean_valid, valid)
    assert not advantages.requires_grad
    assert not gaps.requires_grad


def test_anchor_rloo_dead_zone_clip_invalid_single_and_tie_groups():
    rewards = torch.tensor([[0.50, 0.50, 0.20, 0.42, 0.70, float("nan")]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    advantages, gaps, rloo_valid, group_valid, clean_valid = (
        compute_anchor_rloo_advantages(rewards, valid)
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor([[0.0, 0.0, -1.0, 1.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(
        gaps,
        torch.tensor([[0.0, 0.0, -0.22, 0.22, 0.0, 0.0]]),
    )
    torch.testing.assert_close(group_valid, torch.tensor([[True, True, False]]))
    torch.testing.assert_close(
        rloo_valid,
        torch.tensor([[True, True, True, True, False, False]]),
    )
    torch.testing.assert_close(
        clean_valid,
        torch.tensor([[True, True, True, True, True, False]]),
    )

    boundary_rewards = torch.tensor([[0.0, 0.01, 0.0, 0.21]])
    boundary_advantages = compute_anchor_rloo_advantages(
        boundary_rewards, torch.ones_like(boundary_rewards, dtype=torch.bool)
    )[0]
    torch.testing.assert_close(boundary_advantages[:, :2], torch.zeros(1, 2))
    torch.testing.assert_close(boundary_advantages[:, 2:], torch.tensor([[-1.0, 1.0]]))


def test_anchor_rloo_combines_matching_anchor_terms_and_reports_metrics():
    current = torch.zeros(1, 4, 2, requires_grad=True)
    rewards = torch.tensor([[0.19, 0.41, 0.59, 0.81]], requires_grad=True)
    references = torch.tensor([[0.30, 0.70]], requires_grad=True)
    valid = torch.ones_like(rewards, dtype=torch.bool)
    objective = compute_generation_grpo_objective(
        current,
        torch.zeros_like(current),
        torch.zeros_like(current),
        rewards,
        valid,
        advantage_mode="anchor_rloo",
        reference_anchor_rewards=references,
        reference_margin=0.0,
        reference_scale=0.10,
        rloo_margin=0.0,
        rloo_scale=0.20,
        rollouts_per_mode=2,
    )
    objective["policy_loss"].backward()

    assert current.grad[0, 0].mean() > 0
    assert current.grad[0, 1].mean() < 0
    assert current.grad[0, 2].mean() > 0
    assert current.grad[0, 3].mean() < 0
    assert rewards.grad is None
    assert references.grad is None
    torch.testing.assert_close(objective["ratio_mean"], torch.ones(()))
    torch.testing.assert_close(objective["rloo_absolute_gap_mean"], torch.tensor(0.22))
    torch.testing.assert_close(objective["rloo_dead_zone_fraction"], torch.zeros(()))
    torch.testing.assert_close(objective["rloo_clip_fraction"], torch.ones(()))
    torch.testing.assert_close(objective["rloo_active_fraction"], torch.ones(()))
    torch.testing.assert_close(objective["reference_anchor_valid_fraction"], torch.ones(()))


def test_anchor_rloo_falls_back_per_signal_and_kl_covers_all_valid_modes():
    current = torch.zeros(2, 2, 1, requires_grad=True)
    generation_kl = torch.ones_like(current, requires_grad=True)
    rewards = torch.tensor([[0.20, 0.50], [0.80, 0.10]], requires_grad=True)
    valid = torch.tensor([[True, True], [True, False]])
    references = torch.tensor([[0.30], [0.50]], requires_grad=True)
    reference_valid = torch.tensor([[False], [True]])
    objective = compute_generation_grpo_objective(
        current,
        torch.zeros_like(current),
        generation_kl,
        rewards,
        valid,
        advantage_mode="anchor_rloo",
        reference_anchor_rewards=references,
        reference_anchor_valid_mask=reference_valid,
        rollouts_per_mode=2,
    )
    (objective["policy_loss"] + objective["kl_loss"]).backward()

    assert current.grad[0].abs().sum() > 0
    assert current.grad[1, 0].abs().sum() > 0
    torch.testing.assert_close(current.grad[1, 1], torch.zeros_like(current.grad[1, 1]))
    assert generation_kl.grad is not None
    assert generation_kl.grad[valid.unsqueeze(-1)].abs().sum() > 0
    assert rewards.grad is None
    assert references.grad is None
    torch.testing.assert_close(objective["within_anchor_pair_fraction"], torch.tensor(0.5))
    torch.testing.assert_close(objective["reference_anchor_valid_fraction"], torch.tensor(0.5))


def test_anchor_rloo_zero_advantage_keeps_kl_and_unrelated_branches_frozen():
    current = torch.zeros(1, 2, 2, requires_grad=True)
    generation_kl = torch.ones_like(current, requires_grad=True)
    classification = torch.ones((), requires_grad=True)
    perception = torch.ones((), requires_grad=True)
    rewards = torch.tensor([[0.50, 0.50]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    objective = compute_generation_grpo_objective(
        current,
        torch.zeros_like(current),
        generation_kl,
        rewards,
        valid,
        advantage_mode="anchor_rloo",
        reference_anchor_rewards=torch.tensor([[0.50]]),
        rollouts_per_mode=2,
    )
    (objective["policy_loss"] + objective["kl_loss"]).backward()

    torch.testing.assert_close(current.grad, torch.zeros_like(current.grad))
    assert generation_kl.grad is not None and generation_kl.grad.abs().sum() > 0
    assert classification.grad is None
    assert perception.grad is None
    torch.testing.assert_close(objective["policy_active_scene_fraction"], torch.zeros(()))
    torch.testing.assert_close(objective["rloo_dead_zone_fraction"], torch.ones(()))
