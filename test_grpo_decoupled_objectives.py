from dataclasses import replace

import pytest
import torch

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_loss import (
    compute_generation_grpo_objective,
    compute_grpo_objective,
    compute_pairwise_reward_ranking_objective,
    compute_reference_headroom_scene_weights,
    compute_smoothed_selector_policy,
    transfuser_loss,
)


def test_epsilon_zero_matches_softmax_when_all_modes_are_valid():
    logits = torch.tensor([[1.0, -0.5, 0.25]])
    valid = torch.ones_like(logits, dtype=torch.bool)

    probs, log_probs, has_valid = compute_smoothed_selector_policy(
        logits, valid, exploration_floor=0.0
    )

    torch.testing.assert_close(probs, torch.softmax(logits, dim=-1))
    torch.testing.assert_close(log_probs, torch.log_softmax(logits, dim=-1))
    assert has_valid.tolist() == [True]


def test_smoothed_policy_normalizes_only_valid_modes_and_preserves_argmax():
    logits = torch.tensor([[3.0, 100.0, 1.0]])
    valid = torch.tensor([[True, False, True]])

    probs, _, _ = compute_smoothed_selector_policy(
        logits, valid, exploration_floor=0.3
    )

    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(1))
    assert probs[0, 1] == 0
    assert probs.argmax(dim=-1).item() == logits.masked_fill(~valid, -torch.inf).argmax(
        dim=-1
    ).item()


def test_current_equals_old_has_unit_smoothed_ppo_ratio():
    logits = torch.tensor([[2.0, -1.0, 0.5]])
    rewards = torch.tensor([[0.0, 1.0, 0.4]])
    valid = torch.ones_like(logits, dtype=torch.bool)

    objective = compute_grpo_objective(
        logits,
        logits,
        logits,
        rewards,
        valid,
        exploration_floor=0.3,
    )

    torch.testing.assert_close(objective["ratio_mean"], torch.ones(()))
    torch.testing.assert_close(objective["kl_loss"], torch.zeros(()), atol=1e-7, rtol=0)


def test_floor_gives_low_probability_high_reward_mode_finite_gradient():
    current = torch.tensor([[10.0, -20.0]], requires_grad=True)
    old = current.detach().clone()
    rewards = torch.tensor([[0.0, 1.0]])
    valid = torch.ones_like(rewards, dtype=torch.bool)

    objective = compute_grpo_objective(
        current,
        old,
        old,
        rewards,
        valid,
        exploration_floor=0.1,
    )
    objective["policy_loss"].backward()

    assert torch.isfinite(current.grad).all()
    assert current.grad[0, 1].abs() > 0


def test_uniform_valid_behavior_strengthens_low_probability_reward_gradient():
    rewards = torch.tensor([[0.0, 1.0]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    gradients = {}
    for weighting in ("old_policy", "uniform_valid"):
        current = torch.tensor([[6.0, -6.0]], requires_grad=True)
        objective = compute_grpo_objective(
            current,
            current.detach(),
            current.detach(),
            rewards,
            valid,
            behavior_weighting=weighting,
        )
        objective["policy_loss"].backward()
        gradients[weighting] = current.grad[0, 1].abs().item()

    assert gradients["uniform_valid"] > gradients["old_policy"] * 100


def test_uniform_valid_behavior_masks_invalid_modes_and_has_unit_ratio():
    logits = torch.tensor([[2.0, -1.0, 0.5]], requires_grad=True)
    rewards = torch.tensor([[0.0, 100.0, 1.0]])
    valid = torch.tensor([[True, False, True]])
    objective = compute_grpo_objective(
        logits,
        logits.detach(),
        logits.detach(),
        rewards,
        valid,
        behavior_weighting="uniform_valid",
    )
    objective["policy_loss"].backward()

    torch.testing.assert_close(objective["ratio_mean"], torch.ones(()))
    assert logits.grad[0, 1] == 0
    assert torch.isfinite(logits.grad).all()


def test_selector_behavior_weighting_rejects_unknown_mode():
    logits = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="behavior_weighting"):
        compute_grpo_objective(
            logits,
            logits,
            logits,
            torch.tensor([[0.0, 1.0]]),
            torch.ones_like(logits, dtype=torch.bool),
            behavior_weighting="unknown",
        )


def test_pairwise_ranking_gradient_promotes_higher_reward_mode():
    logits = torch.zeros(1, 3, requires_grad=True)
    rewards = torch.tensor([[0.9, 0.4, 0.1]], requires_grad=True)
    objective = compute_pairwise_reward_ranking_objective(
        logits, rewards, torch.ones_like(rewards, dtype=torch.bool)
    )

    objective["rank_loss"].backward()

    assert logits.grad[0, 0] < 0
    assert logits.grad[0, 2] > 0
    assert rewards.grad is None
    assert objective["rank_active_scene_fraction"] == 1
    assert objective["rank_pair_count"] == 3


def test_pairwise_ranking_ignores_invalid_modes_ties_and_small_gaps():
    logits = torch.zeros(1, 4, requires_grad=True)
    rewards = torch.tensor([[0.50, 0.50, 0.495, 1.0]])
    valid = torch.tensor([[True, True, True, False]])
    objective = compute_pairwise_reward_ranking_objective(
        logits, rewards, valid, reward_gap=0.01
    )

    objective["rank_loss"].backward()

    torch.testing.assert_close(objective["rank_loss"], torch.zeros(()))
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))
    torch.testing.assert_close(objective["rank_active_scene_fraction"], torch.zeros(()))


@pytest.mark.parametrize(
    "valid",
    [torch.tensor([[False, False, False]]), torch.tensor([[False, True, False]])],
)
def test_pairwise_ranking_handles_all_invalid_and_single_valid(valid):
    logits = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    objective = compute_pairwise_reward_ranking_objective(
        logits, torch.tensor([[0.1, 0.5, 0.9]]), valid
    )

    objective["rank_loss"].backward()

    torch.testing.assert_close(objective["rank_loss"], torch.zeros(()))
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))


def test_pairwise_ranking_is_permutation_invariant():
    logits = torch.tensor([[0.2, -0.3, 1.1]])
    rewards = torch.tensor([[0.4, 0.9, 0.1]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    permutation = torch.tensor([2, 0, 1])

    original = compute_pairwise_reward_ranking_objective(logits, rewards, valid)
    permuted = compute_pairwise_reward_ranking_objective(
        logits[:, permutation], rewards[:, permutation], valid[:, permutation]
    )

    torch.testing.assert_close(original["rank_loss"], permuted["rank_loss"])
    torch.testing.assert_close(original["rank_pair_count"], permuted["rank_pair_count"])
    torch.testing.assert_close(
        original["rank_pair_reward_gap"], permuted["rank_pair_reward_gap"]
    )


def test_zero_scene_weight_removes_pairwise_ranking_gradient():
    logits = torch.zeros(1, 2, requires_grad=True)
    objective = compute_pairwise_reward_ranking_objective(
        logits,
        torch.tensor([[1.0, 0.0]]),
        torch.ones(1, 2, dtype=torch.bool),
        scene_weights=torch.zeros(1),
    )

    objective["rank_loss"].backward()

    torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))


def test_rank_weight_zero_preserves_existing_transfuser_objective():
    logits = torch.tensor([[0.0, 0.5, -0.5]], requires_grad=True)
    rewards = torch.tensor([[0.1, 0.8, 0.4]])
    predictions = {
        "final_poses_cls": logits,
        "final_old_poses_cls": logits.detach(),
        "final_ref_poses_cls": logits.detach(),
        "rewards": rewards,
        "raw_rewards": rewards,
        "reward_valid_mask": torch.ones_like(rewards, dtype=torch.bool),
        "grpo_training_rollout": False,
    }
    default_result = transfuser_loss({}, predictions, TransfuserConfig())
    explicit_result = transfuser_loss(
        {}, predictions, replace(TransfuserConfig(), selection_rank_loss_weight=0.0)
    )

    torch.testing.assert_close(default_result["loss"], explicit_result["loss"])
    torch.testing.assert_close(explicit_result["rank_loss"], torch.zeros(()))


def test_rank_only_total_loss_does_not_depend_on_old_policy_logits():
    logits = torch.tensor([[0.0, 0.5, -0.5]])
    rewards = torch.tensor([[0.1, 0.8, 0.4]])
    config = replace(
        TransfuserConfig(),
        grpo_training_mode="selector",
        policy_loss_weight=0.0,
        selection_rank_loss_weight=1.0,
        kl_loss_weight=0.01,
    )
    losses = []
    for old_logits in (logits, torch.tensor([[20.0, -20.0, 0.0]])):
        predictions = {
            "final_poses_cls": logits,
            "final_old_poses_cls": old_logits,
            "final_ref_poses_cls": logits,
            "rewards": rewards,
            "raw_rewards": rewards,
            "reward_valid_mask": torch.ones_like(rewards, dtype=torch.bool),
            "grpo_training_rollout": False,
        }
        losses.append(transfuser_loss({}, predictions, config)["loss"])

    torch.testing.assert_close(losses[0], losses[1])


def test_rank_validation_uses_pdms_rewards_when_raw_rewards_are_absent():
    logits = torch.tensor([[0.0, 0.5, -0.5]])
    rewards = torch.tensor([[0.1, 0.8, 0.4]])
    predictions = {
        "final_poses_cls": logits,
        "final_old_poses_cls": logits,
        "final_ref_poses_cls": logits,
        "rewards": rewards,
        "reward_valid_mask": torch.ones_like(rewards, dtype=torch.bool),
        "grpo_training_rollout": False,
    }
    config = replace(
        TransfuserConfig(),
        grpo_training_mode="selector",
        selection_rank_loss_weight=1.0,
        grpo_reward_mode="pdms",
    )

    result = transfuser_loss({}, predictions, config)

    assert result["rank_loss"] > 0
    assert torch.isfinite(result["loss"])


def test_rank_training_rejects_missing_raw_rewards():
    logits = torch.tensor([[0.0, 0.5, -0.5]])
    predictions = {
        "final_poses_cls": logits,
        "final_old_poses_cls": logits,
        "final_ref_poses_cls": logits,
        "rewards": torch.tensor([[0.1, 0.8, 0.4]]),
        "reward_valid_mask": torch.ones(1, 3, dtype=torch.bool),
        "grpo_training_rollout": True,
    }
    config = replace(
        TransfuserConfig(),
        grpo_training_mode="selector",
        selection_rank_loss_weight=1.0,
    )

    with pytest.raises(ValueError, match="raw_rewards during training"):
        transfuser_loss({}, predictions, config)


def test_smoothed_policy_handles_all_invalid_and_single_valid():
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    valid = torch.tensor([[False, False], [False, True]])

    probs, _, has_valid = compute_smoothed_selector_policy(
        logits, valid, exploration_floor=0.3
    )

    torch.testing.assert_close(probs[0], torch.zeros(2))
    torch.testing.assert_close(probs[1], torch.tensor([0.0, 1.0]))
    assert has_valid.tolist() == [False, True]


@pytest.mark.parametrize("epsilon", [-0.1, 1.0])
def test_smoothed_policy_rejects_invalid_epsilon(epsilon):
    logits = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="0 <= epsilon < 1"):
        compute_smoothed_selector_policy(
            logits,
            torch.ones_like(logits, dtype=torch.bool),
            exploration_floor=epsilon,
        )


def test_reference_headroom_gate_boundaries_and_detach():
    rewards = torch.tensor(
        [[0.4, 0.5], [0.1, 0.2], [0.7, 0.9]], requires_grad=True
    )
    reference = torch.tensor([0.5, 0.16, 0.8], requires_grad=True)
    valid = torch.ones_like(rewards, dtype=torch.bool)

    weights, headroom, scene_valid = compute_reference_headroom_scene_weights(
        rewards, valid, reference, margin=0.01, scale=0.05
    )

    torch.testing.assert_close(weights, torch.tensor([0.0, 0.6, 1.0]))
    torch.testing.assert_close(headroom, torch.tensor([0.0, 0.04, 0.1]))
    assert scene_valid.all()
    assert not weights.requires_grad
    assert not headroom.requires_grad


def test_zero_scene_weight_removes_policy_gradient_but_not_generation_kl():
    current = torch.zeros(1, 3, 2, requires_grad=True)
    old = torch.zeros_like(current)
    generation_kl = torch.ones(1, 3, 2, requires_grad=True)
    rewards = torch.tensor([[0.1, 0.5, 0.9]])
    valid = torch.ones_like(rewards, dtype=torch.bool)

    objective = compute_generation_grpo_objective(
        current,
        old,
        generation_kl,
        rewards,
        valid,
        scene_weights=torch.zeros(1),
    )
    (objective["policy_loss"] + objective["kl_loss"]).backward()

    torch.testing.assert_close(current.grad, torch.zeros_like(current.grad))
    assert generation_kl.grad is not None
    assert generation_kl.grad.abs().sum() > 0


def test_transfuser_gate_keeps_only_kl_without_reference_headroom():
    logits = torch.tensor([[0.0, 0.5, -0.5]], requires_grad=True)
    generation_log_probs = torch.zeros(1, 3, 2, requires_grad=True)
    generation_kl = torch.ones(1, 3, 2, requires_grad=True)
    rewards = torch.tensor([[0.1, 0.8, 0.4]])
    predictions = {
        "final_poses_cls": logits,
        "final_old_poses_cls": logits.detach(),
        "final_ref_poses_cls": logits.detach(),
        "rewards": rewards,
        "raw_rewards": rewards,
        "reward_valid_mask": torch.ones_like(rewards, dtype=torch.bool),
        "reference_selected_reward": torch.tensor([0.8]),
        "reference_reward_valid_mask": torch.tensor([True]),
        "generation_current_log_probs": generation_log_probs,
        "generation_old_log_probs": torch.zeros_like(generation_log_probs),
        "generation_kl": generation_kl,
    }
    config = replace(
        TransfuserConfig(),
        grpo_training_mode="generation",
        grpo_scene_weight_mode="reference_headroom",
        selector_consistency_kl_weight=0.0,
        generation_kl_loss_weight=0.1,
    )

    result = transfuser_loss({}, predictions, config)
    result["loss"].backward()

    torch.testing.assert_close(result["active_scene_fraction"], torch.zeros(()))
    torch.testing.assert_close(
        generation_log_probs.grad, torch.zeros_like(generation_log_probs.grad)
    )
    assert generation_kl.grad is not None and generation_kl.grad.abs().sum() > 0


def test_reference_gate_validation_falls_back_to_ungated_raw_metrics():
    logits = torch.tensor([[0.0, 0.5, -0.5]], requires_grad=True)
    rewards = torch.tensor([[0.1, 0.8, 0.4]])
    predictions = {
        "final_poses_cls": logits,
        "final_old_poses_cls": logits.detach(),
        "final_ref_poses_cls": logits.detach(),
        "rewards": rewards,
        "reward_valid_mask": torch.ones_like(rewards, dtype=torch.bool),
        "grpo_training_rollout": False,
    }
    config = replace(
        TransfuserConfig(),
        grpo_training_mode="selector",
        grpo_scene_weight_mode="reference_headroom",
    )

    result = transfuser_loss({}, predictions, config)

    assert torch.isfinite(result["loss"])
    torch.testing.assert_close(result["active_scene_fraction"], torch.ones(()))
