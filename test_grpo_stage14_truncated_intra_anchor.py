import pytest
import torch

from navsim.agents.diffusiondrive.transfuser_loss import (
    compute_generation_grpo_objective,
)


def _objective(rewards, no_collision, valid=None):
    rewards = torch.tensor([rewards], dtype=torch.float32)
    current = torch.zeros((1, len(rewards[0]), 2), requires_grad=True)
    old = torch.zeros_like(current)
    kl = torch.zeros_like(current)
    if valid is None:
        valid = torch.ones_like(rewards, dtype=torch.bool)
    else:
        valid = torch.tensor([valid], dtype=torch.bool)
    collision = torch.tensor(
        [no_collision], dtype=torch.float32, requires_grad=True
    )
    result = compute_generation_grpo_objective(
        current_log_probs=current,
        old_log_probs=old,
        generation_kl=kl,
        rewards=rewards,
        valid_mask=valid,
        advantage_mode="collision_truncated_intra_anchor",
        rollouts_per_mode=2,
        no_collision_scores=collision,
    )
    return result, current, collision


def test_safe_positive_is_kept_and_safe_negative_is_zeroed():
    result, current, collision = _objective([0.2, 0.8], [1.0, 1.0])

    assert result["truncated_positive_fraction"].item() == pytest.approx(0.5)
    assert result["truncated_safe_zero_fraction"].item() == pytest.approx(0.5)
    assert result["collision_penalty_fraction"].item() == 0.0
    assert result["positive_advantage_fraction"].item() == pytest.approx(0.5)
    assert result["negative_advantage_fraction"].item() == 0.0

    result["policy_loss"].backward()
    assert torch.all(current.grad[0, 0] == 0)
    assert torch.all(current.grad[0, 1] < 0)
    assert collision.grad is None


@pytest.mark.parametrize("collision_score", [0.0, 0.5])
def test_collision_is_negative_even_when_locally_best(collision_score):
    result, current, _ = _objective(
        [0.8, 0.2], [collision_score, 1.0]
    )

    assert result["collision_penalty_fraction"].item() == pytest.approx(0.5)
    assert result["negative_advantage_fraction"].item() == pytest.approx(0.5)
    assert result["positive_advantage_fraction"].item() == 0.0
    assert result["optimized_candidate_count"].item() == 1.0

    result["policy_loss"].backward()
    assert torch.all(current.grad[0, 0] > 0)
    assert torch.all(current.grad[0, 1] == 0)


def test_safe_tie_has_zero_policy_gradient():
    result, current, _ = _objective([0.5, 0.5], [1.0, 1.0])

    assert result["optimized_candidate_count"].item() == 0.0
    assert result["policy_active_scene_fraction"].item() == 0.0
    result["policy_loss"].backward()
    assert torch.all(current.grad == 0)


def test_single_valid_safe_is_ignored_but_single_valid_collision_is_penalized():
    safe, safe_current, _ = _objective(
        [0.7, float("nan")], [1.0, float("nan")], [True, False]
    )
    assert safe["optimized_candidate_count"].item() == 0.0
    safe["policy_loss"].backward()
    assert torch.all(safe_current.grad == 0)

    collision, collision_current, _ = _objective(
        [0.7, float("nan")], [0.0, float("nan")], [True, False]
    )
    assert collision["optimized_candidate_count"].item() == 1.0
    collision["policy_loss"].backward()
    assert torch.all(collision_current.grad[0, 0] > 0)
    assert torch.all(collision_current.grad[0, 1] == 0)


def test_fail_closed_on_missing_bad_shape_or_nonfinite_valid_component():
    rewards = torch.tensor([[0.2, 0.8]])
    log_probs = torch.zeros((1, 2, 2))
    kwargs = dict(
        current_log_probs=log_probs,
        old_log_probs=log_probs,
        generation_kl=log_probs,
        rewards=rewards,
        valid_mask=torch.ones_like(rewards, dtype=torch.bool),
        advantage_mode="collision_truncated_intra_anchor",
        rollouts_per_mode=2,
    )
    with pytest.raises(ValueError, match="requires no_collision_scores"):
        compute_generation_grpo_objective(**kwargs)
    with pytest.raises(ValueError, match="must match"):
        compute_generation_grpo_objective(
            **kwargs, no_collision_scores=torch.ones((1, 1))
        )
    with pytest.raises(ValueError, match="must all be finite"):
        compute_generation_grpo_objective(
            **kwargs, no_collision_scores=torch.tensor([[1.0, float("nan")]])
        )


def test_requires_exactly_two_rollouts_per_anchor():
    rewards = torch.tensor([[0.2, 0.5, 0.8]])
    log_probs = torch.zeros((1, 3, 2))
    with pytest.raises(ValueError, match="rollouts_per_mode=2"):
        compute_generation_grpo_objective(
            current_log_probs=log_probs,
            old_log_probs=log_probs,
            generation_kl=log_probs,
            rewards=rewards,
            valid_mask=torch.ones_like(rewards, dtype=torch.bool),
            advantage_mode="collision_truncated_intra_anchor",
            rollouts_per_mode=3,
            no_collision_scores=torch.ones_like(rewards),
        )


def test_legacy_group_zscore_is_unchanged_by_optional_component_argument():
    rewards = torch.tensor([[0.1, 0.4, 0.9]])
    log_probs = torch.tensor(
        [[[0.1, -0.2], [0.0, 0.2], [-0.1, 0.3]]], dtype=torch.float32
    )
    common = dict(
        current_log_probs=log_probs,
        old_log_probs=torch.zeros_like(log_probs),
        generation_kl=torch.full_like(log_probs, 0.01),
        rewards=rewards,
        valid_mask=torch.ones_like(rewards, dtype=torch.bool),
        advantage_mode="group_zscore",
    )
    before = compute_generation_grpo_objective(**common)
    after = compute_generation_grpo_objective(
        **common, no_collision_scores=torch.zeros_like(rewards)
    )
    assert before.keys() == after.keys()
    for key in before:
        assert torch.equal(before[key], after[key]), key
