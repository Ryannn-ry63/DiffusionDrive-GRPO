"""Checks for order-preserving PDM-component reward shaping."""

import pytest
import torch

from navsim.agents.diffusiondrive.diffusion_grpo import (
    compute_pdm_dense_rewards,
    compute_pdm_tiebreak_rewards,
)


def _components(batch: int, modes: int) -> torch.Tensor:
    values = torch.linspace(0.0, 1.0, modes).view(1, modes, 1)
    return values.expand(batch, modes, 6).clone()


def test_tiebreak_differentiates_equal_aggregate_rewards():
    aggregate = torch.zeros(1, 3)
    components = _components(1, 3)
    valid = torch.ones_like(aggregate, dtype=torch.bool)
    shaped, secondary, epsilon = compute_pdm_tiebreak_rewards(
        aggregate,
        components,
        valid,
        torch.tensor([10.0, 5.0, 2.0, 0.0]),
    )

    assert shaped[0, 0] < shaped[0, 1] < shaped[0, 2]
    assert secondary[0, 0] < secondary[0, 1] < secondary[0, 2]
    torch.testing.assert_close(epsilon, torch.tensor([1e-3]))


def test_tiebreak_never_reverses_distinct_pdms_order():
    aggregate = torch.tensor([[0.0, 0.0, 0.2, 0.2005, 0.9]])
    components = _components(1, 5).flip(dims=(1,))
    valid = torch.ones_like(aggregate, dtype=torch.bool)
    shaped, _, epsilon = compute_pdm_tiebreak_rewards(
        aggregate,
        components,
        valid,
        torch.ones(4),
        max_epsilon=1e-3,
    )

    assert epsilon.item() <= (0.2005 - 0.2) / 4.0 + 1e-8
    original_order = torch.argsort(aggregate[0], stable=True)
    shaped_order = torch.argsort(shaped[0], stable=True)
    # Modes with different aggregate values preserve their relative order.
    for lhs in range(aggregate.shape[1]):
        for rhs in range(aggregate.shape[1]):
            if aggregate[0, lhs] < aggregate[0, rhs]:
                assert shaped[0, lhs] < shaped[0, rhs]
    assert set(original_order.tolist()) == set(shaped_order.tolist())


def test_invalid_modes_remain_invalid_and_do_not_set_epsilon():
    aggregate = torch.tensor([[0.1, float("nan"), 0.5]])
    components = _components(1, 3)
    valid = torch.tensor([[True, False, True]])
    shaped, _, epsilon = compute_pdm_tiebreak_rewards(
        aggregate,
        components,
        valid,
        torch.ones(4),
    )

    assert torch.isnan(shaped[0, 1])
    torch.testing.assert_close(epsilon, torch.tensor([1e-3]))


def test_tiebreak_rejects_invalid_weights():
    aggregate = torch.zeros(1, 2)
    components = _components(1, 2)
    valid = torch.ones_like(aggregate, dtype=torch.bool)
    with pytest.raises(ValueError, match="positive sum"):
        compute_pdm_tiebreak_rewards(
            aggregate,
            components,
            valid,
            torch.zeros(4),
        )


def test_dense_reward_is_safety_gated_and_can_reorder_safe_modes():
    aggregate = torch.tensor([[0.10, 0.09, 0.0, 0.0]])
    components = torch.tensor(
        [[
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        ]]
    )
    valid = torch.ones_like(aggregate, dtype=torch.bool)
    shaped, secondary, gate = compute_pdm_dense_rewards(
        aggregate,
        components,
        valid,
        torch.tensor([10.0, 5.0, 2.0, 0.0]),
        dense_weight=0.1,
    )

    assert shaped[0, 1] > shaped[0, 0]
    torch.testing.assert_close(shaped[0, 2:], aggregate[0, 2:])
    torch.testing.assert_close(gate, torch.tensor([[1.0, 1.0, 0.0, 0.0]]))
    assert torch.isfinite(secondary).all()


def test_dense_reward_rejects_negative_weight():
    with pytest.raises(ValueError, match="non-negative"):
        compute_pdm_dense_rewards(
            torch.zeros(1, 2),
            _components(1, 2),
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones(4),
            dense_weight=-0.1,
        )
