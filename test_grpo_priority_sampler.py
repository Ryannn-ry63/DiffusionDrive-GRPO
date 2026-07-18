"""Tests for exact priority/uniform GRPO batch composition."""

import json

import pytest

from navsim.planning.training.grpo_sampler import (
    BalancedPriorityBatchSampler,
    load_manifest_tokens,
)


def test_priority_sampler_makes_half_priority_half_uniform_slots():
    tokens = [f"token-{index}" for index in range(10)]
    priority = {"token-1", "token-4", "token-8"}
    sampler = BalancedPriorityBatchSampler(
        tokens, sorted(priority), batch_size=2, priority_fraction=0.5, seed=7
    )
    batches = list(sampler)
    assert len(batches) == 5
    assert sampler.common_indices == list(range(len(tokens)))
    for batch in batches:
        batch_tokens = [tokens[index] for index in batch]
        assert sum(token in priority for token in batch_tokens) >= 1


def test_priority_sampler_is_deterministic_per_fresh_instance():
    tokens = [f"token-{index}" for index in range(12)]
    priority = tokens[:4]
    first = list(BalancedPriorityBatchSampler(tokens, priority, 4, 0.5, seed=3))
    second = list(BalancedPriorityBatchSampler(tokens, priority, 4, 0.5, seed=3))
    assert first == second


def test_priority_sampler_rejects_missing_token():
    with pytest.raises(ValueError, match="absent"):
        BalancedPriorityBatchSampler(
            ["a", "b"], ["missing"], batch_size=2, priority_fraction=0.5
        )


def test_priority_sampler_requires_exact_batch_composition():
    with pytest.raises(ValueError, match="must be an integer"):
        BalancedPriorityBatchSampler(
            ["a", "b", "c"], ["a"], batch_size=3, priority_fraction=0.5
        )


def test_load_priority_manifest_records(tmp_path):
    path = tmp_path / "priority.json"
    path.write_text(
        json.dumps({"records": [{"token": "a"}, {"token": "b"}]}),
        encoding="utf-8",
    )
    assert load_manifest_tokens(path) == ["a", "b"]
