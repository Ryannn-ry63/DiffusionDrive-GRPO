"""Tests for paired DiffusionDrive GRPO artifact comparison."""

import pytest

from scripts.evaluation.compare_grpo_artifacts import compare


def artifact(rewards, token_sha="sha"):
    records = []
    for index, reward in enumerate(rewards):
        records.append(
            {
                "token": f"token-{index}",
                "selected_reward": reward,
                "oracle_reward": reward + 0.1,
                "candidate_reward": reward - 0.1,
                "selected_components": {
                    "collision": 1.0,
                    "drivable": 1.0,
                    "ttc": 1.0,
                },
            }
        )
    return {
        "summary": {"token_set_sha256": token_sha},
        "records": records,
    }


def test_compare_reports_exact_paired_delta():
    result = compare(
        artifact([0.2, 0.8]),
        artifact([0.3, 0.9]),
        samples=100,
        seed=7,
    )
    assert result["selected_difference"] == pytest.approx(0.1)
    assert result["oracle_difference"] == pytest.approx(0.1)
    assert result["safety_pass_bucket"]["num_tokens"] == 2
    assert result["wins"] == 2


def test_compare_rejects_token_sha_mismatch():
    with pytest.raises(ValueError, match="SHA"):
        compare(
            artifact([0.2], token_sha="a"),
            artifact([0.3], token_sha="b"),
            samples=10,
            seed=7,
        )
