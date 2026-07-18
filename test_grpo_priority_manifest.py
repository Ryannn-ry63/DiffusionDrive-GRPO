"""Tests for navtrain hard-case manifest construction."""

import pytest

from scripts.evaluation.build_grpo_priority_manifest import (
    build_priority_records,
    priority_reasons,
)


def _record(
    token,
    selected,
    oracle,
    collision=1.0,
    drivable=1.0,
    ttc=1.0,
):
    return {
        "token": token,
        "selected_reward": selected,
        "oracle_reward": oracle,
        "selected_components": {
            "collision": collision,
            "drivable": drivable,
            "ttc": ttc,
        },
    }


def test_priority_reasons_cover_low_reward_safety_and_headroom():
    reasons = priority_reasons(
        _record("a", 0.2, 0.9, collision=0.0, ttc=0.0),
        low_reward_threshold=0.5,
        headroom_threshold=0.05,
        safety_pass_threshold=1.0 - 1e-9,
    )
    assert reasons == [
        "low_selected_reward",
        "collision_fail",
        "ttc_fail",
        "high_oracle_headroom",
    ]


def test_priority_thresholds_are_strict():
    reasons = priority_reasons(
        _record("a", 0.5, 0.55),
        low_reward_threshold=0.5,
        headroom_threshold=0.05,
        safety_pass_threshold=1.0 - 1e-9,
    )
    assert reasons == []


def test_build_priority_records_preserves_source_order():
    records = [
        _record("common", 0.9, 0.92),
        _record("low", 0.1, 0.2),
        _record("headroom", 0.8, 0.9),
    ]
    selected = build_priority_records(records)
    assert [record["token"] for record in selected] == ["low", "headroom"]


def test_missing_components_fail_closed():
    record = {
        "token": "bad",
        "selected_reward": 0.2,
        "oracle_reward": 0.9,
    }
    with pytest.raises(ValueError, match="missing selected_components"):
        priority_reasons(record, 0.5, 0.05, 1.0 - 1e-9)
