"""Tests for the immutable GRPO dev select/confirm split."""

from scripts.evaluation.split_grpo_dev_manifest import (
    build_partition_payload,
    ordered_token_sha256,
    split_records,
)


def _records(count: int) -> list[dict]:
    return [
        {
            "token": f"token-{index:04d}",
            "log_name": f"log-{index % 7}",
            "split": "val",
        }
        for index in range(count)
    ]


def test_dev_split_is_disjoint_complete_and_deterministic():
    records = _records(40)
    select_a, confirm_a = split_records(records, confirm_count=10, seed=9)
    select_b, confirm_b = split_records(records, confirm_count=10, seed=9)

    assert select_a == select_b
    assert confirm_a == confirm_b
    assert len(select_a) == 30
    assert len(confirm_a) == 10
    select_tokens = {record["token"] for record in select_a}
    confirm_tokens = {record["token"] for record in confirm_a}
    assert not select_tokens & confirm_tokens
    assert select_tokens | confirm_tokens == {
        record["token"] for record in records
    }
    assert {record["dev_partition"] for record in select_a} == {"select"}
    assert {record["dev_partition"] for record in confirm_a} == {"confirm"}


def test_dev_split_changes_with_seed():
    records = _records(40)
    _, confirm_a = split_records(records, confirm_count=10, seed=9)
    _, confirm_b = split_records(records, confirm_count=10, seed=10)
    assert {record["token"] for record in confirm_a} != {
        record["token"] for record in confirm_b
    }


def test_partition_payload_records_source_identity(tmp_path):
    records = _records(12)
    source = {"records": records}
    select, _ = split_records(records, confirm_count=3, seed=9)
    payload = build_partition_payload(
        tmp_path / "source.json", source, select, "select", 9
    )

    assert payload["summary"]["num_tokens"] == 9
    assert payload["summary"]["source_num_tokens"] == 12
    assert payload["summary"]["source_ordered_token_sha256"] == (
        ordered_token_sha256([record["token"] for record in records])
    )
