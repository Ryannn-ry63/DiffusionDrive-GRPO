#!/usr/bin/env python3
"""Freeze disjoint select/confirm partitions of an audited GRPO dev manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence


DEFAULT_SOURCE = Path("artifacts/grpo_stage0/dev4096_manifest.json")
DEFAULT_SELECT = Path("artifacts/grpo_stage0/dev_select3072_manifest.json")
DEFAULT_CONFIRM = Path("artifacts/grpo_stage0/dev_confirm1024_manifest.json")


def ordered_token_sha256(tokens: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()


def stable_digest(seed: int, token: str) -> str:
    return hashlib.sha256(
        f"{seed}:dev-confirm:{token}".encode("utf-8")
    ).hexdigest()


def split_records(
    records: list[dict], confirm_count: int, seed: int
) -> tuple[list[dict], list[dict]]:
    if confirm_count <= 0 or confirm_count >= len(records):
        raise ValueError("confirm_count must be between 1 and num_records - 1")
    tokens = [str(record["token"]) for record in records]
    if len(tokens) != len(set(tokens)):
        raise ValueError("source manifest contains duplicate tokens")

    confirm_tokens = set(
        sorted(tokens, key=lambda token: stable_digest(seed, token))[:confirm_count]
    )
    select_records = [
        {**record, "dev_partition": "select"}
        for record in records
        if record["token"] not in confirm_tokens
    ]
    confirm_records = [
        {**record, "dev_partition": "confirm"}
        for record in records
        if record["token"] in confirm_tokens
    ]
    return select_records, confirm_records


def build_partition_payload(
    source_path: Path,
    source_payload: dict,
    records: list[dict],
    partition: str,
    seed: int,
) -> dict:
    tokens = [str(record["token"]) for record in records]
    source_tokens = [
        str(record["token"]) for record in source_payload["records"]
    ]
    return {
        "schema_version": 1,
        "summary": {
            "partition": partition,
            "selection": "sha256_holdout",
            "seed": seed,
            "num_tokens": len(tokens),
            "ordered_token_sha256": ordered_token_sha256(tokens),
            "source_manifest": str(source_path.resolve()),
            "source_num_tokens": len(source_tokens),
            "source_ordered_token_sha256": ordered_token_sha256(source_tokens),
        },
        "records": records,
    }


def write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--select-output", type=Path, default=DEFAULT_SELECT)
    parser.add_argument("--confirm-output", type=Path, default=DEFAULT_CONFIRM)
    parser.add_argument("--confirm-count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_payload = json.loads(args.source.read_text(encoding="utf-8"))
    records = source_payload.get("records")
    if not isinstance(records, list):
        raise ValueError("source manifest must contain a records list")
    select_records, confirm_records = split_records(
        records, args.confirm_count, args.seed
    )
    if set(record["token"] for record in select_records) & set(
        record["token"] for record in confirm_records
    ):
        raise AssertionError("select and confirm partitions overlap")
    if len(select_records) + len(confirm_records) != len(records):
        raise AssertionError("partition sizes do not reconstruct the source")

    select_payload = build_partition_payload(
        args.source, source_payload, select_records, "select", args.seed
    )
    confirm_payload = build_partition_payload(
        args.source, source_payload, confirm_records, "confirm", args.seed
    )
    write_payload(args.select_output, select_payload)
    write_payload(args.confirm_output, confirm_payload)
    print(
        json.dumps(
            {
                "select": select_payload["summary"],
                "confirm": confirm_payload["summary"],
                "overlap": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
