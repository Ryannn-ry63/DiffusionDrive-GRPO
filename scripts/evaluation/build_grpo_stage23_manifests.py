#!/usr/bin/env python3
"""Freeze the Stage23 selector fit/calibration manifest from Stage21 folds 0--3."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from navsim.agents.diffusiondrive.stage23_trajectory_selector import (
    deterministic_log_bucket,
)


def _ordered_sha(values) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold", type=Path, action="append",
        default=None,
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/grpo_stage23/manifests/selector_fit_manifest.json"),
    )
    args = parser.parse_args()
    if args.fold is None:
        args.fold = [
            Path(f"artifacts/grpo_stage21/manifests/fold{index}_manifest.json")
            for index in range(4)
        ]
    records = []
    token_seen = set()
    log_fold = {}
    sources = []
    for fold_index, path in enumerate(args.fold):
        payload = json.loads(path.read_text(encoding="utf-8"))
        fold_records = payload.get("records", [])
        if not fold_records:
            raise RuntimeError(f"empty Stage23 source fold: {path}")
        for record in fold_records:
            token = str(record.get("token", ""))
            log_name = str(record.get("log_name", ""))
            if not token or not log_name or token in token_seen:
                raise RuntimeError("Stage23 source folds require unique token/log records")
            if log_name in log_fold and log_fold[log_name] != fold_index:
                raise RuntimeError(f"whole log crosses source folds: {log_name}")
            token_seen.add(token)
            log_fold[log_name] = fold_index
            stage23_bucket = int(deterministic_log_bucket((log_name,), 5).item())
            records.append(
                {
                    **record,
                    "source_fold": fold_index,
                    "stage23_oof_member": stage23_bucket,
                }
            )
        sources.append(
            {
                "path": str(path),
                "ordered_token_sha256": payload.get("summary", {}).get(
                    "ordered_token_sha256"
                ),
            }
        )
    records.sort(key=lambda record: record["token"])
    log_bucket = {
        log_name: int(deterministic_log_bucket((log_name,), 5).item())
        for log_name in log_fold
    }
    token_counts = Counter(record["stage23_oof_member"] for record in records)
    log_counts = Counter(log_bucket.values())
    logs_by_bucket = defaultdict(list)
    for log_name, bucket in log_bucket.items():
        logs_by_bucket[bucket].append(log_name)
    result = {
        "schema_version": 1,
        "summary": {
            "stage": 23,
            "purpose": "selector_fit_and_whole_log_oof_calibration",
            "source_folds": [0, 1, 2, 3],
            "count": len(records),
            "num_logs": len(log_fold),
            "num_selector_members": 5,
            "num_modes": 20,
            "noise_namespaces": [-1, 20260811, 20260812],
            "ordered_token_sha256": _ordered_sha(
                [record["token"] for record in records]
            ),
            "whole_log_oof": True,
            "member_token_counts": [token_counts[index] for index in range(5)],
            "member_log_counts": [log_counts[index] for index in range(5)],
            "member_ordered_log_sha256": [
                _ordered_sha(sorted(logs_by_bucket[index])) for index in range(5)
            ],
            "sources": sources,
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
