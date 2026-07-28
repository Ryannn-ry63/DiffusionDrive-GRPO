#!/usr/bin/env python3
"""Freeze the Stage27 provisional-generator folds0--4 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_FOLDS = (0, 1, 2, 3, 4)
EXPECTED_COUNT = 5096
EXPECTED_LOGS = 757


def ordered_sha(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/grpo_stage27/manifests/"
            "generator_train_folds0_4.json"
        ),
    )
    args = parser.parse_args()

    records = []
    tokens = set()
    log_folds = {}
    sources = []
    for fold in SOURCE_FOLDS:
        path = Path(
            f"artifacts/grpo_stage21/manifests/fold{fold}_manifest.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        fold_records = payload.get("records", [])
        if not fold_records:
            raise RuntimeError(f"empty Stage27 source fold: {fold}")
        for source_record in fold_records:
            record = dict(source_record)
            token = str(record.get("token", ""))
            log_name = str(record.get("log_name", ""))
            if not token or not log_name or token in tokens:
                raise RuntimeError(
                    "Stage27 folds require unique token/log records"
                )
            if log_name in log_folds and log_folds[log_name] != fold:
                raise RuntimeError(
                    f"whole log crosses Stage27 folds: {log_name}"
                )
            tokens.add(token)
            log_folds[log_name] = fold
            record["source_fold"] = fold
            records.append(record)
        sources.append({
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "ordered_token_sha256": payload["summary"][
                "ordered_token_sha256"
            ],
        })

    records.sort(key=lambda record: str(record["token"]))
    if len(records) != EXPECTED_COUNT or len(log_folds) != EXPECTED_LOGS:
        raise RuntimeError("Stage27 folds0-4 population drifted")
    result = {
        "schema_version": 1,
        "summary": {
            "stage": 27,
            "phase": 3,
            "purpose": "provisional_generator_train_folds0_4",
            "source_folds": list(SOURCE_FOLDS),
            "count": len(records),
            "num_logs": len(log_folds),
            "ordered_token_sha256": ordered_sha([
                str(record["token"]) for record in records
            ]),
            "ordered_log_sha256": ordered_sha(sorted(log_folds)),
            "whole_log_isolation": True,
            "sources": sources,
        },
        "records": records,
    }
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite Stage27 manifest: {args.output}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
