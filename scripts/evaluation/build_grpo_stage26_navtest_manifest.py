#!/usr/bin/env python3
"""Freeze the Stage26 full-6119 generator manifest after fold5 passes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


FOLD5_REPORT_SHA = (
    "80b951729b4d6041a0d7d02b14e18de1846fac013bdc98357acca4e3dc525e3f"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_sha(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def main() -> None:
    report_path = Path("artifacts/grpo_stage26/fold5/final_report.json")
    if sha256(report_path) != FOLD5_REPORT_SHA:
        raise RuntimeError("Stage26 fold5 report SHA drifted")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("passed") or report.get("stop_before_navtest"):
        raise RuntimeError("Stage26 fold5 did not authorize NavTest")

    records = []
    tokens = set()
    logs = set()
    sources = []
    for fold in range(6):
        path = Path(f"artifacts/grpo_stage21/manifests/fold{fold}_manifest.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        fold_records = payload.get("records", [])
        if not fold_records:
            raise RuntimeError(f"empty Stage26 source fold: {fold}")
        fold_logs = set()
        for source_record in fold_records:
            record = dict(source_record)
            token = str(record.get("token", ""))
            log_name = str(record.get("log_name", ""))
            if not token or not log_name or token in tokens:
                raise RuntimeError("full Stage26 manifest requires unique token/log records")
            tokens.add(token)
            fold_logs.add(log_name)
            record["source_fold"] = fold
            records.append(record)
        if logs.intersection(fold_logs):
            raise RuntimeError("whole log crosses Stage26 source folds")
        logs.update(fold_logs)
        sources.append({
            "path": str(path),
            "file_sha256": sha256(path),
            "ordered_token_sha256": payload["summary"]["ordered_token_sha256"],
            "count": len(fold_records),
            "num_logs": len(fold_logs),
        })
    records.sort(key=lambda record: str(record["token"]))
    if len(records) != 6119 or len(logs) != 908:
        raise RuntimeError("Stage26 full-6119 population drifted")
    result = {
        "schema_version": 1,
        "summary": {
            "stage": 26,
            "purpose": "navtest_generator_train_full_6119",
            "source_folds": [0, 1, 2, 3, 4, 5],
            "count": len(records),
            "num_logs": len(logs),
            "ordered_token_sha256": ordered_sha(
                [str(record["token"]) for record in records]
            ),
            "ordered_log_sha256": ordered_sha(sorted(logs)),
            "whole_log_isolation": True,
            "fold5_report_sha256": FOLD5_REPORT_SHA,
            "sources": sources,
        },
        "records": records,
    }
    output = Path(
        "artifacts/grpo_stage26/manifests/generator_train_full_6119.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
