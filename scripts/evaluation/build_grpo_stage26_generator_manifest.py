#!/usr/bin/env python3
"""Freeze the Stage26 final-generator folds0--4 whole-log manifest."""

import hashlib
import json
from pathlib import Path


def ordered_sha(values):
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def main() -> None:
    records = []
    tokens = set()
    log_folds = {}
    sources = []
    for fold in range(5):
        path = Path(f"artifacts/grpo_stage21/manifests/fold{fold}_manifest.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        fold_records = payload.get("records", [])
        if not fold_records:
            raise RuntimeError(f"empty Stage26 source fold: {fold}")
        for source_record in fold_records:
            record = dict(source_record)
            token = str(record.get("token", ""))
            log_name = str(record.get("log_name", ""))
            if not token or not log_name or token in tokens:
                raise RuntimeError("Stage26 folds require unique token/log records")
            if log_name in log_folds and log_folds[log_name] != fold:
                raise RuntimeError(f"whole log crosses Stage26 folds: {log_name}")
            tokens.add(token)
            log_folds[log_name] = fold
            record["source_fold"] = fold
            records.append(record)
        sources.append({
            "path": str(path),
            "ordered_token_sha256": payload["summary"]["ordered_token_sha256"],
        })
    records.sort(key=lambda record: record["token"])
    if len(records) != 5096 or len(log_folds) != 757:
        raise RuntimeError("Stage26 folds0-4 population drifted")
    result = {
        "schema_version": 1,
        "summary": {
            "stage": 26,
            "purpose": "final_generator_train_folds0_4",
            "source_folds": [0, 1, 2, 3, 4],
            "count": len(records),
            "num_logs": len(log_folds),
            "ordered_token_sha256": ordered_sha(
                [str(record["token"]) for record in records]
            ),
            "ordered_log_sha256": ordered_sha(sorted(log_folds)),
            "whole_log_isolation": True,
            "sources": sources,
        },
        "records": records,
    }
    output = Path(
        "artifacts/grpo_stage26/manifests/generator_train_folds0_4.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
