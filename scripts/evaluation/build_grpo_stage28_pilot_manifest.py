#!/usr/bin/env python3
"""Freeze the Stage28 pilot folds0--3 whole-log training manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_FOLDS = (0, 1, 2, 3)
EXPECTED_COUNT = 4075
EXPECTED_LOGS = 606
SOURCE_SHAS = {
    0: "67dfa5fa9654b09d777dd6edfefde7067276dca8ba94c0fffe5e488cb9704321",
    1: "b694f4833d332f67c41ff695636e0d41fc6761c49f26cfd5c374f2e23b4271de",
    2: "fd25dc8bd507d6ae82120a3010375c0a23b033e25947e2a0555090dea5cffd8c",
    3: "b024ff1ccc3f864fc4fc3de7cdb8c82ec036727f13bd54e424e33e2f27ce957a",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_sha(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/grpo_stage28/manifests/pilot_folds0_3.json"),
    )
    args = parser.parse_args()

    records = []
    tokens = set()
    log_folds: dict[str, int] = {}
    sources = []
    for fold in SOURCE_FOLDS:
        source = Path(
            f"artifacts/grpo_stage21/manifests/fold{fold}_manifest.json"
        )
        if file_sha256(source) != SOURCE_SHAS[fold]:
            raise RuntimeError(f"Stage28 source fold{fold} SHA drifted")
        payload = json.loads(source.read_text(encoding="utf-8"))
        for source_record in payload.get("records", []):
            record = dict(source_record)
            token = str(record.get("token", ""))
            log_name = str(record.get("log_name", ""))
            if not token or not log_name or token in tokens:
                raise RuntimeError("Stage28 requires unique nonempty token/log records")
            if log_name in log_folds and log_folds[log_name] != fold:
                raise RuntimeError(f"whole log crosses Stage28 folds: {log_name}")
            tokens.add(token)
            log_folds[log_name] = fold
            record["source_fold"] = fold
            records.append(record)
        sources.append({
            "path": str(source.resolve()),
            "sha256": SOURCE_SHAS[fold],
            "ordered_token_sha256": payload["summary"]["ordered_token_sha256"],
        })

    records.sort(key=lambda item: str(item["token"]))
    ordered_tokens = [str(record["token"]) for record in records]
    if len(records) != EXPECTED_COUNT or len(log_folds) != EXPECTED_LOGS:
        raise RuntimeError("Stage28 folds0-3 population drifted")
    if ordered_sha(ordered_tokens) != (
        "a1e8a6d8c89c4325830abf125672c962542ac9ec481587bb9e1a1ed8c392c6cc"
    ):
        raise RuntimeError("Stage28 folds0-3 token order differs from Stage23")

    result = {
        "schema_version": 1,
        "summary": {
            "stage": 28,
            "purpose": "public_paired_uplift_pilot_folds0_3",
            "source_folds": list(SOURCE_FOLDS),
            "count": len(records),
            "num_logs": len(log_folds),
            "ordered_token_sha256": ordered_sha(ordered_tokens),
            "ordered_log_sha256": ordered_sha(sorted(log_folds)),
            "whole_log_isolation": True,
            "sources": sources,
        },
        "records": records,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Stage28 manifest: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
