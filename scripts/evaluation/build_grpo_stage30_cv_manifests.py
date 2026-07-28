#!/usr/bin/env python3
"""Build four immutable leave-one-fold-out Stage30 training manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from navsim.agents.diffusiondrive.stage30_mode_coverage import (
    STAGE30_BUCKET_NAMES,
    load_stage30_bucket_manifest,
)

SOURCE_SHA = "1763ca12bfd1bf480a1ed6a47cc71f95ef31123517e500551dae24292620ef68"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    source_path = root / "artifacts/grpo_stage26/manifests/generator_train_full_6119.json"
    buckets_path = root / "artifacts/grpo_stage30/manifests/buckets_6119.json"
    if sha256(source_path) != SOURCE_SHA:
        raise RuntimeError("Stage30 CV source manifest drifted")
    token_buckets = load_stage30_bucket_manifest(str(buckets_path), require_full=True)
    buckets_sha = sha256(buckets_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    records = source["records"]
    out_dir = root / "artifacts/grpo_stage30/manifests/cv"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for holdout in range(4):
        train_folds = [fold for fold in range(4) if fold != holdout]
        selected = [
            row for row in records if int(row["source_fold"]) in train_folds
        ]
        counts = Counter(
            STAGE30_BUCKET_NAMES[token_buckets[str(row["token"])]]
            for row in selected
        )
        required = dict(zip(STAGE30_BUCKET_NAMES, (2, 30, 8, 24)))
        sparse = [name for name, count in required.items() if counts[name] < count]
        if sparse:
            raise RuntimeError(
                f"Stage30 holdout {holdout} has sparse buckets: {sparse}"
            )
        payload = {
            "schema_version": 1,
            "summary": {
                "stage": 30,
                "purpose": "four_fold_oof_train",
                "holdout_fold": holdout,
                "source_folds": train_folds,
                "count": len(selected),
                "source_manifest_sha256": SOURCE_SHA,
                "bucket_manifest_sha256": buckets_sha,
                "bucket_counts": {
                    name: counts[name] for name in STAGE30_BUCKET_NAMES
                },
            },
            "records": selected,
        }
        output = out_dir / f"train_except_fold{holdout}.json"
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        holdout_path = root / f"artifacts/grpo_stage21/manifests/fold{holdout}_manifest.json"
        holdout_payload = json.loads(holdout_path.read_text(encoding="utf-8"))
        holdout_summary = holdout_payload["summary"]
        outputs.append({
            "holdout_fold": holdout,
            "path": str(output),
            "sha256": sha256(output),
            "count": len(selected),
            "bucket_counts": payload["summary"]["bucket_counts"],
            "holdout_manifest": str(holdout_path),
            "holdout_manifest_sha256": sha256(holdout_path),
            "holdout_count": int(holdout_summary["count"]),
            "holdout_num_logs": int(holdout_summary["num_logs"]),
        })
    freeze = {
        "schema_version": 1,
        "stage": 30,
        "bucket_manifest": str(buckets_path),
        "bucket_manifest_sha256": buckets_sha,
        "source_manifest_sha256": SOURCE_SHA,
        "folds": outputs,
    }
    freeze_path = out_dir / "freeze.json"
    if freeze_path.exists():
        raise FileExistsError(f"refusing to overwrite {freeze_path}")
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(freeze, indent=2))
    print(f"Stage30 CV freeze SHA256: {sha256(freeze_path)}")


if __name__ == "__main__":
    main()
