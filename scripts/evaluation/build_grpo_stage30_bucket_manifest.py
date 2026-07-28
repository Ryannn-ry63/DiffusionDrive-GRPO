#!/usr/bin/env python3
"""Freeze Stage30 severe/recoverable/boundary/mature labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from navsim.agents.diffusiondrive.stage30_mode_coverage import (
    STAGE30_BUCKET_NAMES,
    STAGE30_CALIBRATION_SHA256,
    STAGE30_NAMESPACES,
    STAGE30_PLAN_SHA256,
    STAGE30_PUBLIC_SHA256,
    STAGE30_SELECTOR_SHA256,
    load_stage30_bucket_manifest,
)

ALL_MANIFEST_SHA = "1763ca12bfd1bf480a1ed6a47cc71f95ef31123517e500551dae24292620ef68"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records_by_token(path: Path, namespace: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    selector = summary.get("stage25_selector", {})
    checks = (
        summary.get("completed") is True,
        summary.get("num_failures") == 0,
        summary.get("num_tokens") == 6119,
        summary.get("checkpoint_sha256") == STAGE30_PUBLIC_SHA256,
        summary.get("reference_checkpoint_sha256") == STAGE30_PUBLIC_SHA256,
        summary.get("evaluation_noise_namespace") == namespace,
        summary.get("selector_logits_source") == "trajectory_relative_harm_v3",
        selector.get("checkpoint_sha256") == STAGE30_SELECTOR_SHA256,
        selector.get("calibration_sha256") == STAGE30_CALIBRATION_SHA256,
    )
    if not all(checks):
        raise RuntimeError(f"Stage30 label provenance failed: {path}")
    result = {str(row["token"]): row for row in payload.get("records", ())}
    if len(result) != 6119 or len(payload.get("records", ())) != 6119:
        raise RuntimeError(f"Stage30 label tokens are incomplete/duplicated: {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/grpo_stage30/manifests/buckets_6119.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    source_manifest = root / "artifacts/grpo_stage26/manifests/generator_train_full_6119.json"
    if sha256(source_manifest) != ALL_MANIFEST_SHA:
        raise RuntimeError("Stage30 all-token source manifest drifted")
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    source_rows = {str(row["token"]): row for row in source["records"]}
    if len(source_rows) != 6119:
        raise RuntimeError("Stage30 source token count drifted")

    label_paths = [
        root / f"artifacts/grpo_stage30/labels/public_smulti_ns{namespace}.json"
        for namespace in STAGE30_NAMESPACES
    ]
    label_maps = [
        records_by_token(path, namespace)
        for path, namespace in zip(label_paths, STAGE30_NAMESPACES)
    ]
    if any(set(mapping) != set(source_rows) for mapping in label_maps):
        raise RuntimeError("Stage30 label/source token identities differ")

    tokens = {}
    counts = Counter()
    for token in sorted(source_rows):
        rows = [mapping[token] for mapping in label_maps]
        scores = [float(row["selected_reward"]) for row in rows]
        if not all(math.isfinite(score) for score in scores):
            raise RuntimeError(f"non-finite Stage30 score for {token}")
        selected_components = [row["selected_components"] for row in rows]
        safety_failure = any(
            float(components[name]) < 1.0 - 1e-6
            for components in selected_components
            for name in ("collision", "drivable", "ttc")
        )
        mean_score = sum(scores) / len(scores)
        if safety_failure or mean_score < 0.5:
            bucket_id = 0
        elif max(scores) < 0.75:
            bucket_id = 1
        elif min(scores) < 0.75 <= max(scores):
            bucket_id = 2
        else:
            bucket_id = 3
        bucket = STAGE30_BUCKET_NAMES[bucket_id]
        counts[bucket] += 1
        tokens[token] = {
            "bucket": bucket,
            "bucket_id": bucket_id,
            "fold": int(source_rows[token]["source_fold"]),
            "log_name": str(source_rows[token]["log_name"]),
            "selected_scores": scores,
            "selected_score_mean": mean_score,
            "selected_score_min": min(scores),
            "selected_score_max": max(scores),
            "safety_failure": safety_failure,
        }
    result = {
        "schema_version": 1,
        "stage": 30,
        "summary": {
            "plan_sha256": STAGE30_PLAN_SHA256,
            "public_checkpoint_sha256": STAGE30_PUBLIC_SHA256,
            "selector_checkpoint_sha256": STAGE30_SELECTOR_SHA256,
            "selector_calibration_sha256": STAGE30_CALIBRATION_SHA256,
            "source_manifest_sha256": ALL_MANIFEST_SHA,
            "namespaces": list(STAGE30_NAMESPACES),
            "bucket_names": list(STAGE30_BUCKET_NAMES),
            "bucket_counts": {name: counts[name] for name in STAGE30_BUCKET_NAMES},
            "count": len(tokens),
            "label_artifacts": [
                {"path": str(path), "sha256": sha256(path)}
                for path in label_paths
            ],
        },
        "tokens": tokens,
    }
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    load_stage30_bucket_manifest(str(output), require_full=True)
    print(json.dumps(result["summary"], indent=2))
    print(f"Stage30 bucket manifest: {output}")
    print(f"SHA256: {sha256(output)}")


if __name__ == "__main__":
    main()

