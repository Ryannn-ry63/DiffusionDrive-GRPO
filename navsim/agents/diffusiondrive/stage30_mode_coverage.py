"""Frozen Stage30 bucket-manifest semantics shared by model and sampler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping


STAGE30_PLAN_SHA256 = (
    "e9867ec48d4806ff97adb0284bae7550305cb1350e4ec66dbfd20d245de435ce"
)
STAGE30_PUBLIC_SHA256 = (
    "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
)
STAGE30_SELECTOR_SHA256 = (
    "023d7b6b77bb8fa2dc3e779849f3716688abf0796b019416494c80b29615b691"
)
STAGE30_CALIBRATION_SHA256 = (
    "fec2a10573e913e4452f8eea92e6334fcd5e6f1104e6244876770973a1f41990"
)
STAGE30_NAMESPACES = (20261301, 20261302, 20261303)
STAGE30_BUCKET_NAMES = ("severe", "recoverable", "boundary", "mature")


def load_stage30_bucket_manifest(path: str, require_full: bool = True) -> Dict[str, int]:
    """Load token buckets and reject provenance or semantic drift."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Stage30 bucket manifest missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    expected = {
        "plan_sha256": STAGE30_PLAN_SHA256,
        "public_checkpoint_sha256": STAGE30_PUBLIC_SHA256,
        "selector_checkpoint_sha256": STAGE30_SELECTOR_SHA256,
        "selector_calibration_sha256": STAGE30_CALIBRATION_SHA256,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"Stage30 manifest {key} drifted")
    if tuple(summary.get("namespaces", ())) != STAGE30_NAMESPACES:
        raise RuntimeError("Stage30 manifest namespace order drifted")
    if tuple(summary.get("bucket_names", ())) != STAGE30_BUCKET_NAMES:
        raise RuntimeError("Stage30 manifest bucket definitions drifted")
    records = payload.get("tokens")
    if not isinstance(records, Mapping) or not records:
        raise RuntimeError("Stage30 manifest contains no token records")
    if int(summary.get("count", -1)) != len(records):
        raise RuntimeError("Stage30 manifest token count is inconsistent")
    if require_full and len(records) != 6119:
        raise RuntimeError("formal Stage30 manifest must contain all 6,119 tokens")

    result: Dict[str, int] = {}
    counts = [0, 0, 0, 0]
    for token, record in records.items():
        if not isinstance(token, str) or not token:
            raise RuntimeError("Stage30 manifest contains an invalid token")
        if not isinstance(record, Mapping):
            raise RuntimeError(f"Stage30 token record is invalid: {token}")
        bucket_id = int(record.get("bucket_id", -1))
        if bucket_id not in range(4):
            raise RuntimeError(f"Stage30 bucket id is invalid for {token}")
        if record.get("bucket") != STAGE30_BUCKET_NAMES[bucket_id]:
            raise RuntimeError(f"Stage30 bucket name/id mismatch for {token}")
        fold = int(record.get("fold", -1))
        if fold not in range(6):
            raise RuntimeError(f"Stage30 fold is invalid for {token}")
        result[token] = bucket_id
        counts[bucket_id] += 1
    declared = summary.get("bucket_counts", {})
    for index, name in enumerate(STAGE30_BUCKET_NAMES):
        if int(declared.get(name, -1)) != counts[index]:
            raise RuntimeError(f"Stage30 declared {name} count drifted")
    return result

