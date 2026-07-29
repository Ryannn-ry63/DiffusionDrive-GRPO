#!/usr/bin/env python3
"""Freeze the disjoint folds-2/3 fitting manifest for the Stage37 JFI heads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PLAN_SHA256 = "4dc469dd0be4d2579796ff70ae7a09ed9fd8e4c92449cd057e42463d268bb5d8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cv_path = root / "artifacts/grpo_stage30/manifests/cv/freeze.json"
    output_dir = root / "artifacts/grpo_stage37/jfi/manifests"
    output = output_dir / "train_folds2_3.json"
    freeze_output = output_dir / "freeze.json"
    if output.exists() or freeze_output.exists():
        raise FileExistsError("refusing to overwrite frozen Stage37 JFI manifests")

    cv = json.loads(cv_path.read_text(encoding="utf-8"))
    if cv.get("stage") != 30 or len(cv.get("folds", [])) != 4:
        raise RuntimeError("Stage30 CV freeze drifted")
    records = []
    seen_tokens: set[str] = set()
    seen_logs: dict[str, int] = {}
    sources = []
    for fold_index in (2, 3):
        entry = cv["folds"][fold_index]
        path = Path(entry["holdout_manifest"])
        if (
            entry.get("holdout_fold") != fold_index
            or not path.is_file()
            or sha256(path) != entry["holdout_manifest_sha256"]
        ):
            raise RuntimeError(f"frozen Stage37 JFI fold {fold_index} drifted")
        payload = json.loads(path.read_text(encoding="utf-8"))
        fold_records = payload.get("records", [])
        if len(fold_records) != int(entry["holdout_count"]):
            raise RuntimeError(f"fold {fold_index} count drifted")
        for record in fold_records:
            token = str(record.get("token", ""))
            log_name = str(record.get("log_name", record.get("log_id", "")))
            if not token or not log_name or token in seen_tokens:
                raise RuntimeError("folds2/3 contain invalid or duplicate tokens")
            if log_name in seen_logs and seen_logs[log_name] != fold_index:
                raise RuntimeError("whole-log isolation failed across folds2/3")
            seen_tokens.add(token)
            seen_logs[log_name] = fold_index
            records.append({**record, "stage37_jfi_source_fold": fold_index})
        sources.append({
            "fold": fold_index,
            "path": str(path),
            "sha256": entry["holdout_manifest_sha256"],
            "count": len(fold_records),
        })

    ordered_sha = hashlib.sha256(
        "\n".join(str(record["token"]) for record in records).encode()
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "summary": {
            "stage": 37,
            "purpose": "jfi_head_fit_folds2_3_only",
            "plan_sha256": PLAN_SHA256,
            "source_folds": [2, 3],
            "excluded_folds": [0, 1],
            "count": len(records),
            "num_logs": len(seen_logs),
            "ordered_token_sha256": ordered_sha,
            "whole_log_isolation": True,
            "calibration_or_test_labels_present": False,
        },
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    freeze = {
        "schema_version": 1,
        "stage": 37,
        "plan_sha256": PLAN_SHA256,
        "train_manifest": str(output),
        "train_manifest_sha256": sha256(output),
        "count": len(records),
        "num_logs": len(seen_logs),
        "ordered_token_sha256": ordered_sha,
        "source_manifests": sources,
    }
    freeze_output.write_text(
        json.dumps(freeze, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
