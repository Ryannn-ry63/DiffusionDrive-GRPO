#!/usr/bin/env python3
"""Freeze Stage24 train/calibration manifests without opening protected fold5."""

import argparse
import hashlib
import json
from pathlib import Path


def ordered_sha(records) -> str:
    return hashlib.sha256(
        "\n".join(record["token"] for record in records).encode("utf-8")
    ).hexdigest()


def load_folds(paths: list[Path], purpose: str) -> dict:
    records = []
    tokens = set()
    log_fold = {}
    sources = []
    for fold_index, path in enumerate(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        fold_records = payload.get("records", [])
        if not fold_records:
            raise RuntimeError(f"empty Stage24 source fold: {path}")
        source_fold = int(path.stem.removeprefix("fold").removesuffix("_manifest"))
        for record in fold_records:
            token = str(record.get("token", ""))
            log_name = str(record.get("log_name", ""))
            if not token or not log_name or token in tokens:
                raise RuntimeError("Stage24 folds require unique token/log records")
            if log_name in log_fold and log_fold[log_name] != source_fold:
                raise RuntimeError(f"whole log crosses folds: {log_name}")
            tokens.add(token)
            log_fold[log_name] = source_fold
            records.append({**record, "source_fold": source_fold})
        sources.append({
            "path": str(path),
            "ordered_token_sha256": payload.get("summary", {}).get(
                "ordered_token_sha256"
            ),
        })
    records.sort(key=lambda item: item["token"])
    return {
        "schema_version": 1,
        "summary": {
            "stage": 24,
            "purpose": purpose,
            "source_folds": sorted(set(log_fold.values())),
            "count": len(records),
            "num_logs": len(log_fold),
            "ordered_token_sha256": ordered_sha(records),
            "whole_log_isolation": True,
            "generator_domains": [
                "official_base", "stage16_epoch8", "stage19_epoch8",
                "stage21_epoch8",
            ],
            "noise_namespaces": [-1, 20260811, 20260812],
            "sources": sources,
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold-dir", type=Path,
        default=Path("artifacts/grpo_stage21/manifests"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/grpo_stage24/manifests"),
    )
    args = parser.parse_args()
    train_paths = [args.fold_dir / f"fold{index}_manifest.json" for index in range(3)]
    calibration_paths = [args.fold_dir / "fold3_manifest.json"]
    for path in train_paths + calibration_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    train = load_folds(train_paths, "stage24_selector_train_folds0_2")
    calibration = load_folds(
        calibration_paths, "stage24_full_ensemble_calibration_fold3"
    )
    train_logs = {record["log_name"] for record in train["records"]}
    calibration_logs = {record["log_name"] for record in calibration["records"]}
    if train_logs & calibration_logs:
        raise RuntimeError("Stage24 train/calibration logs overlap")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "selector_train_manifest.json": train,
        "selector_calibration_manifest.json": calibration,
    }
    for name, payload in outputs.items():
        (args.output_dir / name).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    print(json.dumps({name: value["summary"] for name, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
