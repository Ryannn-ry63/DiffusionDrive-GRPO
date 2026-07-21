#!/usr/bin/env python3
"""Build log-disjoint Stage19 manifests with frozen base selected modes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


BASE_SHA256 = "59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d"
FULL_SCHEDULE = {
    "truncation_timestep": 32,
    "roll_timesteps": [32, 24, 16, 8, 0],
    "scheduler_num_inference_steps": 125,
    "scheduler_step_stride": 8,
    "transitions": [[32, 24], [24, 16], [16, 8], [8, 0], [0, -8]],
}
PARTITIONS = ("fit", "calibration", "test")
FRACTIONS = {"fit": 0.70, "calibration": 0.15, "test": 0.15}


def ordered_sha(values):
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def stable_digest(seed, namespace, value):
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).hexdigest()


def load_metric_token_logs(metric_cache_path):
    metadata = sorted((metric_cache_path / "metadata").glob("*.csv"))
    if not metadata:
        raise FileNotFoundError(f"no metric metadata under {metric_cache_path}")
    result = {}
    for path in metadata:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                metric_path = Path(row["file_name"])
                token = metric_path.parent.name
                log_name = metric_path.parents[2].name
                previous = result.setdefault(token, log_name)
                if previous != log_name:
                    raise RuntimeError(f"token {token} maps to multiple logs")
    return result


def assign_logs(token_logs, seed=20260722):
    """Greedily minimize normalized token-count error using whole logs."""
    counts = Counter(token_logs.values())
    total = sum(counts.values())
    targets = {name: total * FRACTIONS[name] for name in PARTITIONS}
    assigned_counts = {name: 0 for name in PARTITIONS}
    result = {}
    ordered_logs = sorted(
        counts,
        key=lambda log_name: (
            -counts[log_name],
            stable_digest(seed, "log-order", log_name),
        ),
    )
    for log_name in ordered_logs:
        count = counts[log_name]
        candidates = []
        for partition in PARTITIONS:
            hypothetical = dict(assigned_counts)
            hypothetical[partition] += count
            objective = sum(
                ((hypothetical[name] - targets[name]) / targets[name]) ** 2
                for name in PARTITIONS
            )
            candidates.append(
                (
                    objective,
                    stable_digest(seed, f"partition-{partition}", log_name),
                    partition,
                )
            )
        partition = min(candidates)[-1]
        result[log_name] = partition
        assigned_counts[partition] += count
    if set(result) != set(counts) or any(
        not any(value == partition for value in result.values())
        for partition in PARTITIONS
    ):
        raise RuntimeError("invalid log partition assignment")
    return result


def validate_selected_modes(source_tokens, payload):
    summary = payload.get("summary", {})
    records = payload.get("records")
    if not isinstance(records, list):
        raise RuntimeError("selected-mode artifact has no records")
    if summary.get("checkpoint_sha256") != BASE_SHA256:
        raise RuntimeError("selected-mode artifact is not the frozen base")
    if summary.get("reference_checkpoint_sha256") != BASE_SHA256:
        raise RuntimeError("selected-mode reference checkpoint mismatch")
    if summary.get("selector_logits_source") != "reference":
        raise RuntimeError("selected modes were not produced by reference selector")
    if summary.get("schedule") != FULL_SCHEDULE:
        raise RuntimeError("selected modes were not produced by full schedule")
    record_map = {}
    for record in records:
        token = str(record["token"])
        if token in record_map:
            raise RuntimeError(f"duplicate selected-mode token: {token}")
        mode = int(record["selected_mode"])
        if mode < 0 or mode >= 20:
            raise RuntimeError(f"invalid selected mode for {token}: {mode}")
        record_map[token] = mode
    if set(record_map) != set(source_tokens):
        raise RuntimeError(
            "selected-mode artifact token set differs from navtrain source"
        )
    return record_map


def build_manifest(name, records, source, selected_mode_artifact, seed):
    tokens = [record["token"] for record in records]
    logs = sorted({record["log_name"] for record in records})
    return {
        "schema_version": 1,
        "summary": {
            "name": name,
            "selection": "whole_log_greedy_70_15_15",
            "seed": seed,
            "count": len(records),
            "num_logs": len(logs),
            "ordered_token_sha256": ordered_sha(tokens),
            "ordered_log_sha256": ordered_sha(logs),
            "source": str(source.resolve()),
            "selected_mode_artifact": str(selected_mode_artifact.resolve()),
            "selected_mode_artifact_sha256": hashlib.sha256(
                selected_mode_artifact.read_bytes()
            ).hexdigest(),
            "base_checkpoint_sha256": BASE_SHA256,
            "schedule": FULL_SCHEDULE,
            "group_definition": "same_token_same_selected_anchor",
            "group_size": 8,
        },
        "records": records,
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("artifacts/grpo_stage0/navtrain_base_6119.json"),
    )
    parser.add_argument(
        "--metric-cache",
        type=Path,
        default=Path(
            "/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/"
            "metric_cache_trainval"
        ),
    )
    parser.add_argument("--selected-mode-artifact", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/grpo_stage19/manifests"),
    )
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    source_payload = json.loads(args.source.read_text(encoding="utf-8"))
    source_records = source_payload.get("records")
    if not isinstance(source_records, list) or len(source_records) != 6119:
        raise RuntimeError("Stage19 source must contain exactly 6119 records")
    source_tokens = [str(record["token"]) for record in source_records]
    if len(source_tokens) != len(set(source_tokens)):
        raise RuntimeError("Stage19 source contains duplicate tokens")

    metric_logs = load_metric_token_logs(args.metric_cache)
    missing_logs = [token for token in source_tokens if token not in metric_logs]
    if missing_logs:
        raise RuntimeError(f"missing metric log for token {missing_logs[0]}")
    selected_modes = validate_selected_modes(
        source_tokens,
        json.loads(args.selected_mode_artifact.read_text(encoding="utf-8")),
    )
    token_logs = {token: metric_logs[token] for token in source_tokens}
    log_partition = assign_logs(token_logs, args.seed)

    by_partition = {name: [] for name in PARTITIONS}
    all_records = []
    for token in source_tokens:
        partition = log_partition[token_logs[token]]
        record = {
            "token": token,
            "log_name": token_logs[token],
            "split": "train",
            "stage19_partition": partition,
            "selected_mode": selected_modes[token],
        }
        by_partition[partition].append(record)
        all_records.append(record)

    log_sets = {
        name: {record["log_name"] for record in records}
        for name, records in by_partition.items()
    }
    if any(
        log_sets[PARTITIONS[i]] & log_sets[PARTITIONS[j]]
        for i in range(len(PARTITIONS))
        for j in range(i + 1, len(PARTITIONS))
    ):
        raise RuntimeError("Stage19 log partitions overlap")

    outputs = {}
    for name, records in {
        **by_partition,
        "all": all_records,
        "smoke64": by_partition["fit"][:64],
    }.items():
        path = args.output_dir / f"{name}_manifest.json"
        payload = build_manifest(
            name, records, args.source, args.selected_mode_artifact, args.seed
        )
        write_json(path, payload)
        outputs[name] = {
            "path": str(path),
            "tokens": len(records),
            "logs": payload["summary"]["num_logs"],
            "token_sha256": payload["summary"]["ordered_token_sha256"],
        }
    print(json.dumps({"outputs": outputs, "log_overlap": 0}, indent=2))


if __name__ == "__main__":
    main()
