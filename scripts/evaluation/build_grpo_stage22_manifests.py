#!/usr/bin/env python3
"""Build preregistered Stage22 subsets from the frozen Stage21 whole-log folds."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


BASE_SHA = "59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d"
SCHEDULE = {
    "truncation_timestep": 32,
    "roll_timesteps": [32, 24, 16, 8, 0],
    "scheduler_num_inference_steps": 125,
    "scheduler_step_stride": 8,
    "transitions": [[32, 24], [24, 16], [16, 8], [8, 0], [0, -8]],
}
EXPECTED_COUNTS = (1019, 1021, 1016, 1019, 1021, 1023)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_sha(values) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def bounded_unit_mean(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or len(result) == 0 or not np.isfinite(result).all():
        raise ValueError("Stage22 weights require a finite non-empty vector")
    result /= result.mean()
    for _ in range(64):
        result = np.clip(result, 0.5, 2.0)
        error = len(result) - result.sum()
        if abs(error) <= 1e-12:
            break
        free = (result > 0.5 + 1e-12) & (result < 2.0 - 1e-12)
        if not free.any():
            raise RuntimeError("cannot normalize bounded Stage22 weights")
        result[free] += error / free.sum()
    if abs(result.mean() - 1.0) > 1e-9:
        raise RuntimeError("Stage22 scene-weight normalization failed")
    return result


def load_fold(path: Path, index: int):
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    if (
        summary.get("base_checkpoint_sha256") != BASE_SHA
        or summary.get("schedule") != SCHEDULE
        or summary.get("stage21_objective") != "base_anchored_log_robust_v1"
    ):
        raise RuntimeError(f"Stage21 fold provenance mismatch: {path}")
    records = payload.get("records", [])
    if len(records) != EXPECTED_COUNTS[index]:
        raise RuntimeError(f"Stage21 fold{index} count drifted: {len(records)}")
    result = []
    for source in records:
        result.append(
            {
                "token": str(source["token"]),
                "log_name": str(source["log_name"]),
                "selected_mode": int(source["selected_mode"]),
                "stage22_fold": index,
                "audit_base_reward_mean": float(source["base_reward_mean"]),
                "audit_base_collision_mean": float(source["base_collision_mean"]),
                "audit_base_drivable_mean": float(source["base_drivable_mean"]),
                "audit_base_ttc_mean": float(source["base_ttc_mean"]),
            }
        )
    return result


def reweight(records):
    counts = Counter(record["log_name"] for record in records)
    raw = [1.0 / math.log1p(counts[record["log_name"]]) for record in records]
    weights = bounded_unit_mean(raw)
    output = []
    for record, weight in zip(records, weights):
        item = dict(record)
        item["log_frequency"] = counts[record["log_name"]]
        item["scene_weight"] = float(weight)
        output.append(item)
    return output


def write_manifest(path: Path, name: str, records, source_hashes):
    records = reweight(sorted(records, key=lambda item: item["token"]))
    tokens = [record["token"] for record in records]
    logs = sorted({record["log_name"] for record in records})
    if len(tokens) != len(set(tokens)):
        raise RuntimeError(f"duplicate token in Stage22 {name}")
    summary = {
        "name": name,
        "count": len(records),
        "num_logs": len(logs),
        "stage22_objective": "paired_delta_bootstrap_exact_kl_lora_v2",
        "base_checkpoint_sha256": BASE_SHA,
        "schedule": SCHEDULE,
        "group_definition": "same_token_same_selected_anchor_common_random",
        "group_size": 8,
        "scene_weight_definition": "inverse_log_frequency_clip_0.5_2_unit_mean",
        "scene_weight_mean": float(np.mean([item["scene_weight"] for item in records])),
        "scene_weight_min": float(np.min([item["scene_weight"] for item in records])),
        "scene_weight_max": float(np.max([item["scene_weight"] for item in records])),
        "ordered_token_sha256": ordered_sha(tokens),
        "ordered_log_sha256": ordered_sha(logs),
        "source_stage21_fold_sha256": source_hashes,
        "offline_base_reward_used_by_objective": False,
    }
    path.write_text(
        json.dumps(
            {"schema_version": 1, "summary": summary, "records": records},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage21-dir", type=Path,
        default=Path("artifacts/grpo_stage21/manifests"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/grpo_stage22/manifests"),
    )
    args = parser.parse_args()

    folds = []
    hashes = {}
    for index in range(6):
        path = args.stage21_dir / f"fold{index}_manifest.json"
        folds.append(load_fold(path, index))
        hashes[f"fold{index}"] = file_sha256(path)
    all_tokens = [record["token"] for fold in folds for record in fold]
    all_logs = [set(record["log_name"] for record in fold) for fold in folds]
    if len(all_tokens) != 6119 or len(all_tokens) != len(set(all_tokens)):
        raise RuntimeError("Stage22 source folds do not form the 6119-token set")
    if any(all_logs[left] & all_logs[right] for left in range(6)
           for right in range(left + 1, 6)):
        raise RuntimeError("Stage22 source folds overlap by log")

    fit_select = sum(folds[:3], [])
    retrain_fit = sum(folds[:4], [])
    formal_all = sum(folds, [])
    definitions = {
        **{f"fold{index}": fold for index, fold in enumerate(folds)},
        "fit_select": fit_select,
        "inner_calibration": folds[3],
        "retrain_fit": retrain_fit,
        "known_stress": folds[4],
        "internal_test": folds[5],
        "formal_all": formal_all,
        "smoke8": sorted(fit_select, key=lambda item: item["token"])[:8],
        "smoke32": sorted(fit_select, key=lambda item: item["token"])[:32],
        "smoke64": sorted(fit_select, key=lambda item: item["token"])[:64],
    }
    expected = {
        "fit_select": 3056,
        "inner_calibration": 1019,
        "retrain_fit": 4075,
        "known_stress": 1021,
        "internal_test": 1023,
        "formal_all": 6119,
    }
    for name, count in expected.items():
        if len(definitions[name]) != count:
            raise RuntimeError(f"Stage22 {name} count drifted")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, records in definitions.items():
        summaries[name] = write_manifest(
            args.output_dir / f"{name}_manifest.json",
            name,
            records,
            hashes,
        )
    audit = {
        "stage22_objective": "paired_delta_bootstrap_exact_kl_lora_v2",
        "counts": {name: summary["count"] for name, summary in summaries.items()},
        "zero_token_overlap_between_eval_folds": not any(
            set(item["token"] for item in folds[left])
            & set(item["token"] for item in folds[right])
            for left in range(3, 6) for right in range(left + 1, 6)
        ),
        "zero_log_overlap": True,
        "formal_token_sha256": summaries["formal_all"]["ordered_token_sha256"],
        "offline_base_reward_used_by_objective": False,
    }
    (args.output_dir / "manifest_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
