#!/usr/bin/env python3
"""Build preregistered Stage21 whole-log folds and base-anchor metadata."""

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
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
COMPONENT_INDEX = {"collision": 0, "drivable": 1, "ttc": 3}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_sha(values) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def load_artifact(path: Path, namespace: int):
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    if summary.get("checkpoint_sha256") != BASE_SHA:
        raise RuntimeError(f"base checkpoint mismatch: {path}")
    if summary.get("schedule") != SCHEDULE:
        raise RuntimeError(f"schedule mismatch: {path}")
    if int(summary.get("evaluation_noise_namespace", -2)) != namespace:
        raise RuntimeError(f"noise namespace mismatch: {path}")
    records = payload.get("records", [])
    if len(records) != 6119 or not summary.get("completed", False):
        raise RuntimeError(f"incomplete 6119-token artifact: {path}")
    result = {str(record["token"]): record for record in records}
    if len(result) != len(records):
        raise RuntimeError(f"duplicate token in {path}")
    return payload, result


def bounded_unit_mean(raw):
    values = np.asarray(raw, dtype=np.float64)
    values = values / values.mean()
    for _ in range(32):
        values = np.clip(values, 0.25, 4.0)
        free = (values > 0.25 + 1e-12) & (values < 4.0 - 1e-12)
        error = len(values) - values.sum()
        if abs(error) < 1e-12:
            break
        if not free.any():
            raise RuntimeError("cannot normalize bounded scene weights")
        values[free] += error / free.sum()
    if abs(values.mean() - 1.0) > 1e-9:
        raise RuntimeError("scene-weight normalization failed")
    return values


def feature_vector(records):
    vector = np.zeros(27, dtype=np.float64)
    vector[0] = len(records)
    vector[1] = 1.0
    for record in records:
        reward = record["base_reward_mean"]
        reward_bin = 0 if reward < 0.5 else 1 if reward < 0.75 else 2 if reward < 0.9 else 3
        vector[2 + reward_bin] += 1.0
        vector[6] += float(record["safety_failure"])
        vector[7 + record["selected_mode"]] += 1.0
    return vector


def assignment_audit(folds):
    token_counts = np.array([sum(len(records) for records in fold.values()) for fold in folds])
    log_counts = np.array([len(fold) for fold in folds])
    hard_shares = []
    mode_distributions = []
    for fold in folds:
        flat = [record for records in fold.values() for record in records]
        hard_shares.append(sum(record["base_reward_mean"] < 0.5 for record in flat) / len(flat))
        modes = np.bincount([record["selected_mode"] for record in flat], minlength=20)
        mode_distributions.append(modes / modes.sum())
    max_tv = max(
        0.5 * np.abs(mode_distributions[left] - mode_distributions[right]).sum()
        for left in range(6) for right in range(left + 1, 6)
    )
    target = token_counts.sum() / 6.0
    return {
        "token_counts": token_counts.tolist(),
        "log_counts": log_counts.tolist(),
        "hard_scene_shares": hard_shares,
        "max_token_relative_imbalance": float(np.max(np.abs(token_counts - target)) / target),
        "hard_share_range": float(max(hard_shares) - min(hard_shares)),
        "max_selected_mode_tv": float(max_tv),
        "passes": bool(
            log_counts.min() >= 130
            and np.max(np.abs(token_counts - target)) / target <= 0.15
            and max(hard_shares) - min(hard_shares) <= 0.02
            and max_tv <= 0.05
        ),
    }


def build_folds(by_log, seed):
    logs = list(by_log)
    vectors = {log_name: feature_vector(by_log[log_name]) for log_name in logs}
    total = sum(vectors.values(), np.zeros(27, dtype=np.float64))
    target = total / 6.0
    weights = np.array([10.0, 8.0, 3.0, 2.0, 2.0, 2.0, 4.0] + [0.5] * 20)
    best = None
    for trial in range(512):
        rng = random.Random(seed + trial * 104729)
        order = sorted(
            logs,
            key=lambda name: (
                -len(by_log[name]) * (0.85 + 0.30 * rng.random()),
                hashlib.sha256(f"{seed}:{trial}:{name}".encode()).hexdigest(),
            ),
        )
        fold_vectors = [np.zeros(27, dtype=np.float64) for _ in range(6)]
        folds = [defaultdict(list) for _ in range(6)]
        for log_name in order:
            vector = vectors[log_name]
            costs = []
            for fold_index in range(6):
                current = fold_vectors[fold_index]
                projected = fold_vectors[fold_index] + vector
                denominator = np.maximum(target, 1.0)
                before = (current - target) / denominator
                after = (projected - target) / denominator
                before_overflow = max(0.0, current[0] - 1.15 * target[0]) ** 2
                after_overflow = max(0.0, projected[0] - 1.15 * target[0]) ** 2
                costs.append(float(
                    (weights * (np.square(after) - np.square(before))).sum()
                    + after_overflow - before_overflow
                ))
            minimum = min(costs)
            choices = [index for index, value in enumerate(costs) if abs(value - minimum) < 1e-12]
            chosen = choices[rng.randrange(len(choices))]
            fold_vectors[chosen] += vector
            folds[chosen][log_name].extend(by_log[log_name])
        audit = assignment_audit(folds)
        penalty = (
            max(0.0, 130 - min(audit["log_counts"])) * 100.0
            + max(0.0, audit["max_token_relative_imbalance"] - 0.15) * 1000.0
            + max(0.0, audit["hard_share_range"] - 0.02) * 10000.0
            + max(0.0, audit["max_selected_mode_tv"] - 0.05) * 10000.0
            + audit["max_token_relative_imbalance"]
            + audit["hard_share_range"]
            + audit["max_selected_mode_tv"]
        )
        if best is None or penalty < best[0]:
            best = (penalty, folds, audit, trial)
        if audit["passes"]:
            return folds, audit, trial
    raise RuntimeError(f"no fold assignment passed; best={best[2]}, trial={best[3]}")


def write_manifest(path, name, records, summary_base):
    records = sorted(records, key=lambda record: record["token"])
    logs = sorted({record["log_name"] for record in records})
    summary = {
        **summary_base,
        "name": name,
        "count": len(records),
        "num_logs": len(logs),
        "ordered_token_sha256": ordered_sha([record["token"] for record in records]),
        "ordered_log_sha256": ordered_sha(logs),
    }
    path.write_text(
        json.dumps({"schema_version": 1, "summary": summary, "records": records}, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--default-artifact", type=Path, required=True)
    parser.add_argument("--alternate-artifact", type=Path, required=True)
    parser.add_argument("--token-log-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    _, default = load_artifact(args.default_artifact, -1)
    _, alternate = load_artifact(args.alternate_artifact, 20260728)
    mapping_payload = json.loads(args.token_log_manifest.read_text(encoding="utf-8"))
    mapping = {str(record["token"]): str(record["log_name"]) for record in mapping_payload["records"]}
    if set(default) != set(alternate) or set(default) != set(mapping):
        raise RuntimeError("Stage21 token sets differ across base artifacts/log mapping")

    records = []
    log_counts = Counter(mapping.values())
    raw_weights = []
    for token in sorted(default):
        first, second = default[token], alternate[token]
        mode = int(first["selected_mode"])
        second_components = second.get("candidate_components")
        if not isinstance(second_components, list) or len(second_components) != 20:
            raise RuntimeError(f"alternate artifact lacks candidate components: {token}")
        second_reward = second["candidate_rewards"][mode]
        if second_reward is None:
            raise RuntimeError(f"alternate anchor reward invalid: {token}/{mode}")
        base_reward = 0.5 * (float(first["selected_reward"]) + float(second_reward))
        component_means = {}
        for name, index in COMPONENT_INDEX.items():
            component_means[name] = 0.5 * (
                float(first["selected_components"][name])
                + float(second_components[mode][index])
            )
        log_name = mapping[token]
        difficulty = 1.0 + 2.0 * min(max((0.75 - base_reward) / 0.75, 0.0), 1.0)
        raw_weights.append(difficulty / log_counts[log_name])
        records.append(
            {
                "token": token,
                "log_name": log_name,
                "split": "train",
                "selected_mode": mode,
                "base_reward_mean": base_reward,
                "base_collision_mean": component_means["collision"],
                "base_drivable_mean": component_means["drivable"],
                "base_ttc_mean": component_means["ttc"],
                "log_frequency": log_counts[log_name],
                "difficulty_weight": difficulty,
                "safety_failure": any(value < 1.0 - 1e-6 for value in component_means.values()),
                "bc_weight": min(
                    0.3,
                    0.1 if base_reward <= 0.75
                    else 0.1 + 0.2 * (base_reward - 0.75) / 0.25,
                ),
            }
        )
    scene_weights = bounded_unit_mean(raw_weights)
    for record, weight in zip(records, scene_weights):
        record["scene_weight"] = float(weight)

    by_log = defaultdict(list)
    for record in records:
        by_log[record["log_name"]].append(record)
    if len(by_log) != 908:
        raise RuntimeError(f"expected 908 logs, got {len(by_log)}")
    folds, audit, selected_trial = build_folds(by_log, args.seed)
    fold_records = [[record for group in fold.values() for record in group] for fold in folds]
    for fold_index, group in enumerate(fold_records):
        for record in group:
            record["stage21_fold"] = fold_index

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_base = {
        "selection": "whole_log_balanced_six_fold",
        "seed": args.seed,
        "stage21_objective": "base_anchored_log_robust_v1",
        "base_checkpoint_sha256": BASE_SHA,
        "schedule": SCHEDULE,
        "group_definition": "same_token_same_selected_anchor",
        "group_size": 8,
        "base_noise_namespaces": [-1, 20260728],
        "default_base_artifact_sha256": file_sha256(args.default_artifact),
        "alternate_base_artifact_sha256": file_sha256(args.alternate_artifact),
        "token_log_manifest_sha256": file_sha256(args.token_log_manifest),
    }
    subsets = {
        "all": records,
        "fit": sum(fold_records[:4], []),
        "calibration": fold_records[4],
        "internal_test": fold_records[5],
        "smoke64": sorted(sum(fold_records[:4], []), key=lambda record: record["token"])[:64],
    }
    for index, group in enumerate(fold_records):
        subsets[f"fold{index}"] = group
    for name, group in subsets.items():
        write_manifest(args.output_dir / f"{name}_manifest.json", name, group, summary_base)
    audit_payload = {
        "summary": {**summary_base, "selected_assignment_trial": selected_trial},
        "balance": audit,
        "fold_token_sha256": [ordered_sha(sorted(record["token"] for record in group)) for group in fold_records],
        "zero_log_overlap": len(set().union(*(set(fold) for fold in folds))) == sum(len(fold) for fold in folds),
        "scene_weight_mean": float(np.mean(scene_weights)),
        "scene_weight_min": float(np.min(scene_weights)),
        "scene_weight_max": float(np.max(scene_weights)),
    }
    if not audit_payload["zero_log_overlap"] or not audit["passes"]:
        raise RuntimeError(f"Stage21 fold audit failed: {audit_payload}")
    (args.output_dir / "fold_audit.json").write_text(
        json.dumps(audit_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit_payload, indent=2))


if __name__ == "__main__":
    main()
