#!/usr/bin/env python3
"""Strict consumed-fold4 gate for frozen Stage24/25 selectors."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.evaluation.calibrate_grpo_stage24_selector import (
    clopper_pearson_upper,
    whole_log_bootstrap_ci,
)


SAFETY_INDICES = (0, 1, 3)
KNOWN_STAGE23_HARD_NEGATIVE = "d6d624b818c05333"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260822)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if not calibration.get("passed"):
        raise RuntimeError("Stage24 calibration did not pass")
    expected_sha = calibration["selector_checkpoint_sha256"]
    expected_stage = int(calibration.get("stage", 24))
    if expected_stage not in (24, 25):
        raise RuntimeError("unsupported selector stage")
    summary_key = f"stage{expected_stage}_selector"

    deltas = []
    logs = []
    components = []
    groups = defaultdict(list)
    domains = defaultdict(list)
    catastrophes = 0
    hard_negative_seen = 0
    hard_negative_switched = 0
    provenance = []
    seen_combinations = set()
    for path in args.artifact:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        selector = summary.get(summary_key, {})
        if selector.get("checkpoint_sha256") != expected_sha:
            raise RuntimeError("Stage24 fold4 selector SHA mismatch")
        if selector.get("calibration_sha256") != sha256(args.calibration):
            raise RuntimeError("Stage24 fold4 calibration SHA mismatch")
        domain = str(summary.get("generator_domain", ""))
        namespace = int(summary.get("evaluation_noise_namespace", -999))
        combination = (domain, namespace)
        if not domain or combination in seen_combinations:
            raise RuntimeError("Stage24 fold4 domain/namespace provenance invalid")
        seen_combinations.add(combination)
        for record in payload.get("records", []):
            diagnostic = record.get("stage24_selector")
            rewards = np.asarray(record.get("candidate_rewards"), dtype=np.float64)
            candidate_components = np.asarray(
                record.get("candidate_components"), dtype=np.float64
            )
            if diagnostic is None or rewards.shape != (20,) or candidate_components.shape != (20, 6):
                raise RuntimeError("Stage24 fold4 record lacks all-20 labels")
            fallback = int(diagnostic["fallback_mode"])
            selected = int(record["selected_mode"])
            switched = bool(diagnostic["switched"])
            delta = float(rewards[selected] - rewards[fallback])
            component_delta = candidate_components[selected] - candidate_components[fallback]
            catastrophic = switched and (
                delta <= -0.5
                or np.any(component_delta[list(SAFETY_INDICES)] < -0.001)
            )
            token = str(record.get("token", ""))
            if token == KNOWN_STAGE23_HARD_NEGATIVE:
                hard_negative_seen += 1
                hard_negative_switched += int(switched)
            deltas.append(delta)
            logs.append(str(record.get("log_name", token)))
            components.append(component_delta)
            groups[combination].append(delta)
            domains[domain].append(delta)
            catastrophes += int(catastrophic)
        provenance.append({"path": str(path), "sha256": sha256(path)})

    delta_array = np.asarray(deltas, dtype=np.float64)
    component_array = np.asarray(components, dtype=np.float64)
    if not delta_array.size:
        raise RuntimeError("Stage24 fold4 gate has no records")
    ci = whole_log_bootstrap_ci(
        delta_array, logs, args.bootstrap_samples, args.bootstrap_seed
    )
    group_means = {
        f"{domain}:{namespace}": float(np.mean(values))
        for (domain, namespace), values in sorted(groups.items())
    }
    domain_means = {
        domain: float(np.mean(values)) for domain, values in sorted(domains.items())
    }
    stage21_values = [
        values for domain, values in domains.items() if "stage21" in domain.lower()
    ]
    stage21_mean = (
        float(np.mean(np.concatenate(stage21_values))) if stage21_values else float("nan")
    )
    safety_means = component_array[:, SAFETY_INDICES].mean(axis=0)
    catastrophic_upper = clopper_pearson_upper(catastrophes, delta_array.size)
    checks = {
        "pooled_gain_at_least_0.003": bool(delta_array.mean() >= 0.003),
        "whole_log_ci_strictly_positive": bool(ci[0] > 0),
        "stage21_domain_gain_at_least_0.003": bool(stage21_mean >= 0.003),
        "every_domain_namespace_nonnegative": bool(
            all(value >= 0 for value in group_means.values())
        ),
        "safety_components_no_worse_0.001": bool(np.all(safety_means >= -0.001)),
        "catastrophic_upper_at_most_0.005": bool(catastrophic_upper <= 0.005),
    }
    if expected_stage == 24:
        checks.update({
            "known_hard_negative_seen": bool(hard_negative_seen > 0),
            "known_hard_negative_falls_back": bool(
                hard_negative_seen > 0 and hard_negative_switched == 0
            ),
        })
    result = {
        "schema_version": 1,
        "stage": expected_stage,
        "passed": bool(all(checks.values())),
        "stop_before_generator_training": bool(not all(checks.values())),
        "selector_checkpoint_sha256": expected_sha,
        "calibration_path": str(args.calibration),
        "calibration_sha256": sha256(args.calibration),
        "count": int(delta_array.size),
        "pooled_mean_gain": float(delta_array.mean()),
        "whole_log_bootstrap_ci": ci,
        "group_means": group_means,
        "domain_means": domain_means,
        "stage21_domain_mean": stage21_mean,
        "safety_component_mean_deltas": safety_means.tolist(),
        "catastrophic_count": catastrophes,
        "catastrophic_cp_upper": catastrophic_upper,
        "known_hard_negative_seen": hard_negative_seen,
        "known_hard_negative_switched": hard_negative_switched,
        "checks": checks,
        "artifacts": provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
