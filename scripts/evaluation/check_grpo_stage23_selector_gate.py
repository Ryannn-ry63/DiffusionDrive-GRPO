#!/usr/bin/env python3
"""Independent Stage21-fold4 diagnostic gate for the frozen Stage23 selector."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.evaluation.calibrate_grpo_stage23_selector import clopper_pearson_upper


SAFETY_INDICES = (0, 1, 3)


def _bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 500):
        count = min(500, samples - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means[start : start + count] = values[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if not calibration.get("passed"):
        raise RuntimeError("Stage23 calibration did not pass")
    expected_selector_sha = calibration["selector_checkpoint_sha256"]

    pooled_delta = []
    pooled_components = []
    namespace_results = []
    catastrophes = 0
    seen_namespaces = set()
    artifact_provenance = []
    for path in args.artifact:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        selector_summary = summary.get("stage23_selector", {})
        if selector_summary.get("checkpoint_sha256") != expected_selector_sha:
            raise RuntimeError("diagnostic selector SHA differs from calibration")
        namespace = int(summary.get("evaluation_noise_namespace", -999))
        if namespace in seen_namespaces:
            raise RuntimeError(f"duplicate Stage23 diagnostic namespace: {namespace}")
        seen_namespaces.add(namespace)
        deltas = []
        components = []
        namespace_catastrophes = 0
        for record in payload.get("records", []):
            diagnostic = record.get("stage23_selector")
            if diagnostic is None:
                raise RuntimeError("diagnostic artifact lacks Stage23 records")
            fallback = int(diagnostic["fallback_mode"])
            selected = int(record["selected_mode"])
            rewards = np.asarray(record["candidate_rewards"], dtype=float)
            candidate_components = np.asarray(record["candidate_components"], dtype=float)
            delta = float(rewards[selected] - rewards[fallback])
            component_delta = candidate_components[selected] - candidate_components[fallback]
            catastrophic = bool(
                selected != fallback
                and (
                    delta <= -0.5
                    or np.any(component_delta[list(SAFETY_INDICES)] < -0.001)
                )
            )
            deltas.append(delta)
            components.append(component_delta)
            namespace_catastrophes += int(catastrophic)
        if not deltas:
            raise RuntimeError(f"empty Stage23 diagnostic artifact: {path}")
        delta_array = np.asarray(deltas)
        component_array = np.asarray(components)
        pooled_delta.extend(deltas)
        pooled_components.extend(components)
        catastrophes += namespace_catastrophes
        namespace_results.append(
            {
                "namespace": namespace,
                "count": len(deltas),
                "mean_gain": float(delta_array.mean()),
                "safety_component_mean_deltas": component_array[:, SAFETY_INDICES]
                .mean(axis=0).tolist(),
                "catastrophic_count": namespace_catastrophes,
            }
        )
        artifact_provenance.append({"path": str(path), "sha256": _sha256(path)})

    pooled_delta = np.asarray(pooled_delta, dtype=np.float64)
    pooled_components = np.asarray(pooled_components, dtype=np.float64)
    ci = _bootstrap_ci(
        pooled_delta, args.bootstrap_samples, args.bootstrap_seed
    )
    safety_means = pooled_components[:, SAFETY_INDICES].mean(axis=0)
    catastrophic_upper = clopper_pearson_upper(catastrophes, pooled_delta.size)
    checks = {
        "pooled_gain_at_least_0.003": bool(pooled_delta.mean() >= 0.003),
        "pooled_ci_strictly_positive": bool(ci[0] > 0),
        "every_namespace_nonnegative": bool(
            all(item["mean_gain"] >= 0 for item in namespace_results)
        ),
        "safety_components_no_worse_0.001": bool(np.all(safety_means >= -0.001)),
        "catastrophic_cp_upper_at_most_0.005": bool(catastrophic_upper <= 0.005),
    }
    result = {
        "schema_version": 1,
        "passed": bool(all(checks.values())),
        "stop_before_generator_training": bool(not all(checks.values())),
        "selector_checkpoint_sha256": expected_selector_sha,
        "calibration_path": str(args.calibration),
        "calibration_sha256": _sha256(args.calibration),
        "count": int(pooled_delta.size),
        "pooled_mean_gain": float(pooled_delta.mean()),
        "pooled_bootstrap_ci": ci,
        "safety_component_mean_deltas": safety_means.tolist(),
        "catastrophic_count": catastrophes,
        "catastrophic_cp_upper": catastrophic_upper,
        "checks": checks,
        "namespaces": namespace_results,
        "artifacts": artifact_provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
