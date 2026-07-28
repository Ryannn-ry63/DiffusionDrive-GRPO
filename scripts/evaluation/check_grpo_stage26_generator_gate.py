#!/usr/bin/env python3
"""Final fixed-epoch fold4 C-B revalidation for the Stage26 selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluation.check_grpo_stage25_generator_gate import (
    BASE_SHA,
    NOISES,
    load_artifact,
    load_manifest,
    metrics_for_epoch,
    sha256,
)


STAGE25_EPOCH2_SHA = (
    "8a273860a365b2ca0eb6f97fddd8d381714cf7f4a30ceb08267aed1756413989"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--base", action="append", nargs=2, required=True)
    parser.add_argument("--candidate", action="append", nargs=2, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if (
        calibration.get("stage") != 25
        or calibration.get("closure_stage") != 26
        or not calibration.get("passed")
        or calibration.get("num_artifacts") != 10
        or calibration.get("num_scenes") != 10210
    ):
        raise RuntimeError("Stage26 requires the passed ten-cell fold4 calibration")
    calibration_sha = sha256(args.calibration)
    selector_sha = str(calibration["selector_checkpoint_sha256"])
    ood_threshold = float(calibration["ood_threshold"])
    if sha256(args.candidate_checkpoint) != STAGE25_EPOCH2_SHA:
        raise RuntimeError("Stage26 generator checkpoint SHA drifted")

    tokens, logs = load_manifest(args.manifest)
    base_paths = {int(noise): Path(path) for noise, path in args.base}
    candidate_paths = {
        int(noise): Path(path) for noise, path in args.candidate
    }
    if set(base_paths) != set(NOISES) or set(candidate_paths) != set(NOISES):
        raise RuntimeError("Stage26 deployment noise grid drifted")
    bases = {
        noise: load_artifact(
            base_paths[noise], tokens, noise, BASE_SHA, "official_base",
            calibration_sha, selector_sha,
        )
        for noise in NOISES
    }
    candidates = {
        noise: load_artifact(
            candidate_paths[noise], tokens, noise, STAGE25_EPOCH2_SHA,
            "stage25_epoch2", calibration_sha, selector_sha,
        )
        for noise in NOISES
    }
    metrics = metrics_for_epoch(
        2, tokens, logs, bases, candidates, ood_threshold,
        args.bootstrap_samples,
    )
    checks = {
        "every_namespace_positive": all(
            value > 0 for value in metrics["namespace_c_minus_b"].values()
        ),
        "pooled_gain_at_least_0.005": metrics["pooled_c_minus_b"] >= 0.005,
        "whole_log_ci_positive": metrics["whole_log_bootstrap_ci"][0] > 0,
        "hard_scene_gain_at_least_0.010":
            metrics["hard_scene_c_minus_b"] >= 0.010,
        "mature_scene_gain_at_least_minus_0.002":
            metrics["mature_scene_c_minus_b"] >= -0.002,
        "safety_no_worse_0.001": all(
            value >= -0.001
            for value in metrics["safety_component_mean_deltas"].values()
        ),
        "catastrophic_upper_at_most_0.005":
            metrics["catastrophic_cp_upper"] <= 0.005,
        "candidate_ood_rate_at_most_0.05":
            metrics["candidate_ood_rate"] <= 0.05,
    }
    passed = bool(all(checks.values()))
    result = {
        "schema_version": 1,
        "stage": 26,
        "passed": passed,
        "stop_before_fold5": not passed,
        "selection_scope": "fixed_stage25_epoch2_final_selector_fold4_c_minus_b",
        "checkpoint_sha256": STAGE25_EPOCH2_SHA,
        "selector_checkpoint_sha256": selector_sha,
        "calibration_sha256": calibration_sha,
        "manifest": str(args.manifest),
        "noise_namespaces": list(NOISES),
        "checks": checks,
        "metrics": metrics,
        "base_artifacts": {
            str(noise): str(base_paths[noise]) for noise in NOISES
        },
        "candidate_artifacts": {
            str(noise): str(candidate_paths[noise]) for noise in NOISES
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
