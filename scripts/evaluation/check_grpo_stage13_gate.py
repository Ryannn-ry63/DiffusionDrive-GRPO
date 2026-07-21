#!/usr/bin/env python3
"""Apply preregistered Stage-13 frozen-selector confirmation gates."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

try:
    from scripts.evaluation.check_grpo_stage12_gate import (
        FULL_COMPONENTS,
        metric_summary,
        paired_metrics,
        stratified_ci,
        token_clustered_ci,
    )
except ModuleNotFoundError:
    from check_grpo_stage12_gate import (
        FULL_COMPONENTS,
        metric_summary,
        paired_metrics,
        stratified_ci,
        token_clustered_ci,
    )


BASE_CHECKPOINT = "/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model"
SCHEDULE = {
    "truncation_timestep": 8,
    "roll_timesteps": [8, 0],
    "scheduler_num_inference_steps": 125,
}
EXISTING_CHAMPION = 0.0013528957


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate",
        choices=(
            "pilot-dev-select",
            "seed0-fixed1024",
            "three-seed-fixed1024",
            "dev-confirm",
            "full-navtest",
        ),
        required=True,
    )
    parser.add_argument("--baseline-artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--generation-kl-rolling-means", type=float, nargs="*")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load(path: Path) -> Dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("records"), list) or not payload["records"]:
        raise ValueError(f"artifact has no records: {path}")
    return payload


def validate_artifact(
    payload: Dict, label: str, expected_tokens: int, failures: List[str]
) -> None:
    summary = payload.get("summary", {})
    records = payload.get("records", [])
    tokens = [str(record.get("token")) for record in records]
    if len(records) != expected_tokens:
        failures.append(f"{label} does not contain exactly {expected_tokens} records")
    if len(tokens) != len(set(tokens)):
        failures.append(f"{label} contains duplicate tokens")
    digest = hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()
    if summary.get("token_set_sha256") != digest:
        failures.append(f"{label} token-set SHA is missing or incorrect")
    if int(summary.get("num_tokens", -1)) != len(records):
        failures.append(f"{label} summary token count is incorrect")
    if summary.get("selector_logits_source") != "reference":
        failures.append(f"{label} selector source is not reference")
    if summary.get("reference_checkpoint") != BASE_CHECKPOINT:
        failures.append(f"{label} reference checkpoint is not the locked base")
    if summary.get("completed") is not True or int(summary.get("num_failures", -1)) != 0:
        failures.append(f"{label} is incomplete or reports failures")
    if int(summary.get("requested_limit", -1)) != expected_tokens:
        failures.append(f"{label} requested limit is not {expected_tokens}")
    schedule = summary.get("schedule", {})
    for key, expected in SCHEDULE.items():
        if schedule.get(key) != expected:
            failures.append(f"{label} has unregistered schedule field {key}")

    required = ("selected_reward", "candidate_reward", "oracle_reward")
    for index, record in enumerate(records):
        if record.get("selector_logits_source") != "reference":
            failures.append(f"{label} record {index} selector source is not reference")
            break
        values = []
        try:
            values.extend(float(record[name]) for name in required)
            components = record["selected_components"]
            values.extend(float(components[name]) for name in FULL_COMPONENTS)
        except (KeyError, TypeError, ValueError):
            failures.append(f"{label} record {index} is incomplete")
            break
        if not all(math.isfinite(value) for value in values):
            failures.append(f"{label} record {index} contains NaN/Inf")
            break


def _validate_pair(
    baseline: Dict, candidate: Dict, label: str, expected_tokens: int, failures: List[str]
) -> None:
    validate_artifact(baseline, f"{label} baseline", expected_tokens, failures)
    validate_artifact(candidate, f"{label} candidate", expected_tokens, failures)


def evaluate_single(
    gate: str,
    baseline: Dict,
    candidate: Dict,
    rolling_kl: Sequence[float],
    samples: int,
    seed: int,
) -> Dict:
    expected_tokens = 3072 if gate == "pilot-dev-select" else 1024
    failures: List[str] = []
    _validate_pair(baseline, candidate, gate, expected_tokens, failures)
    metric = paired_metrics(baseline, candidate, samples, seed)
    threshold = 0.0015 if gate == "pilot-dev-select" else 0.002
    if metric["selected_mean"] < threshold:
        failures.append(f"selected delta is < +{threshold:g}")
    if metric["selected_ci95"][0] <= 0:
        failures.append("selected CI95 lower bound is not > 0")
    for name in FULL_COMPONENTS:
        if metric["component_differences"][name] < 0:
            failures.append(f"{name} delta is < 0")
    if metric["minimum_token_delta"] <= -0.5:
        failures.append("worst selected-token delta is not > -0.5")

    kl_value = None
    if gate == "seed0-fixed1024":
        if len(rolling_kl) != 1 or not math.isfinite(rolling_kl[0]):
            failures.append("one finite rolling generation KL is required")
        else:
            kl_value = float(rolling_kl[0])
            if kl_value > 5e-4:
                failures.append("rolling generation KL is > 5e-4")
    elif rolling_kl:
        failures.append("pilot gate does not accept a KL override")

    return {
        "gate": gate,
        "passed": not failures,
        "num_tokens": len(baseline["records"]),
        "rolling_generation_kl": kl_value,
        "versus_base": metric_summary(metric),
        "failures": failures,
    }


def evaluate_multi(
    gate: str,
    baselines: Sequence[Dict],
    candidates: Sequence[Dict],
    rolling_kl: Sequence[float],
    samples: int,
    seed: int,
) -> Dict:
    if len(candidates) != 3 or len(baselines) not in (1, 3):
        raise ValueError("multi-seed gates require three candidates and one or three baselines")
    expected_tokens = 12146 if gate == "full-navtest" else 1024
    expanded_baselines = list(baselines) if len(baselines) == 3 else list(baselines) * 3
    failures: List[str] = []
    metrics = []
    for index, (baseline, candidate) in enumerate(zip(expanded_baselines, candidates)):
        _validate_pair(baseline, candidate, f"seed {index}", expected_tokens, failures)
        metrics.append(paired_metrics(baseline, candidate, samples, seed + index))

    matrix = np.stack([metric["selected"] for metric in metrics])
    individual_means = [metric["selected_mean"] for metric in metrics]
    individual_cis = [metric["selected_ci95"] for metric in metrics]
    mean_delta = float(matrix.mean())
    stratified = stratified_ci(matrix, samples, seed + 100)
    clustered = token_clustered_ci(matrix, samples, seed + 200)
    mean_components = {
        name: float(np.mean([metric["component_differences"][name] for metric in metrics]))
        for name in FULL_COMPONENTS
    }

    if not all(value > 0 for value in individual_means):
        failures.append("not every seed has positive selected delta")
    required_ci_count = sum(ci[0] > 0 for ci in individual_cis)
    if gate == "three-seed-fixed1024":
        if mean_delta < 0.002:
            failures.append("three-seed mean selected delta is < +0.002")
        if stratified[0] <= 0:
            failures.append("seed-stratified CI95 lower bound is not > 0")
        if required_ci_count < 2:
            failures.append("fewer than two individual CI95 lower bounds are > 0")
        if len(rolling_kl) != 3 or not all(math.isfinite(value) for value in rolling_kl):
            failures.append("three finite rolling generation KL values are required")
        elif any(value > 5e-4 for value in rolling_kl):
            failures.append("a rolling generation KL is > 5e-4")
    elif gate == "dev-confirm":
        if stratified[0] < 0:
            failures.append("seed-stratified CI95 lower bound is < 0")
        if rolling_kl:
            failures.append("dev-confirm gate does not accept KL overrides")
    elif gate == "full-navtest":
        if required_ci_count < 2:
            failures.append("fewer than two individual CI95 lower bounds are > 0")
        if stratified[0] <= 0:
            failures.append("seed-stratified CI95 lower bound is not > 0")
        if clustered[0] <= 0:
            failures.append("token-clustered CI95 lower bound is not > 0")
        if rolling_kl:
            failures.append("full-navtest gate does not accept KL overrides")
    else:
        raise ValueError(f"unsupported multi-seed gate: {gate}")

    for name, value in mean_components.items():
        if value < 0:
            failures.append(f"mean {name} delta is < 0")
    if gate != "full-navtest" and any(
        metric["minimum_token_delta"] <= -0.5 for metric in metrics
    ):
        failures.append("a worst selected-token delta is not > -0.5")

    return {
        "gate": gate,
        "passed": not failures,
        "num_tokens_per_seed": expected_tokens,
        "seed_metrics": [metric_summary(metric) for metric in metrics],
        "individual_selected_differences": individual_means,
        "mean_selected_difference": mean_delta,
        "seed_stratified_ci95": stratified,
        "token_clustered_ci95": clustered,
        "mean_component_differences": mean_components,
        "rolling_generation_kl": list(rolling_kl),
        "result_classification": {
            "scientific_success": not failures,
            "beats_existing_champion": mean_delta > EXISTING_CHAMPION,
            "strong_at_least_0.003": mean_delta >= 0.003,
            "stretch_at_least_0.005": mean_delta >= 0.005,
        },
        "failures": failures,
    }


def main() -> None:
    args = parse_args()
    baselines = [load(path) for path in args.baseline_artifacts]
    candidates = [load(path) for path in args.candidate_artifacts]
    rolling_kl = args.generation_kl_rolling_means or []
    if args.gate in {"pilot-dev-select", "seed0-fixed1024"}:
        if len(baselines) != 1 or len(candidates) != 1:
            raise ValueError("single-seed gate requires one baseline and one candidate")
        result = evaluate_single(
            args.gate, baselines[0], candidates[0], rolling_kl,
            args.bootstrap_samples, args.bootstrap_seed,
        )
    else:
        result = evaluate_multi(
            args.gate, baselines, candidates, rolling_kl,
            args.bootstrap_samples, args.bootstrap_seed,
        )
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
