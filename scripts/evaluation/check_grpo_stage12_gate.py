#!/usr/bin/env python3
"""Apply preregistered Stage-12 frozen-selector continuation gates."""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


SAFETY_COMPONENTS = ("collision", "drivable", "ttc")
FULL_COMPONENTS = ("collision", "drivable", "progress", "ttc")
DIAGNOSTIC_STEPS = (128, 512, 1024, 2048)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate",
        choices=(
            "diagnostic",
            "seed0-fixed1024",
            "three-seed-fixed1024",
            "dev-select",
            "dev-confirm",
            "full-navtest",
        ),
        required=True,
    )
    parser.add_argument("--baseline-artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--stage10-u128-artifact", type=Path)
    parser.add_argument("--candidate-steps", type=int, nargs="*")
    parser.add_argument("--generation-kl-rolling-means", type=float, nargs="*")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load(path: Path) -> Dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"artifact has no records: {path}")
    tokens = [str(record["token"]) for record in records]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"artifact contains duplicate tokens: {path}")
    summary = payload.get("summary", {})
    if int(summary.get("num_tokens", len(records))) != len(records):
        raise ValueError(f"artifact summary count mismatch: {path}")
    return payload


def align_pair(baseline: Dict, candidate: Dict):
    base_records = baseline["records"]
    candidate_records = candidate["records"]
    base_tokens = [str(record["token"]) for record in base_records]
    candidate_tokens = [str(record["token"]) for record in candidate_records]
    if candidate_tokens != base_tokens:
        raise RuntimeError("paired artifacts have different token order")
    base_sha = baseline.get("summary", {}).get("token_set_sha256")
    candidate_sha = candidate.get("summary", {}).get("token_set_sha256")
    if not base_sha or candidate_sha != base_sha:
        raise RuntimeError("paired artifacts have different token-set SHA")
    return base_records, candidate_records


def field(records: Sequence[Dict], name: str) -> np.ndarray:
    return np.asarray([float(record[name]) for record in records], dtype=np.float64)


def component(records: Sequence[Dict], name: str) -> np.ndarray:
    return np.asarray(
        [float(record["selected_components"][name]) for record in records],
        dtype=np.float64,
    )


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> List[float]:
    if samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        size = min(1000, samples - start)
        indices = rng.integers(0, values.size, size=(size, values.size))
        means[start : start + size] = values[indices].mean(axis=1)
    return np.quantile(means, (0.025, 0.975)).tolist()


def stratified_ci(matrix: np.ndarray, samples: int, seed: int) -> List[float]:
    rng = np.random.default_rng(seed)
    seed_count, token_count = matrix.shape
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 200):
        size = min(200, samples - start)
        sampled_mean = np.zeros(size, dtype=np.float64)
        for seed_index in range(seed_count):
            indices = rng.integers(0, token_count, size=(size, token_count))
            sampled_mean += matrix[seed_index, indices].mean(axis=1)
        means[start : start + size] = sampled_mean / seed_count
    return np.quantile(means, (0.025, 0.975)).tolist()


def token_clustered_ci(matrix: np.ndarray, samples: int, seed: int) -> List[float]:
    rng = np.random.default_rng(seed)
    token_count = matrix.shape[1]
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 200):
        size = min(200, samples - start)
        indices = rng.integers(0, token_count, size=(size, token_count))
        means[start : start + size] = matrix[:, indices].mean(axis=(0, 2))
    return np.quantile(means, (0.025, 0.975)).tolist()


def paired_metrics(baseline: Dict, candidate: Dict, samples: int, seed: int) -> Dict:
    base, cand = align_pair(baseline, candidate)
    selected = field(cand, "selected_reward") - field(base, "selected_reward")
    candidate_delta = field(cand, "candidate_reward") - field(base, "candidate_reward")
    oracle = field(cand, "oracle_reward") - field(base, "oracle_reward")
    component_differences = {
        name: float((component(cand, name) - component(base, name)).mean())
        for name in FULL_COMPONENTS
    }
    safety_mask = np.ones(len(base), dtype=bool)
    for name in SAFETY_COMPONENTS:
        safety_mask &= component(base, name) >= 1.0 - 1e-6
    return {
        "selected": selected,
        "candidate": candidate_delta,
        "oracle": oracle,
        "selected_mean": float(selected.mean()),
        "selected_ci95": bootstrap_ci(selected, samples, seed),
        "candidate_mean": float(candidate_delta.mean()),
        "oracle_mean": float(oracle.mean()),
        "minimum_token_delta": float(selected.min()),
        "component_differences": component_differences,
        "safety_pass_tokens": int(safety_mask.sum()),
        "safety_pass_delta": (
            float(selected[safety_mask].mean()) if safety_mask.any() else None
        ),
    }


def metric_summary(metric: Dict) -> Dict:
    return {
        key: value
        for key, value in metric.items()
        if key not in {"selected", "candidate", "oracle"}
    }


def require_reference_selector(payload: Dict, label: str, failures: List[str]) -> None:
    source = payload.get("summary", {}).get("selector_logits_source")
    if source != "reference":
        failures.append(f"{label} selector source is not reference")


def evaluate_diagnostic(
    baseline: Dict,
    stage10_u128: Dict,
    candidates: Sequence[Dict],
    candidate_steps: Sequence[int],
    samples: int,
    seed: int,
) -> Dict:
    if tuple(candidate_steps) != DIAGNOSTIC_STEPS or len(candidates) != len(
        DIAGNOSTIC_STEPS
    ):
        raise ValueError("diagnostic requires Stage-9 steps 128, 512, 1024, 2048")
    if len(baseline["records"]) != 1024:
        raise ValueError("diagnostic requires a 1024-token baseline")

    failures: List[str] = []
    require_reference_selector(baseline, "baseline", failures)
    require_reference_selector(stage10_u128, "Stage-10 U128", failures)
    u128_metric = paired_metrics(baseline, stage10_u128, samples, seed + 1)
    checkpoint_results = []
    eligible_steps = []
    for index, (step, candidate) in enumerate(zip(candidate_steps, candidates)):
        require_reference_selector(candidate, f"Stage-9 U{step}", failures)
        metric = paired_metrics(baseline, candidate, samples, seed + 10 + index)
        versus_u128 = paired_metrics(
            stage10_u128, candidate, samples, seed + 20 + index
        )
        reasons = []
        if step < 512:
            reasons.append("checkpoint is below the registered U512+ evidence range")
        if metric["selected_mean"] < 0.003:
            reasons.append("selected delta is < +0.003")
        if metric["selected_ci95"][0] <= 0:
            reasons.append("selected CI lower bound is not > 0")
        if versus_u128["selected_mean"] < 0.0005:
            reasons.append("selected improvement over Stage-10 U128 is < +0.0005")
        if metric["candidate_mean"] < u128_metric["candidate_mean"]:
            reasons.append("candidate mean is below Stage-10 U128")
        if metric["oracle_mean"] < 0:
            reasons.append("oracle delta is < 0")
        for name in SAFETY_COMPONENTS:
            if metric["component_differences"][name] < -0.001:
                reasons.append(f"{name} delta is < -0.001")
        if metric["minimum_token_delta"] <= -0.5:
            reasons.append("worst token delta is not > -0.5")
        if not reasons:
            eligible_steps.append(step)
        checkpoint_results.append(
            {
                "step": step,
                "eligible": not reasons,
                "versus_base": metric_summary(metric),
                "versus_stage10_u128": metric_summary(versus_u128),
                "failures": reasons,
            }
        )
    if not eligible_steps:
        failures.append("no Stage-9 U512+ checkpoint demonstrates continuation headroom")
    return {
        "gate": "stage12_diagnostic",
        "passed": not failures,
        "num_tokens": len(baseline["records"]),
        "stage10_u128": metric_summary(u128_metric),
        "eligible_stage9_steps": eligible_steps,
        "checkpoints": checkpoint_results,
        "failures": failures,
    }


def evaluate_seed0(
    baseline: Dict,
    stage10_u128: Dict,
    candidate: Dict,
    rolling_kl: Sequence[float],
    samples: int,
    seed: int,
) -> Dict:
    failures: List[str] = []
    for label, payload in (
        ("baseline", baseline),
        ("Stage-10 U128", stage10_u128),
        ("Stage-12 U256", candidate),
    ):
        require_reference_selector(payload, label, failures)
    metric = paired_metrics(baseline, candidate, samples, seed)
    versus_u128 = paired_metrics(stage10_u128, candidate, samples, seed + 1)
    if len(metric["selected"]) != 1024:
        failures.append("seed0 fixed gate requires exactly 1024 tokens")
    if len(rolling_kl) != 1 or not math.isfinite(rolling_kl[0]):
        failures.append("one finite rolling KL value is required")
        kl = None
    else:
        kl = float(rolling_kl[0])
        if kl > 2.5e-4:
            failures.append("rolling KL is > 2.5e-4")
    if metric["selected_mean"] < 0.003:
        failures.append("selected delta versus base is < +0.003")
    if metric["selected_ci95"][0] <= 0:
        failures.append("selected CI lower bound versus base is not > 0")
    if versus_u128["selected_mean"] < 0.0005:
        failures.append("selected improvement over U128 is < +0.0005")
    if versus_u128["selected_ci95"][0] < 0:
        failures.append("U256-versus-U128 CI lower bound is < 0")
    if metric["candidate_mean"] < 0 or metric["oracle_mean"] < 0:
        failures.append("candidate mean or oracle delta versus base is < 0")
    if versus_u128["candidate_mean"] < 0:
        failures.append("candidate mean is below U128")
    for name in SAFETY_COMPONENTS:
        if metric["component_differences"][name] < -0.0005:
            failures.append(f"{name} delta is < -0.0005")
    if metric["safety_pass_delta"] is None or metric["safety_pass_delta"] < -0.001:
        failures.append("safety-pass bucket delta is < -0.001 or unavailable")
    if metric["minimum_token_delta"] <= -0.5:
        failures.append("worst token delta is not > -0.5")
    return {
        "gate": "stage12_seed0_fixed1024",
        "passed": not failures,
        "num_tokens": len(metric["selected"]),
        "versus_base": metric_summary(metric),
        "versus_stage10_u128": metric_summary(versus_u128),
        "generation_kl_rolling_mean": kl,
        "failures": failures,
    }


def evaluate_multi(
    gate: str,
    baselines: Sequence[Dict],
    candidates: Sequence[Dict],
    rolling_kls: Sequence[float],
    samples: int,
    seed: int,
) -> Dict:
    if len(baselines) == 1 and len(candidates) > 1:
        baselines = list(baselines) * len(candidates)
    if len(baselines) != 3 or len(candidates) != 3:
        raise ValueError(f"{gate} requires exactly three paired seeds")
    failures: List[str] = []
    for index, (baseline, candidate) in enumerate(zip(baselines, candidates)):
        require_reference_selector(baseline, f"baseline seed {index}", failures)
        require_reference_selector(candidate, f"candidate seed {index}", failures)
    metrics = [
        paired_metrics(base, candidate, samples, seed + index)
        for index, (base, candidate) in enumerate(zip(baselines, candidates))
    ]
    matrix = np.stack([metric["selected"] for metric in metrics])
    expected_tokens = {
        "three-seed-fixed1024": 1024,
        "dev-select": 3072,
        "dev-confirm": 1024,
        "full-navtest": 12146,
    }[gate]
    if matrix.shape[1] != expected_tokens:
        failures.append(f"{gate} requires {expected_tokens} tokens per seed")
    seed_means = matrix.mean(axis=1)
    mean_selected = float(matrix.mean())
    component_means = {
        name: float(
            np.mean([metric["component_differences"][name] for metric in metrics])
        )
        for name in FULL_COMPONENTS
    }
    candidate_mean = float(np.mean([metric["candidate_mean"] for metric in metrics]))
    oracle_mean = float(np.mean([metric["oracle_mean"] for metric in metrics]))
    seed_stratified = stratified_ci(matrix, samples, seed + 100)
    token_clustered = token_clustered_ci(matrix, samples, seed + 200)
    if np.any(seed_means <= 0):
        failures.append("not every seed selected delta is positive")

    if gate == "three-seed-fixed1024":
        if len(rolling_kls) != 3 or any(
            not math.isfinite(value) or value > 2.5e-4 for value in rolling_kls
        ):
            failures.append("each seed requires finite rolling KL <= 2.5e-4")
        if mean_selected < 0.003:
            failures.append("three-seed mean selected delta is < +0.003")
        if seed_stratified[0] <= 0:
            failures.append("seed-stratified CI lower bound is not > 0")
        if candidate_mean < 0 or oracle_mean < 0:
            failures.append("candidate mean or oracle mean is < 0")
        for name in SAFETY_COMPONENTS:
            if component_means[name] < 0:
                failures.append(f"mean {name} delta is < 0")
    elif gate == "dev-select":
        if mean_selected < 0.003:
            failures.append("dev-select mean selected delta is < +0.003")
        if seed_stratified[0] <= 0:
            failures.append("dev-select CI lower bound is not > 0")
        for name in SAFETY_COMPONENTS:
            if component_means[name] < 0:
                failures.append(f"dev-select mean {name} delta is < 0")
    elif gate == "dev-confirm":
        if mean_selected <= 0:
            failures.append("dev-confirm mean selected delta is not > 0")
        if seed_stratified[0] < 0:
            failures.append("dev-confirm CI lower bound is < 0")
        for name in SAFETY_COMPONENTS:
            if component_means[name] < 0:
                failures.append(f"dev-confirm mean {name} delta is < 0")
    else:
        if mean_selected < 0.005:
            failures.append("full-navtest mean selected delta is < +0.005")
        significant_seeds = sum(metric["selected_ci95"][0] > 0 for metric in metrics)
        if significant_seeds < 2:
            failures.append("fewer than two single-seed CI lower bounds are > 0")
        if seed_stratified[0] <= 0:
            failures.append("seed-stratified CI lower bound is not > 0")
        if token_clustered[0] <= 0:
            failures.append("token-clustered CI lower bound is not > 0")
        for name in FULL_COMPONENTS:
            if component_means[name] < 0:
                failures.append(f"full-navtest mean {name} delta is < 0")
        for index, candidate in enumerate(candidates):
            if int(candidate.get("summary", {}).get("num_failures", 0)) != 0:
                failures.append(f"candidate seed {index} has evaluation failures")

    result = {
        "gate": f"stage12_{gate}",
        "passed": not failures,
        "num_seeds": 3,
        "num_tokens_per_seed": int(matrix.shape[1]),
        "seed_selected_differences": seed_means.tolist(),
        "mean_selected_difference": mean_selected,
        "seed_stratified_ci95": seed_stratified,
        "token_clustered_ci95": token_clustered,
        "candidate_mean_difference": candidate_mean,
        "oracle_mean_difference": oracle_mean,
        "component_mean_differences": component_means,
        "single_seed_ci95": [metric["selected_ci95"] for metric in metrics],
        "failures": failures,
    }
    if gate == "three-seed-fixed1024":
        result["generation_kl_rolling_means"] = list(rolling_kls)
    if gate == "full-navtest":
        result["significant_seed_count"] = sum(
            metric["selected_ci95"][0] > 0 for metric in metrics
        )
    return result


def main() -> None:
    args = parse_args()
    baselines = [load(path) for path in args.baseline_artifacts]
    candidates = [load(path) for path in args.candidate_artifacts]
    rolling_kls = args.generation_kl_rolling_means or []
    if args.gate == "diagnostic":
        if len(baselines) != 1 or args.stage10_u128_artifact is None:
            raise ValueError("diagnostic requires one baseline and --stage10-u128-artifact")
        result = evaluate_diagnostic(
            baselines[0],
            load(args.stage10_u128_artifact),
            candidates,
            args.candidate_steps or [],
            args.bootstrap_samples,
            args.bootstrap_seed,
        )
    elif args.gate == "seed0-fixed1024":
        if (
            len(baselines) != 1
            or len(candidates) != 1
            or args.stage10_u128_artifact is None
        ):
            raise ValueError(
                "seed0 gate requires one pair and --stage10-u128-artifact"
            )
        result = evaluate_seed0(
            baselines[0],
            load(args.stage10_u128_artifact),
            candidates[0],
            rolling_kls,
            args.bootstrap_samples,
            args.bootstrap_seed,
        )
    else:
        result = evaluate_multi(
            args.gate,
            baselines,
            candidates,
            rolling_kls,
            args.bootstrap_samples,
            args.bootstrap_seed,
        )

    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
