#!/usr/bin/env python3
"""Apply preregistered Stage-11 frozen-selector evaluation gates."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


SAFETY_COMPONENTS = ("collision", "drivable", "ttc")
FULL_COMPONENTS = ("collision", "drivable", "progress", "ttc")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate",
        choices=(
            "base-equivalence",
            "fixed1024",
            "three-seed-fixed1024",
            "dev-select",
            "dev-confirm",
            "full-navtest",
        ),
        required=True,
    )
    parser.add_argument("--baseline-artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--generation-kl-rolling-means", type=float, nargs="*")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"artifact has no records: {path}")
    tokens = [str(record["token"]) for record in records]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"artifact contains duplicate tokens: {path}")
    if int(payload.get("summary", {}).get("num_tokens", len(records))) != len(records):
        raise ValueError(f"artifact summary count mismatch: {path}")
    return payload


def align_pair(baseline, candidate):
    baseline_records = baseline["records"]
    candidate_records = candidate["records"]
    baseline_tokens = [str(record["token"]) for record in baseline_records]
    candidate_tokens = [str(record["token"]) for record in candidate_records]
    if candidate_tokens != baseline_tokens:
        raise RuntimeError("paired artifacts have different token order")
    baseline_sha = baseline["summary"].get("token_set_sha256")
    candidate_sha = candidate["summary"].get("token_set_sha256")
    if baseline_sha != candidate_sha:
        raise RuntimeError("paired artifacts have different token-set SHA")
    return baseline_records, candidate_records


def field(records, name):
    return np.asarray([float(record[name]) for record in records], dtype=np.float64)


def component(records, name):
    return np.asarray(
        [float(record["selected_components"][name]) for record in records],
        dtype=np.float64,
    )


def bootstrap_ci(values, samples, seed):
    if samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        size = min(1000, samples - start)
        indices = rng.integers(0, values.size, size=(size, values.size))
        means[start : start + size] = values[indices].mean(axis=1)
    return np.quantile(means, (0.025, 0.975)).tolist()


def stratified_ci(matrix, samples, seed):
    """Bootstrap tokens independently inside each fixed seed stratum."""
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


def token_clustered_ci(matrix, samples, seed):
    """Bootstrap whole token clusters, retaining all seed observations."""
    rng = np.random.default_rng(seed)
    token_count = matrix.shape[1]
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 200):
        size = min(200, samples - start)
        indices = rng.integers(0, token_count, size=(size, token_count))
        means[start : start + size] = matrix[:, indices].mean(axis=(0, 2))
    return np.quantile(means, (0.025, 0.975)).tolist()


def paired_metrics(baseline, candidate, samples, seed):
    base, cand = align_pair(baseline, candidate)
    selected = field(cand, "selected_reward") - field(base, "selected_reward")
    candidate_delta = field(cand, "candidate_reward") - field(base, "candidate_reward")
    oracle = field(cand, "oracle_reward") - field(base, "oracle_reward")
    components = {
        name: float((component(cand, name) - component(base, name)).mean())
        for name in set(SAFETY_COMPONENTS + FULL_COMPONENTS)
    }
    safety_mask = np.ones(len(base), dtype=bool)
    for name in SAFETY_COMPONENTS:
        safety_mask &= component(base, name) >= 1.0 - 1e-6
    safety_delta = (
        float(selected[safety_mask].mean()) if safety_mask.any() else None
    )
    return {
        "selected": selected,
        "candidate": candidate_delta,
        "oracle": oracle,
        "selected_mean": float(selected.mean()),
        "selected_ci95": bootstrap_ci(selected, samples, seed),
        "candidate_mean": float(candidate_delta.mean()),
        "oracle_mean": float(oracle.mean()),
        "minimum_token_delta": float(selected.min()),
        "component_differences": components,
        "safety_pass_tokens": int(safety_mask.sum()),
        "safety_pass_delta": safety_delta,
    }


def check_base_equivalence(current, reference):
    current_records, reference_records = align_pair(current, reference)
    failures = []
    current_source = current["summary"].get("selector_logits_source")
    reference_source = reference["summary"].get("selector_logits_source")
    if current_source != "current" or reference_source != "reference":
        failures.append("equivalence inputs must use current then reference selector")
    mode_matches = [
        int(left["selected_mode"]) == int(right["selected_mode"])
        for left, right in zip(current_records, reference_records)
    ]
    trajectory_error = max(
        float(
            np.max(
                np.abs(
                    np.asarray(left["selected_trajectory"], dtype=np.float64)
                    - np.asarray(right["selected_trajectory"], dtype=np.float64)
                )
            )
        )
        for left, right in zip(current_records, reference_records)
    )
    reward_delta = field(reference_records, "selected_reward") - field(
        current_records, "selected_reward"
    )
    if not all(mode_matches):
        failures.append("base current/reference selected modes are not 100% equal")
    if trajectory_error > 1e-6:
        failures.append("base selected trajectory maximum error exceeds 1e-6")
    if not np.array_equal(reward_delta, np.zeros_like(reward_delta)):
        failures.append("base paired selected PDMS difference is not exactly zero")
    return {
        "gate": "stage11_base_equivalence",
        "passed": not failures,
        "num_tokens": len(current_records),
        "selected_mode_agreement": float(np.mean(mode_matches)),
        "selected_trajectory_max_error": trajectory_error,
        "selected_pdms_difference": float(reward_delta.mean()),
        "failures": failures,
    }


def require_reference_selector(payload, label, failures):
    source = payload.get("summary", {}).get("selector_logits_source")
    if source != "reference":
        failures.append(f"{label} selector source is not reference")


def main():
    args = parse_args()
    baselines = [load(path) for path in args.baseline_artifacts]
    candidates = [load(path) for path in args.candidate_artifacts]
    if args.gate == "base-equivalence":
        if len(baselines) != 1 or len(candidates) != 1:
            raise ValueError("base-equivalence requires one current and one reference artifact")
        result = check_base_equivalence(baselines[0], candidates[0])
    else:
        if len(baselines) == 1 and len(candidates) > 1:
            baselines *= len(candidates)
        if len(baselines) != len(candidates):
            raise ValueError("baseline/candidate artifact counts differ")
        failures = []
        for index, candidate in enumerate(candidates):
            require_reference_selector(candidate, f"candidate seed {index}", failures)
        metrics = [
            paired_metrics(base, candidate, args.bootstrap_samples, args.bootstrap_seed + index)
            for index, (base, candidate) in enumerate(zip(baselines, candidates))
        ]
        matrix = np.stack([metric["selected"] for metric in metrics])
        seed_means = matrix.mean(axis=1)
        component_means = {
            name: float(np.mean([metric["component_differences"][name] for metric in metrics]))
            for name in FULL_COMPONENTS
        }
        result = {
            "gate": f"stage11_{args.gate}",
            "passed": False,
            "num_seeds": len(metrics),
            "num_tokens_per_seed": int(matrix.shape[1]),
            "seed_selected_differences": seed_means.tolist(),
            "mean_selected_difference": float(matrix.mean()),
            "component_mean_differences": component_means,
            "failures": failures,
        }

        if args.gate == "fixed1024":
            if len(metrics) != 1 or matrix.shape[1] != 1024:
                failures.append("fixed1024 requires exactly one 1024-token pair")
            metric = metrics[0]
            rolling = args.generation_kl_rolling_means or []
            if len(rolling) != 1 or not math.isfinite(rolling[0]) or rolling[0] > 2.5e-4:
                failures.append("rolling KL is missing, non-finite, or > 2.5e-4")
            if metric["selected_mean"] < 0.003:
                failures.append("selected delta is < +0.003")
            if metric["selected_ci95"][0] <= 0:
                failures.append("selected CI lower bound is not > 0")
            if metric["candidate_mean"] < 0 or metric["oracle_mean"] < 0:
                failures.append("candidate mean or oracle delta is < 0")
            for name in SAFETY_COMPONENTS:
                if metric["component_differences"][name] < -0.0005:
                    failures.append(f"{name} delta is < -0.0005")
            if metric["safety_pass_delta"] is None or metric["safety_pass_delta"] < -0.001:
                failures.append("safety-pass bucket delta is < -0.001 or unavailable")
            if metric["minimum_token_delta"] <= -0.5:
                failures.append("worst token delta is not > -0.5")
            result["single_seed"] = {
                key: value for key, value in metric.items()
                if key not in {"selected", "candidate", "oracle"}
            }
            result["generation_kl_rolling_mean"] = rolling[0] if len(rolling) == 1 else None
        else:
            result["seed_stratified_ci95"] = stratified_ci(
                matrix, args.bootstrap_samples, args.bootstrap_seed + 100
            )
            result["token_clustered_ci95"] = token_clustered_ci(
                matrix, args.bootstrap_samples, args.bootstrap_seed + 200
            )
            if len(metrics) != 3:
                failures.append("multi-seed gates require exactly three seeds")
            if np.any(seed_means <= 0):
                failures.append("not every seed selected delta is positive")
            if args.gate == "three-seed-fixed1024":
                if matrix.shape[1] != 1024:
                    failures.append("three-seed fixed gate requires 1024 tokens per seed")
                rolling = args.generation_kl_rolling_means or []
                if len(rolling) != 3 or any(
                    not math.isfinite(value) or value > 2.5e-4 for value in rolling
                ):
                    failures.append("each seed requires finite rolling KL <= 2.5e-4")
                if matrix.mean() < 0.003:
                    failures.append("three-seed mean selected delta is < +0.003")
                if result["seed_stratified_ci95"][0] <= 0:
                    failures.append("seed-stratified CI lower bound is not > 0")
            elif args.gate == "dev-select":
                if matrix.mean() < 0.003:
                    failures.append("dev-select mean selected delta is < +0.003")
                if result["seed_stratified_ci95"][0] <= 0:
                    failures.append("dev-select CI lower bound is not > 0")
                for name in SAFETY_COMPONENTS:
                    if component_means[name] < 0:
                        failures.append(f"dev-select mean {name} delta is < 0")
            elif args.gate == "dev-confirm":
                if matrix.mean() <= 0:
                    failures.append("dev-confirm mean selected delta is not > 0")
                if result["seed_stratified_ci95"][0] < 0:
                    failures.append("dev-confirm CI lower bound is < 0")
                for name in SAFETY_COMPONENTS:
                    if component_means[name] < 0:
                        failures.append(f"dev-confirm mean {name} delta is < 0")
            elif args.gate == "full-navtest":
                if matrix.shape[1] != 12146:
                    failures.append("full-navtest requires 12,146 tokens per seed")
                if matrix.mean() < 0.005:
                    failures.append("full-navtest mean selected delta is < +0.005")
                significant_seeds = sum(metric["selected_ci95"][0] > 0 for metric in metrics)
                if significant_seeds < 2:
                    failures.append("fewer than two single-seed CI lower bounds are > 0")
                if result["seed_stratified_ci95"][0] <= 0:
                    failures.append("seed-stratified CI lower bound is not > 0")
                if result["token_clustered_ci95"][0] <= 0:
                    failures.append("token-clustered CI lower bound is not > 0")
                for name in FULL_COMPONENTS:
                    if component_means[name] < 0:
                        failures.append(f"full-navtest mean {name} delta is < 0")
                for index, candidate in enumerate(candidates):
                    summary = candidate.get("summary", {})
                    failure_count = int(summary.get("num_failures", 0))
                    if failure_count != 0:
                        failures.append(f"candidate seed {index} has evaluation failures")
                result["single_seed_ci95"] = [metric["selected_ci95"] for metric in metrics]
                result["significant_seed_count"] = significant_seeds
        result["passed"] = not failures

    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
