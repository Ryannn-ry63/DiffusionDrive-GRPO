#!/usr/bin/env python3
"""Apply preregistered Stage-10 paired performance and drift gates."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


SAFETY_COMPONENTS = ("collision", "drivable", "ttc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=("fixed256", "fixed1024", "dev-select"), required=True)
    parser.add_argument("--baseline-artifact", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--generation-kl-rolling-mean", type=float, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"artifact has no records: {path}")
    mapping = {str(record["token"]): record for record in records}
    if len(mapping) != len(records):
        raise ValueError(f"artifact contains duplicate tokens: {path}")
    return payload, mapping


def paired(base, candidate, field):
    tokens = list(base)
    return np.asarray(
        [float(candidate[token][field]) - float(base[token][field]) for token in tokens],
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


def component_delta(base, candidate, component):
    tokens = list(base)
    return float(np.mean([
        float(candidate[token]["selected_components"][component])
        - float(base[token]["selected_components"][component])
        for token in tokens
    ]))


def main() -> None:
    args = parse_args()
    baseline_payload, base = load(args.baseline_artifact)
    candidate_payload, candidate = load(args.candidate_artifact)
    if set(base) != set(candidate):
        raise RuntimeError("baseline and candidate token sets differ")
    baseline_sha = baseline_payload.get("summary", {}).get("token_set_sha256")
    candidate_sha = candidate_payload.get("summary", {}).get("token_set_sha256")
    if baseline_sha != candidate_sha:
        raise RuntimeError("baseline and candidate token-set SHA differ")

    selected = paired(base, candidate, "selected_reward")
    oracle = paired(base, candidate, "oracle_reward")
    candidate_mean = paired(base, candidate, "candidate_reward")
    selected_ci = bootstrap_ci(selected, args.bootstrap_samples, args.bootstrap_seed)
    safety = {
        component: component_delta(base, candidate, component)
        for component in SAFETY_COMPONENTS
    }
    safety_pass_tokens = [
        token for token, record in base.items()
        if all(float(record["selected_components"][name]) >= 1.0 for name in SAFETY_COMPONENTS)
    ]
    safety_pass_delta = float(np.mean([
        float(candidate[token]["selected_reward"]) - float(base[token]["selected_reward"])
        for token in safety_pass_tokens
    ]))
    kl = float(args.generation_kl_rolling_mean)
    failures = []

    if args.gate == "fixed256":
        if selected.mean() < -0.002:
            failures.append("selected delta is < -0.002")
        for component, difference in safety.items():
            if difference < -0.004:
                failures.append(f"{component} delta is < -0.004")
        if oracle.mean() < -0.004 or candidate_mean.mean() < -0.004:
            failures.append("oracle or candidate-mean delta is < -0.004")
    elif args.gate == "fixed1024":
        if selected.mean() <= 0:
            failures.append("selected delta is not > 0")
        if selected_ci[0] <= -0.0005:
            failures.append("selected CI lower bound is not > -0.0005")
        for component, difference in safety.items():
            if difference < -0.001:
                failures.append(f"{component} delta is < -0.001")
        if safety_pass_delta < -0.002:
            failures.append("safety-pass bucket delta is < -0.002")
        if oracle.mean() < -0.001:
            failures.append("oracle delta is < -0.001")
        if candidate_mean.mean() < 0:
            failures.append("candidate-mean delta is < 0")
    else:
        if selected.mean() < 0.003:
            failures.append("selected delta is < +0.003")
        if selected_ci[0] <= 0:
            failures.append("selected CI lower bound is not > 0")
        for component, difference in safety.items():
            if difference < -0.0005:
                failures.append(f"{component} delta is < -0.0005")
        if safety_pass_delta < -0.001:
            failures.append("safety-pass bucket delta is < -0.001")
        if oracle.mean() < -0.001:
            failures.append("oracle delta is < -0.001")
        if candidate_mean.mean() < 0:
            failures.append("candidate-mean delta is < 0")

    if selected.min() < -0.5:
        failures.append("a selected token delta is < -0.5")
    if not math.isfinite(kl) or kl > 2.5e-4:
        failures.append("generation KL rolling mean is not finite or > 2.5e-4")

    result = {
        "gate": f"stage10_{args.gate}",
        "passed": not failures,
        "num_tokens": int(selected.size),
        "token_set_sha256": baseline_sha,
        "selected_difference": float(selected.mean()),
        "selected_ci95": selected_ci,
        "minimum_token_selected_difference": float(selected.min()),
        "oracle_difference": float(oracle.mean()),
        "candidate_mean_difference": float(candidate_mean.mean()),
        "safety_component_differences": safety,
        "baseline_safety_pass_tokens": len(safety_pass_tokens),
        "safety_pass_selected_difference": safety_pass_delta,
        "generation_kl_rolling_mean": kl,
        "failures": failures,
    }
    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
