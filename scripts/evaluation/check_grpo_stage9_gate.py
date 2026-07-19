#!/usr/bin/env python3
"""Apply the preregistered Stage-9 selector epoch-1 fixed-1024 gate."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


SAFETY_COMPONENTS = ("collision", "drivable", "ttc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-artifact", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--generation-kl-rolling-mean", type=float, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    return parser.parse_args()


def load_records(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"artifact has no records: {path}")
    mapping = {str(record["token"]): record for record in records}
    if len(mapping) != len(records):
        raise ValueError(f"artifact contains duplicate tokens: {path}")
    return payload, mapping


def paired(base, candidate, field):
    if set(base) != set(candidate):
        raise RuntimeError("baseline and candidate token sets differ")
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


def component_difference(base, candidate, component):
    tokens = list(base)
    return float(
        np.mean(
            [
                float(candidate[token]["selected_components"][component])
                - float(base[token]["selected_components"][component])
                for token in tokens
            ]
        )
    )


def main() -> None:
    args = parse_args()
    baseline_payload, baseline = load_records(args.baseline_artifact)
    candidate_payload, candidate = load_records(args.candidate_artifact)
    baseline_sha = baseline_payload.get("summary", {}).get("token_set_sha256")
    candidate_sha = candidate_payload.get("summary", {}).get("token_set_sha256")
    if baseline_sha != candidate_sha:
        raise RuntimeError("baseline and candidate token-set SHA differ")

    selected = paired(baseline, candidate, "selected_reward")
    selected_ci = bootstrap_ci(
        selected, args.bootstrap_samples, args.bootstrap_seed
    )
    safety = {
        component: component_difference(baseline, candidate, component)
        for component in SAFETY_COMPONENTS
    }
    generation_kl = float(args.generation_kl_rolling_mean)
    failures = []
    if selected.mean() <= 0.0:
        failures.append("fixed-1024 selected delta is not > 0")
    if selected_ci[0] <= -0.0005:
        failures.append("selected CI lower bound is not > -0.0005")
    for component, difference in safety.items():
        if difference < -0.001:
            failures.append(f"{component} delta is < -0.001")
    if selected.min() < -0.5:
        failures.append("a selected token delta is < -0.5")
    if not math.isfinite(generation_kl) or generation_kl > 2.5e-4:
        failures.append("generation KL rolling mean is not finite or > 2.5e-4")

    result = {
        "gate": "stage9_selector_epoch1_fixed1024",
        "passed": not failures,
        "num_tokens": int(selected.size),
        "token_set_sha256": baseline_sha,
        "selected_difference": float(selected.mean()),
        "selected_ci95": selected_ci,
        "minimum_token_selected_difference": float(selected.min()),
        "safety_component_differences": safety,
        "generation_kl_rolling_mean": generation_kl,
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
