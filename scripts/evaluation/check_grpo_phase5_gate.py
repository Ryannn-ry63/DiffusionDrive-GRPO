#!/usr/bin/env python3
"""Hard-stop gates for preregistered DiffusionDrive GRPO Phase 5."""

import argparse
import json
from pathlib import Path

import numpy as np

SAFETY_COMPONENTS = ("collision", "drivable", "ttc")
POST_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate",
        choices=("compatibility", "fixed256", "fixed1024", "dev-select"),
        required=True,
    )
    parser.add_argument("--baseline-artifact", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--carrier-artifact", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    return parser.parse_args()


def load(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"missing records: {path}")
    mapping = {str(record["token"]): record for record in records}
    if len(mapping) != len(records):
        raise ValueError(f"duplicate tokens: {path}")
    return payload, mapping


def paired(base, candidate, field):
    if set(base) != set(candidate):
        raise RuntimeError("paired gate artifacts have different token sets")
    tokens = list(base)
    return np.asarray(
        [float(candidate[t][field]) - float(base[t][field]) for t in tokens],
        dtype=np.float64,
    )


def bootstrap_ci(values, samples, seed):
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


def validate_projection(payload, records):
    summary = payload.get("summary", {})
    trust = summary.get("generation_trust")
    if not isinstance(trust, dict) or trust.get("mode") != "reference_mean_ball":
        raise RuntimeError("candidate artifact is not hard-projected")
    failures = []
    for step_name in ("transition", "final"):
        step = trust.get("steps", {}).get(step_name, {})
        radius = step.get("radius")
        if radius is None:
            failures.append(f"{step_name}: missing radius")
            continue
        if float(step.get("reference_coverage", 0.0)) != 1.0:
            failures.append(f"{step_name}: reference coverage is not 100%")
        if float(step.get("post_distance_max", np.inf)) > float(radius) + POST_TOLERANCE:
            failures.append(f"{step_name}: post-projection bound exceeded")
    for token, record in records.items():
        diagnostics = record.get("generation_trust", {})
        coverage = np.asarray(diagnostics.get("reference_coverage", []))
        post = np.asarray(diagnostics.get("post_distance", []), dtype=np.float64)
        if (
            coverage.size == 0
            or coverage.shape != post.shape
            or post.ndim != 2
            or post.shape[-1] != 2
            or not coverage.all()
            or not np.isfinite(post).all()
        ):
            failures.append(f"{token}: missing reference")
            break
        for step_index, step_name in enumerate(("transition", "final")):
            radius = float(trust["steps"][step_name]["radius"])
            if np.any(post[..., step_index] > radius + POST_TOLERANCE):
                failures.append(f"{token}: {step_name} bound exceeded")
                break
        if failures:
            break
    if failures:
        raise RuntimeError("; ".join(failures))
    return trust


def main() -> None:
    args = parse_args()
    baseline_payload, baseline = load(args.baseline_artifact)
    candidate_payload, candidate = load(args.candidate_artifact)
    trust = validate_projection(candidate_payload, candidate)
    selected = paired(baseline, candidate, "selected_reward")
    oracle = paired(baseline, candidate, "oracle_reward")
    candidate_mean = paired(baseline, candidate, "candidate_reward")
    selected_ci = bootstrap_ci(
        selected, args.bootstrap_samples, args.bootstrap_seed
    )
    oracle_ci = bootstrap_ci(
        oracle, args.bootstrap_samples, args.bootstrap_seed + 1
    )
    safety = {
        name: component_delta(baseline, candidate, name)
        for name in SAFETY_COMPONENTS
    }
    failures = []
    if oracle.min() < -0.5:
        failures.append(f"single-token oracle delta {oracle.min():.6f} < -0.5")

    if args.gate == "compatibility":
        if selected.mean() < -0.0005:
            failures.append("selected compatibility delta < -0.0005")
        if oracle.mean() < -0.0005:
            failures.append("oracle compatibility delta < -0.0005")
    elif args.gate == "fixed256":
        if selected.mean() < -0.001:
            failures.append("selected fixed-256 delta < -0.001")
        if oracle.mean() < -0.001:
            failures.append("oracle fixed-256 delta < -0.001")
        for name, value in safety.items():
            if value < -0.002:
                failures.append(f"{name} fixed-256 delta < -0.002")
    elif args.gate == "fixed1024":
        if selected_ci[0] <= 0:
            failures.append("selected fixed-1024 CI lower bound is not > 0")
        if oracle.mean() < -0.0005:
            failures.append("oracle fixed-1024 delta < -0.0005")
        if candidate_mean.mean() < 0:
            failures.append("candidate mean declined")
        for name, value in safety.items():
            if value < 0:
                failures.append(f"{name} declined")
        tokens = list(baseline)
        bottom_count = max(1, int(np.ceil(0.01 * len(tokens))))
        bottom_tokens = sorted(
            tokens, key=lambda token: float(baseline[token]["selected_reward"])
        )[:bottom_count]
        bottom_delta = np.mean([
            float(candidate[token]["selected_reward"])
            - float(baseline[token]["selected_reward"])
            for token in bottom_tokens
        ])
        if bottom_delta < 0:
            failures.append("baseline-bottom-1% delta < 0")
    else:
        if args.carrier_artifact is None:
            raise ValueError("dev-select gate requires --carrier-artifact")
        _, carrier = load(args.carrier_artifact)
        carrier_delta = paired(carrier, candidate, "selected_reward")
        carrier_ci = bootstrap_ci(
            carrier_delta, args.bootstrap_samples, args.bootstrap_seed + 2
        )
        if selected_ci[0] <= 0:
            failures.append("dev-select selected CI lower bound is not > 0")
        if carrier_delta.mean() < -0.0005 or carrier_ci[1] < 0:
            failures.append("projected policy significantly degrades K2 carrier")
        if oracle.mean() < -0.0005 or oracle_ci[1] < 0:
            failures.append("dev-select oracle gate failed")
        if candidate_mean.mean() < 0:
            failures.append("dev-select candidate mean declined")
        for name, value in safety.items():
            if value < 0:
                failures.append(f"dev-select {name} declined")

    result = {
        "gate": args.gate,
        "passed": not failures,
        "selected_difference": float(selected.mean()),
        "selected_ci95": selected_ci,
        "oracle_difference": float(oracle.mean()),
        "oracle_ci95": oracle_ci,
        "minimum_token_oracle_difference": float(oracle.min()),
        "candidate_mean_difference": float(candidate_mean.mean()),
        "safety_component_differences": safety,
        "projection": trust,
        "seed_expansion_authorized": (
            args.gate == "dev-select" and selected.mean() >= 0.003 and not failures
        ),
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
