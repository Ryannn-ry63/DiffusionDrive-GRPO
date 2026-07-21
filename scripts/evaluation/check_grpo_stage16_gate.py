#!/usr/bin/env python3
"""Apply preregistered Stage-16 schedule, test, fixed, and dev gates."""

import argparse
import json
from pathlib import Path

import numpy as np


COMPONENTS = ("collision", "drivable", "progress", "ttc", "comfort", "direction")
SAFETY = ("collision", "drivable", "ttc")


def load(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = {str(record["token"]): record for record in payload["records"]}
    if len(records) != len(payload["records"]):
        raise ValueError(f"duplicate tokens in {path}")
    return payload, records


def bootstrap(values, samples=10_000, seed=20260721):
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for start in range(0, samples, 1000):
        size = min(1000, samples - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means[start:start + size] = values[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def compare(left, right, label):
    if set(left) != set(right):
        raise ValueError(f"token mismatch for {label}")
    tokens = sorted(left)
    selected = np.asarray([
        right[token]["selected_reward"] - left[token]["selected_reward"]
        for token in tokens
    ], dtype=np.float64)
    return {
        "label": label,
        "count": len(tokens),
        "selected_mean": float(selected.mean()),
        "selected_ci95": bootstrap(selected),
        "minimum_token_delta": float(selected.min()),
        "component_differences": {
            name: float(np.mean([
                right[token]["selected_components"][name]
                - left[token]["selected_components"][name]
                for token in tokens
            ]))
            for name in COMPONENTS
        },
    }


def validate(payload, records, expected, timesteps, algorithm, label, failures):
    summary = payload.get("summary", {})
    if summary.get("selector_logits_source") != "reference":
        failures.append(f"{label} selector source is not reference")
    if len(records) != expected or summary.get("num_tokens") != expected:
        failures.append(f"{label} does not contain {expected} tokens")
    if summary.get("num_failures", 0) != 0 or summary.get("completed") is False:
        failures.append(f"{label} evaluation is incomplete")
    if summary.get("schedule", {}).get("roll_timesteps") != list(timesteps):
        failures.append(f"{label} schedule mismatch")
    if summary.get("generation_policy_algorithm") != algorithm:
        failures.append(f"{label} generation algorithm mismatch")
    if not summary.get("checkpoint_sha256"):
        failures.append(f"{label} checkpoint SHA is missing")
    if not summary.get("reference_checkpoint_sha256"):
        failures.append(f"{label} reference checkpoint SHA is missing")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=("schedule", "selector-test", "fixed1024", "dev-select"), required=True)
    parser.add_argument("--base-original", type=Path, required=True)
    parser.add_argument("--base-full", type=Path, required=True)
    parser.add_argument("--grpo-full", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = {"schedule": 918, "selector-test": 918, "fixed1024": 1024, "dev-select": 3072}[args.gate]
    paths = {"base_original": args.base_original, "base_full": args.base_full}
    if args.grpo_full is not None:
        paths["grpo_full"] = args.grpo_full
    loaded = {name: load(path) for name, path in paths.items()}
    failures = []
    validate(
        *loaded["base_original"], expected, (8, 0), "legacy_ppo",
        "base_original", failures,
    )
    validate(
        *loaded["base_full"], expected, (32, 24, 16, 8, 0),
        "diffgrpo_full_chain", "base_full", failures,
    )
    if args.gate != "schedule":
        if args.grpo_full is None:
            raise ValueError(f"{args.gate} requires --grpo-full")
        validate(
            *loaded["grpo_full"], expected, (32, 24, 16, 8, 0),
            "diffgrpo_full_chain", "grpo_full", failures,
        )

    summaries = {name: payload[0].get("summary", {}) for name, payload in loaded.items()}
    base_sha = summaries["base_original"].get("checkpoint_sha256")
    if summaries["base_full"].get("checkpoint_sha256") != base_sha:
        failures.append("base schedule cells do not use identical weights")
    for name, summary in summaries.items():
        if summary.get("reference_checkpoint_sha256") != base_sha:
            failures.append(f"{name} does not use the registered base reference")

    schedule = compare(loaded["base_original"][1], loaded["base_full"][1], "schedule_contribution")
    metrics = {"schedule_contribution": schedule}
    if args.gate == "schedule":
        if schedule["selected_mean"] < -0.005:
            failures.append("full schedule selected delta is < -0.005")
        if any(schedule["component_differences"][name] < -0.002 for name in SAFETY):
            failures.append("full schedule safety component is < -0.002")
    else:
        total = compare(loaded["base_original"][1], loaded["grpo_full"][1], "system_total")
        grpo = compare(loaded["base_full"][1], loaded["grpo_full"][1], "grpo_contribution")
        metrics.update({"system_total": total, "grpo_contribution": grpo})
        total_threshold = 0.003 if args.gate != "dev-select" else 0.005
        grpo_threshold = 0.002 if args.gate != "dev-select" else 0.003
        if total["selected_mean"] < total_threshold:
            failures.append(f"system total is < +{total_threshold:g}")
        if grpo["selected_mean"] < grpo_threshold:
            failures.append(f"GRPO contribution is < +{grpo_threshold:g}")
        if total["selected_ci95"][0] <= 0:
            failures.append("system-total CI lower bound is not > 0")
        if grpo["selected_ci95"][0] <= 0:
            failures.append("GRPO-contribution CI lower bound is not > 0")
        if any(total["component_differences"][name] < 0 for name in SAFETY):
            failures.append("system safety component declined")
        if total["minimum_token_delta"] <= -0.5:
            failures.append("worst system token is not > -0.5")

    result = {
        "gate": args.gate,
        "passed": not failures,
        "expected_tokens": expected,
        "artifacts": {name: str(path) for name, path in paths.items()},
        "metrics": metrics,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
