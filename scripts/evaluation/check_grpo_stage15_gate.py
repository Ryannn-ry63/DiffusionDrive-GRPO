#!/usr/bin/env python3
"""Apply preregistered Stage-15 selector-test, fixed, and dev gates."""

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
        means[start : start + size] = values[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def compare(left, right, label):
    if set(left) != set(right):
        raise ValueError(f"token mismatch for {label}")
    tokens = sorted(left)
    selected = np.asarray(
        [right[token]["selected_reward"] - left[token]["selected_reward"] for token in tokens],
        dtype=np.float64,
    )
    component = {
        name: float(
            np.mean(
                [
                    right[token]["selected_components"][name]
                    - left[token]["selected_components"][name]
                    for token in tokens
                ]
            )
        )
        for name in COMPONENTS
    }
    return {
        "label": label,
        "count": len(tokens),
        "selected": selected,
        "selected_mean": float(selected.mean()),
        "selected_ci95": bootstrap(selected),
        "minimum_token_delta": float(selected.min()),
        "component_differences": component,
    }


def validate(payload, records, source, expected_count, label, failures):
    summary = payload.get("summary", {})
    if summary.get("selector_logits_source") != source:
        failures.append(f"{label} selector source is not {source}")
    if len(records) != expected_count or summary.get("num_tokens") != expected_count:
        failures.append(f"{label} does not contain {expected_count} tokens")
    if summary.get("num_failures", 0) != 0 or summary.get("completed") is False:
        failures.append(f"{label} evaluation is incomplete")
    if source == "value_top2" and "value_selector" not in summary:
        failures.append(f"{label} lacks value-selector provenance")


def render(metric):
    return {key: value for key, value in metric.items() if key != "selected"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=("selector-test", "fixed1024", "dev-select"), required=True)
    parser.add_argument("--base-frozen", type=Path, required=True)
    parser.add_argument("--grpo-frozen", type=Path, required=True)
    parser.add_argument("--base-value", type=Path)
    parser.add_argument("--grpo-value", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = {"selector-test": 918, "fixed1024": 1024, "dev-select": 3072}[args.gate]
    paths = {
        "base_frozen": args.base_frozen,
        "grpo_frozen": args.grpo_frozen,
        "grpo_value": args.grpo_value,
    }
    if args.base_value is not None:
        paths["base_value"] = args.base_value
    loaded = {name: load(path) for name, path in paths.items()}
    failures = []
    for name, (payload, records) in loaded.items():
        source = "value_top2" if name.endswith("value") else "reference"
        validate(payload, records, source, expected, name, failures)

    if args.gate == "selector-test":
        metric = compare(
            loaded["grpo_frozen"][1], loaded["grpo_value"][1], "grpo_value_vs_frozen"
        )
        value_summary = loaded["grpo_value"][0]["summary"].get("value_selector", {})
        if metric["selected_mean"] < 0.002:
            failures.append("selector-test selected delta is < +0.002")
        if metric["selected_ci95"][0] <= 0:
            failures.append("selector-test CI lower bound is not > 0")
        if float(value_summary.get("switch_rate", 0.0)) < 0.01:
            failures.append("selector-test switch rate is < 1%")
        if any(metric["component_differences"][name] < 0 for name in SAFETY):
            failures.append("selector-test safety component declined")
        if float(value_summary.get("worst_switch_gain", -1.0)) <= -0.5:
            failures.append("selector-test worst switch is not > -0.5")
        if float(value_summary.get("beneficial_gain_sum", 0.0)) <= float(
            value_summary.get("harmful_loss_sum", 0.0)
        ):
            failures.append("selector-test beneficial gain does not exceed harmful loss")
        metrics = {metric["label"]: render(metric)}
    else:
        if args.base_value is None:
            raise ValueError(f"{args.gate} requires --base-value")
        base_frozen = loaded["base_frozen"][1]
        grpo_frozen = loaded["grpo_frozen"][1]
        base_value = loaded["base_value"][1]
        grpo_value = loaded["grpo_value"][1]
        comparisons = [
            compare(base_frozen, grpo_value, "system_total"),
            compare(base_frozen, base_value, "base_selector_gain"),
            compare(grpo_frozen, grpo_value, "grpo_selector_gain"),
            compare(base_value, grpo_value, "same_selector_grpo_gain"),
        ]
        by_name = {metric["label"]: metric for metric in comparisons}
        metrics = {metric["label"]: render(metric) for metric in comparisons}
        total_threshold = 0.003 if args.gate == "fixed1024" else 0.005
        if by_name["system_total"]["selected_mean"] < total_threshold:
            failures.append(f"system total is < +{total_threshold:g}")
        if by_name["base_selector_gain"]["selected_mean"] <= 0:
            failures.append("base selector gain is not positive")
        if by_name["grpo_selector_gain"]["selected_mean"] <= 0:
            failures.append("GRPO selector gain is not positive")
        grpo_threshold = 0.0 if args.gate == "fixed1024" else 0.0005
        if by_name["same_selector_grpo_gain"]["selected_mean"] < grpo_threshold:
            failures.append(f"same-selector GRPO gain is < +{grpo_threshold:g}")
        if args.gate == "dev-select":
            if by_name["system_total"]["selected_ci95"][0] <= 0:
                failures.append("system-total CI lower bound is not > 0")
            if by_name["same_selector_grpo_gain"]["selected_ci95"][0] <= 0:
                failures.append("same-selector GRPO CI lower bound is not > 0")
            safety_tokens = [
                token for token, record in base_frozen.items()
                if all(
                    float(record["selected_components"][name]) >= 1.0 - 1e-6
                    for name in SAFETY
                )
            ]
            safety_pass_delta = float(
                np.mean(
                    [
                        grpo_value[token]["selected_reward"]
                        - base_frozen[token]["selected_reward"]
                        for token in safety_tokens
                    ]
                )
            ) if safety_tokens else None
            metrics["safety_pass_tokens"] = len(safety_tokens)
            metrics["safety_pass_selected_difference"] = safety_pass_delta
            if safety_pass_delta is None or safety_pass_delta < 0:
                failures.append("safety-pass bucket declined")
        if any(
            by_name["system_total"]["component_differences"][name] < 0
            for name in SAFETY
        ):
            failures.append("system safety component declined")
        if by_name["system_total"]["minimum_token_delta"] <= -0.5:
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
