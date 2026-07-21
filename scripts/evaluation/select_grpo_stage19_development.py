#!/usr/bin/env python3
"""Select the locked Stage19 epoch from two-noise log-calibration results."""

import argparse
import json
from pathlib import Path

import numpy as np


EPOCHS = (1, 2, 4, 6, 8, 10)
SAFETY = ("collision", "drivable", "ttc")


def load_artifact(path, expected_noise):
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    records = {str(record["token"]): record for record in payload.get("records", ())}
    if len(records) != 918 or len(records) != len(payload.get("records", ())):
        raise RuntimeError(f"expected 918 unique records: {path}")
    if summary.get("num_tokens") != 918 or not summary.get("completed", False):
        raise RuntimeError(f"incomplete calibration artifact: {path}")
    if summary.get("selector_logits_source") != "reference":
        raise RuntimeError(f"calibration must use frozen reference selector: {path}")
    if summary.get("generation_policy_algorithm") != "diffgrpo_selected_anchor":
        raise RuntimeError(f"Stage19 algorithm mismatch: {path}")
    if summary.get("evaluation_noise_namespace") != expected_noise:
        raise RuntimeError(f"noise namespace mismatch: {path}")
    if summary.get("schedule", {}).get("roll_timesteps") != [32, 24, 16, 8, 0]:
        raise RuntimeError(f"full schedule mismatch: {path}")
    return summary, records


def load_logs(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", ())
    if payload.get("summary", {}).get("name") != "calibration" or len(records) != 918:
        raise RuntimeError("Stage19 selection requires the locked calibration manifest")
    result = {str(record["token"]): str(record["log_name"]) for record in records}
    if len(result) != 918:
        raise RuntimeError("calibration manifest contains duplicate tokens")
    return result


def cluster_bootstrap(values, logs, seed=20260722, samples=10_000):
    values = np.asarray(values, dtype=np.float64)
    logs = np.asarray(logs)
    names = np.unique(logs)
    groups = [values[logs == name] for name in names]
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        indices = rng.integers(0, len(groups), size=len(groups))
        total = sum(float(groups[index].sum()) for index in indices)
        count = sum(int(groups[index].size) for index in indices)
        means[sample] = total / count
    return np.quantile(means, [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-default", type=Path, required=True)
    parser.add_argument("--base-alt", type=Path, required=True)
    parser.add_argument(
        "--candidate", action="append", nargs=3, required=True,
        metavar=("EPOCH", "DEFAULT", "ALT"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    token_logs = load_logs(args.manifest)
    base_default_summary, base_default = load_artifact(args.base_default, -1)
    base_alt_summary, base_alt = load_artifact(args.base_alt, 20260723)
    if set(base_default) != set(token_logs) or set(base_alt) != set(token_logs):
        raise RuntimeError("base calibration token set mismatch")
    if base_default_summary.get("checkpoint_sha256") != base_alt_summary.get("checkpoint_sha256"):
        raise RuntimeError("two base noise cells use different weights")

    candidates = {}
    for epoch_text, default_text, alt_text in args.candidate:
        epoch = int(epoch_text)
        if epoch in candidates:
            raise RuntimeError(f"duplicate epoch: {epoch}")
        candidates[epoch] = (Path(default_text), Path(alt_text))
    if tuple(sorted(candidates)) != EPOCHS:
        raise RuntimeError(f"candidate epochs must be exactly {EPOCHS}")

    tokens = sorted(token_logs)
    logs = [token_logs[token] for token in tokens]
    rows = []
    for epoch in EPOCHS:
        default_summary, default = load_artifact(candidates[epoch][0], -1)
        alt_summary, alt = load_artifact(candidates[epoch][1], 20260723)
        if set(default) != set(tokens) or set(alt) != set(tokens):
            raise RuntimeError(f"epoch {epoch} token set mismatch")
        if default_summary.get("checkpoint_sha256") != alt_summary.get("checkpoint_sha256"):
            raise RuntimeError(f"epoch {epoch} two-noise weights mismatch")
        if default_summary.get("reference_checkpoint_sha256") != base_default_summary.get("checkpoint_sha256"):
            raise RuntimeError(f"epoch {epoch} frozen reference mismatch")

        default_delta = np.asarray([
            default[token]["selected_reward"] - base_default[token]["selected_reward"]
            for token in tokens
        ])
        alt_delta = np.asarray([
            alt[token]["selected_reward"] - base_alt[token]["selected_reward"]
            for token in tokens
        ])
        mean_delta = (default_delta + alt_delta) / 2.0
        safety = {
            name: float(np.mean([
                (
                    default[token]["selected_components"][name]
                    - base_default[token]["selected_components"][name]
                    + alt[token]["selected_components"][name]
                    - base_alt[token]["selected_components"][name]
                ) / 2.0
                for token in tokens
            ]))
            for name in SAFETY
        }
        ci = cluster_bootstrap(mean_delta, logs, seed=20260722 + epoch)
        eligible = (
            float(default_delta.mean()) > 0.0
            and float(mean_delta.mean()) >= 0.003
            and ci[0] > 0.0
            and all(value >= 0.0 for value in safety.values())
        )
        rows.append({
            "epoch": epoch,
            "checkpoint": default_summary.get("checkpoint"),
            "checkpoint_sha256": default_summary.get("checkpoint_sha256"),
            "default_c_minus_b": float(default_delta.mean()),
            "alternate_c_minus_b": float(alt_delta.mean()),
            "two_noise_mean_c_minus_b": float(mean_delta.mean()),
            "log_cluster_ci95": ci,
            "two_noise_safety_deltas": safety,
            "eligible": eligible,
            "artifacts": {
                "default": str(candidates[epoch][0]),
                "alternate": str(candidates[epoch][1]),
            },
        })

    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        result = {
            "passed": False,
            "selection_scope": "log_calibration_only",
            "noise_namespaces": [-1, 20260723],
            "candidates": rows,
            "selected": None,
            "failures": [
                "no Stage19 epoch passes the preregistered calibration gate"
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        raise SystemExit(1)
    best = max(row["two_noise_mean_c_minus_b"] for row in eligible)
    selected = min(
        (row for row in eligible if best - row["two_noise_mean_c_minus_b"] <= 0.0005),
        key=lambda row: row["epoch"],
    )
    result = {
        "passed": True,
        "selection_scope": "log_calibration_only",
        "noise_namespaces": [-1, 20260723],
        "candidates": rows,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
