#!/usr/bin/env python3
"""Select Stage21 epoch 2/4/6/8 using only locked whole-log fold 4."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta


BASE_SHA = "59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d"
EPOCHS = (2, 4, 6, 8)
NOISES = (-1, 20260729)
SAFETY = ("collision", "drivable", "ttc")
FULL_SCHEDULE = {
    "truncation_timestep": 32,
    "roll_timesteps": [32, 24, 16, 8, 0],
    "scheduler_num_inference_steps": 125,
}


def ordered_sha(tokens):
    return hashlib.sha256("\n".join(tokens).encode()).hexdigest()


def core_schedule(summary):
    schedule = summary.get("schedule", {})
    return {name: schedule.get(name) for name in FULL_SCHEDULE}


def load_manifest(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    records = payload.get("records", [])
    tokens = [str(record["token"]) for record in records]
    checks = {
        "name": summary.get("name") == "calibration",
        "objective": summary.get("stage21_objective") == "base_anchored_log_robust_v1",
        "base": summary.get("base_checkpoint_sha256") == BASE_SHA,
        "count": len(records) == 1021 and int(summary.get("count", -1)) == 1021,
        "logs": len({record["log_name"] for record in records}) == 151,
        "unique": len(tokens) == len(set(tokens)),
        "token sha": summary.get("ordered_token_sha256") == ordered_sha(tokens),
        "fold": all(int(record["stage21_fold"]) == 4 for record in records),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"invalid Stage21 calibration manifest: {failed}")
    metadata = {str(record["token"]): record for record in records}
    return tokens, metadata


def load_artifact(path, tokens, noise, candidate):
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    records = payload.get("records", [])
    actual_tokens = [str(record.get("token")) for record in records]
    expected_checkpoint = None if candidate else BASE_SHA
    checks = {
        "checkpoint": candidate or summary.get("checkpoint_sha256") == expected_checkpoint,
        "reference": summary.get("reference_checkpoint_sha256") == BASE_SHA,
        "algorithm": summary.get("generation_policy_algorithm") == "diffgrpo_selected_anchor",
        "selector": summary.get("selector_logits_source") == "reference",
        "noise": summary.get("evaluation_noise_namespace") == noise,
        "schedule": core_schedule(summary) == FULL_SCHEDULE,
        "split": summary.get("log_split") == "train",
        "complete": summary.get("completed") is True and int(summary.get("num_failures", -1)) == 0,
        "count": len(records) == len(tokens) and int(summary.get("num_tokens", -1)) == len(tokens),
        "order": actual_tokens == tokens,
        "sha": summary.get("token_set_sha256") == ordered_sha(tokens),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Stage21 artifact provenance failed {path}: {failed}")
    return summary, {str(record["token"]): record for record in records}


def cluster_ci(values, logs, seed, samples):
    groups = defaultdict(list)
    for value, log_name in zip(values, logs):
        groups[log_name].append(value)
    arrays = [np.asarray(groups[name], dtype=np.float64) for name in sorted(groups)]
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for index in range(samples):
        selected = rng.integers(0, len(arrays), size=len(arrays))
        total = sum(float(arrays[group].sum()) for group in selected)
        count = sum(int(arrays[group].size) for group in selected)
        means[index] = total / count
    return np.quantile(means, [0.025, 0.975]).tolist()


def cp_upper(events, total):
    return 1.0 if events == total else float(beta.ppf(0.95, events + 1, total - events))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base", action="append", nargs=2, required=True, metavar=("NOISE", "PATH"))
    parser.add_argument("--candidate", action="append", nargs=4, required=True, metavar=("EPOCH", "NOISE", "PATH", "SHA"))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    tokens, metadata = load_manifest(args.manifest)
    base_paths = {int(noise): Path(path) for noise, path in args.base}
    if set(base_paths) != set(NOISES):
        raise RuntimeError(f"base noises must be exactly {NOISES}")
    bases = {
        noise: load_artifact(base_paths[noise], tokens, noise, False)
        for noise in NOISES
    }
    candidate_args = defaultdict(dict)
    candidate_shas = {}
    for epoch_text, noise_text, path, sha in args.candidate:
        epoch, noise = int(epoch_text), int(noise_text)
        if noise in candidate_args[epoch]:
            raise RuntimeError("duplicate Stage21 candidate epoch/noise")
        candidate_args[epoch][noise] = Path(path)
        if epoch in candidate_shas and candidate_shas[epoch] != sha:
            raise RuntimeError("candidate SHA differs across namespaces")
        candidate_shas[epoch] = sha
    if set(candidate_args) != set(EPOCHS) or any(set(candidate_args[epoch]) != set(NOISES) for epoch in EPOCHS):
        raise RuntimeError(f"candidate epochs/noises must be {EPOCHS}/{NOISES}")

    rows = []
    for epoch in EPOCHS:
        candidates = {
            noise: load_artifact(candidate_args[epoch][noise], tokens, noise, True)
            for noise in NOISES
        }
        for noise in NOISES:
            if candidates[noise][0].get("checkpoint_sha256") != candidate_shas[epoch]:
                raise RuntimeError(f"candidate epoch {epoch} SHA mismatch")
        deltas, logs, hard, mature = [], [], [], []
        component_deltas = {name: [] for name in SAFETY}
        namespace_means = {}
        for noise in NOISES:
            base_records, candidate_records = bases[noise][1], candidates[noise][1]
            noise_delta = []
            for token in tokens:
                delta = float(candidate_records[token]["selected_reward"]) - float(base_records[token]["selected_reward"])
                noise_delta.append(delta)
                deltas.append(delta)
                logs.append(str(metadata[token]["log_name"]))
                (hard if float(metadata[token]["base_reward_mean"]) < 0.75 else mature).append(delta)
                for name in SAFETY:
                    component_deltas[name].append(
                        float(candidate_records[token]["selected_components"][name])
                        - float(base_records[token]["selected_components"][name])
                    )
            namespace_means[str(noise)] = float(np.mean(noise_delta))
        values, log_array = np.asarray(deltas), np.asarray(logs)
        ci = cluster_ci(values, log_array, 20260729 + epoch, args.bootstrap_samples)
        safety = {}
        for offset, name in enumerate(SAFETY):
            component = np.asarray(component_deltas[name])
            safety[name] = {
                "mean": float(component.mean()),
                "log_cluster_ci95": cluster_ci(
                    component, log_array, 20260800 + 10 * epoch + offset, args.bootstrap_samples
                ),
            }
        catastrophes = int((values <= -0.5).sum())
        metrics = {
            "namespace_c_minus_b": namespace_means,
            "two_noise_c_minus_b": float(values.mean()),
            "log_cluster_ci95": ci,
            "base_below_075_c_minus_b": float(np.mean(hard)),
            "base_at_least_075_c_minus_b": float(np.mean(mature)),
            "safety_c_minus_b": safety,
            "catastrophic_count": catastrophes,
            "catastrophic_cp_upper95": cp_upper(catastrophes, len(values)),
        }
        failures = []
        if metrics["two_noise_c_minus_b"] < 0.005:
            failures.append("C-B mean below +0.005")
        if ci[0] <= 0:
            failures.append("C-B log-cluster CI lower is not positive")
        if metrics["base_below_075_c_minus_b"] < 0.010:
            failures.append("base<0.75 gain below +0.010")
        if metrics["base_at_least_075_c_minus_b"] < -0.002:
            failures.append("base>=0.75 regression exceeds 0.002")
        for name, result in safety.items():
            if result["mean"] < -0.002:
                failures.append(f"{name} mean regression exceeds 0.002")
            if result["log_cluster_ci95"][1] < 0:
                failures.append(f"{name} has significant regression")
        if metrics["catastrophic_cp_upper95"] > 0.005:
            failures.append("catastrophic CP upper exceeds 0.005")
        rows.append({
            "epoch": epoch,
            "checkpoint_sha256": candidate_shas[epoch],
            "checkpoint": candidates[-1][0].get("checkpoint"),
            "metrics": metrics,
            "eligible": not failures,
            "failures": failures,
            "artifacts": {str(noise): str(candidate_args[epoch][noise]) for noise in NOISES},
        })
    eligible = [row for row in rows if row["eligible"]]
    selected = None
    if eligible:
        best = max(row["metrics"]["two_noise_c_minus_b"] for row in eligible)
        selected = min(
            (row for row in eligible if best - row["metrics"]["two_noise_c_minus_b"] <= 0.001),
            key=lambda row: row["epoch"],
        )
    result = {
        "passed": selected is not None,
        "selection_scope": "stage21_whole_log_fold4_only",
        "selected_epoch": selected["epoch"] if selected else None,
        "selected": selected,
        "candidates": rows,
        "noise_namespaces": list(NOISES),
        "manifest": str(args.manifest.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if selected is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
