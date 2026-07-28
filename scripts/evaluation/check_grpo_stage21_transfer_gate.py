#!/usr/bin/env python3
"""Apply locked Stage21 fold5 or consumed dev-select transfer gates."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta


BASE_SHA = "59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d"
SAFETY = ("collision", "drivable", "ttc")
GATE_CONFIG = {
    "internal-test": {
        "noises": (-1, 20260729, 20260730, 20260731),
        "count": 1023,
        "logs": 151,
        "split": "train",
        "manifest_name": "internal_test",
    },
    "dev-select": {
        "noises": (-1, 20260729),
        "count": 3072,
        "logs": 219,
        "split": "val",
        "token_sha": "1d5e2c1ae12f8e5b65edfa9976166b7a24cb77df87fe67b154bff9431a491299",
    },
}


def ordered_sha(tokens):
    return hashlib.sha256("\n".join(tokens).encode()).hexdigest()


def load_manifest(path, gate):
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary, records = payload.get("summary", {}), payload.get("records", [])
    config = GATE_CONFIG[gate]
    tokens = [str(record["token"]) for record in records]
    checks = {
        "count": len(tokens) == config["count"],
        "unique": len(tokens) == len(set(tokens)),
        "logs": len({record["log_name"] for record in records}) == config["logs"],
    }
    if gate == "internal-test":
        checks.update({
            "name": summary.get("name") == config["manifest_name"],
            "objective": summary.get("stage21_objective") == "base_anchored_log_robust_v1",
            "fold": all(int(record["stage21_fold"]) == 5 for record in records),
            "sha": summary.get("ordered_token_sha256") == ordered_sha(tokens),
        })
    else:
        checks["sha"] = ordered_sha(tokens) == config["token_sha"]
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Stage21 {gate} manifest failed: {failed}")
    return tokens, {str(record["token"]): record for record in records}


def load_artifact(path, tokens, noise, split, candidate_sha):
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary, records = payload.get("summary", {}), payload.get("records", [])
    actual_tokens = [str(record.get("token")) for record in records]
    expected_sha = candidate_sha or BASE_SHA
    schedule = summary.get("schedule", {})
    checks = {
        "checkpoint": summary.get("checkpoint_sha256") == expected_sha,
        "reference": summary.get("reference_checkpoint_sha256") == BASE_SHA,
        "algorithm": summary.get("generation_policy_algorithm") == "diffgrpo_selected_anchor",
        "selector": summary.get("selector_logits_source") == "reference",
        "noise": summary.get("evaluation_noise_namespace") == noise,
        "schedule": schedule.get("roll_timesteps") == [32, 24, 16, 8, 0]
            and schedule.get("truncation_timestep") == 32
            and schedule.get("scheduler_num_inference_steps") == 125,
        "split": summary.get("log_split") == split,
        "complete": summary.get("completed") is True and int(summary.get("num_failures", -1)) == 0,
        "count": len(records) == len(tokens) and int(summary.get("num_tokens", -1)) == len(tokens),
        "order": actual_tokens == tokens,
        "sha": summary.get("token_set_sha256") == ordered_sha(tokens),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Stage21 transfer artifact failed {path}: {failed}")
    return {str(record["token"]): record for record in records}


def cluster_ci(values, logs, seed, samples):
    groups = defaultdict(list)
    for value, log_name in zip(values, logs):
        groups[log_name].append(value)
    arrays = [np.asarray(groups[name]) for name in sorted(groups)]
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for sample in range(samples):
        chosen = rng.integers(0, len(arrays), len(arrays))
        means[sample] = sum(float(arrays[index].sum()) for index in chosen) / sum(
            len(arrays[index]) for index in chosen
        )
    return np.quantile(means, [0.025, 0.975]).tolist()


def cp_upper(events, total):
    return 1.0 if events == total else float(beta.ppf(0.95, events + 1, total - events))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=tuple(GATE_CONFIG), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base", action="append", nargs=2, required=True, metavar=("NOISE", "PATH"))
    parser.add_argument("--candidate", action="append", nargs=2, required=True, metavar=("NOISE", "PATH"))
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = GATE_CONFIG[args.gate]
    tokens, metadata = load_manifest(args.manifest, args.gate)
    base_paths = {int(noise): Path(path) for noise, path in args.base}
    candidate_paths = {int(noise): Path(path) for noise, path in args.candidate}
    if set(base_paths) != set(config["noises"]) or set(candidate_paths) != set(config["noises"]):
        raise RuntimeError("Stage21 transfer noise set mismatch")
    bases = {
        noise: load_artifact(base_paths[noise], tokens, noise, config["split"], None)
        for noise in config["noises"]
    }
    candidates = {
        noise: load_artifact(
            candidate_paths[noise], tokens, noise, config["split"], args.expected_candidate_sha256
        ) for noise in config["noises"]
    }

    base_mean = {
        token: float(np.mean([bases[noise][token]["selected_reward"] for noise in config["noises"]]))
        for token in tokens
    }
    deltas, logs, hard, mature = [], [], [], []
    safety_values = {name: [] for name in SAFETY}
    namespace_means = {}
    for noise in config["noises"]:
        noise_values = []
        for token in tokens:
            delta = float(candidates[noise][token]["selected_reward"]) - float(bases[noise][token]["selected_reward"])
            noise_values.append(delta)
            deltas.append(delta)
            logs.append(str(metadata[token]["log_name"]))
            difficulty = (
                float(metadata[token]["base_reward_mean"])
                if args.gate == "internal-test" else base_mean[token]
            )
            (hard if difficulty < 0.75 else mature).append(delta)
            for name in SAFETY:
                safety_values[name].append(
                    float(candidates[noise][token]["selected_components"][name])
                    - float(bases[noise][token]["selected_components"][name])
                )
        namespace_means[str(noise)] = float(np.mean(noise_values))
    values, log_array = np.asarray(deltas), np.asarray(logs)
    ci = cluster_ci(values, log_array, 20260731, args.bootstrap_samples)
    safety = {}
    for offset, name in enumerate(SAFETY):
        component = np.asarray(safety_values[name])
        safety[name] = {
            "mean": float(component.mean()),
            "log_cluster_ci95": cluster_ci(component, log_array, 20260801 + offset, args.bootstrap_samples),
        }
    catastrophes = int((values <= -0.5).sum())
    metrics = {
        "namespace_c_minus_b": namespace_means,
        "mean_c_minus_b": float(values.mean()),
        "log_cluster_ci95": ci,
        "base_below_075_c_minus_b": float(np.mean(hard)),
        "base_at_least_075_c_minus_b": float(np.mean(mature)),
        "safety_c_minus_b": safety,
        "catastrophic_count": catastrophes,
        "catastrophic_cp_upper95": cp_upper(catastrophes, len(values)),
    }
    failures = []
    if args.gate == "internal-test" and any(value <= 0 for value in namespace_means.values()):
        failures.append("at least one namespace C-B is not positive")
    if metrics["mean_c_minus_b"] < 0.005:
        failures.append("mean C-B below +0.005")
    if ci[0] <= 0:
        failures.append("C-B log-cluster CI lower is not positive")
    if args.gate == "internal-test" and metrics["base_below_075_c_minus_b"] < 0.010:
        failures.append("base<0.75 gain below +0.010")
    if metrics["base_at_least_075_c_minus_b"] < -0.002:
        failures.append("base>=0.75 regression exceeds 0.002")
    for name, result in safety.items():
        if result["mean"] < -0.002:
            failures.append(f"{name} mean regression exceeds 0.002")
        if result["log_cluster_ci95"][1] < 0:
            failures.append(f"{name} has significant regression")
    if args.gate == "internal-test" and metrics["catastrophic_cp_upper95"] > 0.005:
        failures.append("catastrophic CP upper exceeds 0.005")
    result = {
        "passed": not failures,
        "gate": args.gate,
        "candidate_checkpoint_sha256": args.expected_candidate_sha256,
        "noise_namespaces": list(config["noises"]),
        "manifest": str(args.manifest.resolve()),
        "metrics": metrics,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
