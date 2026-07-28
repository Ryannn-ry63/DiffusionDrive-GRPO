#!/usr/bin/env python3
"""Apply preregistered Stage22 epoch-selection and transfer gates."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta


BASE_SHA = "59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d"
SAFETY = ("collision", "drivable", "ttc")
CONFIG = {
    "inner": {
        "labels": ("1", "2", "3", "4"), "noises": (-1, 20260801),
        "name": "inner_calibration", "count": 1019, "split": "train",
        "mature": -0.001, "each_positive": True,
    },
    "known-stress": {
        "labels": ("locked",), "noises": (-1, 20260729),
        "name": "known_stress", "count": 1021, "split": "train",
        "mature": -0.002, "each_positive": False,
    },
    "internal-test": {
        "labels": ("locked",), "noises": (-1, 20260801, 20260802, 20260803),
        "name": "internal_test", "count": 1023, "split": "train",
        "mature": -0.002, "each_positive": True,
    },
    "dev-select": {
        "labels": ("locked",), "noises": (-1, 20260804),
        "name": None, "count": 3072, "split": "val",
        "mature": -0.002, "each_positive": True,
    },
    "dev-confirm": {
        "labels": ("locked",), "noises": (-1, 20260804),
        "name": None, "count": 1024, "split": "val",
        "mature": -0.002, "each_positive": True,
    },
}


def ordered_sha(tokens):
    return hashlib.sha256("\n".join(tokens).encode()).hexdigest()


def load_manifest(path, gate):
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary, records = payload.get("summary", {}), payload.get("records", [])
    config = CONFIG[gate]
    tokens = [str(record["token"]) for record in records]
    if len(tokens) != config["count"] or len(tokens) != len(set(tokens)):
        raise RuntimeError(f"Stage22 {gate} manifest count/uniqueness mismatch")
    if config["name"] is not None:
        checks = {
            "name": summary.get("name") == config["name"],
            "objective": summary.get("stage22_objective")
                == "paired_delta_bootstrap_exact_kl_lora_v2",
            "base": summary.get("base_checkpoint_sha256") == BASE_SHA,
            "token_sha": summary.get("ordered_token_sha256") == ordered_sha(tokens),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(f"Stage22 {gate} manifest provenance: {failed}")
    metadata = {str(record["token"]): record for record in records}
    if any("log_name" not in record for record in records):
        raise RuntimeError("Stage22 gate requires whole-log identities")
    return tokens, metadata


def load_artifact(path, tokens, noise, split, expected_sha):
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary, records = payload.get("summary", {}), payload.get("records", [])
    schedule = summary.get("schedule", {})
    actual = [str(record.get("token")) for record in records]
    checks = {
        "checkpoint": summary.get("checkpoint_sha256") == expected_sha,
        "reference": summary.get("reference_checkpoint_sha256") == BASE_SHA,
        "algorithm": summary.get("generation_policy_algorithm")
            == "diffgrpo_paired_residual",
        "selector": summary.get("selector_logits_source") == "reference",
        "noise": int(summary.get("evaluation_noise_namespace", -2)) == noise,
        "schedule": schedule.get("roll_timesteps") == [32, 24, 16, 8, 0]
            and schedule.get("truncation_timestep") == 32
            and schedule.get("scheduler_num_inference_steps") == 125,
        "split": summary.get("log_split") == split,
        "complete": summary.get("completed") is True
            and int(summary.get("num_failures", -1)) == 0,
        "order": actual == tokens,
        "count": len(records) == len(tokens),
        "token_sha": summary.get("token_set_sha256") == ordered_sha(tokens),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Stage22 artifact failed {path}: {failed}")
    return {str(record["token"]): record for record in records}


def cluster_ci(values, logs, seed, samples):
    groups = defaultdict(list)
    for value, log_name in zip(values, logs):
        groups[log_name].append(value)
    arrays = [np.asarray(groups[name], dtype=np.float64) for name in sorted(groups)]
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for index in range(samples):
        chosen = rng.integers(0, len(arrays), len(arrays))
        means[index] = (
            sum(float(arrays[item].sum()) for item in chosen)
            / sum(len(arrays[item]) for item in chosen)
        )
    return np.quantile(means, [0.025, 0.975]).tolist()


def cp_upper(events, total):
    return 1.0 if events == total else float(
        beta.ppf(0.95, events + 1, total - events)
    )


def compute_metrics(tokens, metadata, bases, candidates, noises, seed, samples):
    base_mean = {
        token: float(np.mean([
            bases[noise][token]["selected_reward"] for noise in noises
        ])) for token in tokens
    }
    deltas, logs, hard, mature = [], [], [], []
    safety_values = {name: [] for name in SAFETY}
    namespace_means = {}
    for noise in noises:
        current_namespace = []
        for token in tokens:
            delta = (
                float(candidates[noise][token]["selected_reward"])
                - float(bases[noise][token]["selected_reward"])
            )
            deltas.append(delta)
            current_namespace.append(delta)
            logs.append(str(metadata[token]["log_name"]))
            (hard if base_mean[token] < 0.75 else mature).append(delta)
            for name in SAFETY:
                safety_values[name].append(
                    float(candidates[noise][token]["selected_components"][name])
                    - float(bases[noise][token]["selected_components"][name])
                )
        namespace_means[str(noise)] = float(np.mean(current_namespace))
    values, log_values = np.asarray(deltas), np.asarray(logs)
    safety = {}
    for offset, name in enumerate(SAFETY):
        component = np.asarray(safety_values[name])
        safety[name] = {
            "mean": float(component.mean()),
            "log_cluster_ci95": cluster_ci(
                component, log_values, seed + 100 + offset, samples
            ),
        }
    catastrophes = int((values <= -0.5).sum())
    return {
        "namespace_c_minus_b": namespace_means,
        "mean_c_minus_b": float(values.mean()),
        "log_cluster_ci95": cluster_ci(values, log_values, seed, samples),
        "base_below_075_c_minus_b": float(np.mean(hard)),
        "base_at_least_075_c_minus_b": float(np.mean(mature)),
        "safety_c_minus_b": safety,
        "catastrophic_count": catastrophes,
        "catastrophic_cp_upper95": cp_upper(catastrophes, len(values)),
    }


def gate_failures(metrics, config):
    failures = []
    if config["each_positive"] and any(
        value <= 0 for value in metrics["namespace_c_minus_b"].values()
    ):
        failures.append("at least one namespace C-B is not positive")
    if metrics["mean_c_minus_b"] < 0.005:
        failures.append("mean C-B below +0.005")
    if metrics["log_cluster_ci95"][0] <= 0:
        failures.append("C-B log-cluster CI lower is not positive")
    if metrics["base_below_075_c_minus_b"] < 0.010:
        failures.append("base<0.75 gain below +0.010")
    if metrics["base_at_least_075_c_minus_b"] < config["mature"]:
        failures.append("base>=0.75 regression exceeds gate")
    for name, result in metrics["safety_c_minus_b"].items():
        if result["mean"] < -0.001:
            failures.append(f"{name} average regression exceeds 0.001")
        if result["log_cluster_ci95"][1] < 0:
            failures.append(f"{name} has significant regression")
    if metrics["catastrophic_cp_upper95"] > 0.005:
        failures.append("catastrophic CP upper exceeds 0.005")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=tuple(CONFIG), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--base", action="append", nargs=2, required=True,
        metavar=("NOISE", "PATH"),
    )
    parser.add_argument(
        "--candidate", action="append", nargs=4, required=True,
        metavar=("LABEL", "NOISE", "PATH", "SHA"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = CONFIG[args.gate]
    tokens, metadata = load_manifest(args.manifest, args.gate)
    base_paths = {int(noise): Path(path) for noise, path in args.base}
    if set(base_paths) != set(config["noises"]):
        raise RuntimeError("Stage22 base noise set mismatch")
    bases = {
        noise: load_artifact(
            base_paths[noise], tokens, noise, config["split"], BASE_SHA
        ) for noise in config["noises"]
    }
    paths, shas = defaultdict(dict), {}
    for label, noise_text, path, sha in args.candidate:
        noise = int(noise_text)
        paths[label][noise] = Path(path)
        if label in shas and shas[label] != sha:
            raise RuntimeError("Stage22 candidate SHA differs across noises")
        shas[label] = sha
    if set(paths) != set(config["labels"]) or any(
        set(paths[label]) != set(config["noises"]) for label in config["labels"]
    ):
        raise RuntimeError("Stage22 candidate label/noise grid mismatch")

    rows = []
    for label in config["labels"]:
        candidates = {
            noise: load_artifact(
                paths[label][noise], tokens, noise, config["split"], shas[label]
            ) for noise in config["noises"]
        }
        metrics = compute_metrics(
            tokens, metadata, bases, candidates, config["noises"],
            20260801 + int(label if label.isdigit() else 10),
            args.bootstrap_samples,
        )
        failures = gate_failures(metrics, config)
        rows.append({
            "label": label,
            "epoch": int(label) if label.isdigit() else None,
            "checkpoint_sha256": shas[label],
            "metrics": metrics,
            "eligible": not failures,
            "failures": failures,
            "artifacts": {
                str(noise): str(paths[label][noise])
                for noise in config["noises"]
            },
        })
    eligible = [row for row in rows if row["eligible"]]
    selected = None
    if eligible:
        if args.gate == "inner":
            best = max(row["metrics"]["mean_c_minus_b"] for row in eligible)
            selected = min(
                (
                    row for row in eligible
                    if best - row["metrics"]["mean_c_minus_b"] <= 0.001
                ),
                key=lambda row: row["epoch"],
            )
        else:
            selected = eligible[0]
    result = {
        "passed": selected is not None,
        "gate": args.gate,
        "selection_scope": (
            "stage22_whole_log_fold3_only"
            if args.gate == "inner" else "locked_checkpoint_transfer"
        ),
        "selected_epoch": selected["epoch"] if selected else None,
        "selected": selected,
        "candidates": rows,
        "noise_namespaces": list(config["noises"]),
        "manifest": str(args.manifest.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if selected is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
