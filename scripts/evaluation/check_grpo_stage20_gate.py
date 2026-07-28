#!/usr/bin/env python3
"""Apply the preregistered Stage20 PDMS-first A/B/C transfer gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta


BASE_SHA256 = "59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d"
COMPONENTS = ("collision", "drivable", "progress", "ttc", "comfort", "direction")
SAFETY = ("collision", "drivable", "ttc")
NOISES = (-1, 20260726)
EXPECTED = {
    "dev-select": {
        "count": 3072,
        "token_sha": "1d5e2c1ae12f8e5b65edfa9976166b7a24cb77df87fe67b154bff9431a491299",
        "cb_min": 0.008,
    },
    "dev-confirm": {
        "count": 1024,
        "token_sha": "c98144d252415f74ccdf041a6f610b9a9bedee6adbba91027a7bb48a2df5554d",
        "cb_min": 0.005,
    },
}
SHORT_SCHEDULE = {
    "truncation_timestep": 8,
    "roll_timesteps": [8, 0],
    "scheduler_num_inference_steps": 125,
}
FULL_SCHEDULE = {
    "truncation_timestep": 32,
    "roll_timesteps": [32, 24, 16, 8, 0],
    "scheduler_num_inference_steps": 125,
}


def ordered_sha(tokens: list[str]) -> str:
    return hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("summary"), dict) or not isinstance(
        payload.get("records"), list
    ):
        raise RuntimeError(f"invalid Stage20 artifact schema: {path}")
    return payload


def core_schedule(summary: dict) -> dict:
    schedule = summary.get("schedule", {})
    return {
        "truncation_timestep": schedule.get("truncation_timestep"),
        "roll_timesteps": schedule.get("roll_timesteps"),
        "scheduler_num_inference_steps": schedule.get(
            "scheduler_num_inference_steps"
        ),
    }


def load_manifest(path: Path, gate: str) -> tuple[list[str], dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise RuntimeError("Stage20 manifest has no records")
    tokens = [str(record["token"]) for record in records]
    if len(tokens) != len(set(tokens)):
        raise RuntimeError("Stage20 manifest has duplicate tokens")
    expected = EXPECTED[gate]
    if len(tokens) != expected["count"] or ordered_sha(tokens) != expected["token_sha"]:
        raise RuntimeError("Stage20 manifest count/SHA mismatch")
    logs = {}
    for record in records:
        token = str(record["token"])
        log_name = str(record.get("log_name", ""))
        if not log_name:
            raise RuntimeError(f"manifest token has no log_name: {token}")
        logs[token] = log_name
    return tokens, logs


def validate_artifact(
    payload: dict,
    *,
    label: str,
    noise: int,
    tokens: list[str],
    expected_candidate_sha: str,
) -> None:
    summary = payload["summary"]
    records = payload["records"]
    expected_sha = BASE_SHA256 if label in {"A", "B"} else expected_candidate_sha
    expected_schedule = SHORT_SCHEDULE if label == "A" else FULL_SCHEDULE
    expected_algorithm = "legacy_ppo" if label == "A" else "diffgrpo_selected_anchor"
    actual_tokens = [str(record.get("token")) for record in records]
    checks = {
        "checkpoint SHA": summary.get("checkpoint_sha256") == expected_sha,
        "reference SHA": summary.get("reference_checkpoint_sha256") == BASE_SHA256,
        "algorithm": summary.get("generation_policy_algorithm") == expected_algorithm,
        "noise": summary.get("evaluation_noise_namespace") == noise,
        "selector": summary.get("selector_logits_source") == "reference",
        "schedule": core_schedule(summary) == expected_schedule,
        "log split": summary.get("log_split") == "val",
        "completion": bool(summary.get("completed")),
        "failure count": int(summary.get("num_failures", -1)) == 0,
        "token count": int(summary.get("num_tokens", -1)) == len(tokens),
        "record count": len(records) == len(tokens),
        "token order": actual_tokens == tokens,
        "token SHA": summary.get("token_set_sha256") == ordered_sha(tokens),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Stage20 {label}/{noise} provenance failed: {failed}")
    for record in records:
        reward = float(record.get("selected_reward", np.nan))
        components = record.get("selected_components", {})
        values = np.asarray(
            [components.get(name, np.nan) for name in COMPONENTS], dtype=np.float64
        )
        if not np.isfinite(reward) or not np.isfinite(values).all():
            raise RuntimeError(f"non-finite Stage20 record: {record.get('token')}")


def log_cluster_ci(
    values: np.ndarray,
    logs: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> list[float]:
    if values.ndim != 1 or logs.shape != values.shape or values.size == 0:
        raise ValueError("cluster bootstrap requires aligned nonempty vectors")
    grouped: dict[str, np.ndarray] = {}
    for log_name in sorted(set(logs.tolist())):
        grouped[log_name] = values[logs == log_name]
    names = list(grouped)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        chosen = rng.integers(0, len(names), size=len(names))
        total = 0.0
        count = 0
        for index in chosen:
            cluster = grouped[names[int(index)]]
            total += float(cluster.sum())
            count += int(cluster.size)
        means[sample_index] = total / count
    return np.quantile(means, [0.025, 0.975]).tolist()


def clopper_pearson_upper(events: int, total: int, alpha: float = 0.05) -> float:
    if not 0 <= events <= total or total <= 0:
        raise ValueError("invalid binomial event counts")
    if events == total:
        return 1.0
    return float(beta.ppf(1.0 - alpha, events + 1, total - events))


def evaluate_gate(
    *,
    gate: str,
    manifest: Path,
    cell_paths: dict[str, list[Path]],
    expected_candidate_sha: str,
    bootstrap_samples: int = 10000,
) -> dict:
    tokens, token_logs = load_manifest(manifest, gate)
    artifacts: dict[tuple[str, int], dict] = {}
    for label in ("A", "B", "C"):
        paths = cell_paths[label]
        if len(paths) != len(NOISES):
            raise RuntimeError(f"Stage20 cell {label} needs exactly two artifacts")
        for noise, path in zip(NOISES, paths):
            payload = load_json(path)
            validate_artifact(
                payload,
                label=label,
                noise=noise,
                tokens=tokens,
                expected_candidate_sha=expected_candidate_sha,
            )
            artifacts[(label, noise)] = payload

    namespace_cb = {}
    cb_values, ca_values, pooled_logs = [], [], []
    component_deltas = {name: [] for name in COMPONENTS}
    for noise in NOISES:
        by_label = {
            label: artifacts[(label, noise)]["records"] for label in ("A", "B", "C")
        }
        a = np.asarray([float(record["selected_reward"]) for record in by_label["A"]])
        b = np.asarray([float(record["selected_reward"]) for record in by_label["B"]])
        c = np.asarray([float(record["selected_reward"]) for record in by_label["C"]])
        cb = c - b
        namespace_cb[str(noise)] = float(cb.mean())
        cb_values.append(cb)
        ca_values.append(c - a)
        pooled_logs.extend(token_logs[token] for token in tokens)
        for name in COMPONENTS:
            bv = np.asarray(
                [float(record["selected_components"][name]) for record in by_label["B"]]
            )
            cv = np.asarray(
                [float(record["selected_components"][name]) for record in by_label["C"]]
            )
            component_deltas[name].append(cv - bv)

    cb = np.concatenate(cb_values)
    ca = np.concatenate(ca_values)
    logs = np.asarray(pooled_logs)
    cb_ci = log_cluster_ci(
        cb, logs, seed=20260727, samples=bootstrap_samples
    )
    safety = {}
    for index, name in enumerate(SAFETY):
        values = np.concatenate(component_deltas[name])
        safety[name] = {
            "mean": float(values.mean()),
            "log_cluster_ci95": log_cluster_ci(
                values,
                logs,
                seed=20260728 + index,
                samples=bootstrap_samples,
            ),
        }
    catastrophic_count = int((cb <= -0.5).sum())
    catastrophic_upper = clopper_pearson_upper(catastrophic_count, int(cb.size))
    metrics = {
        "namespace_c_minus_b": namespace_cb,
        "two_noise_c_minus_b": float(cb.mean()),
        "two_noise_c_minus_b_log_cluster_ci95": cb_ci,
        "two_noise_c_minus_a": float(ca.mean()),
        "safety_c_minus_b": safety,
        "catastrophic_count": catastrophic_count,
        "catastrophic_rate": float(catastrophic_count / cb.size),
        "catastrophic_clopper_pearson_upper95": catastrophic_upper,
        "num_token_noise_pairs": int(cb.size),
        "wins": int((cb > 0).sum()),
        "ties": int((cb == 0).sum()),
        "losses": int((cb < 0).sum()),
        "worst_c_minus_b": float(cb.min()),
        "bottom_1pct_c_minus_b": float(
            np.sort(cb)[: max(1, int(np.ceil(0.01 * cb.size)))].mean()
        ),
    }
    failures = []
    if any(value <= 0 for value in namespace_cb.values()):
        failures.append("at least one namespace has non-positive C-B")
    if metrics["two_noise_c_minus_b"] < EXPECTED[gate]["cb_min"]:
        failures.append("two-noise C-B mean is below the registered minimum")
    if cb_ci[0] <= 0:
        failures.append("C-B log-cluster CI lower bound is not positive")
    if metrics["two_noise_c_minus_a"] < 0.010:
        failures.append("two-noise C-A mean is below +0.010")
    for name, values in safety.items():
        if values["mean"] < -0.002:
            failures.append(f"{name} mean regression exceeds 0.002")
        if values["log_cluster_ci95"][1] < 0:
            failures.append(f"{name} has statistically supported regression")
    if catastrophic_upper > 0.005:
        failures.append("catastrophic-rate upper confidence bound exceeds 0.005")
    return {
        "passed": not failures,
        "gate": gate,
        "candidate_checkpoint_sha256": expected_candidate_sha,
        "base_checkpoint_sha256": BASE_SHA256,
        "noise_namespaces": list(NOISES),
        "manifest": str(manifest.resolve()),
        "manifest_ordered_token_sha256": EXPECTED[gate]["token_sha"],
        "metrics": metrics,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=tuple(EXPECTED), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell-a", type=Path, nargs=2, required=True)
    parser.add_argument("--cell-b", type=Path, nargs=2, required=True)
    parser.add_argument("--cell-c", type=Path, nargs=2, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    result = evaluate_gate(
        gate=args.gate,
        manifest=args.manifest,
        cell_paths={"A": args.cell_a, "B": args.cell_b, "C": args.cell_c},
        expected_candidate_sha=args.expected_candidate_sha256,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
