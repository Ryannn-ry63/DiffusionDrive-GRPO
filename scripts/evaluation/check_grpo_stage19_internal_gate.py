#!/usr/bin/env python3
"""Apply the preregistered four-noise Stage19 internal log-test gate."""

import argparse
import json
from pathlib import Path

import numpy as np

NAMESPACES = (-1, 20260723, 20260724, 20260725)
SAFETY = ("collision", "drivable", "ttc")


def cluster_bootstrap(values, logs, seed=20260725, samples=10_000):
    values = np.asarray(values, dtype=np.float64)
    logs = np.asarray(logs)
    groups = [values[logs == name] for name in np.unique(logs)]
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        indices = rng.integers(0, len(groups), size=len(groups))
        total = sum(float(groups[index].sum()) for index in indices)
        count = sum(int(groups[index].size) for index in indices)
        means[sample] = total / count
    return np.quantile(means, [0.025, 0.975]).tolist()


def load(path, namespace, label):
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    records = {str(record["token"]): record for record in payload.get("records", ())}
    if len(records) != 918 or summary.get("num_tokens") != 918:
        raise RuntimeError(f"{label} must contain 918 unique tokens")
    if not summary.get("completed", False) or summary.get("num_failures", 0) != 0:
        raise RuntimeError(f"{label} is incomplete")
    if summary.get("selector_logits_source") != "reference":
        raise RuntimeError(f"{label} did not use the frozen selector")
    if summary.get("generation_policy_algorithm") != "diffgrpo_selected_anchor":
        raise RuntimeError(f"{label} algorithm mismatch")
    if summary.get("evaluation_noise_namespace") != namespace:
        raise RuntimeError(f"{label} noise namespace mismatch")
    if summary.get("schedule", {}).get("roll_timesteps") != [32, 24, 16, 8, 0]:
        raise RuntimeError(f"{label} schedule mismatch")
    return summary, records


def parse_cells(items, label):
    result = {}
    for namespace_text, path_text in items:
        namespace = int(namespace_text)
        if namespace in result:
            raise RuntimeError(f"duplicate {label} namespace {namespace}")
        result[namespace] = Path(path_text)
    if tuple(sorted(result)) != tuple(sorted(NAMESPACES)):
        raise RuntimeError(f"{label} namespaces must be exactly {NAMESPACES}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base", action="append", nargs=2, required=True)
    parser.add_argument("--candidate", action="append", nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_records = manifest.get("records", ())
    if manifest.get("summary", {}).get("name") != "test" or len(manifest_records) != 918:
        raise RuntimeError("internal gate requires the locked Stage19 test manifest")
    token_logs = {
        str(record["token"]): str(record["log_name"])
        for record in manifest_records
    }
    if len(token_logs) != 918:
        raise RuntimeError("test manifest token duplication")

    base_paths = parse_cells(args.base, "base")
    candidate_paths = parse_cells(args.candidate, "candidate")
    base, candidate = {}, {}
    base_sha = candidate_sha = reference_sha = None
    for namespace in NAMESPACES:
        bsummary, brecords = load(base_paths[namespace], namespace, "base")
        csummary, crecords = load(candidate_paths[namespace], namespace, "candidate")
        if set(brecords) != set(token_logs) or set(crecords) != set(token_logs):
            raise RuntimeError(f"namespace {namespace} token set mismatch")
        if base_sha is None:
            base_sha = bsummary.get("checkpoint_sha256")
            candidate_sha = csummary.get("checkpoint_sha256")
            reference_sha = csummary.get("reference_checkpoint_sha256")
        if bsummary.get("checkpoint_sha256") != base_sha:
            raise RuntimeError("base weights differ across namespaces")
        if csummary.get("checkpoint_sha256") != candidate_sha:
            raise RuntimeError("candidate weights differ across namespaces")
        if csummary.get("reference_checkpoint_sha256") != reference_sha:
            raise RuntimeError("reference weights differ across namespaces")
        if reference_sha != base_sha:
            raise RuntimeError("candidate reference is not the evaluated base")
        base[namespace], candidate[namespace] = brecords, crecords

    tokens = sorted(token_logs)
    logs = [token_logs[token] for token in tokens]
    deltas = {}
    for namespace in NAMESPACES:
        deltas[namespace] = np.asarray([
            candidate[namespace][token]["selected_reward"]
            - base[namespace][token]["selected_reward"]
            for token in tokens
        ])
    four_noise = np.mean(np.stack([deltas[name] for name in NAMESPACES]), axis=0)
    safety = {
        name: float(np.mean([
            candidate[namespace][token]["selected_components"][name]
            - base[namespace][token]["selected_components"][name]
            for namespace in NAMESPACES for token in tokens
        ]))
        for name in SAFETY
    }
    namespace_means = {
        str(namespace): float(deltas[namespace].mean())
        for namespace in NAMESPACES
    }
    ci = cluster_bootstrap(four_noise, logs, seed=20260725)
    failures = []
    if namespace_means["-1"] < 0.005:
        failures.append("default C-B is below +0.005")
    if float(four_noise.mean()) < 0.005:
        failures.append("four-noise mean C-B is below +0.005")
    if any(value <= 0.0 for value in namespace_means.values()):
        failures.append("at least one noise namespace is non-positive")
    if ci[0] <= 0.0:
        failures.append("log-cluster CI lower bound is not positive")
    if any(value < 0.0 for value in safety.values()):
        failures.append("aggregate safety delta is negative")
    result = {
        "passed": not failures,
        "gate": "stage19_internal_log_test",
        "candidate_checkpoint_sha256": candidate_sha,
        "base_checkpoint_sha256": base_sha,
        "namespace_c_minus_b": namespace_means,
        "four_noise_mean_c_minus_b": float(four_noise.mean()),
        "four_noise_log_cluster_ci95": ci,
        "four_noise_safety_deltas": safety,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
