#!/usr/bin/env python3
"""Strict paired fold4 C-B gate for Stage25 fresh-generator epochs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta


BASE_SHA = "59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d"
EPOCHS = (2, 4, 6, 8)
NOISES = (20260821, 20260822)
SAFETY = ("collision", "drivable", "ttc")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_sha(tokens: list[str]) -> str:
    return hashlib.sha256("\n".join(tokens).encode()).hexdigest()


def bootstrap_ci(values, logs, samples, seed):
    grouped = defaultdict(list)
    for value, log_name in zip(values, logs):
        grouped[str(log_name)].append(float(value))
    names = sorted(grouped)
    sums = np.asarray([sum(grouped[name]) for name in names])
    counts = np.asarray([len(grouped[name]) for name in names])
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for start in range(0, samples, 500):
        size = min(500, samples - start)
        indices = rng.integers(0, len(names), (size, len(names)))
        means[start:start + size] = (
            sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
        )
    return np.quantile(means, (0.025, 0.975)).tolist()


def cp_upper(events, trials):
    return 1.0 if events == trials else float(
        beta.ppf(0.95, events + 1, trials - events)
    )


def load_manifest(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    tokens = [str(record.get("token", "")) for record in records]
    logs = {str(record["token"]): str(record.get("log_name", "")) for record in records}
    summary = payload.get("summary", {})
    if (
        len(tokens) != 1021 or len(set(tokens)) != 1021
        or not all(tokens) or not all(logs.values())
        or summary.get("name") != "fold4"
        or summary.get("ordered_token_sha256") != ordered_sha(tokens)
    ):
        raise RuntimeError("Stage25 generator fold4 manifest drifted")
    return tokens, logs


def load_artifact(
    path, tokens, noise, checkpoint_sha, domain, calibration_sha, selector_sha,
):
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary, records = payload.get("summary", {}), payload.get("records", [])
    selector = summary.get("stage25_selector", {})
    schedule = summary.get("schedule", {})
    actual_tokens = [str(record.get("token", "")) for record in records]
    checks = {
        "checkpoint": summary.get("checkpoint_sha256") == checkpoint_sha,
        "reference": summary.get("reference_checkpoint_sha256") == BASE_SHA,
        "domain": summary.get("generator_domain") == domain,
        "noise": int(summary.get("evaluation_noise_namespace", -1)) == noise,
        "selector_source": summary.get("selector_logits_source")
            == "trajectory_relative_harm_v3",
        "selector_sha": selector.get("checkpoint_sha256") == selector_sha,
        "calibration_sha": selector.get("calibration_sha256") == calibration_sha,
        "schedule": schedule.get("truncation_timestep") == 32
            and schedule.get("roll_timesteps") == [32, 24, 16, 8, 0]
            and schedule.get("scheduler_num_inference_steps") == 125,
        "complete": summary.get("completed") is True
            and int(summary.get("num_failures", -1)) == 0,
        "count": len(records) == len(tokens),
        "order": actual_tokens == tokens,
        "token_sha": summary.get("token_set_sha256") == ordered_sha(tokens),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Stage25 generator artifact {path} failed {failed}")
    return {str(record["token"]): record for record in records}


def metrics_for_epoch(
    epoch, tokens, logs_by_token, bases, candidates, ood_threshold,
    bootstrap_samples,
):
    base_mean = {
        token: float(np.mean([
            bases[noise][token]["selected_reward"] for noise in NOISES
        ])) for token in tokens
    }
    deltas, logs, hard, mature = [], [], [], []
    safety = {name: [] for name in SAFETY}
    namespace_means, namespace_ood = {}, {}
    ood_events = 0
    ood_trials = 0
    for noise in NOISES:
        namespace_delta = []
        namespace_ood_events = 0
        namespace_ood_trials = 0
        for token in tokens:
            base = bases[noise][token]
            candidate = candidates[noise][token]
            delta = float(candidate["selected_reward"]) - float(base["selected_reward"])
            deltas.append(delta)
            namespace_delta.append(delta)
            logs.append(logs_by_token[token])
            (hard if base_mean[token] < 0.75 else mature).append(delta)
            for name in SAFETY:
                safety[name].append(
                    float(candidate["selected_components"][name])
                    - float(base["selected_components"][name])
                )
            distances = np.asarray(
                candidate["stage24_selector"]["ood_distance"], dtype=np.float64
            )
            if distances.shape != (20,) or not np.isfinite(distances).all():
                raise RuntimeError("Stage25 generator OOD diagnostics drifted")
            count = int((distances > ood_threshold).sum())
            ood_events += count
            ood_trials += distances.size
            namespace_ood_events += count
            namespace_ood_trials += distances.size
        namespace_means[str(noise)] = float(np.mean(namespace_delta))
        namespace_ood[str(noise)] = namespace_ood_events / namespace_ood_trials
    values = np.asarray(deltas, dtype=np.float64)
    log_array = np.asarray(logs)
    safety_means = {name: float(np.mean(items)) for name, items in safety.items()}
    catastrophes = int((values <= -0.5).sum())
    return {
        "namespace_c_minus_b": namespace_means,
        "pooled_c_minus_b": float(values.mean()),
        "whole_log_bootstrap_ci": bootstrap_ci(
            values, log_array, bootstrap_samples, 20260830 + epoch
        ),
        "hard_scene_c_minus_b": float(np.mean(hard)),
        "mature_scene_c_minus_b": float(np.mean(mature)),
        "hard_scene_count": len(hard),
        "mature_scene_count": len(mature),
        "safety_component_mean_deltas": safety_means,
        "catastrophic_count": catastrophes,
        "catastrophic_cp_upper": cp_upper(catastrophes, values.size),
        "candidate_ood_rate": ood_events / ood_trials,
        "namespace_candidate_ood_rate": namespace_ood,
        "wins": int((values > 0).sum()),
        "ties": int((values == 0).sum()),
        "losses": int((values < 0).sum()),
        "worst_c_minus_b": float(values.min()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--base", action="append", nargs=2, required=True)
    parser.add_argument("--candidate", action="append", nargs=5, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if calibration.get("stage") != 25 or not calibration.get("passed"):
        raise RuntimeError("Stage25 generator requires passed calibration")
    calibration_sha = sha256(args.calibration)
    selector_sha = calibration["selector_checkpoint_sha256"]
    ood_threshold = float(calibration["ood_threshold"])
    tokens, logs = load_manifest(args.manifest)
    base_paths = {int(noise): Path(path) for noise, path in args.base}
    if set(base_paths) != set(NOISES):
        raise RuntimeError("Stage25 generator base noise grid drifted")
    bases = {
        noise: load_artifact(
            base_paths[noise], tokens, noise, BASE_SHA, "official_base",
            calibration_sha, selector_sha,
        ) for noise in NOISES
    }
    paths, shas = defaultdict(dict), {}
    for epoch_text, noise_text, path_text, checkpoint_text, sha in args.candidate:
        epoch, noise = int(epoch_text), int(noise_text)
        path, checkpoint = Path(path_text), Path(checkpoint_text)
        if sha256(checkpoint) != sha:
            raise RuntimeError("Stage25 generator checkpoint SHA argument drifted")
        paths[epoch][noise] = path
        if epoch in shas and shas[epoch] != sha:
            raise RuntimeError("Stage25 generator SHA differs across namespaces")
        shas[epoch] = sha
    if set(paths) != set(EPOCHS) or any(set(paths[e]) != set(NOISES) for e in EPOCHS):
        raise RuntimeError("Stage25 generator candidate grid drifted")
    rows = []
    for epoch in EPOCHS:
        candidates = {
            noise: load_artifact(
                paths[epoch][noise], tokens, noise, shas[epoch],
                f"stage25_generator_epoch{epoch}", calibration_sha, selector_sha,
            ) for noise in NOISES
        }
        metrics = metrics_for_epoch(
            epoch, tokens, logs, bases, candidates, ood_threshold,
            args.bootstrap_samples,
        )
        checks = {
            "every_namespace_positive": all(
                value > 0 for value in metrics["namespace_c_minus_b"].values()
            ),
            "pooled_gain_at_least_0.005": metrics["pooled_c_minus_b"] >= 0.005,
            "whole_log_ci_positive": metrics["whole_log_bootstrap_ci"][0] > 0,
            "hard_scene_gain_at_least_0.010": metrics["hard_scene_c_minus_b"] >= 0.010,
            "mature_scene_gain_at_least_minus_0.002": metrics["mature_scene_c_minus_b"] >= -0.002,
            "safety_no_worse_0.001": all(
                value >= -0.001
                for value in metrics["safety_component_mean_deltas"].values()
            ),
            "catastrophic_upper_at_most_0.005": metrics["catastrophic_cp_upper"] <= 0.005,
            "candidate_ood_rate_at_most_0.05": metrics["candidate_ood_rate"] <= 0.05,
        }
        rows.append({
            "epoch": epoch, "checkpoint_sha256": shas[epoch],
            "eligible": all(checks.values()), "checks": checks,
            "metrics": metrics,
            "artifacts": {str(noise): str(paths[epoch][noise]) for noise in NOISES},
        })
    eligible = [row for row in rows if row["eligible"]]
    selected = max(
        eligible, key=lambda row: (row["metrics"]["pooled_c_minus_b"], -row["epoch"])
    ) if eligible else None
    result = {
        "schema_version": 1, "stage": 25,
        "passed": selected is not None,
        "stop_before_final_selector": selected is None,
        "selection_scope": "frozen_stage25_fold4_c_minus_b",
        "selection_rule": "highest pooled C-B among fully eligible epochs; earlier epoch breaks exact ties",
        "selected_epoch": selected["epoch"] if selected else None,
        "selected": selected, "candidates": rows,
        "calibration_sha256": calibration_sha,
        "selector_checkpoint_sha256": selector_sha,
        "manifest": str(args.manifest), "noise_namespaces": list(NOISES),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
