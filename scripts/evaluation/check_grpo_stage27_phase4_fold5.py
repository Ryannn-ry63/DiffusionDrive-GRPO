#!/usr/bin/env python3
"""One-shot Stage27 Phase4 fold5 epoch selection and generator gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


INPUTS_SHA = (
    "f22cb655e8acfbc41a752acb99772ecabdcdd7ad9788953f4e14e6063067fe4b"
)
PUBLIC_SHA = (
    "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
)
SELECTOR_SHA = (
    "023d7b6b77bb8fa2dc3e779849f3716688abf0796b019416494c80b29615b691"
)
CALIBRATION_SHA = (
    "fec2a10573e913e4452f8eea92e6334fcd5e6f1104e6244876770973a1f41990"
)
MANIFEST_SHA = (
    "2424fe88319f81650eec1b8e2dbe265143d5d6d7cc8d113e104caf8d150cf2d4"
)
NOISES = (20261011, 20261012)
SYSTEMS = ("B", "C1", "C2")
DOMAINS = {
    "B": "public88_base",
    "C1": "stage27_epoch1",
    "C2": "stage27_epoch2",
}
COMPONENTS = ("collision", "drivable", "progress", "ttc", "comfort", "direction")
SAFETY_COMPONENTS = ("collision", "ttc")
BOOTSTRAP_SEED = 20261027
BOOTSTRAP_SAMPLES = 10000


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_sha(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def whole_log_bootstrap_ci(
    values: np.ndarray,
    logs: list[str],
) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, log_name in zip(values, logs):
        grouped[log_name].append(float(value))
    names = sorted(grouped)
    if len(names) != 151:
        raise RuntimeError("Stage27 Phase4 whole-log groups drifted")
    sums = np.asarray([sum(grouped[name]) for name in names], dtype=np.float64)
    counts = np.asarray([len(grouped[name]) for name in names], dtype=np.int64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_SAMPLES, 500):
        size = min(500, BOOTSTRAP_SAMPLES - start)
        indices = rng.integers(0, len(names), (size, len(names)))
        means[start:start + size] = (
            sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
        )
    return np.quantile(means, (0.025, 0.975)).tolist()


def load_manifest(path: Path) -> tuple[list[str], dict[str, str]]:
    if file_sha256(path) != MANIFEST_SHA:
        raise RuntimeError("Stage27 protected fold5 manifest SHA drifted")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    summary = payload.get("summary", {})
    tokens = [str(record.get("token", "")) for record in records]
    logs = {
        str(record.get("token", "")): str(record.get("log_name", ""))
        for record in records
    }
    checks = {
        "name": summary.get("name") == "fold5",
        "count": len(tokens) == summary.get("count") == 1023,
        "unique": len(set(tokens)) == 1023,
        "logs": len(set(logs.values())) == summary.get("num_logs") == 151,
        "nonempty": all(tokens) and all(logs.values()),
        "ordered_sha": summary.get("ordered_token_sha256") == ordered_sha(tokens),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Stage27 protected fold5 manifest failed {failed}")
    return tokens, logs


def load_artifact(
    path: Path,
    system: str,
    noise: int,
    tokens: list[str],
    expected_sha: str,
    calibration: dict,
) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    records = payload.get("records", [])
    selector = summary.get("stage25_selector", {})
    schedule = summary.get("schedule", {})
    actual_tokens = [str(record.get("token", "")) for record in records]
    checks = {
        "checkpoint": summary.get("checkpoint_sha256") == expected_sha,
        "reference": summary.get("reference_checkpoint_sha256") == PUBLIC_SHA,
        "domain": summary.get("generator_domain") == DOMAINS[system],
        "noise": summary.get("evaluation_noise_namespace") == noise,
        "selector_source":
            summary.get("selector_logits_source") == "trajectory_relative_harm_v3",
        "selector_sha": selector.get("checkpoint_sha256") == SELECTOR_SHA,
        "calibration_sha": selector.get("calibration_sha256") == CALIBRATION_SHA,
        "calibration_mode": selector.get("calibration_collection") is False,
        "margin": np.isclose(
            selector.get("residual_margin"),
            calibration["residual_margin"],
            atol=0.0,
            rtol=0.0,
        ),
        "risk": np.isclose(
            selector.get("risk_threshold"),
            calibration["risk_threshold"],
            atol=0.0,
            rtol=0.0,
        ),
        "ood": np.isclose(
            selector.get("ood_threshold"),
            calibration["ood_threshold"],
            atol=0.0,
            rtol=0.0,
        ),
        "schedule": (
            schedule.get("truncation_timestep") == 32
            and schedule.get("roll_timesteps") == [32, 24, 16, 8, 0]
            and schedule.get("scheduler_num_inference_steps") == 125
        ),
        "complete": (
            summary.get("completed") is True
            and int(summary.get("num_failures", -1)) == 0
        ),
        "count": len(records) == summary.get("num_tokens") == 1023,
        "order": actual_tokens == tokens,
        "token_sha": summary.get("token_set_sha256") == ordered_sha(tokens),
        "log_split": summary.get("log_split") == "train",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"Stage27 Phase4 artifact {path} failed {failed}"
        )

    by_token = {}
    for record in records:
        token = str(record["token"])
        rewards = np.asarray(record["candidate_rewards"], dtype=np.float64)
        candidate_components = np.asarray(
            record["candidate_components"], dtype=np.float64
        )
        selected_components = np.asarray(
            [record["selected_components"][name] for name in COMPONENTS],
            dtype=np.float64,
        )
        diagnostic = record["stage24_selector"]
        ood_distance = np.asarray(
            diagnostic["ood_distance"], dtype=np.float64
        )
        selected = int(record["selected_mode"])
        fallback = int(diagnostic["fallback_mode"])
        if (
            rewards.shape != (20,)
            or candidate_components.shape != (20, 6)
            or selected_components.shape != (6,)
            or ood_distance.shape != (20,)
            or not np.isfinite(rewards).all()
            or not np.isfinite(candidate_components).all()
            or not np.isfinite(selected_components).all()
            or not np.isfinite(ood_distance).all()
            or not 0 <= selected < 20
            or not 0 <= fallback < 20
            or int(diagnostic["selected_mode"]) != selected
            or not np.isclose(
                float(record["selected_reward"]),
                rewards[selected],
                atol=2e-6,
                rtol=0.0,
            )
            or not np.allclose(
                selected_components,
                candidate_components[selected],
                atol=2e-6,
                rtol=0.0,
            )
            or not np.isclose(
                float(diagnostic["true_selected_minus_fallback"]),
                rewards[selected] - rewards[fallback],
                atol=2e-6,
                rtol=0.0,
            )
        ):
            raise RuntimeError(
                f"Stage27 Phase4 record is invalid: {system}/{noise}/{token}"
            )
        by_token[token] = record
    return by_token


def summarize_epoch(
    epoch: int,
    values: list[float],
    logs: list[str],
    noise_values: dict[int, list[float]],
    component_values: dict[str, list[float]],
) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (2046,) or not np.isfinite(array).all():
        raise RuntimeError(f"Stage27 C{epoch}-B paired values drifted")
    ci = whole_log_bootstrap_ci(array, logs)
    catastrophic_count = int(np.sum(array <= -0.5))
    catastrophic_rate = float(catastrophic_count / array.size)
    mean = float(array.mean())
    namespace_means = {
        f"ns{noise}": float(np.mean(noise_values[noise]))
        for noise in NOISES
    }
    safety_means = {
        name: float(np.mean(component_values[name]))
        for name in SAFETY_COMPONENTS
    }
    checks = {
        "pooled_gain_at_least_0.005": mean >= 0.005,
        "whole_log_ci_lower_strictly_positive": ci[0] > 0.0,
        "both_namespace_means_strictly_positive":
            all(value > 0.0 for value in namespace_means.values()),
        "collision_delta_at_least_negative_0.0005":
            safety_means["collision"] >= -0.0005,
        "ttc_delta_at_least_negative_0.0005":
            safety_means["ttc"] >= -0.0005,
        "catastrophic_rate_at_most_0.005":
            catastrophic_rate <= 0.005,
    }
    return {
        "epoch": epoch,
        "comparison": f"C{epoch}-B",
        "pooled_mean_delta": mean,
        "whole_log_bootstrap_ci": ci,
        "namespace_mean_delta": namespace_means,
        "safety_component_mean_deltas": safety_means,
        "wins": int(np.sum(array > 0)),
        "ties": int(np.sum(array == 0)),
        "losses": int(np.sum(array < 0)),
        "worst_delta": float(array.min()),
        "catastrophic_count": catastrophic_count,
        "catastrophic_rate": catastrophic_rate,
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument(
        "--artifact", action="append", nargs=3,
        metavar=("SYSTEM", "NOISE", "PATH"), required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if file_sha256(args.inputs) != INPUTS_SHA:
        raise RuntimeError("Stage27 Phase4 input-freeze SHA drifted")
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    if (
        not inputs.get("passed")
        or inputs.get("stage") != 27
        or inputs.get("phase") != 4
        or tuple(inputs.get("noise_namespaces", ())) != NOISES
    ):
        raise RuntimeError("Stage27 Phase4 input-freeze semantics drifted")
    manifest_path = Path(inputs["manifest"])
    tokens, logs_by_token = load_manifest(manifest_path)
    calibration_path = Path(inputs["calibration"])
    if file_sha256(calibration_path) != CALIBRATION_SHA:
        raise RuntimeError("Stage27 Phase4 calibration SHA drifted")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))

    artifact_paths = {}
    for system, noise_text, path_text in args.artifact:
        noise = int(noise_text)
        key = (system, noise)
        if key in artifact_paths:
            raise RuntimeError(f"duplicate Stage27 Phase4 artifact cell: {key}")
        artifact_paths[key] = Path(path_text)
    expected_cells = {
        (system, noise) for system in SYSTEMS for noise in NOISES
    }
    if set(artifact_paths) != expected_cells:
        raise RuntimeError("Stage27 Phase4 artifact grid is incomplete")

    artifacts = {}
    provenance = {}
    for system, noise in sorted(expected_cells):
        path = artifact_paths[(system, noise)]
        expected_sha = inputs["systems"][system]["sha256"]
        artifacts[(system, noise)] = load_artifact(
            path, system, noise, tokens, expected_sha, calibration
        )
        provenance[f"{system}:ns{noise}"] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        }

    system_scores = {system: [] for system in SYSTEMS}
    selector_switches = {system: 0 for system in SYSTEMS}
    summaries = {}
    for epoch in (1, 2):
        values = []
        logs = []
        noise_values = {noise: [] for noise in NOISES}
        component_values = {name: [] for name in SAFETY_COMPONENTS}
        for noise in NOISES:
            for token in tokens:
                base = artifacts[("B", noise)][token]
                current = artifacts[(f"C{epoch}", noise)][token]
                base_reward = float(base["selected_reward"])
                current_reward = float(current["selected_reward"])
                delta = current_reward - base_reward
                values.append(delta)
                logs.append(logs_by_token[token])
                noise_values[noise].append(delta)
                for name in SAFETY_COMPONENTS:
                    component_values[name].append(
                        float(current["selected_components"][name])
                        - float(base["selected_components"][name])
                    )
        summaries[f"C{epoch}-B"] = summarize_epoch(
            epoch, values, logs, noise_values, component_values
        )

    for system in SYSTEMS:
        for noise in NOISES:
            for token in tokens:
                record = artifacts[(system, noise)][token]
                system_scores[system].append(float(record["selected_reward"]))
                selector_switches[system] += int(
                    record["stage24_selector"]["switched"]
                )

    candidates = [summaries["C1-B"], summaries["C2-B"]]
    selected = max(
        candidates,
        key=lambda item: (
            item["whole_log_bootstrap_ci"][0],
            item["pooled_mean_delta"],
            -item["catastrophic_count"],
            -item["epoch"],
        ),
    )
    passed = bool(selected["all_checks_pass"])
    mean = selected["pooled_mean_delta"]
    if mean >= 0.015:
        tier = "desired_target"
    elif mean >= 0.010:
        tier = "paper_target"
    elif mean >= 0.005:
        tier = "minimally_useful"
    else:
        tier = "failed"

    result = {
        "schema_version": 1,
        "stage": 27,
        "phase": 4,
        "passed": passed,
        "stop_before_selector_closure": not passed,
        "stop_before_navtest": True,
        "selection_scope": "fold5_provisional_epoch_selection",
        "selection_rule": [
            "larger whole-log CI lower bound",
            "larger pooled mean C-B",
            "fewer catastrophic regressions",
            "earlier epoch",
        ],
        "selected_system": f"C{selected['epoch']}",
        "selected_epoch": selected["epoch"],
        "selected_checkpoint": inputs["systems"][f"C{selected['epoch']}"],
        "performance_tier": tier,
        "inputs": str(args.inputs.resolve()),
        "inputs_sha256": INPUTS_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "selector_checkpoint_sha256": SELECTOR_SHA,
        "calibration_sha256": CALIBRATION_SHA,
        "noise_namespaces": list(NOISES),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "num_tokens": 1023,
        "num_logs": 151,
        "num_paired_observations_per_epoch": 2046,
        "mean_system_scores": {
            system: float(np.mean(values))
            for system, values in system_scores.items()
        },
        "comparisons": summaries,
        "selected_checks": selected["checks"],
        "selector_diagnostics": {
            system: {
                "switch_count": selector_switches[system],
                "switch_rate": selector_switches[system] / 2046,
            }
            for system in SYSTEMS
        },
        "artifacts": provenance,
    }
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite Stage27 Phase4 report: {args.output}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
