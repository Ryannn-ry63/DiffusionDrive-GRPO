#!/usr/bin/env python3
"""Apply the frozen Stage28 fold4 paper-level checkpoint selection gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


PUBLIC_SHA = "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
COMPONENTS = ("collision", "drivable", "progress", "ttc", "comfort", "direction")
GUARD_COMPONENTS = ("collision", "drivable", "ttc", "comfort", "direction")
SYSTEMS = ("P", "A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4")
NOISES = (20261111, 20261112)
BOOTSTRAP_SEED = 20261128
BOOTSTRAP_SAMPLES = 10000


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ordered_sha(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def bootstrap_whole_log(values: np.ndarray, logs: list[str]) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, log_name in zip(values, logs):
        grouped[log_name].append(float(value))
    names = sorted(grouped)
    if len(names) != 151:
        raise RuntimeError("Stage28 fold4 whole-log count drifted")
    sums = np.asarray([sum(grouped[name]) for name in names], dtype=np.float64)
    counts = np.asarray([len(grouped[name]) for name in names], dtype=np.int64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_SAMPLES, 500):
        size = min(500, BOOTSTRAP_SAMPLES - start)
        indices = rng.integers(0, len(names), (size, len(names)))
        samples[start:start + size] = (
            sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
        )
    return np.quantile(samples, (0.025, 0.975)).tolist()


def one_sided_wilson_upper(events: int, total: int) -> float:
    # One-sided 95% Wilson upper confidence bound, preregistered for Stage28.
    z = 1.6448536269514722
    p = events / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return float((center + radius) / denominator)


def trim10_mean(values: np.ndarray) -> float:
    ordered = np.sort(values)
    trim = int(math.floor(0.1 * ordered.size))
    return float(ordered[trim:-trim].mean())


def load_manifest(path: Path, expected_sha: str) -> tuple[list[str], dict[str, str]]:
    if file_sha256(path) != expected_sha:
        raise RuntimeError("Stage28 fold4 manifest SHA drifted")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    summary = payload.get("summary", {})
    tokens = [str(record.get("token", "")) for record in records]
    logs = {str(record.get("token", "")): str(record.get("log_name", "")) for record in records}
    if not (
        len(tokens) == summary.get("count") == 1021
        and len(set(tokens)) == 1021 and all(tokens)
        and len(set(logs.values())) == summary.get("num_logs") == 151
        and all(logs.values())
        and summary.get("ordered_token_sha256") == ordered_sha(tokens)
    ):
        raise RuntimeError("Stage28 fold4 manifest semantics drifted")
    return tokens, logs


def load_artifact(
    path: Path, system: str, noise: int, tokens: list[str],
    inputs: dict, calibration: dict,
) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    records = payload.get("records", [])
    selector = summary.get("stage25_selector", {})
    schedule = summary.get("schedule", {})
    entry = inputs["systems"][system]
    actual_tokens = [str(record.get("token", "")) for record in records]
    checks = {
        "checkpoint": summary.get("checkpoint_sha256") == entry["sha256"],
        "reference": summary.get("reference_checkpoint_sha256") == PUBLIC_SHA,
        "domain": summary.get("generator_domain") == entry["domain"],
        "noise": summary.get("evaluation_noise_namespace") == noise,
        "safe_selector": selector.get("checkpoint_sha256") == inputs["selector_checkpoint_sha256"],
        "safe_calibration": selector.get("calibration_sha256") == inputs["calibration_sha256"],
        "deploy": selector.get("calibration_collection") is False,
        "source": summary.get("selector_logits_source") == "trajectory_relative_harm_v3",
        "margin": np.isclose(selector.get("residual_margin"), calibration["residual_margin"], atol=0, rtol=0),
        "risk": np.isclose(selector.get("risk_threshold"), calibration["risk_threshold"], atol=0, rtol=0),
        "ood": np.isclose(selector.get("ood_threshold"), calibration["ood_threshold"], atol=0, rtol=0),
        "schedule": schedule.get("truncation_timestep") == 32
            and schedule.get("roll_timesteps") == [32, 24, 16, 8, 0]
            and schedule.get("scheduler_num_inference_steps") == 125,
        "complete": summary.get("completed") is True and summary.get("num_failures") == 0,
        "count": len(records) == summary.get("num_tokens") == 1021,
        "order": actual_tokens == tokens,
        "token_sha": summary.get("token_set_sha256") == ordered_sha(tokens),
        "split": summary.get("log_split") == "train",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Stage28 artifact {system}/ns{noise} failed {failed}")
    by_token = {}
    for record in records:
        token = str(record["token"])
        rewards = np.asarray(record["candidate_rewards"], dtype=np.float64)
        candidate_components = np.asarray(record["candidate_components"], dtype=np.float64)
        selected_components = np.asarray(
            [record["selected_components"][name] for name in COMPONENTS], dtype=np.float64
        )
        diagnostic = record["stage24_selector"]
        selected = int(record["selected_mode"])
        fallback = int(diagnostic["fallback_mode"])
        if not (
            rewards.shape == (20,) and candidate_components.shape == (20, 6)
            and selected_components.shape == (6,) and np.isfinite(rewards).all()
            and np.isfinite(candidate_components).all() and np.isfinite(selected_components).all()
            and 0 <= selected < 20 and 0 <= fallback < 20
            and int(diagnostic["selected_mode"]) == selected
            and np.isclose(record["selected_reward"], rewards[selected], atol=2e-6, rtol=0)
            and np.allclose(selected_components, candidate_components[selected], atol=2e-6, rtol=0)
        ):
            raise RuntimeError(f"Stage28 invalid record: {system}/{noise}/{token}")
        by_token[token] = record
    return by_token


def summarize(system: str, deltas: np.ndarray, base_scores: np.ndarray,
              logs: list[str], noise_values: dict[int, list[float]],
              component_values: dict[str, list[float]]) -> dict:
    if deltas.shape != (2042,) or not np.isfinite(deltas).all():
        raise RuntimeError(f"Stage28 paired vector drifted for {system}")
    ci = bootstrap_whole_log(deltas, logs)
    trimmed = trim10_mean(deltas)
    exact_ties = deltas == 0.0
    wins = int(np.sum(deltas > 0.0)); losses = int(np.sum(deltas < 0.0))
    hard = base_scores < 0.75; mature = ~hard
    if not hard.any() or not mature.any():
        raise RuntimeError("Stage28 hard/mature split is empty")
    cats = int(np.sum(deltas <= -0.5))
    cat_upper = one_sided_wilson_upper(cats, deltas.size)
    component_means = {name: float(np.mean(component_values[name])) for name in GUARD_COMPONENTS}
    namespace_means = {f"ns{noise}": float(np.mean(noise_values[noise])) for noise in NOISES}
    checks = {
        "pooled_gain_at_least_0.005": float(deltas.mean()) >= 0.005,
        "whole_log_ci_lower_strictly_positive": ci[0] > 0.0,
        "both_namespace_means_strictly_positive": all(value > 0 for value in namespace_means.values()),
        "trimmed10_mean_nonnegative": trimmed >= 0.0,
        "wins_exceed_losses_excluding_exact_ties": wins > losses,
        "hard_scene_gain_at_least_0.010": float(deltas[hard].mean()) >= 0.010,
        "mature_scene_gain_at_least_negative_0.0005": float(deltas[mature].mean()) >= -0.0005,
        "all_guard_components_at_least_negative_0.0005": all(value >= -0.0005 for value in component_means.values()),
        "catastrophic_one_sided_95_upper_at_most_0.005": cat_upper <= 0.005,
        "finite_and_provenance_complete": True,
    }
    return {
        "system": system, "branch": system[0], "epoch": int(system[1:]),
        "comparison": f"{system}-P", "pooled_mean_delta": float(deltas.mean()),
        "whole_log_bootstrap_ci": ci, "trimmed10_mean_delta": trimmed,
        "namespace_mean_delta": namespace_means,
        "hard_scene_count": int(hard.sum()), "hard_scene_mean_delta": float(deltas[hard].mean()),
        "mature_scene_count": int(mature.sum()), "mature_scene_mean_delta": float(deltas[mature].mean()),
        "component_mean_deltas": component_means,
        "wins": wins, "exact_ties": int(exact_ties.sum()), "losses": losses,
        "worst_delta": float(deltas.min()), "catastrophic_count": cats,
        "catastrophic_rate": float(cats / deltas.size),
        "catastrophic_one_sided_95_wilson_upper": cat_upper,
        "checks": checks, "eligible": bool(all(checks.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--artifact", action="append", nargs=3, required=True,
                        metavar=("SYSTEM", "NOISE", "PATH"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs_sha = file_sha256(args.inputs)
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    if not (
        inputs.get("passed") and inputs.get("stage") == 28 and inputs.get("phase") == "fold4"
        and tuple(inputs.get("noise_namespaces", ())) == NOISES
        and tuple(inputs.get("systems", {}).keys()) == SYSTEMS
        and inputs.get("all_evaluations_use_safe_multi") is True
    ):
        raise RuntimeError("Stage28 fold4 input-freeze semantics drifted")
    tokens, logs_by_token = load_manifest(Path(inputs["manifest"]), inputs["manifest_sha256"])
    calibration_path = Path(inputs["calibration"])
    if file_sha256(calibration_path) != inputs["calibration_sha256"]:
        raise RuntimeError("Stage28 calibration SHA drifted")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    paths = {(system, int(noise)): Path(path) for system, noise, path in args.artifact}
    expected = {(system, noise) for system in SYSTEMS for noise in NOISES}
    if set(paths) != expected or len(paths) != len(args.artifact):
        raise RuntimeError("Stage28 fold4 artifact grid is incomplete or duplicated")
    artifacts = {}; provenance = {}
    for key in sorted(expected):
        artifacts[key] = load_artifact(paths[key], *key, tokens, inputs, calibration)
        provenance[f"{key[0]}:ns{key[1]}"] = {
            "path": str(paths[key].resolve()), "sha256": file_sha256(paths[key])
        }
    summaries = {}
    for system in SYSTEMS[1:]:
        deltas=[]; bases=[]; logs=[]; noise_values={noise:[] for noise in NOISES}
        component_values={name:[] for name in GUARD_COMPONENTS}
        for noise in NOISES:
            for token in tokens:
                base=artifacts[("P",noise)][token]; current=artifacts[(system,noise)][token]
                base_score=float(base["selected_reward"]); delta=float(current["selected_reward"])-base_score
                deltas.append(delta); bases.append(base_score); logs.append(logs_by_token[token]); noise_values[noise].append(delta)
                for name in GUARD_COMPONENTS:
                    component_values[name].append(
                        float(current["selected_components"][name])-float(base["selected_components"][name])
                    )
        summaries[system] = summarize(
            system, np.asarray(deltas), np.asarray(bases), logs, noise_values, component_values
        )
    eligible = [item for item in summaries.values() if item["eligible"]]
    branch_order = {"A": 0, "B": 1, "C": 2}
    selected = max(eligible, key=lambda item: (
        item["whole_log_bootstrap_ci"][0], item["trimmed10_mean_delta"],
        item["pooled_mean_delta"], -item["catastrophic_count"],
        -item["epoch"], -branch_order[item["branch"]],
    )) if eligible else None
    result = {
        "schema_version": 1, "stage": 28, "phase": "fold4", "passed": selected is not None,
        "stop_before_fold5": selected is None,
        "input_freeze": str(args.inputs.resolve()), "input_freeze_sha256": inputs_sha,
        "gate_definition": "stage28_public_paired_uplift_fold4_v1",
        "bootstrap": {"unit": "whole_log", "seed": BOOTSTRAP_SEED, "samples": BOOTSTRAP_SAMPLES},
        "catastrophic_upper_definition": "one_sided_95_wilson",
        "selected_system": selected["system"] if selected else None,
        "selected_branch": selected["branch"] if selected else None,
        "selected_epoch": selected["epoch"] if selected else None,
        "selected_checkpoint": inputs["systems"][selected["system"]] if selected else None,
        "eligible_systems": [item["system"] for item in eligible],
        "summaries": summaries, "provenance": provenance,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Stage28 gate: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if selected is None:
        raise SystemExit("Stage28 stops: no checkpoint passed the frozen fold4 gate")


if __name__ == "__main__":
    main()
