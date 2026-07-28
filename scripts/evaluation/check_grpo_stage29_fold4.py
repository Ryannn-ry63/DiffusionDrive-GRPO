#!/usr/bin/env python3
"""Apply the frozen Stage29 reference-anchored fold4 selection gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


PUBLIC_SHA = "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
AMENDMENT_SHA = "941eef5f370f90ad9e5b9c7ed7a1dbaa12a971f30e79a211c446098f576f3bd9"
SIGNAL_GATE_DEFINITION = "stage29_prefold4_amended_dense_scene_v1"
COMPONENTS = ("collision", "drivable", "progress", "ttc", "comfort", "direction")
SYSTEMS = ("P", "A3", "H1", "H2", "H3", "H4", "HC1", "HC2", "HC3", "HC4")
SELECTABLE_SYSTEMS = SYSTEMS[2:]
NOISES = (20261211, 20261212)
BOOTSTRAP_SEED = 20261229
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
        raise RuntimeError("Stage29 fold4 whole-log count drifted")
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


def trim10_mean(values: np.ndarray) -> float:
    ordered = np.sort(values)
    trim = int(math.floor(0.1 * ordered.size))
    return float(ordered[trim:-trim].mean())


def load_manifest(path: Path, expected_sha: str) -> tuple[list[str], dict[str, str]]:
    if file_sha256(path) != expected_sha:
        raise RuntimeError("Stage29 fold4 manifest SHA drifted")
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
        raise RuntimeError("Stage29 fold4 manifest semantics drifted")
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
        raise RuntimeError(f"Stage29 artifact {system}/ns{noise} failed {failed}")
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
            raise RuntimeError(f"Stage29 invalid record: {system}/{noise}/{token}")
        by_token[token] = record
    return by_token


def summarize(
    system: str,
    metadata: dict,
    selected_deltas: np.ndarray,
    candidate_mean_deltas: np.ndarray,
    oracle_deltas: np.ndarray,
    fallback_oracle_deltas: np.ndarray,
    base_scores: np.ndarray,
    logs: list[str],
    noise_values: dict[int, list[float]],
    component_values: dict[str, list[float]],
) -> dict:
    vectors = (selected_deltas, candidate_mean_deltas, oracle_deltas, fallback_oracle_deltas)
    if any(vector.shape != (2042,) or not np.isfinite(vector).all() for vector in vectors):
        raise RuntimeError(f"Stage29 paired vector drifted for {system}")
    ci = bootstrap_whole_log(selected_deltas, logs)
    trimmed = trim10_mean(selected_deltas)
    exact_ties = selected_deltas == 0.0
    wins = int(np.sum(selected_deltas > 0.0))
    losses = int(np.sum(selected_deltas < 0.0))
    hard = base_scores < 0.75
    mature = ~hard
    if not hard.any() or not mature.any():
        raise RuntimeError("Stage29 hard/mature split is empty")
    cats = int(np.sum(selected_deltas <= -0.5))
    component_means = {
        name: float(np.mean(component_values[name])) for name in COMPONENTS
    }
    namespace_means = {
        f"ns{noise}": float(np.mean(noise_values[noise])) for noise in NOISES
    }
    pooled = float(selected_deltas.mean())
    candidate_mean = float(candidate_mean_deltas.mean())
    oracle_mean = float(oracle_deltas.mean())
    checks = {
        "pooled_gain_at_least_0.003": pooled >= 0.003,
        "whole_log_ci_lower_strictly_positive": ci[0] > 0.0,
        "both_namespace_means_strictly_positive": all(
            value > 0.0 for value in namespace_means.values()
        ),
        "trimmed10_mean_nonnegative": trimmed >= 0.0,
        "wins_exceed_losses_excluding_exact_ties": wins > losses,
        "hard_scene_gain_at_least_0.010": float(selected_deltas[hard].mean()) >= 0.010,
        "mature_scene_gain_at_least_negative_0.0002": float(selected_deltas[mature].mean()) >= -0.0002,
        "candidate_mean_delta_strictly_positive": candidate_mean > 0.0,
        "oracle_candidate_delta_at_least_negative_0.0005": oracle_mean >= -0.0005,
        "all_components_at_least_negative_0.0005": all(
            value >= -0.0005 for value in component_means.values()
        ),
        "catastrophic_count_zero": cats == 0,
        "finite_and_provenance_complete": True,
    }
    selectable = bool(metadata.get("selectable"))
    return {
        "system": system,
        "branch": metadata["branch"],
        "epoch": metadata["epoch"],
        "selectable": selectable,
        "comparison": f"{system}-P",
        "pooled_mean_delta": pooled,
        "strong_gain_at_least_0.005": pooled >= 0.005,
        "whole_log_bootstrap_ci": ci,
        "trimmed10_mean_delta": trimmed,
        "namespace_mean_delta": namespace_means,
        "candidate_mean_delta": candidate_mean,
        "oracle_candidate_mean_delta": oracle_mean,
        "base_fallback_oracle_mean_delta": float(fallback_oracle_deltas.mean()),
        "hard_scene_count": int(hard.sum()),
        "hard_scene_mean_delta": float(selected_deltas[hard].mean()),
        "mature_scene_count": int(mature.sum()),
        "mature_scene_mean_delta": float(selected_deltas[mature].mean()),
        "component_mean_deltas": component_means,
        "wins": wins,
        "exact_ties": int(exact_ties.sum()),
        "losses": losses,
        "worst_delta": float(selected_deltas.min()),
        "catastrophic_count": cats,
        "catastrophic_rate": float(cats / selected_deltas.size),
        "checks": checks,
        "passes_numeric_gate": bool(all(checks.values())),
        "eligible": bool(selectable and all(checks.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument(
        "--artifact", action="append", nargs=3, required=True,
        metavar=("SYSTEM", "NOISE", "PATH"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs_sha = file_sha256(args.inputs)
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    if not (
        inputs.get("passed") and inputs.get("stage") == 29
        and inputs.get("phase") == "fold4"
        and tuple(inputs.get("noise_namespaces", ())) == NOISES
        and tuple(inputs.get("systems", {}).keys()) == SYSTEMS
        and inputs.get("all_evaluations_use_safe_multi") is True
        and inputs.get("historical_control_not_selectable") is True
        and inputs.get("training_signal_gate_definition") == SIGNAL_GATE_DEFINITION
        and inputs.get("training_signal_gate_amendment_sha256") == AMENDMENT_SHA
        and inputs["systems"]["A3"].get("selectable") is False
        and all(inputs["systems"][name].get("selectable") is True for name in SELECTABLE_SYSTEMS)
    ):
        raise RuntimeError("Stage29 fold4 input-freeze semantics drifted")
    amendment_path = Path(inputs["training_signal_gate_amendment"])
    if not amendment_path.is_file() or file_sha256(amendment_path) != AMENDMENT_SHA:
        raise RuntimeError("Stage29 fold4 signal-gate amendment SHA drifted")
    tokens, logs_by_token = load_manifest(
        Path(inputs["manifest"]), inputs["manifest_sha256"]
    )
    calibration_path = Path(inputs["calibration"])
    if file_sha256(calibration_path) != inputs["calibration_sha256"]:
        raise RuntimeError("Stage29 calibration SHA drifted")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not calibration.get("passed"):
        raise RuntimeError("Stage29 safe selector calibration did not pass")
    paths = {
        (system, int(noise)): Path(path) for system, noise, path in args.artifact
    }
    expected = {(system, noise) for system in SYSTEMS for noise in NOISES}
    if set(paths) != expected or len(paths) != len(args.artifact):
        raise RuntimeError("Stage29 fold4 artifact grid is incomplete or duplicated")
    artifacts = {}
    provenance = {}
    for key in sorted(expected):
        artifacts[key] = load_artifact(paths[key], *key, tokens, inputs, calibration)
        provenance[f"{key[0]}:ns{key[1]}"] = {
            "path": str(paths[key].resolve()),
            "sha256": file_sha256(paths[key]),
        }

    summaries = {}
    for system in SYSTEMS[1:]:
        selected_deltas = []
        candidate_mean_deltas = []
        oracle_deltas = []
        fallback_oracle_deltas = []
        bases = []
        logs = []
        noise_values = {noise: [] for noise in NOISES}
        component_values = {name: [] for name in COMPONENTS}
        for noise in NOISES:
            for token in tokens:
                base = artifacts[("P", noise)][token]
                current = artifacts[(system, noise)][token]
                base_score = float(base["selected_reward"])
                current_score = float(current["selected_reward"])
                delta = current_score - base_score
                base_candidates = np.asarray(base["candidate_rewards"], dtype=np.float64)
                current_candidates = np.asarray(current["candidate_rewards"], dtype=np.float64)
                selected_deltas.append(delta)
                candidate_mean_deltas.append(float(current_candidates.mean() - base_candidates.mean()))
                oracle_deltas.append(float(current_candidates.max() - base_candidates.max()))
                fallback_oracle_deltas.append(max(current_score, base_score) - base_score)
                bases.append(base_score)
                logs.append(logs_by_token[token])
                noise_values[noise].append(delta)
                for name in COMPONENTS:
                    component_values[name].append(
                        float(current["selected_components"][name])
                        - float(base["selected_components"][name])
                    )
        summaries[system] = summarize(
            system,
            inputs["systems"][system],
            np.asarray(selected_deltas, dtype=np.float64),
            np.asarray(candidate_mean_deltas, dtype=np.float64),
            np.asarray(oracle_deltas, dtype=np.float64),
            np.asarray(fallback_oracle_deltas, dtype=np.float64),
            np.asarray(bases, dtype=np.float64),
            logs,
            noise_values,
            component_values,
        )

    eligible = [summaries[name] for name in SELECTABLE_SYSTEMS if summaries[name]["eligible"]]
    branch_order = {"H": 0, "HC": 1}
    selected = max(
        eligible,
        key=lambda item: (
            item["whole_log_bootstrap_ci"][0],
            item["trimmed10_mean_delta"],
            item["pooled_mean_delta"],
            -item["epoch"],
            -branch_order[item["branch"]],
        ),
    ) if eligible else None
    result = {
        "schema_version": 1,
        "stage": 29,
        "phase": "fold4",
        "passed": selected is not None,
        "stop_before_fold5": selected is None,
        "input_freeze": str(args.inputs.resolve()),
        "input_freeze_sha256": inputs_sha,
        "gate_definition": "stage29_reference_anchored_headroom_fold4_v1",
        "bootstrap": {
            "unit": "whole_log", "seed": BOOTSTRAP_SEED,
            "samples": BOOTSTRAP_SAMPLES,
        },
        "selected_system": selected["system"] if selected else None,
        "selected_branch": selected["branch"] if selected else None,
        "selected_epoch": selected["epoch"] if selected else None,
        "selected_checkpoint": inputs["systems"][selected["system"]] if selected else None,
        "selected_strong_gate": selected["strong_gain_at_least_0.005"] if selected else False,
        "eligible_systems": [item["system"] for item in eligible],
        "historical_a3_control": summaries["A3"],
        "summaries": summaries,
        "provenance": provenance,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Stage29 gate: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if selected is None:
        raise SystemExit("Stage29 stops: no H/HC checkpoint passed the frozen fold4 gate")


if __name__ == "__main__":
    main()
