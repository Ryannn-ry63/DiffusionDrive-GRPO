#!/usr/bin/env python3
"""Strict one-shot Stage26 fold5 report for B0, B, and C."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta


BASE_SHA = "59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d"
FINAL_SHA = "c0baa2c42023727195f7f28e83e1e9532783e572ddbc8fc60d9ed7a25409ec2b"
SELECTOR_SHA = "77ee768c5f469292d81049192d48797f2f3fae4f80029ae6853dd9d9c6de4207"
CALIBRATION_SHA = (
    "0c14b8f9531d64028bdf4d71f35b34418ab31ad4ae11ccc6b3c60a4325f53a0d"
)
MANIFEST_SHA = "2424fe88319f81650eec1b8e2dbe265143d5d6d7cc8d113e104caf8d150cf2d4"
NOISES = (-1, 20260823, 20260824, 20260825)
NOISE_NAMES = {
    -1: "default",
    20260823: "ns20260823",
    20260824: "ns20260824",
    20260825: "ns20260825",
}
COMPONENTS = ("collision", "drivable", "progress", "ttc", "comfort", "direction")
SAFETY = ("collision", "drivable", "ttc")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_sha(tokens: list[str]) -> str:
    return hashlib.sha256("\n".join(tokens).encode()).hexdigest()


def cp_upper(events: int, trials: int) -> float:
    return 1.0 if events == trials else float(
        beta.ppf(0.95, events + 1, trials - events)
    )


def bootstrap_ci(
    values: list[float], logs: list[str], samples: int, seed: int
) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, log_name in zip(values, logs):
        grouped[log_name].append(float(value))
    names = sorted(grouped)
    sums = np.asarray([sum(grouped[name]) for name in names], dtype=np.float64)
    counts = np.asarray([len(grouped[name]) for name in names], dtype=np.int64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 500):
        size = min(500, samples - start)
        indices = rng.integers(0, len(names), (size, len(names)))
        means[start : start + size] = (
            sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
        )
    return np.quantile(means, (0.025, 0.975)).tolist()


def load_manifest(path: Path) -> tuple[list[str], dict[str, str]]:
    if sha256(path) != MANIFEST_SHA:
        raise RuntimeError("protected fold5 manifest SHA drifted")
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
        raise RuntimeError(f"protected fold5 manifest failed {failed}")
    return tokens, logs


def load_artifact(
    path: Path,
    tokens: list[str],
    noise: int,
    checkpoint_sha: str,
    domain: str,
) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    records = payload.get("records", [])
    selector = summary.get("stage25_selector", {})
    schedule = summary.get("schedule", {})
    actual_tokens = [str(record.get("token", "")) for record in records]
    checks = {
        "checkpoint": summary.get("checkpoint_sha256") == checkpoint_sha,
        "reference": summary.get("reference_checkpoint_sha256") == BASE_SHA,
        "domain": summary.get("generator_domain") == domain,
        "noise": int(summary.get("evaluation_noise_namespace", -2)) == noise,
        "selector_source":
            summary.get("selector_logits_source") == "trajectory_relative_harm_v3",
        "selector_sha": selector.get("checkpoint_sha256") == SELECTOR_SHA,
        "calibration_sha": selector.get("calibration_sha256") == CALIBRATION_SHA,
        "schedule": schedule.get("truncation_timestep") == 32
            and schedule.get("roll_timesteps") == [32, 24, 16, 8, 0]
            and schedule.get("scheduler_num_inference_steps") == 125,
        "complete": summary.get("completed") is True
            and int(summary.get("num_failures", -1)) == 0,
        "count": len(records) == 1023,
        "order": actual_tokens == tokens,
        "token_sha": summary.get("token_set_sha256") == ordered_sha(tokens),
        "log_split": summary.get("log_split") == "train",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"fold5 artifact {path} failed {failed}")
    return {str(record["token"]): record for record in records}


def b0_from_b(record: dict) -> tuple[float, dict[str, float]]:
    diagnostic = record["stage24_selector"]
    mode = int(diagnostic["fallback_mode"])
    candidate_reward = float(record["candidate_rewards"][mode])
    derived_reward = (
        float(record["selected_reward"])
        - float(diagnostic["true_selected_minus_fallback"])
    )
    if not np.isclose(candidate_reward, derived_reward, atol=2e-6, rtol=0):
        raise RuntimeError("B0 fallback reward is inconsistent with B artifact")
    values = diagnostic["fallback_components"]
    if len(values) != len(COMPONENTS):
        raise RuntimeError("B0 fallback component schema drifted")
    return candidate_reward, dict(zip(COMPONENTS, map(float, values)))


def summarize_comparison(
    name: str,
    values: list[float],
    logs: list[str],
    namespace_values: dict[int, list[float]],
    safety_values: dict[str, list[float]],
    hard: list[float],
    mature: list[float],
    bootstrap_samples: int,
    seed: int,
) -> dict:
    array = np.asarray(values, dtype=np.float64)
    catastrophes = int((array <= -0.5).sum())
    return {
        "comparison": name,
        "pooled_mean_delta": float(array.mean()),
        "whole_log_bootstrap_ci": bootstrap_ci(
            values, logs, bootstrap_samples, seed
        ),
        "namespace_mean_delta": {
            NOISE_NAMES[noise]: float(np.mean(namespace_values[noise]))
            for noise in NOISES
        },
        "hard_scene_mean_delta": float(np.mean(hard)),
        "mature_scene_mean_delta": float(np.mean(mature)),
        "hard_scene_count": len(hard),
        "mature_scene_count": len(mature),
        "safety_component_mean_deltas": {
            component: float(np.mean(safety_values[component]))
            for component in SAFETY
        },
        "wins": int((array > 0).sum()),
        "ties": int((array == 0).sum()),
        "losses": int((array < 0).sum()),
        "worst_delta": float(array.min()),
        "catastrophic_count": catastrophes,
        "catastrophic_cp_upper": cp_upper(catastrophes, array.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--b", action="append", nargs=2, required=True)
    parser.add_argument("--c", action="append", nargs=2, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.calibration) != CALIBRATION_SHA:
        raise RuntimeError("Stage26 calibration SHA drifted")
    if sha256(args.final_checkpoint) != FINAL_SHA:
        raise RuntimeError("Stage26 final generator SHA drifted")
    tokens, logs_by_token = load_manifest(args.manifest)
    b_paths = {int(noise): Path(path) for noise, path in args.b}
    c_paths = {int(noise): Path(path) for noise, path in args.c}
    if set(b_paths) != set(NOISES) or set(c_paths) != set(NOISES):
        raise RuntimeError("Stage26 fold5 namespace grid drifted")
    b_artifacts = {
        noise: load_artifact(
            b_paths[noise], tokens, noise, BASE_SHA, "official_base"
        )
        for noise in NOISES
    }
    c_artifacts = {
        noise: load_artifact(
            c_paths[noise], tokens, noise, FINAL_SHA, "stage26_final_epoch2"
        )
        for noise in NOISES
    }

    score_values = {"B0": [], "B": [], "C": []}
    comparison_values = {"B-B0": [], "C-B": [], "C-B0": []}
    comparison_logs = {name: [] for name in comparison_values}
    namespace_values = {
        name: {noise: [] for noise in NOISES} for name in comparison_values
    }
    safety_values = {
        name: {component: [] for component in SAFETY}
        for name in comparison_values
    }
    hard_values = {name: [] for name in comparison_values}
    mature_values = {name: [] for name in comparison_values}
    b_switches = []
    b_switch_gains = []
    c_ood_events = 0
    c_ood_trials = 0

    b0_token_mean = {
        token: float(np.mean([
            b0_from_b(b_artifacts[noise][token])[0] for noise in NOISES
        ]))
        for token in tokens
    }
    b_token_mean = {
        token: float(np.mean([
            b_artifacts[noise][token]["selected_reward"] for noise in NOISES
        ]))
        for token in tokens
    }
    for noise in NOISES:
        for token in tokens:
            b_record = b_artifacts[noise][token]
            c_record = c_artifacts[noise][token]
            b0_reward, b0_components = b0_from_b(b_record)
            rewards = {
                "B0": b0_reward,
                "B": float(b_record["selected_reward"]),
                "C": float(c_record["selected_reward"]),
            }
            components = {
                "B0": b0_components,
                "B": {
                    key: float(value)
                    for key, value in b_record["selected_components"].items()
                },
                "C": {
                    key: float(value)
                    for key, value in c_record["selected_components"].items()
                },
            }
            for system in score_values:
                score_values[system].append(rewards[system])
            pairs = {
                "B-B0": ("B", "B0"),
                "C-B": ("C", "B"),
                "C-B0": ("C", "B0"),
            }
            for name, (right, left) in pairs.items():
                delta = rewards[right] - rewards[left]
                comparison_values[name].append(delta)
                comparison_logs[name].append(logs_by_token[token])
                namespace_values[name][noise].append(delta)
                baseline_mean = (
                    b_token_mean[token] if name == "C-B" else b0_token_mean[token]
                )
                (hard_values[name] if baseline_mean < 0.75
                 else mature_values[name]).append(delta)
                for component in SAFETY:
                    safety_values[name][component].append(
                        components[right][component] - components[left][component]
                    )
            switched = bool(b_record["stage24_selector"]["switched"])
            b_switches.append(switched)
            b_switch_gains.append(
                float(b_record["stage24_selector"]["true_selected_minus_fallback"])
            )
            distances = np.asarray(
                c_record["stage24_selector"]["ood_distance"], dtype=np.float64
            )
            if distances.shape != (20,) or not np.isfinite(distances).all():
                raise RuntimeError("Stage26 fold5 C OOD diagnostics drifted")
            c_ood_events += int((distances > 3.460253834058056).sum())
            c_ood_trials += distances.size

    comparisons = {}
    for index, name in enumerate(("B-B0", "C-B", "C-B0")):
        comparisons[name] = summarize_comparison(
            name,
            comparison_values[name],
            comparison_logs[name],
            namespace_values[name],
            safety_values[name],
            hard_values[name],
            mature_values[name],
            args.bootstrap_samples,
            20260831 + index,
        )
    generator_gain = comparisons["C-B"]["pooled_mean_delta"]
    passed = generator_gain >= 0.005
    if generator_gain >= 0.015:
        tier = "desired_target"
    elif generator_gain >= 0.010:
        tier = "paper_target"
    elif generator_gain >= 0.005:
        tier = "minimally_useful"
    else:
        tier = "failed"
    result = {
        "schema_version": 1,
        "stage": 26,
        "passed": passed,
        "stop_before_navtest": not passed,
        "selection_scope": "one_shot_protected_fold5_final_epoch2",
        "decision_rule": "pooled C-B >= +0.005",
        "performance_tier": tier,
        "checkpoint_sha256": FINAL_SHA,
        "selector_checkpoint_sha256": SELECTOR_SHA,
        "calibration_sha256": CALIBRATION_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "noise_namespaces": list(NOISES),
        "num_tokens": len(tokens),
        "num_logs": len(set(logs_by_token.values())),
        "num_paired_observations": len(comparison_values["C-B"]),
        "mean_system_scores": {
            system: float(np.mean(values))
            for system, values in score_values.items()
        },
        "comparisons": comparisons,
        "selector_diagnostics": {
            "switch_count": int(np.sum(b_switches)),
            "switch_rate": float(np.mean(b_switches)),
            "mean_selected_minus_fallback": float(np.mean(b_switch_gains)),
        },
        "generator_candidate_ood_rate": c_ood_events / c_ood_trials,
        "artifacts": {
            "B": {NOISE_NAMES[n]: str(b_paths[n]) for n in NOISES},
            "C": {NOISE_NAMES[n]: str(c_paths[n]) for n in NOISES},
            "B0": "derived exactly from B final-selector fallback mode",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
