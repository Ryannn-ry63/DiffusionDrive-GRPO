#!/usr/bin/env python3
"""Calibrate the complete Stage24 ensemble on a disjoint whole-log fold."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta


SAFETY_INDICES = (0, 1, 3)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clopper_pearson_upper(events: int, trials: int, alpha: float = 0.05) -> float:
    if trials <= 0 or not 0 <= events <= trials:
        raise ValueError("invalid binomial counts")
    if events == trials:
        return 1.0
    return float(beta.ppf(1.0 - alpha, events + 1, trials - events))


def whole_log_bootstrap_ci(
    values: np.ndarray, logs: list[str], samples: int, seed: int,
) -> list[float]:
    grouped = defaultdict(list)
    for value, log_name in zip(values, logs):
        grouped[log_name].append(float(value))
    names = sorted(grouped)
    group_sum = np.asarray([sum(grouped[name]) for name in names], dtype=np.float64)
    group_count = np.asarray([len(grouped[name]) for name in names], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 500):
        count = min(500, samples - start)
        indices = rng.integers(0, len(names), size=(count, len(names)))
        means[start : start + count] = (
            group_sum[indices].sum(axis=1) / group_count[indices].sum(axis=1)
        )
    return np.quantile(means, [0.025, 0.975]).tolist()


def _normalize_state(checkpoint: dict) -> dict:
    if "state_dict" not in checkpoint:
        raise KeyError("Stage24 selector checkpoint has no state_dict")
    return {
        (key[len("agent."):] if key.startswith("agent.") else key): value
        for key, value in checkpoint["state_dict"].items()
    }


def load_ood_moments(checkpoint_path: Path) -> tuple[np.ndarray, np.ndarray]:
    state = _normalize_state(torch.load(checkpoint_path, map_location="cpu"))
    prefix = "_transfuser_model._trajectory_head."
    required = {
        name: state[prefix + name]
        for name in (
            "_stage24_embedding_count",
            "_stage24_embedding_sum",
            "_stage24_embedding_sum_sq",
        )
        if prefix + name in state
    }
    if len(required) != 3:
        raise RuntimeError("Stage24 checkpoint lacks training embedding moments")
    count = int(required["_stage24_embedding_count"].item())
    if count <= 0:
        raise RuntimeError("Stage24 checkpoint has empty embedding moments")
    total = required["_stage24_embedding_sum"].double().numpy()
    total_sq = required["_stage24_embedding_sum_sq"].double().numpy()
    mean = total / count
    variance = np.maximum(total_sq / count - mean ** 2, 1e-6)
    if not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise RuntimeError("Stage24 embedding moments are non-finite")
    return mean, variance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--residual-quantile", type=float, default=0.95)
    parser.add_argument("--ood-quantile", type=float, default=0.995)
    parser.add_argument("--confidence-z", type=float, default=1.96)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    args = parser.parse_args()
    if not args.selector_checkpoint.is_file():
        raise FileNotFoundError(args.selector_checkpoint)
    if not 0.5 < args.residual_quantile < 1.0:
        raise ValueError("residual quantile must lie in (0.5,1)")
    if not 0.9 < args.ood_quantile < 1.0:
        raise ValueError("OOD quantile must lie in (0.9,1)")

    selector_sha = file_sha256(args.selector_checkpoint)
    ood_mean, ood_variance = load_ood_moments(args.selector_checkpoint)
    scenes = []
    seen = set()
    artifact_provenance = []
    for path in args.artifact:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        selector_summary = summary.get("stage24_selector", {})
        if selector_summary.get("checkpoint_sha256") != selector_sha:
            raise RuntimeError("Stage24 calibration selector SHA mismatch")
        if not selector_summary.get("calibration_collection"):
            raise RuntimeError("Stage24 calibration artifact is not raw collection")
        domain = str(summary.get("generator_domain", ""))
        namespace = int(summary.get("evaluation_noise_namespace", -999))
        if not domain:
            raise RuntimeError("Stage24 calibration artifact lacks generator domain")
        for record in payload.get("records", []):
            token = str(record.get("token", ""))
            log_name = str(record.get("log_name", ""))
            key = (domain, namespace, token)
            if not token or not log_name or key in seen:
                raise RuntimeError("Stage24 calibration has duplicate/invalid provenance")
            seen.add(key)
            diagnostic = record.get("stage24_selector")
            rewards = np.asarray(record.get("candidate_rewards"), dtype=np.float64)
            components = np.asarray(record.get("candidate_components"), dtype=np.float64)
            if diagnostic is None or rewards.shape != (20,) or components.shape != (20, 6):
                raise RuntimeError("Stage24 calibration record lacks all-20 labels")
            safety = np.asarray(diagnostic["safety_probabilities"], dtype=np.float64)
            any_unsafe = np.asarray(
                diagnostic["any_unsafe_probabilities"], dtype=np.float64
            )
            component_delta = np.asarray(
                diagnostic["component_delta_predictions"], dtype=np.float64
            )
            delta = np.asarray(diagnostic["delta_predictions"], dtype=np.float64)
            embedding = np.asarray(diagnostic["embedding_mean"], dtype=np.float64)
            if (
                safety.shape != (20, 8, 3)
                or any_unsafe.shape != (20, 8)
                or component_delta.shape != (20, 8, 6)
                or delta.shape != (20, 8)
                or embedding.shape != (20, ood_mean.size)
            ):
                raise RuntimeError("Stage24 calibration prediction shape mismatch")
            fallback = int(diagnostic["fallback_mode"])
            scenes.append({
                "domain": domain,
                "namespace": namespace,
                "token": token,
                "log_name": log_name,
                "fallback": fallback,
                "reward": rewards,
                "component": components,
                "safety": safety,
                "any_unsafe": any_unsafe,
                "component_delta": component_delta,
                "delta": delta,
                "embedding": embedding,
            })
        artifact_provenance.append({
            "path": str(path), "sha256": file_sha256(path),
            "generator_domain": domain, "namespace": namespace,
        })
    if not scenes:
        raise RuntimeError("Stage24 calibration has no scenes")

    residuals = []
    ood_distances = []
    for scene in scenes:
        fallback = scene["fallback"]
        true_delta = scene["reward"] - scene["reward"][fallback]
        predicted = scene["delta"].mean(axis=-1)
        truly_safe = np.all(scene["component"][:, SAFETY_INDICES] >= 0.999, axis=-1)
        residuals.extend((predicted[truly_safe] - true_delta[truly_safe]).tolist())
        ood_distances.extend(np.mean(
            (scene["embedding"] - ood_mean) ** 2 / ood_variance, axis=-1
        ).tolist())
    residual_margin = max(0.0, float(np.quantile(residuals, args.residual_quantile)))
    ood_threshold = float(np.quantile(ood_distances, args.ood_quantile))

    prepared = []
    risk_grid_values = []
    for scene in scenes:
        fallback = scene["fallback"]
        risk_members = np.maximum(
            scene["any_unsafe"], scene["safety"].max(axis=-1)
        )
        risk_ucb = risk_members.mean(axis=-1) + args.confidence_z * risk_members.std(axis=-1)
        delta_lcb = (
            scene["delta"].mean(axis=-1)
            - args.confidence_z * scene["delta"].std(axis=-1)
            - residual_margin
        )
        safety_delta = scene["component_delta"][:, :, SAFETY_INDICES]
        safety_lcb = (
            safety_delta.mean(axis=1)
            - args.confidence_z * safety_delta.std(axis=1)
        )
        ood = np.mean(
            (scene["embedding"] - ood_mean) ** 2 / ood_variance, axis=-1
        )
        base_eligible = (
            (delta_lcb > 0)
            & np.all(safety_lcb >= -0.001, axis=-1)
            & (ood <= ood_threshold)
        )
        base_eligible[fallback] = False
        risk_grid_values.extend(risk_ucb[base_eligible].tolist())
        prepared.append({
            **scene, "risk_ucb": risk_ucb, "delta_lcb": delta_lcb,
            "base_eligible": base_eligible,
        })

    thresholds = np.unique(np.clip(np.concatenate((
        np.linspace(0.001, 0.999, 500), np.asarray(risk_grid_values),
    )), 0.0, 1.0))
    candidates = []
    for threshold in thresholds:
        deltas = []
        component_deltas = []
        logs = []
        switches = 0
        catastrophes = 0
        groups = defaultdict(list)
        for scene in prepared:
            eligible = scene["base_eligible"] & (scene["risk_ucb"] <= threshold)
            selected = int(np.argmax(np.where(eligible, scene["delta_lcb"], -np.inf)))
            if not eligible.any():
                selected = scene["fallback"]
            switched = selected != scene["fallback"]
            delta = float(scene["reward"][selected] - scene["reward"][scene["fallback"]])
            component_delta = (
                scene["component"][selected] - scene["component"][scene["fallback"]]
            )
            catastrophic = switched and (
                delta <= -0.5 or np.any(component_delta[list(SAFETY_INDICES)] < -0.0005)
            )
            deltas.append(delta)
            component_deltas.append(component_delta)
            logs.append(scene["log_name"])
            switches += int(switched)
            catastrophes += int(catastrophic)
            groups[(scene["domain"], scene["namespace"])].append(delta)
        delta_array = np.asarray(deltas, dtype=np.float64)
        component_array = np.asarray(component_deltas, dtype=np.float64)
        group_means = {
            f"{domain}:{namespace}": float(np.mean(values))
            for (domain, namespace), values in sorted(groups.items())
        }
        cp_upper = clopper_pearson_upper(catastrophes, len(deltas))
        safety_means = component_array[:, SAFETY_INDICES].mean(axis=0)
        checks = {
            "mean_gain_at_least_0.005": bool(delta_array.mean() >= 0.005),
            "every_domain_namespace_nonnegative": bool(
                all(value >= 0 for value in group_means.values())
            ),
            "switch_rate_at_least_0.02": bool(switches / len(deltas) >= 0.02),
            "safety_no_worse_0.0005": bool(np.all(safety_means >= -0.0005)),
            "catastrophic_upper_at_most_0.0025": bool(cp_upper <= 0.0025),
        }
        candidates.append({
            "threshold": float(threshold),
            "mean_gain": float(delta_array.mean()),
            "_deltas": delta_array,
            "_logs": logs,
            "switch_count": switches, "switch_rate": switches / len(deltas),
            "catastrophic_count": catastrophes,
            "catastrophic_cp_upper": cp_upper,
            "safety_component_mean_deltas": safety_means.tolist(),
            "group_means": group_means, "checks": checks,
        })

    # Whole-log bootstrap is expensive. The other constraints are exact and
    # cheap, so inspect thresholds in descending gain order and stop at the
    # first one whose CI is positive. This is equivalent to bootstrapping all
    # thresholds and then choosing the highest-gain passing threshold.
    ordered = sorted(
        candidates, key=lambda item: (-item["mean_gain"], item["threshold"])
    )
    selected = None
    for candidate in ordered:
        if not all(candidate["checks"].values()):
            continue
        ci = whole_log_bootstrap_ci(
            candidate["_deltas"], candidate["_logs"],
            args.bootstrap_samples, args.bootstrap_seed,
        )
        candidate["whole_log_bootstrap_ci"] = ci
        candidate["checks"]["whole_log_ci_positive"] = bool(ci[0] > 0)
        candidate["passed"] = bool(all(candidate["checks"].values()))
        if candidate["passed"]:
            selected = candidate
            break

    if selected is None:
        selected = ordered[0]
        if "whole_log_bootstrap_ci" not in selected:
            selected["whole_log_bootstrap_ci"] = whole_log_bootstrap_ci(
                selected["_deltas"], selected["_logs"],
                args.bootstrap_samples, args.bootstrap_seed,
            )
        selected["checks"]["whole_log_ci_positive"] = bool(
            selected["whole_log_bootstrap_ci"][0] > 0
        )
        selected["passed"] = bool(all(selected["checks"].values()))

    passed = bool(selected["passed"])
    selected.pop("_deltas")
    selected.pop("_logs")
    result = {
        "schema_version": 1,
        "stage": 24,
        "passed": passed,
        "selector_checkpoint": str(args.selector_checkpoint),
        "selector_checkpoint_sha256": selector_sha,
        "calibration_definition": "disjoint_fold_full_eight_member_ensemble",
        "num_scenes": len(scenes),
        "num_artifacts": len(args.artifact),
        "residual_quantile": args.residual_quantile,
        "residual_margin": residual_margin,
        "risk_threshold": float(selected["threshold"]),
        "ood_quantile": args.ood_quantile,
        "ood_threshold": ood_threshold,
        "confidence_z": args.confidence_z,
        "ood": {"mean": ood_mean.tolist(), "variance": ood_variance.tolist()},
        "selected": selected,
        "artifacts": artifact_provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
