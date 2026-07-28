#!/usr/bin/env python3
"""Calibrate the Stage25 relative-harm ensemble on the development fold."""

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


def incremental_mode_grid(
    base_eligible: np.ndarray,
    risk_ucb: np.ndarray,
    delta_lcb: np.ndarray,
    fallbacks: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Return incremental sweep choices; kept small and explicit for exact tests."""
    scene_count, _ = base_eligible.shape
    selected = np.asarray(fallbacks, dtype=np.int64).copy()
    best = np.full(scene_count, -np.inf, dtype=np.float64)
    events = []
    for scene_index in range(scene_count):
        for mode in np.flatnonzero(
            base_eligible[scene_index] & (risk_ucb[scene_index] <= 1.0)
        ):
            events.append((
                float(risk_ucb[scene_index, mode]), scene_index, int(mode)
            ))
    events.sort(key=lambda item: (item[0], item[1], item[2]))
    event_index = 0
    output = []
    for threshold in thresholds:
        while event_index < len(events) and events[event_index][0] <= threshold:
            _, scene_index, mode = events[event_index]
            score = float(delta_lcb[scene_index, mode])
            old_mode = int(selected[scene_index])
            if score > best[scene_index] or (
                score == best[scene_index] and mode < old_mode
            ):
                selected[scene_index] = mode
                best[scene_index] = score
            event_index += 1
        output.append(selected.copy())
    return np.stack(output, axis=0)


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
            raise RuntimeError("Stage25 checkpoint lacks training embedding moments")
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
    parser.add_argument(
        "--profile",
        choices=("stage26", "stage27_phase2"),
        default="stage26",
    )
    args = parser.parse_args()
    if not args.selector_checkpoint.is_file():
        raise FileNotFoundError(args.selector_checkpoint)
    if not 0.5 < args.residual_quantile < 1.0:
        raise ValueError("residual quantile must lie in (0.5,1)")
    if not 0.9 < args.ood_quantile < 1.0:
        raise ValueError("OOD quantile must lie in (0.9,1)")
    stage27_profile = args.profile == "stage27_phase2"
    guard_indices = (0, 1, 3, 4, 5) if stage27_profile else SAFETY_INDICES

    selector_sha = file_sha256(args.selector_checkpoint)
    ood_mean, ood_variance = load_ood_moments(args.selector_checkpoint)
    scenes = []
    seen = set()
    artifact_provenance = []
    for path in args.artifact:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        selector_summary = summary.get("stage25_selector", {})
        if selector_summary.get("checkpoint_sha256") != selector_sha:
            raise RuntimeError("Stage25 calibration selector SHA mismatch")
        if not selector_summary.get("calibration_collection"):
            raise RuntimeError("Stage25 calibration artifact is not raw collection")
        domain = str(summary.get("generator_domain", ""))
        namespace = int(summary.get("evaluation_noise_namespace", -999))
        if not domain:
            raise RuntimeError("Stage24 calibration artifact lacks generator domain")
        for record in payload.get("records", []):
            token = str(record.get("token", ""))
            log_name = str(record.get("log_name", ""))
            key = (domain, namespace, token)
            if not token or not log_name or key in seen:
                raise RuntimeError("Stage25 calibration has duplicate/invalid provenance")
            seen.add(key)
            diagnostic = record.get("stage24_selector")
            rewards = np.asarray(record.get("candidate_rewards"), dtype=np.float64)
            components = np.asarray(record.get("candidate_components"), dtype=np.float64)
            if diagnostic is None or rewards.shape != (20,) or components.shape != (20, 6):
                raise RuntimeError("Stage24 calibration record lacks all-20 labels")
            harm = np.asarray(
                diagnostic["harm_probabilities"], dtype=np.float64
            )
            catastrophe = np.asarray(
                diagnostic["catastrophe_probabilities"], dtype=np.float64
            )
            delta = np.asarray(diagnostic["delta_predictions"], dtype=np.float64)
            embedding = np.asarray(diagnostic["embedding_mean"], dtype=np.float64)
            if (
                harm.shape != (20, 8, 3)
                or catastrophe.shape != (20, 8)
                or delta.shape != (20, 8)
                or embedding.shape != (20, ood_mean.size)
            ):
                raise RuntimeError("Stage25 calibration prediction shape mismatch")
            fallback = int(diagnostic["fallback_mode"])
            scenes.append({
                "domain": domain,
                "namespace": namespace,
                "token": token,
                "log_name": log_name,
                "fallback": fallback,
                "reward": rewards,
                "component": components,
                "harm": harm,
                "catastrophe": catastrophe,
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
            scene["catastrophe"], scene["harm"].max(axis=-1)
        )
        risk_ucb = risk_members.mean(axis=-1) + args.confidence_z * risk_members.std(axis=-1)
        delta_lcb = (
            scene["delta"].mean(axis=-1)
            - args.confidence_z * scene["delta"].std(axis=-1)
            - residual_margin
        )
        ood = np.mean(
            (scene["embedding"] - ood_mean) ** 2 / ood_variance, axis=-1
        )
        base_eligible = (
            (delta_lcb > 0)
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
    scene_count = len(prepared)
    group_keys = sorted({
        (scene["domain"], scene["namespace"]) for scene in prepared
    })
    group_lookup = {key: index for index, key in enumerate(group_keys)}
    scene_groups = np.asarray([
        group_lookup[(scene["domain"], scene["namespace"])]
        for scene in prepared
    ], dtype=np.int64)
    group_counts = np.bincount(
        scene_groups, minlength=len(group_keys)
    ).astype(np.float64)

    # The chosen mode for a scene changes only when a newly admitted challenger
    # has a larger delta LCB. Process those risk breakpoints incrementally
    # instead of replaying every scene for every threshold.
    events = []
    for scene_index, scene in enumerate(prepared):
        for mode in np.flatnonzero(
            scene["base_eligible"] & (scene["risk_ucb"] <= 1.0)
        ):
            events.append((
                float(scene["risk_ucb"][mode]), scene_index, int(mode)
            ))
    events.sort(key=lambda item: (item[0], item[1], item[2]))
    selected_modes = np.asarray(
        [scene["fallback"] for scene in prepared], dtype=np.int64
    )
    best_values = np.full(scene_count, -np.inf, dtype=np.float64)
    current_deltas = np.zeros(scene_count, dtype=np.float64)
    current_components = np.zeros((scene_count, 6), dtype=np.float64)
    current_catastrophic = np.zeros(scene_count, dtype=bool)
    group_sums = np.zeros(len(group_keys), dtype=np.float64)
    component_sum = np.zeros(6, dtype=np.float64)
    total_delta = 0.0
    switches = 0
    catastrophes = 0
    event_index = 0
    candidates = []
    for threshold in thresholds:
        while event_index < len(events) and events[event_index][0] <= threshold:
            _, scene_index, mode = events[event_index]
            scene = prepared[scene_index]
            score = float(scene["delta_lcb"][mode])
            old_mode = int(selected_modes[scene_index])
            if score > best_values[scene_index] or (
                score == best_values[scene_index] and mode < old_mode
            ):
                fallback = scene["fallback"]
                new_delta = float(
                    scene["reward"][mode] - scene["reward"][fallback]
                )
                new_component = (
                    scene["component"][mode] - scene["component"][fallback]
                )
                new_catastrophic = bool(
                    new_delta <= -0.5
                )
                if not stage27_profile:
                    new_catastrophic = bool(
                        new_catastrophic or np.any(
                            new_component[list(SAFETY_INDICES)] < -0.0005
                        )
                    )
                old_delta = current_deltas[scene_index]
                old_component = current_components[scene_index].copy()
                old_catastrophic = bool(
                    current_catastrophic[scene_index]
                )
                delta_change = new_delta - old_delta
                total_delta += delta_change
                component_sum += new_component - old_component
                group_sums[scene_groups[scene_index]] += delta_change
                catastrophes += int(new_catastrophic) - int(old_catastrophic)
                if old_mode == fallback:
                    switches += 1
                selected_modes[scene_index] = mode
                best_values[scene_index] = score
                current_deltas[scene_index] = new_delta
                current_components[scene_index] = new_component
                current_catastrophic[scene_index] = new_catastrophic
            event_index += 1
        group_means = {
            f"{domain}:{namespace}": float(group_sums[index] / group_counts[index])
            for index, (domain, namespace) in enumerate(group_keys)
        }
        mean_gain = total_delta / scene_count
        cp_upper = clopper_pearson_upper(catastrophes, scene_count)
        safety_means = component_sum[list(guard_indices)] / scene_count
        if stage27_profile:
            checks = {
                "mean_gain_at_least_0.003": bool(mean_gain >= 0.003),
                "every_domain_namespace_nonnegative": bool(
                    all(value >= 0 for value in group_means.values())
                ),
                "switch_rate_at_least_0.01": bool(
                    switches / scene_count >= 0.01
                ),
                "switch_rate_at_most_0.15": bool(
                    switches / scene_count <= 0.15
                ),
                "guard_components_no_worse_0.0005": bool(
                    np.all(safety_means >= -0.0005)
                ),
                "catastrophic_count_zero": bool(catastrophes == 0),
            }
        else:
            checks = {
                "mean_gain_at_least_0.005": bool(mean_gain >= 0.005),
                "every_domain_namespace_nonnegative": bool(
                    all(value >= 0 for value in group_means.values())
                ),
                "switch_rate_at_least_0.02": bool(
                    switches / scene_count >= 0.02
                ),
                "safety_no_worse_0.0005": bool(
                    np.all(safety_means >= -0.0005)
                ),
                "catastrophic_upper_at_most_0.0025": bool(
                    cp_upper <= 0.0025
                ),
            }
        candidates.append({
            "threshold": float(threshold),
            "mean_gain": float(mean_gain),
            "switch_count": switches, "switch_rate": switches / scene_count,
            "catastrophic_count": catastrophes,
            "catastrophic_cp_upper": cp_upper,
            "safety_component_mean_deltas": safety_means.tolist(),
            "group_means": group_means, "checks": checks,
        })

    def materialize(threshold):
        deltas = np.zeros(scene_count, dtype=np.float64)
        for scene_index, scene in enumerate(prepared):
            eligible = (
                scene["base_eligible"]
                & (scene["risk_ucb"] <= threshold)
            )
            if eligible.any():
                mode = int(np.argmax(np.where(
                    eligible, scene["delta_lcb"], -np.inf
                )))
                deltas[scene_index] = (
                    scene["reward"][mode]
                    - scene["reward"][scene["fallback"]]
                )
        return deltas, [scene["log_name"] for scene in prepared]

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
        candidate_deltas, candidate_logs = materialize(
            candidate["threshold"]
        )
        if not np.isclose(
            candidate_deltas.mean(), candidate["mean_gain"], atol=1e-12
        ):
            raise RuntimeError("Stage25 incremental calibration drifted")
        ci = whole_log_bootstrap_ci(
            candidate_deltas, candidate_logs,
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
            selected_deltas, selected_logs = materialize(
                selected["threshold"]
            )
            if not np.isclose(
                selected_deltas.mean(), selected["mean_gain"], atol=1e-12
            ):
                raise RuntimeError("Stage25 incremental calibration drifted")
            selected["whole_log_bootstrap_ci"] = whole_log_bootstrap_ci(
                selected_deltas, selected_logs,
                args.bootstrap_samples, args.bootstrap_seed,
            )
        selected["checks"]["whole_log_ci_positive"] = bool(
            selected["whole_log_bootstrap_ci"][0] > 0
        )
        selected["passed"] = bool(all(selected["checks"].values()))

    passed = bool(selected["passed"])
    result = {
        "schema_version": 1,
        "stage": 25,
        "passed": passed,
        "selector_checkpoint": str(args.selector_checkpoint),
        "selector_checkpoint_sha256": selector_sha,
        "calibration_definition": "relative_harm_full_eight_member_ensemble",
        "calibration_profile": args.profile,
        "guard_component_indices": list(guard_indices),
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
