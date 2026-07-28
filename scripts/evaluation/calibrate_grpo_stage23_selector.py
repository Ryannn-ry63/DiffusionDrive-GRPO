#!/usr/bin/env python3
"""Calibrate Stage23 only from whole-log out-of-fold member predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from navsim.agents.diffusiondrive.stage23_trajectory_selector import (
    PDM_SAFETY_INDICES,
    deterministic_log_bucket,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clopper_pearson_upper(events: int, trials: int, alpha: float = 0.05) -> float:
    if trials <= 0 or events < 0 or events > trials:
        raise ValueError("invalid binomial counts")
    if events == trials:
        return 1.0
    return float(beta.ppf(1.0 - alpha, events + 1, trials - events))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--residual-quantile", type=float, default=0.95)
    parser.add_argument("--maximum-catastrophic-upper", type=float, default=0.005)
    parser.add_argument("--safety-tolerance", type=float, default=0.001)
    args = parser.parse_args()
    if not 0.5 < args.residual_quantile < 1.0:
        raise ValueError("residual quantile must lie in (0.5,1)")

    scenes = []
    selector_sha = None
    artifact_provenance = []
    seen = set()
    for path in args.artifact:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        stage23 = summary.get("stage23_selector", {})
        current_sha = stage23.get("checkpoint_sha256")
        if not current_sha:
            raise RuntimeError(f"artifact lacks Stage23 selector provenance: {path}")
        if selector_sha is None:
            selector_sha = current_sha
        elif selector_sha != current_sha:
            raise RuntimeError("calibration artifacts use different selector checkpoints")
        namespace = int(summary.get("evaluation_noise_namespace", -999))
        for record in payload.get("records", []):
            token = str(record.get("token", ""))
            log_name = str(record.get("log_name", ""))
            key = (namespace, token)
            if not token or not log_name or key in seen:
                raise RuntimeError("calibration requires unique token/log/namespace records")
            seen.add(key)
            diagnostic = record.get("stage23_selector")
            rewards = np.asarray(record.get("candidate_rewards"), dtype=float)
            components = np.asarray(record.get("candidate_components"), dtype=float)
            if diagnostic is None or rewards.shape != (20,) or components.shape != (20, 6):
                raise RuntimeError("calibration record lacks all-20 Stage23 targets")
            member_scores = np.asarray(diagnostic["score_predictions"], dtype=float)
            member_components = np.asarray(
                diagnostic["component_predictions"], dtype=float
            )
            if member_scores.shape != (20, 5) or member_components.shape != (20, 5, 6):
                raise RuntimeError("calibration record lacks five-member predictions")
            held_out = int(deterministic_log_bucket((log_name,), 5).item())
            fallback = int(record["reference_mode"])
            scenes.append(
                {
                    "namespace": namespace,
                    "fallback": fallback,
                    "reward": rewards,
                    "component": components,
                    "score": member_scores[:, held_out],
                    "pred_component": member_components[:, held_out],
                }
            )
        artifact_provenance.append(
            {"path": str(path), "sha256": _sha256(path), "namespace": namespace}
        )
    if not scenes:
        raise RuntimeError("no calibration scenes")

    # One-sided residual conformal margin: predicted improvement minus truth.
    residuals = []
    for scene in scenes:
        fallback = scene["fallback"]
        predicted_delta = scene["score"] - scene["score"][fallback]
        true_delta = scene["reward"] - scene["reward"][fallback]
        residuals.extend((predicted_delta - true_delta).tolist())
    residual_margin = max(0.0, float(np.quantile(residuals, args.residual_quantile)))

    threshold_grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.5, 0.999, 500),
                np.asarray(
                    [
                        scene["pred_component"][:, PDM_SAFETY_INDICES].min(axis=-1)
                        for scene in scenes
                    ]
                ).reshape(-1),
            ]
        )
    )
    # Sweep thresholds from high to low. A mode can become eligible only when
    # the threshold crosses its minimum predicted safety component, so this is
    # exactly equivalent to evaluating every scene at every threshold without
    # the O(num_thresholds * num_scenes * num_modes) Python loop.
    num_scenes = len(scenes)
    events = []
    true_deltas = []
    component_deltas = []
    catastrophic = []
    for scene_idx, scene in enumerate(scenes):
        fallback = scene["fallback"]
        advantage = scene["score"] - scene["score"][fallback] - residual_margin
        predicted_safety = scene["pred_component"][:, PDM_SAFETY_INDICES]
        relative_safe = (predicted_safety >= predicted_safety[fallback]).all(axis=-1)
        safety_floor = predicted_safety.min(axis=-1)
        delta = scene["reward"] - scene["reward"][fallback]
        component_delta = scene["component"] - scene["component"][fallback]
        catastrophic_mode = (
            (delta <= -0.5)
            | (
                component_delta[:, list(PDM_SAFETY_INDICES)]
                < -args.safety_tolerance
            ).any(axis=-1)
        )
        valid = relative_safe & (advantage > 0)
        valid[fallback] = False
        for mode in np.flatnonzero(valid):
            events.append(
                (float(safety_floor[mode]), scene_idx, float(advantage[mode]), int(mode))
            )
        true_deltas.append(delta)
        component_deltas.append(component_delta)
        catastrophic.append(catastrophic_mode)

    events.sort(key=lambda item: item[0], reverse=True)
    thresholds = np.sort(threshold_grid)[::-1]
    true_deltas = np.asarray(true_deltas)
    component_deltas = np.asarray(component_deltas)
    catastrophic = np.asarray(catastrophic, dtype=bool)
    selected_modes = np.full(num_scenes, -1, dtype=np.int64)
    selected_advantages = np.full(num_scenes, -np.inf, dtype=np.float64)
    total_gain = 0.0
    total_component_delta = np.zeros(6, dtype=np.float64)
    switch_count = 0
    catastrophic_count = 0
    event_idx = 0
    cp_cache = {}
    selected = None

    for threshold in thresholds:
        while event_idx < len(events) and events[event_idx][0] >= threshold:
            _, scene_idx, advantage, mode = events[event_idx]
            previous_mode = selected_modes[scene_idx]
            previous_advantage = selected_advantages[scene_idx]
            if advantage > previous_advantage or (
                advantage == previous_advantage
                and (previous_mode < 0 or mode < previous_mode)
            ):
                if previous_mode >= 0:
                    total_gain -= float(true_deltas[scene_idx, previous_mode])
                    total_component_delta -= component_deltas[scene_idx, previous_mode]
                    catastrophic_count -= int(catastrophic[scene_idx, previous_mode])
                else:
                    switch_count += 1
                selected_modes[scene_idx] = mode
                selected_advantages[scene_idx] = advantage
                total_gain += float(true_deltas[scene_idx, mode])
                total_component_delta += component_deltas[scene_idx, mode]
                catastrophic_count += int(catastrophic[scene_idx, mode])
            event_idx += 1

        if catastrophic_count not in cp_cache:
            cp_cache[catastrophic_count] = clopper_pearson_upper(
                catastrophic_count, num_scenes
            )
        upper = cp_cache[catastrophic_count]
        safety_means = total_component_delta[list(PDM_SAFETY_INDICES)] / num_scenes
        candidate = {
            "threshold": float(threshold),
            "passed": bool(
                switch_count > 0
                and upper <= args.maximum_catastrophic_upper
                and np.all(safety_means >= -args.safety_tolerance)
            ),
            "mean_gain": float(total_gain / num_scenes),
            "switch_count": int(switch_count),
            "catastrophic_count": int(catastrophic_count),
            "catastrophic_cp_upper": float(upper),
            "safety_component_mean_deltas": safety_means.tolist(),
        }
        if candidate["passed"] and (
            selected is None
            or (candidate["mean_gain"], candidate["threshold"])
            > (selected["mean_gain"], selected["threshold"])
        ):
            selected = candidate

    if selected is None:
        raise RuntimeError("no Stage23 OOF threshold satisfies the preregistered safety bound")
    result = {
        "schema_version": 1,
        "passed": True,
        "selector_checkpoint_sha256": selector_sha,
        "calibration_definition": "whole_log_single_held_out_member",
        "num_scenes": len(scenes),
        "num_artifacts": len(args.artifact),
        "residual_quantile": args.residual_quantile,
        "residual_margin": residual_margin,
        "maximum_catastrophic_upper": args.maximum_catastrophic_upper,
        "num_thresholds": int(len(thresholds)),
        "selected": selected,
        "artifacts": artifact_provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
