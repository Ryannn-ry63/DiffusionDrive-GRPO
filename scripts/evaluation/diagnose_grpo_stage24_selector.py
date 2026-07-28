#!/usr/bin/env python3
"""Report the Stage24 selector calibration gating waterfall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from calibrate_grpo_stage24_selector import (
    clopper_pearson_upper,
    load_ood_moments,
)


SAFETY_INDICES = (0, 1, 3)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        str(q): float(np.quantile(values, q))
        for q in (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--confidence-z", type=float, default=1.96)
    parser.add_argument("--residual-quantile", type=float, default=0.95)
    parser.add_argument("--ood-quantile", type=float, default=0.995)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ood_mean, ood_variance = load_ood_moments(args.selector_checkpoint)
    rows = []
    selection_scenes = []
    for path in args.artifact:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        for record in payload["records"]:
            diagnostic = record["stage24_selector"]
            fallback = int(diagnostic["fallback_mode"])
            reward = np.asarray(record["candidate_rewards"], dtype=np.float64)
            component = np.asarray(record["candidate_components"], dtype=np.float64)
            safety = np.asarray(
                diagnostic["safety_probabilities"], dtype=np.float64
            )
            any_unsafe = np.asarray(
                diagnostic["any_unsafe_probabilities"], dtype=np.float64
            )
            component_delta = np.asarray(
                diagnostic["component_delta_predictions"], dtype=np.float64
            )
            delta = np.asarray(
                diagnostic["delta_predictions"], dtype=np.float64
            )
            embedding = np.asarray(
                diagnostic["embedding_mean"], dtype=np.float64
            )
            scene_start = len(rows)
            nonfallback_modes = []
            for mode in range(20):
                if mode == fallback:
                    continue
                nonfallback_modes.append(mode)
                risk_members = np.maximum(
                    any_unsafe[mode], safety[mode].max(axis=-1)
                )
                safety_members = component_delta[mode][:, SAFETY_INDICES]
                rows.append({
                    "domain": summary["generator_domain"],
                    "namespace": summary["evaluation_noise_namespace"],
                    "log_name": record["log_name"],
                    "true_delta": reward[mode] - reward[fallback],
                    "truly_safe": bool(
                        np.all(component[mode, SAFETY_INDICES] >= 0.999)
                    ),
                    "predicted_delta": delta[mode].mean(),
                    "delta_std": delta[mode].std(),
                    "risk_ucb": (
                        risk_members.mean()
                        + args.confidence_z * risk_members.std()
                    ),
                    "safety_lcb_min": np.min(
                        safety_members.mean(axis=0)
                        - args.confidence_z * safety_members.std(axis=0)
                    ),
                    "ood": np.mean(
                        (embedding[mode] - ood_mean) ** 2 / ood_variance
                    ),
                })
            selection_scenes.append({
                "start": scene_start,
                "stop": len(rows),
                "domain": summary["generator_domain"],
                "namespace": summary["evaluation_noise_namespace"],
                "log_name": record["log_name"],
                "component_delta": (
                    component[nonfallback_modes] - component[fallback]
                ),
            })

    arrays = {
        key: np.asarray([row[key] for row in rows])
        for key in (
            "true_delta", "truly_safe", "predicted_delta", "delta_std",
            "risk_ucb", "safety_lcb_min", "ood",
        )
    }
    residual = arrays["predicted_delta"] - arrays["true_delta"]
    safe_residual = residual[arrays["truly_safe"]]
    residual_margin = max(
        0.0, float(np.quantile(safe_residual, args.residual_quantile))
    )
    ood_threshold = float(np.quantile(arrays["ood"], args.ood_quantile))
    delta_lcb = (
        arrays["predicted_delta"]
        - args.confidence_z * arrays["delta_std"]
        - residual_margin
    )
    gates = {
        "delta_lcb_positive": delta_lcb > 0,
        "safety_lcb_at_least_minus_0.001": arrays["safety_lcb_min"] >= -0.001,
        "in_distribution": arrays["ood"] <= ood_threshold,
    }
    gates["all_base_gates"] = np.logical_and.reduce(list(gates.values()))

    sweep = []
    for tolerance in (0.001, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5):
        best = None
        for risk_threshold in np.linspace(0.2, 1.0, 33):
            deltas = []
            component_deltas = []
            groups = {}
            switches = 0
            catastrophes = 0
            for scene in selection_scenes:
                block = slice(scene["start"], scene["stop"])
                eligible = (
                    (delta_lcb[block] > 0)
                    & (arrays["safety_lcb_min"][block] >= -tolerance)
                    & (arrays["ood"][block] <= ood_threshold)
                    & (arrays["risk_ucb"][block] <= risk_threshold)
                )
                if eligible.any():
                    local = int(np.argmax(np.where(
                        eligible, delta_lcb[block], -np.inf
                    )))
                    delta_value = float(arrays["true_delta"][block][local])
                    component_value = scene["component_delta"][local]
                    switches += 1
                    if (
                        delta_value <= -0.5
                        or np.any(
                            component_value[list(SAFETY_INDICES)] < -0.0005
                        )
                    ):
                        catastrophes += 1
                else:
                    delta_value = 0.0
                    component_value = np.zeros(6, dtype=np.float64)
                deltas.append(delta_value)
                component_deltas.append(component_value)
                group = f'{scene["domain"]}:{scene["namespace"]}'
                groups.setdefault(group, []).append(delta_value)
            deltas = np.asarray(deltas, dtype=np.float64)
            component_deltas = np.asarray(component_deltas, dtype=np.float64)
            group_means = {
                key: float(np.mean(values)) for key, values in groups.items()
            }
            safety_means = component_deltas[:, SAFETY_INDICES].mean(axis=0)
            result = {
                "safety_tolerance": tolerance,
                "risk_threshold": float(risk_threshold),
                "mean_gain": float(deltas.mean()),
                "switch_count": switches,
                "switch_rate": switches / len(selection_scenes),
                "catastrophic_count": catastrophes,
                "catastrophic_cp_upper": clopper_pearson_upper(
                    catastrophes, len(selection_scenes)
                ),
                "minimum_group_mean": min(group_means.values()),
                "safety_component_mean_deltas": safety_means.tolist(),
            }
            result["exact_checks_pass"] = bool(
                result["mean_gain"] >= 0.005
                and result["switch_rate"] >= 0.02
                and result["minimum_group_mean"] >= 0
                and np.all(safety_means >= -0.0005)
                and result["catastrophic_cp_upper"] <= 0.0025
            )
            if best is None or (
                result["exact_checks_pass"], result["mean_gain"]
            ) > (
                best["exact_checks_pass"], best["mean_gain"]
            ):
                best = result
        sweep.append(best)

    result = {
        "num_nonfallback_candidates": len(rows),
        "residual_margin": residual_margin,
        "ood_threshold": ood_threshold,
        "quantiles": {
            "true_delta": _quantiles(arrays["true_delta"]),
            "predicted_delta": _quantiles(arrays["predicted_delta"]),
            "delta_std": _quantiles(arrays["delta_std"]),
            "safe_residual": _quantiles(safe_residual),
            "delta_lcb": _quantiles(delta_lcb),
            "risk_ucb": _quantiles(arrays["risk_ucb"]),
            "safety_lcb_min": _quantiles(arrays["safety_lcb_min"]),
            "ood": _quantiles(arrays["ood"]),
        },
        "gate_counts": {
            key: int(mask.sum()) for key, mask in gates.items()
        },
        "gate_rates": {
            key: float(mask.mean()) for key, mask in gates.items()
        },
        "oracle_checks": {
            "positive_true_delta": int((arrays["true_delta"] > 0).sum()),
            "positive_true_delta_and_truly_safe": int(
                ((arrays["true_delta"] > 0) & arrays["truly_safe"]).sum()
            ),
        },
        "safety_tolerance_sweep": sweep,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
