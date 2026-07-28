#!/usr/bin/env python3
"""Runtime replay gate for the selected Stage27 Phase-2 selector."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.evaluation.calibrate_grpo_stage25_selector import (
    whole_log_bootstrap_ci,
)


PUBLIC_SHA256 = (
    "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
)
GUARD_INDICES = (0, 1, 3, 4, 5)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260927)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if not selection.get("passed") or selection.get("stage") != 27:
        raise RuntimeError("Stage27 selector branch selection did not pass")
    calibration_path = Path(selection["selected_calibration"])
    if file_sha256(calibration_path) != selection[
        "selected_calibration_sha256"
    ]:
        raise RuntimeError("Stage27 selected calibration SHA mismatch")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    selector_sha = selection["selected_selector_checkpoint_sha256"]
    calibration_sha = file_sha256(calibration_path)

    deltas = []
    logs = []
    components = []
    switched_values = []
    groups = defaultdict(list)
    provenance = []
    seen_cells = set()
    for path in args.artifact:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        selector = summary.get("stage25_selector", {})
        domain = str(summary.get("generator_domain", ""))
        namespace = int(summary.get("evaluation_noise_namespace", -999))
        cell = (domain, namespace)
        if (
            cell in seen_cells
            or domain != "public88_base"
            or namespace not in (20260821, 20260822)
            or summary.get("checkpoint_sha256") != PUBLIC_SHA256
            or summary.get("reference_checkpoint_sha256") != PUBLIC_SHA256
            or not summary.get("completed")
            or int(summary.get("num_failures", -1)) != 0
            or int(summary.get("num_tokens", -1)) != 1021
            or selector.get("checkpoint_sha256") != selector_sha
            or selector.get("calibration_sha256") != calibration_sha
            or selector.get("calibration_collection")
        ):
            raise RuntimeError(f"invalid Stage27 selector replay: {path}")
        seen_cells.add(cell)
        local_switches = 0
        local_gain = 0.0
        for record in payload.get("records", []):
            diagnostic = record.get("stage24_selector", {})
            rewards = np.asarray(
                record.get("candidate_rewards"), dtype=np.float64
            )
            candidate_components = np.asarray(
                record.get("candidate_components"), dtype=np.float64
            )
            if rewards.shape != (20,) or candidate_components.shape != (20, 6):
                raise RuntimeError("Stage27 replay lacks all-20 PDM labels")
            fallback = int(diagnostic["fallback_mode"])
            selected = int(record["selected_mode"])
            switched = bool(diagnostic["switched"])
            delta = float(rewards[selected] - rewards[fallback])
            component_delta = (
                candidate_components[selected] - candidate_components[fallback]
            )
            deltas.append(delta)
            logs.append(str(record["log_name"]))
            components.append(component_delta)
            switched_values.append(switched)
            groups[cell].append(delta)
            local_switches += int(switched)
            local_gain += delta
        if local_switches != int(selector["switch_count"]):
            raise RuntimeError("Stage27 replay switch summary drifted")
        if not np.isclose(
            local_gain / 1021,
            float(selector["selected_minus_fallback_mean"]),
            atol=1e-12,
        ):
            raise RuntimeError("Stage27 replay gain summary drifted")
        provenance.append({"path": str(path), "sha256": file_sha256(path)})

    if seen_cells != {
        ("public88_base", 20260821),
        ("public88_base", 20260822),
    }:
        raise RuntimeError("Stage27 replay cell grid incomplete")
    delta_array = np.asarray(deltas, dtype=np.float64)
    component_array = np.asarray(components, dtype=np.float64)
    switch_array = np.asarray(switched_values, dtype=bool)
    ci = whole_log_bootstrap_ci(
        delta_array, logs, args.bootstrap_samples, args.bootstrap_seed
    )
    group_means = {
        f"{domain}:{namespace}": float(np.mean(values))
        for (domain, namespace), values in sorted(groups.items())
    }
    guard_means = component_array[:, GUARD_INDICES].mean(axis=0)
    catastrophic = int(
        np.sum(switch_array & (delta_array <= -0.5))
    )
    switch_rate = float(switch_array.mean())
    checks = {
        "pooled_gain_at_least_0.003": bool(delta_array.mean() >= 0.003),
        "whole_log_ci_strictly_positive": bool(ci[0] > 0),
        "every_namespace_nonnegative": bool(
            all(value >= 0 for value in group_means.values())
        ),
        "switch_rate_at_least_0.01": bool(switch_rate >= 0.01),
        "switch_rate_at_most_0.15": bool(switch_rate <= 0.15),
        "guard_components_no_worse_0.0005": bool(
            np.all(guard_means >= -0.0005)
        ),
        "catastrophic_count_zero": bool(catastrophic == 0),
        "matches_offline_calibration_gain": bool(np.isclose(
            delta_array.mean(),
            float(calibration["selected"]["mean_gain"]),
            atol=1e-12,
        )),
        "matches_offline_calibration_switch_count": bool(
            int(switch_array.sum())
            == int(calibration["selected"]["switch_count"])
        ),
    }
    result = {
        "schema_version": 1,
        "stage": 27,
        "phase": 2,
        "passed": bool(all(checks.values())),
        "stop_before_generator_training": bool(not all(checks.values())),
        "selection": str(args.selection.resolve()),
        "selection_sha256": file_sha256(args.selection),
        "selected_branch": selection["selected_branch"],
        "selector_checkpoint_sha256": selector_sha,
        "calibration": str(calibration_path.resolve()),
        "calibration_sha256": calibration_sha,
        "count": int(delta_array.size),
        "pooled_mean_gain": float(delta_array.mean()),
        "whole_log_bootstrap_ci": ci,
        "group_means": group_means,
        "switch_count": int(switch_array.sum()),
        "switch_rate": switch_rate,
        "guard_component_indices": list(GUARD_INDICES),
        "guard_component_mean_deltas": guard_means.tolist(),
        "catastrophic_count": catastrophic,
        "checks": checks,
        "artifacts": provenance,
    }
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite Stage27 selector gate: {args.output}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
