#!/usr/bin/env python3
"""Select the frozen Stage27 Phase-2 selector branch without NavTest feedback."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PUBLIC_SHA256 = (
    "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
)
PLAN_SHA256 = (
    "06506dbbcc082001c8a1379a55ac4a5af170350b81a3cab072f57831388cd4f1"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-calibration", type=Path, required=True)
    parser.add_argument("--multi-calibration", type=Path, required=True)
    parser.add_argument("--public-freeze", type=Path, required=True)
    parser.add_argument("--multi-freeze", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if file_sha256(args.plan) != PLAN_SHA256:
        raise RuntimeError("Stage27 aligned execution plan SHA mismatch")
    rows = {}
    for branch, calibration_path, freeze_path in (
        ("public", args.public_calibration, args.public_freeze),
        ("multi", args.multi_calibration, args.multi_freeze),
    ):
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if (
            calibration.get("calibration_profile") != "stage27_phase2"
            or calibration.get("num_artifacts") != 2
            or calibration.get("num_scenes") != 2042
            or calibration.get("guard_component_indices") != [0, 1, 3, 4, 5]
            or freeze.get("stage") != 27
            or freeze.get("phase") != "formal"
            or freeze.get("branch") != branch
            or calibration.get("selector_checkpoint_sha256")
            != freeze.get("stage25_checkpoint_sha256")
        ):
            raise RuntimeError(f"invalid Stage27 calibration/freeze: {branch}")
        groups = calibration.get("selected", {}).get("group_means", {})
        if sorted(groups) != [
            "public88_base:20260821",
            "public88_base:20260822",
        ]:
            raise RuntimeError(f"Stage27 calibration groups drifted: {branch}")
        rows[branch] = {
            "branch": branch,
            "passed": bool(calibration.get("passed")),
            "selector_checkpoint": freeze["stage25_checkpoint"],
            "selector_checkpoint_sha256": freeze[
                "stage25_checkpoint_sha256"
            ],
            "calibration": str(calibration_path.resolve()),
            "calibration_sha256": file_sha256(calibration_path),
            "mean_gain": float(calibration["selected"]["mean_gain"]),
            "whole_log_bootstrap_ci": calibration["selected"][
                "whole_log_bootstrap_ci"
            ],
            "switch_count": int(calibration["selected"]["switch_count"]),
            "switch_rate": float(calibration["selected"]["switch_rate"]),
            "catastrophic_count": int(
                calibration["selected"]["catastrophic_count"]
            ),
            "checks": calibration["selected"]["checks"],
            "freeze_sha256": file_sha256(freeze_path),
        }

    passing = [row for row in rows.values() if row["passed"]]
    selected = None
    reason = "neither Stage27 selector branch passed"
    if len(passing) == 1:
        selected = passing[0]
        reason = "only passing branch"
    elif len(passing) == 2:
        public_low = float(rows["public"]["whole_log_bootstrap_ci"][0])
        multi_low = float(rows["multi"]["whole_log_bootstrap_ci"][0])
        if abs(public_low - multi_low) < 0.0005:
            selected = rows["multi"]
            reason = "CI-lower tie within 0.0005; preregistered multi tie-break"
        else:
            selected = max(
                passing,
                key=lambda row: float(row["whole_log_bootstrap_ci"][0]),
            )
            reason = "larger whole-log bootstrap 95% CI lower bound"

    result = {
        "schema_version": 1,
        "stage": 27,
        "phase": 2,
        "passed": selected is not None,
        "stop_before_generator_training": selected is None,
        "public_checkpoint_sha256": PUBLIC_SHA256,
        "plan": str(args.plan.resolve()),
        "plan_sha256": PLAN_SHA256,
        "tie_tolerance": 0.0005,
        "selection_reason": reason,
        "selected_branch": selected["branch"] if selected else None,
        "selected_selector_checkpoint": (
            selected["selector_checkpoint"] if selected else None
        ),
        "selected_selector_checkpoint_sha256": (
            selected["selector_checkpoint_sha256"] if selected else None
        ),
        "selected_calibration": (
            selected["calibration"] if selected else None
        ),
        "selected_calibration_sha256": (
            selected["calibration_sha256"] if selected else None
        ),
        "branches": rows,
    }
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite Stage27 branch selection: {args.output}"
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
