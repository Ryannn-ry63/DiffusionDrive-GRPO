#!/usr/bin/env python3
"""Fail-closed provenance audit for Stage27 Phase-3 generator training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PUBLIC_SHA = (
    "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
)
PLAN_SHA = (
    "06506dbbcc082001c8a1379a55ac4a5af170350b81a3cab072f57831388cd4f1"
)
MANIFEST_SHA = (
    "1b36434147a1e269285eda1472271a741f78b2e4db0745e25940e720e7b67275"
)
SELECTION_SHA = (
    "1083e5b1250a04e6c480ea1a4370315579b26daed46c22a071cb7b972f91a0bb"
)
CALIBRATION_SHA = (
    "fec2a10573e913e4452f8eea92e6334fcd5e6f1104e6244876770973a1f41990"
)
SELECTOR_GATE_SHA = (
    "1b442569dddafb82ea0ad0d07b34c6ded1f93151851c556a346c0dd8b98ca636"
)
SELECTOR_SHA = (
    "023d7b6b77bb8fa2dc3e779849f3716688abf0796b019416494c80b29615b691"
)
FREEZE_SHA = (
    "5cdccfddd175312f7e8d8449f34a7617a1460f71b98a3590cc2ead4a3611e673"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_locked(path: Path, expected_sha: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if actual != expected_sha:
        raise RuntimeError(
            f"Stage27 locked input SHA mismatch: {path}: "
            f"{actual} != {expected_sha}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-checkpoint", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--selector-gate", type=Path, required=True)
    parser.add_argument("--selector-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if file_sha256(args.public_checkpoint) != PUBLIC_SHA:
        raise RuntimeError("Stage27 public checkpoint SHA mismatch")
    if file_sha256(args.plan) != PLAN_SHA:
        raise RuntimeError("Stage27 frozen plan SHA mismatch")
    manifest = load_locked(args.manifest, MANIFEST_SHA)
    selection = load_locked(args.selection, SELECTION_SHA)
    calibration = load_locked(args.calibration, CALIBRATION_SHA)
    gate = load_locked(args.selector_gate, SELECTOR_GATE_SHA)
    freeze = load_locked(args.selector_freeze, FREEZE_SHA)

    records = manifest.get("records", [])
    summary = manifest.get("summary", {})
    tokens = [str(record.get("token", "")) for record in records]
    logs = {}
    for record in records:
        log_name = str(record.get("log_name", ""))
        fold = int(record.get("source_fold", -1))
        if not log_name or fold not in range(5):
            raise RuntimeError("Stage27 generator manifest record drifted")
        if log_name in logs and logs[log_name] != fold:
            raise RuntimeError("Stage27 generator manifest splits one log")
        logs[log_name] = fold
    if (
        summary.get("stage") != 27
        or summary.get("phase") != 3
        or summary.get("source_folds") != [0, 1, 2, 3, 4]
        or int(summary.get("count", -1)) != 5096
        or int(summary.get("num_logs", -1)) != 757
        or not summary.get("whole_log_isolation")
        or len(tokens) != 5096
        or len(set(tokens)) != 5096
        or any(not token for token in tokens)
        or len(logs) != 757
    ):
        raise RuntimeError("Stage27 generator manifest semantics drifted")

    if (
        not selection.get("passed")
        or selection.get("selected_branch") != "multi"
        or selection.get("selected_selector_checkpoint_sha256")
        != SELECTOR_SHA
        or selection.get("selected_calibration_sha256") != CALIBRATION_SHA
    ):
        raise RuntimeError("Stage27 selector selection drifted")
    if (
        not calibration.get("passed")
        or calibration.get("calibration_profile") != "stage27_phase2"
        or calibration.get("selector_checkpoint_sha256") != SELECTOR_SHA
        or calibration.get("num_scenes") != 2042
        or calibration.get("num_artifacts") != 2
    ):
        raise RuntimeError("Stage27 selector calibration drifted")
    if (
        not gate.get("passed")
        or gate.get("stop_before_generator_training")
        or gate.get("selected_branch") != "multi"
        or gate.get("selector_checkpoint_sha256") != SELECTOR_SHA
        or gate.get("calibration_sha256") != CALIBRATION_SHA
        or int(gate.get("count", -1)) != 2042
        or int(gate.get("catastrophic_count", -1)) != 0
        or not all(gate.get("checks", {}).values())
    ):
        raise RuntimeError("Stage27 selector runtime gate drifted")
    if (
        not freeze.get("passed")
        or freeze.get("stage") != 27
        or freeze.get("phase") != "formal"
        or freeze.get("branch") != "multi"
        or freeze.get("stage25_checkpoint_sha256") != SELECTOR_SHA
    ):
        raise RuntimeError("Stage27 selector freeze drifted")

    selector = Path(selection["selected_selector_checkpoint"])
    if (
        selector.resolve()
        != Path(freeze["stage25_checkpoint"]).resolve()
        or not selector.is_file()
        or file_sha256(selector) != SELECTOR_SHA
    ):
        raise RuntimeError("Stage27 selected selector checkpoint drifted")
    if (
        Path(selection["selected_calibration"]).resolve()
        != args.calibration.resolve()
        or Path(gate["calibration"]).resolve() != args.calibration.resolve()
    ):
        raise RuntimeError("Stage27 calibration path provenance drifted")

    result = {
        "schema_version": 1,
        "stage": 27,
        "phase": 3,
        "passed": True,
        "public_checkpoint": str(args.public_checkpoint.resolve()),
        "public_checkpoint_sha256": PUBLIC_SHA,
        "plan": str(args.plan.resolve()),
        "plan_sha256": PLAN_SHA,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": MANIFEST_SHA,
        "num_tokens": 5096,
        "num_logs": 757,
        "source_folds": [0, 1, 2, 3, 4],
        "selection": str(args.selection.resolve()),
        "selection_sha256": SELECTION_SHA,
        "selected_branch": "multi",
        "selector_checkpoint": str(selector.resolve()),
        "selector_checkpoint_sha256": SELECTOR_SHA,
        "selector_freeze": str(args.selector_freeze.resolve()),
        "selector_freeze_sha256": FREEZE_SHA,
        "calibration": str(args.calibration.resolve()),
        "calibration_sha256": CALIBRATION_SHA,
        "selector_gate": str(args.selector_gate.resolve()),
        "selector_gate_sha256": SELECTOR_GATE_SHA,
        "selector_gate_gain": float(gate["pooled_mean_gain"]),
        "selector_gate_ci": gate["whole_log_bootstrap_ci"],
    }
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite Stage27 input audit: {args.output}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
