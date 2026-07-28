#!/usr/bin/env python3
"""Freeze Stage29 H/HC checkpoints and the historical A3 control for fold4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PILOT_INPUTS_SHA = "775e8ac63c8b398e91809e8e4d7222a724f29552fc397dd3f69a215a8fc4d0f7"
PUBLIC_SHA = "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
AMENDMENT_NAME = "GRPO_STAGE29_PREFOLD4_SIGNAL_GATE_AMENDMENT_20260725.md"
AMENDMENT_SHA = "941eef5f370f90ad9e5b9c7ed7a1dbaa12a971f30e79a211c446098f576f3bd9"
SIGNAL_GATE_DEFINITION = "stage29_prefold4_amended_dense_scene_v1"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-inputs", type=Path,
        default=Path("artifacts/grpo_stage29/pilot/inputs.json"),
    )
    parser.add_argument("--control-freeze", type=Path, required=True)
    parser.add_argument("--control-audit", type=Path, required=True)
    parser.add_argument(
        "--branch", action="append", nargs=3, required=True,
        metavar=("NAME", "FREEZE", "AUDIT"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/grpo_stage29/pilot/fold4/inputs.json"),
    )
    args = parser.parse_args()
    if file_sha256(args.pilot_inputs) != PILOT_INPUTS_SHA:
        raise RuntimeError("Stage29 pilot input-freeze SHA drifted")
    amendment_path = Path(__file__).resolve().parents[2] / AMENDMENT_NAME
    if not amendment_path.is_file() or file_sha256(amendment_path) != AMENDMENT_SHA:
        raise RuntimeError("Stage29 pre-fold4 signal-gate amendment SHA drifted")
    pilot = load_json(args.pilot_inputs)
    if not (
        pilot.get("passed") and pilot.get("stage") == 29
        and pilot.get("public_checkpoint_sha256") == PUBLIC_SHA
        and pilot.get("noise_namespaces") == [20261211, 20261212]
    ):
        raise RuntimeError("Stage29 pilot input semantics drifted")

    supplied = {name: (Path(freeze), Path(audit)) for name, freeze, audit in args.branch}
    if set(supplied) != {"H", "HC"} or len(supplied) != len(args.branch):
        raise RuntimeError("Stage29 fold4 requires exactly H and HC")
    systems = {
        "P": {
            "path": pilot["public_checkpoint"],
            "sha256": pilot["public_checkpoint_sha256"],
            "domain": "stage29_public_base",
            "branch": "P",
            "epoch": 0,
            "selectable": False,
        }
    }
    provenance = {}

    control_freeze = load_json(args.control_freeze)
    control_audit = load_json(args.control_audit)
    if not (
        control_freeze.get("passed") and control_freeze.get("stage") == 28
        and control_freeze.get("phase") == "formal"
        and control_freeze.get("branch") == "A"
        and control_audit.get("passed") and control_audit.get("stage") == 28
        and control_audit.get("phase") == "formal"
        and control_audit.get("branch") == "A"
        and control_audit.get("checkpoint_freeze_sha256") == file_sha256(args.control_freeze)
        and control_audit.get("zero_frozen_gradients")
        and control_audit.get("frozen_reference_bitwise_equal_public")
        and control_audit.get("frozen_training_selector_bitwise_equal")
    ):
        raise RuntimeError("Stage29 historical A3 control audit did not pass")
    control_checkpoints = control_freeze.get("checkpoints", [])
    if [item.get("global_step") for item in control_checkpoints] != [64, 128, 192, 256]:
        raise RuntimeError("Stage29 historical A checkpoint schedule drifted")
    a3 = control_checkpoints[2]
    a3_path = Path(a3["path"])
    if not a3_path.is_file() or file_sha256(a3_path) != a3["sha256"]:
        raise RuntimeError("Stage29 historical A3 checkpoint SHA drifted")
    systems["A3"] = {
        "path": str(a3_path.resolve()),
        "sha256": a3["sha256"],
        "domain": "stage28_a_epoch3_control",
        "branch": "A",
        "epoch": 3,
        "selectable": False,
        "evaluation_selector_role": "multi",
    }
    provenance["A3_control"] = {
        "freeze": str(args.control_freeze.resolve()),
        "freeze_sha256": file_sha256(args.control_freeze),
        "audit": str(args.control_audit.resolve()),
        "audit_sha256": file_sha256(args.control_audit),
    }

    for branch in ("H", "HC"):
        freeze_path, audit_path = supplied[branch]
        freeze = load_json(freeze_path)
        audit = load_json(audit_path)
        if not (
            freeze.get("passed") and freeze.get("stage") == 29
            and freeze.get("phase") == "formal" and freeze.get("branch") == branch
            and audit.get("passed") and audit.get("stage") == 29
            and audit.get("phase") == "formal" and audit.get("branch") == branch
            and audit.get("checkpoint_freeze_sha256") == file_sha256(freeze_path)
            and audit.get("zero_frozen_gradients")
            and audit.get("frozen_reference_bitwise_equal_public")
            and audit.get("frozen_training_selector_bitwise_equal")
            and audit.get("training_signal_health", {}).get("passed")
            and audit.get("training_signal_gate_definition") == SIGNAL_GATE_DEFINITION
            and audit.get("training_signal_gate_amendment_sha256") == AMENDMENT_SHA
        ):
            raise RuntimeError(f"Stage29 branch {branch} formal audit did not pass")
        checkpoints = freeze.get("checkpoints", [])
        if [item.get("global_step") for item in checkpoints] != [64, 128, 192, 256]:
            raise RuntimeError(f"Stage29 branch {branch} checkpoint schedule drifted")
        for epoch, item in enumerate(checkpoints, start=1):
            checkpoint = Path(item["path"])
            if not checkpoint.is_file() or file_sha256(checkpoint) != item["sha256"]:
                raise RuntimeError(f"Stage29 branch {branch} checkpoint SHA drifted")
            system = f"{branch}{epoch}"
            systems[system] = {
                "path": str(checkpoint.resolve()),
                "sha256": item["sha256"],
                "domain": f"stage29_{branch.lower()}_epoch{epoch}",
                "branch": branch,
                "epoch": epoch,
                "selectable": True,
                "training_selector_role": pilot["branches"][branch]["training_selector_role"],
                "evaluation_selector_role": "multi",
            }
        provenance[branch] = {
            "freeze": str(freeze_path.resolve()),
            "freeze_sha256": file_sha256(freeze_path),
            "audit": str(audit_path.resolve()),
            "audit_sha256": file_sha256(audit_path),
        }

    expected_systems = ["P", "A3", "H1", "H2", "H3", "H4", "HC1", "HC2", "HC3", "HC4"]
    if list(systems) != expected_systems:
        raise RuntimeError("Stage29 fold4 system order drifted")
    result = {
        "schema_version": 1, "stage": 29, "phase": "fold4", "passed": True,
        "pilot_inputs": str(args.pilot_inputs.resolve()),
        "pilot_inputs_sha256": PILOT_INPUTS_SHA,
        "manifest": pilot["fold4_manifest"],
        "manifest_sha256": pilot["fold4_manifest_sha256"],
        "num_tokens": 1021, "num_logs": 151,
        "selector_checkpoint": pilot["deployment_selector"],
        "selector_checkpoint_sha256": pilot["deployment_selector_sha256"],
        "calibration": pilot["deployment_calibration"],
        "calibration_sha256": pilot["deployment_calibration_sha256"],
        "noise_namespaces": pilot["noise_namespaces"],
        "systems": systems,
        "branch_provenance": provenance,
        "training_signal_gate_definition": SIGNAL_GATE_DEFINITION,
        "training_signal_gate_amendment": str(amendment_path.resolve()),
        "training_signal_gate_amendment_sha256": AMENDMENT_SHA,
        "all_evaluations_use_safe_multi": True,
        "historical_control_not_selectable": True,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Stage29 fold4 inputs: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
