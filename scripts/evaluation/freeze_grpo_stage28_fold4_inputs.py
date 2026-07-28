#!/usr/bin/env python3
"""Freeze passing Stage28 A/B/C checkpoints into the fold4 evaluation grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PILOT_INPUTS_SHA = "489dd645c718fc43de1790e181e9473b36a76aa03c673ac41e22f5d6f4419d9c"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-inputs", type=Path,
        default=Path("artifacts/grpo_stage28/pilot/inputs.json"),
    )
    parser.add_argument(
        "--branch", action="append", nargs=3, required=True,
        metavar=("NAME", "FREEZE", "AUDIT"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/grpo_stage28/pilot/fold4/inputs.json"),
    )
    args = parser.parse_args()
    if file_sha256(args.pilot_inputs) != PILOT_INPUTS_SHA:
        raise RuntimeError("Stage28 pilot input-freeze SHA drifted")
    pilot = json.loads(args.pilot_inputs.read_text(encoding="utf-8"))
    supplied = {name: (Path(freeze), Path(audit)) for name, freeze, audit in args.branch}
    if set(supplied) != {"A", "B", "C"}:
        raise RuntimeError("Stage28 fold4 requires exactly A, B, and C")
    systems = {
        "P": {
            "path": pilot["public_checkpoint"],
            "sha256": pilot["public_checkpoint_sha256"],
            "domain": "stage28_public_base",
            "branch": "P",
            "epoch": 0,
        }
    }
    provenance = {}
    for branch in ("A", "B", "C"):
        freeze_path, audit_path = supplied[branch]
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            not freeze.get("passed") or freeze.get("stage") != 28
            or freeze.get("phase") != "formal" or freeze.get("branch") != branch
            or not audit.get("passed") or audit.get("stage") != 28
            or audit.get("phase") != "formal" or audit.get("branch") != branch
            or audit.get("checkpoint_freeze_sha256") != file_sha256(freeze_path)
            or not audit.get("zero_frozen_gradients")
            or not audit.get("frozen_reference_bitwise_equal_public")
            or not audit.get("frozen_training_selector_bitwise_equal")
        ):
            raise RuntimeError(f"Stage28 branch {branch} formal audit did not pass")
        checkpoints = freeze.get("checkpoints", [])
        if [item.get("global_step") for item in checkpoints] != [64, 128, 192, 256]:
            raise RuntimeError(f"Stage28 branch {branch} checkpoint schedule drifted")
        for epoch, item in enumerate(checkpoints, start=1):
            checkpoint = Path(item["path"])
            if not checkpoint.is_file() or file_sha256(checkpoint) != item["sha256"]:
                raise RuntimeError(f"Stage28 branch {branch} checkpoint SHA drifted")
            system = f"{branch}{epoch}"
            systems[system] = {
                "path": str(checkpoint.resolve()),
                "sha256": item["sha256"],
                "domain": f"stage28_{branch.lower()}_epoch{epoch}",
                "branch": branch,
                "epoch": epoch,
                "training_selector_role": pilot["branches"][branch]["training_selector_role"],
                "evaluation_selector_role": "multi",
            }
        provenance[branch] = {
            "freeze": str(freeze_path.resolve()),
            "freeze_sha256": file_sha256(freeze_path),
            "audit": str(audit_path.resolve()),
            "audit_sha256": file_sha256(audit_path),
        }
    result = {
        "schema_version": 1, "stage": 28, "phase": "fold4", "passed": True,
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
        "all_evaluations_use_safe_multi": True,
        "exploration_selector_deployment_forbidden": True,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Stage28 fold4 inputs: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
