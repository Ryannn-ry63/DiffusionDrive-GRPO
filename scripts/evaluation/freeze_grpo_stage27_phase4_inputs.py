#!/usr/bin/env python3
"""Freeze and validate every Stage27 Phase4 fold5 input before evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PUBLIC_SHA = (
    "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
)
SELECTOR_SHA = (
    "023d7b6b77bb8fa2dc3e779849f3716688abf0796b019416494c80b29615b691"
)
CALIBRATION_SHA = (
    "fec2a10573e913e4452f8eea92e6334fcd5e6f1104e6244876770973a1f41990"
)
SELECTION_SHA = (
    "1083e5b1250a04e6c480ea1a4370315579b26daed46c22a071cb7b972f91a0bb"
)
SELECTOR_GATE_SHA = (
    "1b442569dddafb82ea0ad0d07b34c6ded1f93151851c556a346c0dd8b98ca636"
)
FORMAL_AUDIT_SHA = (
    "68b903d0c12fc5d99a040c176d4773afcdc009a95cec212c962217714dd7cc5a"
)
FORMAL_FREEZE_SHA = (
    "95bab4782996a25083b21309e459da3826e8c6bfcc14b0d240770c4f61f4b3bd"
)
EPOCH_SHA = {
    1: "665e0fda0526e8ffe33d1931f97e48816cf72acd85ba93d4c04dbae90c84d3b3",
    2: "aa4c287814ec25fdb73fafe797251a45d2ac6fc8aad82bfb8ce39f285f4baae1",
}
MANIFEST_SHA = (
    "2424fe88319f81650eec1b8e2dbe265143d5d6d7cc8d113e104caf8d150cf2d4"
)
PLAN_SHA = (
    "06506dbbcc082001c8a1379a55ac4a5af170350b81a3cab072f57831388cd4f1"
)
PHASE4_FREEZE_SHA = (
    "2b1dd1d7d6bccbf0887034f9d41d8490b5daec81a632ca9d5258542ce057dd2a"
)
NOISES = (20261011, 20261012)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_locked(path: Path, sha: str) -> dict:
    if not path.is_file() or file_sha256(path) != sha:
        raise RuntimeError(f"Stage27 Phase4 locked input drifted: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ordered_sha(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--public-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    if file_sha256(args.public_checkpoint) != PUBLIC_SHA:
        raise RuntimeError("Stage27 Phase4 public checkpoint SHA mismatch")
    plan = root / "GRPO_STAGE27_STAGE26_ALIGNED_EXECUTION_PLAN_20260724.md"
    phase4_freeze = (
        root / "GRPO_STAGE27_PHASE4_FOLD5_EXECUTION_FREEZE_20260724.md"
    )
    if file_sha256(plan) != PLAN_SHA:
        raise RuntimeError("Stage27 frozen plan SHA drifted")
    if file_sha256(phase4_freeze) != PHASE4_FREEZE_SHA:
        raise RuntimeError("Stage27 Phase4 execution freeze SHA drifted")

    manifest_path = root / "artifacts/grpo_stage21/manifests/fold5_manifest.json"
    manifest = load_locked(manifest_path, MANIFEST_SHA)
    records = manifest.get("records", [])
    summary = manifest.get("summary", {})
    tokens = [str(record.get("token", "")) for record in records]
    logs = [str(record.get("log_name", "")) for record in records]
    if (
        summary.get("name") != "fold5"
        or len(tokens) != summary.get("count") != 1023
        or len(tokens) != 1023
        or len(set(tokens)) != 1023
        or len(set(logs)) != summary.get("num_logs")
        or len(set(logs)) != 151
        or summary.get("ordered_token_sha256") != ordered_sha(tokens)
    ):
        raise RuntimeError("Stage27 protected fold5 manifest semantics drifted")

    selection_path = root / "artifacts/grpo_stage27/selectors/selection.json"
    selector_gate_path = (
        root / "artifacts/grpo_stage27/selectors/selector_gate.json"
    )
    calibration_path = (
        root / "artifacts/grpo_stage27/selectors/multi/calibration.json"
    )
    selection = load_locked(selection_path, SELECTION_SHA)
    selector_gate = load_locked(selector_gate_path, SELECTOR_GATE_SHA)
    calibration = load_locked(calibration_path, CALIBRATION_SHA)
    if (
        not selection.get("passed")
        or selection.get("selected_branch") != "multi"
        or selection.get("selected_selector_checkpoint_sha256") != SELECTOR_SHA
        or not selector_gate.get("passed")
        or selector_gate.get("stop_before_generator_training")
        or calibration.get("selector_checkpoint_sha256") != SELECTOR_SHA
    ):
        raise RuntimeError("Stage27 selected selector provenance drifted")
    selector_path = Path(selection["selected_selector_checkpoint"])
    if file_sha256(selector_path) != SELECTOR_SHA:
        raise RuntimeError("Stage27 selected selector checkpoint SHA drifted")

    audit_path = (
        root / "artifacts/grpo_stage27/generator/phase3/formal/audit.json"
    )
    formal_freeze_path = (
        root / "artifacts/grpo_stage27/generator/phase3/formal/checkpoints.json"
    )
    audit = load_locked(audit_path, FORMAL_AUDIT_SHA)
    formal_freeze = load_locked(formal_freeze_path, FORMAL_FREEZE_SHA)
    if (
        not audit.get("passed")
        or audit.get("phase") != "formal"
        or audit.get("num_logged_optimizer_steps") != 160
        or not audit.get("frozen_reference_bitwise_equal_public")
        or not audit.get("frozen_selector_bitwise_equal_selected_multi")
        or not audit.get("zero_frozen_gradients")
        or formal_freeze.get("expected_global_steps") != [80, 160]
    ):
        raise RuntimeError("Stage27 Phase3 formal audit drifted")

    checkpoints = {}
    for epoch, (audit_record, freeze_record) in enumerate(
        zip(audit["checkpoints"], formal_freeze["checkpoints"]), start=1
    ):
        expected_step = epoch * 80
        expected_sha = EPOCH_SHA[epoch]
        path = Path(freeze_record["path"])
        if (
            audit_record["global_step"] != expected_step
            or freeze_record["global_step"] != expected_step
            or audit_record["sha256"] != expected_sha
            or freeze_record["sha256"] != expected_sha
            or file_sha256(path) != expected_sha
            or audit_record["changed_allowed_count"] != 64
            or audit_record["changed_forbidden_count"] != 0
        ):
            raise RuntimeError(f"Stage27 epoch-{epoch} checkpoint drifted")
        checkpoints[f"C{epoch}"] = {
            "epoch": epoch,
            "global_step": expected_step,
            "path": str(path.resolve()),
            "sha256": expected_sha,
        }

    result = {
        "schema_version": 1,
        "stage": 27,
        "phase": 4,
        "passed": True,
        "plan": str(plan),
        "plan_sha256": PLAN_SHA,
        "phase4_execution_freeze": str(phase4_freeze),
        "phase4_execution_freeze_sha256": PHASE4_FREEZE_SHA,
        "public_checkpoint": str(args.public_checkpoint.resolve()),
        "public_checkpoint_sha256": PUBLIC_SHA,
        "selector_checkpoint": str(selector_path.resolve()),
        "selector_checkpoint_sha256": SELECTOR_SHA,
        "calibration": str(calibration_path.resolve()),
        "calibration_sha256": CALIBRATION_SHA,
        "selection": str(selection_path.resolve()),
        "selection_sha256": SELECTION_SHA,
        "selector_gate": str(selector_gate_path.resolve()),
        "selector_gate_sha256": SELECTOR_GATE_SHA,
        "phase3_formal_audit": str(audit_path.resolve()),
        "phase3_formal_audit_sha256": FORMAL_AUDIT_SHA,
        "phase3_checkpoint_freeze": str(formal_freeze_path.resolve()),
        "phase3_checkpoint_freeze_sha256": FORMAL_FREEZE_SHA,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": MANIFEST_SHA,
        "num_tokens": 1023,
        "num_logs": 151,
        "noise_namespaces": list(NOISES),
        "systems": {
            "B": {
                "path": str(args.public_checkpoint.resolve()),
                "sha256": PUBLIC_SHA,
                "domain": "public88_base",
            },
            **checkpoints,
        },
    }
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite Stage27 Phase4 inputs: {args.output}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
