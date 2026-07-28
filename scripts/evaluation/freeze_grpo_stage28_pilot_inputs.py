#!/usr/bin/env python3
"""Freeze all Stage28 A/B/C pilot training and fold4 evaluation inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PUBLIC_SHA = "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
PLAN_SHA = "217ef4b31cdc830e0e8fad7aa6d0f53d05c07fffcc381b00ad57f13912bc287b"
MANIFEST_SHA = "878d74b57adc8b195f3d6acd86c7a801a88e648ccddd4992663f30a6cfc3d7a5"
FOLD4_SHA = "8e0f55b18e1faf3e3b12788390abf041ab3049b7e72ff55b0968ebcfd71f4a23"
MULTI_SELECTOR_SHA = "023d7b6b77bb8fa2dc3e779849f3716688abf0796b019416494c80b29615b691"
MULTI_CALIBRATION_SHA = "fec2a10573e913e4452f8eea92e6334fcd5e6f1104e6244876770973a1f41990"
EXPLORE_SELECTOR_SHA = "9a7a202830f0dcd87c5aeac2f91a3b2dc3958584cc965ecf4b7048f56ceca91b"
EXPLORE_CALIBRATION_SHA = "ab1931fc96c00288c32b8d2fa484c2114dd5cd66ef1ccf05693d8fba8885660a"
AUTHORIZATION_SHA = "c347b84b27cb7ec1320f11858e27baf2bcad84605b3577a09843fbb4598e3df6"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locked(path: Path, expected: str, name: str) -> dict:
    if not path.is_file() or file_sha256(path) != expected:
        raise RuntimeError(f"Stage28 locked {name} SHA drifted: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--public", type=Path,
        default=Path("/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/grpo_stage28/pilot/inputs.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    paths = {
        "plan": root / "GRPO_STAGE28_PUBLIC_PAIRED_UPLIFT_EXECUTION_PLAN_20260724.md",
        "manifest": root / "artifacts/grpo_stage28/manifests/pilot_folds0_3.json",
        "fold4_manifest": root / "artifacts/grpo_stage21/manifests/fold4_manifest.json",
        "multi_calibration": root / "artifacts/grpo_stage27/selectors/multi/calibration.json",
        "explore_calibration": root / "artifacts/grpo_stage27/selectors/public/calibration.json",
        "exploration_authorization": root / "artifacts/grpo_stage28/authorization/s_public_training_only.json",
    }
    if file_sha256(args.public) != PUBLIC_SHA:
        raise RuntimeError("Stage28 public checkpoint SHA drifted")
    if file_sha256(paths["plan"]) != PLAN_SHA:
        raise RuntimeError("Stage28 frozen plan SHA drifted")
    manifest = locked(paths["manifest"], MANIFEST_SHA, "pilot manifest")
    fold4 = locked(paths["fold4_manifest"], FOLD4_SHA, "fold4 manifest")
    multi = locked(paths["multi_calibration"], MULTI_CALIBRATION_SHA, "multi calibration")
    explore = locked(paths["explore_calibration"], EXPLORE_CALIBRATION_SHA, "explore calibration")
    authorization = locked(paths["exploration_authorization"], AUTHORIZATION_SHA, "exploration authorization")
    multi_selector = Path(multi["selector_checkpoint"])
    explore_selector = Path(explore["selector_checkpoint"])
    if file_sha256(multi_selector) != MULTI_SELECTOR_SHA:
        raise RuntimeError("Stage28 multi selector SHA drifted")
    if file_sha256(explore_selector) != EXPLORE_SELECTOR_SHA:
        raise RuntimeError("Stage28 explore selector SHA drifted")
    if (
        manifest["summary"].get("count") != 4075
        or manifest["summary"].get("num_logs") != 606
        or manifest["summary"].get("source_folds") != [0, 1, 2, 3]
        or fold4["summary"].get("count") != 1021
        or fold4["summary"].get("num_logs") != 151
        or not multi.get("passed")
        or explore.get("passed") is not False
        or authorization.get("deployment_forbidden") is not True
    ):
        raise RuntimeError("Stage28 frozen input semantics drifted")

    result = {
        "schema_version": 1,
        "stage": 28,
        "phase": "pilot",
        "passed": True,
        "public_checkpoint": str(args.public.resolve()),
        "public_checkpoint_sha256": PUBLIC_SHA,
        "plan": str(paths["plan"]),
        "plan_sha256": PLAN_SHA,
        "manifest": str(paths["manifest"]),
        "manifest_sha256": MANIFEST_SHA,
        "num_train_tokens": 4075,
        "num_train_logs": 606,
        "fold4_manifest": str(paths["fold4_manifest"]),
        "fold4_manifest_sha256": FOLD4_SHA,
        "num_fold4_tokens": 1021,
        "num_fold4_logs": 151,
        "deployment_selector": str(multi_selector.resolve()),
        "deployment_selector_sha256": MULTI_SELECTOR_SHA,
        "deployment_calibration": str(paths["multi_calibration"]),
        "deployment_calibration_sha256": MULTI_CALIBRATION_SHA,
        "exploration_selector": str(explore_selector.resolve()),
        "exploration_selector_sha256": EXPLORE_SELECTOR_SHA,
        "exploration_calibration": str(paths["explore_calibration"]),
        "exploration_calibration_sha256": EXPLORE_CALIBRATION_SHA,
        "exploration_authorization": str(paths["exploration_authorization"]),
        "exploration_authorization_sha256": AUTHORIZATION_SHA,
        "epochs": [1, 2, 3, 4],
        "noise_namespaces": [20261111, 20261112],
        "branches": {
            "A": {
                "objective": "stage27_group_relative_control",
                "training_mode": "stage27_public_diffgrpo_selected_set",
                "training_selector_role": "multi",
            },
            "B": {
                "objective": "stage28_absolute_paired_uplift",
                "training_mode": "stage28_public_paired_uplift_multi",
                "training_selector_role": "multi",
            },
            "C": {
                "objective": "stage28_absolute_paired_uplift",
                "training_mode": "stage28_public_paired_uplift_explore",
                "training_selector_role": "explore",
                "deployment_forbidden": True,
            },
        },
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Stage28 input freeze: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
