#!/usr/bin/env python3
"""Freeze exact Stage39 challenger audit/formal checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

from navsim.agents.diffusiondrive.stage39_challenger import (
    STAGE39_OBJECTIVE_REVISION,
    STAGE39_PLAN_SHA256,
    STAGE39_BRANCHES,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_metadata(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "epoch": int(payload.get("epoch", -1)),
        "global_step": int(payload.get("global_step", -1)),
    }


def choose(records: list[dict], step: int) -> dict:
    matches = [record for record in records if record["global_step"] == step]
    if not matches:
        raise RuntimeError(f"Stage39 checkpoint global_step={step} is missing")
    preferred = [
        record for record in matches
        if re.fullmatch(r"grpo-step-\d+\.ckpt", Path(record["path"]).name)
    ]
    return sorted(preferred or matches, key=lambda record: record["path"])[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("audit", "pilot", "formal"), required=True)
    parser.add_argument("--branch", choices=STAGE39_BRANCHES, required=True)
    parser.add_argument("--holdout", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--selected-step", type=int, choices=(48, 96, 192))
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    experiment_root = args.experiment_root.resolve()
    versions = sorted(experiment_root.glob("**/lightning_logs/version_*"))
    if len(versions) != 1:
        raise RuntimeError(
            f"expected one Stage39 Lightning version under {experiment_root}: {versions}"
        )
    version = versions[0]
    run_dir = version.parent.parent
    records = [
        checkpoint_metadata(path)
        for path in sorted((version / "checkpoints").glob("*.ckpt"))
    ]
    if args.phase == "audit":
        expected_steps = [1]
    elif args.phase == "pilot":
        if args.holdout != 2 or args.selected_step is not None:
            raise ValueError("Stage39 pilot freeze requires holdout=2 and no selected step")
        expected_steps = [48, 96, 192]
    else:
        if args.holdout not in (0, 1, 3) or args.selected_step is None:
            raise ValueError("Stage39 formal freeze requires fold0/1/3 and pilot-selected step")
        expected_steps = [args.selected_step]
    expected_epochs = [(step - 1) // 48 for step in expected_steps]
    selected = [choose(records, step) for step in expected_steps]
    if [record["epoch"] for record in selected] != expected_epochs:
        raise RuntimeError(
            "Stage39 checkpoint schedule drifted: "
            f"{[(r['epoch'], r['global_step']) for r in selected]}"
        )
    events = sorted(version.glob("events.out.tfevents.*"))
    if len(events) != 1:
        raise RuntimeError("Stage39 expected exactly one TensorBoard event file")
    override_candidates = (
        run_dir / "code" / "hydra" / "overrides.yaml",
        run_dir / ".hydra" / "overrides.yaml",
    )
    overrides = next((path for path in override_candidates if path.is_file()), None)
    if overrides is None:
        raise RuntimeError("Stage39 Hydra overrides are missing")
    result = {
        "schema_version": 1,
        "stage": 39,
        "phase": args.phase,
        "branch": args.branch,
        "holdout_fold": args.holdout,
        "passed": True,
        "objective_revision": STAGE39_OBJECTIVE_REVISION,
        "plan_sha256": STAGE39_PLAN_SHA256,
        "experiment_root": str(experiment_root),
        "run_dir": str(run_dir.resolve()),
        "lightning_version_dir": str(version.resolve()),
        "hydra_overrides": str(overrides.resolve()),
        "hydra_overrides_sha256": sha256(overrides),
        "tensorboard_event": str(events[0].resolve()),
        "tensorboard_event_sha256": sha256(events[0]),
        "expected_global_steps": expected_steps,
        "selected_step": args.selected_step,
        "checkpoints": selected,
        "all_checkpoint_metadata": records,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
