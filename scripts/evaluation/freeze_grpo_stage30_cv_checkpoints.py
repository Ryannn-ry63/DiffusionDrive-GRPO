#!/usr/bin/env python3
"""Freeze exact Stage30 audit or 48/96/144/192-step checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "epoch": int(payload.get("epoch", -1)),
        "global_step": int(payload.get("global_step", -1)),
    }


def choose(candidates: list[dict], step: int, phase: str) -> dict:
    matches = [record for record in candidates if record["global_step"] == step]
    if not matches:
        raise RuntimeError(f"Stage30 checkpoint global_step={step} is missing")
    pattern = r"grpo-step-\d+\.ckpt" if phase == "audit" else r"grpo-\d{2}-\d+\.ckpt"
    preferred = [
        record for record in matches
        if re.fullmatch(pattern, Path(record["path"]).name)
    ]
    return sorted(preferred or matches, key=lambda record: record["path"])[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("audit", "formal"), required=True)
    parser.add_argument("--branch", choices=("COV", "MCC"), required=True)
    parser.add_argument("--holdout", type=int, choices=range(4), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    versions = sorted((run_dir / "lightning_logs").glob("version_*"))
    if len(versions) != 1:
        raise RuntimeError(f"expected one Stage30 Lightning version: {versions}")
    version = versions[0]
    candidates = [
        metadata(path)
        for path in sorted((version / "checkpoints").glob("*.ckpt"))
    ]
    expected_steps = [1] if args.phase == "audit" else [48, 96, 144, 192]
    expected_epochs = [0] if args.phase == "audit" else [0, 1, 2, 3]
    selected = [choose(candidates, step, args.phase) for step in expected_steps]
    if [record["epoch"] for record in selected] != expected_epochs:
        raise RuntimeError("Stage30 checkpoint epoch/step schedule drifted")
    events = sorted(version.glob("events.out.tfevents.*"))
    if len(events) != 1:
        raise RuntimeError("Stage30 expected exactly one TensorBoard event")
    override_candidates = (
        run_dir / "code" / "hydra" / "overrides.yaml",
        run_dir / ".hydra" / "overrides.yaml",
    )
    overrides = next((path for path in override_candidates if path.is_file()), None)
    if overrides is None:
        raise RuntimeError("Stage30 Hydra overrides are missing")
    result = {
        "schema_version": 1,
        "stage": 30,
        "phase": args.phase,
        "branch": args.branch,
        "holdout_fold": args.holdout,
        "passed": True,
        "run_dir": str(run_dir),
        "lightning_version_dir": str(version.resolve()),
        "hydra_overrides": str(overrides.resolve()),
        "hydra_overrides_sha256": sha256(overrides),
        "tensorboard_event": str(events[0].resolve()),
        "tensorboard_event_sha256": sha256(events[0]),
        "expected_global_steps": expected_steps,
        "checkpoints": selected,
        "all_checkpoint_metadata": candidates,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

