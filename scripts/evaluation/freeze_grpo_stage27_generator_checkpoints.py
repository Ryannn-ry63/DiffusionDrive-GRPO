#!/usr/bin/env python3
"""Freeze the exact Stage27 Phase-3 audit or provisional checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_metadata(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "epoch": int(payload.get("epoch", -1)),
        "global_step": int(payload.get("global_step", -1)),
    }


def choose_checkpoint(candidates: list[dict], step: int, phase: str) -> dict:
    matches = [item for item in candidates if item["global_step"] == step]
    if not matches:
        raise RuntimeError(f"Stage27 checkpoint at global_step={step} is missing")
    if phase == "audit":
        preferred = [
            item for item in matches
            if Path(item["path"]).name.startswith("grpo-step-")
        ]
    else:
        preferred = [
            item for item in matches
            if re.fullmatch(r"grpo-\d{2}-\d+\.ckpt", Path(item["path"]).name)
        ]
    selected = sorted(preferred or matches, key=lambda item: item["path"])[0]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("audit", "formal"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    version_dirs = sorted((run_dir / "lightning_logs").glob("version_*"))
    if len(version_dirs) != 1:
        raise RuntimeError(
            f"expected exactly one Stage27 Lightning version, got {version_dirs}"
        )
    version_dir = version_dirs[0]
    checkpoint_paths = sorted((version_dir / "checkpoints").glob("*.ckpt"))
    if not checkpoint_paths:
        raise RuntimeError("Stage27 produced no checkpoints")
    candidates = [checkpoint_metadata(path) for path in checkpoint_paths]

    expected_steps = [1] if args.phase == "audit" else [80, 160]
    selected = [
        choose_checkpoint(candidates, step, args.phase)
        for step in expected_steps
    ]
    expected_epochs = [0] if args.phase == "audit" else [0, 1]
    for item, expected_epoch in zip(selected, expected_epochs):
        if item["epoch"] != expected_epoch:
            raise RuntimeError(
                "Stage27 checkpoint epoch mismatch: "
                f"{item['epoch']} != {expected_epoch}"
            )

    events = sorted(version_dir.glob("events.out.tfevents.*"))
    if len(events) != 1:
        raise RuntimeError(
            f"expected exactly one Stage27 TensorBoard event, got {events}"
        )
    override_candidates = (
        run_dir / "code" / "hydra" / "overrides.yaml",
        run_dir / ".hydra" / "overrides.yaml",
    )
    overrides = next((path for path in override_candidates if path.is_file()), None)
    if overrides is None:
        raise RuntimeError(
            f"Stage27 Hydra overrides are missing: {override_candidates}"
        )


    result = {
        "schema_version": 1,
        "stage": 27,
        "phase": args.phase,
        "passed": True,
        "run_dir": str(run_dir),
        "lightning_version_dir": str(version_dir.resolve()),
        "hydra_overrides": str(overrides.resolve()),
        "hydra_overrides_sha256": file_sha256(overrides),
        "tensorboard_event": str(events[0].resolve()),
        "tensorboard_event_sha256": file_sha256(events[0]),
        "expected_global_steps": expected_steps,
        "checkpoints": selected,
        "all_checkpoint_metadata": candidates,
    }
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite Stage27 checkpoint freeze: {args.output}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
