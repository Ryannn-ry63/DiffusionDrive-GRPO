#!/usr/bin/env python3
"""Resolve and freeze one exact Stage27 selector checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--branch", choices=("public", "multi"), required=True)
    parser.add_argument("--phase", choices=("audit", "formal"), required=True)
    parser.add_argument("--selector-stage", choices=("stage24", "stage25"), required=True)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--expected-global-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = []
    for path in sorted(args.experiment_root.glob(
        "*/lightning_logs/version_*/checkpoints/grpo-*.ckpt"
    )):
        payload = torch.load(path, map_location="cpu")
        if (
            int(payload.get("epoch", -1)) == args.expected_epoch
            and int(payload.get("global_step", -1)) == args.expected_global_step
        ):
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one Stage27 selector checkpoint at "
            f"epoch={args.expected_epoch}, step={args.expected_global_step}; "
            f"found {len(candidates)} under {args.experiment_root}"
        )
    path = candidates[0]
    result = {
        "schema_version": 1,
        "stage": 27,
        "phase": args.phase,
        "branch": args.branch,
        "selector_stage": args.selector_stage,
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": file_sha256(path),
        "epoch": args.expected_epoch,
        "global_step": args.expected_global_step,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen pointer: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(str(path.resolve()))


if __name__ == "__main__":
    main()
