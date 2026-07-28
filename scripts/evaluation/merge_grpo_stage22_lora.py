#!/usr/bin/env python3
"""Merge a Stage22 LoRA training checkpoint into one ordinary generator."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from navsim.agents.diffusiondrive.paired_residual_lora import (
    merge_stage22_checkpoint_state_dict,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Stage22 merge output must differ from its input")

    checkpoint = torch.load(args.input, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise KeyError("Stage22 checkpoint has no state_dict")
    merged_state, prefixes = merge_stage22_checkpoint_state_dict(
        checkpoint["state_dict"]
    )
    checkpoint["state_dict"] = merged_state
    checkpoint["stage22_merge"] = {
        "schema_version": 1,
        "algorithm": "diffgrpo_paired_residual",
        "source_checkpoint_sha256": file_sha256(args.input),
        "merged_adapter_prefixes": list(prefixes),
        "rank": 8,
        "alpha": 8.0,
        "scale": 1.0,
        "ordinary_single_generator": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    audit = {
        **checkpoint["stage22_merge"],
        "output_checkpoint": str(args.output.resolve()),
        "output_checkpoint_sha256": file_sha256(args.output),
        "remaining_lora_keys": sorted(
            key for key in merged_state if "lora_" in key
        ),
        "passes": not any("lora_" in key for key in merged_state),
    }
    if not audit["passes"]:
        raise RuntimeError("Stage22 merge left adapter keys in output")
    audit_path = args.audit_output or args.output.with_suffix(
        args.output.suffix + ".merge.json"
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
