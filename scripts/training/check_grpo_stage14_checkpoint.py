#!/usr/bin/env python3
"""Structural audit for Stage-14 U8/U64/U128 generation checkpoints."""

import argparse
import json
from pathlib import Path

try:
    from scripts.training.check_grpo_stage13_checkpoint import (
        inspect_checkpoint,
        load_state,
    )
except ModuleNotFoundError:
    from check_grpo_stage13_checkpoint import inspect_checkpoint, load_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-global-step", type=int, choices=(8, 64, 128), required=True
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = inspect_checkpoint(
        load_state(args.base_checkpoint),
        load_state(args.checkpoint),
        args.expected_global_step,
    )
    result["gate"] = "stage14_checkpoint_audit"
    result["checkpoint"] = str(args.checkpoint.resolve())
    result["base_checkpoint"] = str(args.base_checkpoint.resolve())
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
