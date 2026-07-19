#!/usr/bin/env python3
"""Fail closed when a Stage-12 resume checkpoint lacks trainer/KL state."""

import argparse
import json
import math
from pathlib import Path
from typing import Dict

import torch


CONTROLLER_SUFFIXES = (
    "coefficient",
    "rolling_values",
    "rolling_index",
    "rolling_count",
    "observation_count",
    "consecutive_hard_violations",
    "rolling_mean",
    "should_stop",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-global-step", type=int, required=True)
    return parser.parse_args()


def inspect_resume_payload(payload: Dict, expected_global_step: int) -> Dict:
    failures = []
    global_step = int(payload.get("global_step", -1))
    if global_step != expected_global_step:
        failures.append(
            f"global_step is {global_step}, expected {expected_global_step}"
        )
    optimizer_count = len(payload.get("optimizer_states", []))
    scheduler_count = len(payload.get("lr_schedulers", []))
    if optimizer_count != 1:
        failures.append("checkpoint must contain exactly one optimizer state")
    if scheduler_count != 1:
        failures.append("checkpoint must contain exactly one LR scheduler state")

    state_dict = payload.get("state_dict", {})
    controller = {}
    for suffix in CONTROLLER_SUFFIXES:
        matching = [
            value
            for key, value in state_dict.items()
            if key.endswith(f"_adaptive_kl_controller.{suffix}")
        ]
        if len(matching) != 1:
            failures.append(f"adaptive KL state {suffix!r} is missing or ambiguous")
        else:
            controller[suffix] = matching[0]

    coefficient = controller.get("coefficient")
    rolling_mean = controller.get("rolling_mean")
    should_stop = controller.get("should_stop")
    if coefficient is not None and (
        coefficient.numel() != 1 or not math.isfinite(float(coefficient.item()))
    ):
        failures.append("adaptive KL coefficient is not finite and scalar")
    if rolling_mean is not None and (
        rolling_mean.numel() != 1 or not math.isfinite(float(rolling_mean.item()))
    ):
        failures.append("adaptive KL rolling mean is not finite and scalar")
    if should_stop is not None and bool(should_stop.item()):
        failures.append("adaptive KL checkpoint is already hard-stopped")

    return {
        "passed": not failures,
        "global_step": global_step,
        "optimizer_state_count": optimizer_count,
        "lr_scheduler_state_count": scheduler_count,
        "adaptive_kl_coefficient": (
            float(coefficient.item()) if coefficient is not None else None
        ),
        "adaptive_kl_rolling_mean": (
            float(rolling_mean.item()) if rolling_mean is not None else None
        ),
        "adaptive_kl_should_stop": (
            bool(should_stop.item()) if should_stop is not None else None
        ),
        "failures": failures,
    }


def main() -> None:
    args = parse_args()
    if args.expected_global_step <= 0:
        raise ValueError("expected-global-step must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    result = inspect_resume_payload(payload, args.expected_global_step)
    result["checkpoint"] = str(args.checkpoint.resolve())
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
