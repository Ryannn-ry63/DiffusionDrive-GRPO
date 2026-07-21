#!/usr/bin/env python3
"""Fail-closed TensorBoard audit for Stage-17 U8/U32 runs."""

import argparse
import json
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


FINITE = (
    "train/loss_step",
    "train/paired_risk_classification_loss_step",
    "train/paired_risk_delta_loss_step",
    "train/paired_risk_delta_mae_step",
    "train/paired_risk_grad_norm_step",
)
ZERO = (
    "train/diff_decoder_grad_norm_step",
    "train/perception_grad_norm_step",
    "train/value_selector_grad_norm_step",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--first-step", type=int, required=True)
    parser.add_argument("--last-step", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    accumulator = EventAccumulator(str(args.log_dir)).Reload()
    available = set(accumulator.Tags().get("scalars", ()))
    expected_steps = list(range(args.first_step, args.last_step + 1))
    failures, metrics = [], {}
    for tag in FINITE + ZERO:
        if tag not in available:
            failures.append(f"missing scalar tag: {tag}")
            continue
        events = accumulator.Scalars(tag)
        values = [float(event.value) for event in events]
        if [event.step for event in events] != expected_steps:
            failures.append(f"{tag} step interval mismatch")
        if not values or not all(math.isfinite(value) for value in values):
            failures.append(f"{tag} has non-finite values")
            continue
        if tag in ZERO and any(value != 0.0 for value in values):
            failures.append(f"{tag} violates frozen boundary")
        if tag == "train/paired_risk_grad_norm_step" and any(value <= 0 for value in values):
            failures.append("paired-risk gradient is not positive at every step")
        metrics[tag] = {"minimum": min(values), "maximum": max(values), "count": len(values)}
    result = {"passed": not failures, "metrics": metrics, "failures": failures}
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
