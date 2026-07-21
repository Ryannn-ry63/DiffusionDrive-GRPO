#!/usr/bin/env python3
"""Fail-closed TensorBoard audit for Stage-16 U8/U32 runs."""

import argparse
import json
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


FINITE_TAGS = (
    "train/loss_step",
    "train/generation_grpo_loss_step",
    "train/diffgrpo_bc_loss_step",
    "train/raw_reward_mean_step",
    "train/raw_reward_std_step",
    "train/diffgrpo_mean_current_log_prob_step",
    "train/diffgrpo_mean_bc_log_prob_step",
    "train/diff_decoder_grad_norm_step",
)
ZERO_TAGS = (
    "train/perception_grad_norm_step",
    "train/value_selector_grad_norm_step",
    "train/classification_grad_norm_step",
    "train/decoder_layer_0_classification_grad_norm_step",
    "train/decoder_layer_1_classification_grad_norm_step",
)
POSITIVE_TAGS = (
    "train/diff_decoder_grad_norm_step",
    "train/decoder_layer_0_shared_attention_grad_norm_step",
    "train/decoder_layer_0_regression_grad_norm_step",
    "train/decoder_layer_1_shared_attention_grad_norm_step",
    "train/decoder_layer_1_regression_grad_norm_step",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--first-step", type=int, required=True)
    parser.add_argument("--last-step", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.first_step < 0 or args.last_step < args.first_step:
        raise ValueError("invalid expected step interval")

    accumulator = EventAccumulator(str(args.log_dir)).Reload()
    available = set(accumulator.Tags().get("scalars", ()))
    failures = []
    summaries = {}
    required = set(FINITE_TAGS + ZERO_TAGS + POSITIVE_TAGS) | {
        "train/diffgrpo_discount_first_step",
        "train/diffgrpo_discount_last_step",
    }
    for tag in sorted(required):
        if tag not in available:
            failures.append(f"missing scalar tag: {tag}")
            continue
        events = accumulator.Scalars(tag)
        steps = [event.step for event in events]
        values = [float(event.value) for event in events]
        expected_steps = list(range(args.first_step, args.last_step + 1))
        if steps != expected_steps:
            failures.append(f"{tag} step interval mismatch")
        if not values or not all(math.isfinite(value) for value in values):
            failures.append(f"{tag} contains non-finite values")
            continue
        summaries[tag] = {
            "count": len(values),
            "minimum": min(values),
            "maximum": max(values),
        }
        if tag in ZERO_TAGS and any(value != 0.0 for value in values):
            failures.append(f"{tag} violates the frozen gradient boundary")
        if tag in POSITIVE_TAGS and any(value <= 0.0 for value in values):
            failures.append(f"{tag} is not active at every optimizer step")

    for tag, wanted in (
        ("train/diffgrpo_discount_first_step", 0.6 ** 4),
        ("train/diffgrpo_discount_last_step", 1.0),
    ):
        if tag in available and any(
            not math.isclose(event.value, wanted, rel_tol=0.0, abs_tol=1e-6)
            for event in accumulator.Scalars(tag)
        ):
            failures.append(f"{tag} does not match the locked discount")

    result = {
        "passed": not failures,
        "log_dir": str(args.log_dir.resolve()),
        "expected_step_interval": [args.first_step, args.last_step],
        "metrics": summaries,
        "failures": failures,
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
