#!/usr/bin/env python3
"""Fail-closed structural audit for Stage-15 value-selector checkpoints."""

import argparse
import json
from pathlib import Path

import torch


HEAD = "agent._transfuser_model._trajectory_head."
CURRENT = HEAD + "diff_decoder."
REFERENCE = HEAD + "ref_policy."
VALUE = HEAD + "value_selector."
UPDATE_KEY = HEAD + "_value_selector_training_updates"


def load(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"checkpoint has no state_dict: {path}")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--generator-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-global-step", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = load(args.base_checkpoint)["state_dict"]
    generator = load(args.generator_checkpoint)["state_dict"]
    candidate_payload = load(args.checkpoint)
    candidate = candidate_payload["state_dict"]
    failures = []
    if int(candidate_payload.get("global_step", -1)) != args.expected_global_step:
        failures.append("global_step mismatch")
    if len(candidate_payload.get("optimizer_states", [])) != 1:
        failures.append("checkpoint must contain one optimizer state")
    if len(candidate_payload.get("lr_schedulers", [])) != 1:
        failures.append("checkpoint must contain one scheduler state")

    generator_mismatches = []
    for key, expected in generator.items():
        if not key.startswith(CURRENT):
            continue
        actual = candidate.get(key)
        if actual is None or not torch.equal(actual, expected):
            generator_mismatches.append(key)
    if generator_mismatches:
        failures.append(
            f"{len(generator_mismatches)} frozen generator tensors changed"
        )

    base_mismatches = []
    for key, actual in candidate.items():
        if not key.startswith(REFERENCE):
            continue
        base_key = CURRENT + key[len(REFERENCE):]
        expected = base.get(base_key)
        if expected is None or not torch.equal(actual, expected):
            base_mismatches.append(key)
    if base_mismatches:
        failures.append(
            f"{len(base_mismatches)} reference tensors differ from base"
        )

    value_keys = [key for key in candidate if key.startswith(VALUE)]
    nonfinite_value = [
        key for key in value_keys
        if torch.is_floating_point(candidate[key])
        and not torch.isfinite(candidate[key]).all()
    ]
    if not value_keys:
        failures.append("value selector is absent")
    if nonfinite_value:
        failures.append(f"{len(nonfinite_value)} value tensors are non-finite")
    update_count = None
    if UPDATE_KEY not in candidate or candidate[UPDATE_KEY].numel() != 1:
        failures.append("value-selector update buffer is absent")
    else:
        update_count = int(candidate[UPDATE_KEY].item())
        if update_count != args.expected_global_step:
            failures.append(
                f"value-selector update count is {update_count}, "
                f"expected {args.expected_global_step}"
            )
    result = {
        "passed": not failures,
        "checkpoint": str(args.checkpoint.resolve()),
        "global_step": int(candidate_payload.get("global_step", -1)),
        "expected_global_step": args.expected_global_step,
        "optimizer_states": len(candidate_payload.get("optimizer_states", [])),
        "lr_schedulers": len(candidate_payload.get("lr_schedulers", [])),
        "value_selector_tensor_count": len(value_keys),
        "value_selector_update_count": update_count,
        "generator_mismatch_count": len(generator_mismatches),
        "reference_mismatch_count": len(base_mismatches),
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
