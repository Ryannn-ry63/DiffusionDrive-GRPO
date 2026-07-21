#!/usr/bin/env python3
"""Fail-closed structural audit for Stage-16 full-chain checkpoints."""

import argparse
import json
from pathlib import Path

import torch


HEAD = "agent._transfuser_model._trajectory_head."
CURRENT = HEAD + "diff_decoder."
REFERENCE = HEAD + "ref_policy."


def load(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"checkpoint has no state_dict: {path}")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-global-step", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = load(args.base_checkpoint)["state_dict"]
    payload = load(args.checkpoint)
    candidate = payload["state_dict"]
    failures = []
    if int(payload.get("global_step", -1)) != args.expected_global_step:
        failures.append("global_step mismatch")
    if len(payload.get("optimizer_states", [])) != 1:
        failures.append("checkpoint must contain one optimizer state")
    if len(payload.get("lr_schedulers", [])) != 1:
        failures.append("checkpoint must contain one scheduler state")

    changed = []
    classification_changed = []
    nonfinite = []
    for key, value in candidate.items():
        if not key.startswith(CURRENT):
            continue
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            nonfinite.append(key)
        expected = base.get(key)
        if expected is not None and not torch.equal(value, expected):
            changed.append(key)
            if "plan_cls_branch" in key:
                classification_changed.append(key)
    if not changed:
        failures.append("no current decoder tensor changed")
    if classification_changed:
        failures.append(f"{len(classification_changed)} classification tensors changed")
    if nonfinite:
        failures.append(f"{len(nonfinite)} current decoder tensors are non-finite")

    reference_mismatches = []
    for key, value in candidate.items():
        if not key.startswith(REFERENCE):
            continue
        base_key = CURRENT + key[len(REFERENCE):]
        expected = base.get(base_key)
        if expected is None or not torch.equal(value, expected):
            reference_mismatches.append(key)
    if reference_mismatches:
        failures.append(f"{len(reference_mismatches)} reference tensors differ from base")

    frozen_base_mismatches = []
    for key, expected in base.items():
        if key.startswith(CURRENT):
            continue
        actual = candidate.get(key)
        if actual is None or not torch.equal(actual, expected):
            frozen_base_mismatches.append(key)
    if frozen_base_mismatches:
        failures.append(
            f"{len(frozen_base_mismatches)} non-decoder base tensors changed"
        )

    result = {
        "passed": not failures,
        "checkpoint": str(args.checkpoint.resolve()),
        "global_step": int(payload.get("global_step", -1)),
        "expected_global_step": args.expected_global_step,
        "optimizer_states": len(payload.get("optimizer_states", [])),
        "lr_schedulers": len(payload.get("lr_schedulers", [])),
        "changed_decoder_tensors": len(changed),
        "classification_mismatch_count": len(classification_changed),
        "reference_mismatch_count": len(reference_mismatches),
        "frozen_base_mismatch_count": len(frozen_base_mismatches),
        "nonfinite_decoder_count": len(nonfinite),
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
