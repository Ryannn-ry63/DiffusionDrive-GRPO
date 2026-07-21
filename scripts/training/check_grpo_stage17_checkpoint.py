#!/usr/bin/env python3
"""Fail-closed structural audit for Stage-17 adapter checkpoints."""

import argparse
import json
from pathlib import Path

import torch


HEAD = "agent._transfuser_model._trajectory_head."
ADAPTER = HEAD + "paired_risk_head."
UPDATE = HEAD + "_paired_risk_training_updates"
CURRENT = HEAD + "diff_decoder."
REFERENCE = HEAD + "ref_policy."
OLD = HEAD + "old_policy."


def state(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"checkpoint has no state_dict: {path}")
    return payload


def normalized_base(payload):
    return {
        (key if key.startswith("agent.") else "agent." + key): value
        for key, value in payload["state_dict"].items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-global-step", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    generator = normalized_base(state(args.generator_checkpoint))
    base = normalized_base(state(args.base_checkpoint))
    payload = state(args.checkpoint)
    candidate = payload["state_dict"]
    failures = []
    if int(payload.get("global_step", -1)) != args.expected_global_step:
        failures.append("global_step mismatch")
    if len(payload.get("optimizer_states", [])) != 1 or len(payload.get("lr_schedulers", [])) != 1:
        failures.append("optimizer/scheduler state mismatch")
    adapter_changed = []
    nonfinite = []
    for key, value in candidate.items():
        if key.startswith(ADAPTER):
            expected = generator.get(key)
            if expected is None or not torch.equal(value, expected):
                adapter_changed.append(key)
            if torch.is_floating_point(value) and not torch.isfinite(value).all():
                nonfinite.append(key)
    if not adapter_changed:
        failures.append("no paired-risk adapter tensor changed")
    if nonfinite:
        failures.append("paired-risk adapter has non-finite tensors")
    frozen_mismatches = []
    for key, expected in generator.items():
        if (
            key.startswith(ADAPTER)
            or key == UPDATE
            or key.startswith(REFERENCE)
            or key.startswith(OLD)
        ):
            continue
        actual = candidate.get(key)
        if actual is None or not torch.equal(actual, expected):
            frozen_mismatches.append(key)
    reference_mismatches = []
    for key, value in candidate.items():
        if key.startswith(REFERENCE):
            base_key = CURRENT + key[len(REFERENCE):]
            if base_key not in base or not torch.equal(value, base[base_key]):
                reference_mismatches.append(key)
    if frozen_mismatches:
        failures.append(f"{len(frozen_mismatches)} frozen generator tensors changed")
    if reference_mismatches:
        failures.append(f"{len(reference_mismatches)} reference tensors differ from base")
    result = {
        "passed": not failures,
        "global_step": int(payload.get("global_step", -1)),
        "expected_global_step": args.expected_global_step,
        "changed_adapter_tensors": len(adapter_changed),
        "frozen_generator_mismatches": len(frozen_mismatches),
        "reference_mismatches": len(reference_mismatches),
        "nonfinite_adapter_tensors": len(nonfinite),
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
