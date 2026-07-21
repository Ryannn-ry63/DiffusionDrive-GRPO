#!/usr/bin/env python3
"""Fail-closed structural audit for fresh Stage-13 generation checkpoints."""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch


HEAD = "agent._transfuser_model._trajectory_head."
CURRENT = HEAD + "diff_decoder."
REFERENCE = HEAD + "ref_policy."
OLD = HEAD + "old_policy."
SYNC_KEY = HEAD + "_old_policy_last_sync_step"


def load_state(path: Path) -> Dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"checkpoint has no state_dict: {path}")
    return payload


def inspect_checkpoint(base: Dict, candidate: Dict, expected_step: int) -> Dict:
    failures = []
    if int(candidate.get("global_step", -1)) != expected_step:
        failures.append(f"global_step is not {expected_step}")
    if len(candidate.get("optimizer_states", [])) != 1:
        failures.append("checkpoint must contain exactly one optimizer state")
    if len(candidate.get("lr_schedulers", [])) != 1:
        failures.append("checkpoint must contain exactly one scheduler state")

    base_state = base["state_dict"]
    state = candidate["state_dict"]
    changed_by_layer = {0: 0, 1: 0}
    forbidden_changes = []
    nonfinite = []
    for key, base_value in base_state.items():
        value = state.get(key)
        if value is None or value.shape != base_value.shape:
            forbidden_changes.append(key)
            continue
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            nonfinite.append(key)
        changed = not torch.equal(value, base_value)
        allowed = key.startswith(CURRENT) and ".plan_cls_branch." not in key
        if changed and not allowed:
            forbidden_changes.append(key)
        if changed and allowed:
            for layer in changed_by_layer:
                if f".layers.{layer}." in key:
                    changed_by_layer[layer] += 1

    for layer, count in changed_by_layer.items():
        if count == 0:
            failures.append(f"current decoder layer {layer} has no parameter change")
    if forbidden_changes:
        failures.append(f"{len(forbidden_changes)} frozen/base keys changed or are missing")
    if nonfinite:
        failures.append(f"{len(nonfinite)} base-mapped tensors contain NaN/Inf")

    reference_mismatches = []
    old_invalid = []
    for key, value in state.items():
        if key.startswith(REFERENCE):
            base_key = CURRENT + key[len(REFERENCE):]
            if base_key not in base_state or not torch.equal(value, base_state[base_key]):
                reference_mismatches.append(key)
        elif key.startswith(OLD):
            base_key = CURRENT + key[len(OLD):]
            if base_key not in base_state or value.shape != base_state[base_key].shape:
                old_invalid.append(key)
            elif torch.is_floating_point(value) and not torch.isfinite(value).all():
                old_invalid.append(key)
    if reference_mismatches:
        failures.append(f"{len(reference_mismatches)} reference-policy tensors differ from base")
    if old_invalid:
        failures.append(f"{len(old_invalid)} old-policy tensors lack valid finite provenance")
    reference_count = sum(key.startswith(REFERENCE) for key in state)
    old_count = sum(key.startswith(OLD) for key in state)
    if reference_count == 0:
        failures.append("reference policy is absent")
    if old_count == 0:
        failures.append("old policy is absent")

    sync_value = None
    if SYNC_KEY not in state or state[SYNC_KEY].numel() != 1:
        failures.append("old-policy last-sync buffer is absent or invalid")
    else:
        sync_value = int(state[SYNC_KEY].item())
        expected_sync = ((expected_step - 1) // 32) * 32
        if sync_value != expected_sync:
            failures.append(
                f"old-policy last sync is {sync_value}, expected {expected_sync}"
            )

    return {
        "passed": not failures,
        "expected_global_step": expected_step,
        "global_step": int(candidate.get("global_step", -1)),
        "optimizer_states": len(candidate.get("optimizer_states", [])),
        "lr_schedulers": len(candidate.get("lr_schedulers", [])),
        "changed_current_decoder_keys_by_layer": changed_by_layer,
        "reference_tensor_count": reference_count,
        "old_policy_tensor_count": old_count,
        "old_policy_last_sync_step": sync_value,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-global-step", type=int, choices=(8, 128), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect_checkpoint(
        load_state(args.base_checkpoint), load_state(args.checkpoint), args.expected_global_step
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
