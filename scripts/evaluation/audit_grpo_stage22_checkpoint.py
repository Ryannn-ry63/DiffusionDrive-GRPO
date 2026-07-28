#!/usr/bin/env python3
"""Fail-closed Stage22 LoRA-only checkpoint and training-metric audit."""

import argparse
import json
import math
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


MODEL = "agent._transfuser_model."
DECODER = MODEL + "_trajectory_head.diff_decoder."
REFERENCE = MODEL + "_trajectory_head.ref_policy."
FINAL = ".task_decoder.plan_reg_branch.4."


def scalar_values(events, tag):
    values = [event.value for event in events.Scalars(tag)]
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"missing/non-finite Stage22 scalar: {tag}")
    return values


def current_base_name(name):
    if name.startswith(DECODER) and FINAL in name and name.endswith(
        ("weight", "bias")
    ):
        prefix, field = name.rsplit(".", 1)
        return prefix + ".base." + field
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = torch.load(args.base, map_location="cpu")["state_dict"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    current = checkpoint["state_dict"]
    if int(checkpoint.get("global_step", -1)) != args.expected_step:
        raise RuntimeError("Stage22 global step mismatch")

    frozen_mismatch = []
    for name, value in base.items():
        if not name.startswith(MODEL):
            continue
        mapped = current_base_name(name)
        if mapped not in current or not torch.equal(value.cpu(), current[mapped].cpu()):
            frozen_mismatch.append((name, mapped))
    if frozen_mismatch:
        raise RuntimeError(f"Stage22 changed/missed frozen base: {frozen_mismatch[0]}")

    adapter_keys = sorted(
        key for key in current
        if key.startswith(DECODER) and key.endswith(("lora_A", "lora_B"))
    )
    expected_adapter_keys = sorted(
        DECODER + f"layers.{layer}.task_decoder.plan_reg_branch.4.lora_{part}"
        for layer in range(2) for part in ("A", "B")
    )
    if adapter_keys != expected_adapter_keys:
        raise RuntimeError(f"Stage22 adapter tensor set drifted: {adapter_keys}")
    if not all(torch.isfinite(current[key]).all() for key in adapter_keys):
        raise FloatingPointError("Stage22 adapter contains non-finite values")
    if any(torch.count_nonzero(current[key]) == 0 for key in adapter_keys if key.endswith("lora_B")):
        raise RuntimeError("Stage22 LoRA B remained identically zero")

    base_decoder = {
        name[len(DECODER):]: value for name, value in base.items()
        if name.startswith(DECODER)
    }
    reference_mismatch = [
        suffix for suffix, value in base_decoder.items()
        if REFERENCE + suffix not in current
        or not torch.equal(value.cpu(), current[REFERENCE + suffix].cpu())
    ]
    if reference_mismatch:
        raise RuntimeError(
            f"Stage22 frozen reference mismatch: {reference_mismatch[0]}"
        )

    optimizers = checkpoint.get("optimizer_states", [])
    if len(optimizers) != 1 or len(optimizers[0]["param_groups"]) != 1:
        raise RuntimeError("Stage22 requires one AdamW optimizer group")
    group = optimizers[0]["param_groups"][0]
    if (
        len(group["params"]) != 4
        or not math.isclose(float(group["lr"]), 3e-5, rel_tol=0, abs_tol=1e-12)
        or not math.isclose(float(group["weight_decay"]), 0.0, rel_tol=0, abs_tol=1e-12)
    ):
        raise RuntimeError(f"Stage22 optimizer group drifted: {group}")

    events = EventAccumulator(str(args.events_dir))
    events.Reload()
    tags = set(events.Tags()["scalars"])
    required = {
        "train/loss_step",
        "train/generation_grpo_loss_step",
        "train/stage22_exact_kl_loss_step",
        "train/stage22_paired_delta_mean_step",
        "train/stage22_mature_delta_mean_step",
        "train/stage22_safety_override_fraction_step",
        "train/stage22_bootstrap_advantage_fraction_step",
        "train/stage22_catastrophic_fraction_step",
        "train/stage22_mean_exact_kl_step",
        "train/stage22_mean_scene_weight_step",
    }
    if not required <= tags:
        raise RuntimeError(f"Stage22 missing scalar tags: {sorted(required - tags)}")
    metrics = {tag: scalar_values(events, tag) for tag in required}
    if max(metrics["train/stage22_bootstrap_advantage_fraction_step"]) <= 0:
        raise RuntimeError("Stage22 bootstrap advantage never became active")
    for layer in range(2):
        tag = f"train/decoder_layer_{layer}_regression_grad_norm_step"
        if max(scalar_values(events, tag)) <= 0:
            raise RuntimeError(f"Stage22 inactive adapter gradient: {tag}")
        for group_name in (
            "classification", "shared_attention", "ffn", "time_modulation"
        ):
            frozen_tag = (
                f"train/decoder_layer_{layer}_{group_name}_grad_norm_step"
            )
            if any(abs(value) > 0 for value in scalar_values(events, frozen_tag)):
                raise RuntimeError(f"Stage22 forbidden gradient: {frozen_tag}")
    for tag in (
        "train/perception_grad_norm_step",
        "train/classification_grad_norm_step",
        "train/value_selector_grad_norm_step",
        "train/paired_risk_grad_norm_step",
    ):
        if any(abs(value) > 0 for value in scalar_values(events, tag)):
            raise RuntimeError(f"Stage22 forbidden gradient: {tag}")

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "global_step": args.expected_step,
        "adapter_keys": adapter_keys,
        "adapter_tensor_count": len(adapter_keys),
        "frozen_base_mismatch_count": len(frozen_mismatch),
        "reference_mismatch_count": len(reference_mismatch),
        "optimizer_lr": float(group["lr"]),
        "optimizer_weight_decay": float(group["weight_decay"]),
        "gradient_clip_norm": 1.0,
        "num_logged_steps": len(metrics["train/loss_step"]),
        "finite_metrics": True,
        "zero_forbidden_gradients": True,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
