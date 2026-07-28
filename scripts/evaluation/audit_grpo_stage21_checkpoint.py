#!/usr/bin/env python3
"""Fail-closed Stage21 U8/U32 training, gradient, and freeze audit."""

import argparse
import json
import math
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


MODEL_PREFIX = "agent._transfuser_model."
DECODER_PREFIX = MODEL_PREFIX + "_trajectory_head.diff_decoder."
LAYERS_PREFIX = DECODER_PREFIX + "layers."
CLASSIFICATION_MARKER = ".task_decoder.plan_cls_branch."


def scalar_values(events, tag):
    values = [event.value for event in events.Scalars(tag)]
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"missing or non-finite Stage21 scalar: {tag}")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--replay-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = torch.load(args.base, map_location="cpu")["state_dict"]
    payload = torch.load(args.checkpoint, map_location="cpu")
    current = payload["state_dict"]
    if int(payload.get("global_step", -1)) != args.expected_step:
        raise RuntimeError("Stage21 checkpoint global step mismatch")

    changed_allowed, changed_forbidden = [], []
    layer_changed = {0: 0, 1: 0}
    for name, base_tensor in base.items():
        if not name.startswith(MODEL_PREFIX) or name not in current:
            continue
        if torch.equal(base_tensor.cpu(), current[name].cpu()):
            continue
        allowed = name.startswith(LAYERS_PREFIX) and CLASSIFICATION_MARKER not in name
        if allowed:
            changed_allowed.append(name)
            for layer in (0, 1):
                if f".layers.{layer}." in name:
                    layer_changed[layer] += 1
        else:
            changed_forbidden.append(name)
    if changed_forbidden:
        raise RuntimeError(f"Stage21 changed frozen parameter: {changed_forbidden[0]}")
    if not changed_allowed or any(count == 0 for count in layer_changed.values()):
        raise RuntimeError("Stage21 did not update both permitted decoder layers")

    reference_mismatch = []
    base_decoder = {
        name[len(DECODER_PREFIX):]: value
        for name, value in base.items() if name.startswith(DECODER_PREFIX)
    }
    reference_prefix = MODEL_PREFIX + "_trajectory_head.ref_policy."
    for suffix, base_tensor in base_decoder.items():
        reference_name = reference_prefix + suffix
        if reference_name not in current or not torch.equal(
            base_tensor.cpu(), current[reference_name].cpu()
        ):
            reference_mismatch.append(reference_name)
    if reference_mismatch:
        raise RuntimeError(f"Stage21 frozen reference mismatch: {reference_mismatch[0]}")

    optimizer_groups = payload["optimizer_states"][0]["param_groups"]
    lr_scales = sorted(float(group.get("lr_scale", -1.0)) for group in optimizer_groups)
    if lr_scales != [0.1, 1.0]:
        raise RuntimeError(f"Stage21 optimizer LR scales mismatch: {lr_scales}")
    effective_lrs = sorted(float(group["lr"]) for group in optimizer_groups)
    if not math.isclose(effective_lrs[0] / effective_lrs[1], 0.1, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("Stage21 layer LR ratio is not 0.1")

    replay_equal = None
    replay_numerically_equivalent = None
    replay_model_max_abs_diff = None
    replay_optimizer_max_abs_diff = None
    if args.replay_checkpoint is not None:
        replay = torch.load(args.replay_checkpoint, map_location="cpu")
        if int(replay.get("global_step", -1)) != args.expected_step:
            raise RuntimeError("Stage21 replay global step mismatch")
        replay_state = replay["state_dict"]
        if set(current) != set(replay_state):
            raise RuntimeError("Stage21 resume replay model keys differ")
        replay_equal = True
        replay_model_max_abs_diff = 0.0
        for name, value in current.items():
            other = replay_state[name]
            if torch.equal(value.cpu(), other.cpu()):
                continue
            replay_equal = False
            difference = float((value.float() - other.float()).abs().max())
            replay_model_max_abs_diff = max(replay_model_max_abs_diff, difference)
            allowed = name.startswith(LAYERS_PREFIX) and CLASSIFICATION_MARKER not in name
            if not allowed or difference > 1e-8:
                raise RuntimeError(
                    f"Stage21 resume replay model drift: {name}, max_abs={difference}"
                )
        first_optimizer = payload["optimizer_states"][0]
        replay_optimizer = replay["optimizer_states"][0]
        if first_optimizer["param_groups"] != replay_optimizer["param_groups"]:
            raise RuntimeError("Stage21 resume replay optimizer groups differ")
        if set(first_optimizer["state"]) != set(replay_optimizer["state"]):
            raise RuntimeError("Stage21 resume replay optimizer state keys differ")
        replay_optimizer_max_abs_diff = 0.0
        for parameter_id, state in first_optimizer["state"].items():
            other = replay_optimizer["state"][parameter_id]
            if set(state) != set(other):
                raise RuntimeError("Stage21 resume replay optimizer fields differ")
            for name, value in state.items():
                if torch.is_tensor(value):
                    difference = float(
                        (value.float() - other[name].float()).abs().max()
                    )
                    replay_optimizer_max_abs_diff = max(
                        replay_optimizer_max_abs_diff, difference
                    )
                    equal = difference <= 2e-8
                else:
                    equal = value == other[name]
                if not equal:
                    raise RuntimeError(
                        f"Stage21 resume replay optimizer tensor differs: {parameter_id}/{name}"
                    )
        replay_numerically_equivalent = True

    events = EventAccumulator(str(args.events_dir))
    events.Reload()
    tags = set(events.Tags()["scalars"])
    required_finite = {
        "train/loss_step",
        "train/generation_grpo_loss_step",
        "train/diffgrpo_bc_loss_step",
        "train/raw_reward_mean_step",
        "train/stage21_safety_override_fraction_step",
        "train/stage21_base_delta_mean_step",
        "train/stage21_mean_scene_weight_step",
        "train/stage21_mean_bc_weight_step",
    }
    if not required_finite <= tags:
        raise RuntimeError(f"Stage21 missing scalar tags: {sorted(required_finite - tags)}")
    metrics = {tag: scalar_values(events, tag) for tag in required_finite}
    active_tags = [
        "train/decoder_layer_0_regression_grad_norm_step",
        "train/decoder_layer_1_regression_grad_norm_step",
        "train/decoder_layer_0_shared_attention_grad_norm_step",
        "train/decoder_layer_1_shared_attention_grad_norm_step",
    ]
    for tag in active_tags:
        if max(scalar_values(events, tag)) <= 0:
            raise RuntimeError(f"Stage21 permitted gradient inactive: {tag}")
    frozen_tags = [
        "train/decoder_layer_0_classification_grad_norm_step",
        "train/decoder_layer_1_classification_grad_norm_step",
        "train/perception_grad_norm_step",
        "train/value_selector_grad_norm_step",
        "train/paired_risk_grad_norm_step",
        "train/classification_grad_norm_step",
    ]
    for tag in frozen_tags:
        if any(abs(value) > 0 for value in scalar_values(events, tag)):
            raise RuntimeError(f"Stage21 forbidden gradient active: {tag}")
    for tag in (
        "train/generation_positive_advantage_fraction_step",
        "train/generation_negative_advantage_fraction_step",
    ):
        if max(scalar_values(events, tag)) <= 0:
            raise RuntimeError(f"Stage21 advantage sign never active: {tag}")

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "global_step": args.expected_step,
        "changed_allowed_count": len(changed_allowed),
        "changed_forbidden_count": len(changed_forbidden),
        "changed_by_layer": layer_changed,
        "reference_mismatch_count": len(reference_mismatch),
        "lr_scales": lr_scales,
        "effective_lrs": effective_lrs,
        "num_logged_steps": len(metrics["train/loss_step"]),
        "finite_metrics": True,
        "active_policy_gradients": True,
        "active_bc": True,
        "zero_forbidden_gradients": True,
        "resume_replay_bitwise_equal": replay_equal,
        "resume_replay_numerically_equivalent": replay_numerically_equivalent,
        "resume_replay_model_max_abs_diff": replay_model_max_abs_diff,
        "resume_replay_optimizer_max_abs_diff": replay_optimizer_max_abs_diff,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
