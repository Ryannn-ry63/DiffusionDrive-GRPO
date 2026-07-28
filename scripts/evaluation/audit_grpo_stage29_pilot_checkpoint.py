#!/usr/bin/env python3
"""Fail-closed Stage29 H/HC checkpoint, gradient, and selector audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PUBLIC_SHA = "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
INPUTS_SHA = "775e8ac63c8b398e91809e8e4d7222a724f29552fc397dd3f69a215a8fc4d0f7"
AMENDMENT_NAME = "GRPO_STAGE29_PREFOLD4_SIGNAL_GATE_AMENDMENT_20260725.md"
AMENDMENT_SHA = "941eef5f370f90ad9e5b9c7ed7a1dbaa12a971f30e79a211c446098f576f3bd9"
SIGNAL_GATE_DEFINITION = "stage29_prefold4_amended_dense_scene_v1"
MODEL_PREFIX = "_transfuser_model."
DECODER_PREFIX = MODEL_PREFIX + "_trajectory_head.diff_decoder."
LAYERS_PREFIX = DECODER_PREFIX + "layers."
REFERENCE_PREFIX = MODEL_PREFIX + "_trajectory_head.ref_policy."
CLASSIFICATION_MARKER = ".task_decoder.plan_cls_branch."
SELECTOR_PREFIXES = (
    MODEL_PREFIX + "_trajectory_head.stage24_selector.",
    MODEL_PREFIX + "_trajectory_head.stage25_selector.",
)
SELECTOR_BUFFERS = {
    MODEL_PREFIX + "_trajectory_head." + name
    for name in (
        "_stage24_selector_training_updates",
        "_stage25_selector_training_updates",
        "_stage24_embedding_count",
        "_stage24_embedding_sum",
        "_stage24_embedding_sum_sq",
    )
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_state(payload: dict) -> dict[str, torch.Tensor]:
    return {
        (name[len("agent."):] if name.startswith("agent.") else name): value
        for name, value in payload["state_dict"].items()
    }


def parse_overrides(path: Path) -> dict[str, str]:
    result = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if not line.startswith("- ") or "=" not in line:
            raise RuntimeError(f"invalid Stage29 override line: {raw}")
        key, value = line[2:].split("=", 1)
        result[key.removeprefix("+")] = value
    return result


def require(actual: dict[str, str], expected: dict[str, str]) -> None:
    for key, wanted in expected.items():
        if actual.get(key) != wanted:
            raise RuntimeError(
                f"Stage29 Hydra override drifted: {key}: "
                f"{actual.get(key)!r} != {wanted!r}"
            )


def scalar_values(events: EventAccumulator, tag: str, count: int) -> list[float]:
    values = [float(event.value) for event in events.Scalars(tag)]
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"Stage29 invalid scalar series: {tag}")
    return values


def assess_training_signal_health(
    active_scene_fraction: float,
    positive_fraction: float,
    neutral_fraction: float,
) -> dict:
    values = {
        "policy_active_scene_fraction_mean": float(active_scene_fraction),
        "positive_fraction_mean": float(positive_fraction),
        "neutral_fraction_mean": float(neutral_fraction),
    }
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values.values()):
        raise RuntimeError(f"Stage29 invalid training-signal fractions: {values}")
    if values["positive_fraction_mean"] + values["neutral_fraction_mean"] > 1.0 + 1e-6:
        raise RuntimeError(f"Stage29 inconsistent training-signal fractions: {values}")
    checks = {
        "active_scene_fraction_at_least_0.15": (
            values["policy_active_scene_fraction_mean"] >= 0.15
        ),
        "positive_fraction_at_least_0.01": values["positive_fraction_mean"] >= 0.01,
        "neutral_fraction_at_most_0.85": values["neutral_fraction_mean"] <= 0.85,
        "finite_fractions_within_unit_interval": True,
    }
    return {**values, "checks": checks, "passed": bool(all(checks.values()))}


def audit_checkpoint(
    record: dict,
    base_state: dict[str, torch.Tensor],
    selector_state: dict[str, torch.Tensor],
) -> dict:
    path = Path(record["path"])
    if file_sha256(path) != record["sha256"]:
        raise RuntimeError("Stage29 frozen checkpoint SHA drifted")
    payload = torch.load(path, map_location="cpu")
    current = canonical_state(payload)
    if int(payload.get("epoch", -1)) != record["epoch"] or int(
        payload.get("global_step", -1)
    ) != record["global_step"]:
        raise RuntimeError("Stage29 checkpoint metadata drifted")
    missing = sorted(set(base_state) - set(current))
    if missing:
        raise RuntimeError(f"Stage29 checkpoint lost public tensor: {missing[0]}")
    allowed, forbidden = [], []
    changed_by_layer = {0: 0, 1: 0}
    for name, base_tensor in base_state.items():
        if torch.equal(base_tensor.cpu(), current[name].cpu()):
            continue
        if name.startswith(LAYERS_PREFIX) and CLASSIFICATION_MARKER not in name:
            allowed.append(name)
            for layer in changed_by_layer:
                if f".layers.{layer}." in name:
                    changed_by_layer[layer] += 1
        else:
            forbidden.append(name)
    if forbidden:
        raise RuntimeError(f"Stage29 changed frozen public tensor: {forbidden[0]}")
    if len(allowed) != 64 or changed_by_layer != {0: 32, 1: 32}:
        raise RuntimeError("Stage29 64-tensor decoder boundary drifted")

    base_decoder = {
        name[len(DECODER_PREFIX):]: tensor
        for name, tensor in base_state.items() if name.startswith(DECODER_PREFIX)
    }
    for suffix, base_tensor in base_decoder.items():
        name = REFERENCE_PREFIX + suffix
        if name not in current or not torch.equal(base_tensor.cpu(), current[name].cpu()):
            raise RuntimeError(f"Stage29 frozen reference drifted: {name}")

    selector_names = [
        name for name in selector_state
        if name.startswith(SELECTOR_PREFIXES) or name in SELECTOR_BUFFERS
    ]
    if not selector_names:
        raise RuntimeError("Stage29 expected selector state is empty")
    for name in selector_names:
        if name not in current or not torch.equal(
            selector_state[name].cpu(), current[name].cpu()
        ):
            raise RuntimeError(f"Stage29 frozen training selector drifted: {name}")

    optimizer_states = payload.get("optimizer_states", [])
    if len(optimizer_states) != 1:
        raise RuntimeError("Stage29 expected one optimizer")
    groups = sorted(
        (float(group.get("lr_scale", -1)), float(group["lr"]),
         len(group["params"]), float(group["weight_decay"]))
        for group in optimizer_states[0]["param_groups"]
    )
    if groups != [(0.1, 1e-7, 32, 0.0001), (1.0, 1e-6, 32, 0.0001)]:
        raise RuntimeError(f"Stage29 optimizer groups drifted: {groups}")
    if len(optimizer_states[0]["state"]) != 64:
        raise RuntimeError("Stage29 optimizer state count drifted")
    for state in optimizer_states[0]["state"].values():
        for value in state.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise RuntimeError("Stage29 optimizer contains non-finite state")
    return {
        "path": str(path.resolve()),
        "sha256": record["sha256"],
        "epoch": record["epoch"],
        "global_step": record["global_step"],
        "changed_allowed_count": len(allowed),
        "changed_forbidden_count": len(forbidden),
        "changed_by_layer": changed_by_layer,
        "optimizer_groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("audit", "formal"), required=True)
    parser.add_argument("--branch", choices=("H", "HC"), required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if file_sha256(args.base) != PUBLIC_SHA or file_sha256(args.inputs) != INPUTS_SHA:
        raise RuntimeError("Stage29 public/input SHA drifted")
    amendment_path = Path(__file__).resolve().parents[2] / AMENDMENT_NAME
    if not amendment_path.is_file() or file_sha256(amendment_path) != AMENDMENT_SHA:
        raise RuntimeError("Stage29 pre-fold4 signal-gate amendment SHA drifted")
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    entry = inputs["branches"][args.branch]
    role = entry["training_selector_role"]
    if role != "multi":
        raise RuntimeError("Stage29 requires the frozen S-multi training selector")
    selector_key = "deployment_selector"
    calibration_key = "deployment_calibration"
    selector_path = Path(inputs[selector_key])
    if file_sha256(selector_path) != inputs[f"{selector_key}_sha256"]:
        raise RuntimeError("Stage29 training selector SHA drifted")
    calibration = json.loads(Path(inputs[calibration_key]).read_text(encoding="utf-8"))
    if not calibration.get("passed"):
        raise RuntimeError("Stage29 deployment selector calibration did not pass")
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    expected_steps = [1] if args.phase == "audit" else [64, 128, 192, 256]
    if (
        not freeze.get("passed") or freeze.get("stage") != 29
        or freeze.get("phase") != args.phase or freeze.get("branch") != args.branch
        or freeze.get("expected_global_steps") != expected_steps
    ):
        raise RuntimeError("Stage29 checkpoint freeze semantics drifted")
    overrides_path = Path(freeze["hydra_overrides"])
    event_path = Path(freeze["tensorboard_event"])
    if file_sha256(overrides_path) != freeze["hydra_overrides_sha256"] or file_sha256(
        event_path
    ) != freeze["tensorboard_event_sha256"]:
        raise RuntimeError("Stage29 frozen run artifact SHA drifted")
    overrides = parse_overrides(overrides_path)
    expected = {
        "agent.checkpoint_path": str(args.base.resolve()),
        "agent.reference_checkpoint_path": str(args.base.resolve()),
        "agent.lr": "1e-6",
        "agent.config.grpo_training_mode": entry["training_mode"],
        "agent.config.generation_policy_algorithm": "diffgrpo_selected_set",
        "agent.config.grpo_decoder_gradient_scope": "all_layers",
        "agent.config.grpo_decoder_layer0_lr_mult": "0.1",
        "agent.config.inference_selector_source": "trajectory_relative_harm_v3",
        "agent.config.stage25_selector_checkpoint_path": str(selector_path.resolve()),
        "agent.config.stage25_selector_calibration_path": inputs[calibration_key],
        "agent.config.stage24_selector_residual_margin": str(calibration["residual_margin"]),
        "agent.config.stage25_selector_risk_threshold": str(calibration["risk_threshold"]),
        "agent.config.stage24_selector_ood_threshold": str(calibration["ood_threshold"]),
        "agent.config.stage23_generator_train_manifest_path": inputs["manifest"],
        "agent.config.diffgrpo_bc_weight": "0.1",
        "agent.config.diffgrpo_step_discount": "0.6",
        "agent.config.diffgrpo_group_size": "8",
        "agent.config.diffgrpo_base_advantage_clip": "2.0",
        "agent.config.diffgrpo_safety_regression_tolerance": "0.000001",
        "agent.config.stage29_headroom_low": "0.75",
        "agent.config.stage29_headroom_high": "0.90",
        "agent.config.stage29_delta_scale_floor": "0.002",
        "agent.config.stage29_rank_weight": "0.5",
        "agent.config.stage29_conditional_regularization": str(
            entry["conditional_regularization"]
        ).lower(),
        "agent.config.stage29_fixed_bc_weight": "0.1",
        "agent.config.stage29_fixed_kl_weight": "0.1",
        "agent.config.stage29_hard_bc_weight": "0.05",
        "agent.config.stage29_mature_bc_weight": "0.1",
        "agent.config.stage29_hard_kl_weight": "0.05",
        "agent.config.stage29_mature_kl_weight": "0.5",
        "agent.config.stage29_safety_kl_weight": "0.5",
        "agent.config.diffusion_truncation_timestep": "32",
        "agent.config.diffusion_roll_timesteps": "[32,24,16,8,0]",
        "agent.config.diffusion_scheduler_num_inference_steps": "125",
        "dataloader.params.batch_size": "1",
        "trainer.params.accumulate_grad_batches": "8",
        "trainer.params.gradient_clip_val": "1.0",
        "trainer.params.devices": "8",
        "trainer.params.strategy": "ddp",
        "trainer.params.precision": "32-true",
        "trainer.params.max_epochs": "1" if args.phase == "audit" else "4",
        "trainer.params.limit_train_batches": "8" if args.phase == "audit" else "1.0",
        "agent.config.grpo_checkpoint_every_n_train_steps": "1" if args.phase == "audit" else "0",
    }
    require(overrides, expected)
    if args.phase == "audit":
        require(overrides, {"trainer.params.max_steps": "1"})
    elif "trainer.params.max_steps" in overrides:
        raise RuntimeError("Stage29 formal run unexpectedly limits max_steps")

    base_state = canonical_state(torch.load(args.base, map_location="cpu"))
    if len(base_state) != 763:
        raise RuntimeError("Stage29 public base state count drifted")
    selector_state = canonical_state(torch.load(selector_path, map_location="cpu"))
    checkpoint_audits = [
        audit_checkpoint(record, base_state, selector_state)
        for record in freeze["checkpoints"]
    ]

    expected_count = expected_steps[-1]
    events = EventAccumulator(str(event_path.parent))
    events.Reload()
    tags = set(events.Tags()["scalars"])
    finite_tags = {
        "train/loss_step", "train/generation_grpo_loss_step",
        "train/diffgrpo_bc_loss_step", "train/generation_reference_kl_loss_step",
        "train/raw_reward_mean_step", "train/diffgrpo_mean_current_log_prob_step",
    }
    finite_tags |= {
        "train/stage29_advantage_mean_step",
        "train/stage29_base_delta_mean_step",
        "train/stage29_hard_delta_mean_step",
        "train/stage29_transition_delta_mean_step",
        "train/stage29_mature_delta_mean_step",
        "train/stage29_headroom_mean_step",
        "train/stage29_hard_scene_fraction_step",
        "train/stage29_transition_scene_fraction_step",
        "train/stage29_mature_scene_fraction_step",
        "train/stage29_positive_fraction_step",
        "train/stage29_negative_fraction_step",
        "train/stage29_neutral_fraction_step",
        "train/stage29_policy_active_scene_fraction_step",
        "train/stage29_safety_override_fraction_step",
        "train/stage29_catastrophic_fraction_step",
        "train/stage29_mean_bc_weight_step",
        "train/stage29_mean_kl_weight_step",
        "train/stage29_mean_exact_kl_step",
        "train/stage29_paired_scale_mean_step",
    }
    active_tags = {
        "train/diff_decoder_grad_norm_step",
        *{
            f"train/decoder_layer_{layer}_{part}_grad_norm_step"
            for layer in (0, 1)
            for part in ("shared_attention", "ffn", "time_modulation", "regression")
        },
    }
    frozen_tags = {
        "train/decoder_layer_0_classification_grad_norm_step",
        "train/decoder_layer_1_classification_grad_norm_step",
        "train/perception_grad_norm_step", "train/value_selector_grad_norm_step",
        "train/stage23_selector_grad_norm_step", "train/stage24_selector_grad_norm_step",
        "train/stage25_selector_grad_norm_step", "train/paired_risk_grad_norm_step",
        "train/classification_grad_norm_step",
    }
    missing = sorted((finite_tags | active_tags | frozen_tags) - tags)
    if missing:
        raise RuntimeError(f"Stage29 TensorBoard tags missing: {missing}")
    values = {
        tag: scalar_values(events, tag, expected_count)
        for tag in finite_tags | active_tags | frozen_tags
    }
    if any(min(values[tag]) <= 0 for tag in active_tags):
        raise RuntimeError("Stage29 permitted decoder gradient is inactive")
    if any(any(value != 0 for value in values[tag]) for tag in frozen_tags):
        raise RuntimeError("Stage29 frozen-module gradient is nonzero")
    signal_health = assess_training_signal_health(
        sum(values["train/stage29_policy_active_scene_fraction_step"]) / expected_count,
        sum(values["train/stage29_positive_fraction_step"]) / expected_count,
        sum(values["train/stage29_neutral_fraction_step"]) / expected_count,
    )
    if args.phase == "formal" and not signal_health["passed"]:
        raise RuntimeError(f"Stage29 formal training signal is unhealthy: {signal_health}")
    result = {
        "schema_version": 1, "stage": 29, "phase": args.phase,
        "branch": args.branch, "passed": True,
        "training_mode": entry["training_mode"], "training_selector_role": role,
        "public_checkpoint_sha256": PUBLIC_SHA,
        "input_freeze": str(args.inputs.resolve()), "input_freeze_sha256": INPUTS_SHA,
        "checkpoint_freeze": str(args.freeze.resolve()),
        "checkpoint_freeze_sha256": file_sha256(args.freeze),
        "num_logged_optimizer_steps": expected_count,
        "checkpoints": checkpoint_audits,
        "finite_rewards_advantages_log_probs": True,
        "active_decoder_gradients": True,
        "zero_frozen_gradients": True,
        "training_signal_health": signal_health,
        "training_signal_gate_definition": SIGNAL_GATE_DEFINITION,
        "training_signal_gate_amendment": str(amendment_path.resolve()),
        "training_signal_gate_amendment_sha256": AMENDMENT_SHA,
        "formal_training_signal_health_required": args.phase == "formal",
        "frozen_reference_bitwise_equal_public": True,
        "frozen_training_selector_bitwise_equal": True,
        "deployment_selector_required_for_evaluation": True,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Stage29 audit: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
