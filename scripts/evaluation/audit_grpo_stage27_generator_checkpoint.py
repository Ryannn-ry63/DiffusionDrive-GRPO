#!/usr/bin/env python3
"""Fail-closed Stage27 public-base generator checkpoint and gradient audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PUBLIC_SHA = (
    "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
)
INPUT_AUDIT_SHA = (
    "ea3a9d173d37bd22e3b91d6a4ea3f857edf94afbba63e9dcfbe94febbe98ee89"
)
SELECTOR_SHA = (
    "023d7b6b77bb8fa2dc3e779849f3716688abf0796b019416494c80b29615b691"
)
MODEL_PREFIX = "_transfuser_model."
DECODER_PREFIX = MODEL_PREFIX + "_trajectory_head.diff_decoder."
LAYERS_PREFIX = DECODER_PREFIX + "layers."
REFERENCE_PREFIX = MODEL_PREFIX + "_trajectory_head.ref_policy."
CLASSIFICATION_MARKER = ".task_decoder.plan_cls_branch."
SELECTOR_PREFIXES = (
    MODEL_PREFIX + "_trajectory_head.stage24_selector.",
    MODEL_PREFIX + "_trajectory_head.stage25_selector.",
)
SELECTOR_BUFFER_NAMES = (
    "_stage24_selector_training_updates",
    "_stage25_selector_training_updates",
    "_stage24_embedding_count",
    "_stage24_embedding_sum",
    "_stage24_embedding_sum_sq",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_state(payload: dict) -> dict[str, torch.Tensor]:
    state = payload["state_dict"]
    return {
        (name[len("agent."):] if name.startswith("agent.") else name): value
        for name, value in state.items()
    }


def scalar_values(events: EventAccumulator, tag: str, expected_count: int) -> list[float]:
    values = [float(event.value) for event in events.Scalars(tag)]
    if len(values) != expected_count:
        raise RuntimeError(
            f"Stage27 scalar count mismatch for {tag}: "
            f"{len(values)} != {expected_count}"
        )
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"Stage27 scalar is non-finite: {tag}")
    return values


def parse_overrides(path: Path) -> dict[str, str]:
    result = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not line.startswith("- ") or "=" not in line:
            raise RuntimeError(f"invalid Stage27 Hydra override line: {raw_line}")
        key, value = line[2:].split("=", 1)
        result[key.removeprefix("+")] = value
    return result


def require_overrides(
    actual: dict[str, str],
    expected: dict[str, str],
) -> None:
    for key, value in expected.items():
        if actual.get(key) != value:
            raise RuntimeError(
                f"Stage27 Hydra override drifted: {key}: "
                f"{actual.get(key)!r} != {value!r}"
            )


def audit_checkpoint(
    checkpoint_record: dict,
    base_state: dict[str, torch.Tensor],
    selector_state: dict[str, torch.Tensor],
) -> dict:
    path = Path(checkpoint_record["path"])
    if file_sha256(path) != checkpoint_record["sha256"]:
        raise RuntimeError(f"Stage27 frozen checkpoint SHA drifted: {path}")
    payload = torch.load(path, map_location="cpu")
    if int(payload.get("global_step", -1)) != checkpoint_record["global_step"]:
        raise RuntimeError("Stage27 checkpoint global step drifted")
    if int(payload.get("epoch", -1)) != checkpoint_record["epoch"]:
        raise RuntimeError("Stage27 checkpoint epoch drifted")
    current = canonical_state(payload)

    missing_base = sorted(set(base_state) - set(current))
    if missing_base:
        raise RuntimeError(
            f"Stage27 checkpoint lost public-base tensor: {missing_base[0]}"
        )
    changed_allowed = []
    changed_forbidden = []
    changed_by_layer = {0: 0, 1: 0}
    for name, base_tensor in base_state.items():
        if torch.equal(base_tensor.cpu(), current[name].cpu()):
            continue
        allowed = (
            name.startswith(LAYERS_PREFIX)
            and CLASSIFICATION_MARKER not in name
        )
        if not allowed:
            changed_forbidden.append(name)
            continue
        changed_allowed.append(name)
        for layer in changed_by_layer:
            if f".layers.{layer}." in name:
                changed_by_layer[layer] += 1
    if changed_forbidden:
        raise RuntimeError(
            f"Stage27 changed frozen public tensor: {changed_forbidden[0]}"
        )
    if len(changed_allowed) != 64 or changed_by_layer != {0: 32, 1: 32}:
        raise RuntimeError(
            "Stage27 trainable tensor boundary drifted: "
            f"count={len(changed_allowed)}, layers={changed_by_layer}"
        )

    reference_mismatch = []
    base_decoder = {
        name[len(DECODER_PREFIX):]: value
        for name, value in base_state.items()
        if name.startswith(DECODER_PREFIX)
    }
    for suffix, base_tensor in base_decoder.items():
        name = REFERENCE_PREFIX + suffix
        if name not in current or not torch.equal(
            base_tensor.cpu(), current[name].cpu()
        ):
            reference_mismatch.append(name)
    if reference_mismatch:
        raise RuntimeError(
            f"Stage27 frozen reference drifted: {reference_mismatch[0]}"
        )

    selector_mismatch = []
    expected_selector_names = [
        name for name in selector_state
        if name.startswith(SELECTOR_PREFIXES)
        or name in {
            MODEL_PREFIX + "_trajectory_head." + buffer
            for buffer in SELECTOR_BUFFER_NAMES
        }
    ]
    if not expected_selector_names:
        raise RuntimeError("Stage27 frozen selector state is empty")
    for name in expected_selector_names:
        if name not in current or not torch.equal(
            selector_state[name].cpu(), current[name].cpu()
        ):
            selector_mismatch.append(name)
    if selector_mismatch:
        raise RuntimeError(
            f"Stage27 frozen selector drifted: {selector_mismatch[0]}"
        )

    optimizer_states = payload.get("optimizer_states", [])
    if len(optimizer_states) != 1:
        raise RuntimeError("Stage27 checkpoint must contain exactly one optimizer")
    groups = optimizer_states[0]["param_groups"]
    group_summary = sorted(
        (
            float(group.get("lr_scale", -1.0)),
            float(group["lr"]),
            len(group["params"]),
            float(group["weight_decay"]),
        )
        for group in groups
    )
    if group_summary != [
        (0.1, 1e-7, 32, 0.0001),
        (1.0, 1e-6, 32, 0.0001),
    ]:
        raise RuntimeError(
            f"Stage27 optimizer parameter groups drifted: {group_summary}"
        )
    optimizer_state = optimizer_states[0]["state"]
    if len(optimizer_state) != 64:
        raise RuntimeError(
            f"Stage27 optimizer state count drifted: {len(optimizer_state)}"
        )
    for parameter_id, state in optimizer_state.items():
        for field, value in state.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise RuntimeError(
                    "Stage27 optimizer update is non-finite: "
                    f"{parameter_id}/{field}"
                )

    return {
        "path": str(path.resolve()),
        "sha256": checkpoint_record["sha256"],
        "epoch": checkpoint_record["epoch"],
        "global_step": checkpoint_record["global_step"],
        "public_tensor_count": len(base_state),
        "changed_allowed_count": len(changed_allowed),
        "changed_forbidden_count": len(changed_forbidden),
        "changed_by_layer": changed_by_layer,
        "reference_mismatch_count": len(reference_mismatch),
        "selector_mismatch_count": len(selector_mismatch),
        "optimizer_groups": group_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("audit", "formal"), required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if file_sha256(args.base) != PUBLIC_SHA:
        raise RuntimeError("Stage27 public checkpoint SHA mismatch")
    if file_sha256(args.input_audit) != INPUT_AUDIT_SHA:
        raise RuntimeError("Stage27 Phase3 input-audit SHA mismatch")
    input_audit = json.loads(args.input_audit.read_text(encoding="utf-8"))
    if not input_audit.get("passed"):
        raise RuntimeError("Stage27 Phase3 input audit did not pass")
    selector_path = Path(input_audit["selector_checkpoint"])
    if file_sha256(selector_path) != SELECTOR_SHA:
        raise RuntimeError("Stage27 selector checkpoint SHA mismatch")
    calibration = json.loads(
        Path(input_audit["calibration"]).read_text(encoding="utf-8")
    )

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if (
        not freeze.get("passed")
        or freeze.get("stage") != 27
        or freeze.get("phase") != args.phase
    ):
        raise RuntimeError("Stage27 checkpoint freeze metadata drifted")
    expected_steps = [1] if args.phase == "audit" else [80, 160]
    if freeze.get("expected_global_steps") != expected_steps:
        raise RuntimeError("Stage27 frozen checkpoint steps drifted")

    overrides_path = Path(freeze["hydra_overrides"])
    event_path = Path(freeze["tensorboard_event"])
    if file_sha256(overrides_path) != freeze["hydra_overrides_sha256"]:
        raise RuntimeError("Stage27 Hydra overrides changed after freezing")
    if file_sha256(event_path) != freeze["tensorboard_event_sha256"]:
        raise RuntimeError("Stage27 TensorBoard event changed after freezing")
    overrides = parse_overrides(overrides_path)
    expected_overrides = {
        "agent.checkpoint_path": str(args.base.resolve()),
        "agent.reference_checkpoint_path": str(args.base.resolve()),
        "agent.lr": "1e-6",
        "agent.config.grpo_training_mode":
            "stage27_public_diffgrpo_selected_set",
        "agent.config.generation_policy_algorithm": "diffgrpo_selected_set",
        "agent.config.grpo_decoder_gradient_scope": "all_layers",
        "agent.config.grpo_decoder_layer0_lr_mult": "0.1",
        "agent.config.inference_selector_source":
            "trajectory_relative_harm_v3",
        "agent.config.stage25_selector_checkpoint_path":
            str(selector_path.resolve()),
        "agent.config.stage25_selector_calibration_path":
            input_audit["calibration"],
        "agent.config.stage24_selector_residual_margin":
            str(calibration["residual_margin"]),
        "agent.config.stage25_selector_risk_threshold":
            str(calibration["risk_threshold"]),
        "agent.config.stage24_selector_ood_threshold":
            str(calibration["ood_threshold"]),
        "agent.config.stage23_generator_train_manifest_path":
            input_audit["manifest"],
        "agent.config.grpo_reward_mode": "pdms",
        "agent.config.grpo_scene_weight_mode": "uniform",
        "agent.config.generation_policy_loss_weight": "1.0",
        "agent.config.diffgrpo_bc_weight": "0.1",
        "agent.config.diffgrpo_step_discount": "0.6",
        "agent.config.diffgrpo_group_size": "8",
        "agent.config.diffgrpo_paired_regular_kl_weight": "0.1",
        "agent.config.diffgrpo_paired_mature_kl_weight": "0.5",
        "agent.config.diffusion_truncation_timestep": "32",
        "agent.config.diffusion_roll_timesteps": "[32,24,16,8,0]",
        "agent.config.diffusion_scheduler_num_inference_steps": "125",
        "dataloader.params.batch_size": "1",
        "trainer.params.accumulate_grad_batches": "8",
        "trainer.params.gradient_clip_val": "1.0",
        "trainer.params.devices": "8",
        "trainer.params.strategy": "ddp",
        "trainer.params.precision": "32-true",
        "trainer.params.max_epochs": "1" if args.phase == "audit" else "2",
        "trainer.params.limit_train_batches":
            "8" if args.phase == "audit" else "1.0",
        "agent.config.grpo_checkpoint_every_n_train_steps":
            "1" if args.phase == "audit" else "0",
    }
    require_overrides(overrides, expected_overrides)
    if args.phase == "audit":
        require_overrides(overrides, {"trainer.params.max_steps": "1"})
    elif "trainer.params.max_steps" in overrides:
        raise RuntimeError("Stage27 formal run unexpectedly limits max_steps")

    base_payload = torch.load(args.base, map_location="cpu")
    base_state = canonical_state(base_payload)
    if len(base_state) != 763:
        raise RuntimeError(
            f"Stage27 public-base state count drifted: {len(base_state)}"
        )
    selector_payload = torch.load(selector_path, map_location="cpu")
    selector_state = canonical_state(selector_payload)
    checkpoint_audits = [
        audit_checkpoint(record, base_state, selector_state)
        for record in freeze["checkpoints"]
    ]

    expected_count = expected_steps[-1]
    events = EventAccumulator(str(event_path.parent))
    events.Reload()
    tags = set(events.Tags()["scalars"])
    finite_tags = {
        "train/loss_step",
        "train/generation_grpo_loss_step",
        "train/diffgrpo_bc_loss_step",
        "train/generation_reference_kl_loss_step",
        "train/raw_reward_mean_step",
        "train/stage23_advantage_mean_step",
        "train/diffgrpo_mean_current_log_prob_step",
        "train/stage23_base_delta_mean_step",
        "train/stage23_safety_override_fraction_step",
        "train/stage23_catastrophic_fraction_step",
        "train/stage23_mean_exact_kl_step",
    }
    active_gradient_tags = {
        "train/diff_decoder_grad_norm_step",
        "train/decoder_layer_0_shared_attention_grad_norm_step",
        "train/decoder_layer_0_ffn_grad_norm_step",
        "train/decoder_layer_0_time_modulation_grad_norm_step",
        "train/decoder_layer_0_regression_grad_norm_step",
        "train/decoder_layer_1_shared_attention_grad_norm_step",
        "train/decoder_layer_1_ffn_grad_norm_step",
        "train/decoder_layer_1_time_modulation_grad_norm_step",
        "train/decoder_layer_1_regression_grad_norm_step",
    }
    frozen_gradient_tags = {
        "train/decoder_layer_0_classification_grad_norm_step",
        "train/decoder_layer_1_classification_grad_norm_step",
        "train/perception_grad_norm_step",
        "train/value_selector_grad_norm_step",
        "train/stage23_selector_grad_norm_step",
        "train/stage24_selector_grad_norm_step",
        "train/stage25_selector_grad_norm_step",
        "train/paired_risk_grad_norm_step",
        "train/classification_grad_norm_step",
    }
    required_tags = finite_tags | active_gradient_tags | frozen_gradient_tags
    missing_tags = sorted(required_tags - tags)
    if missing_tags:
        raise RuntimeError(f"Stage27 TensorBoard tags are missing: {missing_tags}")
    metric_values = {
        tag: scalar_values(events, tag, expected_count)
        for tag in required_tags
    }
    for tag in active_gradient_tags:
        if min(metric_values[tag]) <= 0:
            raise RuntimeError(f"Stage27 permitted gradient inactive: {tag}")
    for tag in frozen_gradient_tags:
        if any(value != 0.0 for value in metric_values[tag]):
            raise RuntimeError(f"Stage27 forbidden gradient active: {tag}")

    result = {
        "schema_version": 1,
        "stage": 27,
        "phase": args.phase,
        "passed": True,
        "public_checkpoint": str(args.base.resolve()),
        "public_checkpoint_sha256": PUBLIC_SHA,
        "input_audit": str(args.input_audit.resolve()),
        "input_audit_sha256": INPUT_AUDIT_SHA,
        "checkpoint_freeze": str(args.freeze.resolve()),
        "checkpoint_freeze_sha256": file_sha256(args.freeze),
        "hydra_overrides_sha256": freeze["hydra_overrides_sha256"],
        "tensorboard_event_sha256": freeze["tensorboard_event_sha256"],
        "num_logged_optimizer_steps": expected_count,
        "checkpoints": checkpoint_audits,
        "finite_rewards_advantages_log_probs": True,
        "active_decoder_gradients": True,
        "zero_frozen_gradients": True,
        "frozen_reference_bitwise_equal_public": True,
        "frozen_selector_bitwise_equal_selected_multi": True,
    }
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite Stage27 generator audit: {args.output}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
