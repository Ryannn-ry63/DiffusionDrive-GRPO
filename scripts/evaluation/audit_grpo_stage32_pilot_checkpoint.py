#!/usr/bin/env python3
"""Audit Stage32 pilot provenance, optimizer boundary, and training signals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from navsim.agents.diffusiondrive.stage30_mode_coverage import (
    STAGE30_CALIBRATION_SHA256,
    STAGE30_PUBLIC_SHA256,
    STAGE30_SELECTOR_SHA256,
)

STAGE31_PLAN_SHA256 = (
    "3ff3d3bcd3120b9d73a017876f942458eb3fa16651517e4e30b27cf2fff7bb0d"
)
STAGE32_PLAN_SHA256 = (
    "5c5f3b148c8d03fe3c714558314ebd895a8f143b94fd10fb9bdf8c4ad61cefcd"
)
SCF_OBJECTIVE_REVISION = "mean_normalized_bc_kl_v2"
MODEL_PREFIX = "_transfuser_model."
DECODER_PREFIX = MODEL_PREFIX + "_trajectory_head.diff_decoder."
LAYERS_PREFIX = DECODER_PREFIX + "layers."
REFERENCE_PREFIX = MODEL_PREFIX + "_trajectory_head.ref_policy."
CLASSIFICATION = ".task_decoder.plan_cls_branch."
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state(payload: dict) -> dict[str, torch.Tensor]:
    raw = payload.get("state_dict", payload)
    return {
        (name[len("agent."):] if name.startswith("agent.") else name): value
        for name, value in raw.items()
        if torch.is_tensor(value)
    }


def parse_overrides(path: Path) -> dict[str, str]:
    result = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            if not line.startswith("- ") or "=" not in line:
                raise RuntimeError(f"invalid Stage32 override: {raw}")
            key, value = line[2:].split("=", 1)
            result[key.removeprefix("+")] = value
    return result


def require(actual: dict[str, str], expected: dict[str, str]) -> None:
    for key, wanted in expected.items():
        if actual.get(key) != wanted:
            raise RuntimeError(
                f"Stage32 override drifted: {key}: {actual.get(key)!r} != {wanted!r}"
            )


def scalar_values(events: EventAccumulator, tag: str, count: int) -> list[float]:
    values = [float(event.value) for event in events.Scalars(tag)]
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"Stage32 invalid scalar series: {tag}")
    return values


def audit_checkpoint(record: dict, base: dict, selector: dict) -> dict:
    path = Path(record["path"])
    if sha256(path) != record["sha256"]:
        raise RuntimeError("Stage32 checkpoint SHA drifted")
    payload = torch.load(path, map_location="cpu")
    current = state(payload)
    if (
        int(payload.get("epoch", -1)) != record["epoch"]
        or int(payload.get("global_step", -1)) != record["global_step"]
    ):
        raise RuntimeError("Stage32 checkpoint metadata drifted")
    if not set(base).issubset(current):
        raise RuntimeError("Stage32 checkpoint lost public tensor")
    allowed, forbidden, by_layer = [], [], {0: 0, 1: 0}
    for name, tensor in base.items():
        if not torch.isfinite(current[name]).all():
            raise RuntimeError(f"Stage32 non-finite tensor: {name}")
        if torch.equal(tensor.cpu(), current[name].cpu()):
            continue
        if name.startswith(LAYERS_PREFIX) and CLASSIFICATION not in name:
            allowed.append(name)
            for layer in by_layer:
                if f".layers.{layer}." in name:
                    by_layer[layer] += 1
        else:
            forbidden.append(name)
    if forbidden or len(allowed) != 64 or by_layer != {0: 32, 1: 32}:
        raise RuntimeError(
            f"Stage32 decoder boundary drifted: allowed={len(allowed)}, "
            f"by_layer={by_layer}, forbidden={forbidden[:1]}"
        )
    for suffix, tensor in (
        (name[len(DECODER_PREFIX):], tensor)
        for name, tensor in base.items()
        if name.startswith(DECODER_PREFIX)
    ):
        name = REFERENCE_PREFIX + suffix
        if name not in current or not torch.equal(tensor.cpu(), current[name].cpu()):
            raise RuntimeError(f"Stage32 frozen reference drifted: {name}")
    selector_names = [
        name for name in selector
        if name.startswith(SELECTOR_PREFIXES) or name in SELECTOR_BUFFERS
    ]
    if not selector_names:
        raise RuntimeError("Stage32 selector state is empty")
    for name in selector_names:
        if name not in current or not torch.equal(selector[name].cpu(), current[name].cpu()):
            raise RuntimeError(f"Stage32 frozen selector drifted: {name}")
    optimizers = payload.get("optimizer_states", [])
    if len(optimizers) != 1 or len(optimizers[0].get("state", {})) != 64:
        raise RuntimeError("Stage32 optimizer state boundary drifted")
    groups = sorted(
        (float(group.get("lr_scale", -1)), float(group["lr"]),
         len(group["params"]), float(group["weight_decay"]))
        for group in optimizers[0]["param_groups"]
    )
    if groups != [(0.1, 1e-7, 32, 0.0), (1.0, 1e-6, 32, 0.0)]:
        raise RuntimeError(f"Stage32 optimizer groups drifted: {groups}")
    for item in optimizers[0]["state"].values():
        for value in item.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise RuntimeError("Stage32 optimizer has non-finite state")
    return {
        **record,
        "changed_allowed_count": len(allowed),
        "changed_forbidden_count": len(forbidden),
        "changed_by_layer": by_layer,
        "optimizer_groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("audit", "formal"), required=True)
    parser.add_argument("--branch", choices=("DPEL", "SCF"), required=True)
    parser.add_argument("--holdout", type=int, choices=range(4), required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--cv-freeze", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.base) != STAGE30_PUBLIC_SHA256:
        raise RuntimeError("Stage32 public checkpoint drifted")
    if sha256(args.selector) != STAGE30_SELECTOR_SHA256:
        raise RuntimeError("Stage32 selector checkpoint drifted")
    if sha256(args.calibration) != STAGE30_CALIBRATION_SHA256:
        raise RuntimeError("Stage32 selector calibration drifted")
    cv = json.loads(args.cv_freeze.read_text(encoding="utf-8"))
    entry = cv["folds"][args.holdout]
    if cv.get("stage") != 30 or entry["holdout_fold"] != args.holdout:
        raise RuntimeError("Stage32 CV input freeze drifted")
    for path, expected in (
        (Path(entry["path"]), entry["sha256"]),
        (Path(cv["bucket_manifest"]), cv["bucket_manifest_sha256"]),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"Stage32 frozen input drifted: {path}")
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    expected_steps = [1] if args.phase == "audit" else [192, 384, 576]
    if not (
        freeze.get("passed") and freeze.get("stage") == 32
        and freeze.get("phase") == args.phase
        and freeze.get("branch") == args.branch
        and freeze.get("holdout_fold") == args.holdout
        and freeze.get("expected_global_steps") == expected_steps
    ):
        raise RuntimeError("Stage32 checkpoint freeze semantics drifted")
    overrides_path = Path(freeze["hydra_overrides"])
    event_path = Path(freeze["tensorboard_event"])
    if (
        sha256(overrides_path) != freeze["hydra_overrides_sha256"]
        or sha256(event_path) != freeze["tensorboard_event_sha256"]
    ):
        raise RuntimeError("Stage32 run artifact SHA drifted")
    mode = (
        "stage32_public_deployed_extended"
        if args.branch == "DPEL" else "stage32_selector_aware_frontier"
    )
    overrides = parse_overrides(overrides_path)
    require(overrides, {
        "agent.checkpoint_path": str(args.base.resolve()),
        "agent.reference_checkpoint_path": str(args.base.resolve()),
        "agent.lr": "1e-6",
        "agent.config.grpo_training_mode": mode,
        "agent.config.generation_policy_algorithm": "diffgrpo_deployed_selected_set",
        "agent.config.grpo_decoder_gradient_scope": "all_layers",
        "agent.config.grpo_decoder_layer0_lr_mult": "0.1",
        "agent.config.inference_selector_source": "trajectory_relative_harm_v3",
        "agent.config.stage23_generator_train_manifest_path": entry["path"],
        "agent.config.stage31_bucket_manifest_path": cv["bucket_manifest"],
        "agent.config.stage32_bucket_manifest_path": cv["bucket_manifest"],
        "agent.config.stage31_plan_sha256": STAGE31_PLAN_SHA256,
        "agent.config.stage32_plan_sha256": STAGE32_PLAN_SHA256,
        "agent.config.stage31_global_bucket_composition": "[2,30,8,24]",
        "agent.config.stage32_global_bucket_composition": "[2,30,8,24]",
        "agent.config.stage31_optimizer_steps_per_epoch": "48",
        "agent.config.stage32_optimizer_steps_per_epoch": "48",
        "agent.config.stage31_gradient_accumulation": "8",
        "agent.config.stage32_gradient_accumulation": "8",
        "agent.config.stage32_frontier_pool_size": "4",
        "agent.config.stage32_frontier_risk_margin": "0.10",
        "agent.config.stage32_frontier_owner_margin": "0.001",
        "agent.config.stage32_frontier_weight": "0.5",
        "agent.config.stage32_frontier_mature_cap": "0.25",
        "agent.config.diffgrpo_group_size": "8",
        "agent.config.diffgrpo_bc_weight": "0.1",
        "agent.config.weight_decay": "0.0",
        "dataloader.params.batch_size": "1",
        "trainer.params.accumulate_grad_batches": "8",
        "trainer.params.use_distributed_sampler": "false",
        "trainer.params.devices": "8",
        "trainer.params.strategy": "ddp",
        "trainer.params.precision": "32-true",
        "trainer.params.gradient_clip_val": "1.0",
        "trainer.params.max_epochs": "1" if args.phase == "audit" else "12",
        "trainer.params.limit_train_batches": "8" if args.phase == "audit" else "1.0",
        "agent.config.grpo_checkpoint_every_n_train_steps": "1" if args.phase == "audit" else "48",
    })
    if args.branch == "SCF":
        require(overrides, {
            "agent.config.stage32_scf_objective_revision": SCF_OBJECTIVE_REVISION,
        })
    if args.phase == "audit":
        require(overrides, {"trainer.params.max_steps": "1"})
    elif "trainer.params.max_steps" in overrides:
        raise RuntimeError("Stage32 formal run unexpectedly limits max_steps")

    base_state = state(torch.load(args.base, map_location="cpu"))
    selector_state = state(torch.load(args.selector, map_location="cpu"))
    audits = [
        audit_checkpoint(record, base_state, selector_state)
        for record in freeze["checkpoints"]
    ]
    expected_count = expected_steps[-1]
    events = EventAccumulator(str(event_path.parent))
    events.Reload()
    common = {
        "train/loss_step", "train/generation_grpo_loss_step",
        "train/diffgrpo_bc_loss_step", "train/generation_reference_kl_loss_step",
        "train/diff_decoder_grad_norm_step",
    }
    names = (
        ("advantage_mean", "deployment_delta_mean", "hard_delta_mean",
         "transition_delta_mean", "mature_delta_mean", "positive_fraction",
         "negative_fraction", "neutral_fraction", "policy_active_scene_fraction",
         "safety_override_fraction", "catastrophic_fraction", "mean_bc_weight",
         "mean_kl_weight", "mean_exact_kl", "delta_scale_mean",
         "selector_mode_disagreement", "current_selector_switch_rate",
         "public_selector_switch_rate")
        if args.branch == "SCF" else
        ("advantage_mean", "deployment_delta_mean", "hard_delta_mean",
         "transition_delta_mean", "mature_delta_mean", "headroom_mean",
         "hard_scene_fraction", "transition_scene_fraction", "mature_scene_fraction",
         "positive_fraction", "negative_fraction", "neutral_fraction", "tie_fraction",
         "policy_active_scene_fraction", "safety_override_fraction",
         "catastrophic_fraction", "mean_bc_weight", "mean_kl_weight", "mean_exact_kl",
         "delta_scale_mean", "frontier_rank_enabled", "selector_mode_disagreement",
         "current_selector_switch_rate", "public_selector_switch_rate")
    )
    metric_prefix = "stage32" if args.branch == "SCF" else "stage31"
    finite_tags = common | {f"train/{metric_prefix}_{name}_step" for name in names}
    active_tags = {
        f"train/decoder_layer_{layer}_{part}_grad_norm_step"
        for layer in (0, 1)
        for part in ("shared_attention", "ffn", "time_modulation", "regression")
    }
    active_tags.add("train/diff_decoder_grad_norm_step")
    frozen_tags = {
        "train/decoder_layer_0_classification_grad_norm_step",
        "train/decoder_layer_1_classification_grad_norm_step",
        "train/perception_grad_norm_step", "train/value_selector_grad_norm_step",
        "train/stage23_selector_grad_norm_step", "train/stage24_selector_grad_norm_step",
        "train/stage25_selector_grad_norm_step", "train/paired_risk_grad_norm_step",
        "train/classification_grad_norm_step",
    }
    tags = set(events.Tags()["scalars"])
    missing = sorted((finite_tags | active_tags | frozen_tags) - tags)
    if missing:
        raise RuntimeError(f"Stage32 TensorBoard tags missing: {missing}")
    values = {
        tag: scalar_values(events, tag, expected_count)
        for tag in finite_tags | active_tags | frozen_tags
    }
    if any(min(values[tag]) <= 0 for tag in active_tags):
        raise RuntimeError("Stage32 permitted decoder gradient is inactive")
    if any(any(value != 0 for value in values[tag]) for tag in frozen_tags):
        raise RuntimeError("Stage32 frozen-module gradient is nonzero")
    result = {
        "schema_version": 1, "stage": 32, "phase": args.phase,
        "branch": args.branch, "holdout_fold": args.holdout, "passed": True,
        "training_mode": mode,
        "objective_revision": freeze.get("objective_revision"),
        "public_checkpoint_sha256": STAGE30_PUBLIC_SHA256,
        "selector_checkpoint_sha256": STAGE30_SELECTOR_SHA256,
        "calibration_sha256": STAGE30_CALIBRATION_SHA256,
        "cv_freeze": str(args.cv_freeze.resolve()),
        "cv_freeze_sha256": sha256(args.cv_freeze),
        "checkpoint_freeze": str(args.freeze.resolve()),
        "checkpoint_freeze_sha256": sha256(args.freeze),
        "num_logged_optimizer_steps": expected_count,
        "checkpoints": audits,
        "finite_rewards_advantages_log_probs": True,
        "active_decoder_gradients": True,
        "zero_frozen_gradients": True,
        "frozen_reference_bitwise_equal_public": True,
        "frozen_training_selector_bitwise_equal": True,
        "exact_global_bucket_sampler": True,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
