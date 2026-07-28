#!/usr/bin/env python3
"""Audit Stage31 gradients, 64-tensor boundary, and frozen provenance."""

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
        "_stage24_selector_training_updates", "_stage25_selector_training_updates",
        "_stage24_embedding_count", "_stage24_embedding_sum",
        "_stage24_embedding_sum_sq",
    )
}


def sha256(path: Path) -> str:
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
        if line:
            if not line.startswith("- ") or "=" not in line:
                raise RuntimeError(f"invalid Stage31 override: {raw}")
            key, value = line[2:].split("=", 1)
            result[key.removeprefix("+")] = value
    return result


def require(actual: dict[str, str], expected: dict[str, str]) -> None:
    for key, wanted in expected.items():
        if actual.get(key) != wanted:
            raise RuntimeError(
                f"Stage31 override drifted: {key}: {actual.get(key)!r} != {wanted!r}"
            )


def scalar_values(events: EventAccumulator, tag: str, count: int) -> list[float]:
    values = [float(event.value) for event in events.Scalars(tag)]
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"Stage31 invalid scalar series: {tag}")
    return values


def audit_checkpoint(record, base_state, selector_state):
    path = Path(record["path"])
    if sha256(path) != record["sha256"]:
        raise RuntimeError("Stage31 checkpoint SHA drifted")
    payload = torch.load(path, map_location="cpu")
    current = canonical_state(payload)
    if (
        int(payload.get("epoch", -1)) != record["epoch"]
        or int(payload.get("global_step", -1)) != record["global_step"]
    ):
        raise RuntimeError("Stage31 checkpoint metadata drifted")
    missing = sorted(set(base_state) - set(current))
    if missing:
        raise RuntimeError(f"Stage31 checkpoint lost public tensor: {missing[0]}")
    allowed, forbidden = [], []
    by_layer = {0: 0, 1: 0}
    for name, base_tensor in base_state.items():
        if torch.equal(base_tensor.cpu(), current[name].cpu()):
            continue
        if name.startswith(LAYERS_PREFIX) and CLASSIFICATION not in name:
            allowed.append(name)
            for layer in by_layer:
                if f".layers.{layer}." in name:
                    by_layer[layer] += 1
        else:
            forbidden.append(name)
    if forbidden:
        raise RuntimeError(f"Stage31 changed frozen public tensor: {forbidden[0]}")
    if len(allowed) != 64 or by_layer != {0: 32, 1: 32}:
        raise RuntimeError(
            f"Stage31 changed decoder boundary drifted: {len(allowed)}, {by_layer}"
        )
    base_decoder = {
        name[len(DECODER_PREFIX):]: tensor
        for name, tensor in base_state.items() if name.startswith(DECODER_PREFIX)
    }
    for suffix, tensor in base_decoder.items():
        name = REFERENCE_PREFIX + suffix
        if name not in current or not torch.equal(tensor.cpu(), current[name].cpu()):
            raise RuntimeError(f"Stage31 frozen reference drifted: {name}")
    selector_names = [
        name for name in selector_state
        if name.startswith(SELECTOR_PREFIXES) or name in SELECTOR_BUFFERS
    ]
    if not selector_names:
        raise RuntimeError("Stage31 selector state is empty")
    for name in selector_names:
        if name not in current or not torch.equal(
            selector_state[name].cpu(), current[name].cpu()
        ):
            raise RuntimeError(f"Stage31 frozen selector drifted: {name}")
    optimizers = payload.get("optimizer_states", [])
    if len(optimizers) != 1 or len(optimizers[0].get("state", {})) != 64:
        raise RuntimeError("Stage31 optimizer state boundary drifted")
    groups = sorted(
        (float(group.get("lr_scale", -1)), float(group["lr"]),
         len(group["params"]), float(group["weight_decay"]))
        for group in optimizers[0]["param_groups"]
    )
    if groups != [(0.1, 1e-7, 32, 0.0), (1.0, 1e-6, 32, 0.0)]:
        raise RuntimeError(f"Stage31 optimizer groups drifted: {groups}")
    for state in optimizers[0]["state"].values():
        for value in state.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise RuntimeError("Stage31 optimizer has non-finite state")
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
    parser.add_argument("--branch", choices=("DP", "DPF"), required=True)
    parser.add_argument("--holdout", type=int, choices=range(4), required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--cv-freeze", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.base) != STAGE30_PUBLIC_SHA256:
        raise RuntimeError("Stage31 public checkpoint drifted")
    if sha256(args.selector) != STAGE30_SELECTOR_SHA256:
        raise RuntimeError("Stage31 selector checkpoint drifted")
    if sha256(args.calibration) != STAGE30_CALIBRATION_SHA256:
        raise RuntimeError("Stage31 selector calibration drifted")
    cv = json.loads(args.cv_freeze.read_text(encoding="utf-8"))
    entry = cv["folds"][args.holdout]
    if cv.get("stage") != 30 or entry["holdout_fold"] != args.holdout:
        raise RuntimeError("Stage31 CV input freeze drifted")
    for path, expected in (
        (Path(entry["path"]), entry["sha256"]),
        (Path(cv["bucket_manifest"]), cv["bucket_manifest_sha256"]),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"Stage31 frozen input drifted: {path}")
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    expected_steps = [1] if args.phase == "audit" else [48, 96, 144, 192]
    if not (
        freeze.get("passed") and freeze.get("stage") == 31
        and freeze.get("phase") == args.phase
        and freeze.get("branch") == args.branch
        and freeze.get("holdout_fold") == args.holdout
        and freeze.get("expected_global_steps") == expected_steps
    ):
        raise RuntimeError("Stage31 checkpoint freeze semantics drifted")
    overrides_path = Path(freeze["hydra_overrides"])
    event_path = Path(freeze["tensorboard_event"])
    if (
        sha256(overrides_path) != freeze["hydra_overrides_sha256"]
        or sha256(event_path) != freeze["tensorboard_event_sha256"]
    ):
        raise RuntimeError("Stage31 run artifact SHA drifted")
    mode = (
        "stage31_public_deployed_pair"
        if args.branch == "DP"
        else "stage31_public_deployed_frontier"
    )
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    overrides = parse_overrides(overrides_path)
    require(overrides, {
        "agent.checkpoint_path": str(args.base.resolve()),
        "agent.reference_checkpoint_path": str(args.base.resolve()),
        "agent.lr": "1e-6",
        "agent.config.grpo_training_mode": mode,
        "agent.config.generation_policy_algorithm": "diffgrpo_deployed_selected_set",
        "agent.config.stage23_generator_train_manifest_path": entry["path"],
        "agent.config.stage31_bucket_manifest_path": cv["bucket_manifest"],
        "agent.config.stage25_selector_checkpoint_path": str(args.selector.resolve()),
        "agent.config.stage25_selector_calibration_path": str(args.calibration),
        "agent.config.stage24_selector_residual_margin": str(calibration["residual_margin"]),
        "agent.config.stage25_selector_risk_threshold": str(calibration["risk_threshold"]),
        "agent.config.stage24_selector_ood_threshold": str(calibration["ood_threshold"]),
        "agent.config.diffgrpo_group_size": "8",
        "agent.config.stage31_plan_sha256": "3ff3d3bcd3120b9d73a017876f942458eb3fa16651517e4e30b27cf2fff7bb0d",
        "agent.config.stage31_headroom_low": "0.75",
        "agent.config.stage31_headroom_high": "0.90",
        "agent.config.stage31_delta_scale_floor": "0.002",
        "agent.config.stage31_rank_weight": "0.5",
        "agent.config.stage31_bc_weight": "0.1",
        "agent.config.stage31_kl_weight": "0.1",
        "agent.config.stage31_safety_kl_weight": "0.5",
        "agent.config.stage31_global_bucket_composition": "[2,30,8,24]",
        "agent.config.stage31_optimizer_steps_per_epoch": "48",
        "agent.config.stage31_gradient_accumulation": "8",
        "agent.config.weight_decay": "0.0",
        "dataloader.params.batch_size": "1",
        "trainer.params.accumulate_grad_batches": "8",
        "trainer.params.use_distributed_sampler": "false",
        "trainer.params.devices": "8",
        "trainer.params.strategy": "ddp",
        "trainer.params.precision": "32-true",
        "trainer.params.gradient_clip_val": "1.0",
        "trainer.params.max_epochs": "1" if args.phase == "audit" else "4",
        "trainer.params.limit_train_batches": "8" if args.phase == "audit" else "1.0",
        "agent.config.grpo_checkpoint_every_n_train_steps": "1" if args.phase == "audit" else "48",
    })
    if args.phase == "audit":
        require(overrides, {"trainer.params.max_steps": "1"})
    elif "trainer.params.max_steps" in overrides:
        raise RuntimeError("Stage31 formal run unexpectedly limits max_steps")

    base_state = canonical_state(torch.load(args.base, map_location="cpu"))
    selector_state = canonical_state(torch.load(args.selector, map_location="cpu"))
    audits = [
        audit_checkpoint(record, base_state, selector_state)
        for record in freeze["checkpoints"]
    ]
    expected_count = expected_steps[-1]
    events = EventAccumulator(str(event_path.parent)); events.Reload()
    finite_tags = {
        "train/loss_step", "train/generation_grpo_loss_step",
        "train/diffgrpo_bc_loss_step", "train/generation_reference_kl_loss_step",
        "train/raw_reward_mean_step", "train/diffgrpo_mean_current_log_prob_step",
        *{
            f"train/stage31_{name}_step" for name in (
                "advantage_mean", "deployment_delta_mean", "hard_delta_mean",
                "transition_delta_mean", "mature_delta_mean", "headroom_mean",
                "hard_scene_fraction", "transition_scene_fraction",
                "mature_scene_fraction", "positive_fraction", "negative_fraction",
                "neutral_fraction", "tie_fraction", "policy_active_scene_fraction",
                "safety_override_fraction", "catastrophic_fraction",
                "mean_bc_weight", "mean_kl_weight", "mean_exact_kl",
                "delta_scale_mean", "frontier_rank_enabled",
                "selector_mode_disagreement", "current_selector_switch_rate",
                "public_selector_switch_rate",
            )
        },
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
    tags = set(events.Tags()["scalars"])
    missing = sorted((finite_tags | active_tags | frozen_tags) - tags)
    if missing:
        raise RuntimeError(f"Stage31 TensorBoard tags missing: {missing}")
    values = {
        tag: scalar_values(events, tag, expected_count)
        for tag in finite_tags | active_tags | frozen_tags
    }
    if any(min(values[tag]) <= 0 for tag in active_tags):
        raise RuntimeError("Stage31 permitted decoder gradient is inactive")
    if any(any(value != 0 for value in values[tag]) for tag in frozen_tags):
        raise RuntimeError("Stage31 frozen-module gradient is nonzero")
    result = {
        "schema_version": 1, "stage": 31, "phase": args.phase,
        "branch": args.branch, "holdout_fold": args.holdout, "passed": True,
        "training_mode": mode,
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
