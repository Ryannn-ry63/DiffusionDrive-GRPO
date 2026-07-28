#!/usr/bin/env python3
"""Fail-closed Stage34 provenance, optimizer, trace, and gradient audit."""

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
from navsim.agents.diffusiondrive.stage34_contract import (
    STAGE34_OBJECTIVE_REVISION,
    STAGE34_PLAN_SHA256,
    STAGE34_ROUTE_EXPECTED,
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
        (name[len("agent.") :] if name.startswith("agent.") else name): value
        for name, value in raw.items()
        if torch.is_tensor(value)
    }


def parse_overrides(path: Path) -> dict[str, str]:
    result = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if not line.startswith("- ") or "=" not in line:
            raise RuntimeError(f"invalid Stage34 override: {raw}")
        key, value = line[2:].split("=", 1)
        result[key.removeprefix("+")] = value
    return result


def require(actual: dict[str, str], expected: dict[str, str]) -> None:
    for key, wanted in expected.items():
        if actual.get(key) != wanted:
            raise RuntimeError(
                f"Stage34 override drifted: {key}: "
                f"{actual.get(key)!r} != {wanted!r}"
            )


def scalar_values(
    events: EventAccumulator, tag: str, count: int
) -> list[float]:
    values = [float(event.value) for event in events.Scalars(tag)]
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise RuntimeError(
            f"Stage34 invalid scalar series {tag}: "
            f"count={len(values)} expected={count}"
        )
    return values


def audit_checkpoint(
    record: dict,
    base: dict[str, torch.Tensor],
    selector: dict[str, torch.Tensor],
) -> dict:
    path = Path(record["path"])
    if sha256(path) != record["sha256"]:
        raise RuntimeError("Stage34 checkpoint SHA drifted")
    payload = torch.load(path, map_location="cpu")
    current = state(payload)
    if (
        int(payload.get("epoch", -1)) != record["epoch"]
        or int(payload.get("global_step", -1)) != record["global_step"]
    ):
        raise RuntimeError("Stage34 checkpoint metadata drifted")
    if not set(base).issubset(current):
        raise RuntimeError("Stage34 checkpoint lost a public-base tensor")

    allowed, forbidden, by_layer = [], [], {0: 0, 1: 0}
    for name, before in base.items():
        after = current[name]
        if not torch.isfinite(after).all():
            raise RuntimeError(f"Stage34 non-finite tensor: {name}")
        if torch.equal(before.cpu(), after.cpu()):
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
            f"Stage34 decoder boundary drifted: allowed={len(allowed)} "
            f"by_layer={by_layer} forbidden={forbidden[:2]}"
        )

    for name, before in base.items():
        if not name.startswith(DECODER_PREFIX):
            continue
        reference_name = REFERENCE_PREFIX + name[len(DECODER_PREFIX) :]
        if reference_name not in current or not torch.equal(
            before.cpu(), current[reference_name].cpu()
        ):
            raise RuntimeError(f"Stage34 frozen reference drifted: {reference_name}")
    selector_names = [
        name
        for name in selector
        if name.startswith(SELECTOR_PREFIXES) or name in SELECTOR_BUFFERS
    ]
    if not selector_names:
        raise RuntimeError("Stage34 selector state is empty")
    for name in selector_names:
        if name not in current or not torch.equal(
            selector[name].cpu(), current[name].cpu()
        ):
            raise RuntimeError(f"Stage34 frozen selector drifted: {name}")

    optimizers = payload.get("optimizer_states", [])
    if len(optimizers) != 1 or len(optimizers[0].get("state", {})) != 64:
        raise RuntimeError("Stage34 optimizer state boundary drifted")
    groups = sorted(
        (
            float(group.get("lr_scale", -1)),
            float(group["lr"]),
            len(group["params"]),
            float(group["weight_decay"]),
        )
        for group in optimizers[0]["param_groups"]
    )
    if groups != [(0.1, 1e-7, 32, 0.0), (1.0, 1e-6, 32, 0.0)]:
        raise RuntimeError(f"Stage34 optimizer groups drifted: {groups}")
    for item in optimizers[0]["state"].values():
        for value in item.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise RuntimeError("Stage34 optimizer has non-finite state")
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
    parser.add_argument("--holdout", type=int, choices=(0, 1), required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--cv-freeze", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.base) != STAGE30_PUBLIC_SHA256:
        raise RuntimeError("Stage34 public-88.1 checkpoint drifted")
    if sha256(args.selector) != STAGE30_SELECTOR_SHA256:
        raise RuntimeError("Stage34 frozen S-multi selector drifted")
    if sha256(args.calibration) != STAGE30_CALIBRATION_SHA256:
        raise RuntimeError("Stage34 selector calibration drifted")
    if sha256(args.plan) != STAGE34_PLAN_SHA256:
        raise RuntimeError("Stage34 plan SHA drifted")

    cv = json.loads(args.cv_freeze.read_text(encoding="utf-8"))
    entry = cv["folds"][args.holdout]
    if cv.get("stage") != 30 or entry["holdout_fold"] != args.holdout:
        raise RuntimeError("Stage34 CV freeze semantics drifted")
    for path, expected in (
        (Path(entry["path"]), entry["sha256"]),
        (Path(cv["bucket_manifest"]), cv["bucket_manifest_sha256"]),
    ):
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Stage34 frozen input drifted: {path}")

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    expected_steps = [1] if args.phase == "audit" else [48, 96, 144, 192]
    if not (
        freeze.get("passed")
        and freeze.get("stage") == 34
        and freeze.get("phase") == args.phase
        and freeze.get("holdout_fold") == args.holdout
        and freeze.get("expected_global_steps") == expected_steps
        and freeze.get("objective_revision") == STAGE34_OBJECTIVE_REVISION
        and freeze.get("plan_sha256") == STAGE34_PLAN_SHA256
    ):
        raise RuntimeError("Stage34 checkpoint freeze semantics drifted")
    overrides_path = Path(freeze["hydra_overrides"])
    event_path = Path(freeze["tensorboard_event"])
    if (
        sha256(overrides_path) != freeze["hydra_overrides_sha256"]
        or sha256(event_path) != freeze["tensorboard_event_sha256"]
    ):
        raise RuntimeError("Stage34 run artifact SHA drifted")

    overrides = parse_overrides(overrides_path)
    required = {
        "agent.checkpoint_path": str(args.base.resolve()),
        "agent.reference_checkpoint_path": str(args.base.resolve()),
        "agent.lr": "1e-6",
        "agent.config.grpo_training_mode": "stage34_mode_aligned_frontier_grpo",
        "agent.config.generation_policy_algorithm": "diffgrpo_mode_aligned_frontier",
        "agent.config.grpo_decoder_gradient_scope": "all_layers",
        "agent.config.grpo_decoder_layer0_lr_mult": "0.1",
        "agent.config.inference_selector_source": "trajectory_relative_harm_v3",
        "agent.config.stage23_generator_train_manifest_path": entry["path"],
        "agent.config.stage34_bucket_manifest_path": cv["bucket_manifest"],
        "agent.config.stage34_plan_sha256": STAGE34_PLAN_SHA256,
        "agent.config.stage34_objective_revision": STAGE34_OBJECTIVE_REVISION,
        "agent.config.diffgrpo_group_size": "20",
        "agent.config.diffgrpo_bc_weight": "0.1",
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
        "agent.config.grpo_checkpoint_every_n_train_steps": (
            "1" if args.phase == "audit" else "48"
        ),
    }
    for name, value in STAGE34_ROUTE_EXPECTED.items():
        rendered = (
            "[" + ",".join(str(item) for item in value) + "]"
            if isinstance(value, tuple)
            else str(value).lower()
            if isinstance(value, bool)
            else str(value)
        )
        required[f"agent.config.{name}"] = rendered
    require(overrides, required)
    if args.phase == "audit":
        require(overrides, {"trainer.params.max_steps": "1"})
    elif "trainer.params.max_steps" in overrides:
        raise RuntimeError("Stage34 formal run unexpectedly limits max_steps")

    base_state = state(torch.load(args.base, map_location="cpu"))
    selector_state = state(torch.load(args.selector, map_location="cpu"))
    audits = [
        audit_checkpoint(record, base_state, selector_state)
        for record in freeze["checkpoints"]
    ]

    expected_count = expected_steps[-1]
    events = EventAccumulator(str(event_path.parent))
    events.Reload()
    common_tags = {
        "train/loss_step",
        "train/generation_grpo_loss_step",
        "train/diffgrpo_bc_loss_step",
        "train/generation_reference_kl_loss_step",
        "train/raw_reward_mean_step",
        "train/raw_reward_std_step",
        "train/diff_decoder_grad_norm_step",
    }
    metric_names = (
        "candidate_level_trace",
        "same_mode_delta_mean",
        "candidate_mean_delta",
        "deployment_delta_mean",
        "hard_selected_delta_mean",
        "mature_selected_delta_mean",
        "public_top1_delta_mean",
        "public_top5_delta_mean",
        "raw_oracle_delta_mean",
        "safe_deployable_oracle_delta_mean",
        "headroom_credit_mean",
        "active_mode_fraction",
        "positive_fraction",
        "negative_fraction",
        "safety_override_fraction",
        "catastrophic_fraction",
        "mean_bc_weight",
        "mean_kl_weight",
        "mean_exact_kl",
        "delta_scale_mean",
        "selector_mode_disagreement",
        "current_selector_switch_rate",
        "public_selector_switch_rate",
        "mode_index_alignment",
        "sampled_chain_count",
        "replayed_chain_count",
    )
    finite_tags = common_tags | {
        f"train/stage34_{name}_step" for name in metric_names
    }
    active_tags = {
        f"train/decoder_layer_{layer}_{part}_grad_norm_step"
        for layer in (0, 1)
        for part in ("shared_attention", "ffn", "time_modulation", "regression")
    }
    active_tags.add("train/diff_decoder_grad_norm_step")
    frozen_tags = {
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
    tags = set(events.Tags()["scalars"])
    missing = sorted((finite_tags | active_tags | frozen_tags) - tags)
    if missing:
        raise RuntimeError(f"Stage34 TensorBoard tags missing: {missing}")
    values = {
        tag: scalar_values(events, tag, expected_count)
        for tag in finite_tags | active_tags | frozen_tags
    }
    if any(min(values[tag]) <= 0 for tag in active_tags):
        raise RuntimeError("Stage34 permitted decoder gradient is inactive")
    if any(any(value != 0 for value in values[tag]) for tag in frozen_tags):
        raise RuntimeError("Stage34 frozen-module gradient is nonzero")
    exact_tags = {
        "train/stage34_candidate_level_trace_step": 1.0,
        "train/stage34_mode_index_alignment_step": 1.0,
        "train/stage34_sampled_chain_count_step": 20.0,
        "train/stage34_replayed_chain_count_step": 20.0,
    }
    for tag, wanted in exact_tags.items():
        if any(value != wanted for value in values[tag]):
            raise RuntimeError(f"Stage34 trace invariant drifted: {tag}")

    result = {
        "schema_version": 1,
        "stage": 34,
        "phase": args.phase,
        "holdout_fold": args.holdout,
        "passed": True,
        "training_mode": "stage34_mode_aligned_frontier_grpo",
        "objective_revision": STAGE34_OBJECTIVE_REVISION,
        "plan_sha256": STAGE34_PLAN_SHA256,
        "public_checkpoint_sha256": STAGE30_PUBLIC_SHA256,
        "selector_checkpoint_sha256": STAGE30_SELECTOR_SHA256,
        "calibration_sha256": STAGE30_CALIBRATION_SHA256,
        "cv_freeze": str(args.cv_freeze.resolve()),
        "cv_freeze_sha256": sha256(args.cv_freeze),
        "checkpoint_freeze": str(args.freeze.resolve()),
        "checkpoint_freeze_sha256": sha256(args.freeze),
        "num_logged_optimizer_steps": expected_count,
        "checkpoints": audits,
        "candidate_level_trace": True,
        "mode_index_alignment": True,
        "common_noise_sampled_and_replayed_20": True,
        "finite_rewards_advantages_log_probs": True,
        "active_decoder_gradients": True,
        "zero_frozen_gradients": True,
        "frozen_reference_bitwise_equal_public": True,
        "frozen_selector_bitwise_equal": True,
        "exact_global_bucket_sampler": True,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
