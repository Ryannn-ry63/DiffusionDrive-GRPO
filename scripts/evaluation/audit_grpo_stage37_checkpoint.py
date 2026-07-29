#!/usr/bin/env python3
"""Fail-closed Stage37 provenance, projection, and gradient audit."""

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
from navsim.agents.diffusiondrive.stage37_contract import (
    STAGE37_ALLOWED_INITIAL_SHA256,
    STAGE37_OBJECTIVE_REVISION,
    STAGE37_PLAN_SHA256,
)


MODEL = "_transfuser_model."
DECODER = MODEL + "_trajectory_head.diff_decoder."
REFERENCE = MODEL + "_trajectory_head.ref_policy."
CLASSIFICATION = ".task_decoder.plan_cls_branch."
SELECTOR_PREFIXES = (
    MODEL + "_trajectory_head.stage24_selector.",
    MODEL + "_trajectory_head.stage25_selector.",
)
SELECTOR_BUFFERS = {
    MODEL + "_trajectory_head." + name for name in (
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


def state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    raw = payload.get("state_dict", payload)
    return {
        (name[len("agent."):] if name.startswith("agent.") else name): value
        for name, value in raw.items() if torch.is_tensor(value)
    }


def scalar(events: EventAccumulator, tag: str) -> list[float]:
    values = [float(item.value) for item in events.Scalars(tag)]
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"Stage37 missing/non-finite scalar: {tag}")
    return values


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

    if sha256(args.base) not in STAGE37_ALLOWED_INITIAL_SHA256:
        raise RuntimeError("Stage37 initializer is not public 88.1")
    if sha256(args.base) != STAGE30_PUBLIC_SHA256:
        raise RuntimeError("Stage37 public checkpoint drifted")
    if sha256(args.selector) != STAGE30_SELECTOR_SHA256:
        raise RuntimeError("Stage37 Stage25 selector drifted")
    if sha256(args.calibration) != STAGE30_CALIBRATION_SHA256:
        raise RuntimeError("Stage37 Stage25 calibration drifted")
    if sha256(args.plan) != STAGE37_PLAN_SHA256:
        raise RuntimeError("Stage37 plan SHA drifted")
    freeze = json.loads(args.freeze.read_text())
    if not (
        freeze.get("passed") is True
        and freeze.get("stage") == 37
        and freeze.get("phase") == args.phase
        and freeze.get("holdout_fold") == args.holdout
        and freeze.get("objective_revision") == STAGE37_OBJECTIVE_REVISION
    ):
        raise RuntimeError("Stage37 checkpoint freeze semantics drifted")
    cv = json.loads(args.cv_freeze.read_text())
    entry = cv["folds"][args.holdout]
    if cv.get("stage") != 30 or entry["holdout_fold"] != args.holdout:
        raise RuntimeError("Stage37 CV freeze semantics drifted")

    public = state(args.base)
    selector = state(args.selector)
    audited = []
    for record in freeze["checkpoints"]:
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise RuntimeError("Stage37 checkpoint SHA drifted")
        current = state(path)
        changed = []
        forbidden = []
        by_layer = {0: 0, 1: 0}
        for name, before in public.items():
            if name not in current:
                continue
            after = current[name]
            if not torch.isfinite(after).all():
                raise RuntimeError(f"Stage37 non-finite tensor: {name}")
            if torch.equal(before.cpu(), after.cpu()):
                continue
            if name.startswith(DECODER) and CLASSIFICATION not in name:
                changed.append(name)
                for layer in by_layer:
                    if f".layers.{layer}." in name:
                        by_layer[layer] += 1
            elif not name.startswith((
                MODEL + "_trajectory_head.ref_policy.",
                MODEL + "_trajectory_head.old_policy.",
            )):
                forbidden.append(name)
        if forbidden or len(changed) != 64 or by_layer != {0: 32, 1: 32}:
            raise RuntimeError(
                f"Stage37 decoder boundary drifted changed={len(changed)} "
                f"by_layer={by_layer} forbidden={forbidden[:2]}"
            )
        for name, before in public.items():
            if not name.startswith(DECODER):
                continue
            ref_name = REFERENCE + name[len(DECODER):]
            if ref_name not in current or not torch.equal(
                before.cpu(), current[ref_name].cpu()
            ):
                raise RuntimeError(f"Stage37 frozen reference drifted: {ref_name}")
        selector_names = [
            name for name in selector
            if name.startswith(SELECTOR_PREFIXES) or name in SELECTOR_BUFFERS
        ]
        if not selector_names:
            raise RuntimeError("Stage37 selector state is empty")
        for name in selector_names:
            if name not in current or not torch.equal(
                selector[name].cpu(), current[name].cpu()
            ):
                raise RuntimeError(f"Stage37 frozen selector drifted: {name}")
        payload = torch.load(path, map_location="cpu")
        optimizers = payload.get("optimizer_states", [])
        if len(optimizers) != 1 or len(optimizers[0].get("state", {})) != 64:
            raise RuntimeError("Stage37 optimizer boundary drifted")
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
            raise RuntimeError(f"Stage37 optimizer groups drifted: {groups}")
        audited.append({**record, "changed": len(changed), "by_layer": by_layer})

    events = EventAccumulator(str(freeze["tensorboard_event"]))
    events.Reload()
    tags = {
        "microbatches": "train/stage37_projection_microbatch_count_step",
        "dot": "train/stage37_projection_dot_step",
        "cosine": "train/stage37_projection_cosine_step",
        "conflict": "train/stage37_projection_conflict_step",
        "preservation_norm": "train/stage37_projection_preservation_grad_norm_step",
        "projection_norm": "train/stage37_projection_projection_norm_step",
        "recovery": "train/stage37_projection_recovery_active_step",
        "frontier_delta": "train/stage37_projection_frontier_delta_step",
        "finite": "train/stage37_projection_finite_step",
        "mature_zero": "train/stage37_mature_positive_exploration_zero_step",
        "frontier_width": "train/stage37_frontier_exact_width_step",
        "tail_width": "train/stage37_tail_exact_width_step",
    }
    series = {name: scalar(events, tag) for name, tag in tags.items()}
    if not all(value == 8.0 for value in series["microbatches"]):
        raise RuntimeError("Stage37 projection did not accumulate eight microbatches")
    if not all(value > 0 for value in series["preservation_norm"]):
        raise RuntimeError("Stage37 preservation gradient is zero")
    if not all(value == 1.0 for key in (
        "finite", "mature_zero", "frontier_width", "tail_width"
    ) for value in series[key]):
        raise RuntimeError("Stage37 projection/trace invariant failed")
    result = {
        "schema_version": 1,
        "stage": 37,
        "phase": args.phase,
        "holdout_fold": args.holdout,
        "passed": True,
        "objective_revision": STAGE37_OBJECTIVE_REVISION,
        "plan_sha256": STAGE37_PLAN_SHA256,
        "public_checkpoint_sha256": sha256(args.base),
        "selector_checkpoint_sha256": sha256(args.selector),
        "selector_calibration_sha256": sha256(args.calibration),
        "bistate_current_and_public_frontier": True,
        "gradient_projection_after_accumulation": True,
        "mature_positive_exploration_zero": True,
        "exact_64_tensor_boundary": True,
        "checkpoints": audited,
        "projection_summary": {
            name: {
                "count": len(values),
                "mean": sum(values) / len(values),
                "minimum": min(values),
                "maximum": max(values),
            }
            for name, values in series.items()
        },
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
