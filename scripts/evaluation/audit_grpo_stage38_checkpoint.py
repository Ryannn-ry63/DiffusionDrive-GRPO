#!/usr/bin/env python3
"""Fail-closed provenance and gradient audit for Stage38 ESCR."""

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
from navsim.agents.diffusiondrive.stage38_contract import (
    STAGE38_ALLOWED_INITIAL_SHA256,
    STAGE38_OBJECTIVE_REVISION,
    STAGE38_PLAN_SHA256,
)


MODEL = "_transfuser_model."
DECODER = MODEL + "_trajectory_head.diff_decoder."
REFERENCE = MODEL + "_trajectory_head.ref_policy."
OLD_POLICY = MODEL + "_trajectory_head.old_policy."
OLD_POLICY_SYNC_BUFFER = MODEL + "_trajectory_head._old_policy_last_sync_step"
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
        raise RuntimeError(f"Stage38 missing/non-finite scalar: {tag}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("audit", "formal"), required=True)
    parser.add_argument("--holdout", type=int, choices=(0, 1), required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--cv-freeze", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    public_sha = sha256(args.base)
    initializer_sha = sha256(args.initializer)
    if public_sha != STAGE30_PUBLIC_SHA256:
        raise RuntimeError("Stage38 public checkpoint drifted")
    if initializer_sha not in STAGE38_ALLOWED_INITIAL_SHA256:
        raise RuntimeError("Stage38 initializer is not a fold-matched Stage36 RGT192 checkpoint")
    if sha256(args.selector) != STAGE30_SELECTOR_SHA256:
        raise RuntimeError("Stage38 Stage25 selector drifted")
    if sha256(args.calibration) != STAGE30_CALIBRATION_SHA256:
        raise RuntimeError("Stage38 Stage25 calibration drifted")
    if sha256(args.plan) != STAGE38_PLAN_SHA256:
        raise RuntimeError("Stage38 plan SHA drifted")

    freeze = json.loads(args.freeze.read_text())
    expected_steps = [1] if args.phase == "audit" else [24, 48, 96]
    if not (
        freeze.get("passed") is True
        and freeze.get("stage") == 38
        and freeze.get("phase") == args.phase
        and freeze.get("holdout_fold") == args.holdout
        and freeze.get("objective_revision") == STAGE38_OBJECTIVE_REVISION
        and freeze.get("plan_sha256") == STAGE38_PLAN_SHA256
        and freeze.get("expected_global_steps") == expected_steps
    ):
        raise RuntimeError("Stage38 checkpoint freeze semantics drifted")
    cv = json.loads(args.cv_freeze.read_text())
    entry = cv["folds"][args.holdout]
    if cv.get("stage") != 30 or entry["holdout_fold"] != args.holdout:
        raise RuntimeError("Stage38 CV freeze semantics drifted")

    public = state(args.base)
    initializer = state(args.initializer)
    selector = state(args.selector)
    audited = []
    for record in freeze["checkpoints"]:
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise RuntimeError("Stage38 checkpoint SHA drifted")
        current = state(path)
        # The initializer may contain the previous PPO behavior snapshot.  It
        # is deliberately excluded from the trainable-boundary comparison:
        # the generic Lightning hook synchronizes old_policy at optimizer
        # step 1 and then every configured sync interval.  The snapshot must
        # remain complete and finite, but it is expected to change at those
        # synchronization points.
        initializer_core = {
            name: value
            for name, value in initializer.items()
            if not name.startswith((REFERENCE, OLD_POLICY))
            and name != OLD_POLICY_SYNC_BUFFER
        }
        if not set(initializer_core).issubset(current):
            raise RuntimeError("Stage38 checkpoint lost an initializer tensor")
        changed = []
        forbidden = []
        by_layer = {0: 0, 1: 0}
        for name, before in initializer_core.items():
            if name not in current:
                continue
            after = current[name]
            if not torch.isfinite(after).all():
                raise RuntimeError(f"Stage38 non-finite tensor: {name}")
            if torch.equal(before.cpu(), after.cpu()):
                continue
            if name.startswith(DECODER) and CLASSIFICATION not in name:
                changed.append(name)
                for layer in by_layer:
                    if f".layers.{layer}." in name:
                        by_layer[layer] += 1
            else:
                forbidden.append(name)
        if forbidden or not changed or any(value == 0 for value in by_layer.values()):
            raise RuntimeError(
                f"Stage38 decoder boundary drifted changed={len(changed)} "
                f"by_layer={by_layer} forbidden={forbidden[:2]}"
            )
        decoder_names = [name for name in public if name.startswith(DECODER)]
        old_names = [name for name in current if name.startswith(OLD_POLICY)]
        if len(old_names) != len(decoder_names):
            raise RuntimeError("Stage38 old-policy snapshot is incomplete")
        if not all(torch.isfinite(current[name]).all() for name in old_names):
            raise RuntimeError("Stage38 old-policy snapshot is non-finite")
        for name, before in public.items():
            if not name.startswith(DECODER):
                continue
            ref_name = REFERENCE + name[len(DECODER):]
            if ref_name not in current or not torch.equal(
                before.cpu(), current[ref_name].cpu()
            ):
                raise RuntimeError(f"Stage38 frozen reference drifted: {ref_name}")
        selector_names = [
            name for name in selector
            if name.startswith(SELECTOR_PREFIXES) or name in SELECTOR_BUFFERS
        ]
        if not selector_names:
            raise RuntimeError("Stage38 selector state is empty")
        for name in selector_names:
            if name not in current or not torch.equal(
                selector[name].cpu(), current[name].cpu()
            ):
                raise RuntimeError(f"Stage38 frozen selector drifted: {name}")
        payload = torch.load(path, map_location="cpu")
        optimizers = payload.get("optimizer_states", [])
        if len(optimizers) != 1:
            raise RuntimeError("Stage38 optimizer state count drifted")
        groups = sorted(
            (
                round(float(group.get("lr_scale", -1)), 6),
                round(float(group["lr"]), 12),
                len(group["params"]),
                float(group["weight_decay"]),
            )
            for group in optimizers[0].get("param_groups", [])
        )
        if groups != [(0.1, 1e-7, 32, 0.0), (1.0, 1e-6, 32, 0.0)]:
            raise RuntimeError(f"Stage38 optimizer groups drifted: {groups}")
        audited.append({**record, "changed": len(changed), "by_layer": by_layer})

    events = EventAccumulator(str(freeze["tensorboard_event"]))
    events.Reload()
    tags = {
        "elite_width": "train/stage38_public_elite_exact_width_step",
        "active_bounded": "train/stage38_active_pool_bounded_step",
        "active_fraction": "train/stage38_active_fraction_step",
        "positive_fraction": "train/stage38_positive_fraction_step",
        "positive_challenger": (
            "train/stage38_positive_challenger_fraction_step"
        ),
        "negative_elite": "train/stage38_negative_elite_fraction_step",
        "delta": "train/stage38_delta_set_mean_step",
        "oracle": "train/stage38_union_safe_oracle_gain_mean_step",
        "kl": "train/stage38_mean_exact_kl_step",
    }
    series = {name: scalar(events, tag) for name, tag in tags.items()}
    if not all(
        value == 1.0
        for key in ("elite_width", "active_bounded")
        for value in series[key]
    ):
        raise RuntimeError("Stage38 ESCR replay invariant failed")
    if not all(value > 0 for value in series["active_fraction"]):
        raise RuntimeError("Stage38 active replay pool is empty")
    if max(series["positive_challenger"]) <= 0:
        raise RuntimeError("Stage38 produced no positive challenger credit")
    if max(series["negative_elite"]) <= 0:
        raise RuntimeError("Stage38 produced no negative elite-repair credit")
    result = {
        "schema_version": 1,
        "stage": 38,
        "phase": args.phase,
        "holdout_fold": args.holdout,
        "passed": True,
        "objective_revision": STAGE38_OBJECTIVE_REVISION,
        "plan_sha256": STAGE38_PLAN_SHA256,
        "public_checkpoint_sha256": public_sha,
        "initializer_checkpoint_sha256": initializer_sha,
        "selector_checkpoint_sha256": sha256(args.selector),
        "selector_calibration_sha256": sha256(args.calibration),
        "exact_public_elite_width": True,
        "bounded_active_replay": True,
        "decoder_only_gradient_boundary": True,
        "periodic_old_policy_snapshot": True,
        "checkpoints": audited,
        "signal_summary": {
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
