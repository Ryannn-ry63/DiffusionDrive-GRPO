#!/usr/bin/env python3
"""Fail-closed provenance, boundary and signal audit for Stage39."""

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
    STAGE30_SELECTOR_SHA256,
)
from navsim.agents.diffusiondrive.stage39_challenger import (
    STAGE39_BRANCHES,
    STAGE39_OBJECTIVE_REVISION,
    STAGE39_PLAN_SHA256,
)
from navsim.agents.diffusiondrive.stage39_contract import STAGE39_PUBLIC_SHA256


MODEL = "_transfuser_model."
PUBLIC = MODEL + "_trajectory_head.diff_decoder."
CHALLENGER = MODEL + "_trajectory_head.stage39_challenger_decoder."
REFERENCE = MODEL + "_trajectory_head.ref_policy."
OLD_POLICY = MODEL + "_trajectory_head.old_policy."
CLASSIFICATION = ".task_decoder.plan_cls_branch."
SELECTOR_PREFIXES = (
    MODEL + "_trajectory_head.stage24_selector.",
    MODEL + "_trajectory_head.stage25_selector.",
    MODEL + "_trajectory_head._stage24_",
    MODEL + "_trajectory_head._stage25_",
)
BRANCH_ID = {"BC": 0.0, "STD": 1.0, "SET": 2.0}


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
        raise RuntimeError(f"Stage39 missing/non-finite scalar: {tag}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("audit", "pilot", "formal"), required=True)
    parser.add_argument("--branch", choices=STAGE39_BRANCHES, required=True)
    parser.add_argument("--holdout", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--cv-freeze", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.base) != STAGE39_PUBLIC_SHA256:
        raise RuntimeError("Stage39 public checkpoint drifted")
    if sha256(args.selector) != STAGE30_SELECTOR_SHA256:
        raise RuntimeError("Stage39 Stage25 selector drifted")
    if sha256(args.calibration) != STAGE30_CALIBRATION_SHA256:
        raise RuntimeError("Stage39 Stage25 calibration drifted")
    if sha256(args.plan) != STAGE39_PLAN_SHA256:
        raise RuntimeError("Stage39 plan SHA drifted")

    freeze = json.loads(args.freeze.read_text())
    if args.phase == "audit":
        expected_steps = [1]
    elif args.phase == "pilot":
        expected_steps = [48, 96, 192]
    else:
        selected_step = freeze.get("selected_step")
        if selected_step not in (48, 96, 192):
            raise RuntimeError("Stage39 formal freeze lacks the pilot-selected step")
        expected_steps = [selected_step]
    if not (
        freeze.get("passed") is True
        and freeze.get("stage") == 39
        and freeze.get("phase") == args.phase
        and freeze.get("branch") == args.branch
        and freeze.get("holdout_fold") == args.holdout
        and freeze.get("objective_revision") == STAGE39_OBJECTIVE_REVISION
        and freeze.get("plan_sha256") == STAGE39_PLAN_SHA256
        and freeze.get("expected_global_steps") == expected_steps
    ):
        raise RuntimeError("Stage39 checkpoint freeze semantics drifted")
    cv = json.loads(args.cv_freeze.read_text())
    entry = cv["folds"][args.holdout]
    if cv.get("stage") != 30 or entry["holdout_fold"] != args.holdout:
        raise RuntimeError("Stage39 CV freeze semantics drifted")

    public = state(args.base)
    selector = state(args.selector)
    public_decoder = {
        name[len(PUBLIC):]: value for name, value in public.items()
        if name.startswith(PUBLIC)
    }
    if not public_decoder:
        raise RuntimeError("Stage39 public decoder state is empty")
    audited = []
    for record in freeze["checkpoints"]:
        checkpoint_path = Path(record["path"])
        if sha256(checkpoint_path) != record["sha256"]:
            raise RuntimeError("Stage39 checkpoint SHA drifted")
        current = state(checkpoint_path)
        # Public fallback must remain bitwise equal to the official 88.1 decoder.
        for suffix, before in public_decoder.items():
            public_name = PUBLIC + suffix
            challenger_name = CHALLENGER + suffix
            reference_name = REFERENCE + suffix
            if public_name not in current or not torch.equal(before, current[public_name].cpu()):
                raise RuntimeError(f"Stage39 public fallback drifted: {public_name}")
            if reference_name not in current or not torch.equal(before, current[reference_name].cpu()):
                raise RuntimeError(f"Stage39 frozen reference drifted: {reference_name}")
            if challenger_name not in current or not torch.isfinite(current[challenger_name]).all():
                raise RuntimeError(f"Stage39 challenger state invalid: {challenger_name}")
            if CLASSIFICATION in challenger_name and not torch.equal(
                before, current[challenger_name].cpu()
            ):
                raise RuntimeError(f"Stage39 challenger classification drifted: {challenger_name}")

        changed = []
        by_layer = {0: 0, 1: 0}
        for suffix, before in public_decoder.items():
            name = CHALLENGER + suffix
            if CLASSIFICATION in name or torch.equal(before, current[name].cpu()):
                continue
            changed.append(name)
            for layer in by_layer:
                if f".layers.{layer}." in name:
                    by_layer[layer] += 1
        if not changed or any(count == 0 for count in by_layer.values()):
            raise RuntimeError(
                f"Stage39 challenger boundary has no two-layer update: {by_layer}"
            )

        # Everything inherited from the public checkpoint is frozen except the
        # separately loaded Stage25 selector state and policy snapshots.
        for name, before in public.items():
            if name.startswith((PUBLIC, *SELECTOR_PREFIXES)):
                continue
            if name not in current:
                raise RuntimeError(f"Stage39 checkpoint lost public tensor: {name}")
            if name.startswith((REFERENCE, OLD_POLICY)):
                continue
            if not torch.equal(before.cpu(), current[name].cpu()):
                raise RuntimeError(f"Stage39 forbidden base tensor changed: {name}")
        selector_names = [
            name for name in selector if name.startswith(SELECTOR_PREFIXES)
        ]
        if not selector_names:
            raise RuntimeError("Stage39 selector state is empty")
        for name in selector_names:
            if name not in current or not torch.equal(selector[name].cpu(), current[name].cpu()):
                raise RuntimeError(f"Stage39 frozen selector drifted: {name}")

        payload = torch.load(checkpoint_path, map_location="cpu")
        optimizers = payload.get("optimizer_states", [])
        if len(optimizers) != 1:
            raise RuntimeError("Stage39 optimizer state count drifted")
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
            raise RuntimeError(f"Stage39 optimizer groups drifted: {groups}")
        audited.append({**record, "changed": len(changed), "by_layer": by_layer})

    events = EventAccumulator(str(freeze["tensorboard_event"]))
    events.Reload()
    tags = {
        "sampled": "train/stage39_sampled_chain_count_step",
        "replayed": "train/stage39_replayed_chain_count_step",
        "independent": "train/stage39_independent_initial_noise_step",
        "noise_cosine": "train/stage39_initial_noise_abs_cosine_step",
        "branch_id": "train/stage39_branch_id_step",
        "policy": "train/stage39_policy_loss_step",
        "bc": "train/stage39_bc_loss_step",
        "kl": "train/stage39_kl_loss_step",
        "oracle": "train/stage39_union_safe_oracle_gain_mean_step",
        "set_positive": "train/stage39_set_positive_fraction_step",
        "std_positive": "train/stage39_standard_positive_fraction_step",
        "safety_veto": "train/stage39_safety_veto_fraction_step",
        "regression_grad": "train/regression_grad_norm_step",
        "classification_grad": "train/classification_grad_norm_step",
    }
    series = {name: scalar(events, tag) for name, tag in tags.items()}
    if any(value != 320.0 for key in ("sampled", "replayed") for value in series[key]):
        raise RuntimeError("Stage39 sampled/replayed chain accounting drifted")
    if any(value != 1.0 for value in series["independent"]):
        raise RuntimeError("Stage39 public/challenger initial noise is not independent")
    if max(series["noise_cosine"]) >= 0.1:
        raise RuntimeError("Stage39 independent-noise cosine is implausibly high")
    if any(value != BRANCH_ID[args.branch] for value in series["branch_id"]):
        raise RuntimeError("Stage39 branch provenance drifted")
    if max(series["regression_grad"]) <= 0 or max(series["classification_grad"]) != 0:
        raise RuntimeError("Stage39 challenger gradient boundary failed")
    if args.branch == "BC":
        if max(abs(value) for value in series["policy"] + series["kl"]) != 0:
            raise RuntimeError("Stage39 BC branch leaked GRPO/KL loss")
    elif (
        args.phase in {"pilot", "formal"}
        and max(
            series["std_positive"]
            if args.branch == "STD"
            else series["set_positive"]
        ) <= 0
    ):
        raise RuntimeError(f"Stage39 {args.branch} pilot/formal branch produced no positive credit")

    result = {
        "schema_version": 1,
        "stage": 39,
        "phase": args.phase,
        "branch": args.branch,
        "holdout_fold": args.holdout,
        "passed": True,
        "objective_revision": STAGE39_OBJECTIVE_REVISION,
        "plan_sha256": STAGE39_PLAN_SHA256,
        "public_checkpoint_sha256": sha256(args.base),
        "selector_checkpoint_sha256": sha256(args.selector),
        "selector_calibration_sha256": sha256(args.calibration),
        "public_fallback_bitwise_frozen": True,
        "independent_challenger_decoder": True,
        "decoder_only_gradient_boundary": True,
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
