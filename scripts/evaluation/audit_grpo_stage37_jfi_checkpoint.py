#!/usr/bin/env python3
"""Audit the JFI-only trainable boundary and freeze its exact checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


PUBLIC_SHA256 = "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
PLAN_SHA256 = "4dc469dd0be4d2579796ff70ae7a09ed9fd8e4c92449cd057e42463d268bb5d8"
HEAD = "_transfuser_model._trajectory_head."
JFI_PREFIX = HEAD + "stage37_jfi_selector."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu")
    state = {
        (key[6:] if key.startswith("agent.") else key): value
        for key, value in payload["state_dict"].items()
    }
    return payload, state


def scalar(state: dict[str, torch.Tensor], name: str) -> int:
    key = HEAD + name
    if key not in state:
        raise RuntimeError(f"checkpoint lacks {key}")
    return int(state[key].item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("audit", "formal"), required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--parent-selector", type=Path, required=True)
    parser.add_argument("--public-checkpoint", type=Path, required=True)
    parser.add_argument("--bank-freeze", type=Path, required=True)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--expected-global-step", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.public_checkpoint) != PUBLIC_SHA256:
        raise RuntimeError("public checkpoint SHA drifted")
    bank_freeze = json.loads(args.bank_freeze.read_text())
    if (
        not bank_freeze.get("passed")
        or bank_freeze.get("plan_sha256") != PLAN_SHA256
        or int(bank_freeze.get("num_banks", 0)) != 8
    ):
        raise RuntimeError("JFI bank freeze drifted")

    candidates = []
    for path in args.experiment_root.glob(
        "*/lightning_logs/version_*/checkpoints/*.ckpt"
    ):
        payload = torch.load(path, map_location="cpu")
        if (
            int(payload.get("epoch", -1)) == args.expected_epoch
            and int(payload.get("global_step", -1)) == args.expected_global_step
        ):
            candidates.append(path)
    if not candidates:
        raise RuntimeError("no JFI checkpoint has the expected epoch/global step")
    step_name = f"grpo-step-{args.expected_global_step}.ckpt"
    preferred = [path for path in candidates if path.name == step_name]
    if len(preferred) != 1:
        preferred = [path for path in candidates if path.name.startswith("grpo-")]
    if len(preferred) != 1:
        raise RuntimeError(
            f"expected one canonical JFI checkpoint, found {len(preferred)}"
        )
    checkpoint = preferred[0]
    checkpoint_sha = sha256(checkpoint)
    aliases = candidates
    payload, state = load(checkpoint)
    _, parent = load(args.parent_selector)
    _, public = load(args.public_checkpoint)

    changed_parent = [
        key for key, value in parent.items()
        if key not in state or not torch.equal(value, state[key])
    ]
    if changed_parent:
        raise RuntimeError(
            f"JFI training changed frozen parent state: {changed_parent[:5]}"
        )
    changed_public = [
        key for key, value in public.items()
        if key not in state or not torch.equal(value, state[key])
    ]
    if changed_public:
        raise RuntimeError(
            f"JFI training changed public model state: {changed_public[:5]}"
        )
    jfi = {key: value for key, value in state.items() if key.startswith(JFI_PREFIX)}
    if len(jfi) != 48 or not all(torch.isfinite(value).all() for value in jfi.values()):
        raise RuntimeError(f"JFI must contain 48 finite trainable tensors, got {len(jfi)}")
    for member in range(1, 8):
        left = jfi[f"{JFI_PREFIX}members.0.0.weight"]
        right = jfi[f"{JFI_PREFIX}members.{member}.0.weight"]
        if torch.equal(left, right):
            raise RuntimeError("JFI ensemble heads are not independent")
    if scalar(state, "_stage37_jfi_training_updates") != args.expected_global_step:
        raise RuntimeError("JFI update provenance drifted")
    if scalar(state, "_stage24_selector_training_updates") != scalar(
        parent, "_stage24_selector_training_updates"
    ):
        raise RuntimeError("frozen Stage24 update count changed")
    optimizers = payload.get("optimizer_states", [])
    if len(optimizers) != 1:
        raise RuntimeError("JFI requires exactly one optimizer")
    groups = optimizers[0].get("param_groups", [])
    optimizer_param_count = sum(len(group.get("params", [])) for group in groups)
    if optimizer_param_count != 48:
        raise RuntimeError("JFI optimizer boundary is not exactly 48 tensors")
    optimizer_state = optimizers[0].get("state", {})
    if len(optimizer_state) != 48:
        raise RuntimeError("JFI optimizer did not update all 48 tensors")
    for entry in optimizer_state.values():
        for value in entry.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise RuntimeError("JFI optimizer state is non-finite")

    result = {
        "schema_version": 1,
        "stage": 37,
        "phase": args.phase,
        "passed": True,
        "plan_sha256": PLAN_SHA256,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_aliases": [str(path.resolve()) for path in aliases],
        "parent_selector": str(args.parent_selector.resolve()),
        "parent_selector_sha256": sha256(args.parent_selector),
        "public_checkpoint_sha256": PUBLIC_SHA256,
        "bank_freeze": str(args.bank_freeze.resolve()),
        "bank_freeze_sha256": sha256(args.bank_freeze),
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "jfi_tensor_count": len(jfi),
        "optimizer_tensor_count": len(optimizer_state),
        "frozen_parent_bitwise_equal": True,
        "frozen_public_bitwise_equal": True,
        "independent_eight_heads": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    freeze = {
        "schema_version": 1,
        "stage": 37,
        "phase": args.phase,
        "passed": True,
        "plan_sha256": PLAN_SHA256,
        "checkpoints": [{
            "path": str(checkpoint.resolve()),
            "sha256": checkpoint_sha,
            "epoch": int(payload["epoch"]),
            "global_step": int(payload["global_step"]),
        }],
    }
    (args.output_dir / "checkpoints.json").write_text(
        json.dumps(freeze, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
