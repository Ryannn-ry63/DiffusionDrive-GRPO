#!/usr/bin/env python3
"""Audit Stage27 selector-only checkpoints against the public generator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


PUBLIC_SHA256 = (
    "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
)
HEAD_PREFIX = "_transfuser_model._trajectory_head."
STAGE24_PREFIX = HEAD_PREFIX + "stage24_selector."
STAGE25_PREFIX = HEAD_PREFIX + "stage25_selector."


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> tuple[dict, dict]:
    payload = torch.load(path, map_location="cpu")
    if "state_dict" not in payload:
        raise KeyError(f"checkpoint lacks state_dict: {path}")
    state = {
        (key[len("agent."):] if key.startswith("agent.") else key): value
        for key, value in payload["state_dict"].items()
    }
    return payload, state


def scalar(state: dict, name: str) -> int:
    key = HEAD_PREFIX + name
    if key not in state:
        raise RuntimeError(f"checkpoint lacks selector provenance: {key}")
    return int(state[key].item())


def assert_finite_module(state: dict, prefix: str) -> int:
    values = [value for key, value in state.items() if key.startswith(prefix)]
    if not values:
        raise RuntimeError(f"checkpoint contains no module tensors: {prefix}")
    if not all(torch.isfinite(value).all() for value in values):
        raise RuntimeError(f"checkpoint has non-finite module tensor: {prefix}")
    return len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("stage24", "stage25"), required=True)
    parser.add_argument("--branch", choices=("public", "multi"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--public-checkpoint", type=Path, required=True)
    parser.add_argument("--stage24-parent", type=Path)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--expected-global-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if file_sha256(args.public_checkpoint) != PUBLIC_SHA256:
        raise RuntimeError("Stage27 public checkpoint SHA mismatch")
    payload, state = load_checkpoint(args.checkpoint)
    _, public_state = load_checkpoint(args.public_checkpoint)
    if int(payload.get("epoch", -1)) != args.expected_epoch:
        raise RuntimeError(
            f"selector epoch mismatch: {payload.get('epoch')} "
            f"!= {args.expected_epoch}"
        )
    if int(payload.get("global_step", -1)) != args.expected_global_step:
        raise RuntimeError(
            f"selector global_step mismatch: {payload.get('global_step')} "
            f"!= {args.expected_global_step}"
        )

    missing_public = sorted(set(public_state) - set(state))
    changed_public = []
    for key in sorted(set(public_state) & set(state)):
        if not torch.equal(public_state[key], state[key]):
            changed_public.append(key)
    if missing_public or changed_public:
        raise RuntimeError(
            "Stage27 selector training changed/lost public generator state: "
            f"missing={missing_public[:5]} changed={changed_public[:5]}"
        )

    stage24_count = assert_finite_module(state, STAGE24_PREFIX)
    stage25_count = assert_finite_module(state, STAGE25_PREFIX)
    stage24_updates = scalar(state, "_stage24_selector_training_updates")
    stage25_updates = scalar(state, "_stage25_selector_training_updates")
    embedding_count = scalar(state, "_stage24_embedding_count")
    if embedding_count <= 0:
        raise RuntimeError("Stage27 Stage24 embedding moments are empty")

    parent_sha = None
    stage24_parent_equal = None
    if args.phase == "stage24":
        if args.stage24_parent is not None:
            raise ValueError("stage24 audit does not accept --stage24-parent")
        if (
            stage24_updates != args.expected_global_step
            or stage25_updates != 0
        ):
            raise RuntimeError("Stage27 Stage24 update provenance mismatch")
    else:
        if args.stage24_parent is None:
            raise ValueError("stage25 audit requires --stage24-parent")
        _, parent_state = load_checkpoint(args.stage24_parent)
        parent_sha = file_sha256(args.stage24_parent)
        preserved_prefixes = (
            STAGE24_PREFIX,
            HEAD_PREFIX + "_stage24_selector_training_updates",
            HEAD_PREFIX + "_stage24_embedding_count",
            HEAD_PREFIX + "_stage24_embedding_sum",
            HEAD_PREFIX + "_stage24_embedding_sum_sq",
        )
        preserved_keys = [
            key for key in parent_state
            if key.startswith(preserved_prefixes)
        ]
        if not preserved_keys or any(
            key not in state or not torch.equal(parent_state[key], state[key])
            for key in preserved_keys
        ):
            raise RuntimeError(
                "Stage27 Stage25 training changed the frozen Stage24 selector"
            )
        stage24_parent_equal = True
        parent_updates = scalar(
            parent_state, "_stage24_selector_training_updates"
        )
        if (
            stage24_updates != parent_updates
            or stage25_updates != args.expected_global_step
        ):
            raise RuntimeError("Stage27 Stage25 update provenance mismatch")

    result = {
        "schema_version": 1,
        "stage": 27,
        "phase": args.phase,
        "branch": args.branch,
        "passed": True,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "public_checkpoint": str(args.public_checkpoint.resolve()),
        "public_checkpoint_sha256": PUBLIC_SHA256,
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "public_tensor_count": len(public_state),
        "public_tensors_bitwise_unchanged": len(public_state),
        "stage24_tensor_count": stage24_count,
        "stage25_tensor_count": stage25_count,
        "stage24_training_updates": stage24_updates,
        "stage25_training_updates": stage25_updates,
        "stage24_embedding_count": embedding_count,
        "stage24_parent": (
            str(args.stage24_parent.resolve())
            if args.stage24_parent is not None else None
        ),
        "stage24_parent_sha256": parent_sha,
        "stage24_parent_bitwise_preserved": stage24_parent_equal,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
