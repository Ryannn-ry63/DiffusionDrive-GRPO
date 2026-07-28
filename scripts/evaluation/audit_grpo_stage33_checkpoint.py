#!/usr/bin/env python3
"""Fail-closed Stage33 CDC checkpoint audit.

This audit checks the optimizer boundary and frozen reference/selector state;
it does not claim candidate-level CDC credit until that trace is present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


PLAN_SHA256 = "0ace8d8959360a15cfac8f45bc2c6e19898c09ad1a04aa7b5e5dcc2fb36283ff"
BASE_SHA256 = "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
MODEL_PREFIX = "_transfuser_model."
DECODER_PREFIX = MODEL_PREFIX + "_trajectory_head.diff_decoder."
LAYERS_PREFIX = DECODER_PREFIX + "layers."
REFERENCE_PREFIX = MODEL_PREFIX + "_trajectory_head.ref_policy."
CLASSIFICATION = ".task_decoder.plan_cls_branch."
SELECTOR_PREFIXES = (
    MODEL_PREFIX + "_trajectory_head.stage24_selector.",
    MODEL_PREFIX + "_trajectory_head.stage25_selector.",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensors(payload: dict) -> dict[str, torch.Tensor]:
    raw = payload.get("state_dict", payload)
    return {
        (name[6:] if name.startswith("agent.") else name): value
        for name, value in raw.items()
        if torch.is_tensor(value)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("audit", "formal"), default="formal")
    args = parser.parse_args()
    if sha256(args.plan) != PLAN_SHA256:
        raise RuntimeError("Stage33 plan SHA drifted")
    if sha256(args.base) != BASE_SHA256:
        raise RuntimeError("Stage33 public-88.1 base SHA drifted")
    payload = torch.load(args.checkpoint, map_location="cpu")
    current, base = tensors(payload), tensors(torch.load(args.base, map_location="cpu"))
    changed, forbidden, by_layer = [], [], {0: 0, 1: 0}
    for name, before in base.items():
        if name not in current or not torch.isfinite(current[name]).all():
            raise RuntimeError(f"missing/nonfinite tensor: {name}")
        if torch.equal(before.cpu(), current[name].cpu()):
            continue
        if name.startswith(LAYERS_PREFIX) and CLASSIFICATION not in name:
            changed.append(name)
            for layer in by_layer:
                if f".layers.{layer}." in name:
                    by_layer[layer] += 1
        else:
            forbidden.append(name)
    if forbidden or len(changed) != 64 or by_layer != {0: 32, 1: 32}:
        raise RuntimeError(
            f"Stage33 decoder boundary drifted: changed={len(changed)} "
            f"by_layer={by_layer} forbidden={forbidden[:2]}"
        )
    for name, before in base.items():
        if name.startswith(DECODER_PREFIX):
            ref_name = REFERENCE_PREFIX + name[len(DECODER_PREFIX):]
            if ref_name not in current or not torch.equal(before.cpu(), current[ref_name].cpu()):
                raise RuntimeError(f"reference policy drifted: {ref_name}")
    selector_names = [name for name in base if name.startswith(SELECTOR_PREFIXES)]
    for name in selector_names:
        if name not in current or not torch.equal(base[name].cpu(), current[name].cpu()):
            raise RuntimeError(f"selector drifted: {name}")
    optimizers = payload.get("optimizer_states", [])
    if optimizers and len(optimizers[0].get("state", {})) != 64:
        raise RuntimeError("Stage33 optimizer state boundary drifted")
    result = {
        "passed": True,
        "stage": 33,
        "phase": args.phase,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "plan_sha256": PLAN_SHA256,
        "changed_decoder_tensor_count": len(changed),
        "changed_by_layer": by_layer,
        "candidate_level_trace": bool(payload.get("stage33_candidate_level_trace", False)),
        "cdc_mode": "selected_common_noise_smoke",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

