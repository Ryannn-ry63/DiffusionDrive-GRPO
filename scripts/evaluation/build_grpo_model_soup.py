#!/usr/bin/env python3
"""Average current DiffusionDrive decoder weights across GRPO checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


BASE_CHECKPOINT = Path(
    "/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/"
    "training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/"
    "version_0/checkpoints/eval_model"
)
SOURCE_CHECKPOINTS = (
    Path(
        "/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/"
        "grpo_d_seed0_skl0.0_gkl0.1_eps0.001_dense0.1_mweightuniform_bs2/"
        "2026.07.16.14.53.31/lightning_logs/version_0/checkpoints/grpo-00-128.ckpt"
    ),
    Path(
        "/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/"
        "grpo_d_seed1_skl0.0_gkl0.1_eps0.001_dense0.1_mweightuniform_bs2/"
        "2026.07.16.14.53.31/lightning_logs/version_0/checkpoints/grpo-00-128.ckpt"
    ),
    Path(
        "/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/"
        "grpo_d_seed2_skl0.0_gkl0.1_eps0.001_dense0.1_mweightuniform_bs2/"
        "2026.07.16.14.53.31/lightning_logs/version_0/checkpoints/grpo-00-128.ckpt"
    ),
)
DEFAULT_OUTPUT = Path("artifacts/grpo_stage1/uniform_bs2_d128_soup.ckpt")

DECODER_PREFIX = "_transfuser_model._trajectory_head.diff_decoder."
SNAPSHOT_MARKERS = (
    "_transfuser_model._trajectory_head.old_policy.",
    "_transfuser_model._trajectory_head.ref_policy.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, default=BASE_CHECKPOINT)
    parser.add_argument(
        "--source-checkpoints",
        type=Path,
        nargs="+",
        default=list(SOURCE_CHECKPOINTS),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path)
    return parser.parse_args()


def canonical_key(key: str) -> str:
    return key[len("agent.") :] if key.startswith("agent.") else key


def is_current_decoder_key(key: str) -> bool:
    return canonical_key(key).startswith(DECODER_PREFIX)


def is_snapshot_key(key: str) -> bool:
    normalized = canonical_key(key)
    return any(normalized.startswith(marker) for marker in SNAPSHOT_MARKERS)


def canonical_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized_key = canonical_key(key)
        if normalized_key in normalized:
            raise ValueError(f"Duplicate canonical checkpoint key: {normalized_key}")
        normalized[normalized_key] = value
    return normalized


def checkpoint_state_dict(checkpoint: Mapping[str, Any], path: Path) -> Mapping[str, torch.Tensor]:
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"Checkpoint has no state_dict mapping: {path}")
    return state_dict


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_decoder_state_dicts(
    base_state_dict: Mapping[str, torch.Tensor],
    source_state_dicts: Sequence[Mapping[str, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    if len(source_state_dicts) < 2:
        raise ValueError("Model soup requires at least two source checkpoints")

    base = canonical_state_dict(base_state_dict)
    sources = [canonical_state_dict(state_dict) for state_dict in source_state_dicts]
    decoder_keys = {key for key in base if key.startswith(DECODER_PREFIX)}
    if not decoder_keys:
        raise ValueError("Base checkpoint contains no current diff_decoder weights")
    for index, source in enumerate(sources):
        source_decoder_keys = {key for key in source if key.startswith(DECODER_PREFIX)}
        if source_decoder_keys != decoder_keys:
            missing = sorted(decoder_keys - source_decoder_keys)
            extra = sorted(source_decoder_keys - decoder_keys)
            raise ValueError(
                f"Source {index} decoder keys mismatch: missing={missing[:1]}, "
                f"extra={extra[:1]}"
            )

    output = {
        key: value.detach().cpu().clone()
        for key, value in base.items()
        if not is_snapshot_key(key)
    }
    averaged_tensors = 0
    averaged_numel = 0
    for key in sorted(decoder_keys):
        base_tensor = base[key]
        source_tensors = [source[key] for source in sources]
        if any(tensor.shape != base_tensor.shape for tensor in source_tensors):
            raise ValueError(f"Decoder tensor shape mismatch for {key}")
        if base_tensor.is_floating_point():
            accumulator = torch.zeros_like(base_tensor, dtype=torch.float64, device="cpu")
            for tensor in source_tensors:
                accumulator.add_(tensor.detach().cpu().to(torch.float64))
            output[key] = (accumulator / len(source_tensors)).to(base_tensor.dtype)
            averaged_tensors += 1
            averaged_numel += base_tensor.numel()
        else:
            if any(
                not torch.equal(tensor.detach().cpu(), base_tensor.detach().cpu())
                for tensor in source_tensors
            ):
                raise ValueError(f"Non-floating decoder buffer differs for {key}")

    return output, {
        "averaged_tensors": averaged_tensors,
        "averaged_numel": averaged_numel,
        "output_state_tensors": len(output),
    }


def main() -> None:
    args = parse_args()
    paths = [args.base_checkpoint, *args.source_checkpoints]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    base_checkpoint = torch.load(args.base_checkpoint, map_location="cpu")
    base_state = checkpoint_state_dict(base_checkpoint, args.base_checkpoint)
    source_states = []
    for path in args.source_checkpoints:
        checkpoint = torch.load(path, map_location="cpu")
        source_states.append(checkpoint_state_dict(checkpoint, path))

    soup_state, counts = average_decoder_state_dicts(base_state, source_states)
    metadata = {
        "method": "uniform_absolute_weight_average",
        "scope": "current_diff_decoder_floating_tensors_only",
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_sha256": file_sha256(args.base_checkpoint),
        "source_checkpoints": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in args.source_checkpoints
        ],
        **counts,
    }
    output_payload = {
        "state_dict": {f"agent.{key}": value for key, value in soup_state.items()},
        "grpo_model_soup": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(output_payload, temporary)
    temporary.replace(args.output)

    metadata_output = args.metadata_output or args.output.with_suffix(".json")
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata["output_checkpoint"] = str(args.output.resolve())
    metadata["output_sha256"] = file_sha256(args.output)
    metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))

    del base_checkpoint, source_states, soup_state, output_payload
    gc.collect()


if __name__ == "__main__":
    main()
