#!/usr/bin/env python3
"""Build the immutable train-only Phase-5 functional trust calibration."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

FORMULA_VERSION = "reference_mean_ball_v1"
SEED = 20260719
PERCENTILE = 0.99
EXPECTED_TOKENS = 6119


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_token_sha256(tokens) -> str:
    return hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-artifact", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--champion-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_records(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"artifact has no records list: {path}")
    tokens = [str(record["token"]) for record in records]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"artifact contains duplicate tokens: {path}")
    return payload, records, tokens


def main() -> None:
    args = parse_args()
    for path in (
        args.collection_artifact,
        args.train_manifest,
        args.base_checkpoint,
        args.champion_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(
            f"refusing to replace registered calibration artifact: {args.output}"
        )

    collection, records, tokens = load_records(args.collection_artifact)
    _, _, manifest_tokens = load_records(args.train_manifest)
    summary = collection.get("summary", {})
    trust_summary = summary.get("generation_trust", {})
    if tokens != manifest_tokens:
        raise RuntimeError("calibration collection order differs from train manifest")
    if len(tokens) != EXPECTED_TOKENS:
        raise RuntimeError(
            f"calibration requires {EXPECTED_TOKENS} train tokens, got {len(tokens)}"
        )
    token_sha = ordered_token_sha256(tokens)
    if summary.get("token_set_sha256") != token_sha:
        raise RuntimeError("collection token SHA256 is inconsistent")
    manifest_summary = json.loads(
        args.train_manifest.read_text(encoding="utf-8")
    ).get("summary", {})
    if manifest_summary.get("token_set_sha256") != token_sha:
        raise RuntimeError("train manifest token SHA256 is inconsistent")
    if summary.get("log_split") != "train":
        raise RuntimeError("trust calibration may only use the train split")
    if not trust_summary.get("collect_calibration"):
        raise RuntimeError("input is not a raw trust calibration collection")
    if trust_summary.get("mode") != "none":
        raise RuntimeError("calibration collection must be unprojected")
    if trust_summary.get("calibration_seed") != SEED:
        raise RuntimeError("calibration seed mismatch")
    collected_candidate = Path(str(summary.get("checkpoint", "")))
    collected_reference = Path(
        str(trust_summary.get("reference_checkpoint", ""))
    )
    if collected_candidate.resolve() != args.champion_checkpoint.resolve():
        raise RuntimeError(
            "collection candidate is not the registered D128 champion"
        )
    if collected_reference.resolve() != args.base_checkpoint.resolve():
        raise RuntimeError(
            "collection reference is not the registered frozen base"
        )

    distances = []
    for record in records:
        trust = record.get("generation_trust")
        if not isinstance(trust, dict) or "pre_distance" not in trust:
            raise RuntimeError(f"missing displacement for token {record['token']}")
        value = np.asarray(trust["pre_distance"], dtype=np.float64)
        if value.ndim != 2 or value.shape[-1] != 2 or not np.isfinite(value).all():
            raise RuntimeError(
                f"invalid displacement shape/value for token {record['token']}"
            )
        distances.append(value)
    distances = np.concatenate(distances, axis=0)

    schedule = summary.get("schedule", {})
    roll_timesteps = list(schedule.get("roll_timesteps", ()))
    if len(roll_timesteps) != 2 or roll_timesteps[-1] != 0:
        raise RuntimeError("calibration requires the registered two-step schedule")
    step_payload = {}
    for step_index, step_name in enumerate(("transition", "final")):
        values = distances[:, step_index]
        quantiles = {
            "p50": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, PERCENTILE)),
            "max": float(values.max()),
        }
        step_summary = trust_summary.get("steps", {}).get(step_name, {})
        if int(step_summary.get("count", -1)) != int(values.size):
            raise RuntimeError(f"{step_name} displacement count mismatch")
        step_payload[step_name] = {
            "timestep": int(roll_timesteps[step_index]),
            "sigma": float(step_summary["sigma"]),
            "radius": quantiles["p99"],
            "quantiles": quantiles,
            "count": int(values.size),
        }

    payload = {
        "schema_version": 1,
        "formula_version": FORMULA_VERSION,
        "mode": "reference_mean_ball",
        "seed": SEED,
        "percentile": PERCENTILE,
        "manifest": {
            "path": str(args.train_manifest.resolve()),
            "sha256": file_sha256(args.train_manifest),
            "ordered_token_sha256": token_sha,
            "num_tokens": len(tokens),
            "split": "train",
        },
        "base_checkpoint": {
            "path": str(args.base_checkpoint.resolve()),
            "sha256": file_sha256(args.base_checkpoint),
        },
        "champion_checkpoint": {
            "path": str(args.champion_checkpoint.resolve()),
            "sha256": file_sha256(args.champion_checkpoint),
        },
        "collection_artifact": {
            "path": str(args.collection_artifact.resolve()),
            "sha256": file_sha256(args.collection_artifact),
        },
        "schedule": {
            "truncation_timestep": int(schedule["truncation_timestep"]),
            "roll_timesteps": roll_timesteps,
            "scheduler_num_inference_steps": int(
                schedule["scheduler_num_inference_steps"]
            ),
            "scheduler_step_stride": int(schedule["scheduler_step_stride"]),
        },
        "steps": step_payload,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    os.chmod(args.output, 0o444)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
