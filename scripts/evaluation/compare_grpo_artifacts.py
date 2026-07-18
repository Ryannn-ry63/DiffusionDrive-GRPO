#!/usr/bin/env python3
"""Paired comparison of frozen DiffusionDrive GRPO evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SAFETY_COMPONENTS = ("collision", "drivable", "ttc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--champion", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    return parser.parse_args()


def load_artifact(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"artifact has no records: {path}")
    tokens = [str(record["token"]) for record in records]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"artifact contains duplicate tokens: {path}")
    return payload


def bootstrap_ci(
    differences: np.ndarray, samples: int, seed: int
) -> list[float]:
    if samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        size = min(1000, samples - start)
        indices = rng.integers(
            0, differences.size, size=(size, differences.size)
        )
        means[start : start + size] = differences[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def values(records: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(record[key]) for record in records], dtype=np.float64)


def component_values(records: list[dict], name: str) -> np.ndarray:
    return np.asarray(
        [float(record["selected_components"][name]) for record in records],
        dtype=np.float64,
    )


def compare(
    baseline: dict,
    candidate: dict,
    samples: int,
    seed: int,
) -> dict:
    baseline_records = baseline["records"]
    candidate_records = candidate["records"]
    baseline_tokens = [str(record["token"]) for record in baseline_records]
    candidate_tokens = [str(record["token"]) for record in candidate_records]
    if candidate_tokens != baseline_tokens:
        raise ValueError("artifact token order does not match baseline")
    baseline_sha = baseline["summary"].get("token_set_sha256")
    candidate_sha = candidate["summary"].get("token_set_sha256")
    if baseline_sha != candidate_sha:
        raise ValueError("artifact token SHA does not match baseline")

    selected_difference = (
        values(candidate_records, "selected_reward")
        - values(baseline_records, "selected_reward")
    )
    oracle_difference = (
        values(candidate_records, "oracle_reward")
        - values(baseline_records, "oracle_reward")
    )
    candidate_difference = (
        values(candidate_records, "candidate_reward")
        - values(baseline_records, "candidate_reward")
    )
    baseline_selected = values(baseline_records, "selected_reward")
    safety_pass = np.ones(len(baseline_records), dtype=bool)
    component_differences = {}
    for name in SAFETY_COMPONENTS:
        delta = (
            component_values(candidate_records, name)
            - component_values(baseline_records, name)
        )
        component_differences[name] = float(delta.mean())
        safety_pass &= component_values(baseline_records, name) >= 1.0 - 1e-9

    low_reward = baseline_selected < 0.5
    tail_count = max(1, int(np.ceil(0.01 * len(selected_difference))))
    worst_delta_indices = np.argsort(selected_difference)[:tail_count]
    baseline_tail_indices = np.argsort(baseline_selected)[:tail_count]
    return {
        "num_tokens": len(baseline_records),
        "token_set_sha256": baseline_sha,
        "selected_difference": float(selected_difference.mean()),
        "selected_bootstrap_ci95": bootstrap_ci(
            selected_difference, samples, seed
        ),
        "oracle_difference": float(oracle_difference.mean()),
        "oracle_bootstrap_ci95": bootstrap_ci(
            oracle_difference, samples, seed + 1
        ),
        "candidate_reward_difference": float(candidate_difference.mean()),
        "wins": int((selected_difference > 1e-9).sum()),
        "ties": int((np.abs(selected_difference) <= 1e-9).sum()),
        "losses": int((selected_difference < -1e-9).sum()),
        "selected_component_differences": component_differences,
        "safety_pass_bucket": {
            "num_tokens": int(safety_pass.sum()),
            "selected_difference": (
                float(selected_difference[safety_pass].mean())
                if safety_pass.any()
                else None
            ),
        },
        "baseline_below_0_5_bucket": {
            "num_tokens": int(low_reward.sum()),
            "selected_difference": (
                float(selected_difference[low_reward].mean())
                if low_reward.any()
                else None
            ),
        },
        "worst_1pct_delta_mean": float(
            selected_difference[worst_delta_indices].mean()
        ),
        "baseline_bottom_1pct_selected_difference": float(
            selected_difference[baseline_tail_indices].mean()
        ),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def main() -> None:
    args = parse_args()
    baseline = load_artifact(args.baseline)
    candidate = load_artifact(args.candidate)
    output = {
        "schema_version": 1,
        "baseline_artifact": str(args.baseline.resolve()),
        "candidate_artifact": str(args.candidate.resolve()),
        "candidate_vs_baseline": compare(
            baseline,
            candidate,
            args.bootstrap_samples,
            args.bootstrap_seed,
        ),
    }
    if args.champion is not None:
        champion = load_artifact(args.champion)
        output["champion_artifact"] = str(args.champion.resolve())
        output["candidate_vs_champion"] = compare(
            champion,
            candidate,
            args.bootstrap_samples,
            args.bootstrap_seed + 10,
        )
        output["champion_vs_baseline"] = compare(
            baseline,
            champion,
            args.bootstrap_samples,
            args.bootstrap_seed + 20,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
