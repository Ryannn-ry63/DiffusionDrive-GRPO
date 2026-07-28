#!/usr/bin/env python3
"""Validate and summarize the locked Stage26 P/A/B0/B/C NavTest runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


SYSTEMS = ("P", "A", "B0", "B", "C")
METRICS = (
    "score",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
)
COMPARISONS = (
    ("A-P", "A", "P"),
    ("B0-A", "B0", "A"),
    ("B-B0", "B", "B0"),
    ("C-B", "C", "B"),
    ("C-B0", "C", "B0"),
    ("C-A", "C", "A"),
    ("C-P", "C", "P"),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_complete_csv(exp_root: Path, system: str) -> Tuple[Path, pd.DataFrame]:
    experiment = (
        "stage27_public88_navtest_P"
        if system == "P"
        else f"stage26_navtest_pdms_{system}"
    )
    root = exp_root / experiment
    candidates = sorted(root.glob("*/*.csv"))
    complete = []
    for path in candidates:
        frame = pd.read_csv(path)
        token_rows = frame[frame["token"].astype(str) != "average"].copy()
        if len(token_rows) == 12146:
            complete.append((path, token_rows))
    if len(complete) != 1:
        raise RuntimeError(
            f"{system}: expected exactly one complete 12146-row CSV, "
            f"found {len(complete)} under {root}"
        )
    path, frame = complete[0]
    if frame["token"].duplicated().any():
        raise RuntimeError(f"{system}: duplicate NavTest tokens")
    if "valid" not in frame or not frame["valid"].astype(bool).all():
        bad = 12146 if "valid" not in frame else int(
            (~frame["valid"].astype(bool)).sum()
        )
        raise RuntimeError(f"{system}: {bad} invalid NavTest rows")
    missing = [name for name in METRICS if name not in frame]
    if missing:
        raise RuntimeError(f"{system}: missing PDM columns {missing}")
    frame["token"] = frame["token"].astype(str)
    return path, frame.set_index("token").sort_index()


def load_token_logs(metric_cache: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for path in metric_cache.rglob("metric_cache.pkl"):
        token = path.parent.name
        log_name = path.parents[2].name
        if token in mapping and mapping[token] != log_name:
            raise RuntimeError(f"metric-cache token maps to two logs: {token}")
        mapping[token] = log_name
    if len(mapping) != 12146:
        raise RuntimeError(
            f"expected 12146 token/log mappings, found {len(mapping)}"
        )
    return mapping


def whole_log_ci(
    delta: pd.Series,
    token_logs: Dict[str, str],
    seed: int = 26026,
    samples: int = 10000,
) -> Tuple[float, float]:
    grouped = pd.DataFrame(
        {
            "delta": delta,
            "log_name": [token_logs[token] for token in delta.index],
        },
        index=delta.index,
    ).groupby("log_name")["delta"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(grouped), size=(samples, len(grouped)), endpoint=False
    )
    sums = grouped["sum"].to_numpy()[indices].sum(axis=1)
    counts = grouped["count"].to_numpy()[indices].sum(axis=1)
    estimates = sums / counts
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def comparison_summary(
    left: pd.DataFrame,
    right: pd.DataFrame,
    token_logs: Dict[str, str],
) -> dict:
    score_delta = left["score"] - right["score"]
    epsilon = 1e-12
    return {
        "mean_delta": float(score_delta.mean()),
        "pdm_points_delta": float(100.0 * score_delta.mean()),
        "whole_log_bootstrap_95ci": list(
            whole_log_ci(score_delta, token_logs)
        ),
        "wins": int((score_delta > epsilon).sum()),
        "ties": int((score_delta.abs() <= epsilon).sum()),
        "losses": int((score_delta < -epsilon).sum()),
        "worst_delta": float(score_delta.min()),
        "best_delta": float(score_delta.max()),
        "catastrophic_regressions_le_minus_0_5": int(
            (score_delta <= -0.5).sum()
        ),
        "metric_mean_deltas": {
            name: float((left[name] - right[name]).mean())
            for name in METRICS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-root",
        type=Path,
        default=Path(
            "/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp"
        ),
    )
    parser.add_argument(
        "--metric-cache",
        type=Path,
        default=Path(
            "/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/"
            "metric_cache_navtest"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/grpo_stage26/navtest/final_report.json"
        ),
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen NavTest report: {args.output}"
        )
    token_logs = load_token_logs(args.metric_cache)
    csv_paths: Dict[str, Path] = {}
    frames: Dict[str, pd.DataFrame] = {}
    for system in SYSTEMS:
        csv_paths[system], frames[system] = locate_complete_csv(
            args.exp_root, system
        )
    reference_tokens = set(frames["P"].index)
    if reference_tokens != set(token_logs):
        raise RuntimeError("P tokens do not match the locked NavTest cache")
    for system in SYSTEMS[1:]:
        if set(frames[system].index) != reference_tokens:
            raise RuntimeError(f"{system}: token set differs from A")

    means = {
        system: {
            name: float(frames[system][name].mean())
            for name in METRICS
        }
        for system in SYSTEMS
    }
    comparisons = {
        name: comparison_summary(frames[left], frames[right], token_logs)
        for name, left, right in COMPARISONS
    }
    report = {
        "schema_version": 1,
        "stage": 26,
        "protocol": {
            "P": "actual epoch-99 public 88.1 checkpoint + official selector + [8,0] schedule",
            "A": "locked local epoch-19 Stage13-26 base + official selector + [8,0] schedule",
            "B0": "locked local epoch-19 base + official selector + Stage26 full-chain schedule",
            "B": "locked local epoch-19 base + frozen Stage26 selector + full-chain schedule",
            "C": "full-6119 GRPO generator + same frozen Stage26 selector + full-chain schedule",
            "paired_noise": "SHA256(token)-seeded initial diffusion noise",
            "primary_grpo_comparison": "C-B",
            "selector_comparison": "B-B0",
            "total_stage26_comparison": "C-B0",
            "released_checkpoint_context_comparison": "C-P",
        },
        "num_tokens": 12146,
        "num_logs": len(set(token_logs.values())),
        "mean_metrics": means,
        "comparisons": comparisons,
        "grpo_improved": comparisons["C-B"]["mean_delta"] > 0.0,
        "selector_improved": comparisons["B-B0"]["mean_delta"] > 0.0,
        "total_stage26_improved": comparisons["C-B0"]["mean_delta"] > 0.0,
        "end_to_end_improved_over_released_checkpoint": (
            comparisons["C-P"]["mean_delta"] > 0.0
        ),
        "artifacts": {
            system: {
                "csv": str(csv_paths[system].resolve()),
                "csv_sha256": file_sha256(csv_paths[system]),
            }
            for system in SYSTEMS
        },
        "locked_inputs": {
            "locked_local_epoch19_base_sha256": "59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d",
            "actual_public_88p1_sha256": "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b",
            "full_6119_generator_sha256": "1018f1b1a1cfcb69c27b97c2fbd30099a20a62e55fc02897f6588ca546abad6d",
            "selector_sha256": "77ee768c5f469292d81049192d48797f2f3fae4f80029ae6853dd9d9c6de4207",
            "calibration_sha256": "0c14b8f9531d64028bdf4d71f35b34418ab31ad4ae11ccc6b3c60a4325f53a0d",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
