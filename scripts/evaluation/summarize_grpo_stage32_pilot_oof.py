#!/usr/bin/env python3
"""Summarize Stage32 pilot OOF artifacts with paired, log-bootstrap deltas."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from navsim.agents.diffusiondrive.stage30_mode_coverage import (
    STAGE30_CALIBRATION_SHA256,
    STAGE30_PUBLIC_SHA256,
    STAGE30_SELECTOR_SHA256,
)

NOISES = (20261411, 20261412)
STEPS = (192, 384, 576)
BRANCHES = ("DPEL", "SCF")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_records(path: Path) -> list[dict]:
    """Read only the small per-record fields needed for paired OOF statistics."""
    records = []
    current = None
    in_records = False
    token_re = re.compile(r'^\s*"token":\s*"([^"]+)"')
    log_re = re.compile(r'^\s*"log_name":\s*"([^"]+)"')
    selected_re = re.compile(r'^\s*"selected_reward":\s*([-+0-9.eE]+)')
    candidate_re = re.compile(r'^\s*"candidate_reward":\s*([-+0-9.eE]+)')
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not in_records:
                if line.lstrip().startswith('"records":'):
                    in_records = True
                continue
            match = token_re.match(line)
            if match:
                if current is not None:
                    records.append(current)
                current = {"token": match.group(1)}
                continue
            if current is None:
                continue
            match = log_re.match(line)
            if match:
                current["log_name"] = match.group(1)
                continue
            match = selected_re.match(line)
            if match:
                current["selected_reward"] = float(match.group(1))
                continue
            match = candidate_re.match(line)
            if match:
                current["candidate_reward"] = float(match.group(1))
    if current is not None:
        records.append(current)
    if not records or any(
        set(item) != {"token", "log_name", "selected_reward", "candidate_reward"}
        for item in records
    ):
        raise RuntimeError(f"invalid compact record extraction: {path}")
    return records


def load_artifact(
    path: Path,
    expected_checkpoint_sha: str,
    noise: int,
    expected_count: int,
    expected_domain: str,
) -> list[dict]:
    # The summary is small relative to the records; read only its prefix.
    with path.open("r", encoding="utf-8") as stream:
        prefix = []
        for line in stream:
            prefix.append(line)
            if line.lstrip().startswith('"records":'):
                break
    summary_text = "".join(prefix)
    marker = summary_text.rfind('"records":')
    if marker < 0:
        raise RuntimeError(f"Stage32 records array is missing: {path}")
    summary = json.loads(summary_text[:marker].rstrip().rstrip(",") + "\n}")["summary"]
    selector = summary["stage25_selector"]
    checks = (
        summary.get("completed") is True,
        summary.get("num_failures") == 0,
        summary.get("num_tokens") == expected_count,
        summary.get("checkpoint_sha256") == expected_checkpoint_sha,
        summary.get("reference_checkpoint_sha256") == STAGE30_PUBLIC_SHA256,
        summary.get("evaluation_noise_namespace") == noise,
        summary.get("generator_domain") == expected_domain,
        summary.get("selector_logits_source") == "trajectory_relative_harm_v3",
        selector.get("checkpoint_sha256") == STAGE30_SELECTOR_SHA256,
        selector.get("calibration_sha256") == STAGE30_CALIBRATION_SHA256,
    )
    if not all(checks):
        raise RuntimeError(f"Stage32 artifact provenance drifted: {path}")
    records = stream_records(path)
    if len(records) != expected_count:
        raise RuntimeError(f"Stage32 record count drifted: {path}")
    values = np.asarray(
        [[item["selected_reward"], item["candidate_reward"]] for item in records],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise RuntimeError(f"Stage32 non-finite reward in {path}")
    return records


def bootstrap_by_log(values: np.ndarray, logs: list[str], seed: int) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, log_name in zip(values, logs):
        grouped[log_name].append(float(value))
    names = sorted(grouped)
    sums = np.asarray([sum(grouped[name]) for name in names], dtype=np.float64)
    counts = np.asarray([len(grouped[name]) for name in names], dtype=np.int64)
    rng = np.random.default_rng(seed)
    samples = np.empty(10000, dtype=np.float64)
    for start in range(0, len(samples), 500):
        size = min(500, len(samples) - start)
        indices = rng.integers(0, len(names), (size, len(names)))
        samples[start:start + size] = (
            sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
        )
    return np.quantile(samples, (0.025, 0.975)).tolist()


def vector(records: list[dict], key: str) -> np.ndarray:
    return np.asarray([record[key] for record in records], dtype=np.float64)


def summarize_delta(
    delta: np.ndarray,
    candidate_delta: np.ndarray,
    records: list[dict],
    bucket_ids: np.ndarray,
    seed: int,
) -> dict:
    logs = [record["log_name"] for record in records]
    hard = bucket_ids < 3
    mature = bucket_ids == 3
    wins = int(np.count_nonzero(delta > 0))
    losses = int(np.count_nonzero(delta < 0))
    return {
        "count": int(delta.size),
        "mean_selected_delta": float(delta.mean()),
        "mean_candidate_delta": float(candidate_delta.mean()),
        "whole_log_ci95": bootstrap_by_log(delta, logs, seed),
        "hard_selected_delta": float(delta[hard].mean()),
        "mature_selected_delta": float(delta[mature].mean()),
        "wins": wins,
        "losses": losses,
        "ties": int(delta.size - wins - losses),
        "catastrophic_count_delta_below_-0.01": int(np.count_nonzero(delta < -0.01)),
        "finite": bool(np.isfinite(delta).all() and np.isfinite(candidate_delta).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--bucket-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bucket_payload = json.loads(args.bucket_manifest.read_text(encoding="utf-8"))
    if sha256(args.bucket_manifest) != "e067680e9d601929b69b932c59de1f092e3c229d6aeb0bf5ee5aa0408b9a7cf7":
        raise RuntimeError("Stage32 bucket manifest SHA drifted")
    bucket_map = {
        str(token): int(record["bucket_id"])
        for token, record in bucket_payload["tokens"].items()
    }
    result = {
        "schema_version": 1,
        "stage": 32,
        "pilot_oof": True,
        "public_checkpoint_sha256": STAGE30_PUBLIC_SHA256,
        "selector_checkpoint_sha256": STAGE30_SELECTOR_SHA256,
        "calibration_sha256": STAGE30_CALIBRATION_SHA256,
        "folds": [0, 1],
        "noise_namespaces": list(NOISES),
        "steps": list(STEPS),
        "branches": {},
    }
    loaded: dict[tuple[str, int, int, int], list[dict]] = {}
    checkpoint_shas = {}
    for branch in ("P",) + BRANCHES:
        result["branches"][branch] = {}
        for fold in (0, 1):
            result["branches"][branch][str(fold)] = {}
            for step in STEPS if branch != "P" else (0,):
                for noise in NOISES:
                    name = f"{branch if branch == 'P' else branch + str(step)}_ns{noise}.json"
                    path = args.eval_root / branch if branch != "P" else args.eval_root / "DPEL"
                    path = path / f"fold{fold}" / name
                    expected_domain = (
                        f"stage32_pilot_p_fold{fold}"
                        if branch == "P"
                        else f"stage32_pilot_{branch.lower()}{step}_fold{fold}"
                    )
                    expected_sha = (
                        STAGE30_PUBLIC_SHA256 if branch == "P"
                        else json.loads(
                            (args.eval_root.parent / "training" / branch
                             / f"fold{fold}" / "formal" / "audit.json").read_text()
                        )["checkpoints"][
                            {192: 0, 384: 1, 576: 2}[step]
                        ]["sha256"]
                    )
                    records = load_artifact(
                        path, expected_sha, noise,
                        1019 if fold == 0 else 1021,
                        expected_domain,
                    )
                    loaded[(branch, fold, step, noise)] = records
                    if branch != "P":
                        checkpoint_shas[f"{branch}{step}/fold{fold}"] = expected_sha

    for branch in BRANCHES:
        result["branches"][branch] = {}
        for step in STEPS:
            rows = []
            pooled_delta, pooled_candidate, pooled_records, pooled_buckets = [], [], [], []
            paired_delta, paired_candidate, paired_records, paired_buckets = [], [], [], []
            for fold in (0, 1):
                fold_rows = []
                for noise in NOISES:
                    public = loaded[("P", fold, 0, noise)]
                    candidate = loaded[(branch, fold, step, noise)]
                    if [x["token"] for x in public] != [x["token"] for x in candidate]:
                        raise RuntimeError("Stage32 token order drifted")
                    ids = np.asarray([bucket_map[x["token"]] for x in public], dtype=np.int64)
                    delta = vector(candidate, "selected_reward") - vector(public, "selected_reward")
                    candidate_delta = vector(candidate, "candidate_reward") - vector(public, "candidate_reward")
                    records = [
                        {"log_name": f"{fold}:{noise}:{item['log_name']}"}
                        for item in public
                    ]
                    pooled_delta.append(delta)
                    pooled_candidate.append(candidate_delta)
                    pooled_records.extend(records)
                    pooled_buckets.append(ids)
                    fold_rows.append(summarize_delta(delta, candidate_delta, records, ids, 20263200 + fold * 100 + noise % 100))
                    if branch == "SCF":
                        dpel = loaded[("DPEL", fold, step, noise)]
                        sdelta = vector(candidate, "selected_reward") - vector(dpel, "selected_reward")
                        scandidate = vector(candidate, "candidate_reward") - vector(dpel, "candidate_reward")
                        paired_delta.append(sdelta)
                        paired_candidate.append(scandidate)
                        paired_records.extend(records)
                        paired_buckets.append(ids)
                rows.append({"fold": fold, "noise_rows": fold_rows})
            pooled_delta_array = np.concatenate(pooled_delta)
            pooled_candidate_array = np.concatenate(pooled_candidate)
            pooled_records_flat = pooled_records
            pooled_buckets_array = np.concatenate(pooled_buckets)
            summary = summarize_delta(
                pooled_delta_array, pooled_candidate_array,
                pooled_records_flat, pooled_buckets_array, 20263232 + step,
            )
            summary["fold_noise_rows"] = rows
            if branch == "SCF":
                summary["paired_vs_DPEL"] = summarize_delta(
                    np.concatenate(paired_delta), np.concatenate(paired_candidate),
                    paired_records, np.concatenate(paired_buckets), 20263264 + step,
                )
            result["branches"][branch][str(step)] = summary

    scf_rows = [result["branches"]["SCF"][str(step)] for step in STEPS]
    result["pilot_screening"] = {}
    for step, row in zip(STEPS, scf_rows):
        paired = row["paired_vs_DPEL"]
        row["pilot_screening"] = {
            "selected_gain_at_least_0.0015": row["mean_selected_delta"] >= 0.0015,
            "paired_gain_at_least_0.0005": paired["mean_selected_delta"] >= 0.0005,
            "whole_log_ci_lower_positive": row["whole_log_ci95"][0] > 0.0,
            "hard_gain_at_least_0.005": row["hard_selected_delta"] >= 0.005,
            "mature_not_worse_than_-0.0002": row["mature_selected_delta"] >= -0.0002,
            "catastrophic_count_zero": row["catastrophic_count_delta_below_-0.01"] == 0,
            "passed": all((
                row["mean_selected_delta"] >= 0.0015,
                paired["mean_selected_delta"] >= 0.0005,
                row["whole_log_ci95"][0] > 0.0,
                row["hard_selected_delta"] >= 0.005,
                row["mature_selected_delta"] >= -0.0002,
                row["catastrophic_count_delta_below_-0.01"] == 0,
            )),
        }
        result["pilot_screening"][str(step)] = row["pilot_screening"]
    result["best_scf_step_by_selected_delta"] = int(max(
        STEPS, key=lambda step: result["branches"]["SCF"][str(step)]["mean_selected_delta"]
    ))
    result["best_scf_step_passes_pilot_screening"] = result["pilot_screening"][
        str(result["best_scf_step_by_selected_delta"])
    ]["passed"]
    result["checkpoint_shas"] = checkpoint_shas
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
