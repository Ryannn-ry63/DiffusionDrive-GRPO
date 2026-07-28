#!/usr/bin/env python3
"""Paired two-fold Stage34 pilot summary and frozen promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from navsim.agents.diffusiondrive.stage30_mode_coverage import (
    STAGE30_CALIBRATION_SHA256,
    STAGE30_PUBLIC_SHA256,
    STAGE30_SELECTOR_SHA256,
)
from navsim.agents.diffusiondrive.stage34_contract import (
    STAGE34_OBJECTIVE_REVISION,
    STAGE34_PLAN_SHA256,
)


NOISES = (20261511, 20261512)
STEPS = (48, 96, 144, 192)
COMPONENTS = ("collision", "drivable", "ttc")
BUCKET_SHA256 = "e067680e9d601929b69b932c59de1f092e3c229d6aeb0bf5ee5aa0408b9a7cf7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_checkpoint(
    root: Path, system: str, fold: int
) -> tuple[str, str]:
    if system == "P":
        return STAGE30_PUBLIC_SHA256, f"stage34_pilot_p_fold{fold}"
    if system == "DPEL192":
        freeze_path = (
            root.parent.parent.parent
            / "grpo_stage32"
            / "pilot"
            / "training"
            / "DPEL"
            / f"fold{fold}"
            / "formal"
            / "checkpoints.json"
        )
        freeze = json.loads(freeze_path.read_text())
        matches = [
            record
            for record in freeze["checkpoints"]
            if int(record["global_step"]) == 192
        ]
        if len(matches) != 1:
            raise RuntimeError("Stage34 DPEL192 checkpoint freeze drifted")
        return matches[0]["sha256"], f"stage34_pilot_dpel192_fold{fold}"
    step = int(system.removeprefix("MAF"))
    freeze_path = (
        root.parent
        / "training"
        / "MAF"
        / f"fold{fold}"
        / "formal"
        / "checkpoints.json"
    )
    freeze = json.loads(freeze_path.read_text())
    matches = [
        record
        for record in freeze["checkpoints"]
        if int(record["global_step"]) == step
    ]
    if (
        not freeze.get("passed")
        or freeze.get("stage") != 34
        or freeze.get("objective_revision") != STAGE34_OBJECTIVE_REVISION
        or freeze.get("plan_sha256") != STAGE34_PLAN_SHA256
        or len(matches) != 1
    ):
        raise RuntimeError(f"Stage34 MAF{step} checkpoint freeze drifted")
    return matches[0]["sha256"], f"stage34_pilot_maf{step}_fold{fold}"


def load_artifact(
    path: Path,
    expected_sha: str,
    expected_domain: str,
    noise: int,
    expected_count: int,
) -> dict:
    payload = json.loads(path.read_text())
    summary, raw_records = payload["summary"], payload["records"]
    selector = summary["stage25_selector"]
    checks = (
        summary.get("completed") is True,
        summary.get("num_failures") == 0,
        summary.get("num_tokens") == expected_count == len(raw_records),
        summary.get("checkpoint_sha256") == expected_sha,
        summary.get("reference_checkpoint_sha256") == STAGE30_PUBLIC_SHA256,
        summary.get("generator_domain") == expected_domain,
        summary.get("evaluation_noise_namespace") == noise,
        summary.get("selector_logits_source") == "trajectory_relative_harm_v3",
        selector.get("checkpoint_sha256") == STAGE30_SELECTOR_SHA256,
        selector.get("calibration_sha256") == STAGE30_CALIBRATION_SHA256,
        summary.get("schedule", {}).get("roll_timesteps") == [32, 24, 16, 8, 0],
    )
    if not all(checks):
        raise RuntimeError(f"Stage34 artifact provenance drifted: {path}")

    tokens, logs, selected, candidate_mean = [], [], [], []
    candidate_rewards, raw_oracle, safe_oracle = [], [], []
    selected_components = {name: [] for name in COMPONENTS}
    for record in raw_records:
        rewards = np.asarray(record["candidate_rewards"], dtype=np.float64)
        if rewards.shape != (20,) or not np.isfinite(rewards).all():
            raise RuntimeError(f"Stage34 candidate reward shape drifted: {path}")
        stage24 = record["stage24_selector"]
        eligible = np.asarray(stage24["eligible"], dtype=bool)
        fallback = int(stage24["fallback_mode"])
        if eligible.shape != (20,) or not 0 <= fallback < 20:
            raise RuntimeError(f"Stage34 eligibility shape drifted: {path}")
        deployable = eligible.copy()
        deployable[fallback] = True
        tokens.append(record["token"])
        logs.append(record["log_name"])
        selected.append(float(record["selected_reward"]))
        candidate_mean.append(float(rewards.mean()))
        candidate_rewards.append(rewards)
        raw_oracle.append(float(rewards.max()))
        safe_oracle.append(float(rewards[deployable].max()))
        for name in COMPONENTS:
            selected_components[name].append(
                float(record["selected_components"][name])
            )
    arrays = {
        "tokens": tokens,
        "logs": logs,
        "selected": np.asarray(selected, dtype=np.float64),
        "candidate_mean": np.asarray(candidate_mean, dtype=np.float64),
        "candidate_rewards": np.stack(candidate_rewards),
        "raw_oracle": np.asarray(raw_oracle, dtype=np.float64),
        "safe_oracle": np.asarray(safe_oracle, dtype=np.float64),
        "components": {
            name: np.asarray(values, dtype=np.float64)
            for name, values in selected_components.items()
        },
    }
    numeric = [
        arrays["selected"],
        arrays["candidate_mean"],
        arrays["candidate_rewards"],
        arrays["raw_oracle"],
        arrays["safe_oracle"],
        *arrays["components"].values(),
    ]
    if not all(np.isfinite(value).all() for value in numeric):
        raise RuntimeError(f"Stage34 non-finite compact artifact: {path}")
    return arrays


def bootstrap_by_log(
    values: np.ndarray, logs: list[str], seed: int
) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, log_name in zip(values, logs):
        grouped[log_name].append(float(value))
    names = sorted(grouped)
    sums = np.asarray([sum(grouped[name]) for name in names], dtype=np.float64)
    counts = np.asarray([len(grouped[name]) for name in names], dtype=np.int64)
    rng = np.random.default_rng(seed)
    samples = np.empty(10000, dtype=np.float64)
    for start in range(0, samples.size, 500):
        size = min(500, samples.size - start)
        indices = rng.integers(0, len(names), (size, len(names)))
        samples[start : start + size] = (
            sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
        )
    return np.quantile(samples, (0.025, 0.975)).tolist()


def trimmed10_mean(values: np.ndarray) -> float:
    trim = int(np.floor(values.size * 0.10))
    ordered = np.sort(values)
    retained = ordered[trim : values.size - trim] if trim else ordered
    return float(retained.mean())


def summarize(
    *,
    selected_delta: np.ndarray,
    candidate_delta: np.ndarray,
    raw_oracle_delta: np.ndarray,
    safe_oracle_delta: np.ndarray,
    top5_delta: np.ndarray,
    component_deltas: dict[str, np.ndarray],
    buckets: np.ndarray,
    logs: list[str],
    folds: np.ndarray,
    noises: np.ndarray,
    seed: int,
) -> dict:
    hard = buckets < 3
    mature = buckets == 3
    wins = int(np.count_nonzero(selected_delta > 0))
    losses = int(np.count_nonzero(selected_delta < 0))
    fold_means = {
        str(fold): float(selected_delta[folds == fold].mean())
        for fold in (0, 1)
    }
    namespace_means = {
        str(noise): float(selected_delta[noises == noise].mean())
        for noise in NOISES
    }
    return {
        "count": int(selected_delta.size),
        "mean_selected_delta": float(selected_delta.mean()),
        "mean_candidate_delta": float(candidate_delta.mean()),
        "mean_raw_oracle_delta": float(raw_oracle_delta.mean()),
        "mean_safe_deployable_oracle_delta": float(safe_oracle_delta.mean()),
        "mean_public_top5_delta": float(top5_delta.mean()),
        "whole_log_bootstrap_ci95": bootstrap_by_log(
            selected_delta, logs, seed
        ),
        "fold_mean_selected_delta": fold_means,
        "namespace_mean_selected_delta": namespace_means,
        "hard_selected_delta": float(selected_delta[hard].mean()),
        "mature_selected_delta": float(selected_delta[mature].mean()),
        "component_mean_deltas": {
            name: float(values.mean())
            for name, values in component_deltas.items()
        },
        "trimmed10_mean_selected_delta": trimmed10_mean(selected_delta),
        "wins": wins,
        "losses": losses,
        "ties": int(selected_delta.size - wins - losses),
        "catastrophic_count_delta_at_most_-0.5": int(
            np.count_nonzero(selected_delta <= -0.5)
        ),
        "finite": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--bucket-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.bucket_manifest) != BUCKET_SHA256:
        raise RuntimeError("Stage34 bucket manifest SHA drifted")
    bucket_payload = json.loads(args.bucket_manifest.read_text())
    bucket_map = {
        token: int(record["bucket_id"])
        for token, record in bucket_payload["tokens"].items()
    }
    systems = ("P", "DPEL192") + tuple(f"MAF{step}" for step in STEPS)
    loaded = {}
    checkpoint_shas = {}
    for system in systems:
        for fold in (0, 1):
            checkpoint_sha, domain = expected_checkpoint(
                args.eval_root, system, fold
            )
            checkpoint_shas[f"{system}/fold{fold}"] = checkpoint_sha
            expected_count = 1019 if fold == 0 else 1021
            for noise in NOISES:
                path = (
                    args.eval_root
                    / f"fold{fold}"
                    / f"{system}_ns{noise}.json"
                )
                loaded[(system, fold, noise)] = load_artifact(
                    path, checkpoint_sha, domain, noise, expected_count
                )

    comparisons = {}
    for step in STEPS:
        system = f"MAF{step}"
        vectors = defaultdict(list)
        component_vectors = {name: [] for name in COMPONENTS}
        logs, fold_ids, noise_ids, bucket_ids = [], [], [], []
        dpel_selected, dpel_candidate = [], []
        for fold in (0, 1):
            for noise in NOISES:
                public = loaded[("P", fold, noise)]
                current = loaded[(system, fold, noise)]
                dpel = loaded[("DPEL192", fold, noise)]
                if not (
                    public["tokens"] == current["tokens"] == dpel["tokens"]
                ):
                    raise RuntimeError("Stage34 paired token order drifted")
                public_top5 = np.argpartition(
                    public["candidate_rewards"], -5, axis=1
                )[:, -5:]
                row = np.arange(len(public["tokens"]))[:, None]
                current_top5 = current["candidate_rewards"][row, public_top5]
                public_top5_rewards = public["candidate_rewards"][row, public_top5]
                vectors["selected"].append(
                    current["selected"] - public["selected"]
                )
                vectors["candidate"].append(
                    current["candidate_mean"] - public["candidate_mean"]
                )
                vectors["raw_oracle"].append(
                    current["raw_oracle"] - public["raw_oracle"]
                )
                vectors["safe_oracle"].append(
                    current["safe_oracle"] - public["safe_oracle"]
                )
                vectors["top5"].append(
                    (current_top5 - public_top5_rewards).mean(axis=1)
                )
                dpel_selected.append(current["selected"] - dpel["selected"])
                dpel_candidate.append(
                    current["candidate_mean"] - dpel["candidate_mean"]
                )
                for name in COMPONENTS:
                    component_vectors[name].append(
                        current["components"][name] - public["components"][name]
                    )
                count = len(public["tokens"])
                logs.extend(
                    f"{fold}:{noise}:{name}" for name in public["logs"]
                )
                fold_ids.append(np.full(count, fold, dtype=np.int64))
                noise_ids.append(np.full(count, noise, dtype=np.int64))
                bucket_ids.append(
                    np.asarray(
                        [bucket_map[token] for token in public["tokens"]],
                        dtype=np.int64,
                    )
                )
        arrays = {
            name: np.concatenate(parts) for name, parts in vectors.items()
        }
        folds = np.concatenate(fold_ids)
        noises = np.concatenate(noise_ids)
        buckets = np.concatenate(bucket_ids)
        summary = summarize(
            selected_delta=arrays["selected"],
            candidate_delta=arrays["candidate"],
            raw_oracle_delta=arrays["raw_oracle"],
            safe_oracle_delta=arrays["safe_oracle"],
            top5_delta=arrays["top5"],
            component_deltas={
                name: np.concatenate(parts)
                for name, parts in component_vectors.items()
            },
            buckets=buckets,
            logs=logs,
            folds=folds,
            noises=noises,
            seed=20263400 + step,
        )
        paired_dpel = np.concatenate(dpel_selected)
        summary["paired_vs_DPEL192"] = {
            "mean_selected_delta": float(paired_dpel.mean()),
            "mean_candidate_delta": float(
                np.concatenate(dpel_candidate).mean()
            ),
            "whole_log_bootstrap_ci95": bootstrap_by_log(
                paired_dpel, logs, 20263450 + step
            ),
        }
        checks = {
            "pooled_selected_gain_at_least_0.0015":
                summary["mean_selected_delta"] >= 0.0015,
            "both_fold_means_strictly_positive": all(
                value > 0
                for value in summary["fold_mean_selected_delta"].values()
            ),
            "both_namespace_means_strictly_positive": all(
                value > 0
                for value in summary[
                    "namespace_mean_selected_delta"
                ].values()
            ),
            "whole_log_ci_lower_strictly_positive":
                summary["whole_log_bootstrap_ci95"][0] > 0,
            "gain_over_DPEL192_at_least_0.0005":
                summary["paired_vs_DPEL192"]["mean_selected_delta"] >= 0.0005,
            "safe_deployable_oracle_gain_at_least_0.0005":
                summary["mean_safe_deployable_oracle_delta"] >= 0.0005,
            "raw_oracle_nonnegative":
                summary["mean_raw_oracle_delta"] >= 0,
            "public_top5_nonnegative":
                summary["mean_public_top5_delta"] >= 0,
            "mature_selected_at_least_negative_0.0001":
                summary["mature_selected_delta"] >= -0.0001,
            "guard_components_at_least_negative_0.0005": all(
                summary["component_mean_deltas"][name] >= -0.0005
                for name in COMPONENTS
            ),
            "trimmed10_mean_nonnegative":
                summary["trimmed10_mean_selected_delta"] >= 0,
            "wins_exceed_losses": summary["wins"] > summary["losses"],
            "catastrophic_count_zero":
                summary["catastrophic_count_delta_at_most_-0.5"] == 0,
        }
        checks["passed"] = all(checks.values())
        summary["promotion_checks"] = checks
        comparisons[str(step)] = summary

    passing = [
        step
        for step in STEPS
        if comparisons[str(step)]["promotion_checks"]["passed"]
    ]
    best_step = max(
        STEPS,
        key=lambda step: comparisons[str(step)]["mean_selected_delta"],
    )
    result = {
        "schema_version": 1,
        "stage": 34,
        "pilot_oof": True,
        "method": "MAF-GRPO",
        "objective_revision": STAGE34_OBJECTIVE_REVISION,
        "plan_sha256": STAGE34_PLAN_SHA256,
        "public_checkpoint_sha256": STAGE30_PUBLIC_SHA256,
        "selector_checkpoint_sha256": STAGE30_SELECTOR_SHA256,
        "calibration_sha256": STAGE30_CALIBRATION_SHA256,
        "folds": [0, 1],
        "noise_namespaces": list(NOISES),
        "steps": list(STEPS),
        "comparisons": comparisons,
        "passing_steps": passing,
        "pilot_passed": bool(passing),
        "best_step_by_selected_gain": best_step,
        "best_step_passes": best_step in passing,
        "all_checkpoints_safe_oracle_nonpositive": all(
            comparisons[str(step)]["mean_safe_deployable_oracle_delta"] <= 0
            for step in STEPS
        ),
        "stage34_stop_for_no_generator_ceiling": all(
            comparisons[str(step)]["mean_safe_deployable_oracle_delta"] <= 0
            for step in STEPS
        ),
        "checkpoint_shas": checkpoint_shas,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
