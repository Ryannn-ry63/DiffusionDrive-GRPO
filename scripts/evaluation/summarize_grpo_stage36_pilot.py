#!/usr/bin/env python3
"""Paired two-fold Stage36 pilot summary and frozen promotion gate."""

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
from navsim.agents.diffusiondrive.stage35_counterfactual import (
    STAGE35_OBJECTIVE_REVISION,
    STAGE35_PLAN_SHA256,
)
from navsim.agents.diffusiondrive.stage36_contract import (
    STAGE36_OBJECTIVE_REVISION,
    STAGE36_PLAN_SHA256,
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
            raise RuntimeError("Stage36 DPEL192 checkpoint freeze drifted")
        return matches[0]["sha256"], f"stage34_pilot_dpel192_fold{fold}"
    if system == "NCD192":
        freeze_path = (
            root.parent / "training" / "NCD" / f"fold{fold}"
            / "formal" / "checkpoints.json"
        )
        freeze = json.loads(freeze_path.read_text())
        matches = [
            record for record in freeze["checkpoints"]
            if int(record["global_step"]) == 192
        ]
        if (
            not freeze.get("passed")
            or freeze.get("stage") != 35
            or freeze.get("objective_revision") != STAGE35_OBJECTIVE_REVISION
            or freeze.get("plan_sha256") != STAGE35_PLAN_SHA256
            or len(matches) != 1
        ):
            raise RuntimeError("Stage36 NCD192 control freeze drifted")
        return matches[0]["sha256"], f"stage35_pilot_ncd192_fold{fold}"
    step = int(system.removeprefix("RGT"))
    freeze_path = (
        root.parent
        / "training"
        / "RGT"
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
        or freeze.get("stage") != 36
        or freeze.get("objective_revision") != STAGE36_OBJECTIVE_REVISION
        or freeze.get("plan_sha256") != STAGE36_PLAN_SHA256
        or len(matches) != 1
    ):
        raise RuntimeError(f"Stage36 RGT{step} checkpoint freeze drifted")
    return matches[0]["sha256"], f"stage36_pilot_rgt{step}_fold{fold}"


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
        raise RuntimeError(f"Stage36 artifact provenance drifted: {path}")

    tokens, logs, selected, candidate_mean = [], [], [], []
    candidate_rewards, raw_oracle, safe_oracle = [], [], []
    selected_components = {name: [] for name in COMPONENTS}
    for record in raw_records:
        rewards = np.asarray(record["candidate_rewards"], dtype=np.float64)
        if rewards.shape != (20,) or not np.isfinite(rewards).all():
            raise RuntimeError(f"Stage36 candidate reward shape drifted: {path}")
        stage24 = record["stage24_selector"]
        eligible = np.asarray(stage24["eligible"], dtype=bool)
        fallback = int(stage24["fallback_mode"])
        if eligible.shape != (20,) or not 0 <= fallback < 20:
            raise RuntimeError(f"Stage36 eligibility shape drifted: {path}")
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
        raise RuntimeError(f"Stage36 non-finite compact artifact: {path}")
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
    parser.add_argument("--baseline-eval-root", type=Path, required=True)
    parser.add_argument("--ncd-eval-root", type=Path, required=True)
    parser.add_argument("--bucket-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.bucket_manifest) != BUCKET_SHA256:
        raise RuntimeError("Stage36 bucket manifest SHA drifted")
    bucket_payload = json.loads(args.bucket_manifest.read_text())
    bucket_map = {
        token: int(record["bucket_id"])
        for token, record in bucket_payload["tokens"].items()
    }
    systems = ("P", "DPEL192", "NCD192") + tuple(f"RGT{step}" for step in STEPS)
    loaded = {}
    checkpoint_shas = {}
    artifact_shas = {}
    for system in systems:
        artifact_root = (
            args.baseline_eval_root if system in {"P", "DPEL192"}
            else args.ncd_eval_root if system == "NCD192"
            else args.eval_root
        )
        for fold in (0, 1):
            checkpoint_sha, domain = expected_checkpoint(
                artifact_root, system, fold
            )
            checkpoint_shas[f"{system}/fold{fold}"] = checkpoint_sha
            expected_count = 1019 if fold == 0 else 1021
            for noise in NOISES:
                path = (
                    artifact_root
                    / f"fold{fold}"
                    / f"{system}_ns{noise}.json"
                )
                artifact_shas[str(path.resolve())] = sha256(path)
                loaded[(system, fold, noise)] = load_artifact(
                    path, checkpoint_sha, domain, noise, expected_count
                )

    comparisons = {}
    for step in STEPS:
        system = f"RGT{step}"
        vectors = defaultdict(list)
        component_vectors = {name: [] for name in COMPONENTS}
        logs, fold_ids, noise_ids, bucket_ids = [], [], [], []
        dpel_selected, dpel_candidate, ncd_selected = [], [], []
        for fold in (0, 1):
            for noise in NOISES:
                public = loaded[("P", fold, noise)]
                current = loaded[(system, fold, noise)]
                dpel = loaded[("DPEL192", fold, noise)]
                ncd = loaded[("NCD192", fold, noise)]
                if not (
                    public["tokens"] == current["tokens"] == dpel["tokens"] == ncd["tokens"]
                ):
                    raise RuntimeError("Stage36 paired token order drifted")
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
                ncd_selected.append(current["selected"] - ncd["selected"])
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
            seed=20263500 + step,
        )
        paired_dpel = np.concatenate(dpel_selected)
        summary["paired_vs_DPEL192"] = {
            "mean_selected_delta": float(paired_dpel.mean()),
            "mean_candidate_delta": float(
                np.concatenate(dpel_candidate).mean()
            ),
            "whole_log_bootstrap_ci95": bootstrap_by_log(
                paired_dpel, logs, 20263550 + step
            ),
        }
        paired_ncd = np.concatenate(ncd_selected)
        summary["paired_vs_NCD192"] = {
            "mean_selected_delta": float(paired_ncd.mean()),
            "whole_log_bootstrap_ci95": bootstrap_by_log(
                paired_ncd, logs, 20263650 + step
            ),
        }
        # The two frozen namespaces are the top-2 replica set per token/mode.
        summary["mean_same_anchor_top2_delta"] = float(
            arrays["candidate"].mean()
        )
        summary["same_anchor_tail_definition"] = (
            "top2_of_two_namespaces_per_mode"
        )
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
            "gain_over_NCD192_at_least_0.00015":
                summary["paired_vs_NCD192"]["mean_selected_delta"] >= 0.00015,
            "candidate_mean_nonnegative":
                summary["mean_candidate_delta"] >= 0,
            "same_anchor_top2_nonnegative":
                summary["mean_same_anchor_top2_delta"] >= 0,
            "hard_selected_gain_at_least_0.0045":
                summary["hard_selected_delta"] >= 0.0045,
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
        "stage": 36,
        "pilot_oof": True,
        "method": "RGT-NCD-GRPO",
        "objective_revision": STAGE36_OBJECTIVE_REVISION,
        "plan_sha256": STAGE36_PLAN_SHA256,
        "public_checkpoint_sha256": STAGE30_PUBLIC_SHA256,
        "selector_checkpoint_sha256": STAGE30_SELECTOR_SHA256,
        "calibration_sha256": STAGE30_CALIBRATION_SHA256,
        "eval_root": str(args.eval_root.resolve()),
        "baseline_eval_root": str(args.baseline_eval_root.resolve()),
        "ncd_eval_root": str(args.ncd_eval_root.resolve()),
        "input_artifact_shas": artifact_shas,
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
        "stage36_stop_for_no_generator_ceiling": (
            all(
                comparisons[str(step)]["mean_raw_oracle_delta"] < 0
                for step in STEPS
            )
            or all(
                comparisons[str(step)]["mean_public_top5_delta"] < 0
                for step in STEPS
            )
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
