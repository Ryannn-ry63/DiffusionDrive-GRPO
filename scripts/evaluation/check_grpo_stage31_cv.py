#!/usr/bin/env python3
"""Apply the frozen Stage31 four-fold OOF DPF selection gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from navsim.agents.diffusiondrive.stage30_mode_coverage import (
    STAGE30_BUCKET_NAMES,
    STAGE30_CALIBRATION_SHA256,
    STAGE30_PUBLIC_SHA256,
    STAGE30_SELECTOR_SHA256,
    load_stage30_bucket_manifest,
)

BRANCHES = ("DP", "DPF")
STEPS = (48, 96, 144, 192)
NOISES = (20261411, 20261412)
COMPONENTS = ("collision", "drivable", "progress", "ttc", "comfort", "direction")
BOOTSTRAP_SEED = 20261430
BOOTSTRAP_SAMPLES = 10000
PLAN_SHA256 = "3ff3d3bcd3120b9d73a017876f942458eb3fa16651517e4e30b27cf2fff7bb0d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_whole_log(values: np.ndarray, logs: list[str]) -> list[float]:
    grouped = defaultdict(list)
    for value, log_name in zip(values, logs):
        grouped[log_name].append(float(value))
    if len(grouped) != 606:
        raise RuntimeError(f"Stage31 OOF log count drifted: {len(grouped)}")
    names = sorted(grouped)
    sums = np.asarray([sum(grouped[name]) for name in names], dtype=np.float64)
    counts = np.asarray([len(grouped[name]) for name in names], dtype=np.int64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_SAMPLES, 500):
        size = min(500, BOOTSTRAP_SAMPLES - start)
        indices = rng.integers(0, len(names), (size, len(names)))
        samples[start:start + size] = (
            sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
        )
    return np.quantile(samples, (0.025, 0.975)).tolist()


def trim10(values: np.ndarray) -> float:
    ordered = np.sort(values)
    amount = int(math.floor(0.1 * ordered.size))
    return float(ordered[amount:-amount].mean())


def load_artifact(
    path: Path,
    expected_sha: str,
    noise: int,
    tokens: list[str],
    expected_domain: str,
) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    selector = summary["stage25_selector"]
    records = payload["records"]
    checks = (
        summary.get("completed") is True,
        summary.get("num_failures") == 0,
        summary.get("num_tokens") == len(tokens) == len(records),
        summary.get("checkpoint_sha256") == expected_sha,
        summary.get("reference_checkpoint_sha256") == STAGE30_PUBLIC_SHA256,
        summary.get("evaluation_noise_namespace") == noise,
        summary.get("generator_domain") == expected_domain,
        summary.get("selector_logits_source") == "trajectory_relative_harm_v3",
        selector.get("checkpoint_sha256") == STAGE30_SELECTOR_SHA256,
        selector.get("calibration_sha256") == STAGE30_CALIBRATION_SHA256,
        [str(record["token"]) for record in records] == tokens,
    )
    if not all(checks):
        raise RuntimeError(f"Stage31 artifact provenance drifted: {path}")
    result = {}
    for record in records:
        token = str(record["token"])
        rewards = np.asarray(record["candidate_rewards"], dtype=np.float64)
        components = np.asarray(record["candidate_components"], dtype=np.float64)
        selected = int(record["selected_mode"])
        selected_components = np.asarray(
            [record["selected_components"][name] for name in COMPONENTS],
            dtype=np.float64,
        )
        if not (
            rewards.shape == (20,) and components.shape == (20, 6)
            and 0 <= selected < 20 and np.isfinite(rewards).all()
            and np.isfinite(components).all() and np.isfinite(selected_components).all()
            and np.isclose(record["selected_reward"], rewards[selected], atol=2e-6, rtol=0)
            and np.allclose(selected_components, components[selected], atol=2e-6, rtol=0)
        ):
            raise RuntimeError(f"Stage31 invalid record: {path}/{token}")
        result[token] = record
    return result


def summarize(branch, step, vectors, buckets, folds, noises, logs, component_values):
    selected = np.asarray(vectors["selected"], dtype=np.float64)
    candidate_mean = np.asarray(vectors["candidate_mean"], dtype=np.float64)
    oracle = np.asarray(vectors["oracle"], dtype=np.float64)
    fallback = np.asarray(vectors["fallback"], dtype=np.float64)
    bucket_ids = np.asarray(buckets, dtype=np.int64)
    fold_ids = np.asarray(folds, dtype=np.int64)
    noise_ids = np.asarray(noises, dtype=np.int64)
    if not all(np.isfinite(value).all() for value in (selected, candidate_mean, oracle, fallback)):
        raise RuntimeError(f"Stage31 non-finite summary vector: {branch}/{step}")
    hard = bucket_ids <= 1
    mature = bucket_ids == 3
    if not hard.any() or not mature.any():
        raise RuntimeError("Stage31 OOF hard/mature bucket is empty")
    ci = bootstrap_whole_log(selected, logs)
    fold_means = {
        f"fold{fold}": float(selected[fold_ids == fold].mean())
        for fold in range(4)
    }
    fold_mature = {
        f"fold{fold}": float(selected[(fold_ids == fold) & mature].mean())
        for fold in range(4)
    }
    namespace_means = {
        f"ns{noise}": float(selected[noise_ids == noise].mean())
        for noise in NOISES
    }
    component_means = {
        name: float(np.mean(component_values[name])) for name in COMPONENTS
    }
    pooled = float(selected.mean())
    hard_mean = float(selected[hard].mean())
    mature_mean = float(selected[mature].mean())
    wins = int(np.sum(selected > 0)); losses = int(np.sum(selected < 0))
    catastrophes = int(np.sum(selected <= -0.5))
    checks = {
        "pooled_gain_at_least_0.003": pooled >= 0.003,
        "whole_log_ci_lower_strictly_positive": ci[0] > 0.0,
        "both_namespace_means_strictly_positive": all(x > 0 for x in namespace_means.values()),
        "no_fold_below_negative_0.0005": all(x >= -0.0005 for x in fold_means.values()),
        "hard_scene_gain_at_least_0.010": hard_mean >= 0.010,
        "mature_scene_gain_at_least_negative_0.0001": mature_mean >= -0.0001,
        "no_fold_mature_below_negative_0.0005": all(x >= -0.0005 for x in fold_mature.values()),
        "candidate_mean_delta_at_least_negative_0.0005": float(candidate_mean.mean()) >= -0.0005,
        "candidate_max_oracle_delta_at_least_negative_0.0005": float(oracle.mean()) >= -0.0005,
        "selected_fallback_oracle_at_least_0.004": float(fallback.mean()) >= 0.004,
        "trimmed10_mean_nonnegative": trim10(selected) >= 0.0,
        "wins_exceed_losses": wins > losses,
        "all_components_at_least_negative_0.0005": all(x >= -0.0005 for x in component_means.values()),
        "catastrophic_count_zero": catastrophes == 0,
        "finite_and_provenance_complete": True,
    }
    selectable = branch == "DPF"
    return {
        "branch": branch, "step": step, "epoch": step // 48,
        "selectable": selectable, "pooled_mean_delta": pooled,
        "whole_log_bootstrap_ci": ci, "trimmed10_mean_delta": trim10(selected),
        "namespace_mean_delta": namespace_means, "fold_mean_delta": fold_means,
        "candidate_mean_delta": float(candidate_mean.mean()),
        "candidate_max_oracle_mean_delta": float(oracle.mean()),
        "selected_fallback_oracle_mean_delta": float(fallback.mean()),
        "hard_scene_count": int(hard.sum()), "hard_scene_mean_delta": hard_mean,
        "mature_scene_count": int(mature.sum()), "mature_scene_mean_delta": mature_mean,
        "fold_mature_mean_delta": fold_mature,
        "component_mean_deltas": component_means,
        "wins": wins, "exact_ties": int(np.sum(selected == 0)), "losses": losses,
        "worst_delta": float(selected.min()), "catastrophic_count": catastrophes,
        "checks": checks, "passes_numeric_gate": bool(all(checks.values())),
        "eligible": bool(selectable and all(checks.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/grpo_stage31/cv/report.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    plan_path = root / "GRPO_STAGE31_SELECTOR_CONSISTENT_DECISION_GRPO_PLAN_20260725.md"
    if sha256(plan_path) != PLAN_SHA256:
        raise RuntimeError("Stage31 frozen plan SHA drifted")
    cv_path = root / "artifacts/grpo_stage30/manifests/cv/freeze.json"
    cv = json.loads(cv_path.read_text(encoding="utf-8"))
    if cv.get("stage") != 30:
        raise RuntimeError("Stage31 must reuse the frozen Stage30 CV manifests")
    token_buckets = load_stage30_bucket_manifest(
        cv["bucket_manifest"], require_full=True
    )
    artifacts = {}
    provenance = {}
    checkpoint_metadata = {}
    fold_tokens, fold_logs = {}, {}
    for fold in range(4):
        entry = cv["folds"][fold]
        manifest_path = Path(entry["holdout_manifest"])
        if sha256(manifest_path) != entry["holdout_manifest_sha256"]:
            raise RuntimeError(f"Stage31 heldout fold{fold} manifest drifted")
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))["records"]
        fold_tokens[fold] = [str(row["token"]) for row in rows]
        fold_logs[fold] = {str(row["token"]): str(row["log_name"]) for row in rows}
        for branch in BRANCHES:
            audit_path = root / f"artifacts/grpo_stage31/cv/training/{branch}/fold{fold}/formal/audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if not (
                audit.get("passed") and audit.get("stage") == 31
                and audit.get("phase") == "formal" and audit.get("branch") == branch
                and audit.get("holdout_fold") == fold
                and audit.get("cv_freeze_sha256") == sha256(cv_path)
            ):
                raise RuntimeError(f"Stage31 formal audit drifted: {branch}/fold{fold}")
            checkpoint_metadata[(branch, fold)] = {
                item["global_step"]: item for item in audit["checkpoints"]
            }
            for step in STEPS:
                checkpoint = checkpoint_metadata[(branch, fold)][step]
                for noise in NOISES:
                    path = root / f"artifacts/grpo_stage31/cv/eval/{branch}/fold{fold}/{branch}{step}_ns{noise}.json"
                    artifacts[(branch, step, fold, noise)] = load_artifact(
                        path, checkpoint["sha256"], noise, fold_tokens[fold],
                        f"stage31_cv_{branch.lower()}{step}_fold{fold}",
                    )
                    provenance[f"{branch}{step}:fold{fold}:ns{noise}"] = {
                        "path": str(path), "sha256": sha256(path),
                    }
        for noise in NOISES:
            path = root / f"artifacts/grpo_stage31/cv/eval/DPF/fold{fold}/P_ns{noise}.json"
            artifacts[("P", 0, fold, noise)] = load_artifact(
                path, STAGE30_PUBLIC_SHA256, noise, fold_tokens[fold],
                f"stage31_cv_p_fold{fold}",
            )
            provenance[f"P:fold{fold}:ns{noise}"] = {
                "path": str(path), "sha256": sha256(path),
            }

    summaries = {}
    for branch in BRANCHES:
        for step in STEPS:
            vectors = {name: [] for name in ("selected", "candidate_mean", "oracle", "fallback")}
            buckets, folds, noises, logs = [], [], [], []
            components = {name: [] for name in COMPONENTS}
            for fold in range(4):
                for noise in NOISES:
                    base_map = artifacts[("P", 0, fold, noise)]
                    current_map = artifacts[(branch, step, fold, noise)]
                    for token in fold_tokens[fold]:
                        base, current = base_map[token], current_map[token]
                        base_score = float(base["selected_reward"])
                        current_score = float(current["selected_reward"])
                        base_candidates = np.asarray(base["candidate_rewards"], dtype=np.float64)
                        current_candidates = np.asarray(current["candidate_rewards"], dtype=np.float64)
                        delta = current_score - base_score
                        vectors["selected"].append(delta)
                        vectors["candidate_mean"].append(float(current_candidates.mean() - base_candidates.mean()))
                        vectors["oracle"].append(float(current_candidates.max() - base_candidates.max()))
                        vectors["fallback"].append(max(current_score, base_score) - base_score)
                        buckets.append(token_buckets[token]); folds.append(fold); noises.append(noise)
                        logs.append(fold_logs[fold][token])
                        for name in COMPONENTS:
                            components[name].append(
                                float(current["selected_components"][name])
                                - float(base["selected_components"][name])
                            )
            key = f"{branch}{step}"
            summaries[key] = summarize(
                branch, step, vectors, buckets, folds, noises, logs, components
            )
    eligible = [
        summaries[f"DPF{step}"] for step in STEPS
        if summaries[f"DPF{step}"]["eligible"]
    ]
    selected = max(
        eligible,
        key=lambda item: (
            item["whole_log_bootstrap_ci"][0],
            min(item["fold_mean_delta"].values()),
            item["pooled_mean_delta"],
            -item["step"],
        ),
    ) if eligible else None
    selected_checkpoints = None
    if selected:
        selected_checkpoints = {
            f"fold{fold}": checkpoint_metadata[("DPF", fold)][selected["step"]]
            for fold in range(4)
        }
    result = {
        "schema_version": 1, "stage": 31, "phase": "four_fold_oof",
        "passed": selected is not None, "stop_before_confirmation": selected is None,
        "cv_freeze": str(cv_path), "cv_freeze_sha256": sha256(cv_path),
        "plan": str(plan_path), "plan_sha256": PLAN_SHA256,
        "gate_definition": "stage31_selector_consistent_decision_oof_v1",
        "bootstrap": {"unit": "whole_log", "seed": BOOTSTRAP_SEED, "samples": BOOTSTRAP_SAMPLES},
        "selected_branch": "DPF" if selected else None,
        "selected_step": selected["step"] if selected else None,
        "selected_epoch": selected["epoch"] if selected else None,
        "selected_checkpoints": selected_checkpoints,
        "eligible_steps": [item["step"] for item in eligible],
        "summaries": summaries, "provenance": provenance,
    }
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if selected is None:
        raise SystemExit("Stage31 stops: no common DPF step passed the frozen OOF gate")


if __name__ == "__main__":
    main()
