#!/usr/bin/env python3
"""Apply the preregistered Stage38 ESCR kill-test or two-fold OOF gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PUBLIC_SHA = "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
SELECTOR_SHA = "023d7b6b77bb8fa2dc3e779849f3716688abf0796b019416494c80b29615b691"
CALIBRATION_SHA = "fec2a10573e913e4452f8eea92e6334fcd5e6f1104e6244876770973a1f41990"
PLAN_SHA = "8cd826b6ab8a4376c6e4a4fdffc97e96fe00fba00a25a1029d5fa5bcf4a8b027"
BUCKET_SHA = "e067680e9d601929b69b932c59de1f092e3c229d6aeb0bf5ee5aa0408b9a7cf7"
RGT_SHAS = {
    0: "2a2302da415e38aa654c172858d3f89f8112f774a6215d89b9c150112f0dab33",
    1: "0bafe817ebfe98032b3d4e9499eb78d547efcf7230ddc3c83aa4b1bd4e4e7a9b",
}
STEPS = (24, 48, 96)
NOISES = (20261611, 20261612)
COMPONENTS = ("collision", "drivable", "progress", "ttc", "comfort", "direction")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_entry(root: Path, fold: int, step: int) -> dict:
    path = root / f"pilot/training/ESCR/fold{fold}/formal/checkpoints.json"
    freeze = json.loads(path.read_text())
    matches = [
        record for record in freeze.get("checkpoints", [])
        if int(record["global_step"]) == step
    ]
    if not (
        freeze.get("passed") is True
        and freeze.get("stage") == 38
        and freeze.get("phase") == "formal"
        and freeze.get("holdout_fold") == fold
        and freeze.get("plan_sha256") == PLAN_SHA
        and len(matches) == 1
    ):
        raise RuntimeError(f"Stage38 checkpoint freeze drifted: {path}")
    checkpoint = Path(matches[0]["path"])
    if not checkpoint.is_file() or sha256(checkpoint) != matches[0]["sha256"]:
        raise RuntimeError(f"Stage38 checkpoint SHA drifted: {checkpoint}")
    return matches[0]


def load_artifact(
    path: Path,
    *,
    checkpoint_sha: str,
    domain: str,
    noise: int,
) -> dict:
    payload = json.loads(path.read_text())
    summary = payload["summary"]
    records = payload["records"]
    selector = summary["stage25_selector"]
    if not all((
        summary.get("completed") is True,
        summary.get("num_failures") == 0,
        summary.get("checkpoint_sha256") == checkpoint_sha,
        summary.get("reference_checkpoint_sha256") == PUBLIC_SHA,
        summary.get("generator_domain") == domain,
        summary.get("evaluation_noise_namespace") == noise,
        summary.get("generation_policy_algorithm")
        == "diffgrpo_elite_set_counterfactual_repair",
        summary.get("selector_logits_source") == "trajectory_relative_harm_v3",
        selector.get("checkpoint_sha256") == SELECTOR_SHA,
        selector.get("calibration_sha256") == CALIBRATION_SHA,
    )):
        raise RuntimeError(f"Stage38 artifact provenance drifted: {path}")

    tokens: list[str] = []
    logs: list[str] = []
    selected = []
    candidate_rewards = []
    safe_oracle = []
    selected_components = []
    for record in records:
        rewards = np.asarray(record["candidate_rewards"], dtype=np.float64)
        components = np.asarray(record["candidate_components"], dtype=np.float64)
        eligible = np.asarray(record["stage24_selector"]["eligible"], dtype=bool)
        fallback = int(record["stage24_selector"]["fallback_mode"])
        if (
            rewards.shape != (20,)
            or components.shape != (20, 6)
            or eligible.shape != (20,)
            or not np.isfinite(rewards).all()
            or not np.isfinite(components).all()
        ):
            raise RuntimeError(f"Stage38 invalid all-20 record: {path}")
        eligible[fallback] = True
        tokens.append(str(record["token"]))
        logs.append(str(record["log_name"]))
        selected.append(float(record["selected_reward"]))
        candidate_rewards.append(rewards)
        safe_oracle.append(float(rewards[eligible].max()))
        selected_components.append([
            float(record["selected_components"][name]) for name in COMPONENTS
        ])
    reward_bank = np.stack(candidate_rewards)
    return {
        "tokens": tokens,
        "logs": logs,
        "selected": np.asarray(selected),
        "candidate_rewards": reward_bank,
        "candidate_mean": reward_bank.mean(axis=1),
        "raw_oracle": reward_bank.max(axis=1),
        "safe_oracle": np.asarray(safe_oracle),
        "selected_components": np.asarray(selected_components),
    }


def log_bootstrap(values: np.ndarray, logs: list[str], seed: int) -> list[float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for value, log_name in zip(values, logs):
        groups[log_name].append(float(value))
    names = sorted(groups)
    sums = np.asarray([sum(groups[name]) for name in names], dtype=np.float64)
    counts = np.asarray([len(groups[name]) for name in names], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(10000, dtype=np.float64)
    for start in range(0, 10000, 500):
        indices = rng.integers(0, len(names), size=(500, len(names)))
        samples[start:start + 500] = (
            sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
        )
    return np.quantile(samples, (0.025, 0.975)).tolist()


def trimmed_mean(values: np.ndarray, fraction: float = 0.10) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    trim = int(np.floor(fraction * ordered.size))
    retained = ordered[trim:ordered.size - trim] if trim else ordered
    if retained.size == 0:
        raise RuntimeError("Stage38 trimmed mean has no samples")
    return float(retained.mean())


def compare_step(
    *,
    stage38_root: Path,
    bucket_by_token: dict[str, int],
    folds: tuple[int, ...],
    step: int,
) -> tuple[dict, dict[str, str], dict[int, dict]]:
    vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    component_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    artifact_shas: dict[str, str] = {}
    checkpoint_entries: dict[int, dict] = {}
    fold_ids = []
    noise_ids = []
    bucket_ids = []
    log_ids: list[str] = []

    for fold in folds:
        current_entry = checkpoint_entry(stage38_root, fold, step)
        checkpoint_entries[fold] = current_entry
        for noise in NOISES:
            public_path = (
                stage38_root / f"pilot/eval/fold{fold}/P_ns{noise}.json"
            )
            rgt_path = (
                stage38_root / f"pilot/eval/fold{fold}/RGT192_ns{noise}.json"
            )
            current_path = (
                stage38_root
                / f"pilot/eval/fold{fold}/ESCR{step}_ns{noise}.json"
            )
            for path in (public_path, rgt_path, current_path):
                artifact_shas[str(path.resolve())] = sha256(path)
            public = load_artifact(
                public_path,
                checkpoint_sha=PUBLIC_SHA,
                domain=f"stage38_pilot_p_fold{fold}",
                noise=noise,
            )
            rgt = load_artifact(
                rgt_path,
                checkpoint_sha=RGT_SHAS[fold],
                domain=f"stage38_pilot_rgt192_fold{fold}",
                noise=noise,
            )
            current = load_artifact(
                current_path,
                checkpoint_sha=current_entry["sha256"],
                domain=f"stage38_pilot_escr{step}_fold{fold}",
                noise=noise,
            )
            if not public["tokens"] == rgt["tokens"] == current["tokens"]:
                raise RuntimeError("Stage38 paired token order drifted")
            if not public["logs"] == rgt["logs"] == current["logs"]:
                raise RuntimeError("Stage38 paired log order drifted")

            public_order = np.argsort(
                public["candidate_rewards"], axis=1, kind="stable"
            )
            public_top5 = public_order[:, -5:]
            row = np.arange(len(public_top5))[:, None]
            public_top5_utility = (
                public["candidate_rewards"][row, public_top5].mean(axis=1)
            )
            current_top5_utility = (
                current["candidate_rewards"][row, public_top5].mean(axis=1)
            )
            rgt_top5_utility = (
                rgt["candidate_rewards"][row, public_top5].mean(axis=1)
            )
            mode_aligned_union = np.maximum(
                public["candidate_rewards"], current["candidate_rewards"]
            )
            union_top5 = np.sort(mode_aligned_union, axis=1)[:, -5:].mean(axis=1)

            vectors["selected"].append(current["selected"] - public["selected"])
            vectors["vs_rgt"].append(current["selected"] - rgt["selected"])
            vectors["candidate_mean"].append(
                current["candidate_mean"] - public["candidate_mean"]
            )
            vectors["raw_oracle"].append(
                current["raw_oracle"] - public["raw_oracle"]
            )
            vectors["public_top5"].append(
                current_top5_utility - public_top5_utility
            )
            vectors["public_top5_vs_rgt"].append(
                current_top5_utility - rgt_top5_utility
            )
            vectors["union_top5"].append(union_top5 - public_top5_utility)
            vectors["union_safe_oracle"].append(
                np.maximum(current["safe_oracle"], public["safe_oracle"])
                - public["safe_oracle"]
            )
            component_delta = (
                current["selected_components"] - public["selected_components"]
            )
            for index, name in enumerate(COMPONENTS):
                component_vectors[name].append(component_delta[:, index])

            count = len(public["tokens"])
            fold_ids.append(np.full(count, fold, dtype=np.int64))
            noise_ids.append(np.full(count, noise, dtype=np.int64))
            bucket_ids.append(np.asarray([
                bucket_by_token[token] for token in public["tokens"]
            ]))
            # Keep both namespaces from the same physical log in one bootstrap
            # cluster; fold is retained to avoid accidental cross-fold merging.
            log_ids.extend(f"{fold}:{name}" for name in public["logs"])

    values = {name: np.concatenate(parts) for name, parts in vectors.items()}
    component_values = {
        name: np.concatenate(parts) for name, parts in component_vectors.items()
    }
    fold_array = np.concatenate(fold_ids)
    noise_array = np.concatenate(noise_ids)
    bucket_array = np.concatenate(bucket_ids)
    selected_delta = values["selected"]
    wins = int((selected_delta > 0).sum())
    losses = int((selected_delta < 0).sum())
    summary = {
        "count": int(selected_delta.size),
        "mean_selected_delta": float(selected_delta.mean()),
        "mean_selected_delta_vs_stage36_rgt192": float(
            values["vs_rgt"].mean()
        ),
        "mean_candidate_delta": float(values["candidate_mean"].mean()),
        "mean_raw_oracle_delta": float(values["raw_oracle"].mean()),
        "mean_public_top5_delta": float(values["public_top5"].mean()),
        "mean_public_top5_improvement_vs_stage36_rgt192": float(
            values["public_top5_vs_rgt"].mean()
        ),
        "mean_union_top5_utility_delta": float(values["union_top5"].mean()),
        "mean_union_safe_oracle_delta": float(
            values["union_safe_oracle"].mean()
        ),
        "whole_log_bootstrap_ci95": log_bootstrap(
            selected_delta, log_ids, 203800 + step
        ),
        "fold_means": {
            str(fold): float(selected_delta[fold_array == fold].mean())
            for fold in folds
        },
        "namespace_means": {
            str(noise): float(selected_delta[noise_array == noise].mean())
            for noise in NOISES
        },
        "hard_scene_gain": float(selected_delta[bucket_array < 3].mean()),
        "mature_scene_gain": float(selected_delta[bucket_array == 3].mean()),
        "selected_component_deltas": {
            name: float(component_values[name].mean())
            for name in COMPONENTS
        },
        "trimmed10_mean": trimmed_mean(selected_delta),
        "wins": wins,
        "losses": losses,
        "ties": int(selected_delta.size - wins - losses),
        "catastrophic_count": int((selected_delta <= -0.5).sum()),
    }
    if not all(
        np.isfinite(value).all()
        for value in (*values.values(), *component_values.values())
    ):
        raise RuntimeError("Stage38 summary contains non-finite values")
    return summary, artifact_shas, checkpoint_entries


def gates(summary: dict, phase: str) -> dict:
    components = summary["selected_component_deltas"]
    common = {
        "namespace_means_positive": all(
            value > 0 for value in summary["namespace_means"].values()
        ),
        "collision_drivable_ttc_at_least_negative_0.0005": all(
            components[name] >= -0.0005
            for name in ("collision", "drivable", "ttc")
        ),
        "catastrophic_count_zero": summary["catastrophic_count"] == 0,
    }
    if phase == "kill":
        checks = {
            "selected_gain_at_least_0.0015":
                summary["mean_selected_delta"] >= 0.0015,
            "vs_stage36_rgt192_at_least_negative_0.00025":
                summary["mean_selected_delta_vs_stage36_rgt192"] >= -0.00025,
            "public_top5_at_least_negative_0.0005":
                summary["mean_public_top5_delta"] >= -0.0005,
            "public_top5_improves_stage36_by_0.0005":
                summary[
                    "mean_public_top5_improvement_vs_stage36_rgt192"
                ] >= 0.0005,
            "union_safe_oracle_at_least_0.001":
                summary["mean_union_safe_oracle_delta"] >= 0.001,
            "mature_at_least_negative_0.0002":
                summary["mature_scene_gain"] >= -0.0002,
            **common,
        }
    else:
        checks = {
            "selected_gain_at_least_0.002":
                summary["mean_selected_delta"] >= 0.002,
            "whole_log_ci_lower_positive":
                summary["whole_log_bootstrap_ci95"][0] > 0,
            "fold_means_positive": all(
                value > 0 for value in summary["fold_means"].values()
            ),
            "vs_stage36_rgt192_nonnegative":
                summary["mean_selected_delta_vs_stage36_rgt192"] >= 0,
            "public_top5_nonnegative":
                summary["mean_public_top5_delta"] >= 0,
            "candidate_mean_nonnegative":
                summary["mean_candidate_delta"] >= 0,
            "raw_oracle_nonnegative":
                summary["mean_raw_oracle_delta"] >= 0,
            "union_top5_nonnegative":
                summary["mean_union_top5_utility_delta"] >= 0,
            "union_safe_oracle_at_least_0.002":
                summary["mean_union_safe_oracle_delta"] >= 0.002,
            "hard_scene_gain_at_least_0.008":
                summary["hard_scene_gain"] >= 0.008,
            "mature_at_least_negative_0.0001":
                summary["mature_scene_gain"] >= -0.0001,
            "trimmed10_nonnegative": summary["trimmed10_mean"] >= 0,
            "wins_exceed_losses": summary["wins"] > summary["losses"],
            **common,
        }
    checks["passed"] = all(checks.values())
    return checks


def choose_step(passing: list[int], comparisons: dict[str, dict]) -> int | None:
    if not passing:
        return None
    # Preregistered tie-break: selected gain, then union safe oracle if selected
    # gains are within 1e-4, then the earlier optimizer step.
    ordered = sorted(passing)
    best = ordered[0]
    for step in ordered[1:]:
        current = comparisons[str(step)]
        incumbent = comparisons[str(best)]
        gain_gap = (
            current["mean_selected_delta"]
            - incumbent["mean_selected_delta"]
        )
        if gain_gap > 1e-4:
            best = step
        elif abs(gain_gap) <= 1e-4 and (
            current["mean_union_safe_oracle_delta"]
            > incumbent["mean_union_safe_oracle_delta"]
        ):
            best = step
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("kill", "formal"), required=True)
    parser.add_argument("--stage38-root", type=Path, required=True)
    parser.add_argument("--bucket-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.bucket_manifest) != BUCKET_SHA:
        raise RuntimeError("Stage38 bucket manifest drifted")
    bucket_payload = json.loads(args.bucket_manifest.read_text())
    bucket_by_token = {
        token: int(record["bucket_id"])
        for token, record in bucket_payload["tokens"].items()
    }
    folds = (0,) if args.phase == "kill" else (0, 1)
    comparisons = {}
    all_artifact_shas: dict[str, str] = {}
    entries: dict[tuple[int, int], dict] = {}
    for step in STEPS:
        summary, artifact_shas, checkpoint_entries = compare_step(
            stage38_root=args.stage38_root,
            bucket_by_token=bucket_by_token,
            folds=folds,
            step=step,
        )
        summary["checks"] = gates(summary, args.phase)
        comparisons[str(step)] = summary
        all_artifact_shas.update(artifact_shas)
        for fold, entry in checkpoint_entries.items():
            entries[(fold, step)] = entry

    passing = [
        step for step in STEPS if comparisons[str(step)]["checks"]["passed"]
    ]
    selected = choose_step(passing, comparisons)
    result = {
        "schema_version": 1,
        "stage": 38,
        "phase": args.phase,
        "plan_sha256": PLAN_SHA,
        "objective_revision": "elite_set_counterfactual_repair_v1",
        "folds": list(folds),
        "noise_namespaces": list(NOISES),
        "comparisons": comparisons,
        "passing_steps": passing,
        "passed": selected is not None,
        "selected_step": selected,
        "result_class": (
            "fail"
            if selected is None
            else (
                "mainline_pass"
                if args.phase == "formal"
                and comparisons[str(selected)][
                    "mean_union_safe_oracle_delta"
                ] >= 0.005
                else "positive_ablation"
            )
        ),
        "input_artifact_shas": all_artifact_shas,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    if selected is not None:
        selection = {
            "schema_version": 1,
            "stage": 38,
            "phase": args.phase,
            "passed": True,
            "plan_sha256": PLAN_SHA,
            "selected_step": selected,
            "result_class": result["result_class"],
            "generator_gate": comparisons[str(selected)],
            "folds": {
                str(fold): {
                    "checkpoint": entries[(fold, selected)]["path"],
                    "checkpoint_sha256": entries[(fold, selected)]["sha256"],
                }
                for fold in folds
            },
        }
        if args.selection_output.exists():
            raise FileExistsError(
                f"refusing to overwrite {args.selection_output}"
            )
        args.selection_output.parent.mkdir(parents=True, exist_ok=True)
        args.selection_output.write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
