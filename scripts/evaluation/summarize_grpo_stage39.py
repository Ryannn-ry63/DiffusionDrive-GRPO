#!/usr/bin/env python3
"""Summarize Stage39 pilot/formal attribution and freeze the global step."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from navsim.agents.diffusiondrive.stage39_challenger import STAGE39_PLAN_SHA256


SAFETY = (0, 1, 3)
PILOT_STEPS = (48, 96, 192)
PILOT_NOISES = (20261711, 20261712)
FORMAL_NOISES = (20261721, 20261722)
FORMAL_FOLDS = (0, 1, 3)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path, expected_source: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    summary = payload.get("summary", {})
    records = payload.get("records", [])
    if not (
        summary.get("completed") is True
        and summary.get("num_failures") == 0
        and summary.get("generation_policy_algorithm")
        == "diffgrpo_non_destructive_challenger"
        and summary.get("stage39_candidate_source") == expected_source
        and len(records) == int(summary.get("num_tokens", -1))
    ):
        raise RuntimeError(f"Stage39 evaluation artifact is incomplete: {path}")
    return {"path": str(path.resolve()), "sha256": sha256(path), **payload}


def index(records: list[dict]) -> dict[str, dict]:
    result = {str(record["token"]): record for record in records}
    if len(result) != len(records):
        raise RuntimeError("Stage39 artifact contains duplicate tokens")
    return result


def finite_rewards(record: dict) -> np.ndarray:
    return np.asarray(
        [np.nan if value is None else float(value) for value in record["candidate_rewards"]],
        dtype=np.float64,
    )


def exact_public20(base: dict, arm: dict) -> bool:
    return (
        arm["candidate_rewards"][:20] == base["candidate_rewards"]
        and arm["candidate_components"][:20] == base["candidate_components"]
        and arm["candidate_trajectories"][:20] == base["candidate_trajectories"]
        and arm["candidate_reference_logits"][:20]
        == base["candidate_reference_logits"][:20]
    )


def record_metrics(record: dict) -> dict[str, float]:
    rewards = finite_rewards(record)
    components = np.asarray(record["candidate_components"], dtype=np.float64)
    if rewards.size not in (20, 40) or components.shape != (rewards.size, 6):
        raise RuntimeError("Stage39 candidate bank shape drifted")
    public_valid = np.isfinite(rewards[:20]) & np.isfinite(components[:20]).all(axis=1)
    if int(public_valid.sum()) < 5:
        raise RuntimeError("Stage39 public bank has fewer than five valid candidates")
    public_ids = np.flatnonzero(public_valid)
    elite_ids = public_ids[np.argsort(-rewards[public_ids], kind="stable")[:5]]
    public_utility = float(rewards[elite_ids].mean())
    safety_floor = components[elite_ids][:, SAFETY].min(axis=0)
    union_top5 = public_utility
    safe_top5 = public_utility
    positive = 0
    catastrophic = 0
    extra_count = max(rewards.size - 20, 0)
    if extra_count:
        extra_rewards = rewards[20:]
        extra_components = components[20:]
        extra_valid = np.isfinite(extra_rewards) & np.isfinite(extra_components).all(axis=1)
        catastrophic_mask = (~extra_valid) | np.any(
            np.nan_to_num(extra_components[:, SAFETY], nan=0.0) <= 0.0, axis=1
        )
        catastrophic = int(catastrophic_mask.sum())
        all_valid = np.concatenate((rewards[public_ids], extra_rewards[extra_valid]))
        union_top5 = float(np.sort(all_valid)[-5:].mean())
        strict_safe = extra_valid & np.all(
            extra_components[:, SAFETY] >= safety_floor[None, :] - 1e-6,
            axis=1,
        )
        safe_values = np.concatenate((rewards[elite_ids], extra_rewards[strict_safe]))
        safe_top5 = float(np.sort(safe_values)[-5:].mean())
        for reward, safe in zip(extra_rewards, strict_safe):
            if not safe:
                continue
            marginal = float(
                np.sort(np.concatenate((rewards[elite_ids], [reward])))[-5:].mean()
                - public_utility
            )
            positive += int(marginal > 0.001)
    return {
        "public_top5": public_utility,
        "union_top5": union_top5,
        "safe_union_top5": safe_top5,
        "union_top5_gain": union_top5 - public_utility,
        "safe_union_gain": safe_top5 - public_utility,
        "selected_reward": float(record["selected_reward"]),
        "positive_fraction": positive / max(extra_count, 1),
        "catastrophic_rate": catastrophic / max(extra_count, 1),
    }


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values or not np.isfinite(values).all():
        raise RuntimeError("Stage39 aggregate contains missing/non-finite values")
    return float(np.mean(values))


def bootstrap_ci(values: list[float], seed: int, samples: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("Stage39 bootstrap input is empty/non-finite")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        size = min(1000, samples - start)
        ids = rng.integers(0, array.size, size=(size, array.size))
        means[start : start + size] = array[ids].mean(axis=1)
    return np.quantile(means, (0.025, 0.975)).tolist()


def cell(eval_root: Path, fold: int, noise: int, step: int) -> dict:
    folder = eval_root / f"fold{fold}"
    artifacts = {
        "P20": load(folder / f"P20_ns{noise}.json", "public20"),
        "P40": load(folder / f"P40_ns{noise}.json", "public40_extra"),
        "BC": load(folder / f"BC{step}_ns{noise}.json", "challenger_union"),
        "STD": load(folder / f"STD{step}_ns{noise}.json", "challenger_union"),
        "SET": load(folder / f"SET{step}_ns{noise}.json", "challenger_union"),
    }
    maps = {name: index(artifact["records"]) for name, artifact in artifacts.items()}
    tokens = list(maps["P20"])
    if any(set(mapping) != set(tokens) for mapping in maps.values()):
        raise RuntimeError(f"Stage39 token alignment drifted fold={fold} noise={noise}")
    rows = []
    public_exact = True
    for token in tokens:
        base = maps["P20"][token]
        metrics = {name: record_metrics(mapping[token]) for name, mapping in maps.items()}
        for name in ("P40", "BC", "STD", "SET"):
            public_exact &= exact_public20(base, maps[name][token])
        rows.append({"token": token, **metrics})
    arm_summary = {}
    for name in artifacts:
        arm_summary[name] = {
            metric: mean(row[name][metric] for row in rows)
            for metric in (
                "public_top5", "union_top5", "safe_union_top5",
                "union_top5_gain", "safe_union_gain", "selected_reward",
                "positive_fraction", "catastrophic_rate",
            )
        }
    deltas = {
        "sampling_effect": arm_summary["P40"]["safe_union_top5"]
        - arm_summary["P20"]["safe_union_top5"],
        "architecture_bc_effect": arm_summary["BC"]["safe_union_top5"]
        - arm_summary["P40"]["safe_union_top5"],
        "standard_grpo_effect": arm_summary["STD"]["safe_union_top5"]
        - arm_summary["BC"]["safe_union_top5"],
        "set_credit_effect": arm_summary["SET"]["safe_union_top5"]
        - arm_summary["STD"]["safe_union_top5"],
        "set_vs_public20": arm_summary["SET"]["safe_union_top5"]
        - arm_summary["P20"]["safe_union_top5"],
        "set_vs_public40": arm_summary["SET"]["safe_union_top5"]
        - arm_summary["P40"]["safe_union_top5"],
        "set_vs_bc": arm_summary["SET"]["safe_union_top5"]
        - arm_summary["BC"]["safe_union_top5"],
        "selected_set_vs_public20": arm_summary["SET"]["selected_reward"]
        - arm_summary["P20"]["selected_reward"],
    }
    return {
        "fold": fold,
        "noise": noise,
        "step": step,
        "num_tokens": len(tokens),
        "public20_exact": bool(public_exact),
        "artifacts": {
            name: {"path": artifact["path"], "sha256": artifact["sha256"]}
            for name, artifact in artifacts.items()
        },
        "arms": arm_summary,
        "deltas": deltas,
        "rows": rows,
    }


def aggregate(cells: list[dict], bootstrap_samples: int) -> dict:
    arms = ("P20", "P40", "BC", "STD", "SET")
    metrics = (
        "public_top5", "union_top5", "safe_union_top5", "union_top5_gain",
        "safe_union_gain", "selected_reward", "positive_fraction",
        "catastrophic_rate",
    )
    arm_summary = {
        arm: {
            metric: mean(
                row[arm][metric] for item in cells for row in item["rows"]
            )
            for metric in metrics
        }
        for arm in arms
    }
    comparisons = {}
    for label, left, right in (
        ("sampling_effect", "P40", "P20"),
        ("architecture_bc_effect", "BC", "P40"),
        ("standard_grpo_effect", "STD", "BC"),
        ("set_credit_effect", "SET", "STD"),
        ("set_vs_public20", "SET", "P20"),
        ("set_vs_public40", "SET", "P40"),
        ("set_vs_bc", "SET", "BC"),
    ):
        values = [
            row[left]["safe_union_top5"] - row[right]["safe_union_top5"]
            for item in cells for row in item["rows"]
        ]
        comparisons[label] = {
            "mean": mean(values),
            "bootstrap_95_ci": bootstrap_ci(
                values, 203900 + len(comparisons), bootstrap_samples
            ),
        }
    selected_values = [
        row["SET"]["selected_reward"] - row["P20"]["selected_reward"]
        for item in cells for row in item["rows"]
    ]
    comparisons["selected_set_vs_public20"] = {
        "mean": mean(selected_values),
        "bootstrap_95_ci": bootstrap_ci(
            selected_values, 203999, bootstrap_samples
        ),
    }
    by_namespace = {}
    for noise in sorted({item["noise"] for item in cells}):
        selected = [item for item in cells if item["noise"] == noise]
        by_namespace[str(noise)] = {
            label: mean(item["deltas"][label] for item in selected)
            for label in selected[0]["deltas"]
        }
    return {
        "public20_exact": all(item["public20_exact"] for item in cells),
        "arms": arm_summary,
        "comparisons": comparisons,
        "by_namespace": by_namespace,
        "cell_summaries": [
            {key: value for key, value in item.items() if key != "rows"}
            for item in cells
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "formal"), required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--selection-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")

    if args.phase == "pilot":
        if args.selection_output is None or args.selection is not None:
            raise ValueError("pilot requires --selection-output only")
        step_results = {}
        for step in PILOT_STEPS:
            cells = [cell(args.eval_root, 2, noise, step) for noise in PILOT_NOISES]
            step_results[step] = aggregate(cells, args.bootstrap_samples)
        # One step for all branches: maximize the worst-namespace Set-vs-BC
        # safe-oracle delta, then Set-vs-P20 headroom, then prefer the earlier step.
        def rank(step: int):
            result = step_results[step]
            worst_set_bc = min(
                values["set_vs_bc"] for values in result["by_namespace"].values()
            )
            return (
                worst_set_bc,
                result["comparisons"]["set_vs_public20"]["mean"],
                -step,
            )
        selected_step = max(PILOT_STEPS, key=rank)
        selected = step_results[selected_step]
        passed = bool(selected["public20_exact"])
        selection_payload = {
            "schema_version": 1,
            "stage": 39,
            "phase": "pilot_selection",
            "passed": passed,
            "plan_sha256": STAGE39_PLAN_SHA256,
            "selection_rule": "maximin_set_vs_bc_then_set_vs_public20_then_earlier",
            "selected_step": selected_step,
            "pilot_sanity": {
                "public20_exact": selected["public20_exact"],
                "extra_sampling_safe_oracle_at_least_0.005": (
                    selected["comparisons"]["sampling_effect"]["mean"] >= 0.005
                ),
                "set_safe_oracle_positive": (
                    selected["comparisons"]["set_vs_public20"]["mean"] > 0
                ),
            },
            "step_ranks": {str(step): list(rank(step)) for step in PILOT_STEPS},
        }
        if args.selection_output.exists():
            raise FileExistsError(f"refusing to overwrite {args.selection_output}")
        args.selection_output.parent.mkdir(parents=True, exist_ok=True)
        args.selection_output.write_text(json.dumps(selection_payload, indent=2) + "\n")
        result = {
            "schema_version": 1,
            "stage": 39,
            "phase": "pilot",
            "passed": passed,
            "plan_sha256": STAGE39_PLAN_SHA256,
            "selected_step": selected_step,
            "selection": selection_payload,
            "steps": {str(step): value for step, value in step_results.items()},
        }
    else:
        if args.selection is None or args.selection_output is not None:
            raise ValueError("formal requires --selection only")
        selection = json.loads(args.selection.read_text())
        step = int(selection.get("selected_step", -1))
        if not (
            selection.get("passed") is True
            and selection.get("plan_sha256") == STAGE39_PLAN_SHA256
            and step in PILOT_STEPS
        ):
            raise RuntimeError("Stage39 pilot selection is invalid")
        cells = [
            cell(args.eval_root, fold, noise, step)
            for fold in FORMAL_FOLDS for noise in FORMAL_NOISES
        ]
        summary = aggregate(cells, args.bootstrap_samples)
        comp = summary["comparisons"]
        every_cell_set_positive = all(
            item["deltas"]["set_vs_public20"] > 0
            for item in summary["cell_summaries"]
        )
        std_both_namespaces_positive = all(
            values["standard_grpo_effect"] > 0
            for values in summary["by_namespace"].values()
        )
        checks = {
            "public20_bitwise_exact": summary["public20_exact"],
            "std_minus_bc_both_namespaces_positive": std_both_namespaces_positive,
            "std_minus_bc_ci_lower_positive": (
                comp["standard_grpo_effect"]["bootstrap_95_ci"][0] > 0
            ),
            "set_safe_oracle_vs_public20_at_least_0.012": (
                comp["set_vs_public20"]["mean"] >= 0.012
            ),
            "set_vs_public40_at_least_0.001": (
                comp["set_vs_public40"]["mean"] >= 0.001
            ),
            "set_vs_bc_at_least_0.001": comp["set_vs_bc"]["mean"] >= 0.001,
            "set_vs_std_at_least_0.0005": (
                comp["set_credit_effect"]["mean"] >= 0.0005
            ),
            "every_fold_namespace_set_positive": every_cell_set_positive,
            "catastrophic_rate_no_more_than_public40_plus_0.001": (
                summary["arms"]["SET"]["catastrophic_rate"]
                <= summary["arms"]["P40"]["catastrophic_rate"] + 0.001
            ),
        }
        result = {
            "schema_version": 1,
            "stage": 39,
            "phase": "formal",
            "passed": all(checks.values()),
            "plan_sha256": STAGE39_PLAN_SHA256,
            "selected_step": step,
            "selection_path": str(args.selection.resolve()),
            "selection_sha256": sha256(args.selection),
            "checks": checks,
            "summary": summary,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
