#!/usr/bin/env python3
"""Select a Stage-17 checkpoint by the locked risk-validation AUPRC metric."""

import argparse
import json
import re
from pathlib import Path

import numpy as np


EVENTS = (
    "base_better", "loss_0p1", "loss_0p5",
    "collision_regression", "drivable_regression", "ttc_regression",
)


def average_precision(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    if positives == 0 or not np.isfinite(scores).all():
        raise RuntimeError("AUPRC requires finite scores and positive labels")
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float(precision[ranked].sum() / positives)


def artifact_metrics(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    all_labels, all_scores = [], []
    for record in payload["records"]:
        pair = record["paired_risk"]
        for current, base, current_c, base_c, scores in zip(
            record["candidate_rewards"], pair["base_candidate_rewards"],
            pair["current_candidate_components"], pair["base_candidate_components"],
            pair["event_probabilities_all"],
        ):
            if current is None or base is None:
                continue
            values = np.asarray((current, base, *current_c, *base_c, *scores), dtype=float)
            if not np.isfinite(values).all():
                continue
            delta = float(current - base)
            all_labels.append((
                delta < 0.0, delta <= -0.1, delta <= -0.5,
                current_c[0] < base_c[0] - 1e-6,
                current_c[1] < base_c[1] - 1e-6,
                current_c[3] < base_c[3] - 1e-6,
            ))
            all_scores.append(scores)
    labels = np.asarray(all_labels, dtype=bool)
    scores = np.asarray(all_scores, dtype=float)
    aps = [average_precision(labels[:, i], scores[:, i]) for i in range(6)]
    metric = 0.5 * float(np.mean(aps)) + 0.5 * float(np.min(aps))
    match = re.search(r"epoch(\d+)", path.stem)
    if not match:
        raise ValueError(f"artifact name must contain epochNN: {path}")
    return {
        "artifact": str(path.resolve()),
        "epoch": int(match.group(1)),
        "checkpoint": payload["summary"]["paired_risk"]["checkpoint"],
        "checkpoint_sha256": payload["summary"]["paired_risk"]["checkpoint_sha256"],
        "per_event_auprc": dict(zip(EVENTS, aps)),
        "selection_metric": metric,
        "num_pairs": int(labels.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = sorted((artifact_metrics(path) for path in args.artifacts), key=lambda x: x["epoch"])
    best_metric = max(item["selection_metric"] for item in candidates)
    eligible = [item for item in candidates if best_metric - item["selection_metric"] <= 0.005]
    selected = min(eligible, key=lambda item: item["epoch"])
    by_epoch = {item["epoch"]: item for item in candidates}
    extend_to_24 = (
        selected["epoch"] == 12
        and 10 in by_epoch
        and selected["selection_metric"] - by_epoch[10]["selection_metric"] >= 0.005
    )
    result = {
        "selection_rule": "0.5*macro_AUPRC+0.5*minimum_AUPRC; within_0.005_earlier",
        "selected": selected,
        "extend_to_epoch24": extend_to_24,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
