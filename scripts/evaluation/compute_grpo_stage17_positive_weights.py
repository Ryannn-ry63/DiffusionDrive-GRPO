#!/usr/bin/env python3
"""Compute locked Stage-17 positive BCE weights from risk-fit paired labels."""

import argparse
import json
from pathlib import Path

import numpy as np


EVENTS = (
    "base_better", "loss_0p1", "loss_0p5",
    "collision_regression", "drivable_regression", "ttc_regression",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    labels = []
    for record in payload["records"]:
        pair = record["paired_risk"]
        current_rewards = record["candidate_rewards"]
        base_rewards = pair["base_candidate_rewards"]
        current_components = pair["current_candidate_components"]
        base_components = pair["base_candidate_components"]
        for current, base, current_c, base_c in zip(
            current_rewards, base_rewards, current_components, base_components
        ):
            if current is None or base is None:
                continue
            values = np.asarray((current, base, *current_c, *base_c), dtype=float)
            if not np.isfinite(values).all():
                continue
            delta = float(current - base)
            labels.append((
                delta < 0.0,
                delta <= -0.1,
                delta <= -0.5,
                current_c[0] < base_c[0] - 1e-6,
                current_c[1] < base_c[1] - 1e-6,
                current_c[3] < base_c[3] - 1e-6,
            ))
    array = np.asarray(labels, dtype=bool)
    if array.ndim != 2 or array.shape[1] != len(EVENTS):
        raise RuntimeError("paired artifact produced no valid labels")
    positives = array.sum(axis=0)
    negatives = array.shape[0] - positives
    if (positives == 0).any():
        raise RuntimeError("risk-fit has an event without positive examples")
    weights = np.minimum(negatives / positives, 100.0)
    result = {
        "artifact": str(args.artifact.resolve()),
        "num_valid_pairs": int(array.shape[0]),
        "events": {
            name: {
                "positives": int(positives[index]),
                "negatives": int(negatives[index]),
                "positive_weight": float(weights[index]),
            }
            for index, name in enumerate(EVENTS)
        },
        "hydra_list": [float(value) for value in weights],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
