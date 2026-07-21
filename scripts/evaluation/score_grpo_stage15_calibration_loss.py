#!/usr/bin/env python3
"""Recompute the locked Stage-15 selector loss from a calibration artifact."""

import argparse
import json
from pathlib import Path

import torch

from navsim.agents.diffusiondrive.trajectory_value_selector import (
    compute_value_selector_loss,
    deterministic_bootstrap_mask,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=918)
    parser.add_argument("--reward-gap", type=float, default=0.01)
    parser.add_argument("--bootstrap-fraction", type=float, default=0.8)
    args = parser.parse_args()

    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if len(records) != args.expected_count:
        raise RuntimeError(
            f"Calibration artifact must have {args.expected_count} records; "
            f"got {len(records)}"
        )
    if payload.get("summary", {}).get("selector_logits_source") != "value_top2":
        raise RuntimeError("Calibration artifact must use value_top2")

    tokens = [str(record["token"]) for record in records]
    value = [record.get("value_selector") for record in records]
    if any(item is None for item in value):
        raise RuntimeError("Calibration artifact lacks value-selector diagnostics")
    component_predictions = torch.tensor(
        [item["component_predictions"] for item in value], dtype=torch.float32
    )
    score_predictions = torch.tensor(
        [item["score_predictions"] for item in value], dtype=torch.float32
    )
    component_targets = torch.tensor(
        [item["candidate_components"] for item in value], dtype=torch.float32
    )
    reference_logits = torch.tensor(
        [item["reference_logits"] for item in value], dtype=torch.float32
    )
    rewards_list = [record["candidate_rewards"] for record in records]
    valid = torch.tensor(
        [[reward is not None for reward in rewards] for rewards in rewards_list],
        dtype=torch.bool,
    )
    rewards = torch.tensor(
        [[0.0 if reward is None else reward for reward in row] for row in rewards_list],
        dtype=torch.float32,
    )
    heads = component_predictions.shape[2]
    bootstrap = deterministic_bootstrap_mask(
        tokens, heads, torch.device("cpu"), args.bootstrap_fraction
    )
    losses = compute_value_selector_loss(
        {
            "value_component_predictions": component_predictions,
            "value_score_predictions": score_predictions,
            "component_scores": component_targets,
            "raw_rewards": rewards,
            "reward_valid_mask": valid,
            "value_bootstrap_mask": bootstrap,
            "value_reference_logits": reference_logits,
            "value_group_size": rewards.shape[1],
        },
        reward_gap=args.reward_gap,
    )

    result = {
        "artifact": str(args.artifact),
        "num_tokens": len(records),
        "reward_gap": args.reward_gap,
        "bootstrap_fraction": args.bootstrap_fraction,
        "losses": {
            **{name: float(value.item()) for name, value in losses.items()},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
