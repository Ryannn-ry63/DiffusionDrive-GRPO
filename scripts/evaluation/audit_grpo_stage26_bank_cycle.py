#!/usr/bin/env python3
"""Audit the exact Stage26 15-epoch candidate-bank cycle."""

import argparse
import json
from pathlib import Path


DOMAINS = (
    "official_base",
    "stage16_epoch8",
    "stage19_epoch8",
    "stage21_epoch8",
    "stage25_epoch2",
)
LABELS = ("default", "ns20260811", "ns20260812")
STEPS_PER_EPOCH = 2038


def combination_at_update(update: int) -> str:
    combinations = [
        f"{domain}_{label}.json"
        for domain in DOMAINS
        for label in LABELS
    ]
    epoch = update // STEPS_PER_EPOCH
    return combinations[epoch % len(combinations)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    boundaries = {
        "epoch0_first_update": combination_at_update(0),
        "epoch0_last_update": combination_at_update(STEPS_PER_EPOCH - 1),
        "epoch14_first_update": combination_at_update(14 * STEPS_PER_EPOCH),
        "epoch14_last_update": combination_at_update(
            15 * STEPS_PER_EPOCH - 1
        ),
        "epoch15_wrap_update": combination_at_update(15 * STEPS_PER_EPOCH),
    }
    expected = {
        "epoch0_first_update": "official_base_default.json",
        "epoch0_last_update": "official_base_default.json",
        "epoch14_first_update": "stage25_epoch2_ns20260812.json",
        "epoch14_last_update": "stage25_epoch2_ns20260812.json",
        "epoch15_wrap_update": "official_base_default.json",
    }
    if boundaries != expected:
        raise RuntimeError("Stage26 candidate-bank cycle boundary mismatch")
    result = {
        "schema_version": 1,
        "stage": 26,
        "passed": True,
        "num_combinations": 15,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "formal_epochs": 15,
        "total_updates": 15 * STEPS_PER_EPOCH,
        "ordered_combinations": [
            f"{domain}_{label}.json"
            for domain in DOMAINS
            for label in LABELS
        ],
        "boundaries": boundaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
