#!/usr/bin/env python3
"""Select the preregistered Stage-16 development epoch on calibration only."""

import argparse
import json
from pathlib import Path

import numpy as np


EPOCHS = (1, 2, 4, 6, 8, 10)
SAFETY = ("collision", "drivable", "ttc")


def load(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = {str(record["token"]): record for record in payload["records"]}
    if len(records) != 918 or len(records) != len(payload["records"]):
        raise ValueError(f"expected 918 unique calibration tokens: {path}")
    summary = payload.get("summary", {})
    if summary.get("num_tokens") != 918 or not summary.get("completed", False):
        raise ValueError(f"incomplete calibration artifact: {path}")
    return summary, records


def paired(left, right, field):
    if set(left) != set(right):
        raise ValueError("candidate/base token mismatch")
    return float(np.mean([
        right[token][field] - left[token][field] for token in sorted(left)
    ]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-original", type=Path, required=True)
    parser.add_argument("--base-full", type=Path, required=True)
    parser.add_argument(
        "--candidate", action="append", required=True, metavar="EPOCH=ARTIFACT"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    parsed = {}
    for item in args.candidate:
        epoch_text, separator, path_text = item.partition("=")
        if not separator:
            raise ValueError("candidate must use EPOCH=ARTIFACT")
        epoch = int(epoch_text)
        if epoch in parsed:
            raise ValueError(f"duplicate candidate epoch: {epoch}")
        parsed[epoch] = Path(path_text)
    if tuple(sorted(parsed)) != EPOCHS:
        raise ValueError(f"candidate epochs must be exactly {EPOCHS}")

    original_summary, original = load(args.base_original)
    full_summary, full = load(args.base_full)
    base_sha = original_summary.get("checkpoint_sha256")
    if full_summary.get("checkpoint_sha256") != base_sha:
        raise ValueError("base attribution cells use different weights")
    rows = []
    for epoch in EPOCHS:
        summary, records = load(parsed[epoch])
        if summary.get("selector_logits_source") != "reference":
            raise ValueError(f"epoch {epoch} does not use the reference selector")
        if summary.get("generation_policy_algorithm") != "diffgrpo_full_chain":
            raise ValueError(f"epoch {epoch} algorithm mismatch")
        if summary.get("schedule", {}).get("roll_timesteps") != [32, 24, 16, 8, 0]:
            raise ValueError(f"epoch {epoch} schedule mismatch")
        if summary.get("reference_checkpoint_sha256") != base_sha:
            raise ValueError(f"epoch {epoch} frozen reference mismatch")
        selected = float(summary["selected_reward"])
        total = paired(original, records, "selected_reward")
        grpo = paired(full, records, "selected_reward")
        safety = {}
        for name in SAFETY:
            safety[name] = float(np.mean([
                records[token]["selected_components"][name]
                - original[token]["selected_components"][name]
                for token in sorted(original)
            ]))
        rows.append({
            "epoch": epoch,
            "artifact": str(parsed[epoch]),
            "checkpoint": summary.get("checkpoint"),
            "checkpoint_sha256": summary.get("checkpoint_sha256"),
            "selected_reward": selected,
            "system_total_delta": total,
            "grpo_delta": grpo,
            "safety_deltas_vs_original": safety,
            "eligible": all(value >= 0.0 for value in safety.values()),
        })

    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise RuntimeError("no development epoch satisfies the safety constraint")
    best_reward = max(row["selected_reward"] for row in eligible)
    selected = min(
        (row for row in eligible if best_reward - row["selected_reward"] <= 0.0005),
        key=lambda row: row["epoch"],
    )
    result = {
        "passed": True,
        "selection_rule": (
            "highest calibration selected_reward with nonnegative collision/"
            "drivable/ttc deltas; within 0.0005 choose earlier epoch"
        ),
        "base_original": str(args.base_original),
        "base_full": str(args.base_full),
        "candidates": rows,
        "best_observed_reward": best_reward,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
