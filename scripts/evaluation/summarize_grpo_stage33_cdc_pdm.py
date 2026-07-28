#!/usr/bin/env python3
"""Aggregate Stage33 holdout PDM cells across two common-noise namespaces."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped = defaultdict(list)
    for path in sorted(args.eval_dir.glob("fold[01]/*.json")):
        payload = json.loads(path.read_text())
        summary = payload["summary"]
        domain = summary["generator_domain"]
        if not domain.startswith("stage33_cdc_"):
            continue
        step = domain.split("stage33_cdc_", 1)[1].split("_fold", 1)[0]
        fold = int(domain.rsplit("_fold", 1)[1])
        grouped[(fold, step)].append(summary)
    if not grouped:
        raise RuntimeError("No Stage33 PDM evaluation cells found")

    rows = {}
    for (fold, step), summaries in sorted(grouped.items()):
        rows[f"fold{fold}/{step}"] = {
            "fold": fold,
            "step": step,
            "num_cells": len(summaries),
            "num_tokens_per_cell": sorted({int(s["num_tokens"]) for s in summaries}),
            "selected_reward_mean": mean(float(s["selected_reward"]) for s in summaries),
            "oracle_reward_mean": mean(float(s["oracle_reward"]) for s in summaries),
            "selection_regret_mean": mean(float(s["selection_regret"]) for s in summaries),
            "oracle_hit_rate_mean": mean(float(s["oracle_hit_rate"]) for s in summaries),
            "catastrophic_switch_count": sum(
                int(s.get("stage25_selector", {}).get("catastrophic_switch_count", 0))
                for s in summaries
            ),
        }
    for row in rows.values():
        baseline = rows.get(f"fold{row['fold']}/p")
        if baseline is not None and row["step"] != "p":
            row["gain_vs_public_same_fold"] = (
                row["selected_reward_mean"] - baseline["selected_reward_mean"]
            )
            row["regret_change_vs_public_same_fold"] = (
                row["selection_regret_mean"] - baseline["selection_regret_mean"]
            )

    formal = [row for row in rows.values() if row["step"] != "p"]
    result = {
        "stage": 33,
        "objective": "cdc_deployment_credit_v1",
        "num_groups": len(rows),
        "rows": rows,
        "best_formal_by_gain": max(
            formal,
            key=lambda row: row.get("gain_vs_public_same_fold", float("-inf")),
        ) if formal else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

