#!/usr/bin/env python3
"""Offline Stage39 target/budget audit from frozen public20/public40 banks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from navsim.agents.diffusiondrive.stage39_challenger import STAGE39_PLAN_SHA256
from scripts.evaluation.summarize_grpo_stage39 import (
    PILOT_NOISES,
    exact_public20,
    index,
    load,
    mean,
    record_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    cells = []
    for noise in PILOT_NOISES:
        public20 = load(args.eval_root / "fold2" / f"P20_ns{noise}.json", "public20")
        public40 = load(
            args.eval_root / "fold2" / f"P40_ns{noise}.json", "public40_extra"
        )
        base = index(public20["records"])
        extra = index(public40["records"])
        if set(base) != set(extra):
            raise RuntimeError("Stage39 Phase0 token alignment drifted")
        exact = all(exact_public20(base[token], extra[token]) for token in base)
        rows = [record_metrics(extra[token]) for token in base]
        cells.append(
            {
                "noise": noise,
                "num_tokens": len(rows),
                "public20_exact": exact,
                "safe_positive_fraction": mean(
                    row["positive_fraction"] for row in rows
                ),
                "strict_safe_oracle_gain": mean(
                    row["safe_union_gain"] for row in rows
                ),
                "catastrophic_rate": mean(
                    row["catastrophic_rate"] for row in rows
                ),
                "public20_path": public20["path"],
                "public20_sha256": public20["sha256"],
                "public40_path": public40["path"],
                "public40_sha256": public40["sha256"],
            }
        )
    aggregate_positive = mean(cell["safe_positive_fraction"] for cell in cells)
    aggregate_oracle = mean(cell["strict_safe_oracle_gain"] for cell in cells)
    checks = {
        "public20_bitwise_exact": all(cell["public20_exact"] for cell in cells),
        "safe_positive_fraction_in_0.05_0.40": 0.05 <= aggregate_positive <= 0.40,
        "strict_safe_oracle_at_least_0.005": aggregate_oracle >= 0.005,
        "each_namespace_strict_safe_oracle_positive": all(
            cell["strict_safe_oracle_gain"] > 0 for cell in cells
        ),
        "finite_targets": bool(
            np.isfinite(
                [
                    value
                    for cell in cells
                    for value in (
                        cell["safe_positive_fraction"],
                        cell["strict_safe_oracle_gain"],
                        cell["catastrophic_rate"],
                    )
                ]
            ).all()
        ),
    }
    result = {
        "schema_version": 1,
        "stage": 39,
        "phase": "offline_target_audit",
        "passed": all(checks.values()),
        "plan_sha256": STAGE39_PLAN_SHA256,
        "checks": checks,
        "safe_positive_fraction": aggregate_positive,
        "strict_safe_oracle_gain": aggregate_oracle,
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
