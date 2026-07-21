#!/usr/bin/env python3
"""Choose the one locked Stage-17 fallback threshold on calibration-918."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta


SAFETY = (0, 1, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    records = payload["records"]
    if len(records) != 918:
        raise RuntimeError("Stage-17 calibration requires exactly 918 tokens")
    score = np.asarray([r["paired_risk"]["fallback_score"] for r in records])
    current = np.asarray([r["paired_risk"]["current_reward"] for r in records])
    base = np.asarray([r["paired_risk"]["base_reward"] for r in records])
    current_c = np.asarray([r["paired_risk"]["current_components"] for r in records])
    base_c = np.asarray([r["paired_risk"]["base_components"] for r in records])
    delta = current - base
    c_minus_b = float(delta.mean())
    candidates = []
    thresholds = np.unique(np.concatenate(([0.0], score, [1.0])))
    for threshold in thresholds:
        fallback = score > threshold
        admitted = ~fallback
        hybrid = np.where(fallback, base, current)
        hybrid_c = np.where(fallback[:, None], base_c, current_c)
        d_minus_b = hybrid - base
        d_minus_c = hybrid - current
        fallback_rate = float(fallback.mean())
        tail_count = max(1, int(np.ceil(0.01 * len(records))))
        catastrophic = admitted & (delta <= -0.5)
        admitted_count = int(admitted.sum())
        catastrophic_count = int(catastrophic.sum())
        cp_upper = (
            float(beta.ppf(0.95, catastrophic_count + 1, admitted_count - catastrophic_count))
            if admitted_count > catastrophic_count else 1.0
        )
        safety_delta = (hybrid_c[:, SAFETY] - base_c[:, SAFETY]).mean(axis=0)
        eligible = (
            float(d_minus_c.mean()) >= 0.001
            and float(d_minus_b.mean()) >= c_minus_b
            and 0.01 <= fallback_rate <= 0.35
            and bool((safety_delta >= 0).all())
            and catastrophic_count == 0
            and float(d_minus_b.min()) > -0.5
            and float(np.sort(d_minus_b)[:tail_count].mean()) >= -0.1
            and cp_upper <= 0.005
        )
        candidates.append({
            "threshold": float(threshold),
            "eligible": bool(eligible),
            "selected_reward": float(hybrid.mean()),
            "d_minus_b": float(d_minus_b.mean()),
            "d_minus_c": float(d_minus_c.mean()),
            "fallback_rate": fallback_rate,
            "admitted_count": admitted_count,
            "admitted_catastrophic_count": catastrophic_count,
            "catastrophic_cp95_upper": cp_upper,
            "worst_d_minus_b": float(d_minus_b.min()),
            "bottom_1pct_cvar": float(np.sort(d_minus_b)[:tail_count].mean()),
            "safety_deltas": safety_delta.tolist(),
        })
    eligible = [item for item in candidates if item["eligible"]]
    if not eligible:
        result = {"passed": False, "failure": "no eligible threshold", "candidates": candidates}
    else:
        best_reward = max(item["selected_reward"] for item in eligible)
        near = [item for item in eligible if best_reward - item["selected_reward"] <= 0.0005]
        selected = min(near, key=lambda item: item["fallback_rate"])
        result = {
            "passed": True,
            "selected": selected,
            "raw_stage16_c_minus_base": c_minus_b,
            "num_thresholds": len(candidates),
            "artifact": str(args.artifact.resolve()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
