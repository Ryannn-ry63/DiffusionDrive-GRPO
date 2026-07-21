#!/usr/bin/env python3
"""Select the one allowed Stage-15 deployment margin on calibration tokens."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SAFETY_INDICES = (0, 1, 3)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safety-threshold", type=float, default=0.9)
    args = parser.parse_args()
    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if len(records) != 918:
        raise RuntimeError(f"Calibration artifact must have 918 records; got {len(records)}")
    predicted = []
    eligible = []
    reward_delta = []
    component_delta = []
    for record in records:
        value = record.get("value_selector")
        rewards = record.get("candidate_rewards")
        if value is None or rewards is None:
            raise RuntimeError("Calibration artifact lacks Stage-15 diagnostics")
        fallback = int(value["fallback_mode"])
        challenger = int(value["challenger_mode"])
        fallback_reward = rewards[fallback]
        challenger_reward = rewards[challenger]
        valid = fallback_reward is not None and challenger_reward is not None
        fallback_lower = np.asarray(value["fallback_safety_lower"], dtype=float)
        challenger_lower = np.asarray(value["challenger_safety_lower"], dtype=float)
        safe = bool(
            valid
            and np.all(challenger_lower >= args.safety_threshold)
            and np.all(challenger_lower >= fallback_lower)
        )
        predicted.append(float(value["predicted_advantage"]))
        eligible.append(safe)
        reward_delta.append(
            float(challenger_reward - fallback_reward) if valid else 0.0
        )
        fallback_components = np.asarray(value["fallback_components"], dtype=float)
        challenger_components = np.asarray(value["challenger_components"], dtype=float)
        component_delta.append(challenger_components - fallback_components)
    predicted = np.asarray(predicted)
    eligible = np.asarray(eligible, dtype=bool)
    reward_delta = np.asarray(reward_delta)
    component_delta = np.asarray(component_delta)
    thresholds = sorted(
        {0.0, *[float(value) for value in predicted[eligible] if value >= 0.0]}
    )
    candidates = []
    for margin in thresholds:
        switch = eligible & (predicted > margin)
        switched_delta = reward_delta[switch]
        selected_delta = np.where(switch, reward_delta, 0.0)
        selected_component_delta = np.where(
            switch[:, None], component_delta, 0.0
        )
        beneficial = switched_delta[switched_delta > 0]
        harmful = switched_delta[switched_delta < 0]
        beneficial_sum = float(beneficial.sum())
        harmful_loss = float(-harmful.sum())
        safety_means = selected_component_delta[:, SAFETY_INDICES].mean(axis=0)
        worst = float(switched_delta.min()) if switched_delta.size else 0.0
        passed = bool(
            switch.any()
            and np.all(safety_means >= -1e-12)
            and worst > -0.5
            and harmful_loss <= 0.25 * beneficial_sum
        )
        candidates.append(
            {
                "margin": margin,
                "passed": passed,
                "switch_count": int(switch.sum()),
                "switch_rate": float(switch.mean()),
                "selected_difference": float(selected_delta.mean()),
                "safety_component_differences": safety_means.tolist(),
                "beneficial_gain_sum": beneficial_sum,
                "harmful_loss_sum": harmful_loss,
                "worst_switch_gain": worst,
            }
        )
    passed = [candidate for candidate in candidates if candidate["passed"]]
    if not passed:
        raise RuntimeError("No Stage-15 calibration margin satisfies safety constraints")
    selected = max(
        passed,
        key=lambda candidate: (
            candidate["selected_difference"], -candidate["margin"]
        ),
    )
    result = {
        "passed": True,
        "artifact": str(args.artifact),
        "artifact_sha256": sha256(args.artifact),
        "token_set_sha256": payload.get("summary", {}).get("token_set_sha256"),
        "selector_checkpoint": payload.get("summary", {})
        .get("value_selector", {})
        .get("checkpoint"),
        "selector_checkpoint_sha256": payload.get("summary", {})
        .get("value_selector", {})
        .get("checkpoint_sha256"),
        "safety_threshold": args.safety_threshold,
        "selected": selected,
        "num_thresholds": len(candidates),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
