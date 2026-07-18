#!/usr/bin/env python3
"""Build a frozen DiffusionDrive GRPO hard-case manifest from base evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_OUTPUT = Path("artifacts/grpo_stage0/navtrain_priority_manifest.json")
SAFETY_COMPONENTS = ("collision", "drivable", "ttc")


def ordered_token_sha256(tokens: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()


def load_tokens(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        tokens = payload
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        tokens = [record["token"] for record in payload["records"]]
    else:
        raise ValueError(f"unsupported token manifest schema: {path}")
    return [str(token) for token in tokens]


def priority_reasons(
    record: dict,
    low_reward_threshold: float,
    headroom_threshold: float,
    safety_pass_threshold: float,
) -> list[str]:
    reasons: list[str] = []
    selected_reward = float(record["selected_reward"])
    oracle_reward = float(record["oracle_reward"])
    if selected_reward < low_reward_threshold:
        reasons.append("low_selected_reward")
    components = record.get("selected_components")
    if not isinstance(components, dict):
        raise ValueError(
            f"record {record.get('token')} is missing selected_components"
        )
    for name in SAFETY_COMPONENTS:
        if float(components[name]) < safety_pass_threshold:
            reasons.append(f"{name}_fail")
    if oracle_reward - selected_reward > headroom_threshold + 1e-12:
        reasons.append("high_oracle_headroom")
    return reasons


def build_priority_records(
    records: Iterable[dict],
    low_reward_threshold: float = 0.5,
    headroom_threshold: float = 0.05,
    safety_pass_threshold: float = 1.0 - 1e-9,
) -> list[dict]:
    output = []
    for record in records:
        reasons = priority_reasons(
            record,
            low_reward_threshold,
            headroom_threshold,
            safety_pass_threshold,
        )
        if reasons:
            output.append(
                {
                    "token": str(record["token"]),
                    "priority_reasons": reasons,
                    "selected_reward": float(record["selected_reward"]),
                    "oracle_reward": float(record["oracle_reward"]),
                    "oracle_headroom": (
                        float(record["oracle_reward"])
                        - float(record["selected_reward"])
                    ),
                    "selected_components": {
                        name: float(record["selected_components"][name])
                        for name in SAFETY_COMPONENTS
                    },
                }
            )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--protected-manifest",
        type=Path,
        action="append",
        default=[],
        help="Manifest that must have zero token overlap; may be repeated",
    )
    parser.add_argument("--low-reward-threshold", type=float, default=0.5)
    parser.add_argument("--headroom-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.evaluation_artifact.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("evaluation artifact must contain non-empty records")
    source_tokens = [str(record["token"]) for record in records]
    if len(source_tokens) != len(set(source_tokens)):
        raise ValueError("evaluation artifact contains duplicate tokens")
    priority_records = build_priority_records(
        records,
        low_reward_threshold=args.low_reward_threshold,
        headroom_threshold=args.headroom_threshold,
    )
    if not priority_records or len(priority_records) == len(records):
        raise RuntimeError("priority and common pools must both be non-empty")
    priority_tokens = [record["token"] for record in priority_records]
    protected_overlap = {}
    for protected_path in args.protected_manifest:
        overlap = set(priority_tokens) & set(load_tokens(protected_path))
        protected_overlap[str(protected_path)] = len(overlap)
        if overlap:
            raise RuntimeError(
                f"priority tokens overlap protected manifest {protected_path}: "
                f"count={len(overlap)}, first={sorted(overlap)[0]}"
            )

    reason_counts = Counter(
        reason
        for record in priority_records
        for reason in record["priority_reasons"]
    )
    output = {
        "schema_version": 1,
        "summary": {
            "source_evaluation_artifact": str(
                args.evaluation_artifact.resolve()
            ),
            "source_num_tokens": len(source_tokens),
            "source_ordered_token_sha256": ordered_token_sha256(source_tokens),
            "priority_num_tokens": len(priority_records),
            "common_num_tokens": len(records) - len(priority_records),
            "priority_ordered_token_sha256": ordered_token_sha256(priority_tokens),
            "low_reward_threshold": args.low_reward_threshold,
            "headroom_threshold": args.headroom_threshold,
            "safety_pass_threshold": 1.0 - 1e-9,
            "reason_counts": dict(sorted(reason_counts.items())),
            "protected_overlap": protected_overlap,
        },
        "records": priority_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
