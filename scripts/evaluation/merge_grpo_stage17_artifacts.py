#!/usr/bin/env python3
"""Merge deterministic Stage-17 evaluation shards in source-manifest order."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


COMPONENTS = ("collision", "drivable", "progress", "ttc", "comfort", "direction")


def validate_shard_provenance(summaries):
    """Reject mixed generator, reference, schedule, adapter, or calibration shards."""
    if not summaries:
        raise RuntimeError("no Stage-17 shard summaries")
    for name in (
        "checkpoint_sha256", "reference_checkpoint_sha256",
        "selector_logits_source", "generation_policy_algorithm",
        "evaluation_noise_namespace", "schedule", "cache_path",
        "metric_cache_path", "log_split",
    ):
        if any(summary.get(name) != summaries[0].get(name) for summary in summaries[1:]):
            raise RuntimeError(f"shard provenance mismatch: {name}")
    if summaries[0].get("selector_logits_source") == "paired_tail_risk":
        first_paired = summaries[0].get("paired_risk", {})
        for name in ("checkpoint_sha256", "threshold", "calibration_path"):
            if any(
                summary.get("paired_risk", {}).get(name) != first_paired.get(name)
                for summary in summaries[1:]
            ):
                raise RuntimeError(f"shard paired-risk provenance mismatch: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    tokens = [str(record["token"]) for record in source["records"]]
    record_map, summaries = {}, []
    for path in args.artifacts:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summaries.append(payload["summary"])
        for record in payload["records"]:
            token = str(record["token"])
            if token in record_map:
                raise RuntimeError(f"duplicate token across shards: {token}")
            record_map[token] = record
    missing = [token for token in tokens if token not in record_map]
    extra = sorted(set(record_map) - set(tokens))
    if missing or extra:
        raise RuntimeError(f"shard token mismatch: missing={len(missing)} extra={len(extra)}")
    validate_shard_provenance(summaries)
    records = [record_map[token] for token in tokens]
    summary = dict(summaries[0])
    selected = np.asarray([record["selected_reward"] for record in records])
    oracle = np.asarray([record["oracle_reward"] for record in records])
    summary.update({
        "tokens_file": str(args.source_manifest.resolve()),
        "requested_limit": len(tokens),
        "num_tokens": len(tokens),
        "completed": True,
        "selected_reward": float(selected.mean()),
        "oracle_reward": float(oracle.mean()),
        "selection_regret": float(oracle.mean() - selected.mean()),
        "candidate_reward": float(np.mean([record["candidate_reward"] for record in records])),
        "oracle_hit_rate": float(np.mean([record["oracle_hit"] for record in records])),
        "classification_entropy": float(np.mean([record["entropy"] for record in records])),
        "trajectory_diversity": float(np.mean([record["diversity"] for record in records])),
        "reward_logit_spearman": float(np.mean([record["reward_logit_spearman"] for record in records])),
        "current_reference_mode_agreement": float(np.mean([
            record["current_mode"] == record["reference_mode"] for record in records
        ])),
        "selected_component_means": {
            name: float(np.mean([record["selected_components"][name] for record in records]))
            for name in COMPONENTS
        },
        "token_set_sha256": hashlib.sha256("\n".join(tokens).encode()).hexdigest(),
    })
    if summary.get("selector_logits_source") == "paired_tail_risk":
        pairs = [record["paired_risk"] for record in records]
        fallback = np.asarray([pair["fallback"] for pair in pairs], dtype=bool)
        delta = np.asarray([pair["true_current_minus_base"] for pair in pairs])
        tail = max(1, int(np.ceil(0.01 * len(records))))
        paired = dict(summary.get("paired_risk", {}))
        paired.update({
            "fallback_count": int(fallback.sum()),
            "fallback_rate": float(fallback.mean()),
            "current_minus_base_mean": float(delta.mean()),
            "current_minus_base_minimum": float(delta.min()),
            "current_minus_base_bottom_1pct_cvar": float(np.sort(delta)[:tail].mean()),
            "catastrophic_current_count": int((delta <= -0.5).sum()),
            "admitted_catastrophic_count": int(((~fallback) & (delta <= -0.5)).sum()),
        })
        summary["paired_risk"] = paired
    result = {"summary": summary, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "num_tokens": len(records)}, indent=2))


if __name__ == "__main__":
    main()
