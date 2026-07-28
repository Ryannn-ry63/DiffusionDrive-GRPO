#!/usr/bin/env python3
"""Fail-closed audit of the frozen Stage24 selector-training candidate grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SAFETY_INDICES = (0, 1, 3)
EXPECTED_DOMAINS = (
    "official_base", "stage16_epoch8", "stage19_epoch8", "stage21_epoch8",
)
EXPECTED_NAMESPACES = (-1, 20260811, 20260812)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_artifact(path: Path) -> tuple[dict, list[dict], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    records = payload.get("records", [])
    provenance = {"path": str(path), "sha256": sha256(path)}
    source_text = str(summary.get("source_artifact", ""))
    if source_text:
        source = Path(source_text)
        if not source.is_file():
            raise FileNotFoundError(source)
        source_sha = sha256(source)
        if source_sha != str(summary.get("source_artifact_sha256", "")):
            raise RuntimeError(f"source SHA mismatch: {path}")
        source_payload = json.loads(source.read_text(encoding="utf-8"))
        records = source_payload.get("records", [])
        summary = {**source_payload.get("summary", {}), **summary}
        provenance.update({"source_path": str(source), "source_sha256": source_sha})
    return summary, records, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_records = manifest.get("records", [])
    token_order = [str(record["token"]) for record in manifest_records]
    token_logs = {
        str(record["token"]): str(record["log_name"])
        for record in manifest_records
    }
    if (
        len(token_order) != 3056
        or len(token_order) != len(set(token_order))
        or manifest.get("summary", {}).get("source_folds") != [0, 1, 2]
    ):
        raise RuntimeError("Stage24 candidate audit requires locked folds0-2 manifest")
    token_set = set(token_order)

    combinations = set()
    domain_checkpoints: dict[str, str] = {}
    artifact_results = []
    pooled_fallback = []
    pooled_headroom = []
    for path in args.artifact:
        summary, records, provenance = resolve_artifact(path)
        domain = str(summary.get("generator_domain", ""))
        namespace = int(summary.get("evaluation_noise_namespace", -999))
        checkpoint_sha = str(summary.get("checkpoint_sha256", ""))
        combination = (domain, namespace)
        if combination in combinations:
            raise RuntimeError(f"duplicate Stage24 combination: {combination}")
        combinations.add(combination)
        if (
            domain not in EXPECTED_DOMAINS
            or namespace not in EXPECTED_NAMESPACES
            or len(checkpoint_sha) != 64
            or not summary.get("completed")
            or not summary.get("stores_candidate_trajectories")
        ):
            raise RuntimeError(f"invalid Stage24 bank provenance: {path}")
        previous_sha = domain_checkpoints.setdefault(domain, checkpoint_sha)
        if previous_sha != checkpoint_sha:
            raise RuntimeError(f"generator checkpoint drift within domain {domain}")

        indexed = {}
        for record in records:
            token = str(record.get("token", ""))
            if token not in token_set:
                continue
            if token in indexed:
                raise RuntimeError(f"duplicate Stage24 token in {path}: {token}")
            indexed[token] = record
        if set(indexed) != token_set:
            raise RuntimeError(f"Stage24 token coverage mismatch: {path}")

        fallback_values = []
        headroom_values = []
        no_safe_challenger = 0
        for token in token_order:
            record = indexed[token]
            if str(record.get("log_name", "")) != token_logs[token]:
                raise RuntimeError(f"Stage24 token/log drift: {token}")
            trajectories = np.asarray(
                record.get("candidate_trajectories"), dtype=np.float64
            )
            logits = np.asarray(
                record.get("candidate_reference_logits"), dtype=np.float64
            )
            components = np.asarray(
                record.get("candidate_components"), dtype=np.float64
            )
            reward_values = record.get("candidate_rewards")
            if (
                trajectories.shape != (20, 8, 3)
                or logits.shape != (20,)
                or components.shape != (20, 6)
                or not isinstance(reward_values, list)
                or len(reward_values) != 20
            ):
                raise RuntimeError(f"malformed Stage24 all-20 record: {token}")
            rewards = np.asarray([
                np.nan if value is None else float(value)
                for value in reward_values
            ])
            fallback = int(np.argmax(logits))
            if not np.isfinite(rewards[fallback]):
                raise RuntimeError(f"invalid Stage24 fallback reward: {token}")
            safe = (
                np.isfinite(rewards)
                & np.isfinite(components).all(axis=-1)
                & np.all(components[:, SAFETY_INDICES] >= 0.999, axis=-1)
            )
            safe[fallback] = False
            fallback_values.append(float(rewards[fallback]))
            if safe.any():
                headroom_values.append(
                    float(np.max(rewards[safe]) - rewards[fallback])
                )
            else:
                headroom_values.append(float("-inf"))
                no_safe_challenger += 1

        fallback_array = np.asarray(fallback_values, dtype=np.float64)
        headroom_array = np.asarray(headroom_values, dtype=np.float64)
        finite = np.isfinite(headroom_array)
        pooled_fallback.extend(fallback_values)
        pooled_headroom.extend(headroom_values)
        artifact_results.append({
            **provenance,
            "generator_domain": domain,
            "evaluation_noise_namespace": namespace,
            "generator_checkpoint_sha256": checkpoint_sha,
            "count": len(token_order),
            "fallback_reward_mean": float(fallback_array.mean()),
            "safe_challenger_headroom_mean": float(headroom_array[finite].mean()),
            "safe_challenger_gain_at_least_0.005_fraction": float(
                np.mean(headroom_array >= 0.005)
            ),
            "safe_challenger_gain_at_least_0.010_fraction": float(
                np.mean(headroom_array >= 0.010)
            ),
            "no_safe_challenger_count": no_safe_challenger,
        })

    expected = {
        (domain, namespace)
        for domain in EXPECTED_DOMAINS
        for namespace in EXPECTED_NAMESPACES
    }
    if combinations != expected:
        raise RuntimeError(
            f"Stage24 candidate grid mismatch; missing={sorted(expected-combinations)}"
        )
    pooled_headroom_array = np.asarray(pooled_headroom, dtype=np.float64)
    pooled_finite = np.isfinite(pooled_headroom_array)
    result = {
        "schema_version": 1,
        "stage": 24,
        "passed": True,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "manifest_token_sha256": manifest["summary"]["ordered_token_sha256"],
        "count_per_artifact": len(token_order),
        "num_artifacts": len(artifact_results),
        "num_domain_namespace_records": len(pooled_headroom),
        "pooled_fallback_reward_mean": float(np.mean(pooled_fallback)),
        "pooled_safe_challenger_headroom_mean": float(
            pooled_headroom_array[pooled_finite].mean()
        ),
        "pooled_gain_at_least_0.005_fraction": float(
            np.mean(pooled_headroom_array >= 0.005)
        ),
        "pooled_gain_at_least_0.010_fraction": float(
            np.mean(pooled_headroom_array >= 0.010)
        ),
        "artifacts": artifact_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
