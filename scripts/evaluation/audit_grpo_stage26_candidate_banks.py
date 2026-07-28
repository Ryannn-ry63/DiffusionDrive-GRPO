#!/usr/bin/env python3
"""Fail-closed audit of the Stage26 five-domain candidate-bank grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED = {
    "official_base": "59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d",
    "stage16_epoch8": "3a7641d4ac2a9d644eda4cb945d1902d4ac4bebfdad09b156676dc0cbed94e23",
    "stage19_epoch8": "f0bc08a76d5c3aa27f2f19fd46dcf569406bdd41ff5d9af0078fa75e12986c4b",
    "stage21_epoch8": "7a8d850280aa043ef41734fc9854e5fe8f4c2bb6376cf2d7274c1511d06a9492",
    "stage25_epoch2": "8a273860a365b2ca0eb6f97fddd8d381714cf7f4a30ceb08267aed1756413989",
}
NAMESPACES = (-1, 20260811, 20260812)
SAFETY_INDICES = (0, 1, 3)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path: Path) -> tuple[dict, list[dict], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    records = payload.get("records", [])
    provenance = {"path": str(path), "sha256": file_sha256(path)}
    source_text = str(summary.get("source_artifact", ""))
    if source_text:
        source = Path(source_text)
        if not source.is_file():
            raise FileNotFoundError(source)
        source_sha = file_sha256(source)
        if source_sha != summary.get("source_artifact_sha256"):
            raise RuntimeError(f"source artifact SHA mismatch: {path}")
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
        len(token_order) != 4075
        or len(token_order) != len(set(token_order))
        or manifest.get("summary", {}).get("source_folds") != [0, 1, 2, 3]
    ):
        raise RuntimeError("Stage26 audit requires locked folds0-3 manifest")

    expected_grid = {
        (domain, namespace)
        for domain in EXPECTED
        for namespace in NAMESPACES
    }
    seen_grid = set()
    artifact_results = []
    pooled_headroom = []
    pooled_fallback = []
    for path in args.artifact:
        summary, records, provenance = resolve(path)
        domain = str(summary.get("generator_domain", ""))
        namespace = int(summary.get("evaluation_noise_namespace", -999))
        checkpoint_sha = str(summary.get("checkpoint_sha256", ""))
        key = (domain, namespace)
        if key in seen_grid:
            raise RuntimeError(f"duplicate Stage26 grid cell: {key}")
        if (
            key not in expected_grid
            or checkpoint_sha != EXPECTED.get(domain)
            or not summary.get("completed")
            or int(summary.get("num_failures", 0)) != 0
            or not summary.get("stores_candidate_trajectories")
            or len(records) != 4075
        ):
            raise RuntimeError(f"invalid Stage26 bank provenance: {path}")
        seen_grid.add(key)
        actual_order = [str(record.get("token", "")) for record in records]
        if actual_order != token_order:
            raise RuntimeError(f"Stage26 token order mismatch: {path}")

        fallback_values = []
        headroom_values = []
        for token, record in zip(token_order, records):
            if str(record.get("log_name", "")) != token_logs[token]:
                raise RuntimeError(f"Stage26 token/log mismatch: {token}")
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
                raise RuntimeError(f"malformed Stage26 all-20 record: {token}")
            rewards = np.asarray([
                np.nan if value is None else float(value)
                for value in reward_values
            ])
            fallback = int(np.argmax(logits))
            if not np.isfinite(rewards[fallback]):
                raise RuntimeError(f"invalid Stage26 fallback reward: {token}")
            safe = (
                np.isfinite(rewards)
                & np.isfinite(components).all(axis=-1)
                & np.all(components[:, SAFETY_INDICES] >= 0.999, axis=-1)
            )
            safe[fallback] = False
            fallback_values.append(float(rewards[fallback]))
            headroom_values.append(
                float(np.max(rewards[safe]) - rewards[fallback])
                if safe.any() else float("-inf")
            )
        headroom = np.asarray(headroom_values, dtype=np.float64)
        finite = np.isfinite(headroom)
        pooled_headroom.extend(headroom_values)
        pooled_fallback.extend(fallback_values)
        artifact_results.append({
            **provenance,
            "generator_domain": domain,
            "evaluation_noise_namespace": namespace,
            "generator_checkpoint_sha256": checkpoint_sha,
            "count": len(records),
            "fallback_reward_mean": float(np.mean(fallback_values)),
            "safe_challenger_headroom_mean": float(headroom[finite].mean()),
            "safe_gain_at_least_0.005_fraction": float(np.mean(headroom >= 0.005)),
            "safe_gain_at_least_0.010_fraction": float(np.mean(headroom >= 0.010)),
        })

    if seen_grid != expected_grid:
        raise RuntimeError(
            f"Stage26 grid mismatch: missing={sorted(expected_grid-seen_grid)}"
        )
    pooled = np.asarray(pooled_headroom, dtype=np.float64)
    result = {
        "schema_version": 1,
        "stage": 26,
        "passed": True,
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "manifest_token_sha256": manifest["summary"]["ordered_token_sha256"],
        "count_per_artifact": len(token_order),
        "num_artifacts": len(artifact_results),
        "num_domain_namespace_records": len(pooled),
        "pooled_fallback_reward_mean": float(np.mean(pooled_fallback)),
        "pooled_safe_challenger_headroom_mean": float(
            pooled[np.isfinite(pooled)].mean()
        ),
        "pooled_gain_at_least_0.005_fraction": float(np.mean(pooled >= 0.005)),
        "pooled_gain_at_least_0.010_fraction": float(np.mean(pooled >= 0.010)),
        "generator_checkpoint_sha256": EXPECTED,
        "artifacts": artifact_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
