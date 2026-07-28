#!/usr/bin/env python3
"""Fail-closed audit of Stage27 public and multi-domain selector banks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.evaluation.audit_grpo_stage26_candidate_banks import (
    file_sha256,
    resolve,
)


EXPECTED_SHA = {
    "official_base": "59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d",
    "stage16_epoch8": "3a7641d4ac2a9d644eda4cb945d1902d4ac4bebfdad09b156676dc0cbed94e23",
    "stage19_epoch8": "f0bc08a76d5c3aa27f2f19fd46dcf569406bdd41ff5d9af0078fa75e12986c4b",
    "stage21_epoch8": "7a8d850280aa043ef41734fc9854e5fe8f4c2bb6376cf2d7274c1511d06a9492",
    "stage25_epoch2": "8a273860a365b2ca0eb6f97fddd8d381714cf7f4a30ceb08267aed1756413989",
    "public88_base": "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b",
}
OLD_DOMAINS = (
    "official_base",
    "stage16_epoch8",
    "stage19_epoch8",
    "stage21_epoch8",
    "stage25_epoch2",
)
PUBLIC_DOMAIN = ("public88_base",)
NAMESPACES = (-1, 20260811, 20260812)
LABELS = ("default", "ns20260811", "ns20260812")
SAFETY_INDICES = (0, 1, 3)
MANIFEST_SHA256 = (
    "cefe6cdc470f9e14a5244ba422d7ff7265eed7769d4147189d705c1df1604f7f"
)


def domains_for_branch(branch: str) -> tuple[str, ...]:
    if branch == "public":
        return PUBLIC_DOMAIN
    if branch == "multi":
        return OLD_DOMAINS + PUBLIC_DOMAIN
    raise ValueError(f"unsupported Stage27 selector branch: {branch}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("public", "multi"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if file_sha256(args.manifest) != MANIFEST_SHA256:
        raise RuntimeError("Stage27 selector-fit manifest SHA mismatch")
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
        raise RuntimeError("Stage27 requires the locked folds0-3 manifest")

    domains = domains_for_branch(args.branch)
    expected_order = [
        (domain, namespace)
        for domain in domains
        for namespace in NAMESPACES
    ]
    if len(args.artifact) != len(expected_order):
        raise RuntimeError(
            f"{args.branch}: expected {len(expected_order)} artifacts, "
            f"got {len(args.artifact)}"
        )

    artifact_results = []
    pooled_fallback = []
    pooled_headroom = []
    actual_order = []
    for path, expected_key in zip(args.artifact, expected_order):
        summary, records, provenance = resolve(path)
        domain = str(summary.get("generator_domain", ""))
        namespace = int(summary.get("evaluation_noise_namespace", -999))
        key = (domain, namespace)
        checkpoint_sha = str(summary.get("checkpoint_sha256", ""))
        if key != expected_key:
            raise RuntimeError(
                f"Stage27 bank order mismatch: expected {expected_key}, got {key}"
            )
        if (
            checkpoint_sha != EXPECTED_SHA[domain]
            or not summary.get("completed")
            or int(summary.get("num_failures", 0)) != 0
            or not summary.get("stores_candidate_trajectories")
            or len(records) != len(token_order)
        ):
            raise RuntimeError(f"invalid Stage27 bank provenance: {path}")
        if [str(record.get("token", "")) for record in records] != token_order:
            raise RuntimeError(f"Stage27 token order mismatch: {path}")

        fallback_values = []
        headroom_values = []
        for token, record in zip(token_order, records):
            if str(record.get("log_name", "")) != token_logs[token]:
                raise RuntimeError(f"Stage27 token/log mismatch: {token}")
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
                raise RuntimeError(f"malformed Stage27 all-20 record: {token}")
            rewards = np.asarray([
                np.nan if value is None else float(value)
                for value in reward_values
            ])
            fallback = int(np.argmax(logits))
            if not np.isfinite(rewards[fallback]):
                raise RuntimeError(f"invalid Stage27 fallback reward: {token}")
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
        if not finite.any():
            raise RuntimeError(f"Stage27 bank has no safe challenger: {path}")
        pooled_fallback.extend(fallback_values)
        pooled_headroom.extend(headroom_values)
        actual_order.append(key)
        artifact_results.append({
            **provenance,
            "generator_domain": domain,
            "evaluation_noise_namespace": namespace,
            "generator_checkpoint_sha256": checkpoint_sha,
            "count": len(records),
            "fallback_reward_mean": float(np.mean(fallback_values)),
            "safe_challenger_headroom_mean": float(headroom[finite].mean()),
            "safe_gain_at_least_0.005_fraction": float(
                np.mean(headroom >= 0.005)
            ),
            "safe_gain_at_least_0.010_fraction": float(
                np.mean(headroom >= 0.010)
            ),
        })

    pooled = np.asarray(pooled_headroom, dtype=np.float64)
    finite = np.isfinite(pooled)
    epochs = len(expected_order)
    result = {
        "schema_version": 1,
        "stage": 27,
        "phase": 2,
        "branch": args.branch,
        "passed": True,
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "manifest_token_sha256": manifest["summary"]["ordered_token_sha256"],
        "count_per_artifact": len(token_order),
        "num_artifacts": len(artifact_results),
        "num_domain_namespace_records": len(pooled),
        "steps_per_epoch": 2038,
        "formal_epochs": epochs,
        "total_updates": epochs * 2038,
        "ordered_combinations": [
            f"{domain}_{label}.json"
            for domain in domains
            for label in LABELS
        ],
        "pooled_fallback_reward_mean": float(np.mean(pooled_fallback)),
        "pooled_safe_challenger_headroom_mean": float(pooled[finite].mean()),
        "pooled_gain_at_least_0.005_fraction": float(
            np.mean(pooled >= 0.005)
        ),
        "pooled_gain_at_least_0.010_fraction": float(
            np.mean(pooled >= 0.010)
        ),
        "generator_checkpoint_sha256": {
            domain: EXPECTED_SHA[domain] for domain in domains
        },
        "cycle_boundaries": {
            "first": f"{domains[0]}_{LABELS[0]}.json",
            "last": f"{domains[-1]}_{LABELS[-1]}.json",
            "wrap": f"{domains[0]}_{LABELS[0]}.json",
        },
        "artifacts": artifact_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
