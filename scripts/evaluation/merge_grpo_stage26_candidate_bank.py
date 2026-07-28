#!/usr/bin/env python3
"""Merge Stage26 folds0-2 and fold3 candidate banks in folds0-3 order."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_complete(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    records = payload.get("records", [])
    if (
        not summary.get("completed")
        or int(summary.get("num_failures", 0)) != 0
        or not summary.get("stores_candidate_trajectories")
        or int(summary.get("num_tokens", -1)) != len(records)
    ):
        raise RuntimeError(f"incomplete or malformed candidate bank: {path}")
    return summary, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--suffix", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--namespace", type=int, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_records = manifest.get("records", [])
    ordered_tokens = [str(record["token"]) for record in manifest_records]
    token_logs = {
        str(record["token"]): str(record["log_name"])
        for record in manifest_records
    }
    if (
        len(ordered_tokens) != 4075
        or len(ordered_tokens) != len(set(ordered_tokens))
        or manifest.get("summary", {}).get("source_folds") != [0, 1, 2, 3]
    ):
        raise RuntimeError("Stage26 merge requires the locked folds0-3 manifest")

    indexed: dict[str, dict] = {}
    provenance = []
    for path, expected_count in ((args.prefix, 3056), (args.suffix, 1019)):
        summary, records = load_complete(path)
        if (
            str(summary.get("generator_domain", "")) != args.domain
            or int(summary.get("evaluation_noise_namespace", -999))
            != args.namespace
            or str(summary.get("checkpoint_sha256", ""))
            != args.checkpoint_sha256
            or len(records) != expected_count
        ):
            raise RuntimeError(f"Stage26 bank provenance mismatch: {path}")
        for record in records:
            token = str(record.get("token", ""))
            if token in indexed:
                raise RuntimeError(f"duplicate token across Stage26 banks: {token}")
            if token not in token_logs:
                raise RuntimeError(f"unexpected token in Stage26 bank: {token}")
            if str(record.get("log_name", "")) != token_logs[token]:
                raise RuntimeError(f"token/log mismatch in Stage26 bank: {token}")
            if (
                len(record.get("candidate_trajectories", [])) != 20
                or len(record.get("candidate_reference_logits", [])) != 20
                or len(record.get("candidate_rewards", [])) != 20
                or len(record.get("candidate_components", [])) != 20
            ):
                raise RuntimeError(f"malformed all-20 Stage26 record: {token}")
            indexed[token] = record
        provenance.append({
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "count": len(records),
        })

    if set(indexed) != set(ordered_tokens):
        missing = set(ordered_tokens) - set(indexed)
        extra = set(indexed) - set(ordered_tokens)
        raise RuntimeError(
            f"Stage26 merged coverage mismatch: missing={len(missing)}, "
            f"extra={len(extra)}"
        )
    records = [indexed[token] for token in ordered_tokens]
    result = {
        "schema_version": 1,
        "summary": {
            "stage": 26,
            "checkpoint": records[0].get("checkpoint", ""),
            "checkpoint_sha256": args.checkpoint_sha256,
            "generator_domain": args.domain,
            "reference_checkpoint": records[0].get("reference_checkpoint", ""),
            "reference_checkpoint_sha256": records[0].get(
                "reference_checkpoint_sha256", ""
            ),
            "generation_policy_algorithm": records[0].get(
                "generation_policy_algorithm", "legacy_ppo"
            ),
            "evaluation_noise_namespace": args.namespace,
            "tokens_file": str(args.manifest),
            "requested_limit": len(records),
            "token_offset": 0,
            "num_tokens": len(records),
            "num_failures": 0,
            "completed": True,
            "stores_candidate_trajectories": True,
            "ordered_token_sha256": manifest["summary"]["ordered_token_sha256"],
            "merged_sources": provenance,
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "count": len(records),
        "domain": args.domain,
        "namespace": args.namespace,
        "sha256": file_sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
