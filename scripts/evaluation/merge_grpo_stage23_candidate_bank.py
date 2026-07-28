#!/usr/bin/env python3
"""Merge completed Stage23 candidate shards in manifest order."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--namespace", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_tokens = [str(record["token"]) for record in manifest["records"]]
    records = []
    summary = None
    expected_offset = 0
    checkpoint_sha = None
    shard_provenance = []
    for path in args.shard:
        payload = json.loads(path.read_text(encoding="utf-8"))
        shard_summary = payload.get("summary", {})
        if not shard_summary.get("completed"):
            raise RuntimeError(f"incomplete Stage23 shard: {path}")
        if int(shard_summary.get("evaluation_noise_namespace", -999)) != args.namespace:
            raise RuntimeError("Stage23 shard namespace mismatch")
        if int(shard_summary.get("token_offset", -1)) != expected_offset:
            raise RuntimeError("Stage23 shard offsets are not contiguous/in order")
        current_sha = shard_summary.get("checkpoint_sha256")
        if checkpoint_sha is None:
            checkpoint_sha = current_sha
            summary = dict(shard_summary)
        elif checkpoint_sha != current_sha:
            raise RuntimeError("Stage23 shards use different generators")
        shard_records = payload.get("records", [])
        if len(shard_records) != int(shard_summary.get("num_tokens", -1)):
            raise RuntimeError("Stage23 shard record count mismatch")
        records.extend(shard_records)
        expected_offset += len(shard_records)
        shard_provenance.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "count": len(shard_records),
            }
        )
    actual_tokens = [str(record["token"]) for record in records]
    if actual_tokens != expected_tokens:
        raise RuntimeError("merged Stage23 tokens differ from frozen manifest order")
    if not all(
        len(record.get("candidate_trajectories", [])) == 20
        and len(record.get("candidate_reference_logits", [])) == 20
        and len(record.get("candidate_rewards", [])) == 20
        and len(record.get("candidate_components", [])) == 20
        and bool(record.get("log_name"))
        for record in records
    ):
        raise RuntimeError("merged Stage23 bank has malformed all-20 records")
    summary.update(
        {
            "requested_limit": len(records),
            "token_offset": 0,
            "num_tokens": len(records),
            "completed": True,
            "stores_candidate_trajectories": True,
            "token_set_sha256": hashlib.sha256(
                "\n".join(actual_tokens).encode("utf-8")
            ).hexdigest(),
            "resumable_shards": shard_provenance,
        }
    )
    output = {"summary": summary, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output), "namespace": args.namespace,
                "count": len(records), "checkpoint_sha256": checkpoint_sha,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
