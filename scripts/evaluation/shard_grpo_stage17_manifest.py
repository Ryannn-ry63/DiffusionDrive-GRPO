#!/usr/bin/env python3
"""Split a Stage-17 token manifest into deterministic contiguous shards."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=8)
    args = parser.parse_args()
    payload = json.loads(args.source.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or args.shards <= 0 or args.shards > len(records):
        raise ValueError("invalid source records or shard count")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index in range(args.shards):
        start = len(records) * index // args.shards
        end = len(records) * (index + 1) // args.shards
        shard = {
            "schema_version": 1,
            "summary": {
                "source": str(args.source.resolve()),
                "shard_index": index,
                "num_shards": args.shards,
                "count": end - start,
                "start": start,
                "end": end,
            },
            "records": records[start:end],
        }
        path = args.output_dir / f"shard{index:02d}.json"
        path.write_text(json.dumps(shard, indent=2) + "\n", encoding="utf-8")
        outputs.append(str(path))
    print(json.dumps({"count": len(records), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
