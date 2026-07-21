#!/usr/bin/env python3
"""Create immutable Stage-15 train/calibration/test token manifests."""

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_TOKENS = 6119
SPLIT_COUNTS = (4283, 918, 918)
SPLIT_NAMES = ("selector_train", "selector_calibration", "selector_test")
SEED = 20260721


def load_tokens(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        tokens = payload
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        tokens = [record["token"] for record in payload["records"]]
    else:
        raise ValueError(f"Unsupported token source: {path}")
    tokens = [str(token) for token in tokens]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Duplicate tokens in {path}")
    return tokens


def token_sha(tokens):
    return hashlib.sha256("\n".join(tokens).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protected", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    source_tokens = load_tokens(args.source)
    if len(source_tokens) != EXPECTED_TOKENS:
        raise RuntimeError(
            f"Stage-15 source must contain {EXPECTED_TOKENS} tokens; "
            f"got {len(source_tokens)}"
        )
    protected = set()
    for path in args.protected:
        protected.update(load_tokens(path))
    overlap = set(source_tokens).intersection(protected)
    if overlap:
        raise RuntimeError(
            f"Stage-15 source overlaps protected evaluation tokens: {len(overlap)}"
        )
    ordered = sorted(
        source_tokens,
        key=lambda token: hashlib.sha256(f"{SEED}:{token}".encode()).hexdigest(),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = 0
    summary = {
        "source": str(args.source),
        "source_count": len(source_tokens),
        "source_ordered_sha256": token_sha(source_tokens),
        "seed": SEED,
        "protected": [str(path) for path in args.protected],
        "splits": {},
    }
    for name, count in zip(SPLIT_NAMES, SPLIT_COUNTS):
        tokens = ordered[start : start + count]
        start += count
        payload = {
            "summary": {
                "name": name,
                "count": len(tokens),
                "ordered_token_sha256": token_sha(tokens),
                "seed": SEED,
                "source": str(args.source),
            },
            "records": [{"token": token} for token in tokens],
        }
        path = args.output_dir / f"{name}_manifest.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summary["splits"][name] = payload["summary"] | {"path": str(path)}
    if start != EXPECTED_TOKENS:
        raise RuntimeError("Stage-15 split counts do not cover the source")
    (args.output_dir / "manifest_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
