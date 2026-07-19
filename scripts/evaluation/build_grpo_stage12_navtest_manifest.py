#!/usr/bin/env python3
"""Build the locked Stage-12 navtest token order from a validated score CSV."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-csv", type=Path, required=True)
    parser.add_argument("--expected-tokens", type=int, default=12146)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_valid_tokens(path: Path, expected_tokens: int) -> list[str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "token" not in rows[0] or "valid" not in rows[0]:
        raise ValueError("score CSV must contain token and valid columns")
    aggregate_rows = [row for row in rows if row["token"] == "average"]
    if len(aggregate_rows) > 1:
        raise ValueError("score CSV contains multiple average rows")
    rows = [row for row in rows if row["token"] != "average"]
    invalid = [row.get("token", "") for row in rows if row["valid"].lower() != "true"]
    if invalid:
        raise ValueError(f"score CSV contains {len(invalid)} invalid scenarios")
    tokens = [str(row["token"]) for row in rows]
    if len(tokens) != expected_tokens:
        raise ValueError(f"expected {expected_tokens} tokens, found {len(tokens)}")
    if len(tokens) != len(set(tokens)):
        raise ValueError("score CSV contains duplicate tokens")
    return tokens


def main() -> None:
    args = parse_args()
    if args.expected_tokens <= 0:
        raise ValueError("expected-tokens must be positive")
    tokens = load_valid_tokens(args.score_csv, args.expected_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tokens, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()
    print(
        json.dumps(
            {
                "num_tokens": len(tokens),
                "token_set_sha256": digest,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
