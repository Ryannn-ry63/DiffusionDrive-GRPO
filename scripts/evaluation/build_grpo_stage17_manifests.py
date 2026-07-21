#!/usr/bin/env python3
"""Create the immutable Stage-17 risk-fit/validation manifests."""

import argparse
import hashlib
import json
from pathlib import Path


FIT_COUNT = 3427
VALIDATION_COUNT = 856
SEED_PREFIX = "stage17:20260721:"


def load_tokens(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError("Stage-17 source must contain records")
    tokens = [str(record["token"]) for record in records]
    if len(tokens) != len(set(tokens)):
        raise ValueError("Stage-17 source contains duplicate tokens")
    return tokens


def ordered_sha(tokens: list[str]) -> str:
    return hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()


def payload(name: str, tokens: list[str], source: Path) -> dict:
    return {
        "schema_version": 1,
        "summary": {
            "name": name,
            "count": len(tokens),
            "selection": "sha256_stage17",
            "seed_prefix": SEED_PREFIX,
            "source": str(source),
            "ordered_token_sha256": ordered_sha(tokens),
        },
        "records": [{"token": token} for token in tokens],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    tokens = load_tokens(args.source)
    if len(tokens) != FIT_COUNT + VALIDATION_COUNT:
        raise RuntimeError("Stage-17 source must contain exactly 4,283 tokens")
    ordered = sorted(
        tokens,
        key=lambda token: hashlib.sha256(
            f"{SEED_PREFIX}{token}".encode("utf-8")
        ).hexdigest(),
    )
    fit = ordered[:FIT_COUNT]
    validation = ordered[FIT_COUNT:]
    smoke = fit[:64]
    if set(fit) & set(validation):
        raise AssertionError("Stage-17 fit/validation overlap")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "risk_fit": payload("risk_fit", fit, args.source),
        "risk_validation": payload("risk_validation", validation, args.source),
        "risk_smoke64": payload("risk_smoke64", smoke, args.source),
    }
    paths = {}
    for name, data in outputs.items():
        path = args.output_dir / f"{name}_manifest.json"
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        paths[name] = str(path)
    audit = {
        "source": str(args.source),
        "source_count": len(tokens),
        "source_ordered_sha256": ordered_sha(tokens),
        "seed_prefix": SEED_PREFIX,
        "fit_count": len(fit),
        "validation_count": len(validation),
        "overlap": 0,
        "outputs": paths,
    }
    (args.output_dir / "manifest_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
