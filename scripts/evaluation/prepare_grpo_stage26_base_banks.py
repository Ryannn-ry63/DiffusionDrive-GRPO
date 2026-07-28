#!/usr/bin/env python3
"""Create SHA-pinned Stage26 wrappers for the full official-base banks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BASE_SHA256 = "59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d"
SOURCES = {
    "official_base_default.json": ("base_default.json", -1),
    "official_base_ns20260811.json": ("base_ns20260811.json", 20260811),
    "official_base_ns20260812.json": ("base_ns20260812.json", 20260812),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir", type=Path,
        default=Path("artifacts/grpo_stage23/candidate_bank"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/grpo_stage26/candidate_bank"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for output_name, (source_name, namespace) in SOURCES.items():
        source = (args.source_dir / source_name).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        payload = json.loads(source.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        if (
            summary.get("checkpoint_sha256") != BASE_SHA256
            or int(summary.get("evaluation_noise_namespace", -999)) != namespace
            or not summary.get("completed")
            or int(summary.get("num_failures", 0)) != 0
            or summary.get("num_tokens") != 4075
            or not summary.get("stores_candidate_trajectories")
        ):
            raise RuntimeError(f"invalid Stage23 base bank: {source}")
        wrapper = {
            "schema_version": 1,
            "summary": {
                "stage": 26,
                "generator_domain": "official_base",
                "checkpoint_sha256": BASE_SHA256,
                "evaluation_noise_namespace": namespace,
                "stores_candidate_trajectories": True,
                "source_artifact": str(source),
                "source_artifact_sha256": file_sha256(source),
                "source_num_tokens": 4075,
            },
        }
        output = args.output_dir / output_name
        output.write_text(json.dumps(wrapper, indent=2), encoding="utf-8")
        outputs.append({"path": str(output), "sha256": file_sha256(output)})
    print(json.dumps({"outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
