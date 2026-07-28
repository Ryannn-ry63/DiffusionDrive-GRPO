#!/usr/bin/env python3
"""Create lightweight, SHA-pinned Stage24 wrappers for Stage23 base banks."""

import argparse
import hashlib
import json
from pathlib import Path


BASE_SHA256 = "59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d"
SOURCES = {
    "official_base_default.json": "base_default.json",
    "official_base_ns20260811.json": "base_ns20260811.json",
    "official_base_ns20260812.json": "base_ns20260812.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir", type=Path,
        default=Path("artifacts/grpo_stage23/candidate_bank"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/grpo_stage24/candidate_bank"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for output_name, source_name in SOURCES.items():
        source = (args.source_dir / source_name).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        payload = json.loads(source.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        if (
            summary.get("checkpoint_sha256") != BASE_SHA256
            or not summary.get("completed")
            or summary.get("num_tokens") != 4075
            or not summary.get("stores_candidate_trajectories")
        ):
            raise RuntimeError(f"invalid Stage23 base bank: {source}")
        wrapper = {
            "schema_version": 1,
            "summary": {
                "stage": 24,
                "generator_domain": "official_base",
                "checkpoint_sha256": BASE_SHA256,
                "evaluation_noise_namespace": int(
                    summary["evaluation_noise_namespace"]
                ),
                "stores_candidate_trajectories": True,
                "source_artifact": str(source),
                "source_artifact_sha256": sha256(source),
                "source_num_tokens": 4075,
            },
        }
        output = args.output_dir / output_name
        output.write_text(json.dumps(wrapper, indent=2), encoding="utf-8")
        outputs.append(str(output))
    print(json.dumps({"outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
