#!/usr/bin/env python3
"""Validate and freeze the eight Stage37 JFI fitting banks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PLAN_SHA256 = "4dc469dd0be4d2579796ff70ae7a09ed9fd8e4c92449cd057e42463d268bb5d8"
SYSTEMS = ("P", "DPEL192", "NCD192", "RGT192")
NOISES = (20263721, 20263722)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    bank_dir = root / "artifacts/grpo_stage37/jfi/banks"
    manifest_freeze_path = root / "artifacts/grpo_stage37/jfi/manifests/freeze.json"
    output = bank_dir / "freeze.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    manifest_freeze = json.loads(manifest_freeze_path.read_text())
    expected_count = int(manifest_freeze["count"])
    expected_tokens = str(manifest_freeze["ordered_token_sha256"])
    artifacts = []
    checkpoint_by_system: dict[str, str] = {}
    for system in SYSTEMS:
        for noise in NOISES:
            path = bank_dir / f"{system}_ns{noise}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            summary = payload.get("summary", {})
            records = payload.get("records", [])
            token_sha = hashlib.sha256(
                "\n".join(str(record.get("token", "")) for record in records).encode()
            ).hexdigest()
            expected_domain = f"stage37_jfi_{system.lower()}_folds23"
            if (
                not summary.get("completed")
                or int(summary.get("num_failures", -1)) != 0
                or int(summary.get("num_tokens", -1)) != expected_count
                or len(records) != expected_count
                or int(summary.get("evaluation_noise_namespace", -1)) != noise
                or summary.get("generator_domain") != expected_domain
                or not summary.get("stores_candidate_trajectories")
                or token_sha != expected_tokens
                or not all(
                    len(record.get("candidate_trajectories", [])) == 20
                    and len(record.get("candidate_rewards", [])) == 20
                    and len(record.get("candidate_components", [])) == 20
                    for record in records
                )
            ):
                raise RuntimeError(f"invalid Stage37 JFI bank: {path}")
            checkpoint_sha = str(summary.get("checkpoint_sha256", ""))
            if system in checkpoint_by_system and checkpoint_by_system[system] != checkpoint_sha:
                raise RuntimeError(f"generator SHA differs across namespaces: {system}")
            checkpoint_by_system[system] = checkpoint_sha
            artifacts.append({
                "system": system,
                "namespace": noise,
                "domain": expected_domain,
                "path": str(path),
                "sha256": sha256(path),
                "generator_checkpoint": summary.get("checkpoint"),
                "generator_checkpoint_sha256": checkpoint_sha,
                "count": expected_count,
                "ordered_token_sha256": token_sha,
            })
            del payload, records
    result = {
        "schema_version": 1,
        "stage": 37,
        "passed": True,
        "purpose": "jfi_head_fit_folds2_3_only",
        "plan_sha256": PLAN_SHA256,
        "manifest_freeze": str(manifest_freeze_path),
        "manifest_freeze_sha256": sha256(manifest_freeze_path),
        "count": expected_count,
        "num_banks": len(artifacts),
        "artifacts": artifacts,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"PASS frozen {len(artifacts)} Stage37 JFI banks: {output}")


if __name__ == "__main__":
    main()
