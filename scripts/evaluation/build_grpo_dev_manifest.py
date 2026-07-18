#!/usr/bin/env python3
"""Build and audit a fixed, train-disjoint GRPO development manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from omegaconf import OmegaConf


DEFAULT_FEATURE_CACHE = Path(
    "/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache"
)
DEFAULT_METRIC_CACHE = Path(
    "/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/metric_cache_trainval"
)
DEFAULT_NAVTEST_METRIC_CACHE = Path(
    "/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/metric_cache_navtest"
)
DEFAULT_FIXED_TOKENS = Path("artifacts/grpo_stage0/holdout1024_aligned_t8.json")
DEFAULT_SPLIT_CONFIG = Path(
    "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
)
DEFAULT_OUTPUT = Path("artifacts/grpo_stage0/dev4096_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache-path", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--metric-cache-path", type=Path, default=DEFAULT_METRIC_CACHE)
    parser.add_argument(
        "--navtest-metric-cache-path", type=Path, default=DEFAULT_NAVTEST_METRIC_CACHE
    )
    parser.add_argument("--fixed-tokens-file", type=Path, default=DEFAULT_FIXED_TOKENS)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--feature-file", default="transfuser_feature.gz")
    parser.add_argument("--target-file", default="transfuser_target.gz")
    return parser.parse_args()


def load_ordered_tokens(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        tokens = payload
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        tokens = [record["token"] for record in payload["records"]]
    else:
        raise ValueError(f"Unsupported token manifest schema: {path}")
    tokens = [str(token) for token in tokens]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Token manifest contains duplicates: {path}")
    return tokens


def load_metric_token_logs(cache_path: Path) -> dict[str, str]:
    metadata_dir = cache_path / "metadata"
    metadata_files = sorted(metadata_dir.glob("*.csv"))
    if not metadata_files:
        raise FileNotFoundError(f"No metric-cache metadata CSV found in {metadata_dir}")

    token_logs: dict[str, str] = {}
    for metadata_file in metadata_files:
        with metadata_file.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                metric_path = Path(row["file_name"])
                token = metric_path.parent.name
                log_name = metric_path.parents[2].name
                previous = token_logs.setdefault(token, log_name)
                if previous != log_name:
                    raise ValueError(
                        f"Metric token {token} appears in both {previous} and {log_name}"
                    )
    return token_logs


def scan_ready_feature_tokens(
    cache_path: Path,
    log_names: Iterable[str],
    feature_file: str,
    target_file: str,
) -> dict[str, str]:
    token_logs: dict[str, str] = {}
    for log_name in sorted(set(log_names)):
        log_path = cache_path / log_name
        if not log_path.is_dir():
            continue
        for token_path in log_path.iterdir():
            if not token_path.is_dir():
                continue
            if (token_path / feature_file).is_file() and (
                token_path / target_file
            ).is_file():
                previous = token_logs.setdefault(token_path.name, log_name)
                if previous != log_name:
                    raise ValueError(
                        f"Feature token {token_path.name} appears in both "
                        f"{previous} and {log_name}"
                    )
    return token_logs


def stable_digest(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode("utf-8")).hexdigest()


def stratified_hash_sample(
    token_logs: Mapping[str, str], limit: int, seed: int
) -> list[str]:
    """Proportionally allocate samples to logs, then hash-rank within each log."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit > len(token_logs):
        raise ValueError(f"Requested {limit} tokens from a pool of {len(token_logs)}")

    by_log: dict[str, list[str]] = defaultdict(list)
    for token, log_name in token_logs.items():
        by_log[log_name].append(token)

    exact_allocations = {
        log_name: len(tokens) * limit / len(token_logs)
        for log_name, tokens in by_log.items()
    }
    allocations = {
        log_name: math.floor(exact) for log_name, exact in exact_allocations.items()
    }
    remainder = limit - sum(allocations.values())
    remainder_order = sorted(
        by_log,
        key=lambda log_name: (
            -(exact_allocations[log_name] - allocations[log_name]),
            stable_digest(seed, "log", log_name),
        ),
    )
    for log_name in remainder_order[:remainder]:
        allocations[log_name] += 1

    # A development set should cover every available log when its requested
    # size permits. Repair zero allocations by moving one slot from the most
    # over-allocated donor; proportional counts otherwise stay unchanged.
    if limit >= len(by_log):
        empty_logs = sorted(
            (log_name for log_name, count in allocations.items() if count == 0),
            key=lambda log_name: stable_digest(seed, "empty-log", log_name),
        )
        for empty_log in empty_logs:
            donors = [
                log_name for log_name, count in allocations.items() if count > 1
            ]
            if not donors:
                raise AssertionError("Unable to give every log a sample")
            donor = min(
                donors,
                key=lambda log_name: (
                    -(allocations[log_name] - exact_allocations[log_name]),
                    -allocations[log_name],
                    stable_digest(seed, "donor-log", log_name),
                ),
            )
            allocations[donor] -= 1
            allocations[empty_log] = 1

    selected: list[str] = []
    for log_name, tokens in by_log.items():
        ranked = sorted(tokens, key=lambda token: stable_digest(seed, "token", token))
        selected.extend(ranked[: allocations[log_name]])
    selected.sort(key=lambda token: stable_digest(seed, "order", token))
    if len(selected) != limit or len(selected) != len(set(selected)):
        raise AssertionError("Stratified sampler did not return the requested unique count")
    return selected


def ordered_token_sha256(tokens: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()


def count_by_split(token_logs: Mapping[str, str], logs: set[str]) -> int:
    return sum(log_name in logs for log_name in token_logs.values())


def main() -> None:
    args = parse_args()
    if not args.feature_cache_path.is_dir():
        raise FileNotFoundError(args.feature_cache_path)
    if not args.metric_cache_path.is_dir():
        raise FileNotFoundError(args.metric_cache_path)
    if not args.navtest_metric_cache_path.is_dir():
        raise FileNotFoundError(args.navtest_metric_cache_path)

    split = OmegaConf.load(args.split_config)
    train_logs = set(map(str, split.train_logs))
    val_logs = set(map(str, split.val_logs))
    if train_logs & val_logs:
        raise RuntimeError("Configured train_logs and val_logs overlap")

    metric_token_logs = load_metric_token_logs(args.metric_cache_path)
    navtest_tokens = set(load_metric_token_logs(args.navtest_metric_cache_path))
    ready_token_logs = scan_ready_feature_tokens(
        args.feature_cache_path,
        train_logs | val_logs,
        args.feature_file,
        args.target_file,
    )
    fixed_tokens = load_ordered_tokens(args.fixed_tokens_file)
    fixed_set = set(fixed_tokens)

    metric_train = {
        token for token, log_name in metric_token_logs.items() if log_name in train_logs
    }
    metric_val = {
        token for token, log_name in metric_token_logs.items() if log_name in val_logs
    }
    unknown_metric = set(metric_token_logs) - metric_train - metric_val
    if unknown_metric:
        first = sorted(unknown_metric)[0]
        raise RuntimeError(
            f"{len(unknown_metric)} metric tokens are outside train/val logs; first={first}"
        )
    if not fixed_set <= metric_val:
        missing = sorted(fixed_set - metric_val)
        raise RuntimeError(
            f"Fixed manifest is not a subset of metric-cache val tokens; first={missing[0]}"
        )

    candidate_logs = {
        token: metric_token_logs[token] for token in metric_val - fixed_set
    }
    selected = stratified_hash_sample(candidate_logs, args.limit, args.seed)
    selected_set = set(selected)
    train_overlap = selected_set & metric_train
    fixed_overlap = selected_set & fixed_set
    navtest_overlap = selected_set & navtest_tokens
    if train_overlap or fixed_overlap or navtest_overlap:
        raise RuntimeError(
            "Selected dev tokens overlap a protected split: "
            f"train={len(train_overlap)}, fixed={len(fixed_overlap)}, "
            f"navtest={len(navtest_overlap)}"
        )

    ready_tokens = set(ready_token_logs)
    records = [
        {
            "token": token,
            "log_name": candidate_logs[token],
            "split": "val",
            "feature_target_ready": token in ready_tokens,
        }
        for token in selected
    ]
    ready_selected = sum(record["feature_target_ready"] for record in records)
    rewardable_train = metric_train & ready_tokens
    rewardable_val = metric_val & ready_tokens

    payload = {
        "schema_version": 1,
        "summary": {
            "selection": "proportional_log_stratified_sha256",
            "seed": args.seed,
            "num_tokens": len(records),
            "ordered_token_sha256": ordered_token_sha256(selected),
            "candidate_val_tokens": len(candidate_logs),
            "selected_feature_target_ready": ready_selected,
            "selected_feature_target_missing": len(records) - ready_selected,
            "feature_target_train": count_by_split(ready_token_logs, train_logs),
            "feature_target_val": count_by_split(ready_token_logs, val_logs),
            "metric_train": len(metric_train),
            "metric_val": len(metric_val),
            "rewardable_train": len(rewardable_train),
            "rewardable_val": len(rewardable_val),
            "fixed_tokens": len(fixed_set),
            "train_overlap": len(train_overlap),
            "fixed_overlap": len(fixed_overlap),
            "navtest_overlap": len(navtest_overlap),
            "feature_cache_path": str(args.feature_cache_path.resolve()),
            "metric_cache_path": str(args.metric_cache_path.resolve()),
            "navtest_metric_cache_path": str(args.navtest_metric_cache_path.resolve()),
            "fixed_tokens_file": str(args.fixed_tokens_file.resolve()),
            "split_config": str(args.split_config.resolve()),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
