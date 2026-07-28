#!/usr/bin/env python3
"""Train the Stage33 selector reranker from candidate-bank PDM labels.

This is an offline selector experiment (C2).  It never updates the diffusion
generator and never exposes PDM labels to the reranker at inference time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

import torch

from navsim.agents.diffusiondrive.stage33_external_reranker import (
    FEATURE_DIM,
    NUM_CANDIDATES,
    Stage33ExternalReranker,
    build_candidate_feature_matrix,
    compute_stage33_reranker_loss,
)


def _load_records(paths: Iterable[Path]) -> List[Mapping]:
    records = []
    for path in paths:
        payload = json.loads(path.read_text())
        records.extend(payload.get("records", []))
    if not records:
        raise RuntimeError("No candidate-bank records were found")
    return records


def _sha256_paths(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.resolve()).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _split(records: Sequence[Mapping], holdout_fraction: float, seed: int):
    train, valid = [], []
    for record in records:
        key = f"{seed}:{record.get('log_name', '')}:{record.get('token', '')}".encode()
        bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
        (valid if bucket < holdout_fraction else train).append(record)
    if not train or not valid:
        raise RuntimeError("Deterministic split produced an empty train/validation partition")
    return train, valid


def _tensor_records(records: Sequence[Mapping]):
    features, rewards = [], []
    for record in records:
        reward = torch.as_tensor(record.get("candidate_rewards"), dtype=torch.float32)
        if reward.shape != (NUM_CANDIDATES,):
            continue
        features.append(build_candidate_feature_matrix(record))
        rewards.append(torch.nan_to_num(reward, nan=-1e4, posinf=-1e4, neginf=-1e4))
    if not features:
        raise RuntimeError("No records with candidate_rewards and candidate_trajectories")
    return torch.stack(features), torch.stack(rewards)


def _run_epoch(model, features, rewards, optimizer, batch_size, device, train):
    model.train(train)
    order = torch.randperm(features.shape[0]) if train else torch.arange(features.shape[0])
    losses = []
    for start in range(0, features.shape[0], batch_size):
        idx = order[start : start + batch_size]
        x, y = features[idx].to(device), rewards[idx].to(device)
        with torch.set_grad_enabled(train):
            outputs = model(x)
            result = compute_stage33_reranker_loss(outputs, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        losses.append(float(result["loss"].detach().cpu()))
    return sum(losses) / max(1, len(losses))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, action="append", default=[])
    parser.add_argument("--bank-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3300)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.bank and args.bank_dir:
        args.bank = sorted(args.bank_dir.glob("*.json"))
    if not args.bank:
        raise SystemExit("Pass --bank or --bank-dir")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_records, valid_records = _split(
        _load_records(args.bank), args.holdout_fraction, args.seed
    )
    train_x, train_y = _tensor_records(train_records)
    valid_x, valid_y = _tensor_records(valid_records)
    device = torch.device(args.device)
    model = Stage33ExternalReranker(feature_dim=FEATURE_DIM, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(model, train_x, train_y, optimizer, args.batch_size, device, True)
        valid_loss = _run_epoch(model, valid_x, valid_y, optimizer, args.batch_size, device, False)
        print(f"epoch={epoch} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}", flush=True)
        if valid_loss < best:
            best = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No valid reranker checkpoint was produced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_hash = _sha256_paths(args.bank)
    torch.save(
        {
            "version": "stage33_relative_harm_reranker_v1",
            "feature_dim": FEATURE_DIM,
            "hidden_dim": args.hidden_dim,
            "state_dict": best_state,
            "manifest_sha256": manifest_hash,
            "seed": args.seed,
            "train_records": len(train_x),
            "valid_records": len(valid_x),
            "best_valid_loss": best,
        },
        args.output,
    )
    print(f"saved={args.output} train={len(train_x)} valid={len(valid_x)}", flush=True)


if __name__ == "__main__":
    main()
