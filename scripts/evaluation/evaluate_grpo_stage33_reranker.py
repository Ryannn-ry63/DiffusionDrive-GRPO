#!/usr/bin/env python3
"""Offline C2 evaluation for the Stage33 external selector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch

from navsim.agents.diffusiondrive.stage33_external_reranker import (
    FEATURE_DIM,
    NUM_CANDIDATES,
    Stage33ExternalReranker,
    build_candidate_feature_matrix,
    select_stage33,
)


def _records(paths: Iterable[Path]):
    for path in paths:
        payload = json.loads(path.read_text())
        yield from payload.get("records", [])


def _find_eligible(record: Mapping[str, Any]) -> Optional[torch.Tensor]:
    for key in ("stage25_eligible", "eligible_modes", "eligible_mask"):
        value = record.get(key)
        if value is not None:
            mask = torch.as_tensor(value, dtype=torch.bool)
            if mask.shape == (NUM_CANDIDATES,):
                return mask
    nested = record.get("stage25")
    if isinstance(nested, Mapping):
        for key in ("eligible", "eligible_modes", "eligible_mask"):
            value = nested.get(key)
            if value is not None:
                mask = torch.as_tensor(value, dtype=torch.bool)
                if mask.shape == (NUM_CANDIDATES,):
                    return mask
    return None


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    xr = {v: i for i, v in enumerate(sorted(x))}
    yr = {v: i for i, v in enumerate(sorted(y))}
    a = torch.tensor([xr[v] for v in x], dtype=torch.float32)
    b = torch.tensor([yr[v] for v in y], dtype=torch.float32)
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.square().sum().sqrt() * b.square().sum().sqrt()).clamp_min(1e-8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, action="append", default=[])
    parser.add_argument("--bank-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fallback-mode", type=int, default=0)
    parser.add_argument("--risk-z", type=float, default=1.0)
    parser.add_argument("--fallback-margin", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.bank and args.bank_dir:
        args.bank = sorted(args.bank_dir.glob("*.json"))
    if not args.bank:
        raise SystemExit("Pass --bank or --bank-dir")
    payload = torch.load(args.checkpoint, map_location="cpu")
    model = Stage33ExternalReranker(
        feature_dim=int(payload.get("feature_dim", FEATURE_DIM)),
        hidden_dim=int(payload.get("hidden_dim", 128)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(torch.device(args.device)).eval()

    gains, regrets, fallback_flags, catastrophic = [], [], [], []
    rank_corr, count = [], 0
    for record in _records(args.bank):
        rewards = torch.as_tensor(record.get("candidate_rewards"), dtype=torch.float32)
        if rewards.shape != (NUM_CANDIDATES,):
            continue
        features = build_candidate_feature_matrix(record).unsqueeze(0).to(args.device)
        eligible = _find_eligible(record)
        if eligible is not None:
            eligible = eligible.unsqueeze(0).to(args.device)
        with torch.no_grad():
            outputs = model(features)
            selected = select_stage33(
                outputs,
                eligible_mask=eligible,
                fallback_mode=args.fallback_mode,
                risk_z=args.risk_z,
                fallback_margin=args.fallback_margin,
            )
        mode = int(selected["selected_mode"][0].cpu())
        chosen = float(rewards[mode])
        oracle = float(torch.nan_to_num(rewards, nan=-1e4).max())
        baseline = float(rewards[int(record.get("selected_mode", args.fallback_mode))])
        gains.append(chosen - baseline)
        regrets.append(oracle - chosen)
        fallback_flags.append(bool(selected["fallback_used"][0].cpu()))
        catastrophic.append(float(torch.sigmoid(outputs["catastrophe_logit"][0, mode]).cpu()))
        rank_corr.append(_spearman(rewards.tolist(), outputs["score_mean"][0].cpu().tolist()))
        count += 1
    if not count:
        raise RuntimeError("No valid records found for evaluation")
    summary = {
        "version": "stage33_relative_harm_reranker_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "num_records": count,
        "mean_gain_vs_stored_selector": sum(gains) / count,
        "mean_regret_to_oracle": sum(regrets) / count,
        "fallback_rate": sum(fallback_flags) / count,
        "mean_catastrophe_probability": sum(catastrophic) / count,
        "mean_reward_rank_spearman": sum(rank_corr) / count,
        "banks": [str(path.resolve()) for path in args.bank],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

