#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 CHECKPOINT OUTPUT_JSON [BASELINE_ARTIFACT=none] [CUDA_DEVICE=0]"
  exit 2
fi

CHECKPOINT="$1"
OUTPUT_JSON="$2"
BASELINE_ARTIFACT="${3:-none}"
CUDA_DEVICE="${4:-0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEV_MANIFEST="$ROOT_DIR/artifacts/grpo_stage0/dev4096_manifest.json"
DEV_CACHE="${GRPO_DEV_CACHE:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/grpo_dev4096_feature_cache_20260717}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint does not exist: $CHECKPOINT"
  exit 2
fi
if [[ ! -f "$DEV_MANIFEST" ]]; then
  echo "Dev manifest does not exist: $DEV_MANIFEST"
  exit 2
fi

cd "$ROOT_DIR"
EVAL_ARGS=(
  scripts/evaluation/evaluate_grpo_schedule.py
  --checkpoint "$CHECKPOINT"
  --cache-path "$DEV_CACHE"
  --tokens-file "$DEV_MANIFEST"
  --limit 4096
  --batch-size 2
  --num-workers 4
  --device cuda:0
  --truncation-timestep 8
  --roll-timesteps 8 0
  --scheduler-num-inference-steps 125
  --bootstrap-samples 10000
  --bootstrap-seed 20260717
  --output "$OUTPUT_JSON"
)
if [[ "$BASELINE_ARTIFACT" != "none" ]]; then
  if [[ ! -f "$BASELINE_ARTIFACT" ]]; then
    echo "Baseline artifact does not exist: $BASELINE_ARTIFACT"
    exit 2
  fi
  EVAL_ARGS+=(--baseline-artifact "$BASELINE_ARTIFACT")
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" "${EVAL_ARGS[@]}"
