#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 CHECKPOINT OUTPUT_JSON [LIMIT=256] [CUDA_DEVICE=0]"
  exit 2
fi

CHECKPOINT="$1"
OUTPUT_JSON="$2"
LIMIT="${3:-256}"
CUDA_DEVICE="${4:-0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_ARTIFACT="$ROOT_DIR/artifacts/grpo_stage0/holdout1024_aligned_t8.json"

if [[ "$LIMIT" != "256" && "$LIMIT" != "1024" ]]; then
  echo "LIMIT must be 256 or 1024"
  exit 2
fi

cd "$ROOT_DIR"
EVAL_ARGS=(
  scripts/evaluation/evaluate_grpo_schedule.py
  --checkpoint "$CHECKPOINT"
  --tokens-file "$BASE_ARTIFACT"
  --baseline-artifact "$BASE_ARTIFACT"
  --limit "$LIMIT"
  --batch-size 2
  --num-workers 4
  --device cuda:0
  --truncation-timestep 8
  --roll-timesteps 8 0
  --scheduler-num-inference-steps 125
  --bootstrap-samples 10000
  --bootstrap-seed 20260716
  --output "$OUTPUT_JSON"
)
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" "${EVAL_ARGS[@]}"
