#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 10 ]]; then
  echo "Usage: $0 current|reference CHECKPOINT OUTPUT_JSON TOKENS_FILE LIMIT [BASELINE_ARTIFACT=none] [CACHE_PATH=default] [METRIC_CACHE_PATH=default] [LOG_SPLIT=val] [CUDA_DEVICE=0]"
  exit 2
fi

SELECTOR_SOURCE="$1"
CHECKPOINT="$2"
OUTPUT_JSON="$3"
TOKENS_FILE="$4"
LIMIT="$5"
BASELINE_ARTIFACT="${6:-none}"
CACHE_PATH="${7:-default}"
METRIC_CACHE_PATH="${8:-default}"
LOG_SPLIT="${9:-val}"
CUDA_DEVICE="${10:-0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"
DEFAULT_CACHE="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache"
DEFAULT_METRIC_CACHE="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/metric_cache_trainval"

[[ "$SELECTOR_SOURCE" == current || "$SELECTOR_SOURCE" == reference ]] || { echo "selector source must be current or reference"; exit 2; }
[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint does not exist: $CHECKPOINT"; exit 2; }
[[ -f "$BASE_CHECKPOINT" ]] || { echo "Base checkpoint does not exist: $BASE_CHECKPOINT"; exit 2; }
[[ -f "$TOKENS_FILE" ]] || { echo "Tokens file does not exist: $TOKENS_FILE"; exit 2; }
[[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "LIMIT must be positive"; exit 2; }
[[ "$LOG_SPLIT" == train || "$LOG_SPLIT" == val ]] || { echo "LOG_SPLIT must be train or val"; exit 2; }
if [[ "$BASELINE_ARTIFACT" != none && ! -f "$BASELINE_ARTIFACT" ]]; then
  echo "Baseline artifact does not exist: $BASELINE_ARTIFACT"
  exit 2
fi
[[ "$CACHE_PATH" == default ]] && CACHE_PATH="$DEFAULT_CACHE"
[[ "$METRIC_CACHE_PATH" == default ]] && METRIC_CACHE_PATH="$DEFAULT_METRIC_CACHE"

ARGS=(
  scripts/evaluation/evaluate_grpo_schedule.py
  --checkpoint "$CHECKPOINT"
  --reference-checkpoint "$BASE_CHECKPOINT"
  --selector-logits-source "$SELECTOR_SOURCE"
  --cache-path "$CACHE_PATH"
  --metric-cache-path "$METRIC_CACHE_PATH"
  --tokens-file "$TOKENS_FILE"
  --limit "$LIMIT"
  --log-split "$LOG_SPLIT"
  --batch-size 2
  --num-workers 4
  --device cuda:0
  --truncation-timestep 8
  --roll-timesteps 8 0
  --scheduler-num-inference-steps 125
  --bootstrap-samples 10000
  --bootstrap-seed 20260719
  --output "$OUTPUT_JSON"
)
if [[ "$BASELINE_ARTIFACT" != none ]]; then
  ARGS+=(--baseline-artifact "$BASELINE_ARTIFACT")
fi

mkdir -p "$(dirname "$OUTPUT_JSON")"
cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" "${ARGS[@]}"
