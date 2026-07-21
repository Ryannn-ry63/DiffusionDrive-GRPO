#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 || $# -gt 10 ]]; then
  echo "Usage: $0 GENERATOR_CHECKPOINT SELECTOR_CHECKPOINT OUTPUT_JSON TOKENS_FILE LIMIT MARGIN LOG_SPLIT [CUDA_DEVICE=0] [CACHE_PATH=default] [METRIC_CACHE_PATH=default]"
  exit 2
fi

GENERATOR_CHECKPOINT="$1"
SELECTOR_CHECKPOINT="$2"
OUTPUT_JSON="$3"
TOKENS_FILE="$4"
LIMIT="$5"
MARGIN="$6"
LOG_SPLIT="$7"
CUDA_DEVICE="${8:-0}"
CACHE_PATH="${9:-default}"
METRIC_CACHE_PATH="${10:-default}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"
DEFAULT_CACHE="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache"
DEFAULT_METRIC_CACHE="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/metric_cache_trainval"

[[ -f "$GENERATOR_CHECKPOINT" ]] || { echo "Generator checkpoint does not exist: $GENERATOR_CHECKPOINT"; exit 2; }
[[ -f "$SELECTOR_CHECKPOINT" ]] || { echo "Selector checkpoint does not exist: $SELECTOR_CHECKPOINT"; exit 2; }
[[ -f "$BASE_CHECKPOINT" ]] || { echo "Base checkpoint does not exist: $BASE_CHECKPOINT"; exit 2; }
[[ -f "$TOKENS_FILE" ]] || { echo "Tokens file does not exist: $TOKENS_FILE"; exit 2; }
[[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "LIMIT must be positive"; exit 2; }
[[ "$MARGIN" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "MARGIN must be non-negative"; exit 2; }
[[ "$LOG_SPLIT" == train || "$LOG_SPLIT" == val || "$LOG_SPLIT" == test ]] || { echo "LOG_SPLIT must be train, val, or test"; exit 2; }
[[ "$CUDA_DEVICE" =~ ^[0-7]$ ]] || { echo "CUDA_DEVICE must be 0 through 7"; exit 2; }
[[ "$CACHE_PATH" == default ]] && CACHE_PATH="$DEFAULT_CACHE"
[[ "$METRIC_CACHE_PATH" == default ]] && METRIC_CACHE_PATH="$DEFAULT_METRIC_CACHE"

mkdir -p "$(dirname "$OUTPUT_JSON")"
cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" \
  scripts/evaluation/evaluate_grpo_schedule.py \
  --checkpoint "$GENERATOR_CHECKPOINT" \
  --reference-checkpoint "$BASE_CHECKPOINT" \
  --selector-logits-source value_top2 \
  --value-selector-checkpoint "$SELECTOR_CHECKPOINT" \
  --value-selector-calibration-margin "$MARGIN" \
  --cache-path "$CACHE_PATH" \
  --metric-cache-path "$METRIC_CACHE_PATH" \
  --tokens-file "$TOKENS_FILE" \
  --limit "$LIMIT" \
  --log-split "$LOG_SPLIT" \
  --batch-size 2 \
  --num-workers 4 \
  --device cuda:0 \
  --truncation-timestep 8 \
  --roll-timesteps 8 0 \
  --scheduler-num-inference-steps 125 \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260721 \
  --output "$OUTPUT_JSON"
