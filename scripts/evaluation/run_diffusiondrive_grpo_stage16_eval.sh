#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 || $# -gt 10 ]]; then
  echo "Usage: $0 original|full CHECKPOINT OUTPUT_JSON TOKENS_FILE LIMIT LOG_SPLIT CUDA_DEVICE [CACHE_PATH=default] [METRIC_CACHE_PATH=default] [BATCH_SIZE=2]"
  exit 2
fi

SCHEDULE="$1"
CHECKPOINT="$2"
OUTPUT_JSON="$3"
TOKENS_FILE="$4"
LIMIT="$5"
LOG_SPLIT="$6"
CUDA_DEVICE="$7"
CACHE_PATH="${8:-default}"
METRIC_CACHE_PATH="${9:-default}"
BATCH_SIZE="${10:-2}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"
DEFAULT_CACHE="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache"
DEFAULT_METRIC_CACHE="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/metric_cache_trainval"

[[ "$SCHEDULE" == original || "$SCHEDULE" == full ]] || { echo "SCHEDULE must be original or full"; exit 2; }
[[ -f "$CHECKPOINT" && -f "$BASE_CHECKPOINT" && -f "$TOKENS_FILE" ]] || { echo "checkpoint, base, or tokens file missing"; exit 2; }
[[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "LIMIT must be positive"; exit 2; }
[[ "$LOG_SPLIT" == train || "$LOG_SPLIT" == val || "$LOG_SPLIT" == test ]] || { echo "invalid LOG_SPLIT"; exit 2; }
[[ "$CUDA_DEVICE" =~ ^[0-7]$ ]] || { echo "invalid CUDA_DEVICE"; exit 2; }
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "BATCH_SIZE must be positive"; exit 2; }
[[ "$CACHE_PATH" == default ]] && CACHE_PATH="$DEFAULT_CACHE"
[[ "$METRIC_CACHE_PATH" == default ]] && METRIC_CACHE_PATH="$DEFAULT_METRIC_CACHE"

TRUNCATION=8
TIMESTEPS=(8 0)
ALGORITHM=legacy_ppo
if [[ "$SCHEDULE" == full ]]; then
  TRUNCATION=32
  TIMESTEPS=(32 24 16 8 0)
  ALGORITHM=diffgrpo_full_chain
fi

mkdir -p "$(dirname "$OUTPUT_JSON")"
cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" scripts/evaluation/evaluate_grpo_schedule.py \
  --checkpoint "$CHECKPOINT" --reference-checkpoint "$BASE_CHECKPOINT" \
  --selector-logits-source reference \
  --generation-policy-algorithm "$ALGORITHM" \
  --cache-path "$CACHE_PATH" --metric-cache-path "$METRIC_CACHE_PATH" \
  --tokens-file "$TOKENS_FILE" --limit "$LIMIT" --log-split "$LOG_SPLIT" \
  --batch-size "$BATCH_SIZE" --num-workers 4 --device cuda:0 \
  --truncation-timestep "$TRUNCATION" --roll-timesteps "${TIMESTEPS[@]}" \
  --scheduler-num-inference-steps 125 \
  --bootstrap-samples 10000 --bootstrap-seed 20260721 --output "$OUTPUT_JSON"
