#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 9 ]]; then
  echo "Usage: $0 GENERATOR SELECTOR MARGIN SAFETY_THRESHOLD MANIFEST LIMIT NOISE OUTPUT GPU"
  exit 2
fi
GENERATOR="$1"; SELECTOR="$2"; MARGIN="$3"; SAFETY_THRESHOLD="$4"
MANIFEST="$5"; LIMIT="$6"; NOISE="$7"; OUTPUT="$8"; GPU="$9"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"
TRAINING_CACHE="${NAVSIM_TRAINING_CACHE:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache}"
METRIC_CACHE="${NAVSIM_METRIC_CACHE:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/metric_cache_trainval}"

for path in "$GENERATOR" "$SELECTOR" "$MANIFEST" "$BASE_CHECKPOINT"; do
  [[ -f "$path" ]] || { echo "missing Stage23 input: $path"; exit 2; }
done
[[ "$LIMIT" =~ ^[1-9][0-9]*$ && "$GPU" =~ ^[0-7]$ ]] || { echo "invalid limit/GPU"; exit 2; }
mkdir -p "$(dirname "$OUTPUT")"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
  "$ROOT_DIR/scripts/evaluation/evaluate_grpo_schedule.py" \
  --checkpoint "$GENERATOR" --reference-checkpoint "$BASE_CHECKPOINT" \
  --cache-path "$TRAINING_CACHE" --metric-cache-path "$METRIC_CACHE" \
  --tokens-file "$MANIFEST" --limit "$LIMIT" --log-split train \
  --batch-size 2 --num-workers 4 --device cuda:0 \
  --truncation-timestep 32 --roll-timesteps 32 24 16 8 0 \
  --scheduler-num-inference-steps 125 \
  --generation-policy-algorithm diffgrpo_selected_anchor \
  --evaluation-noise-namespace "$NOISE" \
  --selector-logits-source trajectory_oof \
  --stage23-selector-checkpoint "$SELECTOR" \
  --stage23-selector-residual-margin "$MARGIN" \
  --stage23-selector-safety-threshold "$SAFETY_THRESHOLD" \
  --output "$OUTPUT"
