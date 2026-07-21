#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 8 || $# -gt 10 ]]; then
  echo "Usage: $0 CHECKPOINT ADAPTER THRESHOLD OUTPUT TOKENS LIMIT LOG_SPLIT CUDA_DEVICE [CACHE=default] [METRIC_CACHE=default]"
  exit 2
fi
CHECKPOINT="$1"; ADAPTER="$2"; THRESHOLD="$3"; OUTPUT="$4"; TOKENS="$5"
LIMIT="$6"; LOG_SPLIT="$7"; CUDA_DEVICE="$8"; CACHE="${9:-default}"; METRIC_CACHE="${10:-default}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"
CALIBRATION_PATH="${STAGE17_CALIBRATION_PATH:-}"
GENERATION_POLICY_ALGORITHM="${GENERATION_POLICY_ALGORITHM:-diffgrpo_full_chain}"
EVALUATION_NOISE_NAMESPACE="${EVALUATION_NOISE_NAMESPACE:--1}"
EXPECTED_GENERATOR_SHA256="${EXPECTED_GENERATOR_SHA256:-}"
[[ "$CACHE" == default ]] && CACHE="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache"
[[ "$METRIC_CACHE" == default ]] && METRIC_CACHE="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/metric_cache_trainval"
[[ -f "$CHECKPOINT" && -f "$ADAPTER" && -f "$TOKENS" && -f "$BASE_CHECKPOINT" ]] || { echo "missing input"; exit 2; }
[[ "$CUDA_DEVICE" =~ ^[0-7]$ && "$LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "invalid device/limit"; exit 2; }
[[ -z "$CALIBRATION_PATH" || -f "$CALIBRATION_PATH" ]] || { echo "calibration file missing"; exit 2; }
[[ "$GENERATION_POLICY_ALGORITHM" == diffgrpo_full_chain || "$GENERATION_POLICY_ALGORITHM" == diffgrpo_selected_anchor ]] || { echo "invalid generation algorithm"; exit 2; }
[[ "$EVALUATION_NOISE_NAMESPACE" =~ ^-1$|^[0-9]+$ ]] || { echo "invalid noise namespace"; exit 2; }
CALIBRATION_ARGS=()
[[ -n "$CALIBRATION_PATH" ]] && CALIBRATION_ARGS+=(--paired-risk-calibration-path "$CALIBRATION_PATH")
[[ -n "$EXPECTED_GENERATOR_SHA256" ]] && CALIBRATION_ARGS+=(--paired-risk-expected-generator-sha256 "$EXPECTED_GENERATOR_SHA256")
mkdir -p "$(dirname "$OUTPUT")"
cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" scripts/evaluation/evaluate_grpo_schedule.py \
  --checkpoint "$CHECKPOINT" --reference-checkpoint "$BASE_CHECKPOINT" \
  --selector-logits-source paired_tail_risk --paired-risk-checkpoint "$ADAPTER" \
  --paired-risk-threshold "$THRESHOLD" --generation-policy-algorithm "$GENERATION_POLICY_ALGORITHM" \
  --evaluation-noise-namespace "$EVALUATION_NOISE_NAMESPACE" \
  "${CALIBRATION_ARGS[@]}" \
  --cache-path "$CACHE" --metric-cache-path "$METRIC_CACHE" --tokens-file "$TOKENS" \
  --limit "$LIMIT" --log-split "$LOG_SPLIT" --batch-size 2 --num-workers 4 --device cuda:0 \
  --truncation-timestep 32 --roll-timesteps 32 24 16 8 0 \
  --scheduler-num-inference-steps 125 --bootstrap-samples 10000 --bootstrap-seed 20260721 \
  --output "$OUTPUT"
