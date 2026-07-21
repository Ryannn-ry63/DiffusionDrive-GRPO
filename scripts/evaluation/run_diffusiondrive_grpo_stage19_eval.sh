#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 8 || $# -gt 11 ]]; then
  echo "Usage: $0 original|full|risk CHECKPOINT OUTPUT TOKENS LIMIT LOG_SPLIT CUDA_DEVICE NOISE_NAMESPACE [CACHE=default] [METRIC_CACHE=default] [BATCH_SIZE=2]"
  exit 2
fi

CELL="$1"; CHECKPOINT="$2"; OUTPUT="$3"; TOKENS="$4"; LIMIT="$5"
LOG_SPLIT="$6"; CUDA_DEVICE="$7"; NOISE_NAMESPACE="$8"
CACHE="${9:-default}"; METRIC_CACHE="${10:-default}"; BATCH_SIZE="${11:-2}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"
STAGE17_ADAPTER="${STAGE17_ADAPTER:-}"
STAGE17_THRESHOLD="${STAGE17_THRESHOLD:-0.7683204412460327}"
STAGE17_CALIBRATION_PATH="${STAGE17_CALIBRATION_PATH:-}"
[[ "$CACHE" == default ]] && CACHE="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache"
[[ "$METRIC_CACHE" == default ]] && METRIC_CACHE="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/metric_cache_trainval"

[[ "$CELL" == original || "$CELL" == full || "$CELL" == risk ]] || { echo "invalid cell"; exit 2; }
[[ -f "$CHECKPOINT" && -f "$BASE_CHECKPOINT" && -f "$TOKENS" ]] || { echo "missing checkpoint/base/tokens"; exit 2; }
[[ "$LIMIT" =~ ^[1-9][0-9]*$ && "$CUDA_DEVICE" =~ ^[0-7]$ && "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "invalid numeric argument"; exit 2; }
[[ "$NOISE_NAMESPACE" =~ ^-1$|^[0-9]+$ ]] || { echo "invalid noise namespace"; exit 2; }

TRUNCATION=32
TIMESTEPS=(32 24 16 8 0)
ALGORITHM=diffgrpo_selected_anchor
SELECTOR_ARGS=(--selector-logits-source reference)
if [[ "$CELL" == original ]]; then
  TRUNCATION=8
  TIMESTEPS=(8 0)
  ALGORITHM=legacy_ppo
elif [[ "$CELL" == risk ]]; then
  [[ -f "$STAGE17_ADAPTER" ]] || { echo "risk cell requires STAGE17_ADAPTER"; exit 2; }
  EXPECTED_SHA="$($PYTHON_BIN -c 'import hashlib,sys; h=hashlib.sha256(); f=open(sys.argv[1],"rb"); [h.update(b) for b in iter(lambda:f.read(1048576),b"")]; print(h.hexdigest())' "$CHECKPOINT")"
  SELECTOR_ARGS=(
    --selector-logits-source paired_tail_risk
    --paired-risk-checkpoint "$STAGE17_ADAPTER"
    --paired-risk-threshold "$STAGE17_THRESHOLD"
    --paired-risk-expected-generator-sha256 "$EXPECTED_SHA"
  )
  [[ -z "$STAGE17_CALIBRATION_PATH" || -f "$STAGE17_CALIBRATION_PATH" ]] || { echo "calibration file missing"; exit 2; }
  [[ -z "$STAGE17_CALIBRATION_PATH" ]] || SELECTOR_ARGS+=(--paired-risk-calibration-path "$STAGE17_CALIBRATION_PATH")
fi

mkdir -p "$(dirname "$OUTPUT")"
cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" scripts/evaluation/evaluate_grpo_schedule.py \
  --checkpoint "$CHECKPOINT" --reference-checkpoint "$BASE_CHECKPOINT" \
  --generation-policy-algorithm "$ALGORITHM" --evaluation-noise-namespace "$NOISE_NAMESPACE" \
  "${SELECTOR_ARGS[@]}" --cache-path "$CACHE" --metric-cache-path "$METRIC_CACHE" \
  --tokens-file "$TOKENS" --limit "$LIMIT" --log-split "$LOG_SPLIT" \
  --batch-size "$BATCH_SIZE" --num-workers 4 --device cuda:0 \
  --truncation-timestep "$TRUNCATION" --roll-timesteps "${TIMESTEPS[@]}" \
  --scheduler-num-inference-steps 125 --bootstrap-samples 10000 \
  --bootstrap-seed 20260722 --output "$OUTPUT"
