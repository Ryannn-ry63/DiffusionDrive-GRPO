#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 10 ]]; then
  echo "Usage: $0 collect|deploy GENERATOR DOMAIN SELECTOR CALIBRATION|- MANIFEST LIMIT NOISE OUTPUT GPU"
  exit 2
fi
MODE="$1"; GENERATOR="$2"; DOMAIN="$3"; SELECTOR="$4"; CALIBRATION="$5"
MANIFEST="$6"; LIMIT="$7"; NOISE="$8"; OUTPUT="$9"; GPU="${10}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"
TRAINING_CACHE="${NAVSIM_TRAINING_CACHE:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache}"
METRIC_CACHE="${NAVSIM_METRIC_CACHE:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/metric_cache_trainval}"
GENERATION_ALGORITHM="${GRPO_EVAL_GENERATION_ALGORITHM:-legacy_ppo}"

[[ "$MODE" == collect || "$MODE" == deploy ]] || { echo "invalid mode"; exit 2; }
for path in "$GENERATOR" "$SELECTOR" "$MANIFEST" "$BASE_CHECKPOINT"; do
  [[ -f "$path" ]] || { echo "missing Stage25 input: $path"; exit 2; }
done
[[ "$LIMIT" =~ ^[1-9][0-9]*$ && "$GPU" =~ ^[0-7]$ ]] || { echo "invalid limit/GPU"; exit 2; }
[[ "$DOMAIN" =~ ^[a-z0-9_]+$ ]] || { echo "invalid generator domain"; exit 2; }
mkdir -p "$(dirname "$OUTPUT")"

SELECTOR_ARGS=(
  --selector-logits-source trajectory_relative_harm_v3
  --stage25-selector-checkpoint "$SELECTOR"
)
if [[ "$MODE" == collect ]]; then
  [[ "$CALIBRATION" == - ]] || { echo "collect mode requires calibration '-'"; exit 2; }
  SELECTOR_ARGS+=(--stage25-collect-calibration)
else
  [[ -f "$CALIBRATION" ]] || { echo "missing Stage25 calibration"; exit 2; }
  read -r MARGIN RISK OOD < <("$PYTHON_BIN" -c '
import json,sys
p=json.load(open(sys.argv[1])); assert p["passed"]
print(p["residual_margin"],p["risk_threshold"],p["ood_threshold"])
' "$CALIBRATION")
  SELECTOR_ARGS+=(
    --stage25-selector-calibration "$CALIBRATION"
    --stage24-selector-residual-margin "$MARGIN"
    --stage25-selector-risk-threshold "$RISK"
    --stage24-selector-ood-threshold "$OOD"
  )
fi

if [[ "${GRPO_SELECTOR_EVAL_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "PASS Stage25 selector eval preflight domain=$DOMAIN limit=$LIMIT GPU=$GPU"
  exit 0
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
  "$ROOT_DIR/scripts/evaluation/evaluate_grpo_schedule.py" \
  --checkpoint "$GENERATOR" --reference-checkpoint "$BASE_CHECKPOINT" \
  --cache-path "$TRAINING_CACHE" --metric-cache-path "$METRIC_CACHE" \
  --tokens-file "$MANIFEST" --limit "$LIMIT" --log-split train \
  --batch-size 2 --num-workers 4 --device cuda:0 \
  --truncation-timestep 32 --roll-timesteps 32 24 16 8 0 \
  --scheduler-num-inference-steps 125 --generation-policy-algorithm "$GENERATION_ALGORITHM" \
  --evaluation-noise-namespace "$NOISE" --generator-domain "$DOMAIN" \
  "${SELECTOR_ARGS[@]}" --output "$OUTPUT"
