#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 || $# -gt 9 ]]; then
  echo "Usage: $0 A|B|C CHECKPOINT OUTPUT TOKENS LIMIT CUDA_DEVICE NOISE_NAMESPACE [CACHE=dev4096] [METRIC_CACHE=default]"
  exit 2
fi

CELL="$1"; CHECKPOINT="$2"; OUTPUT="$3"; TOKENS="$4"; LIMIT="$5"
CUDA_DEVICE="$6"; NOISE_NAMESPACE="$7"; CACHE="${8:-dev4096}"; METRIC_CACHE="${9:-default}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"

[[ "$CELL" == A || "$CELL" == B || "$CELL" == C ]] || { echo "invalid Stage20 cell"; exit 2; }
[[ -f "$CHECKPOINT" && -f "$TOKENS" && -f "$BASE_CHECKPOINT" ]] || { echo "missing checkpoint/tokens/base"; exit 2; }
[[ "$LIMIT" =~ ^[1-9][0-9]*$ && "$CUDA_DEVICE" =~ ^[0-7]$ ]] || { echo "invalid limit/device"; exit 2; }
[[ "$NOISE_NAMESPACE" == -1 || "$NOISE_NAMESPACE" == 20260726 ]] || { echo "unregistered Stage20 noise"; exit 2; }
if [[ "$CELL" == A || "$CELL" == B ]]; then
  [[ "$CHECKPOINT" -ef "$BASE_CHECKPOINT" ]] || { echo "Stage20 A/B must use the frozen base"; exit 2; }
fi
[[ "$CELL" != C || ! "$CHECKPOINT" -ef "$BASE_CHECKPOINT" ]] || { echo "Stage20 C cannot be the base checkpoint"; exit 2; }

if [[ "$CACHE" == dev4096 ]]; then
  CACHE="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/grpo_dev4096_feature_cache_20260717"
fi
if [[ "$METRIC_CACHE" == default ]]; then
  METRIC_CACHE="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/metric_cache_trainval"
fi
[[ -d "$CACHE" && -d "$METRIC_CACHE" ]] || { echo "Stage20 cache is missing"; exit 2; }

LEGACY_CELL=full
[[ "$CELL" == A ]] && LEGACY_CELL=original

exec bash "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage19_eval.sh" \
  "$LEGACY_CELL" "$CHECKPOINT" "$OUTPUT" "$TOKENS" "$LIMIT" val \
  "$CUDA_DEVICE" "$NOISE_NAMESPACE" "$CACHE" "$METRIC_CACHE" 2

