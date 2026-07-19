#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: $0 1|2 8|128 EXPERIMENT_NAME [RESUME_CHECKPOINT=none] [CUDA_DEVICE=0]"
  exit 2
fi

SEED="$1"
TARGET_STEPS="$2"
EXPERIMENT_NAME="$3"
RESUME_CHECKPOINT="${4:-none}"
CUDA_DEVICE="${5:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"

[[ "$SEED" == 1 || "$SEED" == 2 ]] || { echo "Stage 11 trains only independent seeds 1 and 2"; exit 2; }
[[ "$TARGET_STEPS" == 8 || "$TARGET_STEPS" == 128 ]] || { echo "TARGET_STEPS must be the U8 audit or U128 run"; exit 2; }
[[ -f "$BASE_CHECKPOINT" ]] || { echo "Base checkpoint does not exist: $BASE_CHECKPOINT"; exit 2; }
if [[ "$TARGET_STEPS" == 128 && "$RESUME_CHECKPOINT" == none ]]; then
  echo "U128 must resume the audited U8 run"
  exit 2
fi

exec "$ROOT_DIR/scripts/training/run_diffusiondrive_grpo_stage10.sh" \
  layer0_01 "$BASE_CHECKPOINT" "$EXPERIMENT_NAME" "$TARGET_STEPS" \
  "$RESUME_CHECKPOINT" "$SEED" "$CUDA_DEVICE"
