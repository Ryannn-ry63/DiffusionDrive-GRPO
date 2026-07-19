#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 SEED TARGET_STEPS EXPERIMENT_NAME RESUME_CHECKPOINT [CUDA_DEVICE=0]"
  exit 2
fi

SEED="$1"
TARGET_STEPS="$2"
EXPERIMENT_NAME="$3"
RESUME_CHECKPOINT="$4"
CUDA_DEVICE="${5:-0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"
STAGE10_U128_CHECKPOINT="${GRPO_STAGE10_U128_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage10_layer0_01_u128/2026.07.19.04.43.02/lightning_logs/version_0/checkpoints/grpo-step-128.ckpt}"

[[ "$SEED" == 0 || "$SEED" == 1 || "$SEED" == 2 ]] || { echo "SEED must be 0, 1, or 2"; exit 2; }
[[ -f "$BASE_CHECKPOINT" ]] || { echo "Base checkpoint does not exist: $BASE_CHECKPOINT"; exit 2; }

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$CUDA_DEVICE")"
[[ "$GPU_NAME" == *"RTX 4090"* ]] || { echo "Stage 12 formal runs require RTX 4090; got: $GPU_NAME"; exit 2; }

if [[ "$SEED" == 0 ]]; then
  [[ "$TARGET_STEPS" == 256 ]] || { echo "Seed 0 only permits the registered U128-to-U256 continuation"; exit 2; }
  [[ -f "$RESUME_CHECKPOINT" ]] || { echo "Resume checkpoint does not exist: $RESUME_CHECKPOINT"; exit 2; }
  [[ "$(readlink -f "$RESUME_CHECKPOINT")" == "$(readlink -f "$STAGE10_U128_CHECKPOINT")" ]] || {
    echo "Seed 0 must resume the registered Stage-10 U128 checkpoint"
    exit 2
  }
  EXPECTED_GLOBAL_STEP=128
else
  if [[ "$TARGET_STEPS" == 8 ]]; then
    [[ "$RESUME_CHECKPOINT" == none ]] || { echo "Seeds 1/2 U8 must start from base"; exit 2; }
    EXPECTED_GLOBAL_STEP=0
  else
    [[ "$TARGET_STEPS" == 256 ]] || { echo "Seeds 1/2 only permit U8 or audited U8-to-U256 continuation"; exit 2; }
    [[ -f "$RESUME_CHECKPOINT" ]] || { echo "Resume checkpoint does not exist: $RESUME_CHECKPOINT"; exit 2; }
    EXPECTED_GLOBAL_STEP=8
  fi
fi

if [[ "$EXPECTED_GLOBAL_STEP" -gt 0 ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/scripts/training/check_grpo_stage12_resume.py" \
    --checkpoint "$RESUME_CHECKPOINT" \
    --expected-global-step "$EXPECTED_GLOBAL_STEP"
fi

exec "$ROOT_DIR/scripts/training/run_diffusiondrive_grpo_stage10.sh" \
  layer0_01 "$BASE_CHECKPOINT" "$EXPERIMENT_NAME" "$TARGET_STEPS" \
  "$RESUME_CHECKPOINT" "$SEED" "$CUDA_DEVICE"
