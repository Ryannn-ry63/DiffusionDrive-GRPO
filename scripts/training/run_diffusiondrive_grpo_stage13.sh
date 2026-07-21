#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 audit|formal SEED EXPERIMENT_NAME CUDA_DEVICE"
  exit 2
fi

PHASE="$1"
SEED="$2"
EXPERIMENT_NAME="$3"
CUDA_DEVICE="$4"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_CHECKPOINT="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model"

[[ "$PHASE" == audit || "$PHASE" == formal ]] || { echo "PHASE must be audit or formal"; exit 2; }
[[ "$SEED" =~ ^[012]$ ]] || { echo "SEED must be 0, 1, or 2"; exit 2; }
[[ "$CUDA_DEVICE" =~ ^[0-7]$ ]] || { echo "CUDA_DEVICE must be an integer from 0 to 7"; exit 2; }
[[ "$EXPERIMENT_NAME" =~ ^stage13_[A-Za-z0-9_.-]+$ ]] || { echo "EXPERIMENT_NAME must start with stage13_"; exit 2; }
[[ -f "$BASE_CHECKPOINT" ]] || { echo "Locked base checkpoint does not exist"; exit 2; }
if [[ -n "${GRPO_BASE_CHECKPOINT:-}" && "$GRPO_BASE_CHECKPOINT" != "$BASE_CHECKPOINT" ]]; then
  echo "GRPO_BASE_CHECKPOINT must equal the locked Stage-13 base"
  exit 2
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$CUDA_DEVICE")"
[[ "$GPU_NAME" == *"RTX 4090"* ]] || { echo "Stage 13 requires RTX 4090; got: $GPU_NAME"; exit 2; }

MAX_STEPS=8
[[ "$PHASE" == formal ]] && MAX_STEPS=128

echo "stage13 phase=$PHASE seed=$SEED max_steps=$MAX_STEPS fresh_base=$BASE_CHECKPOINT gpu=$CUDA_DEVICE ($GPU_NAME)"
GRPO_BASE_CHECKPOINT="$BASE_CHECKPOINT" \
  bash "$ROOT_DIR/scripts/training/run_diffusiondrive_grpo_stage9.sh" \
  generation_group "$BASE_CHECKPOINT" "$EXPERIMENT_NAME" 1 none \
  "$SEED" "$CUDA_DEVICE" "$MAX_STEPS"
