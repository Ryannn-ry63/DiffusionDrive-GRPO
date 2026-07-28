#!/usr/bin/env bash
set -euo pipefail

# Stage26 selector training is intentionally single-GPU.  Its locked bank
# scheduler defines 2,038 optimizer updates per epoch; converting this job to
# DDP would change the optimization trajectory and invalidate preregistration.
CUDA_DEVICE="${1:-0}"
[[ "$CUDA_DEVICE" =~ ^[0-7]$ ]] || { echo "Usage: $0 [CUDA_DEVICE=0]"; exit 2; }

ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/navsim}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
export NAVSIM_TRAINING_CACHE="${NAVSIM_TRAINING_CACHE:-$NAVSIM_EXP_ROOT/training_cache}"

STAGE24_EXPERIMENT=stage26_stage24_selector_formal_seed26024
STAGE25_EXPERIMENT=stage26_stage25_selector_formal_seed26025

cd "$ROOT_DIR"
scripts/training/run_diffusiondrive_grpo_stage26_stage24_selector.sh \
  formal "$STAGE24_EXPERIMENT" "$CUDA_DEVICE"

STAGE24_CHECKPOINT="$(
  find "$NAVSIM_EXP_ROOT/$STAGE24_EXPERIMENT" -type f \
    -name 'grpo-14-30570.ckpt' -printf '%T@ %p\n' \
    | sort -nr | head -1 | cut -d' ' -f2-
)"
[[ -n "$STAGE24_CHECKPOINT" && -f "$STAGE24_CHECKPOINT" ]] || {
  echo "mandatory Stage26 Stage24 final checkpoint was not found"
  exit 2
}

scripts/training/run_diffusiondrive_grpo_stage26_stage25_selector.sh \
  formal "$STAGE24_CHECKPOINT" "$STAGE25_EXPERIMENT" "$CUDA_DEVICE"

STAGE25_CHECKPOINT="$(
  find "$NAVSIM_EXP_ROOT/$STAGE25_EXPERIMENT" -type f \
    -name 'grpo-14-30570.ckpt' -printf '%T@ %p\n' \
    | sort -nr | head -1 | cut -d' ' -f2-
)"
[[ -n "$STAGE25_CHECKPOINT" && -f "$STAGE25_CHECKPOINT" ]] || {
  echo "mandatory Stage26 Stage25 final checkpoint was not found"
  exit 2
}

echo "STAGE26_STAGE24_CHECKPOINT=$STAGE24_CHECKPOINT"
sha256sum "$STAGE24_CHECKPOINT"
echo "STAGE26_STAGE25_CHECKPOINT=$STAGE25_CHECKPOINT"
sha256sum "$STAGE25_CHECKPOINT"
