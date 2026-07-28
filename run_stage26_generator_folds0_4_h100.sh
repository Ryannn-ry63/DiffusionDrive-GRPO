#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/navsim}"
if [[ ! -f "$NAVSIM_DEVKIT_ROOT/planning/script/run_training.py" ]]; then
  export NAVSIM_DEVKIT_ROOT="$ROOT_DIR/navsim"
fi
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
export GRPO_TRAIN_CACHE="${GRPO_TRAIN_CACHE:-$NAVSIM_EXP_ROOT/training_cache}"

cd "$ROOT_DIR"
scripts/training/run_diffusiondrive_grpo_stage26_generator.sh \
  final stage26_generator_final_folds0_4_seed0 0,1,2,3,4,5,6,7
