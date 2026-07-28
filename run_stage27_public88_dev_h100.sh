#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 candidate|selector JOB_ID [GPU=0]"
  echo "candidate JOB_ID: 0=default, 1=20260811, 2=20260812"
  echo "selector  JOB_ID: 0=20260821, 1=20260822"
  exit 2
fi

MODE="$1"
JOB_ID="$2"
GPU="${3:-0}"
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
export NAVSIM_METRIC_CACHE="${NAVSIM_METRIC_CACHE:-$NAVSIM_EXP_ROOT/metric_cache_trainval}"

cd "$ROOT_DIR"
scripts/evaluation/run_diffusiondrive_grpo_stage27_public88_dev.sh \
  "$MODE" "$JOB_ID" "$GPU"
