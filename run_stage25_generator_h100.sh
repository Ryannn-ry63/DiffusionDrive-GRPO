#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
EXP_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/inspire/hdd/global_public/public_datas/NAVSIM/maps
export NAVSIM_DEVKIT_ROOT="$ROOT_DIR/navsim"
export NAVSIM_EXP_ROOT="$EXP_ROOT"
export OPENSCENE_DATA_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export PYTHON_BIN=/root/miniconda3/envs/navsimH100/bin/python

SELECTOR="$EXP_ROOT/stage25_selector_formal_seed25025/2026.07.23.02.45.45/lightning_logs/version_0/checkpoints/grpo-11-18336.ckpt"
CALIBRATION="$ROOT_DIR/artifacts/grpo_stage25/stage25_selector_calibration_fold3.json"
GATE="$ROOT_DIR/artifacts/grpo_stage25/fold4/gate.json"
EXPERIMENT="${1:-stage25_generator_development8_seed0}"

cd "$ROOT_DIR"
exec scripts/training/run_diffusiondrive_grpo_stage25_generator.sh \
  development \
  "$SELECTOR" \
  "$CALIBRATION" \
  "$GATE" \
  "$EXPERIMENT" \
  0,1,2,3,4,5,6,7
