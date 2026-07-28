#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [P|A|B0|B|C|all]"
  exit 2
fi
SYSTEM="${1:-all}"
[[ "$SYSTEM" == P || "$SYSTEM" == A || "$SYSTEM" == B0 || "$SYSTEM" == B || \
   "$SYSTEM" == C || "$SYSTEM" == all ]] || {
  echo "SYSTEM must be P, A, B0, B, C, or all"
  exit 2
}

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
export STAGE26_NAVTEST_METRIC_CACHE="${STAGE26_NAVTEST_METRIC_CACHE:-$NAVSIM_EXP_ROOT/metric_cache_navtest}"

cd "$ROOT_DIR"
if [[ "$SYSTEM" == all ]]; then
  for item in A B0 B C; do
    scripts/evaluation/run_diffusiondrive_grpo_stage26_navtest_pdms.sh \
      final "$item" 0,1,2,3,4,5,6,7
  done
else
  scripts/evaluation/run_diffusiondrive_grpo_stage26_navtest_pdms.sh \
    final "$SYSTEM" 0,1,2,3,4,5,6,7
fi
