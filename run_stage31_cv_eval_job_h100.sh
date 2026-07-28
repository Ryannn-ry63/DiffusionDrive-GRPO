#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then echo "Usage: $0 DP|DPF HOLDOUT(0..3)"; exit 2; fi
BRANCH="$1"; HOLDOUT="$2"
ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
export NAVSIM_DEVKIT_ROOT="$ROOT_DIR/navsim"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1 RAY_DEDUP_LOGS=0
export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
[[ "$BRANCH" == DP || "$BRANCH" == DPF ]] || { echo "bad branch"; exit 2; }
[[ "$HOLDOUT" =~ ^[0-3]$ ]] || { echo "bad holdout"; exit 2; }
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" -eq 8 ]] || { echo "Stage31 requires 8 GPUs"; exit 2; }
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$ROOT_DIR/artifacts/grpo_stage31/h100_logs/${RUN_TAG}_eval_${BRANCH}_fold${HOLDOUT}"
mkdir -p "$LOG_DIR"; cd "$ROOT_DIR"
TOTAL=8; [[ "$BRANCH" == DPF ]] && TOTAL=10
for wave in 0 8; do
  (( wave < TOTAL )) || continue
  pids=(); names=()
  for gpu in 0 1 2 3 4 5 6 7; do
    job=$((wave + gpu)); (( job < TOTAL )) || continue
    name="job${job}"; names+=("$name")
    bash scripts/evaluation/run_diffusiondrive_grpo_stage31_cv_eval_cell.sh \
      "$BRANCH" "$HOLDOUT" "$job" "$gpu" >"$LOG_DIR/$name.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then echo "PASS ${names[$index]}";
    else status=$?; echo "FAIL ${names[$index]} exit=$status"; failed=1; fi
  done
  (( failed == 0 )) || { echo "inspect $LOG_DIR"; exit 1; }
done
echo "PASS Stage31 CV evaluation branch=$BRANCH fold=$HOLDOUT"
echo "Logs: $LOG_DIR"
