#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
export NAVSIM_DEVKIT_ROOT="$ROOT_DIR/navsim"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"

RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$ROOT_DIR/artifacts/grpo_stage30/h100_logs/${RUN_TAG}_collect_labels"
OUTPUT="$ROOT_DIR/artifacts/grpo_stage30/manifests/buckets_6119.json"
[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite $OUTPUT"; exit 2; }
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" -eq 8 ]] || { echo "Stage30 requires exactly 8 visible GPUs"; exit 2; }
mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

pids=(); names=()
for job in 0 1 2; do
  namespace=$((20261301 + job))
  names+=("ns${namespace}")
  bash scripts/evaluation/run_diffusiondrive_grpo_stage30_label_cell.sh \
    "$job" "$job" >"$LOG_DIR/ns${namespace}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "PASS Stage30 label ${names[$index]}"
  else
    status=$?; echo "FAIL Stage30 label ${names[$index]} exit=$status"; failed=1
  fi
done
(( failed == 0 )) || { echo "inspect Stage30 logs: $LOG_DIR"; exit 1; }
"$PYTHON_BIN" scripts/evaluation/build_grpo_stage30_bucket_manifest.py \
  --root "$ROOT_DIR" --output "$OUTPUT"
"$PYTHON_BIN" scripts/evaluation/build_grpo_stage30_cv_manifests.py \
  --root "$ROOT_DIR"
echo "PASS Stage30 label collection and bucket freeze"
echo "Manifest: $OUTPUT"
echo "Logs: $LOG_DIR"
