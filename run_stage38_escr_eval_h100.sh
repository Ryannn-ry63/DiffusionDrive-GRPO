#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[01]$ ]]; then
  echo "Usage: $0 FOLD(0|1)"
  exit 2
fi

FOLD="$1"
ROOT_DIR="/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive"
CELL="$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage38_escr_eval_cell.sh"
FINALIZE="$ROOT_DIR/scripts/evaluation/finalize_grpo_stage38_escr.sh"
FREEZE="$ROOT_DIR/artifacts/grpo_stage38/pilot/training/ESCR/fold${FOLD}/formal/checkpoints.json"
[[ -x "$CELL" && -x "$FINALIZE" && -f "$FREEZE" ]] || {
  echo "Stage38 eval scripts or formal freeze are missing"; exit 2;
}
if [[ "$FOLD" == 1 ]]; then
  [[ -f "$ROOT_DIR/artifacts/grpo_stage38/pilot/kill_selection.json" ]] || {
    echo "Fold1 evaluation is locked until the fold0 kill-test passes"; exit 2;
  }
fi

if command -v tmux >/dev/null 2>&1 \
    && tmux has-session -t gpu-occupy 2>/dev/null; then
  echo "Stopping fixed gpu-occupy session before Stage38 evaluation..."
  tmux kill-session -t gpu-occupy
  for _ in $(seq 1 20); do
    if ! pgrep -f '[o]ccupy_multi.py' >/dev/null; then break; fi
    sleep 1
  done
  if pgrep -f '[o]ccupy_multi.py' >/dev/null; then
    echo "occupy_multi.py did not exit; refusing to start GPU evaluation"
    exit 2
  fi
  echo "gpu-occupy stopped and occupy_multi.py exited."
fi

export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
export NAVSIM_DEVKIT_ROOT="$ROOT_DIR/navsim"
export NAVSIM_EXP_ROOT="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp"
export OPENSCENE_DATA_ROOT="/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/inspire/hdd/global_public/public_datas/NAVSIM/maps"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0

TASKS=(
  "public 0 20261611"
  "public 0 20261612"
  "rgt 192 20261611"
  "rgt 192 20261612"
  "escr 24 20261611"
  "escr 24 20261612"
  "escr 48 20261611"
  "escr 48 20261612"
  "escr 96 20261611"
  "escr 96 20261612"
)

for task in "${TASKS[@]}"; do
  read -r role step noise <<<"$task"
  GRPO_STAGE38_EVAL_PREFLIGHT_ONLY=1 \
    "$CELL" "$role" "$FOLD" "$step" "$noise" 0
done

LOG_DIR="$ROOT_DIR/artifacts/grpo_stage38/h100_logs/$(date -u +%Y%m%dT%H%M%SZ)_eval_fold${FOLD}"
mkdir -p "$LOG_DIR"

run_wave() {
  local start="$1"
  local end="$2"
  local pids=()
  local names=()
  local failed=0
  for ((index=start; index<end; index++)); do
    read -r role step noise <<<"${TASKS[$index]}"
    local gpu=$((index - start))
    local name="${role}_${step}_ns${noise}"
    echo "Launching $name on GPU $gpu; log=$LOG_DIR/$name.log"
    "$CELL" "$role" "$FOLD" "$step" "$noise" "$gpu" \
      >"$LOG_DIR/$name.log" 2>&1 &
    pids+=("$!")
    names+=("$name")
  done
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      echo "PASS ${names[$index]}"
    else
      echo "FAIL ${names[$index]}; inspect $LOG_DIR/${names[$index]}.log"
      failed=1
    fi
  done
  [[ "$failed" -eq 0 ]]
}

run_wave 0 8
run_wave 8 10

if [[ "$FOLD" == 0 ]]; then
  "$FINALIZE" kill
else
  "$FINALIZE" formal
fi
echo "PASS Stage38 ESCR evaluation fold=$FOLD; logs=$LOG_DIR"
