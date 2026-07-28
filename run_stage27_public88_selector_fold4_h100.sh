#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 collect|deploy"
  exit 2
fi
MODE="$1"
[[ "$MODE" == collect || "$MODE" == deploy ]] || { echo "invalid mode"; exit 2; }
ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/navsim}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$ROOT_DIR/artifacts/grpo_stage27/h100_logs/${RUN_TAG}_fold4_${MODE}"
mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

if [[ "$MODE" == collect ]]; then
  BRANCHES=(public public multi multi)
  JOBS=(0 1 0 1)
  GPUS=(0 1 2 3)
else
  SELECTION="$ROOT_DIR/artifacts/grpo_stage27/selectors/selection.json"
  [[ -f "$SELECTION" ]] || { echo "missing selector selection: $SELECTION"; exit 2; }
  SELECTED_BRANCH="$("$PYTHON_BIN" -c \
    'import json,sys;p=json.load(open(sys.argv[1]));assert p["passed"];print(p["selected_branch"])' \
    "$SELECTION")"
  BRANCHES=("$SELECTED_BRANCH" "$SELECTED_BRANCH")
  JOBS=(0 1)
  GPUS=(0 1)
fi

pids=()
names=()
for index in "${!JOBS[@]}"; do
  branch="${BRANCHES[$index]}"
  job="${JOBS[$index]}"
  gpu="${GPUS[$index]}"
  name="${branch}_${job}"
  names+=("$name")
  bash scripts/evaluation/run_diffusiondrive_grpo_stage27_selector_fold4.sh \
    "$MODE" "$branch" "$job" "$gpu" >"$LOG_DIR/$name.log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "PASS Stage27 fold4 $MODE ${names[$index]}"
  else
    status=$?
    echo "FAIL Stage27 fold4 $MODE ${names[$index]} exit=$status"
    failed=1
  fi
done
(( failed == 0 )) || exit 1

if [[ "$MODE" == collect ]]; then
  scripts/evaluation/calibrate_grpo_stage27_selectors.sh
else
  SELECTION="$ROOT_DIR/artifacts/grpo_stage27/selectors/selection.json"
  SELECTED_BRANCH="$("$PYTHON_BIN" -c \
    'import json,sys;print(json.load(open(sys.argv[1]))["selected_branch"])' \
    "$SELECTION")"
  "$PYTHON_BIN" \
    scripts/evaluation/check_grpo_stage27_selector_gate.py \
    --selection "$SELECTION" \
    --artifact "$ROOT_DIR/artifacts/grpo_stage27/selectors/$SELECTED_BRANCH/fold4/deploy_ns20260821.json" \
    --artifact "$ROOT_DIR/artifacts/grpo_stage27/selectors/$SELECTED_BRANCH/fold4/deploy_ns20260822.json" \
    --output "$ROOT_DIR/artifacts/grpo_stage27/selectors/selector_gate.json"
fi
echo "Logs: $LOG_DIR"
