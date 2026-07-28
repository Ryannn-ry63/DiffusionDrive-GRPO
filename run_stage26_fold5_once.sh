#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
LOG_DIR="$ROOT_DIR/artifacts/grpo_stage26/logs/fold5"
OUT_DIR="$ROOT_DIR/artifacts/grpo_stage26/fold5"
mkdir -p "$LOG_DIR" "$OUT_DIR"
cd "$ROOT_DIR"

pids=()
for job in 0 1 2 3 4 5 6 7; do
  log="$LOG_DIR/job${job}.log"
  [[ ! -e "$log" ]] || {
    echo "protected fold5 log already exists; refusing a second launch: $log"
    exit 2
  }
  scripts/evaluation/run_diffusiondrive_grpo_stage26_fold5.sh \
    "$job" "$job" >"$log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" != 0 ]]; then
  echo "at least one fold5 cell failed; inspect $LOG_DIR and resume only missing cells"
  exit 1
fi
scripts/evaluation/check_grpo_stage26_fold5.sh
