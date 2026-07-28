#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$ROOT_DIR/artifacts/grpo_stage27/h100_logs/$RUN_TAG"
mkdir -p "$LOG_DIR"

cd "$ROOT_DIR"

NAMES=(
  candidate_default
  candidate_ns20260811
  candidate_ns20260812
  selector_ns20260821
  selector_ns20260822
)
COMMANDS=(
  "candidate 0 0"
  "candidate 1 1"
  "candidate 2 2"
  "selector 0 3"
  "selector 1 4"
)

pids=()
for index in "${!NAMES[@]}"; do
  name="${NAMES[$index]}"
  read -r mode job_id gpu <<<"${COMMANDS[$index]}"
  echo "Launching $name on GPU $gpu; log=$LOG_DIR/$name.log"
  bash "$ROOT_DIR/run_stage27_public88_dev_h100.sh" \
    "$mode" "$job_id" "$gpu" >"$LOG_DIR/$name.log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  name="${NAMES[$index]}"
  if wait "${pids[$index]}"; then
    echo "PASS $name"
  else
    status=$?
    echo "FAIL $name exit=$status; inspect $LOG_DIR/$name.log"
    failed=1
  fi
done

if (( failed != 0 )); then
  echo "One or more Stage27 public-88.1 development jobs failed."
  exit 1
fi

echo "All five Stage27 public-88.1 development jobs completed."
echo "Logs: $LOG_DIR"
