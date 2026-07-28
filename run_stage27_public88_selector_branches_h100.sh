#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 audit|formal"
  exit 2
fi
PHASE="$1"
[[ "$PHASE" == audit || "$PHASE" == formal ]] || {
  echo "PHASE must be audit or formal"
  exit 2
}

ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$ROOT_DIR/artifacts/grpo_stage27/h100_logs/${RUN_TAG}_selector_${PHASE}"
mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

bash ./run_stage27_public88_selector_branch_h100.sh \
  "$PHASE" public 0 >"$LOG_DIR/public.log" 2>&1 &
public_pid=$!
bash ./run_stage27_public88_selector_branch_h100.sh \
  "$PHASE" multi 1 >"$LOG_DIR/multi.log" 2>&1 &
multi_pid=$!

failed=0
if wait "$public_pid"; then
  echo "PASS Stage27 S-public $PHASE"
else
  status=$?
  echo "FAIL Stage27 S-public $PHASE exit=$status; inspect $LOG_DIR/public.log"
  failed=1
fi
if wait "$multi_pid"; then
  echo "PASS Stage27 S-multi $PHASE"
else
  status=$?
  echo "FAIL Stage27 S-multi $PHASE exit=$status; inspect $LOG_DIR/multi.log"
  failed=1
fi

if (( failed != 0 )); then
  exit 1
fi
echo "Both Stage27 selector branches completed phase=$PHASE"
echo "Logs: $LOG_DIR"
