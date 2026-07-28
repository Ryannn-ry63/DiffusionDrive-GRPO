#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 FOLD(0|1) [CUDA_DEVICES]"
  exit 2
fi

FOLD="$1"
CUDA_DEVICES="${2:-0,1,2,3,4,5,6,7}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CELL="$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage34_eval_cell.sh"
LOG_DIR="$ROOT_DIR/artifacts/grpo_stage34/h100_logs/$(date -u +%Y%m%dT%H%M%SZ)_eval_fold${FOLD}"

[[ "$FOLD" =~ ^[01]$ ]] || { echo "fold must be 0 or 1"; exit 2; }
[[ "$CUDA_DEVICES" =~ ^[0-7](,[0-7]){7}$ ]] || {
  echo "Stage34 evaluation requires eight GPU indices"; exit 2;
}
[[ -x "$CELL" ]] || { echo "missing executable Stage34 eval cell: $CELL"; exit 2; }
mkdir -p "$LOG_DIR"
IFS=',' read -r -a GPUS <<< "$CUDA_DEVICES"

JOBS=(
  "P 20261511" "P 20261512"
  "DPEL192 20261511" "DPEL192 20261512"
  "MAF48 20261511" "MAF48 20261512"
  "MAF96 20261511" "MAF96 20261512"
  "MAF144 20261511" "MAF144 20261512"
  "MAF192 20261511" "MAF192 20261512"
)
PIDS=()
LABELS=()
FAILED=0

launch() {
  local system="$1" noise="$2" gpu="$3"
  local label="${system}_ns${noise}"
  echo "Launching fold=$FOLD system=$system noise=$noise on GPU=$gpu"
  "$CELL" "$FOLD" "$system" "$noise" "$gpu" \
    >"$LOG_DIR/${label}.log" 2>&1 &
  PIDS+=("$!")
  LABELS+=("$label")
}

wait_batch() {
  local index
  for index in "${!PIDS[@]}"; do
    if wait "${PIDS[$index]}"; then
      echo "PASS ${LABELS[$index]}"
    else
      echo "FAIL ${LABELS[$index]} (see $LOG_DIR/${LABELS[$index]}.log)"
      FAILED=1
    fi
  done
  PIDS=()
  LABELS=()
}

for index in "${!JOBS[@]}"; do
  read -r system noise <<< "${JOBS[$index]}"
  launch "$system" "$noise" "${GPUS[$((index % 8))]}"
  if [[ "${#PIDS[@]}" -eq 8 ]]; then
    wait_batch
  fi
done
if [[ "${#PIDS[@]}" -gt 0 ]]; then
  wait_batch
fi

if [[ "$FAILED" -ne 0 ]]; then
  echo "Stage34 fold=$FOLD evaluation failed; logs=$LOG_DIR"
  exit 1
fi
echo "PASS Stage34 fold=$FOLD evaluation; logs=$LOG_DIR"
echo "After both folds run:"
echo "  $ROOT_DIR/scripts/evaluation/summarize_grpo_stage34_pilot.py"
echo "    --eval-root $ROOT_DIR/artifacts/grpo_stage34/pilot/eval"
echo "    --bucket-manifest $ROOT_DIR/artifacts/grpo_stage30/manifests/buckets_6119.json"
echo "    --output $ROOT_DIR/artifacts/grpo_stage34/pilot/summary.json"
