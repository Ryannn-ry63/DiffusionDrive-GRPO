#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 FOLD(0|1) [CUDA_DEVICES]"
  exit 2
fi
FOLD="$1"
CUDA_DEVICES="${2:-0,1,2,3,4,5,6,7}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CELL="$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage33_cdc_eval_cell.sh"
LOG_DIR="$ROOT_DIR/artifacts/grpo_stage33/eval_logs/fold${FOLD}/$(date -u +%Y%m%dT%H%M%SZ)"

[[ "$FOLD" =~ ^[01]$ ]] || { echo "fold must be 0 or 1"; exit 2; }
[[ "$CUDA_DEVICES" =~ ^[0-7](,[0-7]){7}$ ]] || { echo "requires eight GPU indices"; exit 2; }
[[ -x "$CELL" ]] || { echo "missing eval cell: $CELL"; exit 2; }
mkdir -p "$LOG_DIR"
IFS=',' read -r -a GPUS <<< "$CUDA_DEVICES"

JOBS=(
  "P 20261411" "P 20261412"
  "48 20261411" "48 20261412"
  "96 20261411" "96 20261412"
  "144 20261411" "144 20261412"
  "192 20261411" "192 20261412"
)
PIDS=(); LABELS=(); FAILED=0

launch() {
  local job_index="$1" step="$2" noise="$3" gpu="$4"
  local label="${step}_ns${noise}"
  echo "Launching fold=$FOLD step=$step noise=$noise on GPU=$gpu"
  "$CELL" "$FOLD" "$step" "$noise" "$gpu" \
    >"$LOG_DIR/${label}.log" 2>&1 &
  PIDS+=("$!")
  LABELS+=("$label")
}

for i in "${!JOBS[@]}"; do
  read -r step noise <<< "${JOBS[$i]}"
  gpu="${GPUS[$((i % 8))]}"
  launch "$i" "$step" "$noise" "$gpu"
  if [[ "${#PIDS[@]}" -eq 8 ]]; then
    for j in "${!PIDS[@]}"; do
      if wait "${PIDS[$j]}"; then
        echo "PASS ${LABELS[$j]}"
      else
        echo "FAIL ${LABELS[$j]} (see $LOG_DIR/${LABELS[$j]}.log)"
        FAILED=1
      fi
    done
    PIDS=(); LABELS=()
  fi
done

for j in "${!PIDS[@]}"; do
  if wait "${PIDS[$j]}"; then
    echo "PASS ${LABELS[$j]}"
  else
    echo "FAIL ${LABELS[$j]} (see $LOG_DIR/${LABELS[$j]}.log)"
    FAILED=1
  fi
done

if [[ "$FAILED" -ne 0 ]]; then
  echo "Stage33 fold=$FOLD evaluation failed; logs=$LOG_DIR"
  exit 1
fi
echo "Stage33 fold=$FOLD evaluation complete; logs=$LOG_DIR"
echo "Aggregate after both folds:"
echo "  $ROOT_DIR/scripts/evaluation/summarize_grpo_stage33_cdc_pdm.py --eval-dir $ROOT_DIR/artifacts/grpo_stage33/eval --output $ROOT_DIR/artifacts/grpo_stage33/pdm_summary.json"
