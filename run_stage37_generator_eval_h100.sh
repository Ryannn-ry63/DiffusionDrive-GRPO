#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then echo "Usage: $0 FOLD(0|1) [CUDA_DEVICES]"; exit 2; fi
FOLD="$1"; CUDA_DEVICES="${2:-0,1,2,3,4,5,6,7}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; CELL="$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage37_generator_eval_cell.sh"
LOG_DIR="$ROOT_DIR/artifacts/grpo_stage37/h100_logs/$(date -u +%Y%m%dT%H%M%SZ)_generator_eval_f${FOLD}"
[[ "$FOLD" =~ ^[01]$ && "$CUDA_DEVICES" =~ ^[0-7](,[0-7]){7}$ && -x "$CELL" ]] || { echo "invalid input/cell"; exit 2; }
mkdir -p "$LOG_DIR"; IFS=',' read -r -a GPUS <<< "$CUDA_DEVICES"
JOBS=("48 20261511" "48 20261512" "96 20261511" "96 20261512" "144 20261511" "144 20261512" "192 20261511" "192 20261512")
PIDS=(); LABELS=(); FAILED=0
for i in "${!JOBS[@]}"; do read -r step noise <<< "${JOBS[$i]}"; label="BPD${step}_ns${noise}"; "$CELL" "$FOLD" "$step" "$noise" "${GPUS[$i]}" >"$LOG_DIR/$label.log" 2>&1 & PIDS+=("$!"); LABELS+=("$label"); done
for i in "${!PIDS[@]}"; do if wait "${PIDS[$i]}"; then echo "PASS ${LABELS[$i]}"; else echo "FAIL ${LABELS[$i]} log=$LOG_DIR/${LABELS[$i]}.log"; FAILED=1; fi; done
[[ "$FAILED" -eq 0 ]] || exit 1
echo "PASS Stage37 generator evaluation fold=$FOLD; logs=$LOG_DIR"
