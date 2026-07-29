#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; CELL="$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage37_jfi_calibration_cell.sh"
LOG_DIR="$ROOT_DIR/artifacts/grpo_stage37/h100_logs/$(date -u +%Y%m%dT%H%M%SZ)_jfi_calibration"; mkdir -p "$LOG_DIR"
JOBS=("0 P 20261511" "0 P 20261512" "0 BPD 20261511" "0 BPD 20261512" "1 P 20261511" "1 P 20261512" "1 BPD 20261511" "1 BPD 20261512")
PIDS=();LABELS=();FAILED=0
for i in "${!JOBS[@]}"; do read -r h s n <<< "${JOBS[$i]}"; label="h${h}_${s}_ns${n}"; "$CELL" "$h" "$s" "$n" "$i" >"$LOG_DIR/$label.log" 2>&1 & PIDS+=("$!");LABELS+=("$label"); done
for i in "${!PIDS[@]}"; do if wait "${PIDS[$i]}";then echo "PASS ${LABELS[$i]}";else echo "FAIL ${LABELS[$i]} log=$LOG_DIR/${LABELS[$i]}.log";FAILED=1;fi;done
[[ "$FAILED" -eq 0 ]] || exit 1
echo "PASS all JFI calibration collections; logs=$LOG_DIR"
