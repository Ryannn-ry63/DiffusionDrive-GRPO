#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELL="$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage37_jfi_bank_cell.sh"
LOG_DIR="$ROOT_DIR/artifacts/grpo_stage37/h100_logs/$(date -u +%Y%m%dT%H%M%SZ)_jfi_banks"
[[ -x "$CELL" ]] || { echo "missing executable bank cell: $CELL"; exit 2; }
mkdir -p "$LOG_DIR"
JOBS=(
  "P 20263721" "P 20263722"
  "DPEL192 20263721" "DPEL192 20263722"
  "NCD192 20263721" "NCD192 20263722"
  "RGT192 20263721" "RGT192 20263722"
)
PIDS=(); LABELS=(); FAILED=0
for index in "${!JOBS[@]}"; do
  read -r system noise <<< "${JOBS[$index]}"
  label="${system}_ns${noise}"
  echo "Launching $label on GPU $index"
  "$CELL" "$system" "$noise" "$index" >"$LOG_DIR/$label.log" 2>&1 &
  PIDS+=("$!"); LABELS+=("$label")
done
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then echo "PASS ${LABELS[$index]}";
  else echo "FAIL ${LABELS[$index]} log=$LOG_DIR/${LABELS[$index]}.log"; FAILED=1; fi
done
[[ "$FAILED" -eq 0 ]] || exit 1
if [[ "${GRPO_STAGE37_BANK_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "PASS all Stage37 JFI bank preflights; no bank freeze was attempted."
  exit 0
fi
"${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}" \
  "$ROOT_DIR/scripts/evaluation/freeze_grpo_stage37_jfi_banks.py"
echo "PASS all Stage37 JFI candidate banks; logs=$LOG_DIR"
