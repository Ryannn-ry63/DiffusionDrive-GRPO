#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive"
CELL="$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage39_eval_cell.sh"
AUDITOR="$ROOT_DIR/scripts/evaluation/audit_grpo_stage39_phase0.py"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
[[ -d "$ROOT_DIR" && -x "$CELL" && -x "$AUDITOR" && -x "$PYTHON_BIN" ]] || {
  echo "Stage39 Phase0 paths are incomplete under $ROOT_DIR"; exit 2;
}
PHASE0_OUTPUT="$ROOT_DIR/artifacts/grpo_stage39/phase0/audit.json"
if [[ -f "$PHASE0_OUTPUT" ]]; then
  if "$PYTHON_BIN" - "$PHASE0_OUTPUT" <<'PYREUSE'
import json
import sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if (
    p.get("stage") == 39
    and p.get("phase") == "offline_target_audit"
    and p.get("passed") is True
) else 1)
PYREUSE
  then
    echo "Reusing passed Stage39 Phase0: $PHASE0_OUTPUT"
    exit 0
  fi
  echo "Existing Stage39 Phase0 audit is invalid: $PHASE0_OUTPUT"
  if [[ "${STAGE39_PHASE0_REPAIR:-0}" != 1 ]]; then
    echo "Set STAGE39_PHASE0_REPAIR=1 to quarantine the failed Phase0 controls and regenerate them."
    exit 2
  fi
  REPAIR_DIR="$ROOT_DIR/artifacts/grpo_stage39/phase0/repair_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$REPAIR_DIR"
  for artifact in \
      "$PHASE0_OUTPUT" \
      "$ROOT_DIR/artifacts/grpo_stage39/pilot/eval/fold2/P20_ns20261711.json" \
      "$ROOT_DIR/artifacts/grpo_stage39/pilot/eval/fold2/P20_ns20261712.json" \
      "$ROOT_DIR/artifacts/grpo_stage39/pilot/eval/fold2/P40_ns20261711.json" \
      "$ROOT_DIR/artifacts/grpo_stage39/pilot/eval/fold2/P40_ns20261712.json"; do
    if [[ -e "$artifact" ]]; then
      mv "$artifact" "$REPAIR_DIR/"
    fi
  done
  echo "Quarantined failed Stage39 Phase0 artifacts under $REPAIR_DIR"
fi

export PYTHON_BIN
export NAVSIM_DEVKIT_ROOT="$ROOT_DIR/navsim"
export NAVSIM_EXP_ROOT="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp"
export OPENSCENE_DATA_ROOT="/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/inspire/hdd/global_public/public_datas/NAVSIM/maps"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0

TASKS=(
  "P20 20261711 0"
  "P20 20261712 1"
  "P40 20261711 2"
  "P40 20261712 3"
)
for task in "${TASKS[@]}"; do
  read -r role noise gpu <<<"$task"
  GRPO_STAGE39_EVAL_PREFLIGHT_ONLY=1 "$CELL" "$role" 2 0 "$noise" "$gpu"
done
if [[ "${GRPO_STAGE39_PHASE0_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "PASS Stage39 Phase0 root preflight"
  exit 0
fi

if command -v tmux >/dev/null 2>&1 && tmux has-session -t gpu-occupy 2>/dev/null; then
  echo "Stopping fixed gpu-occupy session before Stage39 Phase0..."
  tmux kill-session -t gpu-occupy
  for _ in $(seq 1 20); do
    if ! pgrep -f '[o]ccupy_multi.py' >/dev/null; then break; fi
    sleep 1
  done
  if pgrep -f '[o]ccupy_multi.py' >/dev/null; then
    echo "occupy_multi.py did not exit; refusing to start GPU evaluation"; exit 2
  fi
  echo "gpu-occupy stopped and memory release requested."
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits || true
fi

LOG_DIR="$ROOT_DIR/artifacts/grpo_stage39/h100_logs/$(date -u +%Y%m%dT%H%M%SZ)_phase0"
mkdir -p "$LOG_DIR"
pids=(); names=()
for task in "${TASKS[@]}"; do
  read -r role noise gpu <<<"$task"
  name="${role}_ns${noise}"
  echo "Launching $name on GPU $gpu; log=$LOG_DIR/$name.log"
  "$CELL" "$role" 2 0 "$noise" "$gpu" >"$LOG_DIR/$name.log" 2>&1 &
  pids+=("$!"); names+=("$name")
done
failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "PASS ${names[$index]}"
  else
    echo "FAIL ${names[$index]}; inspect $LOG_DIR/${names[$index]}.log"
    failed=1
  fi
done
[[ "$failed" -eq 0 ]] || exit 1
OUTPUT="$PHASE0_OUTPUT"
"$PYTHON_BIN" "$AUDITOR" \
  --eval-root "$ROOT_DIR/artifacts/grpo_stage39/pilot/eval" \
  --output "$OUTPUT"
echo "PASS Stage39 Phase0; audit=$OUTPUT; logs=$LOG_DIR"
