#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive"
CELL="$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage39_eval_cell.sh"
SUMMARIZER="$ROOT_DIR/scripts/evaluation/summarize_grpo_stage39.py"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
DEVICES="${STAGE39_CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
PLAN_SHA="7e1e6142412778524714a0b919f971edf1d2bf14138b0efcf21fc9dd7c4d6f1b"
PHASE0="$ROOT_DIR/artifacts/grpo_stage39/phase0/audit.json"
EVAL_ROOT="$ROOT_DIR/artifacts/grpo_stage39/pilot/eval"
SELECTION_OUT="$ROOT_DIR/artifacts/grpo_stage39/pilot/selection.json"
SUMMARY_OUT="$ROOT_DIR/artifacts/grpo_stage39/pilot/summary.json"

[[ -x "$CELL" && -x "$PYTHON_BIN" && -f "$SUMMARIZER" ]] || {
  echo "Stage39 pilot evaluation paths are incomplete under $ROOT_DIR"; exit 2;
}
[[ "$DEVICES" =~ ^[0-7](,[0-7]){7}$ ]] || {
  echo "Stage39 requires exactly eight GPU indices"; exit 2;
}

export NAVSIM_DEVKIT_ROOT="$ROOT_DIR/navsim"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0

wait_for_file() {
  local path="$1"
  local label="$2"
  local waited=0
  local timeout="${STAGE39_WAIT_SECONDS:-259200}"
  while (( waited < timeout )); do
    if [[ -f "$path" ]]; then
      echo "Found $label: $path"
      return 0
    fi
    if (( waited == 0 || waited % 300 == 0 )); then
      echo "Waiting for $label (${waited}s/${timeout}s): $path"
    fi
    sleep 30
    waited=$((waited + 30))
  done
  echo "Timed out waiting for $label: $path"
  exit 2
}

validate_phase0() {
  "$PYTHON_BIN" - "$PHASE0" "$PLAN_SHA" <<'PY'
import json
import sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text())
assert p.get("stage") == 39 and p.get("phase") == "offline_target_audit"
assert p.get("passed") is True and p.get("plan_sha256") == sys.argv[2]
PY
}

validate_freeze() {
  local branch="$1"
  local path="$ROOT_DIR/artifacts/grpo_stage39/pilot/training/$branch/fold2/checkpoints.json"
  "$PYTHON_BIN" - "$path" "$branch" "$PLAN_SHA" <<'PY'
import json
import sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text())
assert p.get("stage") == 39 and p.get("phase") == "pilot"
assert p.get("branch") == sys.argv[2] and p.get("holdout_fold") == 2
assert p.get("passed") is True and p.get("plan_sha256") == sys.argv[3]
assert [int(x["global_step"]) for x in p["checkpoints"]] == [48, 96, 192]
PY
}

valid_output() {
  local path="$1"
  local source="$2"
  local expected="$3"
  "$PYTHON_BIN" - "$path" "$source" "$expected" <<'PY'
import json
import sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text())
s = p.get("summary", {})
records = p.get("records", [])
ok = (
    s.get("completed") is True
    and s.get("num_failures") == 0
    and s.get("generation_policy_algorithm") == "diffgrpo_non_destructive_challenger"
    and s.get("stage39_candidate_source") == sys.argv[2]
    and len(records) == int(s.get("num_tokens", -1))
    and all(len(r.get("candidate_rewards", [])) == int(sys.argv[3]) for r in records)
)
raise SystemExit(0 if ok else 1)
PY
}

stop_gpu_occupy() {
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t gpu-occupy 2>/dev/null; then
    echo "Stopping fixed gpu-occupy session before Stage39 pilot evaluation..."
    tmux kill-session -t gpu-occupy
    for _ in $(seq 1 20); do
      pgrep -f '[o]ccupy_multi.py' >/dev/null || break
      sleep 1
    done
    if pgrep -f '[o]ccupy_multi.py' >/dev/null; then
      echo "occupy_multi.py did not exit; refusing to start Stage39 evaluation"
      exit 2
    fi
    echo "gpu-occupy stopped and memory release confirmed."
    nvidia-smi --query-gpu=index,memory.used,memory.total \
      --format=csv,noheader,nounits || true
  fi
}

cd "$ROOT_DIR"
if [[ -e "$SUMMARY_OUT" || -e "$SELECTION_OUT" ]]; then
  echo "Stage39 pilot summary/selection already exists; refusing overwrite:"
  echo "  $SUMMARY_OUT"
  echo "  $SELECTION_OUT"
  exit 2
fi
wait_for_file "$PHASE0" "Stage39 Phase0 audit"
validate_phase0
for branch in BC STD SET; do
  wait_for_file \
    "$ROOT_DIR/artifacts/grpo_stage39/pilot/training/$branch/fold2/checkpoints.json" \
    "Stage39 pilot freeze ($branch)"
  validate_freeze "$branch"
done

TASKS=()
# Phase0 owns the four controls.  They are reused bitwise; only the 18 trained
# candidate cells are launched here.
for role in BC STD SET; do
  for step in 48 96 192; do
    for noise in 20261711 20261712; do
      TASKS+=("$role 2 $step $noise")
    done
  done
done

for task in "${TASKS[@]}"; do
  read -r role fold step noise <<<"$task"
  GRPO_STAGE39_EVAL_PREFLIGHT_ONLY=1 "$CELL" "$role" "$fold" "$step" "$noise" 0 >/dev/null
  output="$EVAL_ROOT/fold2/${role}${step}_ns${noise}.json"
  if [[ -e "$output" ]]; then
    if valid_output "$output" challenger_union 40; then
      echo "Reusing valid Stage39 pilot evaluation: $output"
    else
      echo "Existing Stage39 pilot evaluation is invalid; refusing overwrite: $output"
      exit 2
    fi
  fi
done

PENDING=()
for task in "${TASKS[@]}"; do
  read -r role fold step noise <<<"$task"
  output="$EVAL_ROOT/fold2/${role}${step}_ns${noise}.json"
  valid_output "$output" challenger_union 40 >/dev/null 2>&1 || PENDING+=("$task")
done

LOG_DIR="$ROOT_DIR/artifacts/grpo_stage39/h100_logs/$(date -u +%Y%m%dT%H%M%SZ)_pilot_eval"
mkdir -p "$LOG_DIR"
failed=0
if (( ${#PENDING[@]} > 0 )); then
  stop_gpu_occupy
fi
for ((start=0; start<${#PENDING[@]}; start+=8)); do
  end=$((start + 8))
  (( end > ${#PENDING[@]} )) && end=${#PENDING[@]}
  pids=()
  names=()
  for ((index=start; index<end; index++)); do
    read -r role fold step noise <<<"${PENDING[$index]}"
    gpu=$((index - start))
    name="${role}${step}_ns${noise}"
    echo "Launching $name on GPU $gpu; log=$LOG_DIR/$name.log"
    "$CELL" "$role" "$fold" "$step" "$noise" "$gpu" \
      >"$LOG_DIR/$name.log" 2>&1 &
    pids+=("$!")
    names+=("$name")
  done
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      echo "PASS ${names[$index]}"
    else
      echo "FAIL ${names[$index]}; inspect $LOG_DIR/${names[$index]}.log"
      failed=1
    fi
  done
  [[ "$failed" -eq 0 ]] || exit 1
done

mkdir -p "$ROOT_DIR/artifacts/grpo_stage39/pilot"
"$PYTHON_BIN" "$SUMMARIZER" \
  --phase pilot --eval-root "$EVAL_ROOT" \
  --selection-output "$SELECTION_OUT" --output "$SUMMARY_OUT"
echo "PASS Stage39 pilot evaluation and step selection"
echo "Summary: $SUMMARY_OUT"
echo "Selection: $SELECTION_OUT"
echo "Logs: $LOG_DIR"
