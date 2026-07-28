#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
export NAVSIM_DEVKIT_ROOT="$ROOT_DIR/navsim"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"

INPUTS="$ROOT_DIR/artifacts/grpo_stage29/pilot/fold4/inputs.json"
OUT_DIR="$ROOT_DIR/artifacts/grpo_stage29/pilot/fold4/results"
REPORT="$ROOT_DIR/artifacts/grpo_stage29/pilot/fold4/report.json"
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$ROOT_DIR/artifacts/grpo_stage29/h100_logs/${RUN_TAG}_pilot_fold4"
[[ -f "$INPUTS" ]] || { echo "missing Stage29 fold4 inputs: $INPUTS"; exit 2; }
[[ ! -e "$REPORT" ]] || { echo "refusing to overwrite Stage29 fold4 report"; exit 2; }
if [[ -e "$OUT_DIR" ]]; then
  [[ -d "$OUT_DIR" && -z "$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
    echo "refusing to overwrite Stage29 fold4 results"; exit 2;
  }
fi
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" -eq 8 ]] || { echo "Stage29 fold4 requires exactly 8 visible GPUs"; exit 2; }
mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"
SYSTEMS=(P A3 H1 H2 H3 H4 HC1 HC2 HC3 HC4)
NOISES=(20261211 20261212)

for wave_start in 0 8 16; do
  pids=(); names=()
  for offset in 0 1 2 3 4 5 6 7; do
    job=$((wave_start + offset))
    (( job < 20 )) || continue
    system_index=$((job / 2)); noise_index=$((job % 2))
    name="${SYSTEMS[$system_index]}_ns${NOISES[$noise_index]}"
    names+=("$name")
    bash scripts/evaluation/run_diffusiondrive_grpo_stage29_fold4_cell.sh \
      "$job" "$offset" >"$LOG_DIR/$name.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      echo "PASS Stage29 fold4 cell ${names[$index]}"
    else
      status=$?; echo "FAIL Stage29 fold4 cell ${names[$index]} exit=$status"; failed=1
    fi
  done
  (( failed == 0 )) || { echo "inspect Stage29 logs: $LOG_DIR"; exit 1; }
done

GATE_ARGS=(--inputs "$INPUTS")
for system in "${SYSTEMS[@]}"; do
  for noise in "${NOISES[@]}"; do
    GATE_ARGS+=(--artifact "$system" "$noise" "$OUT_DIR/${system}_ns${noise}.json")
  done
done
"$PYTHON_BIN" scripts/evaluation/check_grpo_stage29_fold4.py \
  "${GATE_ARGS[@]}" --output "$REPORT"
echo "PASS Stage29 pilot fold4 gate"
echo "Report: $REPORT"
echo "Logs: $LOG_DIR"
