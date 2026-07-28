#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/navsim}"
if [[ ! -f "$NAVSIM_DEVKIT_ROOT/planning/script/run_training.py" ]]; then
  export NAVSIM_DEVKIT_ROOT="$ROOT_DIR/navsim"
fi
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
EXPECTED_GPU_SUBSTRING="${STAGE27_EXPECTED_GPU_SUBSTRING:-H100}"

INPUTS="$ROOT_DIR/artifacts/grpo_stage27/generator/phase4/inputs.json"
OUT_DIR="$ROOT_DIR/artifacts/grpo_stage27/generator/phase4/fold5"
REPORT="$ROOT_DIR/artifacts/grpo_stage27/generator/phase4/report.json"
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$ROOT_DIR/artifacts/grpo_stage27/h100_logs/${RUN_TAG}_phase4_fold5"

[[ -x "$PYTHON_BIN" ]] || {
  echo "missing Stage27 H100 Python: $PYTHON_BIN"
  exit 2
}
mapfile -t GPU_NAMES < <(
  nvidia-smi --query-gpu=name --format=csv,noheader
)
[[ "${#GPU_NAMES[@]}" -eq 8 ]] || {
  echo "Stage27 Phase4 requires exactly eight visible GPUs"
  exit 2
}
for gpu_name in "${GPU_NAMES[@]}"; do
  [[ "$gpu_name" == *"$EXPECTED_GPU_SUBSTRING"* ]] || {
    echo "Stage27 Phase4 GPU gate rejected GPU: $gpu_name"
    exit 2
  }
done
[[ -f "$INPUTS" ]] || {
  echo "missing Stage27 Phase4 frozen inputs: $INPUTS"
  exit 2
}
[[ ! -e "$REPORT" ]] || {
  echo "refusing to overwrite protected Stage27 Phase4 report"
  exit 2
}
if [[ -e "$OUT_DIR" ]]; then
  [[ -d "$OUT_DIR" ]] || {
    echo "refusing non-directory Stage27 Phase4 output path: $OUT_DIR"
    exit 2
  }
  [[ -z "$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
    echo "refusing to overwrite protected Stage27 Phase4 cell results"
    exit 2
  }
fi

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"
pids=()
names=()
for job in 0 1 2 3 4 5; do
  system_index=$((job / 2))
  noise_index=$((job % 2))
  systems=(B C1 C2)
  noises=(20261011 20261012)
  name="${systems[$system_index]}_ns${noises[$noise_index]}"
  names+=("$name")
  bash scripts/evaluation/run_diffusiondrive_grpo_stage27_phase4_fold5.sh \
    "$job" "$job" >"$LOG_DIR/$name.log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "PASS Stage27 Phase4 cell ${names[$index]}"
  else
    status=$?
    echo "FAIL Stage27 Phase4 cell ${names[$index]} exit=$status"
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "Stage27 Phase4 has failed cells; inspect $LOG_DIR"
  exit 1
fi

"$PYTHON_BIN" scripts/evaluation/check_grpo_stage27_phase4_fold5.py \
  --inputs "$INPUTS" \
  --artifact B 20261011 "$OUT_DIR/B_ns20261011.json" \
  --artifact B 20261012 "$OUT_DIR/B_ns20261012.json" \
  --artifact C1 20261011 "$OUT_DIR/C1_ns20261011.json" \
  --artifact C1 20261012 "$OUT_DIR/C1_ns20261012.json" \
  --artifact C2 20261011 "$OUT_DIR/C2_ns20261011.json" \
  --artifact C2 20261012 "$OUT_DIR/C2_ns20261012.json" \
  --output "$REPORT"

echo "PASS Stage27 Phase4 provisional generator gate"
echo "Report: $REPORT"
echo "Logs: $LOG_DIR"
