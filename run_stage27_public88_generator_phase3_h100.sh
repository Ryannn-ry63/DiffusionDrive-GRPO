#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != audit && "$1" != formal ) ]]; then
  echo "Usage: $0 audit|formal"
  exit 2
fi

PHASE="$1"
ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/navsim}"
if [[ ! -f "$NAVSIM_DEVKIT_ROOT/planning/script/run_training.py" ]]; then
  export NAVSIM_DEVKIT_ROOT="$ROOT_DIR/navsim"
fi
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
export GRPO_TRAIN_CACHE="${GRPO_TRAIN_CACHE:-$NAVSIM_EXP_ROOT/training_cache}"

PUBLIC_CHECKPOINT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS
INPUT_AUDIT="$ROOT_DIR/artifacts/grpo_stage27/audits/generator_phase3_inputs.json"
ARTIFACT_DIR="$ROOT_DIR/artifacts/grpo_stage27/generator/phase3/$PHASE"
EXPERIMENT="stage27_generator_phase3_${PHASE}_seed0"
FREEZE="$ARTIFACT_DIR/checkpoints.json"
AUDIT="$ARTIFACT_DIR/audit.json"

[[ -x "$PYTHON_BIN" ]] || {
  echo "missing Stage27 H100 Python: $PYTHON_BIN"
  exit 2
}
mapfile -t GPU_NAMES < <(
  nvidia-smi --query-gpu=name --format=csv,noheader
)
[[ "${#GPU_NAMES[@]}" -eq 8 ]] || {
  echo "Stage27 Phase3 requires exactly eight visible H100 GPUs"
  exit 2
}
for gpu_name in "${GPU_NAMES[@]}"; do
  [[ "$gpu_name" == *H100* ]] || {
    echo "Stage27 Phase3 H100 gate rejected GPU: $gpu_name"
    exit 2
  }
done
[[ ! -e "$ARTIFACT_DIR" ]] || {
  echo "refusing to overwrite Stage27 Phase3 artifacts: $ARTIFACT_DIR"
  exit 2
}
if [[ "$PHASE" == formal ]]; then
  "$PYTHON_BIN" - "$ROOT_DIR/artifacts/grpo_stage27/generator/phase3/audit/audit.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise RuntimeError("Stage27 formal training is blocked until the H100 audit passes")
audit = json.loads(path.read_text(encoding="utf-8"))
if (
    not audit.get("passed")
    or audit.get("stage") != 27
    or audit.get("phase") != "audit"
    or audit.get("num_logged_optimizer_steps") != 1
    or not audit.get("frozen_reference_bitwise_equal_public")
    or not audit.get("frozen_selector_bitwise_equal_selected_multi")
    or not audit.get("zero_frozen_gradients")
):
    raise RuntimeError("Stage27 H100 one-step audit is not a valid passing gate")
PY
fi

cd "$ROOT_DIR"
scripts/training/run_diffusiondrive_grpo_stage27_generator.sh \
  "$PHASE" "$EXPERIMENT" 0,1,2,3,4,5,6,7

shopt -s nullglob
RUN_DIRS=("$NAVSIM_EXP_ROOT/$EXPERIMENT"/*)
[[ "${#RUN_DIRS[@]}" -eq 1 ]] || {
  echo "expected one Stage27 run directory, got ${#RUN_DIRS[@]}"
  exit 2
}
RUN_DIR="${RUN_DIRS[0]}"

"$PYTHON_BIN" scripts/evaluation/freeze_grpo_stage27_generator_checkpoints.py \
  --phase "$PHASE" \
  --run-dir "$RUN_DIR" \
  --output "$FREEZE"
"$PYTHON_BIN" scripts/evaluation/audit_grpo_stage27_generator_checkpoint.py \
  --phase "$PHASE" \
  --base "$PUBLIC_CHECKPOINT" \
  --input-audit "$INPUT_AUDIT" \
  --freeze "$FREEZE" \
  --output "$AUDIT"

echo "PASS Stage27 public-88.1 generator phase=$PHASE"
echo "Run: $RUN_DIR"
echo "Checkpoint freeze: $FREEZE"
echo "Audit: $AUDIT"
