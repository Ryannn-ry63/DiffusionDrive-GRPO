#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || ( "$1" != audit && "$1" != formal ) || ! "$2" =~ ^[ABC]$ ]]; then
  echo "Usage: $0 audit|formal A|B|C"
  exit 2
fi
PHASE="$1"
BRANCH="$2"
ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
export NAVSIM_DEVKIT_ROOT="$ROOT_DIR/navsim"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
export GRPO_TRAIN_CACHE="${GRPO_TRAIN_CACHE:-$NAVSIM_EXP_ROOT/training_cache}"

PUBLIC=/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS
INPUTS="$ROOT_DIR/artifacts/grpo_stage28/pilot/inputs.json"
ARTIFACT_DIR="$ROOT_DIR/artifacts/grpo_stage28/pilot/training/$BRANCH/$PHASE"
EXPERIMENT="stage28_pilot_${BRANCH}_${PHASE}_seed0"
FREEZE="$ARTIFACT_DIR/checkpoints.json"
AUDIT="$ARTIFACT_DIR/audit.json"
EXPECTED_GPU_SUBSTRING="${STAGE28_EXPECTED_GPU_SUBSTRING:-H100}"

[[ -x "$PYTHON_BIN" ]] || { echo "missing Stage28 Python: $PYTHON_BIN"; exit 2; }
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" -eq 8 ]] || { echo "Stage28 requires exactly 8 visible GPUs"; exit 2; }
for name in "${GPU_NAMES[@]}"; do
  [[ "$name" == *"$EXPECTED_GPU_SUBSTRING"* ]] || {
    echo "Stage28 GPU gate rejected: $name"; exit 2;
  }
done
[[ ! -e "$ARTIFACT_DIR" ]] || {
  echo "refusing to overwrite Stage28 $BRANCH/$PHASE artifacts"; exit 2;
}
if [[ "$PHASE" == formal ]]; then
  "$PYTHON_BIN" - "$ROOT_DIR/artifacts/grpo_stage28/pilot/training/$BRANCH/audit/audit.json" "$BRANCH" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    raise RuntimeError("Stage28 formal run requires the same branch one-step audit")
d = json.loads(p.read_text())
if not (
    d.get("passed") and d.get("stage") == 28 and d.get("phase") == "audit"
    and d.get("branch") == sys.argv[2] and d.get("num_logged_optimizer_steps") == 1
    and d.get("active_decoder_gradients") and d.get("zero_frozen_gradients")
    and d.get("frozen_reference_bitwise_equal_public")
    and d.get("frozen_training_selector_bitwise_equal")
):
    raise RuntimeError("Stage28 one-step audit is not a valid passing gate")
PY
fi

cd "$ROOT_DIR"
scripts/training/run_diffusiondrive_grpo_stage28_pilot.sh \
  "$PHASE" "$BRANCH" "$EXPERIMENT" 0,1,2,3,4,5,6,7
shopt -s nullglob
RUN_DIRS=("$NAVSIM_EXP_ROOT/$EXPERIMENT"/*)
[[ "${#RUN_DIRS[@]}" -eq 1 ]] || {
  echo "expected one Stage28 run directory, got ${#RUN_DIRS[@]}"; exit 2;
}
RUN_DIR="${RUN_DIRS[0]}"
"$PYTHON_BIN" scripts/evaluation/freeze_grpo_stage28_pilot_checkpoints.py \
  --phase "$PHASE" --branch "$BRANCH" --run-dir "$RUN_DIR" --output "$FREEZE"
"$PYTHON_BIN" scripts/evaluation/audit_grpo_stage28_pilot_checkpoint.py \
  --phase "$PHASE" --branch "$BRANCH" --base "$PUBLIC" --inputs "$INPUTS" \
  --freeze "$FREEZE" --output "$AUDIT"

echo "PASS Stage28 pilot branch=$BRANCH phase=$PHASE"
echo "Run: $RUN_DIR"
echo "Checkpoint freeze: $FREEZE"
echo "Audit: $AUDIT"
