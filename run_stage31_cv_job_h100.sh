#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 audit|formal DP|DPF HOLDOUT_FOLD(0..3)"
  exit 2
fi
PHASE="$1"; BRANCH="$2"; HOLDOUT="$3"
ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
export NAVSIM_DEVKIT_ROOT="$ROOT_DIR/navsim"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1 RAY_DEDUP_LOGS=0
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
export GRPO_TRAIN_CACHE="${GRPO_TRAIN_CACHE:-$NAVSIM_EXP_ROOT/training_cache}"

TRAIN_SCRIPT="$ROOT_DIR/scripts/training/run_diffusiondrive_grpo_stage31_cv.sh"
FREEZER="$ROOT_DIR/scripts/evaluation/freeze_grpo_stage31_cv_checkpoints.py"
AUDITOR="$ROOT_DIR/scripts/evaluation/audit_grpo_stage31_cv_checkpoint.py"
PUBLIC=/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS
SELECTOR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage27_selector_multi_stage25_formal_seed27125/2026.07.24.09.28.29/lightning_logs/version_0/checkpoints/grpo-17-36684.ckpt
CALIBRATION="$ROOT_DIR/artifacts/grpo_stage27/selectors/multi/calibration.json"
CV_FREEZE="$ROOT_DIR/artifacts/grpo_stage30/manifests/cv/freeze.json"

[[ "$PHASE" == audit || "$PHASE" == formal ]] || { echo "bad phase"; exit 2; }
[[ "$BRANCH" == DP || "$BRANCH" == DPF ]] || { echo "bad branch"; exit 2; }
[[ "$HOLDOUT" =~ ^[0-3]$ ]] || { echo "bad holdout"; exit 2; }
for path in "$PYTHON_BIN" "$TRAIN_SCRIPT" "$FREEZER" "$AUDITOR" "$PUBLIC" "$SELECTOR" "$CALIBRATION" "$CV_FREEZE"; do
  [[ -e "$path" ]] || { echo "missing Stage31 job input: $path"; exit 2; }
done
bash -n "$TRAIN_SCRIPT"
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" -eq 8 ]] || { echo "Stage31 requires exactly 8 GPUs"; exit 2; }
EXPERIMENT="stage31_cv_${BRANCH}_fold${HOLDOUT}_${PHASE}_seed$((203100 + HOLDOUT))"
ARTIFACT_DIR="$ROOT_DIR/artifacts/grpo_stage31/cv/training/$BRANCH/fold$HOLDOUT/$PHASE"
FREEZE="$ARTIFACT_DIR/checkpoints.json"
AUDIT="$ARTIFACT_DIR/audit.json"
[[ ! -e "$ARTIFACT_DIR" ]] || { echo "refusing to overwrite $ARTIFACT_DIR"; exit 2; }
if [[ "$PHASE" == formal ]]; then
  "$PYTHON_BIN" - "$ROOT_DIR/artifacts/grpo_stage31/cv/training/DPF/fold0/audit/audit.json" <<'PY'
import json, sys
from pathlib import Path
path=Path(sys.argv[1])
if not path.is_file():
    raise RuntimeError('Stage31 formal jobs require DPF/fold0 one-step audit')
p=json.loads(path.read_text())
if not (p.get('passed') and p.get('stage')==31 and p.get('phase')=='audit'
        and p.get('branch')=='DPF' and p.get('holdout_fold')==0
        and p.get('num_logged_optimizer_steps')==1
        and p.get('active_decoder_gradients') and p.get('zero_frozen_gradients')
        and p.get('frozen_reference_bitwise_equal_public')
        and p.get('frozen_training_selector_bitwise_equal')
        and p.get('exact_global_bucket_sampler')):
    raise RuntimeError('Stage31 one-step audit is not a passing gate')
PY
fi
cd "$ROOT_DIR"
scripts/training/run_diffusiondrive_grpo_stage31_cv.sh \
  "$PHASE" "$BRANCH" "$HOLDOUT" "$EXPERIMENT" 0,1,2,3,4,5,6,7
shopt -s nullglob
RUN_DIRS=("$NAVSIM_EXP_ROOT/$EXPERIMENT"/*)
[[ "${#RUN_DIRS[@]}" -eq 1 ]] || {
  echo "expected one Stage31 run directory, got ${#RUN_DIRS[@]}"; exit 2;
}
RUN_DIR="${RUN_DIRS[0]}"
"$PYTHON_BIN" scripts/evaluation/freeze_grpo_stage31_cv_checkpoints.py \
  --phase "$PHASE" --branch "$BRANCH" --holdout "$HOLDOUT" \
  --run-dir "$RUN_DIR" --output "$FREEZE"
"$PYTHON_BIN" scripts/evaluation/audit_grpo_stage31_cv_checkpoint.py \
  --phase "$PHASE" --branch "$BRANCH" --holdout "$HOLDOUT" \
  --base "$PUBLIC" --selector "$SELECTOR" --calibration "$CALIBRATION" \
  --cv-freeze "$CV_FREEZE" --freeze "$FREEZE" --output "$AUDIT"
echo "PASS Stage31 training phase=$PHASE branch=$BRANCH holdout=$HOLDOUT"
echo "Run: $RUN_DIR"
echo "Checkpoint freeze: $FREEZE"
echo "Audit: $AUDIT"
