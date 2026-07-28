#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [audit|final]"
  exit 2
fi
PHASE="${1:-final}"
[[ "$PHASE" == audit || "$PHASE" == final ]] || {
  echo "PHASE must be audit or final"
  exit 2
}

ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/navsim}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"

PUBLIC=/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS
METRIC_CACHE="${STAGE27_NAVTEST_METRIC_CACHE:-$NAVSIM_EXP_ROOT/metric_cache_navtest}"
EVAL_ENTRY="$NAVSIM_DEVKIT_ROOT/planning/script/run_pdm_score.py"
[[ -f "$PUBLIC" ]] || { echo "missing public 88.1 checkpoint: $PUBLIC"; exit 2; }
[[ -d "$METRIC_CACHE" ]] || { echo "missing NavTest metric cache: $METRIC_CACHE"; exit 2; }
[[ -f "$EVAL_ENTRY" ]] || { echo "missing NavTest entry: $EVAL_ENTRY"; exit 2; }

"$PYTHON_BIN" -c '
import hashlib,pathlib,sys
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()
assert sha(sys.argv[1]) == (
    "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
)
count=sum(1 for _ in pathlib.Path(sys.argv[2]).rglob("metric_cache.pkl"))
assert count == 12146, f"expected 12146 NavTest caches, got {count}"
' "$PUBLIC" "$METRIC_CACHE"

EXPERIMENT=stage27_public88_navtest_P
WORKER=ray_distributed
EXTRA_ARGS=(worker.threads_per_node="${STAGE27_NAVTEST_WORKERS:-64}")
if [[ "$PHASE" == audit ]]; then
  EXPERIMENT=stage27_public88_navtest_audit_P
  WORKER=sequential
  EXTRA_ARGS=(train_test_split.scene_filter.max_scenes=1)
else
  "$PYTHON_BIN" -c '
from pathlib import Path
import sys
root=Path(sys.argv[1])
csvs=list(root.glob("*/*.csv")) if root.is_dir() else []
if csvs:
    raise SystemExit("completed Stage27 P CSV already exists: " + str(csvs[-1]))
' "$NAVSIM_EXP_ROOT/$EXPERIMENT"
fi

cd "$ROOT_DIR"
echo "Stage27 public-88.1 NavTest phase=$PHASE experiment=$EXPERIMENT"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PYTHON_BIN" "$EVAL_ENTRY" \
  train_test_split=navtest agent=diffusiondrive_agent worker="$WORKER" \
  metric_cache_path="$METRIC_CACHE" experiment_name="$EXPERIMENT" \
  agent.checkpoint_path="$PUBLIC" agent.reference_checkpoint_path="$PUBLIC" \
  agent.config.inference_selector_source=current \
  agent.config.diffusion_truncation_timestep=8 \
  agent.config.diffusion_roll_timesteps='[8,0]' \
  agent.config.diffusion_scheduler_num_inference_steps=125 \
  agent.config.evaluation_noise_namespace=-1 \
  "${EXTRA_ARGS[@]}"
