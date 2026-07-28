#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
CV_FREEZE="$ROOT_DIR/artifacts/grpo_stage30/manifests/cv/freeze.json"
BUCKETS="$ROOT_DIR/artifacts/grpo_stage30/manifests/buckets_6119.json"
PLAN="$ROOT_DIR/GRPO_STAGE31_SELECTOR_CONSISTENT_DECISION_GRPO_PLAN_20260725.md"
JOB_SCRIPT="$ROOT_DIR/run_stage31_cv_job_h100.sh"
TRAIN_SCRIPT="$ROOT_DIR/scripts/training/run_diffusiondrive_grpo_stage31_cv.sh"
FREEZER="$ROOT_DIR/scripts/evaluation/freeze_grpo_stage31_cv_checkpoints.py"
AUDITOR="$ROOT_DIR/scripts/evaluation/audit_grpo_stage31_cv_checkpoint.py"
AUDIT="$ROOT_DIR/artifacts/grpo_stage31/cv/training/DPF/fold0/audit/audit.json"
PLAN_SHA256=3ff3d3bcd3120b9d73a017876f942458eb3fa16651517e4e30b27cf2fff7bb0d

[[ -d "$ROOT_DIR" ]] || { echo "missing Stage31 root: $ROOT_DIR"; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "missing Python: $PYTHON_BIN"; exit 2; }
for path in "$CV_FREEZE" "$BUCKETS" "$PLAN" "$JOB_SCRIPT" "$TRAIN_SCRIPT" "$FREEZER" "$AUDITOR"; do
  [[ -f "$path" ]] || { echo "missing required Stage31 path: $path"; exit 2; }
done
[[ "$(sha256sum "$PLAN" | awk '{print $1}')" == "$PLAN_SHA256" ]] || {
  echo "Stage31 plan SHA256 mismatch: $PLAN"; exit 2;
}
bash -n "$JOB_SCRIPT" "$TRAIN_SCRIPT"
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" -eq 8 ]] || { echo "Stage31 audit requires exactly 8 GPUs"; exit 2; }

"$PYTHON_BIN" - "$ROOT_DIR" "$CV_FREEZE" "$BUCKETS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root, freeze_path, buckets_path = map(Path, sys.argv[1:])
sys.path.insert(0, str(root))
from navsim.agents.diffusiondrive.stage30_mode_coverage import load_stage30_bucket_manifest

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

freeze = json.loads(freeze_path.read_text())
load_stage30_bucket_manifest(str(buckets_path), require_full=True)
if not (
    freeze.get("stage") == 30
    and len(freeze.get("folds", ())) == 4
    and freeze.get("bucket_manifest_sha256") == sha(buckets_path)
):
    raise RuntimeError("immutable Stage30 four-fold manifests are incomplete")
for fold, entry in enumerate(freeze["folds"]):
    path = Path(entry["path"])
    if entry.get("holdout_fold") != fold or not path.is_file() or sha(path) != entry["sha256"]:
        raise RuntimeError(f"Stage30 fold{fold} manifest is incomplete")
PY

if [[ -f "$AUDIT" ]]; then
  "$PYTHON_BIN" - "$AUDIT" <<'PY'
import json
import sys

p = json.load(open(sys.argv[1]))
checks = (
    p.get("passed"), p.get("stage") == 31, p.get("phase") == "audit",
    p.get("branch") == "DPF", p.get("holdout_fold") == 0,
    p.get("num_logged_optimizer_steps") == 1,
    p.get("active_decoder_gradients"), p.get("zero_frozen_gradients"),
    p.get("frozen_reference_bitwise_equal_public"),
    p.get("frozen_training_selector_bitwise_equal"),
    p.get("exact_global_bucket_sampler"),
)
if not all(checks):
    raise RuntimeError("existing Stage31 DPF/fold0 audit is not passing")
PY
  echo "PASS existing Stage31 DPF/fold0 audit: $AUDIT"
  exit 0
fi

echo "All paths and immutable manifests passed; starting Stage31 DPF/fold0 optimizer-step audit."
exec bash "$JOB_SCRIPT" audit DPF 0
