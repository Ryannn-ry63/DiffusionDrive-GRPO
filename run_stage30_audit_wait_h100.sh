#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
CV_FREEZE="$ROOT_DIR/artifacts/grpo_stage30/manifests/cv/freeze.json"
BUCKETS="$ROOT_DIR/artifacts/grpo_stage30/manifests/buckets_6119.json"
AUDIT="$ROOT_DIR/artifacts/grpo_stage30/cv/training/MCC/fold0/audit/audit.json"
TIMEOUT_SECONDS="${STAGE30_WAIT_TIMEOUT_SECONDS:-21600}"

[[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || { echo "invalid timeout"; exit 2; }
[[ -d "$ROOT_DIR" ]] || { echo "missing Stage30 root: $ROOT_DIR"; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "missing Python: $PYTHON_BIN"; exit 2; }
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" -eq 8 ]] || { echo "Stage30 audit requires exactly 8 GPUs"; exit 2; }

deadline=$((SECONDS + TIMEOUT_SECONDS))
echo "Waiting for the locally generated Stage30 Phase-0 manifests..."
while true; do
  if [[ -f "$CV_FREEZE" && -f "$BUCKETS" ]] && \
    "$PYTHON_BIN" - "$ROOT_DIR" "$CV_FREEZE" "$BUCKETS" >/dev/null 2>&1 <<'PY'
import hashlib, json, sys
from pathlib import Path

root, freeze_path, buckets_path = map(Path, sys.argv[1:])
sys.path.insert(0, str(root))
from navsim.agents.diffusiondrive.stage30_mode_coverage import load_stage30_bucket_manifest

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

freeze = json.loads(freeze_path.read_text())
load_stage30_bucket_manifest(str(buckets_path), require_full=True)
if not (freeze.get('stage') == 30 and len(freeze.get('folds', ())) == 4
        and freeze.get('bucket_manifest_sha256') == sha(buckets_path)):
    raise RuntimeError('Stage30 Phase-0 freeze is incomplete')
for fold, entry in enumerate(freeze['folds']):
    path = Path(entry['path'])
    if entry.get('holdout_fold') != fold or not path.is_file() or sha(path) != entry['sha256']:
        raise RuntimeError(f'Stage30 fold{fold} manifest is incomplete')
PY
  then
    break
  fi
  (( SECONDS < deadline )) || { echo "timed out waiting for Stage30 Phase 0"; exit 1; }
  echo "Phase 0 is still running; retrying in 15 seconds..."
  sleep 15
done

if [[ -f "$AUDIT" ]]; then
  "$PYTHON_BIN" - "$AUDIT" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
if not (p.get('passed') and p.get('stage') == 30 and p.get('phase') == 'audit'
        and p.get('branch') == 'MCC' and p.get('holdout_fold') == 0):
    raise RuntimeError('existing Stage30 audit is not passing')
PY
  echo "PASS existing Stage30 MCC/fold0 audit: $AUDIT"
  exit 0
fi

echo "Phase 0 passed; starting the real 8-GPU Stage30 optimizer-step audit."
exec bash "$ROOT_DIR/run_stage30_cv_job_h100.sh" audit MCC 0
