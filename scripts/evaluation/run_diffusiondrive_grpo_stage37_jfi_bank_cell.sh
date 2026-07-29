#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 P|DPEL192|NCD192|RGT192 NOISE(20263721|20263722) GPU"
  exit 2
fi
SYSTEM="$1"; NOISE="$2"; GPU="$3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
PUBLIC="${STAGE37_PUBLIC_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS}"
MANIFEST_FREEZE="$ROOT_DIR/artifacts/grpo_stage37/jfi/manifests/freeze.json"
OUT_DIR="$ROOT_DIR/artifacts/grpo_stage37/jfi/banks"

[[ "$SYSTEM" =~ ^(P|DPEL192|NCD192|RGT192)$ ]] || { echo "invalid system"; exit 2; }
[[ "$NOISE" == 20263721 || "$NOISE" == 20263722 ]] || { echo "invalid noise"; exit 2; }
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "GPU must be 0..7"; exit 2; }
for path in "$PYTHON_BIN" "$PUBLIC" "$MANIFEST_FREEZE"; do
  [[ -f "$path" ]] || { echo "missing Stage37 bank input: $path"; exit 2; }
done

readarray -t LOCKED < <(
  "$PYTHON_BIN" - "$ROOT_DIR" "$SYSTEM" "$PUBLIC" "$MANIFEST_FREEZE" <<'PY'
import hashlib, json, sys
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

root, system, public, mf = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
frozen=json.loads(mf.read_text()); manifest=Path(frozen['train_manifest'])
if frozen.get('stage') != 37 or not manifest.is_file() or sha(manifest) != frozen['train_manifest_sha256']:
    raise RuntimeError('Stage37 JFI manifest freeze drifted')
if sha(public) != '008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b':
    raise RuntimeError('public checkpoint drifted')
if system == 'P':
    checkpoint=public
else:
    stage={'DPEL192':32,'NCD192':35,'RGT192':36}[system]
    branch={'DPEL192':'DPEL','NCD192':'NCD','RGT192':'RGT'}[system]
    freeze_path=root/f'artifacts/grpo_stage{stage}/pilot/training/{branch}/fold0/formal/checkpoints.json'
    freeze=json.loads(freeze_path.read_text())
    matches=[x for x in freeze['checkpoints'] if int(x['global_step']) == 192]
    if not freeze.get('passed') or freeze.get('stage') != stage or len(matches) != 1:
        raise RuntimeError(f'{system} freeze drifted')
    checkpoint=Path(matches[0]['path'])
    if not checkpoint.is_file() or sha(checkpoint) != matches[0]['sha256']:
        raise RuntimeError(f'{system} checkpoint drifted')
print(checkpoint)
print(manifest)
print(frozen.get('count', 2035) if system == 'P' else json.loads(mf.read_text())['count'])
print(json.loads(mf.read_text())['train_manifest_sha256'])
PY
)
[[ "${#LOCKED[@]}" -eq 4 ]] || { echo "failed bank input resolution"; exit 2; }
GENERATOR="${LOCKED[0]}"; MANIFEST="${LOCKED[1]}"; LIMIT="${LOCKED[2]}"; MANIFEST_SHA="${LOCKED[3]}"
DOMAIN="stage37_jfi_${SYSTEM,,}_folds23"
OUTPUT="$OUT_DIR/${SYSTEM}_ns${NOISE}.json"
[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite $OUTPUT"; exit 2; }
mkdir -p "$OUT_DIR"

export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/navsim}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

if [[ "${GRPO_STAGE37_BANK_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "PASS Stage37 JFI bank preflight system=$SYSTEM noise=$NOISE generator=$GENERATOR"
  exit 0
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/evaluate_grpo_schedule.py" \
  --checkpoint "$GENERATOR" --reference-checkpoint "$PUBLIC" \
  --cache-path "${NAVSIM_TRAINING_CACHE:-$NAVSIM_EXP_ROOT/training_cache}" \
  --metric-cache-path "${NAVSIM_METRIC_CACHE:-$NAVSIM_EXP_ROOT/metric_cache_trainval}" \
  --tokens-file "$MANIFEST" --limit "$LIMIT" --log-split train \
  --batch-size 2 --num-workers 4 --device cuda:0 \
  --truncation-timestep 32 --roll-timesteps 32 24 16 8 0 \
  --scheduler-num-inference-steps 125 --generation-policy-algorithm legacy_ppo \
  --evaluation-noise-namespace "$NOISE" --selector-logits-source current \
  --generator-domain "$DOMAIN" --store-candidate-trajectories --output "$OUTPUT"

"$PYTHON_BIN" - "$OUTPUT" "$LIMIT" "$DOMAIN" "$MANIFEST_SHA" <<'PY'
import hashlib,json,sys
p=json.load(open(sys.argv[1])); s=p['summary']; r=p['records']
token_sha=hashlib.sha256('\n'.join(x['token'] for x in r).encode()).hexdigest()
assert s['completed'] and s['num_failures']==0 and len(r)==int(sys.argv[2])
assert s['generator_domain']==sys.argv[3] and s['stores_candidate_trajectories']
assert all(len(x['candidate_trajectories'])==20 for x in r)
assert token_sha==s['token_set_sha256']
print(f"PASS Stage37 JFI bank {sys.argv[3]} records={len(r)} manifest={sys.argv[4]}")
PY
