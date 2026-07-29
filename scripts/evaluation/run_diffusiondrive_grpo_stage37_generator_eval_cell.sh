#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then echo "Usage: $0 FOLD(0|1) STEP(48|96|144|192) NOISE GPU"; exit 2; fi
FOLD="$1"; STEP="$2"; NOISE="$3"; GPU="$4"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
PUBLIC="${STAGE37_PUBLIC_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS}"
SELECTOR="${STAGE37_SELECTOR_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage27_selector_multi_stage25_formal_seed27125/2026.07.24.09.28.29/lightning_logs/version_0/checkpoints/grpo-17-36684.ckpt}"
CALIBRATION="${STAGE37_SELECTOR_CALIBRATION:-$ROOT_DIR/artifacts/grpo_stage27/selectors/multi/calibration.json}"
CV_FREEZE="$ROOT_DIR/artifacts/grpo_stage30/manifests/cv/freeze.json"
GEN_FREEZE="$ROOT_DIR/artifacts/grpo_stage37/generator/fold${FOLD}/formal/checkpoints.json"
OUT_DIR="$ROOT_DIR/artifacts/grpo_stage37/pilot/generator_eval/fold${FOLD}"
[[ "$FOLD" =~ ^[01]$ && "$STEP" =~ ^(48|96|144|192)$ ]] || { echo "invalid fold/step"; exit 2; }
[[ "$NOISE" == 20261511 || "$NOISE" == 20261512 ]] || { echo "invalid noise"; exit 2; }
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "invalid GPU"; exit 2; }
for path in "$PYTHON_BIN" "$PUBLIC" "$SELECTOR" "$CALIBRATION" "$CV_FREEZE" "$GEN_FREEZE"; do
  [[ -f "$path" ]] || { echo "missing Stage37 generator eval input: $path"; exit 2; }
done
readarray -t LOCKED < <(
 "$PYTHON_BIN" - "$CV_FREEZE" "$GEN_FREEZE" "$FOLD" "$STEP" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
cv,gf=map(Path,sys.argv[1:3]); fold,step=map(int,sys.argv[3:])
c=json.loads(cv.read_text()); g=json.loads(gf.read_text()); entry=c['folds'][fold]
manifest=Path(entry['holdout_manifest']); matches=[x for x in g['checkpoints'] if int(x['global_step'])==step]
if entry['holdout_fold']!=fold or sha(manifest)!=entry['holdout_manifest_sha256']: raise RuntimeError('holdout drift')
if not g.get('passed') or g.get('stage')!=37 or g.get('plan_sha256')!='4dc469dd0be4d2579796ff70ae7a09ed9fd8e4c92449cd057e42463d268bb5d8' or len(matches)!=1: raise RuntimeError('generator freeze drift')
checkpoint=Path(matches[0]['path'])
if sha(checkpoint)!=matches[0]['sha256']: raise RuntimeError('checkpoint SHA drift')
print(checkpoint); print(matches[0]['sha256']); print(manifest); print(entry['holdout_count']); print(entry['holdout_manifest_sha256'])
PY
)
[[ "${#LOCKED[@]}" -eq 5 ]] || exit 2
GENERATOR="${LOCKED[0]}"; GEN_SHA="${LOCKED[1]}"; MANIFEST="${LOCKED[2]}"; LIMIT="${LOCKED[3]}"; MANIFEST_SHA="${LOCKED[4]}"
DOMAIN="stage37_pilot_bpd${STEP}_fold${FOLD}"; OUTPUT="$OUT_DIR/BPD${STEP}_ns${NOISE}.json"
[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite $OUTPUT"; exit 2; }; mkdir -p "$OUT_DIR"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/navsim}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
if [[ "${GRPO_STAGE37_EVAL_PREFLIGHT_ONLY:-0}" == 1 ]]; then echo "PASS Stage37 generator eval preflight fold=$FOLD step=$STEP noise=$NOISE"; exit 0; fi
GRPO_BASE_CHECKPOINT="$PUBLIC" GRPO_EVAL_GENERATION_ALGORITHM=diffgrpo_bistate_projected_deployment PYTHON_BIN="$PYTHON_BIN" \
 "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage25_selector_eval.sh" deploy "$GENERATOR" "$DOMAIN" "$SELECTOR" "$CALIBRATION" "$MANIFEST" "$LIMIT" "$NOISE" "$OUTPUT" "$GPU"
"$PYTHON_BIN" - "$OUTPUT" "$GEN_SHA" "$DOMAIN" "$NOISE" "$LIMIT" "$MANIFEST_SHA" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); s=p['summary']
assert s['completed'] and s['num_failures']==0 and s['checkpoint_sha256']==sys.argv[2]
assert s['generator_domain']==sys.argv[3] and s['evaluation_noise_namespace']==int(sys.argv[4]) and s['num_tokens']==int(sys.argv[5])
assert s['generation_policy_algorithm']=='diffgrpo_bistate_projected_deployment'
print(f"PASS Stage37 BPD eval selected={s['selected_reward']:.9f} manifest={sys.argv[6]}")
PY
