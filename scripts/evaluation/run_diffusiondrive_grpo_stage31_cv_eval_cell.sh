#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 DP|DPF HOLDOUT(0..3) JOB_ID GPU"
  exit 2
fi
BRANCH="$1"; HOLDOUT="$2"; JOB_ID="$3"; GPU="$4"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
PUBLIC=/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS
SELECTOR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage27_selector_multi_stage25_formal_seed27125/2026.07.24.09.28.29/lightning_logs/version_0/checkpoints/grpo-17-36684.ckpt
CALIBRATION="$ROOT_DIR/artifacts/grpo_stage27/selectors/multi/calibration.json"
CV_FREEZE="$ROOT_DIR/artifacts/grpo_stage30/manifests/cv/freeze.json"
AUDIT="$ROOT_DIR/artifacts/grpo_stage31/cv/training/$BRANCH/fold$HOLDOUT/formal/audit.json"
OUT_DIR="$ROOT_DIR/artifacts/grpo_stage31/cv/eval/$BRANCH/fold$HOLDOUT"
NOISES=(20261411 20261412)
STEPS=(48 96 144 192)

[[ "$BRANCH" == DP || "$BRANCH" == DPF ]] || { echo "bad branch"; exit 2; }
[[ "$HOLDOUT" =~ ^[0-3]$ ]] || { echo "bad holdout"; exit 2; }
MAX_JOB=7; [[ "$BRANCH" == DPF ]] && MAX_JOB=9
(( JOB_ID >= 0 && JOB_ID <= MAX_JOB )) || { echo "bad job id"; exit 2; }
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "bad GPU"; exit 2; }
for path in "$PUBLIC" "$SELECTOR" "$CALIBRATION" "$CV_FREEZE" "$AUDIT"; do
  [[ -f "$path" ]] || { echo "missing Stage31 eval input: $path"; exit 2; }
done

if (( JOB_ID < 8 )); then
  STEP_INDEX=$((JOB_ID / 2)); NOISE_INDEX=$((JOB_ID % 2))
  STEP="${STEPS[$STEP_INDEX]}"; NOISE="${NOISES[$NOISE_INDEX]}"
  SYSTEM="${BRANCH}${STEP}"
else
  NOISE_INDEX=$((JOB_ID - 8)); NOISE="${NOISES[$NOISE_INDEX]}"
  SYSTEM=P
fi
readarray -t LOCKED < <(
  "$PYTHON_BIN" - "$CV_FREEZE" "$AUDIT" "$BRANCH" "$HOLDOUT" "$SYSTEM" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
cvp,ap,branch,fold,system=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],int(sys.argv[4]),sys.argv[5]
cv=json.loads(cvp.read_text()); audit=json.loads(ap.read_text()); entry=cv['folds'][fold]
if not (audit.get('passed') and audit.get('stage')==31 and audit.get('phase')=='formal'
        and audit.get('branch')==branch and audit.get('holdout_fold')==fold
        and audit.get('cv_freeze_sha256')==sha(cvp)):
 raise RuntimeError('Stage31 formal audit drifted')
manifest=Path(entry['holdout_manifest'])
if sha(manifest)!=entry['holdout_manifest_sha256']: raise RuntimeError('holdout drifted')
if system=='P':
 checkpoint=Path('/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS')
 checkpoint_sha='008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b'
else:
 step=int(system[len(branch):]); matches=[x for x in audit['checkpoints'] if x['global_step']==step]
 if len(matches)!=1: raise RuntimeError('Stage31 checkpoint step missing')
 checkpoint=Path(matches[0]['path']); checkpoint_sha=matches[0]['sha256']
 if sha(checkpoint)!=checkpoint_sha: raise RuntimeError('checkpoint drifted')
print(checkpoint); print(checkpoint_sha); print(manifest)
print(entry['holdout_count']); print(entry['holdout_manifest_sha256'])
PY
)
[[ "${#LOCKED[@]}" -eq 5 ]] || { echo "failed Stage31 eval resolution"; exit 2; }
GENERATOR="${LOCKED[0]}"; GENERATOR_SHA="${LOCKED[1]}"; MANIFEST="${LOCKED[2]}"
COUNT="${LOCKED[3]}"; MANIFEST_SHA="${LOCKED[4]}"
DOMAIN="stage31_cv_${SYSTEM,,}_fold${HOLDOUT}"
OUTPUT="$OUT_DIR/${SYSTEM}_ns${NOISE}.json"
[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite $OUTPUT"; exit 2; }
[[ "$DOMAIN" =~ ^[a-z0-9_]+$ ]] || { echo "invalid Stage31 generator domain"; exit 2; }
mkdir -p "$OUT_DIR"
if [[ "${GRPO_STAGE31_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  GRPO_BASE_CHECKPOINT="$PUBLIC" PYTHON_BIN="$PYTHON_BIN" \
    GRPO_SELECTOR_EVAL_PREFLIGHT_ONLY=1 \
    "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage25_selector_eval.sh" \
    deploy "$GENERATOR" "$DOMAIN" "$SELECTOR" "$CALIBRATION" \
    "$MANIFEST" "$COUNT" "$NOISE" "$OUTPUT" "$GPU"
  echo "PASS Stage31 eval preflight $SYSTEM fold=$HOLDOUT ns=$NOISE count=$COUNT"
  exit 0
fi
GRPO_BASE_CHECKPOINT="$PUBLIC" PYTHON_BIN="$PYTHON_BIN" \
  "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage25_selector_eval.sh" \
  deploy "$GENERATOR" "$DOMAIN" "$SELECTOR" "$CALIBRATION" \
  "$MANIFEST" "$COUNT" "$NOISE" "$OUTPUT" "$GPU"

"$PYTHON_BIN" - "$OUTPUT" "$GENERATOR_SHA" "$NOISE" "$COUNT" "$MANIFEST_SHA" "$DOMAIN" <<'PY'
import hashlib,json,sys
def ordered(values): return hashlib.sha256('\n'.join(values).encode()).hexdigest()
p=json.load(open(sys.argv[1])); s=p['summary']; records=p['records']; selector=s['stage25_selector']
checks={
 'complete':s['completed'] and s['num_failures']==0,
 'count':s['num_tokens']==int(sys.argv[4])==len(records),
 'checkpoint':s['checkpoint_sha256']==sys.argv[2],
 'reference':s['reference_checkpoint_sha256']=='008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b',
 'noise':s['evaluation_noise_namespace']==int(sys.argv[3]),
 'domain':s['generator_domain']==sys.argv[6],
 'selector':selector['checkpoint_sha256']=='023d7b6b77bb8fa2dc3e779849f3716688abf0796b019416494c80b29615b691',
 'calibration':selector['calibration_sha256']=='fec2a10573e913e4452f8eea92e6334fcd5e6f1104e6244876770973a1f41990',
 'source':s['selector_logits_source']=='trajectory_relative_harm_v3',
 'schedule':s['schedule']['roll_timesteps']==[32,24,16,8,0],
 'tokens':s['token_set_sha256']==ordered([str(x['token']) for x in records]),
}
bad=[k for k,v in checks.items() if not v]
if bad: raise RuntimeError(f'Stage31 eval artifact failed {bad}')
PY
echo "PASS Stage31 eval $SYSTEM fold=$HOLDOUT ns=$NOISE: $OUTPUT"
