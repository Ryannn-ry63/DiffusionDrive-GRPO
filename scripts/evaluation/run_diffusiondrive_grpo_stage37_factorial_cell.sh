#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 ]]; then echo "Usage: $0 FOLD(0|1) P_JFI|BPD_JFI NOISE GPU"; exit 2; fi
FOLD="$1"; SYSTEM="$2"; NOISE="$3"; GPU="$4"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
PUBLIC="${STAGE37_PUBLIC_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS}"
SELECTION="$ROOT_DIR/artifacts/grpo_stage37/pilot/generator_selection.json"; JFI_FREEZE="$ROOT_DIR/artifacts/grpo_stage37/jfi/training/formal/checkpoints.json"
CALIBRATION="$ROOT_DIR/artifacts/grpo_stage37/jfi/calibration/h${FOLD}/calibration.json"; CV="$ROOT_DIR/artifacts/grpo_stage30/manifests/cv/freeze.json"
[[ "$FOLD" =~ ^[01]$ && "$SYSTEM" =~ ^(P_JFI|BPD_JFI)$ && "$NOISE" =~ ^(20261511|20261512)$ && "$GPU" =~ ^[0-7]$ ]] || { echo "invalid factorial cell args"; exit 2; }
for p in "$PYTHON_BIN" "$PUBLIC" "$SELECTION" "$JFI_FREEZE" "$CALIBRATION" "$CV"; do [[ -f "$p" ]] || { echo "missing factorial input: $p"; exit 2; }; done
readarray -t LOCKED < <("$PYTHON_BIN" - "$SELECTION" "$JFI_FREEZE" "$CALIBRATION" "$CV" "$FOLD" "$SYSTEM" "$PUBLIC" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
s,j,k,c=map(lambda x:json.loads(Path(x).read_text()),sys.argv[1:5]); fold=int(sys.argv[5]); system=sys.argv[6]; public=Path(sys.argv[7])
je=j['checkpoints'][0]; jp=Path(je['path']); entry=c['folds'][fold]; manifest=Path(entry['holdout_manifest'])
if not s.get('passed') or not k.get('passed') or k.get('holdout_fold')!=fold or k.get('selector_checkpoint_sha256')!=je['sha256']:raise RuntimeError('factorial freeze/calibration drift')
if sha(jp)!=je['sha256'] or sha(manifest)!=entry['holdout_manifest_sha256']:raise RuntimeError('JFI/manifest SHA drift')
if system=='P_JFI':generator=public
else:
 ge=s['folds'][str(fold)];generator=Path(ge['checkpoint'])
 if sha(generator)!=ge['checkpoint_sha256']:raise RuntimeError('generator SHA drift')
print(generator);print(jp);print(manifest);print(entry['holdout_count']);print(s['selected_step'])
PY
)
GENERATOR="${LOCKED[0]}"; JFI="${LOCKED[1]}"; MANIFEST="${LOCKED[2]}"; LIMIT="${LOCKED[3]}"; STEP="${LOCKED[4]}"
DOMAIN="stage37_factorial_${SYSTEM,,}_fold${FOLD}"; OUTPUT="$ROOT_DIR/artifacts/grpo_stage37/pilot/factorial/fold${FOLD}/${SYSTEM}_ns${NOISE}.json"
[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite $OUTPUT"; exit 2; }
ALGO=legacy_ppo; [[ "$SYSTEM" == BPD_JFI ]] && ALGO=diffgrpo_bistate_projected_deployment
GRPO_EVAL_GENERATION_ALGORITHM="$ALGO" PYTHON_BIN="$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage37_jfi_eval.sh" deploy "$GENERATOR" "$DOMAIN" "$JFI" "$CALIBRATION" "$MANIFEST" "$LIMIT" "$NOISE" "$OUTPUT" "$GPU"
"$PYTHON_BIN" - "$OUTPUT" "$DOMAIN" "$LIMIT" "$NOISE" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))['summary'];j=s['stage37_jfi_selector']
assert s['completed'] and s['generator_domain']==sys.argv[2] and s['num_tokens']==int(sys.argv[3]) and s['evaluation_noise_namespace']==int(sys.argv[4])
assert j['max_candidates']==4 and 1<=j['mean_candidate_pool_size']<=4
print(f"PASS factorial selected={s['selected_reward']:.9f} pool={j['mean_candidate_pool_size']:.4f}")
PY
echo "PASS Stage37 factorial fold=$FOLD system=$SYSTEM step=$STEP noise=$NOISE"
