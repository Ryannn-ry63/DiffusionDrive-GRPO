#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 ]]; then echo "Usage: $0 HOLDOUT(0|1) P|BPD NOISE GPU"; exit 2; fi
HOLDOUT="$1"; SYSTEM="$2"; NOISE="$3"; GPU="$4"; CALIB_FOLD=$((1-HOLDOUT))
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
PUBLIC="${STAGE37_PUBLIC_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS}"
SELECTION="$ROOT_DIR/artifacts/grpo_stage37/pilot/generator_selection.json"; JFI_FREEZE="$ROOT_DIR/artifacts/grpo_stage37/jfi/training/formal/checkpoints.json"; CV="$ROOT_DIR/artifacts/grpo_stage30/manifests/cv/freeze.json"
[[ "$HOLDOUT" =~ ^[01]$ && "$SYSTEM" =~ ^(P|BPD)$ && "$NOISE" =~ ^(20261511|20261512)$ && "$GPU" =~ ^[0-7]$ ]] || { echo "invalid calibration cell args"; exit 2; }
for p in "$PYTHON_BIN" "$PUBLIC" "$SELECTION" "$JFI_FREEZE" "$CV"; do [[ -f "$p" ]] || { echo "missing JFI calibration input: $p"; exit 2; }; done
readarray -t LOCKED < <("$PYTHON_BIN" - "$SELECTION" "$JFI_FREEZE" "$CV" "$HOLDOUT" "$SYSTEM" "$PUBLIC" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
s,j,c=map(lambda x:json.loads(Path(x).read_text()),sys.argv[1:4]); h=int(sys.argv[4]); system=sys.argv[5]; public=Path(sys.argv[6]); cf=1-h
if not s.get('passed') or s.get('plan_sha256')!='4dc469dd0be4d2579796ff70ae7a09ed9fd8e4c92449cd057e42463d268bb5d8':raise RuntimeError('generator selection drift')
je=j['checkpoints'][0]; jp=Path(je['path']); entry=c['folds'][cf]; manifest=Path(entry['holdout_manifest'])
if sha(jp)!=je['sha256'] or sha(manifest)!=entry['holdout_manifest_sha256']:raise RuntimeError('JFI/manifest drift')
if system=='P': generator=public
else:
 ge=s['folds'][str(h)]; generator=Path(ge['checkpoint'])
 if sha(generator)!=ge['checkpoint_sha256']:raise RuntimeError('generator drift')
print(generator);print(jp);print(manifest);print(entry['holdout_count']);print(s['selected_step'])
PY
)
GENERATOR="${LOCKED[0]}"; JFI="${LOCKED[1]}"; MANIFEST="${LOCKED[2]}"; LIMIT="${LOCKED[3]}"; STEP="${LOCKED[4]}"
DOMAIN="stage37_jfi_calib_h${HOLDOUT}_${SYSTEM,,}_fold${CALIB_FOLD}"; OUTPUT="$ROOT_DIR/artifacts/grpo_stage37/jfi/calibration/h${HOLDOUT}/collect/${SYSTEM}_ns${NOISE}.json"
[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite $OUTPUT"; exit 2; }
ALGO=legacy_ppo; [[ "$SYSTEM" == BPD ]] && ALGO=diffgrpo_bistate_projected_deployment
GRPO_EVAL_GENERATION_ALGORITHM="$ALGO" PYTHON_BIN="$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage37_jfi_eval.sh" collect "$GENERATOR" "$DOMAIN" "$JFI" - "$MANIFEST" "$LIMIT" "$NOISE" "$OUTPUT" "$GPU"
echo "PASS JFI calibration collect holdout=$HOLDOUT system=$SYSTEM step=$STEP noise=$NOISE"
