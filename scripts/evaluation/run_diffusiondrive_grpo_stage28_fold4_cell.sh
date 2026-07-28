#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 JOB_ID(0..25) GPU(0..7)"
  exit 2
fi
JOB_ID="$1"
GPU="$2"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
INPUTS="$ROOT_DIR/artifacts/grpo_stage28/pilot/fold4/inputs.json"
OUT_DIR="$ROOT_DIR/artifacts/grpo_stage28/pilot/fold4/results"
PUBLIC=/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS

[[ "$JOB_ID" =~ ^([0-9]|1[0-9]|2[0-5])$ ]] || { echo "JOB_ID must be 0..25"; exit 2; }
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "GPU must be 0..7"; exit 2; }
for path in "$INPUTS" "$PUBLIC"; do [[ -f "$path" ]] || { echo "missing $path"; exit 2; }; done

SYSTEM_INDEX=$((JOB_ID / 2))
NOISE_INDEX=$((JOB_ID % 2))
SYSTEMS=(P A1 A2 A3 A4 B1 B2 B3 B4 C1 C2 C3 C4)
NOISES=(20261111 20261112)
SYSTEM="${SYSTEMS[$SYSTEM_INDEX]}"
NOISE="${NOISES[$NOISE_INDEX]}"

readarray -t LOCKED < <(
  "$PYTHON_BIN" - "$INPUTS" "$SYSTEM" "$NOISE" <<'PY'
import hashlib, json, sys
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

path, system, noise = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
p=json.loads(path.read_text())
if not (p.get('passed') and p.get('stage')==28 and p.get('phase')=='fold4'
        and p.get('all_evaluations_use_safe_multi') is True
        and p.get('exploration_selector_deployment_forbidden') is True
        and system in p['systems'] and noise in p['noise_namespaces']):
    raise RuntimeError('Stage28 fold4 frozen grid drifted')
for key in ('manifest','selector_checkpoint','calibration'):
    q=Path(p[key])
    if not q.is_file() or sha(q)!=p[f'{key}_sha256']:
        raise RuntimeError(f'Stage28 fold4 locked input drifted: {key}')
entry=p['systems'][system]
q=Path(entry['path'])
if not q.is_file() or sha(q)!=entry['sha256']:
    raise RuntimeError('Stage28 fold4 generator SHA drifted')
print(entry['path']); print(entry['sha256']); print(entry['domain'])
print(p['selector_checkpoint']); print(p['selector_checkpoint_sha256'])
print(p['calibration']); print(p['calibration_sha256']); print(p['manifest'])
print(sha(path))
PY
)
[[ "${#LOCKED[@]}" -eq 9 ]] || { echo "failed to resolve Stage28 cell"; exit 2; }
GENERATOR="${LOCKED[0]}"; GENERATOR_SHA="${LOCKED[1]}"; DOMAIN="${LOCKED[2]}"
SELECTOR="${LOCKED[3]}"; SELECTOR_SHA="${LOCKED[4]}"; CALIBRATION="${LOCKED[5]}"
CALIBRATION_SHA="${LOCKED[6]}"; MANIFEST="${LOCKED[7]}"; INPUTS_SHA="${LOCKED[8]}"
OUTPUT="$OUT_DIR/${SYSTEM}_ns${NOISE}.json"
[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite $OUTPUT"; exit 2; }

if [[ "${GRPO_STAGE28_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "PASS Stage28 fold4 preflight $SYSTEM namespace=$NOISE inputs_sha=$INPUTS_SHA"
  exit 0
fi
GRPO_BASE_CHECKPOINT="$PUBLIC" PYTHON_BIN="$PYTHON_BIN" \
  "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage25_selector_eval.sh" \
  deploy "$GENERATOR" "$DOMAIN" "$SELECTOR" "$CALIBRATION" \
  "$MANIFEST" 1021 "$NOISE" "$OUTPUT" "$GPU"

"$PYTHON_BIN" - "$OUTPUT" "$GENERATOR_SHA" "$DOMAIN" "$NOISE" \
  "$SELECTOR_SHA" "$CALIBRATION_SHA" <<'PY'
import json, sys
p=json.load(open(sys.argv[1])); s=p['summary']; sel=s['stage25_selector']; sch=s['schedule']
checks={
 'complete':s['completed'] and s['num_failures']==0,
 'count':s['num_tokens']==1021 and len(p['records'])==1021,
 'generator':s['checkpoint_sha256']==sys.argv[2],
 'domain':s['generator_domain']==sys.argv[3],
 'reference':s['reference_checkpoint_sha256']=='008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b',
 'noise':s['evaluation_noise_namespace']==int(sys.argv[4]),
 'safe_selector':sel['checkpoint_sha256']==sys.argv[5],
 'safe_calibration':sel['calibration_sha256']==sys.argv[6] and not sel['calibration_collection'],
 'source':s['selector_logits_source']=='trajectory_relative_harm_v3',
 'schedule':sch['truncation_timestep']==32 and sch['roll_timesteps']==[32,24,16,8,0]
            and sch['scheduler_num_inference_steps']==125,
}
bad=[k for k,v in checks.items() if not v]
if bad: raise RuntimeError(f'Stage28 fold4 artifact failed {bad}')
PY
echo "PASS Stage28 fold4 $SYSTEM namespace=$NOISE: $OUTPUT"
