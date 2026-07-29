#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 P20|P40|BC|STD|SET FOLD(0|1|2|3) STEP(0|48|96|192) NOISE GPU"
  exit 2
fi
ROLE="${1^^}"
FOLD="$2"
STEP="$3"
NOISE="$4"
GPU="$5"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
PUBLIC="${STAGE39_PUBLIC_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS}"
PUBLIC_SHA="008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
CV_FREEZE="${STAGE39_CV_FREEZE:-$ROOT_DIR/artifacts/grpo_stage30/manifests/cv/freeze.json}"
TRAIN_CACHE="${NAVSIM_TRAINING_CACHE:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache}"
METRIC_CACHE="${NAVSIM_METRIC_CACHE:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/metric_cache_trainval}"
PLAN_SHA="7e1e6142412778524714a0b919f971edf1d2bf14138b0efcf21fc9dd7c4d6f1b"

[[ "$ROLE" =~ ^(P20|P40|BC|STD|SET)$ ]] || { echo "invalid Stage39 eval role"; exit 2; }
[[ "$FOLD" =~ ^[0-3]$ ]] || { echo "invalid Stage39 eval fold"; exit 2; }
if [[ "$FOLD" == 2 ]]; then
  EVAL_PHASE=pilot
  [[ "$NOISE" == 20261711 || "$NOISE" == 20261712 ]] || {
    echo "Stage39 pilot requires namespace 20261711 or 20261712"; exit 2;
  }
else
  EVAL_PHASE=formal
  [[ "$FOLD" =~ ^(0|1|3)$ ]] || { echo "Stage39 formal folds are 0/1/3"; exit 2; }
  [[ "$NOISE" == 20261721 || "$NOISE" == 20261722 ]] || {
    echo "Stage39 formal requires namespace 20261721 or 20261722"; exit 2;
  }
fi
OUT_DIR="$ROOT_DIR/artifacts/grpo_stage39/$EVAL_PHASE/eval/fold${FOLD}"
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "invalid Stage39 eval GPU"; exit 2; }
for path in "$PYTHON_BIN" "$PUBLIC" "$CV_FREEZE" "$ROOT_DIR/scripts/evaluation/evaluate_grpo_schedule.py"; do
  [[ -e "$path" ]] || { echo "missing Stage39 eval input: $path"; exit 2; }
done
[[ "$(sha256sum "$PUBLIC" | awk '{print $1}')" == "$PUBLIC_SHA" ]] || {
  echo "Stage39 public checkpoint SHA drifted"; exit 2;
}

case "$ROLE:$STEP" in
  P20:0)
    GENERATOR="$PUBLIC"; GEN_SHA="$PUBLIC_SHA"; SOURCE=public20; LABEL=P20
    ;;
  P40:0)
    GENERATOR="$PUBLIC"; GEN_SHA="$PUBLIC_SHA"; SOURCE=public40_extra; LABEL=P40
    ;;
  BC:48|BC:96|BC:192|STD:48|STD:96|STD:192|SET:48|SET:96|SET:192)
    TRAIN_FREEZE="$ROOT_DIR/artifacts/grpo_stage39/$EVAL_PHASE/training/$ROLE/fold${FOLD}/checkpoints.json"
    if [[ "$EVAL_PHASE" == formal ]]; then
      SELECTION="${STAGE39_SELECTION:-$ROOT_DIR/artifacts/grpo_stage39/pilot/selection.json}"
      [[ -f "$SELECTION" ]] || { echo "missing Stage39 pilot selection: $SELECTION"; exit 2; }
      EXPECTED_STEP=$("$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p.get("passed") is True; print(int(p["selected_step"]))' "$SELECTION")
      [[ "$STEP" == "$EXPECTED_STEP" ]] || { echo "formal step must equal pilot-selected $EXPECTED_STEP"; exit 2; }
    fi
    [[ -f "$TRAIN_FREEZE" ]] || { echo "missing Stage39 training freeze: $TRAIN_FREEZE"; exit 2; }
    readarray -t LOCKED < <(
      "$PYTHON_BIN" - "$TRAIN_FREEZE" "$ROLE" "$FOLD" "$STEP" "$PLAN_SHA" "$EVAL_PHASE" <<'PYLOCK'
import hashlib,json,sys
from pathlib import Path

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()
freeze=json.loads(Path(sys.argv[1]).read_text())
branch,fold,step,plan,phase=sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),sys.argv[5],sys.argv[6]
matches=[r for r in freeze.get('checkpoints',[]) if int(r['global_step'])==step]
if not (freeze.get('passed') is True and freeze.get('stage')==39 and
        freeze.get('phase')==phase and freeze.get('branch')==branch and
        freeze.get('holdout_fold')==fold and freeze.get('plan_sha256')==plan and
        len(matches)==1):
    raise RuntimeError('Stage39 checkpoint freeze drifted')
path=Path(matches[0]['path'])
if not path.is_file() or sha(path)!=matches[0]['sha256']:
    raise RuntimeError('Stage39 checkpoint SHA drifted')
print(path)
print(matches[0]['sha256'])
PYLOCK
    )
    [[ "${#LOCKED[@]}" -eq 2 ]] || { echo "failed Stage39 checkpoint resolution"; exit 2; }
    GENERATOR="${LOCKED[0]}"; GEN_SHA="${LOCKED[1]}"; SOURCE=challenger_union; LABEL="${ROLE}${STEP}"
    ;;
  *) echo "role/step mismatch: P20/P40=0; BC/STD/SET=48|96|192"; exit 2 ;;
esac

readarray -t CV_LOCKED < <(
  "$PYTHON_BIN" - "$CV_FREEZE" "$FOLD" <<'PYCV'
import hashlib,json,sys
from pathlib import Path

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()
freeze=json.loads(Path(sys.argv[1]).read_text())
fold=int(sys.argv[2]); entry=freeze['folds'][fold]; manifest=Path(entry['holdout_manifest'])
if not (freeze.get('stage')==30 and entry['holdout_fold']==fold and manifest.is_file() and
        sha(manifest)==entry['holdout_manifest_sha256']):
    raise RuntimeError('Stage39 held-out manifest drifted')
print(manifest); print(entry['holdout_count']); print(entry['holdout_manifest_sha256'])
PYCV
)
[[ "${#CV_LOCKED[@]}" -eq 3 ]] || { echo "failed Stage39 CV resolution"; exit 2; }
MANIFEST="${CV_LOCKED[0]}"; LIMIT="${CV_LOCKED[1]}"; MANIFEST_SHA="${CV_LOCKED[2]}"
OUTPUT="$OUT_DIR/${LABEL}_ns${NOISE}.json"
DOMAIN="stage39_${EVAL_PHASE}_${LABEL,,}_fold${FOLD}"
[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite $OUTPUT"; exit 2; }

export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/navsim}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

if [[ "${GRPO_STAGE39_EVAL_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "PASS Stage39 eval preflight role=$ROLE fold=$FOLD step=$STEP noise=$NOISE source=$SOURCE"
  echo "Generator: $GENERATOR"
  echo "Manifest: $MANIFEST"
  echo "Output: $OUTPUT"
  exit 0
fi
mkdir -p "$OUT_DIR"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/evaluate_grpo_schedule.py" \
  --checkpoint "$GENERATOR" --reference-checkpoint "$PUBLIC" \
  --cache-path "$TRAIN_CACHE" --metric-cache-path "$METRIC_CACHE" \
  --tokens-file "$MANIFEST" --limit "$LIMIT" --log-split train \
  --batch-size 2 --num-workers 4 --device cuda:0 \
  --truncation-timestep 32 --roll-timesteps 32 24 16 8 0 \
  --scheduler-num-inference-steps 125 \
  --generation-policy-algorithm diffgrpo_non_destructive_challenger \
  --stage39-candidate-source "$SOURCE" --selector-logits-source current \
  --evaluation-noise-namespace "$NOISE" --generator-domain "$DOMAIN" \
  --store-candidate-trajectories --output "$OUTPUT"

"$PYTHON_BIN" - "$OUTPUT" "$GEN_SHA" "$DOMAIN" "$NOISE" "$LIMIT" "$MANIFEST_SHA" "$SOURCE" <<'PYCHECK'
import json,sys
payload=json.load(open(sys.argv[1])); summary=payload['summary']; records=payload['records']
assert summary['completed'] and summary['num_failures']==0
assert summary['checkpoint_sha256']==sys.argv[2]
assert summary['generator_domain']==sys.argv[3]
assert summary['evaluation_noise_namespace']==int(sys.argv[4])
assert summary['num_tokens']==int(sys.argv[5])
assert summary['generation_policy_algorithm']=='diffgrpo_non_destructive_challenger'
assert summary['stage39_candidate_source']==sys.argv[7]
expected_modes=20 if sys.argv[7]=='public20' else 40
assert all(len(record['candidate_rewards'])==expected_modes for record in records)
assert all(len(record['candidate_trajectories'])==expected_modes for record in records)
print(f"PASS Stage39 eval selected={summary['selected_reward']:.9f} manifest_sha={sys.argv[6]}")
PYCHECK
