#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 JOB_ID(0..2) GPU(0..7)"
  exit 2
fi
JOB_ID="$1"
GPU="$2"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
PUBLIC=/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS
SELECTOR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage27_selector_multi_stage25_formal_seed27125/2026.07.24.09.28.29/lightning_logs/version_0/checkpoints/grpo-17-36684.ckpt
CALIBRATION="$ROOT_DIR/artifacts/grpo_stage27/selectors/multi/calibration.json"
MANIFEST="$ROOT_DIR/artifacts/grpo_stage26/manifests/generator_train_full_6119.json"
PLAN="$ROOT_DIR/GRPO_STAGE30_MODE_COVERAGE_CONSTRAINED_PLAN_20260725.md"
OUT_DIR="$ROOT_DIR/artifacts/grpo_stage30/labels"
NAMESPACES=(20261301 20261302 20261303)

[[ "$JOB_ID" =~ ^[0-2]$ ]] || { echo "JOB_ID must be 0..2"; exit 2; }
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "GPU must be 0..7"; exit 2; }
for path in "$PUBLIC" "$SELECTOR" "$CALIBRATION" "$MANIFEST" "$PLAN"; do
  [[ -f "$path" ]] || { echo "missing locked Stage30 input: $path"; exit 2; }
done
"$PYTHON_BIN" - "$PUBLIC" "$SELECTOR" "$CALIBRATION" "$MANIFEST" "$PLAN" <<'PY'
import hashlib, sys
from pathlib import Path
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
expected=(
 '008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b',
 '023d7b6b77bb8fa2dc3e779849f3716688abf0796b019416494c80b29615b691',
 'fec2a10573e913e4452f8eea92e6334fcd5e6f1104e6244876770973a1f41990',
 '1763ca12bfd1bf480a1ed6a47cc71f95ef31123517e500551dae24292620ef68',
 'e9867ec48d4806ff97adb0284bae7550305cb1350e4ec66dbfd20d245de435ce',
)
actual=tuple(sha(path) for path in sys.argv[1:])
if actual != expected: raise RuntimeError((actual, expected))
PY

NAMESPACE="${NAMESPACES[$JOB_ID]}"
OUTPUT="$OUT_DIR/public_smulti_ns${NAMESPACE}.json"
[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite $OUTPUT"; exit 2; }
mkdir -p "$OUT_DIR"
if [[ "${GRPO_STAGE30_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "PASS Stage30 label preflight namespace=$NAMESPACE"
  exit 0
fi
GRPO_BASE_CHECKPOINT="$PUBLIC" PYTHON_BIN="$PYTHON_BIN" \
  "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage25_selector_eval.sh" \
  deploy "$PUBLIC" public88_base "$SELECTOR" "$CALIBRATION" \
  "$MANIFEST" 6119 "$NAMESPACE" "$OUTPUT" "$GPU"

"$PYTHON_BIN" - "$OUTPUT" "$NAMESPACE" <<'PY'
import json, sys
p=json.load(open(sys.argv[1])); s=p['summary']; selector=s['stage25_selector']
checks={
 'complete':s['completed'] and s['num_failures']==0,
 'count':s['num_tokens']==6119 and len(p['records'])==6119,
 'public':s['checkpoint_sha256']==s['reference_checkpoint_sha256']==
   '008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b',
 'namespace':s['evaluation_noise_namespace']==int(sys.argv[2]),
 'selector':selector['checkpoint_sha256']==
   '023d7b6b77bb8fa2dc3e779849f3716688abf0796b019416494c80b29615b691',
 'calibration':selector['calibration_sha256']==
   'fec2a10573e913e4452f8eea92e6334fcd5e6f1104e6244876770973a1f41990',
 'source':s['selector_logits_source']=='trajectory_relative_harm_v3',
 'schedule':s['schedule']['roll_timesteps']==[32,24,16,8,0],
}
bad=[name for name, passed in checks.items() if not passed]
if bad: raise RuntimeError(f'Stage30 label artifact failed {bad}')
PY
echo "PASS Stage30 public label namespace=$NAMESPACE: $OUTPUT"

