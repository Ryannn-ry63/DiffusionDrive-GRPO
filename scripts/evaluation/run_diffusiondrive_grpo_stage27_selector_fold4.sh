#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 collect|deploy public|multi JOB_ID GPU"
  exit 2
fi
MODE="$1"
BRANCH="$2"
JOB_ID="$3"
GPU="$4"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
PUBLIC=/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS
FREEZE="$ROOT_DIR/artifacts/grpo_stage27/selectors/$BRANCH/formal/frozen_selector.json"
MANIFEST="$ROOT_DIR/artifacts/grpo_stage21/manifests/fold4_manifest.json"
NOISES=(20260821 20260822)

[[ "$MODE" == collect || "$MODE" == deploy ]] || { echo "invalid mode"; exit 2; }
[[ "$BRANCH" == public || "$BRANCH" == multi ]] || { echo "invalid branch"; exit 2; }
[[ "$JOB_ID" =~ ^[0-1]$ ]] || { echo "JOB_ID must be 0 or 1"; exit 2; }
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "GPU must be 0..7"; exit 2; }
for path in "$PUBLIC" "$FREEZE" "$MANIFEST"; do
  [[ -f "$path" ]] || { echo "missing Stage27 selector input: $path"; exit 2; }
done

read -r SELECTOR SELECTOR_SHA < <("$PYTHON_BIN" -c '
import hashlib,json,pathlib,sys
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
p=json.load(open(sys.argv[1])); branch=sys.argv[2]
assert p["passed"] and p["stage"] == 27 and p["phase"] == "formal"
assert p["branch"] == branch
path=pathlib.Path(p["stage25_checkpoint"])
assert path.is_file() and sha(path) == p["stage25_checkpoint_sha256"]
print(path,p["stage25_checkpoint_sha256"])
' "$FREEZE" "$BRANCH")

NOISE="${NOISES[$JOB_ID]}"
if [[ "$MODE" == collect ]]; then
  CALIBRATION=-
  OUTPUT="$ROOT_DIR/artifacts/grpo_stage27/selectors/$BRANCH/fold4/collect_ns${NOISE}.json"
else
  CALIBRATION="$ROOT_DIR/artifacts/grpo_stage27/selectors/$BRANCH/calibration.json"
  [[ -f "$CALIBRATION" ]] || { echo "missing Stage27 calibration: $CALIBRATION"; exit 2; }
  OUTPUT="$ROOT_DIR/artifacts/grpo_stage27/selectors/$BRANCH/fold4/deploy_ns${NOISE}.json"
fi
[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite: $OUTPUT"; exit 2; }
mkdir -p "$(dirname "$OUTPUT")"

GRPO_BASE_CHECKPOINT="$PUBLIC" PYTHON_BIN="$PYTHON_BIN" \
  "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage25_selector_eval.sh" \
  "$MODE" "$PUBLIC" public88_base "$SELECTOR" "$CALIBRATION" \
  "$MANIFEST" 1021 "$NOISE" "$OUTPUT" "$GPU"

"$PYTHON_BIN" -c '
import json,sys
p=json.load(open(sys.argv[1])); s=p["summary"]; selector=s["stage25_selector"]
assert s["completed"] and s["num_failures"] == 0 and s["num_tokens"] == 1021
assert s["checkpoint_sha256"] == s["reference_checkpoint_sha256"] == (
 "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b")
assert s["generator_domain"] == "public88_base"
assert s["evaluation_noise_namespace"] == int(sys.argv[2])
assert selector["checkpoint_sha256"] == sys.argv[3]
assert selector["calibration_collection"] == (sys.argv[4] == "collect")
' "$OUTPUT" "$NOISE" "$SELECTOR_SHA" "$MODE"
echo "Stage27 selector fold4 complete: $OUTPUT"
