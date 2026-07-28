#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 JOB_ID GPU"
  echo "JOB_ID 0-3 evaluates B; JOB_ID 4-7 evaluates C"
  exit 2
fi

JOB_ID="$1"
GPU="$2"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
BASE=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model
FINAL=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage26_generator_final_folds0_4_seed0/2026.07.23.15.32.57/lightning_logs/version_0/checkpoints/grpo-01-160.ckpt
SELECTOR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage26_stage25_selector_formal_seed26025/2026.07.23.13.16.13/lightning_logs/version_0/checkpoints/grpo-14-30570.ckpt
CALIBRATION="$ROOT_DIR/artifacts/grpo_stage26/stage26_selector_calibration_fold4.json"
FOLD4_GATE="$ROOT_DIR/artifacts/grpo_stage26/fold4/generator_gate.json"
MANIFEST="$ROOT_DIR/artifacts/grpo_stage21/manifests/fold5_manifest.json"
OUT_DIR="$ROOT_DIR/artifacts/grpo_stage26/fold5"

[[ "$JOB_ID" =~ ^[0-7]$ ]] || { echo "JOB_ID must be 0-7"; exit 2; }
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "GPU must be 0-7"; exit 2; }
for path in "$BASE" "$FINAL" "$SELECTOR" "$CALIBRATION" "$FOLD4_GATE" "$MANIFEST"; do
  [[ -f "$path" ]] || { echo "missing locked Stage26 input: $path"; exit 2; }
done

"$PYTHON_BIN" -c '
import hashlib,json,sys
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
base,final,selector,calibration,gate,manifest=sys.argv[1:]
assert sha(base) == "59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d"
assert sha(final) == "c0baa2c42023727195f7f28e83e1e9532783e572ddbc8fc60d9ed7a25409ec2b"
assert sha(selector) == "77ee768c5f469292d81049192d48797f2f3fae4f80029ae6853dd9d9c6de4207"
assert sha(calibration) == "0c14b8f9531d64028bdf4d71f35b34418ab31ad4ae11ccc6b3c60a4325f53a0d"
assert sha(gate) == "3436534cdca9e8aefd2b4410b25a1198be3c267bcd2dc7dd7c4efdd6fae6a5a2"
assert sha(manifest) == "2424fe88319f81650eec1b8e2dbe265143d5d6d7cc8d113e104caf8d150cf2d4"
g=json.load(open(gate)); assert g["passed"] and not g["stop_before_fold5"]
m=json.load(open(manifest)); s=m["summary"]
assert s["name"] == "fold5" and s["count"] == 1023 and s["num_logs"] == 151
' "$BASE" "$FINAL" "$SELECTOR" "$CALIBRATION" "$FOLD4_GATE" "$MANIFEST"

NOISES=(-1 20260823 20260824 20260825)
NAMES=(default ns20260823 ns20260824 ns20260825)
SYSTEM_INDEX=$((JOB_ID / 4))
NS_INDEX=$((JOB_ID % 4))
if [[ "$SYSTEM_INDEX" == 0 ]]; then
  SYSTEM=B
  GENERATOR="$BASE"
  DOMAIN=official_base
else
  SYSTEM=C
  GENERATOR="$FINAL"
  DOMAIN=stage26_final_epoch2
fi
NOISE="${NOISES[$NS_INDEX]}"
NAME="${NAMES[$NS_INDEX]}"
OUTPUT="$OUT_DIR/${SYSTEM}_${NAME}.json"
[[ ! -e "$OUTPUT" ]] || {
  echo "protected fold5 artifact already exists; refusing to overwrite: $OUTPUT"
  exit 2
}

"$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage25_selector_eval.sh" \
  deploy "$GENERATOR" "$DOMAIN" "$SELECTOR" "$CALIBRATION" \
  "$MANIFEST" 1023 "$NOISE" "$OUTPUT" "$GPU"
