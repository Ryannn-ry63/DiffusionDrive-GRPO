#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 candidate|selector JOB_ID GPU"
  echo "candidate JOB_ID: 0=default, 1=20260811, 2=20260812"
  echo "selector  JOB_ID: 0=20260821, 1=20260822"
  exit 2
fi

MODE="$1"
JOB_ID="$2"
GPU="$3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
PUBLIC=/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS
SELECTOR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage26_stage25_selector_formal_seed26025/2026.07.23.13.16.13/lightning_logs/version_0/checkpoints/grpo-14-30570.ckpt
CALIBRATION="$ROOT_DIR/artifacts/grpo_stage26/stage26_selector_calibration_fold4.json"
FIT_MANIFEST="$ROOT_DIR/artifacts/grpo_stage23/manifests/selector_fit_manifest.json"
DEV_MANIFEST="$ROOT_DIR/artifacts/grpo_stage21/manifests/fold4_manifest.json"
OUTPUT_ROOT="$ROOT_DIR/artifacts/grpo_stage27"

[[ "$MODE" == candidate || "$MODE" == selector ]] || {
  echo "MODE must be candidate or selector"
  exit 2
}
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "GPU must be 0..7"; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "missing Python: $PYTHON_BIN"; exit 2; }
for path in "$PUBLIC" "$SELECTOR" "$CALIBRATION" "$FIT_MANIFEST" \
  "$DEV_MANIFEST"; do
  [[ -f "$path" ]] || { echo "missing locked Stage27 input: $path"; exit 2; }
done

"$PYTHON_BIN" -c '
import hashlib,sys
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()
expected=(
    "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b",
    "77ee768c5f469292d81049192d48797f2f3fae4f80029ae6853dd9d9c6de4207",
    "0c14b8f9531d64028bdf4d71f35b34418ab31ad4ae11ccc6b3c60a4325f53a0d",
    "cefe6cdc470f9e14a5244ba422d7ff7265eed7769d4147189d705c1df1604f7f",
    "8e0f55b18e1faf3e3b12788390abf041ab3049b7e72ff55b0968ebcfd71f4a23",
)
actual=tuple(sha(p) for p in sys.argv[1:])
assert actual == expected, (actual,expected)
' "$PUBLIC" "$SELECTOR" "$CALIBRATION" "$FIT_MANIFEST" "$DEV_MANIFEST"

DOMAIN=public88_base
if [[ "$MODE" == candidate ]]; then
  [[ "$JOB_ID" =~ ^[0-2]$ ]] || {
    echo "candidate JOB_ID must be 0, 1, or 2"
    exit 2
  }
  NAMESPACES=(-1 20260811 20260812)
  LABELS=(default ns20260811 ns20260812)
  NAMESPACE="${NAMESPACES[$JOB_ID]}"
  LABEL="${LABELS[$JOB_ID]}"
  OUTPUT="$OUTPUT_ROOT/candidate_bank/${DOMAIN}_${LABEL}.json"
  [[ ! -e "$OUTPUT" ]] || {
    echo "refusing to overwrite Stage27 artifact: $OUTPUT"
    exit 2
  }
  mkdir -p "$(dirname "$OUTPUT")"
  GRPO_BASE_CHECKPOINT="$PUBLIC" PYTHON_BIN="$PYTHON_BIN" \
    "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage24_candidate_bank.sh" \
    "$FIT_MANIFEST" 4075 "$DOMAIN" "$PUBLIC" "$NAMESPACE" "$OUTPUT" "$GPU"
else
  [[ "$JOB_ID" =~ ^[0-1]$ ]] || {
    echo "selector JOB_ID must be 0 or 1"
    exit 2
  }
  NAMESPACES=(20260821 20260822)
  NAMESPACE="${NAMESPACES[$JOB_ID]}"
  OUTPUT="$OUTPUT_ROOT/fold4/old_selector_${DOMAIN}_ns${NAMESPACE}.json"
  [[ ! -e "$OUTPUT" ]] || {
    echo "refusing to overwrite Stage27 artifact: $OUTPUT"
    exit 2
  }
  mkdir -p "$(dirname "$OUTPUT")"
  GRPO_BASE_CHECKPOINT="$PUBLIC" PYTHON_BIN="$PYTHON_BIN" \
    "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage25_selector_eval.sh" \
    deploy "$PUBLIC" "$DOMAIN" "$SELECTOR" "$CALIBRATION" \
    "$DEV_MANIFEST" 1021 "$NAMESPACE" "$OUTPUT" "$GPU"
fi

"$PYTHON_BIN" -c '
import json,sys
p=json.load(open(sys.argv[1]))
s=p["summary"]
public="008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
assert s["completed"] and s["num_failures"] == 0
assert s["checkpoint_sha256"] == public
assert s["reference_checkpoint_sha256"] == public
assert s["generator_domain"] == "public88_base"
assert s["schedule"]["roll_timesteps"] == [32,24,16,8,0]
assert s["evaluation_noise_namespace"] == int(sys.argv[2])
assert s["num_tokens"] == int(sys.argv[3])
if sys.argv[4] == "candidate":
    assert s["selector_logits_source"] == "current"
    assert s["stores_candidate_trajectories"]
    assert all(len(r["candidate_trajectories"]) == 20 for r in p["records"])
else:
    assert s["selector_logits_source"] == "trajectory_relative_harm_v3"
    assert not s["stores_candidate_trajectories"]
    assert s["stage25_selector"]["checkpoint_sha256"] == (
        "77ee768c5f469292d81049192d48797f2f3fae4f80029ae6853dd9d9c6de4207"
    )
print(json.dumps(s,indent=2))
' "$OUTPUT" "$NAMESPACE" "$([[ "$MODE" == candidate ]] && echo 4075 || echo 1021)" "$MODE"

echo "Stage27 $MODE job complete: $OUTPUT"
