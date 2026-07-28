#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 audit|formal public|multi GPU"
  exit 2
fi
PHASE="$1"
BRANCH="$2"
GPU="$3"
[[ "$PHASE" == audit || "$PHASE" == formal ]] || { echo "invalid phase"; exit 2; }
[[ "$BRANCH" == public || "$BRANCH" == multi ]] || { echo "invalid branch"; exit 2; }
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "GPU must be 0..7"; exit 2; }

ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/navsim}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
export NAVSIM_TRAINING_CACHE="${NAVSIM_TRAINING_CACHE:-$NAVSIM_EXP_ROOT/training_cache}"

PUBLIC=/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS
OUTPUT_DIR="$ROOT_DIR/artifacts/grpo_stage27/selectors/$BRANCH/$PHASE"
STAGE24_EXPERIMENT="stage27_selector_${BRANCH}_stage24_${PHASE}_seed$(
  [[ "$BRANCH" == public ]] && echo 27024 || echo 27124
)"
STAGE25_EXPERIMENT="stage27_selector_${BRANCH}_stage25_${PHASE}_seed$(
  [[ "$BRANCH" == public ]] && echo 27025 || echo 27125
)"
for experiment in "$STAGE24_EXPERIMENT" "$STAGE25_EXPERIMENT"; do
  [[ ! -e "$NAVSIM_EXP_ROOT/$experiment" ]] || {
    echo "refusing to mix an existing Stage27 experiment: $NAVSIM_EXP_ROOT/$experiment"
    exit 2
  }
done
[[ ! -e "$OUTPUT_DIR" ]] || {
  echo "refusing to overwrite Stage27 selector freeze directory: $OUTPUT_DIR"
  exit 2
}
mkdir -p "$OUTPUT_DIR"

if [[ "$PHASE" == audit ]]; then
  EXPECTED_EPOCH=0
  EXPECTED_STEP=8
else
  if [[ "$BRANCH" == public ]]; then
    EXPECTED_EPOCH=2
    EXPECTED_STEP=6114
  else
    EXPECTED_EPOCH=17
    EXPECTED_STEP=36684
  fi
fi

cd "$ROOT_DIR"
scripts/training/run_diffusiondrive_grpo_stage27_stage24_selector.sh \
  "$PHASE" "$BRANCH" "$STAGE24_EXPERIMENT" "$GPU"

"$PYTHON_BIN" scripts/evaluation/freeze_grpo_stage27_selector_checkpoint.py \
  --experiment-root "$NAVSIM_EXP_ROOT/$STAGE24_EXPERIMENT" \
  --branch "$BRANCH" --phase "$PHASE" --selector-stage stage24 \
  --expected-epoch "$EXPECTED_EPOCH" --expected-global-step "$EXPECTED_STEP" \
  --output "$OUTPUT_DIR/stage24_pointer.json" >"$OUTPUT_DIR/stage24_path.txt"
STAGE24_CHECKPOINT="$(tail -n 1 "$OUTPUT_DIR/stage24_path.txt")"

"$PYTHON_BIN" scripts/evaluation/audit_grpo_stage27_selector_checkpoint.py \
  --phase stage24 --branch "$BRANCH" --checkpoint "$STAGE24_CHECKPOINT" \
  --public-checkpoint "$PUBLIC" --expected-epoch "$EXPECTED_EPOCH" \
  --expected-global-step "$EXPECTED_STEP" \
  --output "$OUTPUT_DIR/stage24_checkpoint_audit.json"

scripts/training/run_diffusiondrive_grpo_stage27_stage25_selector.sh \
  "$PHASE" "$BRANCH" "$STAGE24_CHECKPOINT" "$STAGE25_EXPERIMENT" "$GPU"

"$PYTHON_BIN" scripts/evaluation/freeze_grpo_stage27_selector_checkpoint.py \
  --experiment-root "$NAVSIM_EXP_ROOT/$STAGE25_EXPERIMENT" \
  --branch "$BRANCH" --phase "$PHASE" --selector-stage stage25 \
  --expected-epoch "$EXPECTED_EPOCH" --expected-global-step "$EXPECTED_STEP" \
  --output "$OUTPUT_DIR/stage25_pointer.json" >"$OUTPUT_DIR/stage25_path.txt"
STAGE25_CHECKPOINT="$(tail -n 1 "$OUTPUT_DIR/stage25_path.txt")"

"$PYTHON_BIN" scripts/evaluation/audit_grpo_stage27_selector_checkpoint.py \
  --phase stage25 --branch "$BRANCH" --checkpoint "$STAGE25_CHECKPOINT" \
  --stage24-parent "$STAGE24_CHECKPOINT" --public-checkpoint "$PUBLIC" \
  --expected-epoch "$EXPECTED_EPOCH" --expected-global-step "$EXPECTED_STEP" \
  --output "$OUTPUT_DIR/stage25_checkpoint_audit.json"

"$PYTHON_BIN" -c '
import hashlib,json,pathlib,sys
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
branch,phase,out,bank_audit=sys.argv[1:]
root=pathlib.Path(out)
s24=json.load(open(root/"stage24_pointer.json"))
s25=json.load(open(root/"stage25_pointer.json"))
a24=json.load(open(root/"stage24_checkpoint_audit.json"))
a25=json.load(open(root/"stage25_checkpoint_audit.json"))
assert a24["passed"] and a25["passed"]
assert s24["checkpoint_sha256"] == a24["checkpoint_sha256"]
assert s25["checkpoint_sha256"] == a25["checkpoint_sha256"]
payload={
  "schema_version":1,"stage":27,"phase":phase,"branch":branch,"passed":True,
  "stage24_checkpoint":s24["checkpoint"],
  "stage24_checkpoint_sha256":s24["checkpoint_sha256"],
  "stage25_checkpoint":s25["checkpoint"],
  "stage25_checkpoint_sha256":s25["checkpoint_sha256"],
  "stage24_audit_sha256":sha(root/"stage24_checkpoint_audit.json"),
  "stage25_audit_sha256":sha(root/"stage25_checkpoint_audit.json"),
  "bank_audit":bank_audit,"bank_audit_sha256":sha(bank_audit),
}
path=root/"frozen_selector.json"
path.write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps(payload,indent=2))
' "$BRANCH" "$PHASE" "$OUTPUT_DIR" \
  "$ROOT_DIR/artifacts/grpo_stage27/audits/selector_banks_${BRANCH}.json"

echo "Stage27 selector branch complete: $OUTPUT_DIR/frozen_selector.json"
