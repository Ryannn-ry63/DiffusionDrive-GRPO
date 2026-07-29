#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 public|rgt|escr FOLD(0|1) STEP(0|192|24|48|96) NOISE GPU"
  exit 2
fi

ROLE="$1"
FOLD="$2"
STEP="$3"
NOISE="$4"
GPU="$5"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
PUBLIC="${STAGE38_PUBLIC_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS}"
SELECTOR="${STAGE38_SELECTOR_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage27_selector_multi_stage25_formal_seed27125/2026.07.24.09.28.29/lightning_logs/version_0/checkpoints/grpo-17-36684.ckpt}"
CALIBRATION="${STAGE38_SELECTOR_CALIBRATION:-$ROOT_DIR/artifacts/grpo_stage27/selectors/multi/calibration.json}"
CV_FREEZE="${STAGE38_CV_FREEZE:-$ROOT_DIR/artifacts/grpo_stage30/manifests/cv/freeze.json}"
TRAIN_FREEZE="$ROOT_DIR/artifacts/grpo_stage38/pilot/training/ESCR/fold${FOLD}/formal/checkpoints.json"
OUT_DIR="$ROOT_DIR/artifacts/grpo_stage38/pilot/eval/fold${FOLD}"
PLAN_SHA="8cd826b6ab8a4376c6e4a4fdffc97e96fe00fba00a25a1029d5fa5bcf4a8b027"
PUBLIC_SHA="008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
RGT_F0="${STAGE38_INITIALIZER_F0:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage36_rgt_pilot_f0_formal/2026.07.28.09.38.46/lightning_logs/version_0/checkpoints/grpo-step-192.ckpt}"
RGT_F1="${STAGE38_INITIALIZER_F1:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage36_rgt_pilot_f1_formal/2026.07.28.09.56.24/lightning_logs/version_0/checkpoints/grpo-step-192.ckpt}"
RGT_SHA_F0="2a2302da415e38aa654c172858d3f89f8112f774a6215d89b9c150112f0dab33"
RGT_SHA_F1="0bafe817ebfe98032b3d4e9499eb78d547efcf7230ddc3c83aa4b1bd4e4e7a9b"

[[ "$ROLE" =~ ^(public|rgt|escr)$ ]] || { echo "invalid Stage38 eval role"; exit 2; }
[[ "$FOLD" =~ ^[01]$ ]] || { echo "invalid Stage38 eval fold"; exit 2; }
[[ "$NOISE" == 20261611 || "$NOISE" == 20261612 ]] || {
  echo "Stage38 requires fresh noise namespace 20261611 or 20261612"
  exit 2
}
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "invalid Stage38 eval GPU"; exit 2; }
for path in "$PYTHON_BIN" "$PUBLIC" "$SELECTOR" "$CALIBRATION" "$CV_FREEZE"; do
  [[ -f "$path" ]] || { echo "missing Stage38 eval input: $path"; exit 2; }
done
[[ "$(sha256sum "$PUBLIC" | awk '{print $1}')" == "$PUBLIC_SHA" ]] || {
  echo "Stage38 public checkpoint SHA drifted"; exit 2;
}

if [[ "$FOLD" == 0 ]]; then
  RGT="$RGT_F0"
  RGT_SHA="$RGT_SHA_F0"
else
  RGT="$RGT_F1"
  RGT_SHA="$RGT_SHA_F1"
fi

case "$ROLE:$STEP" in
  public:0)
    GENERATOR="$PUBLIC"
    GEN_SHA="$PUBLIC_SHA"
    LABEL="P"
    DOMAIN="stage38_pilot_p_fold${FOLD}"
    ;;
  rgt:192)
    GENERATOR="$RGT"
    GEN_SHA="$RGT_SHA"
    LABEL="RGT192"
    DOMAIN="stage38_pilot_rgt192_fold${FOLD}"
    ;;
  escr:24|escr:48|escr:96)
    [[ -f "$TRAIN_FREEZE" ]] || {
      echo "missing Stage38 formal checkpoint freeze: $TRAIN_FREEZE"; exit 2;
    }
    readarray -t ESCR_LOCKED < <(
      "$PYTHON_BIN" - "$TRAIN_FREEZE" "$FOLD" "$STEP" "$PLAN_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

path = Path(sys.argv[1])
fold, step = int(sys.argv[2]), int(sys.argv[3])
plan_sha = sys.argv[4]
freeze = json.loads(path.read_text())
matches = [
    record for record in freeze.get("checkpoints", [])
    if int(record["global_step"]) == step
]
if not (
    freeze.get("passed") is True
    and freeze.get("stage") == 38
    and freeze.get("phase") == "formal"
    and freeze.get("holdout_fold") == fold
    and freeze.get("plan_sha256") == plan_sha
    and len(matches) == 1
):
    raise RuntimeError("Stage38 formal checkpoint freeze drifted")
checkpoint = Path(matches[0]["path"])
if not checkpoint.is_file() or sha(checkpoint) != matches[0]["sha256"]:
    raise RuntimeError("Stage38 formal checkpoint SHA drifted")
print(checkpoint)
print(matches[0]["sha256"])
PY
    )
    [[ "${#ESCR_LOCKED[@]}" -eq 2 ]] || {
      echo "failed to resolve Stage38 ESCR checkpoint"; exit 2;
    }
    GENERATOR="${ESCR_LOCKED[0]}"
    GEN_SHA="${ESCR_LOCKED[1]}"
    LABEL="ESCR${STEP}"
    DOMAIN="stage38_pilot_escr${STEP}_fold${FOLD}"
    ;;
  *)
    echo "role/step mismatch: public=0, rgt=192, escr=24|48|96"
    exit 2
    ;;
esac

[[ -f "$GENERATOR" ]] || { echo "missing Stage38 generator: $GENERATOR"; exit 2; }
[[ "$(sha256sum "$GENERATOR" | awk '{print $1}')" == "$GEN_SHA" ]] || {
  echo "Stage38 generator SHA drifted"; exit 2;
}
readarray -t CV_LOCKED < <(
  "$PYTHON_BIN" - "$CV_FREEZE" "$FOLD" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

freeze = json.loads(Path(sys.argv[1]).read_text())
fold = int(sys.argv[2])
entry = freeze["folds"][fold]
manifest = Path(entry["holdout_manifest"])
if not (
    freeze.get("stage") == 30
    and entry["holdout_fold"] == fold
    and manifest.is_file()
    and sha(manifest) == entry["holdout_manifest_sha256"]
):
    raise RuntimeError("Stage38 held-out manifest drifted")
print(manifest)
print(entry["holdout_count"])
print(entry["holdout_manifest_sha256"])
PY
)
[[ "${#CV_LOCKED[@]}" -eq 3 ]] || { echo "failed Stage38 CV resolution"; exit 2; }
MANIFEST="${CV_LOCKED[0]}"
LIMIT="${CV_LOCKED[1]}"
MANIFEST_SHA="${CV_LOCKED[2]}"
OUTPUT="$OUT_DIR/${LABEL}_ns${NOISE}.json"
[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite $OUTPUT"; exit 2; }

export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/navsim}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

if [[ "${GRPO_STAGE38_EVAL_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "PASS Stage38 eval preflight role=$ROLE fold=$FOLD step=$STEP noise=$NOISE"
  echo "Generator: $GENERATOR"
  echo "Manifest: $MANIFEST"
  echo "Output: $OUTPUT"
  exit 0
fi

mkdir -p "$OUT_DIR"
GRPO_BASE_CHECKPOINT="$PUBLIC" \
GRPO_EVAL_GENERATION_ALGORITHM=diffgrpo_elite_set_counterfactual_repair \
PYTHON_BIN="$PYTHON_BIN" \
  "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage25_selector_eval.sh" \
  deploy "$GENERATOR" "$DOMAIN" "$SELECTOR" "$CALIBRATION" \
  "$MANIFEST" "$LIMIT" "$NOISE" "$OUTPUT" "$GPU"

"$PYTHON_BIN" - "$OUTPUT" "$GEN_SHA" "$DOMAIN" "$NOISE" "$LIMIT" "$MANIFEST_SHA" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
summary = payload["summary"]
assert summary["completed"] and summary["num_failures"] == 0
assert summary["checkpoint_sha256"] == sys.argv[2]
assert summary["generator_domain"] == sys.argv[3]
assert summary["evaluation_noise_namespace"] == int(sys.argv[4])
assert summary["num_tokens"] == int(sys.argv[5])
assert summary["generation_policy_algorithm"] == (
    "diffgrpo_elite_set_counterfactual_repair"
)
print(
    f"PASS Stage38 eval selected={summary['selected_reward']:.9f} "
    f"manifest_sha={sys.argv[6]}"
)
PY
