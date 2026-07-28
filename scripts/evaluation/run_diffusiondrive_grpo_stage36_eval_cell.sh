#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 FOLD(0|1) SYSTEM(P|DPEL192|RGT48|RGT96|RGT144|RGT192) NOISE GPU"
  exit 2
fi

FOLD="$1"
SYSTEM="$2"
NOISE="$3"
GPU="$4"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
PUBLIC="${STAGE36_PUBLIC_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS}"
SELECTOR="${STAGE36_SELECTOR_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage27_selector_multi_stage25_formal_seed27125/2026.07.24.09.28.29/lightning_logs/version_0/checkpoints/grpo-17-36684.ckpt}"
CALIBRATION="${STAGE36_SELECTOR_CALIBRATION:-$ROOT_DIR/artifacts/grpo_stage27/selectors/multi/calibration.json}"
CV_FREEZE="${STAGE36_CV_FREEZE:-$ROOT_DIR/artifacts/grpo_stage30/manifests/cv/freeze.json}"
DPEL_FREEZE="$ROOT_DIR/artifacts/grpo_stage32/pilot/training/DPEL/fold${FOLD}/formal/checkpoints.json"
STAGE36_FREEZE="$ROOT_DIR/artifacts/grpo_stage36/pilot/training/RGT/fold${FOLD}/formal/checkpoints.json"
OUT_DIR="$ROOT_DIR/artifacts/grpo_stage36/pilot/eval/fold${FOLD}"

[[ "$FOLD" =~ ^[01]$ ]] || { echo "fold must be 0 or 1"; exit 2; }
[[ "$SYSTEM" =~ ^(P|DPEL192|RGT48|RGT96|RGT144|RGT192)$ ]] || {
  echo "invalid Stage36 evaluation system: $SYSTEM"; exit 2;
}
[[ "$NOISE" == 20261511 || "$NOISE" == 20261512 ]] || {
  echo "Stage36 noise must be 20261511 or 20261512"; exit 2;
}
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "GPU must be 0..7"; exit 2; }
for path in "$PYTHON_BIN" "$PUBLIC" "$SELECTOR" "$CALIBRATION" "$CV_FREEZE"; do
  [[ -f "$path" ]] || { echo "missing Stage36 eval input: $path"; exit 2; }
done
if [[ "$SYSTEM" == DPEL192 ]]; then
  [[ -f "$DPEL_FREEZE" ]] || { echo "missing DPEL freeze: $DPEL_FREEZE"; exit 2; }
elif [[ "$SYSTEM" == RGT* ]]; then
  [[ -f "$STAGE36_FREEZE" ]] || {
    echo "missing Stage36 formal freeze: $STAGE36_FREEZE"; exit 2;
  }
fi

readarray -t LOCKED < <(
  "$PYTHON_BIN" - \
    "$CV_FREEZE" "$FOLD" "$SYSTEM" "$PUBLIC" "$DPEL_FREEZE" "$STAGE36_FREEZE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

cv_path, fold, system = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
public, dpel_path, stage36_path = map(Path, sys.argv[4:7])
cv = json.loads(cv_path.read_text())
entry = cv["folds"][fold]
manifest = Path(entry["holdout_manifest"])
if (
    cv.get("stage") != 30
    or entry["holdout_fold"] != fold
    or not manifest.is_file()
    or sha(manifest) != entry["holdout_manifest_sha256"]
):
    raise RuntimeError("Stage36 holdout manifest drifted")
if system == "P":
    checkpoint = public
elif system == "DPEL192":
    freeze = json.loads(dpel_path.read_text())
    matches = [
        item for item in freeze["checkpoints"]
        if int(item["global_step"]) == 192
    ]
    if len(matches) != 1 or sha(Path(matches[0]["path"])) != matches[0]["sha256"]:
        raise RuntimeError("Stage36 DPEL192 positive-control freeze drifted")
    checkpoint = Path(matches[0]["path"])
else:
    step = int(system.removeprefix("RGT"))
    freeze = json.loads(stage36_path.read_text())
    matches = [
        item for item in freeze["checkpoints"]
        if int(item["global_step"]) == step
    ]
    if (
        not freeze.get("passed")
        or freeze.get("stage") != 36
        or freeze.get("objective_revision")
            != "reference_gated_tail_ncd_v1"
        or freeze.get("plan_sha256")
            != "858e6688faa9dff6d9027254e6a0262c67e47582d51a5829cc12a9723a25e521"
        or len(matches) != 1
        or sha(Path(matches[0]["path"])) != matches[0]["sha256"]
    ):
        raise RuntimeError(f"Stage36 RGT{step} checkpoint freeze drifted")
    checkpoint = Path(matches[0]["path"])
print(checkpoint)
print(sha(checkpoint))
print(manifest)
print(entry["holdout_count"])
print(entry["holdout_manifest_sha256"])
PY
)
[[ "${#LOCKED[@]}" -eq 5 ]] || {
  echo "failed Stage36 evaluation input resolution"; exit 2;
}
GENERATOR="${LOCKED[0]}"
GENERATOR_SHA="${LOCKED[1]}"
MANIFEST="${LOCKED[2]}"
LIMIT="${LOCKED[3]}"
MANIFEST_SHA="${LOCKED[4]}"
DOMAIN="stage36_pilot_${SYSTEM,,}_fold${FOLD}"
OUTPUT="$OUT_DIR/${SYSTEM}_ns${NOISE}.json"
[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite $OUTPUT"; exit 2; }
mkdir -p "$OUT_DIR"

export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/navsim}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0

if [[ "${GRPO_STAGE36_EVAL_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  GRPO_BASE_CHECKPOINT="$PUBLIC" PYTHON_BIN="$PYTHON_BIN" \
    GRPO_SELECTOR_EVAL_PREFLIGHT_ONLY=1 \
    "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage25_selector_eval.sh" \
    deploy "$GENERATOR" "$DOMAIN" "$SELECTOR" "$CALIBRATION" \
    "$MANIFEST" "$LIMIT" "$NOISE" "$OUTPUT" "$GPU"
  echo "PASS Stage36 eval preflight fold=$FOLD system=$SYSTEM noise=$NOISE"
  exit 0
fi

GRPO_BASE_CHECKPOINT="$PUBLIC" PYTHON_BIN="$PYTHON_BIN" \
  "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage25_selector_eval.sh" \
  deploy "$GENERATOR" "$DOMAIN" "$SELECTOR" "$CALIBRATION" \
  "$MANIFEST" "$LIMIT" "$NOISE" "$OUTPUT" "$GPU"

"$PYTHON_BIN" - \
  "$OUTPUT" "$GENERATOR_SHA" "$NOISE" "$LIMIT" "$MANIFEST_SHA" "$DOMAIN" <<'PY'
import hashlib
import json
import sys

path, checkpoint_sha, noise, count, manifest_sha, domain = sys.argv[1:]
payload = json.loads(open(path).read())
summary, records = payload["summary"], payload["records"]
selector = summary["stage25_selector"]
token_hash = hashlib.sha256(
    "\n".join(record["token"] for record in records).encode()
).hexdigest()
checks = {
    "complete": summary["completed"] and summary["num_failures"] == 0,
    "count": summary["num_tokens"] == int(count) == len(records),
    "checkpoint": summary["checkpoint_sha256"] == checkpoint_sha,
    "reference": summary["reference_checkpoint_sha256"]
        == "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b",
    "noise": summary["evaluation_noise_namespace"] == int(noise),
    "domain": summary["generator_domain"] == domain,
    "selector": selector["checkpoint_sha256"]
        == "023d7b6b77bb8fa2dc3e779849f3716688abf0796b019416494c80b29615b691",
    "calibration": selector["calibration_sha256"]
        == "fec2a10573e913e4452f8eea92e6334fcd5e6f1104e6244876770973a1f41990",
    "source": summary["selector_logits_source"] == "trajectory_relative_harm_v3",
    "schedule": summary["schedule"]["roll_timesteps"] == [32, 24, 16, 8, 0],
    "token_order": summary["token_set_sha256"] == token_hash,
    "manifest": manifest_sha != "",
    "candidate_rewards": all(len(record["candidate_rewards"]) == 20 for record in records),
    "candidate_components": all(len(record["candidate_components"]) == 20 for record in records),
    "eligibility": all(len(record["stage24_selector"]["eligible"]) == 20 for record in records),
}
bad = [name for name, passed in checks.items() if not passed]
if bad:
    raise RuntimeError(f"Stage36 PDM artifact failed provenance checks: {bad}")
print(
    f"PASS Stage36 eval domain={domain} noise={noise} "
    f"selected={summary['selected_reward']:.9f} "
    f"oracle={summary['oracle_reward']:.9f}"
)
PY
