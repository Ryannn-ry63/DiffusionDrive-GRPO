#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 FOLD(0|1) STEP(P|48|96|144|192) NOISE(20261411|20261412) GPU(0..7)"
  exit 2
fi
FOLD="$1"; STEP="$2"; NOISE="$3"; GPU="$4"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
PUBLIC="${STAGE33_PUBLIC_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS}"
SELECTOR="${STAGE33_SELECTOR_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage27_selector_multi_stage25_formal_seed27125/2026.07.24.09.28.29/lightning_logs/version_0/checkpoints/grpo-17-36684.ckpt}"
CALIBRATION="${STAGE33_SELECTOR_CALIBRATION:-$ROOT_DIR/artifacts/grpo_stage27/selectors/multi/calibration.json}"
CV_FREEZE="${STAGE33_CV_FREEZE:-$ROOT_DIR/artifacts/grpo_stage30/manifests/cv/freeze.json}"
FORMAL_ROOT="${STAGE33_FORMAL_ROOT:-$EXP_ROOT/stage33_cdc_chain_f${FOLD}_fix1_formal}"
OUT_DIR="$ROOT_DIR/artifacts/grpo_stage33/eval/fold${FOLD}"

[[ "$FOLD" =~ ^[01]$ ]] || { echo "fold must be 0 or 1"; exit 2; }
[[ "$STEP" == P || "$STEP" =~ ^(48|96|144|192)$ ]] || { echo "bad checkpoint step"; exit 2; }
[[ "$NOISE" == 20261411 || "$NOISE" == 20261412 ]] || { echo "bad evaluation noise"; exit 2; }
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "GPU must be 0..7"; exit 2; }
for path in "$PUBLIC" "$SELECTOR" "$CALIBRATION" "$CV_FREEZE"; do
  [[ -f "$path" ]] || { echo "missing Stage33 eval input: $path"; exit 2; }
done

readarray -t LOCKED < <("$PYTHON_BIN" - "$CV_FREEZE" "$FOLD" "$STEP" "$FORMAL_ROOT" <<'PY'
import hashlib, json, sys
from pathlib import Path
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
cv = json.loads(Path(sys.argv[1]).read_text())
fold, step, root = int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
entry = cv["folds"][fold]
manifest = Path(entry["holdout_manifest"])
if sha(manifest) != entry["holdout_manifest_sha256"]:
    raise RuntimeError("Stage33 holdout manifest drifted")
if step == "P":
    checkpoint = Path("/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS")
else:
    matches = sorted(root.rglob(f"grpo-step-{step}.ckpt"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one Stage33 checkpoint for step={step}, got {len(matches)}")
    checkpoint = matches[0]
print(checkpoint)
print(manifest)
print(entry["holdout_count"])
print(entry["holdout_manifest_sha256"])
print(sha(checkpoint))
PY
)
[[ "${#LOCKED[@]}" -eq 5 ]] || { echo "failed Stage33 evaluation input resolution"; exit 2; }
GENERATOR="${LOCKED[0]}"; MANIFEST="${LOCKED[1]}"; LIMIT="${LOCKED[2]}"
MANIFEST_SHA="${LOCKED[3]}"; GENERATOR_SHA="${LOCKED[4]}"
DOMAIN="stage33_cdc_${STEP,,}_fold${FOLD}"
OUTPUT="$OUT_DIR/${STEP}_ns${NOISE}.json"
[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite $OUTPUT"; exit 2; }
mkdir -p "$OUT_DIR"

if [[ "${GRPO_STAGE33_EVAL_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  GRPO_BASE_CHECKPOINT="$PUBLIC" PYTHON_BIN="$PYTHON_BIN" \
    GRPO_SELECTOR_EVAL_PREFLIGHT_ONLY=1 \
    "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage25_selector_eval.sh" \
    deploy "$GENERATOR" "$DOMAIN" "$SELECTOR" "$CALIBRATION" \
    "$MANIFEST" "$LIMIT" "$NOISE" "$OUTPUT" "$GPU"
  echo "PASS Stage33 eval preflight fold=$FOLD step=$STEP noise=$NOISE"
  exit 0
fi
GRPO_BASE_CHECKPOINT="$PUBLIC" PYTHON_BIN="$PYTHON_BIN" \
  "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage25_selector_eval.sh" \
  deploy "$GENERATOR" "$DOMAIN" "$SELECTOR" "$CALIBRATION" \
  "$MANIFEST" "$LIMIT" "$NOISE" "$OUTPUT" "$GPU"

"$PYTHON_BIN" - "$OUTPUT" "$GENERATOR_SHA" "$NOISE" "$LIMIT" "$MANIFEST_SHA" "$DOMAIN" <<'PY'
import hashlib, json, sys
payload = json.loads(open(sys.argv[1]).read())
summary = payload["summary"]
records = payload["records"]
selector = summary["stage25_selector"]
token_hash = hashlib.sha256("\n".join(r["token"] for r in records).encode()).hexdigest()
checks = {
    "complete": summary["completed"] and summary["num_failures"] == 0,
    "count": summary["num_tokens"] == int(sys.argv[4]) == len(records),
    "checkpoint": summary["checkpoint_sha256"] == sys.argv[2],
    "reference": summary["reference_checkpoint_sha256"] == "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b",
    "noise": summary["evaluation_noise_namespace"] == int(sys.argv[3]),
    "domain": summary["generator_domain"] == sys.argv[6],
    "selector": selector["checkpoint_sha256"] == "023d7b6b77bb8fa2dc3e779849f3716688abf0796b019416494c80b29615b691",
    "calibration": selector["calibration_sha256"] == "fec2a10573e913e4452f8eea92e6334fcd5e6f1104e6244876770973a1f41990",
    "source": summary["selector_logits_source"] == "trajectory_relative_harm_v3",
    "schedule": summary["schedule"]["roll_timesteps"] == [32, 24, 16, 8, 0],
    "tokens": summary["token_set_sha256"] == token_hash,
}
bad = [key for key, value in checks.items() if not value]
if bad:
    raise RuntimeError(f"Stage33 PDM evaluation artifact failed: {bad}")
print(
    f"PASS Stage33 eval fold={sys.argv[6].split('_fold')[1]} "
    f"noise={sys.argv[3]} selected={summary['selected_reward']:.9f} "
    f"oracle={summary['oracle_reward']:.9f} regret={summary['selection_regret']:.9f}"
)
PY
