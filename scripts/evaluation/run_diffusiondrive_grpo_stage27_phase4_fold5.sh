#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 JOB_ID GPU"
  echo "JOB_ID 0-1=B, 2-3=C1, 4-5=C2"
  exit 2
fi

JOB_ID="$1"
GPU="$2"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
INPUTS="$ROOT_DIR/artifacts/grpo_stage27/generator/phase4/inputs.json"
OUT_DIR="$ROOT_DIR/artifacts/grpo_stage27/generator/phase4/fold5"
PUBLIC=/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS
INPUTS_SHA=f22cb655e8acfbc41a752acb99772ecabdcdd7ad9788953f4e14e6063067fe4b

[[ "$JOB_ID" =~ ^[0-5]$ ]] || {
  echo "Stage27 Phase4 JOB_ID must be 0..5"
  exit 2
}
[[ "$GPU" =~ ^[0-7]$ ]] || {
  echo "Stage27 Phase4 GPU must be 0..7"
  exit 2
}
for path in "$INPUTS" "$PUBLIC"; do
  [[ -f "$path" ]] || {
    echo "missing Stage27 Phase4 input: $path"
    exit 2
  }
done

SYSTEM_INDEX=$((JOB_ID / 2))
NOISE_INDEX=$((JOB_ID % 2))
SYSTEMS=(B C1 C2)
NOISES=(20261011 20261012)
SYSTEM="${SYSTEMS[$SYSTEM_INDEX]}"
NOISE="${NOISES[$NOISE_INDEX]}"

readarray -t LOCKED < <(
  "$PYTHON_BIN" - "$INPUTS" "$INPUTS_SHA" "$SYSTEM" "$NOISE" <<'PY'
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


path = Path(sys.argv[1])
expected_inputs_sha = sys.argv[2]
system = sys.argv[3]
noise = int(sys.argv[4])
if sha(path) != expected_inputs_sha:
    raise RuntimeError("Stage27 Phase4 input-freeze SHA drifted")
p = json.loads(path.read_text(encoding="utf-8"))
if (
    not p.get("passed")
    or p.get("stage") != 27
    or p.get("phase") != 4
    or noise not in p["noise_namespaces"]
    or system not in p["systems"]
):
    raise RuntimeError("Stage27 Phase4 frozen grid drifted")
entry = p["systems"][system]
generator = Path(entry["path"])
if not generator.is_file() or sha(generator) != entry["sha256"]:
    raise RuntimeError("Stage27 Phase4 generator SHA drifted")
locked = {
    "manifest": "2424fe88319f81650eec1b8e2dbe265143d5d6d7cc8d113e104caf8d150cf2d4",
    "selector_checkpoint": "023d7b6b77bb8fa2dc3e779849f3716688abf0796b019416494c80b29615b691",
    "calibration": "fec2a10573e913e4452f8eea92e6334fcd5e6f1104e6244876770973a1f41990",
    "selector_gate": "1b442569dddafb82ea0ad0d07b34c6ded1f93151851c556a346c0dd8b98ca636",
}
for field, expected in locked.items():
    locked_path = Path(p[field])
    if not locked_path.is_file() or sha(locked_path) != expected:
        raise RuntimeError(f"Stage27 Phase4 locked input drifted: {field}")
domain = {
    "B": "public88_base",
    "C1": "stage27_epoch1",
    "C2": "stage27_epoch2",
}[system]
print(generator)
print(entry["sha256"])
print(domain)
print(p["selector_checkpoint"])
print(p["selector_checkpoint_sha256"])
print(p["calibration"])
print(p["manifest"])
PY
)
[[ "${#LOCKED[@]}" -eq 7 ]] || {
  echo "failed to resolve Stage27 Phase4 locked cell"
  exit 2
}
GENERATOR="${LOCKED[0]}"
GENERATOR_SHA="${LOCKED[1]}"
DOMAIN="${LOCKED[2]}"
SELECTOR="${LOCKED[3]}"
SELECTOR_SHA="${LOCKED[4]}"
CALIBRATION="${LOCKED[5]}"
MANIFEST="${LOCKED[6]}"
OUTPUT="$OUT_DIR/${SYSTEM}_ns${NOISE}.json"
[[ ! -e "$OUTPUT" ]] || {
  echo "refusing to overwrite protected Stage27 Phase4 artifact: $OUTPUT"
  exit 2
}

if [[ "${GRPO_STAGE27_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "PASS Stage27 Phase4 preflight $SYSTEM namespace=$NOISE"
  echo "Generator: $GENERATOR"
  echo "Selector: $SELECTOR"
  echo "Manifest: $MANIFEST"
  exit 0
fi

GRPO_BASE_CHECKPOINT="$PUBLIC" PYTHON_BIN="$PYTHON_BIN" \
  "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage25_selector_eval.sh" \
  deploy "$GENERATOR" "$DOMAIN" "$SELECTOR" "$CALIBRATION" \
  "$MANIFEST" 1023 "$NOISE" "$OUTPUT" "$GPU"

"$PYTHON_BIN" - "$OUTPUT" "$GENERATOR_SHA" "$DOMAIN" "$NOISE" \
  "$SELECTOR_SHA" <<'PY'
import json
import sys

p = json.load(open(sys.argv[1]))
s = p["summary"]
selector = s["stage25_selector"]
schedule = s["schedule"]
public_sha = "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
calibration_sha = "fec2a10573e913e4452f8eea92e6334fcd5e6f1104e6244876770973a1f41990"
checks = {
    "complete": s["completed"] and s["num_failures"] == 0,
    "count": s["num_tokens"] == 1023 and len(p["records"]) == 1023,
    "generator": s["checkpoint_sha256"] == sys.argv[2],
    "domain": s["generator_domain"] == sys.argv[3],
    "reference": s["reference_checkpoint_sha256"] == public_sha,
    "noise": s["evaluation_noise_namespace"] == int(sys.argv[4]),
    "selector": selector["checkpoint_sha256"] == sys.argv[5],
    "calibration": selector["calibration_sha256"] == calibration_sha,
    "deploy": not selector["calibration_collection"],
    "source": s["selector_logits_source"] == "trajectory_relative_harm_v3",
    "schedule": (
        schedule["truncation_timestep"] == 32
        and schedule["roll_timesteps"] == [32, 24, 16, 8, 0]
        and schedule["scheduler_num_inference_steps"] == 125
    ),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise RuntimeError(f"Stage27 Phase4 artifact failed {failed}")
PY
echo "PASS Stage27 Phase4 $SYSTEM namespace=$NOISE: $OUTPUT"
