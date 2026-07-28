#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 audit|final P|A|B0|B|C [CUDA_DEVICES=0,1,2,3,4,5,6,7]"
  exit 2
fi

PHASE="$1"
SYSTEM="$2"
CUDA_DEVICES="${3:-0,1,2,3,4,5,6,7}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
NAVSIM_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/navsim}"
EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
BASE="$EXP_ROOT/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model"
PUBLIC=/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS
FINAL="$EXP_ROOT/stage26_navtest_generator_full6119_seed0/2026.07.23.17.12.16/lightning_logs/version_0/checkpoints/grpo-01-192.ckpt"
SELECTOR="$EXP_ROOT/stage26_stage25_selector_formal_seed26025/2026.07.23.13.16.13/lightning_logs/version_0/checkpoints/grpo-14-30570.ckpt"
CALIBRATION="$ROOT_DIR/artifacts/grpo_stage26/stage26_selector_calibration_fold4.json"
CHECKPOINT_AUDIT="$ROOT_DIR/artifacts/grpo_stage26/navtest_generator_checkpoint_audit.json"
METRIC_CACHE="${STAGE26_NAVTEST_METRIC_CACHE:-$EXP_ROOT/metric_cache_navtest}"
EVAL_ENTRY="$NAVSIM_ROOT/planning/script/run_pdm_score.py"

[[ "$PHASE" == audit || "$PHASE" == final ]] || {
  echo "PHASE must be audit or final"
  exit 2
}
[[ "$SYSTEM" == P || "$SYSTEM" == A || "$SYSTEM" == B0 || \
   "$SYSTEM" == B || "$SYSTEM" == C ]] || {
  echo "SYSTEM must be P, A, B0, B, or C"
  exit 2
}
[[ "$CUDA_DEVICES" =~ ^[0-7](,[0-7])*$ ]] || {
  echo "CUDA_DEVICES must be a comma-separated subset of 0..7"
  exit 2
}
if [[ ! -f "$EVAL_ENTRY" ]]; then
  NAVSIM_ROOT="$ROOT_DIR/navsim"
  EVAL_ENTRY="$NAVSIM_ROOT/planning/script/run_pdm_score.py"
fi
for path in "$BASE" "$PUBLIC" "$FINAL" "$SELECTOR" "$CALIBRATION" \
  "$CHECKPOINT_AUDIT" "$EVAL_ENTRY"; do
  [[ -f "$path" ]] || { echo "missing locked Stage26 input: $path"; exit 2; }
done
[[ -d "$METRIC_CACHE" ]] || {
  echo "missing NavTest metric cache: $METRIC_CACHE"
  exit 2
}

read -r MARGIN RISK OOD < <(
  "$PYTHON_BIN" -c '
import hashlib,json,pathlib,sys
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
base,public,final,selector,calibration,audit,cache=sys.argv[1:]
assert sha(base) == "59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d"
assert sha(public) == "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
assert sha(final) == "1018f1b1a1cfcb69c27b97c2fbd30099a20a62e55fc02897f6588ca546abad6d"
assert sha(selector) == "77ee768c5f469292d81049192d48797f2f3fae4f80029ae6853dd9d9c6de4207"
assert sha(calibration) == "0c14b8f9531d64028bdf4d71f35b34418ab31ad4ae11ccc6b3c60a4325f53a0d"
assert sha(audit) == "be20e5c5ed1ad2d58f87fef11f2a5d28d1b85637ee843e2a358ced0c553befb6"
a=json.load(open(audit))
assert a["passed"] and a["navtest_authorized"]
assert a["checkpoint_sha256"] == sha(final)
c=json.load(open(calibration))
assert c["passed"] and c["selector_checkpoint_sha256"] == sha(selector)
count=sum(1 for _ in pathlib.Path(cache).rglob("metric_cache.pkl"))
assert count == 12146, f"expected 12146 NavTest caches, got {count}"
print(c["residual_margin"],c["risk_threshold"],c["ood_threshold"])
' "$BASE" "$PUBLIC" "$FINAL" "$SELECTOR" "$CALIBRATION" "$CHECKPOINT_AUDIT" \
  "$METRIC_CACHE"
)

GENERATOR="$BASE"
REFERENCE="$BASE"
SELECTOR_SOURCE=current
TRUNCATION=8
ROLL_TIMESTEPS='[8,0]'
EXPERIMENT="stage26_navtest_pdms_${SYSTEM}"

case "$SYSTEM" in
  P)
    # Actual epoch-99 checkpoint released for the public 88.1 result.
    GENERATOR="$PUBLIC"
    REFERENCE="$PUBLIC"
    ;;
  A)
    # Locked Stage13-26 local epoch-19 base under the public two-step schedule.
    # It is retained for paired attribution and is not the epoch-99 release.
    ;;
  B0)
    # Schedule-matched official fallback used for Stage26 decomposition.
    TRUNCATION=32
    ROLL_TIMESTEPS='[32,24,16,8,0]'
    ;;
  B)
    # Official generator plus the frozen cross-generator Stage26 selector.
    SELECTOR_SOURCE=trajectory_relative_harm_v3
    TRUNCATION=32
    ROLL_TIMESTEPS='[32,24,16,8,0]'
    ;;
  C)
    # Full-6119 GRPO generator plus the same frozen Stage26 selector.
    GENERATOR="$FINAL"
    SELECTOR_SOURCE=trajectory_relative_harm_v3
    TRUNCATION=32
    ROLL_TIMESTEPS='[32,24,16,8,0]'
    ;;
esac

WORKER=ray_distributed
EXTRA_ARGS=(worker.threads_per_node="${STAGE26_NAVTEST_WORKERS:-64}")
if [[ "$PHASE" == audit ]]; then
  EXPERIMENT="stage26_navtest_pdms_audit_${SYSTEM}"
  WORKER=sequential
  EXTRA_ARGS=(train_test_split.scene_filter.max_scenes=1)
else
  "$PYTHON_BIN" -c '
from pathlib import Path
import sys
root=Path(sys.argv[1])
csvs=list(root.glob("*/*.csv")) if root.is_dir() else []
if csvs:
    raise SystemExit("completed final NavTest CSV already exists: " + str(csvs[-1]))
' "$EXP_ROOT/$EXPERIMENT"
fi

AGENT_ARGS=(
  agent.checkpoint_path="$GENERATOR"
  agent.reference_checkpoint_path="$REFERENCE"
  agent.config.inference_selector_source="$SELECTOR_SOURCE"
  agent.config.diffusion_truncation_timestep="$TRUNCATION"
  agent.config.diffusion_roll_timesteps="$ROLL_TIMESTEPS"
  agent.config.diffusion_scheduler_num_inference_steps=125
  agent.config.evaluation_noise_namespace=-1
)
if [[ "$SELECTOR_SOURCE" == trajectory_relative_harm_v3 ]]; then
  AGENT_ARGS+=(
    agent.config.stage25_selector_checkpoint_path="$SELECTOR"
    agent.config.stage25_selector_calibration_path="$CALIBRATION"
    agent.config.stage24_selector_residual_margin="$MARGIN"
    agent.config.stage25_selector_risk_threshold="$RISK"
    agent.config.stage24_selector_ood_threshold="$OOD"
  )
fi

cd "$ROOT_DIR"
echo "Stage26 NavTest phase=$PHASE system=$SYSTEM experiment=$EXPERIMENT"
echo "generator=$GENERATOR"
echo "selector=$SELECTOR_SOURCE schedule=$ROLL_TIMESTEPS"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" "$EVAL_ENTRY" \
  train_test_split=navtest agent=diffusiondrive_agent worker="$WORKER" \
  metric_cache_path="$METRIC_CACHE" experiment_name="$EXPERIMENT" \
  "${AGENT_ARGS[@]}" "${EXTRA_ARGS[@]}"
