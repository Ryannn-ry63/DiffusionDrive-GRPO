#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 10 ]]; then echo "Usage: $0 collect|deploy GENERATOR DOMAIN JFI_CHECKPOINT CALIBRATION|- MANIFEST LIMIT NOISE OUTPUT GPU"; exit 2; fi
MODE="$1"; GENERATOR="$2"; DOMAIN="$3"; JFI="$4"; CALIBRATION="$5"; MANIFEST="$6"; LIMIT="$7"; NOISE="$8"; OUTPUT="$9"; GPU="${10}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
PUBLIC="${STAGE37_PUBLIC_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS}"
[[ "$MODE" == collect || "$MODE" == deploy ]] || { echo "invalid mode"; exit 2; }
[[ "$DOMAIN" =~ ^[a-z0-9_]+$ && "$LIMIT" =~ ^[1-9][0-9]*$ && "$GPU" =~ ^[0-7]$ ]] || { echo "invalid domain/limit/GPU"; exit 2; }
for path in "$PYTHON_BIN" "$GENERATOR" "$JFI" "$MANIFEST" "$PUBLIC"; do [[ -f "$path" ]] || { echo "missing JFI eval input: $path"; exit 2; }; done
ARGS=(--selector-logits-source joint_feasible_improvement_v1 --stage37-jfi-checkpoint "$JFI" --stage37-jfi-max-candidates 4)
if [[ "$MODE" == collect ]]; then
  [[ "$CALIBRATION" == - ]] || { echo "collect requires '-' calibration"; exit 2; }
  ARGS+=(--stage37-jfi-collect-calibration)
else
  [[ -f "$CALIBRATION" ]] || { echo "missing calibration: $CALIBRATION"; exit 2; }
  read -r JOINT Q10 OOD < <("$PYTHON_BIN" -c 'import json,sys;p=json.load(open(sys.argv[1]));assert p["passed"];print(p["joint_threshold"],p["q10_floor"],p["ood_threshold"])' "$CALIBRATION")
  ARGS+=(--stage37-jfi-calibration "$CALIBRATION" --stage37-jfi-joint-threshold "$JOINT" --stage37-jfi-q10-floor "$Q10" --stage24-selector-ood-threshold "$OOD")
fi
mkdir -p "$(dirname "$OUTPUT")"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/navsim}"; export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"; export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"; export PYTHONPATH="$ROOT_DIR:$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
if [[ "${GRPO_STAGE37_JFI_EVAL_PREFLIGHT_ONLY:-0}" == 1 ]]; then echo "PASS JFI eval preflight mode=$MODE domain=$DOMAIN"; exit 0; fi
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/evaluate_grpo_schedule.py" \
 --checkpoint "$GENERATOR" --reference-checkpoint "$PUBLIC" \
 --cache-path "${NAVSIM_TRAINING_CACHE:-$NAVSIM_EXP_ROOT/training_cache}" --metric-cache-path "${NAVSIM_METRIC_CACHE:-$NAVSIM_EXP_ROOT/metric_cache_trainval}" \
 --tokens-file "$MANIFEST" --limit "$LIMIT" --log-split train --batch-size 2 --num-workers 4 --device cuda:0 \
 --truncation-timestep 32 --roll-timesteps 32 24 16 8 0 --scheduler-num-inference-steps 125 \
 --generation-policy-algorithm "${GRPO_EVAL_GENERATION_ALGORITHM:-legacy_ppo}" --evaluation-noise-namespace "$NOISE" \
 --generator-domain "$DOMAIN" "${ARGS[@]}" --output "$OUTPUT"
