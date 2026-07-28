#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 HOLDOUT(0|1) [EXPERIMENT] [CUDA_DEVICES]"
  exit 2
fi

HOLDOUT="$1"
EXPERIMENT="${2:-stage35_ncd_chain_f${HOLDOUT}}"
CUDA_DEVICES="${3:-0,1,2,3,4,5,6,7}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$ROOT_DIR/scripts/training/run_diffusiondrive_grpo_stage35_pilot.sh"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
AUDIT="$ROOT_DIR/artifacts/grpo_stage35/pilot/training/NCD/fold${HOLDOUT}/audit/audit.json"

[[ "$HOLDOUT" =~ ^[01]$ ]] || { echo "holdout must be 0 or 1"; exit 2; }
[[ -x "$LAUNCHER" ]] || { echo "missing executable Stage35 launcher: $LAUNCHER"; exit 2; }

cd "$ROOT_DIR"
if [[ -f "$AUDIT" ]]; then
  "$PYTHON_BIN" - "$AUDIT" "$HOLDOUT" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
fold = int(sys.argv[2])
if not (
    payload.get("passed") is True
    and payload.get("stage") == 35
    and payload.get("phase") == "audit"
    and payload.get("holdout_fold") == fold
    and payload.get("candidate_level_trace") is True
    and payload.get("same_anchor_counterfactuals_only") is True
    and payload.get("common_noise_current_public_8x20") is True
    and payload.get("counterfactual_signal_health", {}).get("passed") is True
):
    raise RuntimeError("existing Stage35 audit is not reusable")
PY
  echo "Reusing frozen PASS Stage35 audit for fold=$HOLDOUT; starting formal."
  exec "$LAUNCHER" formal NCD "$HOLDOUT" "${EXPERIMENT}_formal" "$CUDA_DEVICES"
fi
exec "$LAUNCHER" chain NCD "$HOLDOUT" "$EXPERIMENT" "$CUDA_DEVICES"
