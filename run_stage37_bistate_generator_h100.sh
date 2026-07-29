#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 HOLDOUT(0|1) [EXPERIMENT] [CUDA_DEVICES]"
  exit 2
fi

HOLDOUT="$1"
EXPERIMENT="${2:-stage37_bpd_chain_f${HOLDOUT}}"
CUDA_DEVICES="${3:-0,1,2,3,4,5,6,7}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$ROOT_DIR/scripts/training/run_diffusiondrive_grpo_stage37_generator.sh"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
AUDIT="$ROOT_DIR/artifacts/grpo_stage37/generator/fold${HOLDOUT}/audit/audit.json"

[[ "$HOLDOUT" =~ ^[01]$ ]] || { echo "holdout must be 0 or 1"; exit 2; }
[[ -x "$LAUNCHER" ]] || {
  echo "missing executable Stage37 launcher: $LAUNCHER"; exit 2;
}
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
    and payload.get("stage") == 37
    and payload.get("phase") == "audit"
    and payload.get("holdout_fold") == fold
    and payload.get("bistate_current_and_public_frontier") is True
    and payload.get("gradient_projection_after_accumulation") is True
    and payload.get("exact_64_tensor_boundary") is True
):
    raise RuntimeError("existing Stage37 audit is not reusable")
PY
  echo "Reusing PASS Stage37 audit fold=$HOLDOUT; starting formal."
  exec "$LAUNCHER" formal "$HOLDOUT" "${EXPERIMENT}_formal" "$CUDA_DEVICES"
fi
exec "$LAUNCHER" chain "$HOLDOUT" "$EXPERIMENT" "$CUDA_DEVICES"
