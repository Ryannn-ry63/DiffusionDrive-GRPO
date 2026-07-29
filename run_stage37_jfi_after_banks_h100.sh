#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FREEZE="$ROOT_DIR/artifacts/grpo_stage37/jfi/banks/freeze.json"
LAUNCHER="$ROOT_DIR/scripts/training/run_diffusiondrive_grpo_stage37_jfi_selector.sh"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
EXPERIMENT="${1:-stage37_jfi_chain}"
GPU="${2:-0}"
MAX_POLLS="${STAGE37_JFI_MAX_POLLS:-2880}"
[[ "$MAX_POLLS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid MAX_POLLS"; exit 2; }
for ((poll=0; poll<MAX_POLLS; poll++)); do
  if [[ -f "$FREEZE" ]] && "$PYTHON_BIN" - "$FREEZE" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); assert p.get('passed') and p.get('num_banks')==8
PY
  then
    echo "Stage37 JFI banks are frozen; starting selector audit/formal."
    exec "$LAUNCHER" chain "$EXPERIMENT" "$GPU"
  fi
  if (( poll % 20 == 0 )); then echo "Waiting for Stage37 JFI banks ($poll/$MAX_POLLS)..."; fi
  sleep 15
done
echo "timed out waiting for Stage37 JFI bank freeze: $FREEZE"
exit 2
