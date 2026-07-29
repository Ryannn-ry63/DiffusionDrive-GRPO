#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
OUTPUT="$ROOT_DIR/artifacts/grpo_stage37/pilot/factorial_summary.json"
"$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/summarize_grpo_stage37_factorial.py" --root "$ROOT_DIR/artifacts/grpo_stage37" --baseline-root "$ROOT_DIR/artifacts/grpo_stage34/pilot/eval" --output "$OUTPUT"
"$PYTHON_BIN" - "$OUTPUT" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]));assert p['pilot_passed'],p['promotion_checks']
print(f"PASS Stage37 full-system gate gain={p['metrics']['mean_full_gain']:.6f} same_jfi_generator={p['metrics']['mean_generator_effect_same_jfi']:.6f}")
PY
