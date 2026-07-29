#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
SUMMARIZER="$ROOT_DIR/scripts/evaluation/summarize_grpo_stage39.py"
SELECTION="${STAGE39_SELECTION:-$ROOT_DIR/artifacts/grpo_stage39/pilot/selection.json}"
OUTPUT="$ROOT_DIR/artifacts/grpo_stage39/formal/gate.json"
EVAL_ROOT="$ROOT_DIR/artifacts/grpo_stage39/formal/eval"

[[ -x "$PYTHON_BIN" && -f "$SUMMARIZER" && -f "$SELECTION" ]] || {
  echo "Stage39 formal finalization inputs are incomplete"; exit 2;
}
[[ ! -e "$OUTPUT" ]] || {
  echo "refusing to overwrite Stage39 formal gate: $OUTPUT"; exit 2;
}

SELECTED_STEP=$("$PYTHON_BIN" - "$SELECTION" <<PY
import json
import sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text())
assert p.get("stage") == 39 and p.get("passed") is True
step = int(p.get("selected_step", -1))
assert step in (48, 96, 192)
print(step)
PY
)
for fold in 0 1 3; do
  for noise in 20261721 20261722; do
    for label in P20 P40 "BC${SELECTED_STEP}" "STD${SELECTED_STEP}" "SET${SELECTED_STEP}"; do
      [[ -f "$EVAL_ROOT/fold${fold}/${label}_ns${noise}.json" ]] || {
        echo "missing Stage39 formal evaluation: fold=$fold $label ns=$noise"; exit 2;
      }
    done
  done
done

mkdir -p "$(dirname "$OUTPUT")"
"$PYTHON_BIN" "$SUMMARIZER" \
  --phase formal --eval-root "$EVAL_ROOT" \
  --selection "$SELECTION" --output "$OUTPUT"

"$PYTHON_BIN" - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text())
if p.get("passed") is not True:
    raise SystemExit("Stage39 formal gate did not pass; inspect " + sys.argv[1])
PY
echo "PASS Stage39 formal gate: $OUTPUT"
