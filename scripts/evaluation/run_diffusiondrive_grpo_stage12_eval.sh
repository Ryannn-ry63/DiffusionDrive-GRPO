#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 9 ]]; then
  echo "Usage: $0 CHECKPOINT OUTPUT_JSON TOKENS_FILE LIMIT [BASELINE_ARTIFACT=none] [CACHE_PATH=default] [METRIC_CACHE_PATH=default] [LOG_SPLIT=val] [CUDA_DEVICE=0]"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

exec "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage11_eval.sh" \
  reference "$1" "$2" "$3" "$4" \
  "${5:-none}" "${6:-default}" "${7:-default}" "${8:-val}" "${9:-0}"
