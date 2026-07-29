#!/usr/bin/env bash
set -euo pipefail

# Produce the three independent 8×H100 pilot commands as one immutable queue
# manifest.  A single 8-GPU DDP allocation cannot run BC/STD/SET concurrently;
# each line is therefore meant to be submitted as a separate queue job.

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [RUN_ID]"
  exit 2
fi

RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT_DIR="/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive"
PILOT="$ROOT_DIR/run_stage39_pilot_train_h100.sh"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid RUN_ID"; exit 2; }
[[ -x "$PILOT" ]] || { echo "missing pilot launcher: $PILOT"; exit 2; }

MANIFEST="$ROOT_DIR/artifacts/grpo_stage39/queue/stage39_pilot_${RUN_ID}.sh"
if [[ -e "$MANIFEST" ]]; then
  echo "refusing to overwrite existing queue manifest: $MANIFEST"
  exit 2
fi
mkdir -p "$(dirname "$MANIFEST")"
{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "# Submit each line as its own 8×H100 queue task."
  echo "cd $ROOT_DIR"
  printf 'bash ./run_stage39_pilot_train_h100.sh BC %q\n' "${RUN_ID}_bc"
  printf 'bash ./run_stage39_pilot_train_h100.sh STD %q\n' "${RUN_ID}_std"
  printf 'bash ./run_stage39_pilot_train_h100.sh SET %q\n' "${RUN_ID}_set"
} >"$MANIFEST"
chmod +x "$MANIFEST"

echo "PASS Stage39 pilot queue manifest: $MANIFEST"
echo "Submit the three branch lines as separate 8×H100 jobs:"
sed -n '4,6p' "$MANIFEST"
