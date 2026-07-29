#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 audit|formal|chain EXPERIMENT [CUDA_DEVICE]"
  exit 2
fi
PHASE="$1"; EXPERIMENT="$2"; CUDA_DEVICE="${3:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SELF="$ROOT_DIR/scripts/training/run_diffusiondrive_grpo_stage37_jfi_selector.sh"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
NAVSIM_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/navsim}"
[[ -f "$NAVSIM_ROOT/planning/script/run_training.py" ]] || NAVSIM_ROOT="$ROOT_DIR/navsim"
EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
TRAIN_CACHE="${GRPO_TRAIN_CACHE:-$EXP_ROOT/training_cache}"
PUBLIC="${STAGE37_PUBLIC_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS}"
PARENT="${STAGE37_JFI_PARENT_SELECTOR:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage27_selector_multi_stage25_formal_seed27125/2026.07.24.09.28.29/lightning_logs/version_0/checkpoints/grpo-17-36684.ckpt}"
MANIFEST_FREEZE="$ROOT_DIR/artifacts/grpo_stage37/jfi/manifests/freeze.json"
BANK_FREEZE="$ROOT_DIR/artifacts/grpo_stage37/jfi/banks/freeze.json"
AUDITOR="$ROOT_DIR/scripts/evaluation/audit_grpo_stage37_jfi_checkpoint.py"
PLAN_SHA="4dc469dd0be4d2579796ff70ae7a09ed9fd8e4c92449cd057e42463d268bb5d8"

[[ "$PHASE" == audit || "$PHASE" == formal || "$PHASE" == chain ]] || { echo "invalid phase"; exit 2; }
[[ "$EXPERIMENT" =~ ^stage37_jfi_[A-Za-z0-9_.-]+$ ]] || { echo "invalid experiment"; exit 2; }
[[ "$CUDA_DEVICE" =~ ^[0-7]$ ]] || { echo "GPU must be 0..7"; exit 2; }
if [[ "$PHASE" == chain ]]; then
  "$SELF" audit "${EXPERIMENT}_audit" "$CUDA_DEVICE"
  "$SELF" formal "${EXPERIMENT}_formal" "$CUDA_DEVICE"
  echo "PASS Stage37 JFI audit+formal chain"
  exit 0
fi
for path in "$PYTHON_BIN" "$NAVSIM_ROOT/planning/script/run_training.py" "$PUBLIC" "$PARENT" "$MANIFEST_FREEZE" "$AUDITOR"; do
  [[ -f "$path" ]] || { echo "missing Stage37 JFI input: $path"; exit 2; }
done
if [[ "${GRPO_STAGE37_JFI_PREFLIGHT_ONLY:-0}" != 1 ]]; then
  [[ -f "$BANK_FREEZE" ]] || { echo "missing Stage37 JFI input: $BANK_FREEZE"; exit 2; }
fi
[[ -d "$TRAIN_CACHE" ]] || { echo "missing training cache: $TRAIN_CACHE"; exit 2; }
[[ ! -e "$EXP_ROOT/$EXPERIMENT" ]] || { echo "refusing to reuse $EXP_ROOT/$EXPERIMENT"; exit 2; }

if [[ "${GRPO_STAGE37_JFI_PREFLIGHT_ONLY:-0}" == 1 && ! -f "$BANK_FREEZE" ]]; then
  readarray -t LOCKED < <(
    "$PYTHON_BIN" - "$MANIFEST_FREEZE" "$ROOT_DIR" <<'PY'
import json,sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text()); root=Path(sys.argv[2])
print(m['train_manifest'])
for system in ('P','DPEL192','NCD192','RGT192'):
 for noise in (20263721,20263722):
  print(root/f'artifacts/grpo_stage37/jfi/banks/{system}_ns{noise}.json')
PY
  )
else
  readarray -t LOCKED < <(
  "$PYTHON_BIN" - "$MANIFEST_FREEZE" "$BANK_FREEZE" "$PUBLIC" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
mf,bf,public=map(Path,sys.argv[1:])
m=json.loads(mf.read_text()); b=json.loads(bf.read_text())
if m.get('stage')!=37 or m.get('plan_sha256')!='4dc469dd0be4d2579796ff70ae7a09ed9fd8e4c92449cd057e42463d268bb5d8': raise RuntimeError('manifest drift')
manifest=Path(m['train_manifest'])
if not manifest.is_file() or sha(manifest)!=m['train_manifest_sha256'] or int(m['count'])!=2035: raise RuntimeError('manifest SHA/count drift')
if not b.get('passed') or b.get('num_banks')!=8 or b.get('plan_sha256')!=m['plan_sha256']: raise RuntimeError('bank freeze drift')
if sha(public)!='008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b': raise RuntimeError('public drift')
print(manifest)
for item in b['artifacts']:
 p=Path(item['path'])
 if not p.is_file() or sha(p)!=item['sha256']: raise RuntimeError(f"bank drift: {p}")
 print(p)
PY
  )
fi
[[ "${#LOCKED[@]}" -eq 9 ]] || { echo "JFI requires one manifest and eight banks"; exit 2; }
MANIFEST="${LOCKED[0]}"; BANKS=("${LOCKED[@]:1}")
BANK_LIST="[$(IFS=,; echo "${BANKS[*]}")]"

MAX_EPOCHS=12; LIMIT_BATCHES=1.0; STEPS_PER_EPOCH=1018; EXPECTED_EPOCH=11; EXPECTED_STEP=12216
EXTRA_ARGS=()
if [[ "$PHASE" == audit ]]; then
  MAX_EPOCHS=1; LIMIT_BATCHES=8; STEPS_PER_EPOCH=8; EXPECTED_EPOCH=0; EXPECTED_STEP=8
  EXTRA_ARGS+=(+trainer.params.max_steps=8)
fi
OUTPUT_DIR="$ROOT_DIR/artifacts/grpo_stage37/jfi/training/$PHASE"
[[ ! -e "$OUTPUT_DIR" ]] || { echo "refusing to reuse $OUTPUT_DIR"; exit 2; }

export NAVSIM_DEVKIT_ROOT="$NAVSIM_ROOT"
export NAVSIM_EXP_ROOT="$EXP_ROOT"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

ARGS=(
  agent=diffusiondrive_agent train_test_split=trainval split=trainval
  cache_path="$TRAIN_CACHE" use_cache_without_dataset=true force_cache_computation=false
  seed=203737 dataloader.params.batch_size=2 dataloader.params.num_workers=4
  trainer.params.max_epochs="$MAX_EPOCHS" trainer.params.limit_train_batches="$LIMIT_BATCHES"
  trainer.params.limit_val_batches=0 trainer.params.accumulate_grad_batches=1
  trainer.params.accelerator=gpu trainer.params.strategy=auto trainer.params.precision=32-true
  +trainer.params.devices=1 +trainer.params.enable_progress_bar=false +trainer.params.log_every_n_steps=1
  agent.checkpoint_path="$PARENT" agent.reference_checkpoint_path="$PUBLIC" agent.lr=1e-4
  agent.config.grpo_training_mode=stage37_jfi_selector
  agent.config.stage37_jfi_train_manifest_path="$MANIFEST"
  "agent.config.stage37_jfi_candidate_bank_paths=$BANK_LIST"
  agent.config.stage37_jfi_num_members=8
  agent.config.stage37_jfi_focal_gamma=2.0
  agent.config.stage37_jfi_positive_weight=1.0
  agent.config.stage24_selector_steps_per_epoch="$STEPS_PER_EPOCH"
  agent.config.stage24_selector_subbag_fraction=0.8
  agent.config.stage37_plan_sha256="$PLAN_SHA"
  agent.config.grpo_checkpoint_save_top_k=-1
  agent.config.grpo_checkpoint_every_n_train_steps="$STEPS_PER_EPOCH"
  agent.config.diffusion_truncation_timestep=32
  agent.config.diffusion_roll_timesteps=[32,24,16,8,0]
  agent.config.diffusion_scheduler_num_inference_steps=125
  experiment_name="$EXPERIMENT"
)
cd "$ROOT_DIR"
if [[ "${GRPO_STAGE37_JFI_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" "$NAVSIM_ROOT/planning/script/run_training.py" --cfg job --resolve "${ARGS[@]}" "${EXTRA_ARGS[@]}" >/dev/null
  echo "PASS Stage37 JFI preflight phase=$PHASE manifest=$MANIFEST banks=${#BANKS[@]}"
  exit 0
fi
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" "$NAVSIM_ROOT/planning/script/run_training.py" "${ARGS[@]}" "${EXTRA_ARGS[@]}"
"$PYTHON_BIN" "$AUDITOR" --phase "$PHASE" --experiment-root "$EXP_ROOT/$EXPERIMENT" \
  --parent-selector "$PARENT" --public-checkpoint "$PUBLIC" --bank-freeze "$BANK_FREEZE" \
  --expected-epoch "$EXPECTED_EPOCH" --expected-global-step "$EXPECTED_STEP" --output-dir "$OUTPUT_DIR"
echo "PASS Stage37 JFI training phase=$PHASE; audit=$OUTPUT_DIR/audit.json"
