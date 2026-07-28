#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 audit|formal EXPERIMENT [CUDA_DEVICES=0,1,2,3,4,5,6,7]"
  exit 2
fi

PHASE="$1"
EXPERIMENT="$2"
CUDA_DEVICES="${3:-0,1,2,3,4,5,6,7}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x /root/miniconda3/envs/navsimH100/bin/python ]]; then
    PYTHON_BIN=/root/miniconda3/envs/navsimH100/bin/python
  else
    PYTHON_BIN=/root/miniconda3/envs/navsim/bin/python
  fi
fi
NAVSIM_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/navsim}"
if [[ ! -f "$NAVSIM_ROOT/planning/script/run_training.py" ]]; then
  NAVSIM_ROOT="$ROOT_DIR/navsim"
fi
EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
TRAIN_CACHE="${GRPO_TRAIN_CACHE:-$EXP_ROOT/training_cache}"
TRAIN_ENTRY="$NAVSIM_ROOT/planning/script/run_training.py"
PUBLIC_CHECKPOINT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS
INPUT_AUDIT="$ROOT_DIR/artifacts/grpo_stage27/audits/generator_phase3_inputs.json"

[[ "$PHASE" == audit || "$PHASE" == formal ]] || {
  echo "Stage27 generator phase must be audit or formal"
  exit 2
}
[[ "$EXPERIMENT" =~ ^stage27_generator_phase3_(audit|formal)_[A-Za-z0-9_.-]+$ ]] || {
  echo "invalid Stage27 experiment name: $EXPERIMENT"
  exit 2
}
[[ "$CUDA_DEVICES" =~ ^[0-7](,[0-7]){7}$ ]] || {
  echo "Stage27 Phase3 requires exactly eight distinct H100 device indices"
  exit 2
}
IFS=',' read -r -a DEVICE_ARRAY <<< "$CUDA_DEVICES"
declare -A SEEN_DEVICES=()
for device in "${DEVICE_ARRAY[@]}"; do
  [[ -z "${SEEN_DEVICES[$device]:-}" ]] || {
    echo "duplicate CUDA device in Stage27 list: $device"
    exit 2
  }
  SEEN_DEVICES[$device]=1
done

for path in "$PUBLIC_CHECKPOINT" "$INPUT_AUDIT" "$TRAIN_ENTRY"; do
  [[ -f "$path" ]] || {
    echo "missing locked Stage27 input: $path"
    exit 2
  }
done
[[ -d "$TRAIN_CACHE" ]] || {
  echo "missing Stage27 training cache: $TRAIN_CACHE"
  exit 2
}
[[ ! -e "$EXP_ROOT/$EXPERIMENT" ]] || {
  echo "refusing to reuse Stage27 experiment directory: $EXP_ROOT/$EXPERIMENT"
  exit 2
}

readarray -t LOCKED_VALUES < <(
  "$PYTHON_BIN" - "$INPUT_AUDIT" "$PUBLIC_CHECKPOINT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

INPUT_AUDIT_SHA = "ea3a9d173d37bd22e3b91d6a4ea3f857edf94afbba63e9dcfbe94febbe98ee89"
PUBLIC_SHA = "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


audit_path = Path(sys.argv[1])
public_path = Path(sys.argv[2])
if sha(audit_path) != INPUT_AUDIT_SHA:
    raise RuntimeError("Stage27 Phase3 input-audit SHA drifted")
audit = json.loads(audit_path.read_text(encoding="utf-8"))
if not audit.get("passed") or audit.get("stage") != 27 or audit.get("phase") != 3:
    raise RuntimeError("Stage27 Phase3 input audit is not a passing audit")
if sha(public_path) != PUBLIC_SHA or audit["public_checkpoint_sha256"] != PUBLIC_SHA:
    raise RuntimeError("Stage27 public checkpoint SHA drifted")
locked = {
    "manifest": "1b36434147a1e269285eda1472271a741f78b2e4db0745e25940e720e7b67275",
    "selector_checkpoint": "023d7b6b77bb8fa2dc3e779849f3716688abf0796b019416494c80b29615b691",
    "calibration": "fec2a10573e913e4452f8eea92e6334fcd5e6f1104e6244876770973a1f41990",
    "selector_gate": "1b442569dddafb82ea0ad0d07b34c6ded1f93151851c556a346c0dd8b98ca636",
    "selection": "1083e5b1250a04e6c480ea1a4370315579b26daed46c22a071cb7b972f91a0bb",
    "selector_freeze": "5cdccfddd175312f7e8d8449f34a7617a1460f71b98a3590cc2ead4a3611e673",
}
for field, expected in locked.items():
    path = Path(audit[field])
    if not path.is_file() or sha(path) != expected:
        raise RuntimeError(f"Stage27 locked input drifted: {field}")
    if audit[f"{field}_sha256"] != expected:
        raise RuntimeError(f"Stage27 audit hash drifted: {field}")
calibration = json.loads(Path(audit["calibration"]).read_text(encoding="utf-8"))
print(audit["manifest"])
print(audit["selector_checkpoint"])
print(audit["calibration"])
print(calibration["residual_margin"])
print(calibration["risk_threshold"])
print(calibration["ood_threshold"])
PY
)
[[ "${#LOCKED_VALUES[@]}" -eq 6 ]] || {
  echo "failed to resolve Stage27 locked inputs"
  exit 2
}
MANIFEST="${LOCKED_VALUES[0]}"
SELECTOR="${LOCKED_VALUES[1]}"
CALIBRATION="${LOCKED_VALUES[2]}"
MARGIN="${LOCKED_VALUES[3]}"
RISK="${LOCKED_VALUES[4]}"
OOD="${LOCKED_VALUES[5]}"

MAX_EPOCHS=2
LIMIT_BATCHES=1.0
CHECKPOINT_EVERY=0
EXTRA_ARGS=()
if [[ "$PHASE" == audit ]]; then
  MAX_EPOCHS=1
  LIMIT_BATCHES=8
  CHECKPOINT_EVERY=1
  EXTRA_ARGS+=(+trainer.params.max_steps=1)
fi

TRAIN_ARGS=(
  agent=diffusiondrive_agent
  train_test_split=trainval
  cache_path="$TRAIN_CACHE"
  use_cache_without_dataset=true
  force_cache_computation=false
  seed=0
  dataloader.params.batch_size=1
  dataloader.params.num_workers=4
  trainer.params.max_epochs="$MAX_EPOCHS"
  trainer.params.limit_train_batches="$LIMIT_BATCHES"
  trainer.params.limit_val_batches=0
  trainer.params.accumulate_grad_batches=8
  trainer.params.gradient_clip_val=1.0
  trainer.params.accelerator=gpu
  trainer.params.strategy=ddp
  trainer.params.precision=32-true
  +trainer.params.devices=8
  +trainer.params.enable_progress_bar=false
  +trainer.params.log_every_n_steps=1
  agent.checkpoint_path="$PUBLIC_CHECKPOINT"
  agent.reference_checkpoint_path="$PUBLIC_CHECKPOINT"
  agent.lr=1e-6
  agent.config.grpo_training_mode=stage27_public_diffgrpo_selected_set
  agent.config.generation_policy_algorithm=diffgrpo_selected_set
  agent.config.grpo_decoder_gradient_scope=all_layers
  agent.config.grpo_decoder_layer0_lr_mult=0.1
  agent.config.inference_selector_source=trajectory_relative_harm_v3
  agent.config.stage25_selector_checkpoint_path="$SELECTOR"
  agent.config.stage25_selector_calibration_path="$CALIBRATION"
  agent.config.stage24_selector_residual_margin="$MARGIN"
  agent.config.stage25_selector_risk_threshold="$RISK"
  agent.config.stage24_selector_ood_threshold="$OOD"
  agent.config.stage23_generator_train_manifest_path="$MANIFEST"
  agent.config.grpo_reward_mode=pdms
  agent.config.grpo_scene_weight_mode=uniform
  agent.config.selection_behavior_weighting=old_policy
  agent.config.selection_entropy_weight=0.0
  agent.config.selection_exploration_floor=0.0
  agent.config.selection_rank_loss_weight=0.0
  agent.config.selector_consistency_kl_weight=0.0
  agent.config.selector_generation_kl_weight=0.0
  agent.config.policy_loss_weight=0.0
  agent.config.kl_loss_weight=0.0
  agent.config.generation_policy_loss_weight=1.0
  agent.config.generation_kl_loss_weight=0.0
  agent.config.generation_adaptive_kl_enabled=false
  agent.config.diffgrpo_bc_weight=0.1
  agent.config.diffgrpo_step_discount=0.6
  agent.config.diffgrpo_logprob_reduction=mean
  agent.config.diffgrpo_group_size=8
  agent.config.diffgrpo_paired_regular_kl_weight=0.1
  agent.config.diffgrpo_paired_mature_kl_weight=0.5
  agent.config.diffgrpo_safety_regression_tolerance=0.000001
  agent.config.generation_advantage_mode=group_zscore
  agent.config.grpo_rollouts_per_mode=1
  agent.config.generation_mode_weighting=uniform
  agent.config.generation_trust_projection_mode=none
  agent.config.grpo_old_policy_sync_steps=32
  agent.config.grpo_priority_manifest_path=
  agent.config.grpo_priority_sample_fraction=0.0
  agent.config.diffusion_truncation_timestep=32
  agent.config.diffusion_roll_timesteps=[32,24,16,8,0]
  agent.config.diffusion_scheduler_num_inference_steps=125
  agent.config.grpo_checkpoint_save_top_k=-1
  agent.config.grpo_checkpoint_every_n_train_steps="$CHECKPOINT_EVERY"
  experiment_name="$EXPERIMENT"
)

if [[ "${GRPO_STAGE27_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "PASS Stage27 Phase3 preflight phase=$PHASE"
  echo "Public checkpoint: $PUBLIC_CHECKPOINT"
  echo "Manifest: $MANIFEST"
  echo "Selector: $SELECTOR"
  echo "Calibration: $CALIBRATION"
  exit 0
fi

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" "$TRAIN_ENTRY" \
  "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}"
