#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 audit|formal|chain CDC FOLD(0|1|2|3) EXPERIMENT [CUDA_DEVICES]"
  exit 2
fi
PHASE="$1"; BRANCH="$2"; FOLD="$3"; EXPERIMENT="$4"
CUDA_DEVICES="${5:-0,1,2,3,4,5,6,7}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
TRAIN_CACHE="${GRPO_TRAIN_CACHE:-$EXP_ROOT/training_cache}"
PUBLIC="${STAGE33_PUBLIC_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS}"
SELECTOR="${STAGE33_SELECTOR_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage27_selector_multi_stage25_formal_seed27125/2026.07.24.09.28.29/lightning_logs/version_0/checkpoints/grpo-17-36684.ckpt}"
CALIBRATION="${STAGE33_SELECTOR_CALIBRATION:-$ROOT_DIR/artifacts/grpo_stage27/selectors/multi/calibration.json}"
CV_FREEZE="${STAGE33_CV_FREEZE:-$ROOT_DIR/artifacts/grpo_stage30/manifests/cv/freeze.json}"
PLAN="$ROOT_DIR/GRPO_STAGE33_CDC_GPRO_PLAN_20260726.md"
PLAN_SHA=0ace8d8959360a15cfac8f45bc2c6e19898c09ad1a04aa7b5e5dcc2fb36283ff
TRAIN_ENTRY="$ROOT_DIR/navsim/planning/script/run_training.py"
SELF="$ROOT_DIR/scripts/training/run_diffusiondrive_grpo_stage33_cdc_pilot.sh"

[[ "$PHASE" == audit || "$PHASE" == formal || "$PHASE" == chain ]] || { echo "phase must be audit, formal, or chain"; exit 2; }
[[ "$BRANCH" == CDC ]] || { echo "Stage33 pilot currently exposes only the CDC mainline; V2 control is a separate ablation"; exit 2; }
[[ "$FOLD" =~ ^[0-3]$ ]] || { echo "fold must be 0,1,2,3"; exit 2; }
[[ "$CUDA_DEVICES" =~ ^[0-7](,[0-7]){7}$ ]] || { echo "Stage33 requires exactly 8 GPU indices"; exit 2; }
for path in "$PYTHON_BIN" "$TRAIN_ENTRY" "$PUBLIC" "$SELECTOR" "$CALIBRATION" "$CV_FREEZE" "$PLAN"; do
  [[ -e "$path" ]] || { echo "missing Stage33 input: $path"; exit 2; }
done
[[ "$(sha256sum "$PLAN" | awk '{print $1}')" == "$PLAN_SHA" ]] || { echo "Stage33 plan SHA drifted"; exit 2; }

# A chain job is safe to submit while another H100 job is queued: it runs its
# own one-step audit, audits the produced checkpoint, and starts formal only
# after every audit gate passes.  The two phases use separate directories so a
# resumed/partial audit can never be mistaken for a formal run.
if [[ "$PHASE" == chain ]]; then
  AUDIT_EXPERIMENT="${EXPERIMENT}_audit"
  FORMAL_EXPERIMENT="${EXPERIMENT}_formal"
  if [[ "${GRPO_STAGE33_PREFLIGHT_ONLY:-0}" == 1 ]]; then
    "$SELF" audit "$BRANCH" "$FOLD" "$AUDIT_EXPERIMENT" "$CUDA_DEVICES"
    "$SELF" formal "$BRANCH" "$FOLD" "$FORMAL_EXPERIMENT" "$CUDA_DEVICES"
    exit 0
  fi
  "$SELF" audit "$BRANCH" "$FOLD" "$AUDIT_EXPERIMENT" "$CUDA_DEVICES"
  AUDIT_CHECKPOINT=$("$PYTHON_BIN" - "$EXP_ROOT/$AUDIT_EXPERIMENT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
candidates = sorted(root.rglob("last.ckpt"), key=lambda p: p.stat().st_mtime)
if not candidates:
    raise SystemExit("Stage33 chain could not find audit last.ckpt")
print(candidates[-1])
PY
  )
  AUDIT_REPORT="$ROOT_DIR/artifacts/grpo_stage33/audit/${EXPERIMENT}_audit.json"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/audit_grpo_stage33_checkpoint.py" \
    --phase audit --checkpoint "$AUDIT_CHECKPOINT" --base "$PUBLIC" \
    --plan "$PLAN" --output "$AUDIT_REPORT"
  "$SELF" formal "$BRANCH" "$FOLD" "$FORMAL_EXPERIMENT" "$CUDA_DEVICES"
  exit 0
fi

[[ ! -e "$EXP_ROOT/$EXPERIMENT" ]] || { echo "refusing to reuse experiment directory $EXP_ROOT/$EXPERIMENT"; exit 2; }

readarray -t LOCKED < <("$PYTHON_BIN" - "$CV_FREEZE" "$FOLD" "$CALIBRATION" <<'PY'
import hashlib, json, sys
from pathlib import Path
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
freeze = json.loads(Path(sys.argv[1]).read_text())
fold = int(sys.argv[2])
calibration = Path(sys.argv[3])
entry = freeze["folds"][fold]
bucket = Path(freeze["bucket_manifest"])
if sha(Path(entry["path"])) != entry["sha256"] or sha(bucket) != freeze["bucket_manifest_sha256"]:
    raise RuntimeError("Stage30 CV freeze drifted")
cal = json.loads(calibration.read_text())
if not cal.get("passed"):
    raise RuntimeError("Stage25 calibration did not pass")
print(entry["path"])
print(bucket)
print(cal["residual_margin"])
print(cal["risk_threshold"])
print(cal["ood_threshold"])
PY
)
[[ "${#LOCKED[@]}" -eq 5 ]] || { echo "failed Stage33 manifest resolution"; exit 2; }
TRAIN_MANIFEST="${LOCKED[0]}"; BUCKET_MANIFEST="${LOCKED[1]}"
MARGIN="${LOCKED[2]}"; RISK="${LOCKED[3]}"; OOD="${LOCKED[4]}"

export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/navsim}"
export NAVSIM_EXP_ROOT="$EXP_ROOT"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1 RAY_DEDUP_LOGS=0

MAX_EPOCHS=4; LIMIT_BATCHES=1.0; CHECKPOINT_EVERY=48; EXTRA_ARGS=()
if [[ "$PHASE" == audit ]]; then
  MAX_EPOCHS=1; LIMIT_BATCHES=8; CHECKPOINT_EVERY=1
  EXTRA_ARGS+=(+trainer.params.max_steps=1)
fi
SEED=$((203300 + FOLD))
TRAIN_ARGS=(
  agent=diffusiondrive_agent train_test_split=trainval split=trainval
  cache_path="$TRAIN_CACHE" use_cache_without_dataset=true force_cache_computation=false
  seed="$SEED" dataloader.params.batch_size=1 dataloader.params.num_workers=4
  trainer.params.max_epochs="$MAX_EPOCHS" trainer.params.limit_train_batches="$LIMIT_BATCHES"
  trainer.params.limit_val_batches=0 trainer.params.accumulate_grad_batches=8
  trainer.params.gradient_clip_val=1.0 trainer.params.accelerator=gpu
  trainer.params.strategy=ddp trainer.params.precision=32-true
  +trainer.params.devices=8 +trainer.params.use_distributed_sampler=false
  +trainer.params.enable_progress_bar=false +trainer.params.log_every_n_steps=1
  agent.checkpoint_path="$PUBLIC" agent.reference_checkpoint_path="$PUBLIC" agent.lr=1e-6
  agent.config.grpo_training_mode=stage33_cdc_grpo
  agent.config.generation_policy_algorithm=diffgrpo_deployed_selected_set
  agent.config.grpo_decoder_gradient_scope=all_layers agent.config.grpo_decoder_layer0_lr_mult=0.1
  agent.config.inference_selector_source=trajectory_relative_harm_v3
  agent.config.stage25_selector_checkpoint_path="$SELECTOR"
  agent.config.stage25_selector_calibration_path="$CALIBRATION"
  agent.config.stage24_selector_residual_margin="$MARGIN"
  agent.config.stage25_selector_risk_threshold="$RISK"
  agent.config.stage24_selector_ood_threshold="$OOD"
  agent.config.stage23_generator_train_manifest_path="$TRAIN_MANIFEST"
  agent.config.stage31_bucket_manifest_path="$BUCKET_MANIFEST"
  agent.config.stage33_bucket_manifest_path="$BUCKET_MANIFEST"
  agent.config.stage31_plan_sha256=3ff3d3bcd3120b9d73a017876f942458eb3fa16651517e4e30b27cf2fff7bb0d
  agent.config.stage31_optimizer_steps_per_epoch=48 agent.config.stage31_gradient_accumulation=8
  agent.config.stage31_global_bucket_composition=[2,30,8,24]
  agent.config.stage33_plan_sha256="$PLAN_SHA" agent.config.stage33_cdc_objective_revision=cdc_deployment_credit_v1
  agent.config.stage33_deployment_weight=0.5 agent.config.stage33_headroom_weight=0.5
  agent.config.stage33_headroom_margin=0.001 agent.config.stage33_advantage_clip=2.0
  agent.config.stage33_optimizer_steps_per_epoch=48 agent.config.stage33_gradient_accumulation=8
  agent.config.stage33_global_bucket_composition=[2,30,8,24]
  agent.config.grpo_reward_mode=pdms agent.config.grpo_scene_weight_mode=uniform
  agent.config.selection_behavior_weighting=old_policy agent.config.selection_entropy_weight=0.0
  agent.config.selection_exploration_floor=0.0 agent.config.selection_rank_loss_weight=0.0
  agent.config.selector_consistency_kl_weight=0.0 agent.config.selector_generation_kl_weight=0.0
  agent.config.policy_loss_weight=0.0 agent.config.kl_loss_weight=0.0
  agent.config.generation_policy_loss_weight=1.0 agent.config.generation_kl_loss_weight=0.0
  agent.config.generation_adaptive_kl_enabled=false agent.config.diffgrpo_bc_weight=0.1
  agent.config.diffgrpo_step_discount=0.6 agent.config.diffgrpo_logprob_reduction=mean
  agent.config.diffgrpo_group_size=8 agent.config.diffgrpo_base_advantage_clip=2.0
  agent.config.diffgrpo_safety_regression_tolerance=0.000001
  agent.config.generation_advantage_mode=group_zscore agent.config.grpo_rollouts_per_mode=1
  agent.config.generation_mode_weighting=uniform agent.config.generation_trust_projection_mode=none
  agent.config.grpo_old_policy_sync_steps=32 agent.config.weight_decay=0.0
  agent.config.diffusion_truncation_timestep=32 agent.config.diffusion_roll_timesteps=[32,24,16,8,0]
  agent.config.diffusion_scheduler_num_inference_steps=125
  agent.config.grpo_checkpoint_save_top_k=-1 agent.config.grpo_checkpoint_every_n_train_steps="$CHECKPOINT_EVERY"
  experiment_name="$EXPERIMENT"
)

cd "$ROOT_DIR"
if [[ "${GRPO_STAGE33_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" "$TRAIN_ENTRY" \
    --cfg job --resolve "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}" >/dev/null
  echo "PASS Stage33 CDC preflight phase=$PHASE fold=$FOLD"
  echo "Train manifest: $TRAIN_MANIFEST"
  echo "Bucket manifest: $BUCKET_MANIFEST"
  exit 0
fi
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" "$TRAIN_ENTRY" "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}"
