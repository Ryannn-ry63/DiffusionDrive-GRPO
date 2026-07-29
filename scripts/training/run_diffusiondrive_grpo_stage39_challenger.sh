#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 audit|pilot|formal|chain BC|STD|SET HOLDOUT(0|1|2|3) EXPERIMENT [CUDA_DEVICES]"
  exit 2
fi

PHASE="$1"
BRANCH="${2^^}"
HOLDOUT="$3"
EXPERIMENT="$4"
CUDA_DEVICES="${5:-0,1,2,3,4,5,6,7}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
NAVSIM_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/navsim}"
[[ -f "$NAVSIM_ROOT/planning/script/run_training.py" ]] || NAVSIM_ROOT="$ROOT_DIR/navsim"
EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
TRAIN_CACHE="${GRPO_TRAIN_CACHE:-$EXP_ROOT/training_cache}"
TRAIN_ENTRY="$NAVSIM_ROOT/planning/script/run_training.py"
PUBLIC="${STAGE39_PUBLIC_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS}"
SELECTOR="${STAGE39_SELECTOR_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage27_selector_multi_stage25_formal_seed27125/2026.07.24.09.28.29/lightning_logs/version_0/checkpoints/grpo-17-36684.ckpt}"
CALIBRATION="${STAGE39_SELECTOR_CALIBRATION:-$ROOT_DIR/artifacts/grpo_stage27/selectors/multi/calibration.json}"
CV_FREEZE="${STAGE39_CV_FREEZE:-$ROOT_DIR/artifacts/grpo_stage30/manifests/cv/freeze.json}"
PLAN="$ROOT_DIR/GRPO_STAGE39_40_NON_DESTRUCTIVE_CHALLENGER_PLAN_20260729.md"
PLAN_SHA="7e1e6142412778524714a0b919f971edf1d2bf14138b0efcf21fc9dd7c4d6f1b"
SELF="$ROOT_DIR/scripts/training/run_diffusiondrive_grpo_stage39_challenger.sh"
FREEZER="$ROOT_DIR/scripts/evaluation/freeze_grpo_stage39_checkpoints.py"
AUDITOR="$ROOT_DIR/scripts/evaluation/audit_grpo_stage39_checkpoint.py"
PUBLIC_SHA="008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"

[[ "$PHASE" == audit || "$PHASE" == pilot || "$PHASE" == formal || "$PHASE" == chain ]] || {
  echo "phase must be audit, pilot, formal, or chain"; exit 2;
}
[[ "$BRANCH" =~ ^(BC|STD|SET)$ ]] || { echo "branch must be BC, STD, or SET"; exit 2; }
[[ "$HOLDOUT" =~ ^[0-3]$ ]] || { echo "holdout must be 0, 1, 2, or 3"; exit 2; }
if [[ "$PHASE" == pilot && "$HOLDOUT" != 2 ]]; then
  echo "Stage39 pilot is frozen to holdout fold2"; exit 2
fi
if [[ "$PHASE" == formal && ! "$HOLDOUT" =~ ^(0|1|3)$ ]]; then
  echo "Stage39 formal is frozen to holdout folds0/1/3"; exit 2
fi
[[ "$EXPERIMENT" =~ ^stage39_[A-Za-z0-9_.-]+$ ]] || {
  echo "invalid Stage39 experiment name: $EXPERIMENT"; exit 2;
}
[[ "$CUDA_DEVICES" =~ ^[0-7](,[0-7]){7}$ ]] || {
  echo "Stage39 requires exactly eight GPU indices"; exit 2;
}
for path in "$PYTHON_BIN" "$TRAIN_ENTRY" "$PUBLIC" "$SELECTOR" "$CALIBRATION" "$CV_FREEZE" "$PLAN" "$FREEZER" "$AUDITOR"; do
  [[ -e "$path" ]] || { echo "missing Stage39 input: $path"; exit 2; }
done
[[ "$(sha256sum "$PLAN" | awk '{print $1}')" == "$PLAN_SHA" ]] || {
  echo "Stage39 plan SHA drifted"; exit 2;
}
[[ "$(sha256sum "$PUBLIC" | awk '{print $1}')" == "$PUBLIC_SHA" ]] || {
  echo "Stage39 public checkpoint SHA drifted"; exit 2;
}
[[ -d "$TRAIN_CACHE" ]] || { echo "missing Stage39 training cache: $TRAIN_CACHE"; exit 2; }

case "$BRANCH" in
  BC) TRAINING_MODE=stage39_challenger_bc; BRANCH_OFFSET=0 ;;
  STD) TRAINING_MODE=stage39_challenger_standard_grpo; BRANCH_OFFSET=10 ;;
  SET) TRAINING_MODE=stage39_challenger_set_grpo; BRANCH_OFFSET=20 ;;
esac

export NAVSIM_DEVKIT_ROOT="$NAVSIM_ROOT"
export NAVSIM_EXP_ROOT="$EXP_ROOT"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0

if [[ "${GRPO_STAGE39_PREFLIGHT_ONLY:-0}" != 1 && "$PHASE" =~ ^(audit|pilot)$ ]]; then
  PHASE0="$ROOT_DIR/artifacts/grpo_stage39/phase0/audit.json"
  [[ -f "$PHASE0" ]] || { echo "Stage39 $PHASE requires passing Phase0: $PHASE0"; exit 2; }
  "$PYTHON_BIN" - "$PHASE0" "$PLAN_SHA" <<'PYPHASE0'
import json,sys
p=json.load(open(sys.argv[1]))
assert p.get('stage')==39 and p.get('phase')=='offline_target_audit'
assert p.get('passed') is True and p.get('plan_sha256')==sys.argv[2]
PYPHASE0
fi

if [[ "$PHASE" == chain ]]; then
  NEXT_PHASE=formal
  [[ "$HOLDOUT" == 2 ]] && NEXT_PHASE=pilot
  "$SELF" audit "$BRANCH" "$HOLDOUT" "${EXPERIMENT}_audit" "$CUDA_DEVICES"
  "$SELF" "$NEXT_PHASE" "$BRANCH" "$HOLDOUT" "${EXPERIMENT}_${NEXT_PHASE}" "$CUDA_DEVICES"
  echo "PASS Stage39 queue-safe chain phase=$NEXT_PHASE branch=$BRANCH holdout=$HOLDOUT"
  exit 0
fi

[[ ! -e "$EXP_ROOT/$EXPERIMENT" ]] || {
  echo "refusing to reuse Stage39 experiment directory $EXP_ROOT/$EXPERIMENT"; exit 2;
}

readarray -t LOCKED < <(
  "$PYTHON_BIN" - "$CV_FREEZE" "$HOLDOUT" "$CALIBRATION" <<'PYLOCK'
import hashlib
import json
import sys
from pathlib import Path

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

freeze = json.loads(Path(sys.argv[1]).read_text())
holdout = int(sys.argv[2])
calibration = json.loads(Path(sys.argv[3]).read_text())
entry = freeze["folds"][holdout]
bucket = Path(freeze["bucket_manifest"])
if freeze.get("stage") != 30 or entry["holdout_fold"] != holdout:
    raise RuntimeError("Stage39 CV freeze semantics drifted")
for path, expected in (
    (Path(entry["path"]), entry["sha256"]),
    (bucket, freeze["bucket_manifest_sha256"]),
):
    if not path.is_file() or sha(path) != expected:
        raise RuntimeError(f"Stage39 frozen manifest drifted: {path}")
if not calibration.get("passed"):
    raise RuntimeError("Stage39 frozen Stage25 calibration did not pass")
print(entry["path"])
print(bucket)
print(calibration["residual_margin"])
print(calibration["risk_threshold"])
print(calibration["ood_threshold"])
PYLOCK
)
[[ "${#LOCKED[@]}" -eq 5 ]] || { echo "failed Stage39 manifest resolution"; exit 2; }
TRAIN_MANIFEST="${LOCKED[0]}"
BUCKET_MANIFEST="${LOCKED[1]}"
MARGIN="${LOCKED[2]}"
RISK="${LOCKED[3]}"
OOD="${LOCKED[4]}"

MAX_EPOCHS=4
LIMIT_BATCHES=1.0
CHECKPOINT_EVERY=48
EXTRA_ARGS=()
SELECTED_STEP=
SELECTION="${STAGE39_SELECTION:-$ROOT_DIR/artifacts/grpo_stage39/pilot/selection.json}"
if [[ "$PHASE" == audit ]]; then
  MAX_EPOCHS=1
  LIMIT_BATCHES=8
  CHECKPOINT_EVERY=1
  EXTRA_ARGS+=(+trainer.params.max_steps=1)
elif [[ "$PHASE" == formal ]]; then
  [[ -f "$SELECTION" ]] || { echo "Stage39 formal requires pilot selection: $SELECTION"; exit 2; }
  SELECTED_STEP=$("$PYTHON_BIN" - "$SELECTION" <<'PYSELECT'
import json,sys
p=json.load(open(sys.argv[1])); step=int(p.get('selected_step',-1))
assert p.get('passed') is True and step in (48,96,192)
print(step)
PYSELECT
  )
  CHECKPOINT_EVERY="$SELECTED_STEP"
  EXTRA_ARGS+=(+trainer.params.max_steps="$SELECTED_STEP")
fi
SEED=$((203900 + HOLDOUT + BRANCH_OFFSET))
TRAIN_ARGS=(
  agent=diffusiondrive_agent
  train_test_split=trainval
  split=trainval
  cache_path="$TRAIN_CACHE"
  use_cache_without_dataset=true
  force_cache_computation=false
  seed="$SEED"
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
  +trainer.params.use_distributed_sampler=false
  +trainer.params.enable_progress_bar=false
  +trainer.params.log_every_n_steps=1
  agent.checkpoint_path="$PUBLIC"
  agent.reference_checkpoint_path="$PUBLIC"
  agent.lr=1e-6
  agent.config.grpo_training_mode="$TRAINING_MODE"
  agent.config.generation_policy_algorithm=diffgrpo_non_destructive_challenger
  agent.config.grpo_decoder_gradient_scope=all_layers
  agent.config.grpo_decoder_layer0_lr_mult=0.1
  agent.config.inference_selector_source=trajectory_relative_harm_v3
  agent.config.stage25_selector_checkpoint_path="$SELECTOR"
  agent.config.stage25_selector_calibration_path="$CALIBRATION"
  agent.config.stage24_selector_residual_margin="$MARGIN"
  agent.config.stage25_selector_risk_threshold="$RISK"
  agent.config.stage24_selector_ood_threshold="$OOD"
  agent.config.stage23_generator_train_manifest_path="$TRAIN_MANIFEST"
  agent.config.stage39_bucket_manifest_path="$BUCKET_MANIFEST"
  agent.config.stage39_plan_sha256="$PLAN_SHA"
  agent.config.stage39_objective_revision=non_destructive_challenger_set_grpo_v1
  agent.config.stage39_public_candidate_count=20
  agent.config.stage39_challenger_candidate_count=20
  agent.config.stage39_group_size=8
  agent.config.stage39_elite_width=5
  agent.config.stage39_positive_margin=0.001
  agent.config.stage39_advantage_scale=0.02
  agent.config.stage39_advantage_clip=2.0
  agent.config.stage39_std_floor=0.0001
  agent.config.stage39_kl_weight=0.1
  agent.config.stage39_step_discount=0.6
  agent.config.stage39_optimizer_steps_per_epoch=48
  agent.config.stage39_gradient_accumulation=8
  agent.config.stage39_global_bucket_composition=[2,30,8,24]
  agent.config.stage39_candidate_source=public20
  agent.config.stage39_challenger_noise_offset=390001
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
  agent.config.diffgrpo_base_advantage_clip=2.0
  agent.config.diffgrpo_safety_regression_tolerance=0.000001
  agent.config.generation_advantage_mode=stage39_branch_objective
  agent.config.grpo_rollouts_per_mode=1
  agent.config.generation_mode_weighting=uniform
  agent.config.generation_trust_projection_mode=none
  agent.config.grpo_old_policy_sync_steps=32
  agent.config.grpo_priority_manifest_path=
  agent.config.grpo_priority_sample_fraction=0.0
  agent.config.weight_decay=0.0
  agent.config.diffusion_truncation_timestep=32
  agent.config.diffusion_roll_timesteps=[32,24,16,8,0]
  agent.config.diffusion_scheduler_num_inference_steps=125
  agent.config.grpo_checkpoint_save_top_k=-1
  agent.config.grpo_checkpoint_every_n_train_steps="$CHECKPOINT_EVERY"
  experiment_name="$EXPERIMENT"
)

cd "$ROOT_DIR"
if [[ "${GRPO_STAGE39_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" "$TRAIN_ENTRY" --cfg job --resolve "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}" >/dev/null
  echo "PASS Stage39 preflight phase=$PHASE branch=$BRANCH holdout=$HOLDOUT"
  echo "Repository: $ROOT_DIR"
  echo "Training entry: $TRAIN_ENTRY"
  echo "Public/current initializer: $PUBLIC"
  echo "Train manifest: $TRAIN_MANIFEST"
  echo "Bucket manifest: $BUCKET_MANIFEST"
  echo "Experiment root: $EXP_ROOT/$EXPERIMENT"
  exit 0
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" "$TRAIN_ENTRY" "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}"

ARTIFACT_DIR="$ROOT_DIR/artifacts/grpo_stage39/$PHASE/training/$BRANCH/fold${HOLDOUT}"
FREEZE="$ARTIFACT_DIR/checkpoints.json"
AUDIT="$ARTIFACT_DIR/audit.json"
[[ ! -e "$FREEZE" && ! -e "$AUDIT" ]] || {
  echo "refusing to overwrite Stage39 artifacts in $ARTIFACT_DIR"; exit 2;
}
FREEZE_ARGS=(--phase "$PHASE" --branch "$BRANCH" --holdout "$HOLDOUT" --experiment-root "$EXP_ROOT/$EXPERIMENT" --output "$FREEZE")
[[ -n "$SELECTED_STEP" ]] && FREEZE_ARGS+=(--selected-step "$SELECTED_STEP")
"$PYTHON_BIN" "$FREEZER" "${FREEZE_ARGS[@]}"
"$PYTHON_BIN" "$AUDITOR" --phase "$PHASE" --branch "$BRANCH" --holdout "$HOLDOUT" --base "$PUBLIC" --selector "$SELECTOR" --calibration "$CALIBRATION" --cv-freeze "$CV_FREEZE" --plan "$PLAN" --freeze "$FREEZE" --output "$AUDIT"

echo "PASS Stage39 training phase=$PHASE branch=$BRANCH holdout=$HOLDOUT"
echo "Run: $EXP_ROOT/$EXPERIMENT"
echo "Checkpoint freeze: $FREEZE"
echo "Audit: $AUDIT"
