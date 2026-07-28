#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 audit|formal public|multi STAGE24_CHECKPOINT EXPERIMENT_NAME GPU"
  exit 2
fi
PHASE="$1"
BRANCH="$2"
STAGE24_CHECKPOINT="$3"
EXPERIMENT_NAME="$4"
GPU="$5"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
PUBLIC=/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS
TRAINING_CACHE="${NAVSIM_TRAINING_CACHE:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache}"
MANIFEST="$ROOT_DIR/artifacts/grpo_stage23/manifests/selector_fit_manifest.json"
OLD_DIR="$ROOT_DIR/artifacts/grpo_stage26/candidate_bank"
PUBLIC_DIR="$ROOT_DIR/artifacts/grpo_stage27/candidate_bank"

[[ "$PHASE" == audit || "$PHASE" == formal ]] || { echo "invalid phase"; exit 2; }
[[ "$BRANCH" == public || "$BRANCH" == multi ]] || { echo "invalid branch"; exit 2; }
[[ "$EXPERIMENT_NAME" =~ ^stage27_selector_(public|multi)_stage25_(audit|formal)_[A-Za-z0-9_.-]+$ ]] || {
  echo "invalid experiment name"
  exit 2
}
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "GPU must be 0..7"; exit 2; }

LABELS=(default ns20260811 ns20260812)
BANKS=()
if [[ "$BRANCH" == multi ]]; then
  for domain in official_base stage16_epoch8 stage19_epoch8 stage21_epoch8 stage25_epoch2; do
    for label in "${LABELS[@]}"; do
      BANKS+=("$OLD_DIR/${domain}_${label}.json")
    done
  done
  AUDIT="$ROOT_DIR/artifacts/grpo_stage27/audits/selector_banks_multi.json"
  AUDIT_SHA=39e37419c25f9cd89ffe20ce079ca97fe7055d049de6072592e9d5f136ae74fd
  FORMAL_EPOCHS=18
  SEED=27125
else
  AUDIT="$ROOT_DIR/artifacts/grpo_stage27/audits/selector_banks_public.json"
  AUDIT_SHA=4071409175ff5f69c8c85c62ab49a912b7a64d8172e4772fb07a81bcfdae71e1
  FORMAL_EPOCHS=3
  SEED=27025
fi
for label in "${LABELS[@]}"; do
  BANKS+=("$PUBLIC_DIR/public88_base_${label}.json")
done

for path in "$PUBLIC" "$STAGE24_CHECKPOINT" "$TRAINING_CACHE" "$MANIFEST" \
  "$AUDIT" "${BANKS[@]}"; do
  [[ -e "$path" ]] || { echo "missing locked Stage27 input: $path"; exit 2; }
done
"$PYTHON_BIN" -c '
import hashlib,json,pathlib,sys,torch
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
branch,audit_sha,public,stage24,manifest,audit,*banks=sys.argv[1:]
assert sha(public) == "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
assert sha(manifest) == "cefe6cdc470f9e14a5244ba422d7ff7265eed7769d4147189d705c1df1604f7f"
assert sha(audit) == audit_sha
p=json.load(open(audit))
assert p["passed"] and p["stage"] == 27 and p["branch"] == branch
assert p["num_artifacts"] == len(banks)
for expected,path in zip(p["artifacts"],banks):
    assert pathlib.Path(expected["path"]).resolve() == pathlib.Path(path).resolve()
    assert expected["sha256"] == sha(path)
c=torch.load(stage24,map_location="cpu")
assert "state_dict" in c and int(c.get("global_step",-1)) > 0
' "$BRANCH" "$AUDIT_SHA" "$PUBLIC" "$STAGE24_CHECKPOINT" "$MANIFEST" "$AUDIT" "${BANKS[@]}"

MAX_EPOCHS="$FORMAL_EPOCHS"
LIMIT_TRAIN=1.0
STEPS_PER_EPOCH=2038
EXTRA_ARGS=()
if [[ "$PHASE" == audit ]]; then
  MAX_EPOCHS=1
  LIMIT_TRAIN=8
  STEPS_PER_EPOCH=8
  EXTRA_ARGS+=(+trainer.params.max_steps=8)
fi
BANK_LIST="[$(IFS=,; echo "${BANKS[*]}")]"

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" navsim/planning/script/run_training.py \
  agent=diffusiondrive_agent train_test_split=trainval \
  cache_path="$TRAINING_CACHE" use_cache_without_dataset=true force_cache_computation=false \
  seed="$SEED" dataloader.params.batch_size=2 dataloader.params.num_workers=4 \
  trainer.params.max_epochs="$MAX_EPOCHS" trainer.params.limit_train_batches="$LIMIT_TRAIN" \
  trainer.params.limit_val_batches=0 trainer.params.accumulate_grad_batches=1 \
  trainer.params.accelerator=gpu trainer.params.strategy=auto trainer.params.precision=32-true \
  +trainer.params.devices=1 +trainer.params.enable_progress_bar=false +trainer.params.log_every_n_steps=1 \
  agent.checkpoint_path="$STAGE24_CHECKPOINT" agent.reference_checkpoint_path="$PUBLIC" \
  agent.lr=1e-4 agent.config.grpo_training_mode=stage25_relative_harm_selector \
  agent.config.stage24_selector_train_manifest_path="$MANIFEST" \
  "agent.config.stage24_candidate_bank_paths=$BANK_LIST" \
  agent.config.stage24_selector_num_members=8 agent.config.stage24_selector_num_modes=20 \
  agent.config.stage24_selector_dim=128 agent.config.stage24_selector_steps_per_epoch="$STEPS_PER_EPOCH" \
  agent.config.stage24_selector_subbag_fraction=0.8 \
  agent.config.stage24_selector_pair_reward_gap=0.005 \
  agent.config.stage24_selector_focal_gamma=2.0 \
  agent.config.stage24_selector_hard_negative_count=4 \
  agent.config.stage25_selector_focal_gamma=2.0 \
  agent.config.stage25_selector_hard_negative_count=4 \
  agent.config.stage25_selector_hard_risk_margin=0.9 \
  agent.config.grpo_checkpoint_save_top_k=-1 \
  agent.config.diffusion_truncation_timestep=32 \
  agent.config.diffusion_roll_timesteps='[32,24,16,8,0]' \
  agent.config.diffusion_scheduler_num_inference_steps=125 \
  experiment_name="$EXPERIMENT_NAME" "${EXTRA_ARGS[@]}"
