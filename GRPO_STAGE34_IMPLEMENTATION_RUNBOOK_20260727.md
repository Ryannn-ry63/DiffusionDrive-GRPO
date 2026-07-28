# Stage34 MAF-GRPO Implementation and Runbook

Date: 2026-07-27

## Current status

Stage34 implements Mode-Aligned Deployable-Frontier GRPO (MAF-GRPO) on the
released `diffusiondrive_navsim_88p1_PDMS` checkpoint.  The selector is the
frozen Stage27 S-multi selector; it is not trained in Stage34.

The fold0 real eight-GPU one-step audit passed:

- candidate-level 20-mode trace: passed;
- current/public mode index alignment: passed;
- common-noise sampled and replayed chain count: exactly 20;
- finite PDM rewards, advantages, and diffusion log probabilities: passed;
- active decoder gradients: passed;
- frozen-parameter gradients: exactly zero;
- public/reference generator: bitwise frozen;
- Stage27 selector: bitwise frozen;
- changed parameters: exactly 64 allowed decoder tensors;
- optimizer groups: layer0 `1e-7`, layer1 `1e-6`, 32 tensors each;
- Stage34 global bucket sampler: exact.

Evidence:

- audit:
  `artifacts/grpo_stage34/pilot/training/MAF/fold0/audit/audit.json`
- checkpoint freeze:
  `artifacts/grpo_stage34/pilot/training/MAF/fold0/audit/checkpoints.json`
- run:
  `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage34_maf_runtime_f0_audit_r2`

The audit initially exposed an implementation bug where the model-side bucket
lookup still used the Stage30 config field.  Stage34 now explicitly resolves
`stage34_bucket_manifest_path`, and a regression test locks this behavior.

## Frozen scientific boundary

The method and promotion rules are frozen in
`GRPO_STAGE34_MODE_ALIGNED_FRONTIER_PLAN_20260727.md`, SHA256
`ad3e9dace53cd3196ef236c7d3630605e64aa46c908e5dd18c81b8600eaf155a`.

Stage34 changes only the diffusion generator.  It trains same-anchor paired
PDM deltas from a complete 20-mode current/public bank under identical
diffusion noise.  Credit is restricted to deployment owners, public top-five
frontier modes, and safe headroom modes, with asymmetric tail, safety, and
mature-scene protection.

Stage33 remains the negative selected-only ablation.  It must not be silently
extended or mixed into Stage34.

## Formal pilot training

Submit these as two independent eight-H100 jobs.  The scripts are
self-locating, so `cd` is not required.

Fold0 reuses the frozen passing audit and starts formal training:

```bash
/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/run_stage34_mode_aligned_frontier_h100.sh \
  0 stage34_maf_release_f0
```

Fold1 runs its own one-step audit and automatically starts formal only after
that audit passes:

```bash
/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/run_stage34_mode_aligned_frontier_h100.sh \
  1 stage34_maf_release_f1
```

Each formal job trains four epochs and freezes checkpoints at optimizer steps
`48, 96, 144, 192`.  Expected formal artifacts:

```text
artifacts/grpo_stage34/pilot/training/MAF/fold0/formal/checkpoints.json
artifacts/grpo_stage34/pilot/training/MAF/fold0/formal/audit.json
artifacts/grpo_stage34/pilot/training/MAF/fold1/formal/checkpoints.json
artifacts/grpo_stage34/pilot/training/MAF/fold1/formal/audit.json
```

Do not start evaluation unless both formal audits report `"passed": true`.

## Two-fold holdout evaluation

After both formal jobs pass, submit two independent eight-H100 evaluation
jobs:

```bash
/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/scripts/evaluation/run_diffusiondrive_grpo_stage34_eval_h100.sh 0
```

```bash
/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/scripts/evaluation/run_diffusiondrive_grpo_stage34_eval_h100.sh 1
```

Each fold evaluates Public, DPEL192, and MAF steps 48/96/144/192 under noise
namespaces `20261511` and `20261512`.

When both folds finish:

```bash
/root/miniconda3/envs/navsimH100/bin/python \
  /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/scripts/evaluation/summarize_grpo_stage34_pilot.py \
  --eval-root /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage34/pilot/eval \
  --bucket-manifest /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage30/manifests/buckets_6119.json \
  --output /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage34/pilot/summary.json
```

The summarizer is fail-closed on checkpoint, selector, calibration, noise,
token count, and schedule provenance.

## Decision after pilot

Promote a checkpoint only if every frozen gate passes, including selected
gain at least `+0.0015`, positive fold/noise means and whole-log CI, gain over
DPEL192 at least `+0.0005`, positive safe-deployable ceiling, preserved public
top-five/raw oracle, and no safety or catastrophic regression.

If no checkpoint improves the safe-deployable oracle, stop Stage34.  More
epochs or a selector change are not authorized by this pilot.  If a checkpoint
passes, expand that fixed checkpoint and objective to four-fold OOF; NavTest
remains blocked until the four-fold gate passes.

## Verification completed

- Stage30-Stage34 focused regression suite: `47 passed`;
- shell syntax checks: passed;
- fold0 absolute-path formal preflight: passed;
- fold1 absolute-path audit-to-formal chain preflight: passed;
- fold0 real eight-GPU optimizer-step and freeze audit: passed.
