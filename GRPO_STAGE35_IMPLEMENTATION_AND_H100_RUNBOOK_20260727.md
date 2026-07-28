# Stage35 NCD-GRPO implementation and H100 runbook

Date: 2026-07-27

Frozen scientific plan:

- `GRPO_STAGE35_NESTED_COUNTERFACTUAL_DEPLOYMENT_PLAN_20260727.md`
- SHA256: `301e5fb37066c34f6a3e224be08fd1ca435cc9a8d961121869cf6dcd21d2fae7`

## What Stage35 changes

Stage35 keeps the Stage31 DPEL bank-level objective as the outer GRPO
signal and adds an inner, same-anchor counterfactual deployment signal.
For one active candidate, the candidate is replaced by the same anchor
sampled in another bank, the frozen S-multi selector is rerun, and the
change in final deployed PDM reward is assigned back to that candidate.

The current policy initializes from the fold-matched frozen Stage32
DPEL192 checkpoint. The GRPO reference remains the official public-88.1
checkpoint. The selector, calibration, perception stack, classification
heads, and all non-regression decoder parameters remain frozen.

This is not a copy of DiffusionDriveV2: the Stage35 contribution is
deployment-path credit assignment under a frozen selector, with
same-anchor counterfactual normalization and a public-reference safety
gate.

## Locked inputs

- public reference SHA256:
  `008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b`
- S-multi selector SHA256:
  `023d7b6b77bb8fa2dc3e779849f3716688abf0796b019416494c80b29615b691`
- selector calibration SHA256:
  `fec2a10573e913e4452f8eea92e6334fcd5e6f1104e6244876770973a1f41990`
- fold0 DPEL192 initializer SHA256:
  `2c01b93fb35b246b6c897db149f38c13cc4a4259933f8ef21aee4d311dfee951`
- fold1 DPEL192 initializer SHA256:
  `b2bf4243e622235de4003938fba85240d2fd660ad0a7123b750cbc83abf6a557`

The launcher resolves the DPEL checkpoint path from the Stage32 freeze
JSON and verifies stage, branch, fold, global step, recorded SHA, actual
file SHA, and the fold-specific allow-list. No checkpoint path is typed
by hand.

## Training gate

Each fold runs a one-optimizer-step audit before formal training.
Formal training starts only if all of the following pass:

- 320 sampled chains per scene: current/public `8 x 20`;
- 64 replayed chains per scene: current/public `8 x 4`;
- 224 same-anchor counterfactual selector reruns per scene;
- counterfactual nonzero fraction at least `0.10`;
- scenes with non-selected active-mode influence at least `0.25`;
- exactly 64 trainable regression tensors, split `32 + 32` by layer;
- active finite gradients on all allowed decoder groups;
- zero gradients on selector, perception, and classification modules;
- bitwise-frozen public reference and S-multi selector;
- finite rewards, advantages, log probabilities, optimizer state.

Formal schedule is four epochs, 48 optimizer steps per epoch, with
checkpoints at steps `48`, `96`, `144`, and `192`.

## H100 commands

The two folds are independent and may be submitted as two separate
8-H100 jobs:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage35_nested_counterfactual_h100.sh 0
```

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage35_nested_counterfactual_h100.sh 1
```

Each command is queue-safe: it runs audit first and automatically runs
formal only after the audit JSON passes. If the counterfactual signal is
too weak, the job exits before spending the formal-training budget.

Expected artifacts:

```text
artifacts/grpo_stage35/pilot/training/NCD/fold0/{audit,formal}/
artifacts/grpo_stage35/pilot/training/NCD/fold1/{audit,formal}/
```

## OOF PDM evaluation

After both formal jobs pass, submit two independent 8-H100 evaluation
jobs:

```bash
bash ./scripts/evaluation/run_diffusiondrive_grpo_stage35_eval_h100.sh 0
bash ./scripts/evaluation/run_diffusiondrive_grpo_stage35_eval_h100.sh 1
```

Each job evaluates only the eight new NCD cells: four checkpoints times
two frozen noise namespaces. Public and DPEL192 controls are reused from
the exact Stage34 paired artifacts; their checkpoint, reference,
selector, calibration, token order, domain, and input JSON SHA are
revalidated.

Then summarize:

```bash
/root/miniconda3/envs/navsimH100/bin/python \
  scripts/evaluation/summarize_grpo_stage35_pilot.py \
  --eval-root artifacts/grpo_stage35/pilot/eval \
  --baseline-eval-root artifacts/grpo_stage34/pilot/eval \
  --bucket-manifest artifacts/grpo_stage30/manifests/buckets_6119.json \
  --output artifacts/grpo_stage35/pilot/summary.json
```

## Frozen promotion rule

A checkpoint is promoted only if it satisfies every gate:

- selected PDM gain over public-88.1 at least `+0.0015`;
- selected PDM gain over fold-matched DPEL192 at least `+0.0005`;
- both fold means and both noise-namespace means are positive;
- whole-log bootstrap 95% CI lower bound is positive;
- safe-deployable oracle gain at least `+0.0005`;
- raw oracle and public-top5 candidate means are nonnegative;
- mature-scene selected delta at least `-0.0001`;
- collision, drivable, and TTC component deltas each at least `-0.0005`;
- trimmed mean nonnegative, wins exceed losses, and no delta at most
  `-0.5`.

No navtest is started from a merely interesting checkpoint. A navtest
candidate must first pass this two-fold, two-noise OOF gate.

## Local validation completed

- Stage35 targeted tests: 13 passed.
- Stage31/32/34/35 cross-stage regression tests: 40 passed.
- Python compilation and Bash syntax checks passed.
- Both fold0 and fold1 audit/formal Hydra preflights passed.
