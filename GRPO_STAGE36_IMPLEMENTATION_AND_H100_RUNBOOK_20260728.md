# Stage36 Implementation and H100 Runbook

Date: 2026-07-28

## Outcome

Stage36 implements the frozen Reference-Gated Tail NCD-GRPO plan on the
released `diffusiondrive_navsim_88p1_PDMS` baseline.

The method keeps Stage35 nested counterfactual deployment credit and adds two
generator-side objectives:

1. public-frontier retention on regressed public top-five trajectories;
2. same-anchor top-two tail expansion when the current trajectory safely
   exceeds its paired public threshold.

The selector remains frozen. The trainable boundary remains the registered
64 diffusion-decoder tensors.

## Verified implementation

The fold0 real one-step audit ran on eight RTX 4090 GPUs and passed:

- reference-retention active fraction: `0.025000`;
- safe tail-expansion active fraction: `0.465625`;
- required minima: `0.01` and `0.005`;
- changed allowed tensors: `64`;
- changed forbidden tensors: `0`;
- decoder layers 0/1 changed tensors: `32/32`;
- reference and selector: bitwise frozen;
- all forbidden gradient groups: zero;
- current replay unique/physical rank means: `51.375/83.0`;
- public replay unique/physical rank means: `15.875/19.0`.

Frozen audit artifacts:

- `artifacts/grpo_stage36/pilot/training/RGT/fold0/audit/checkpoints.json`
- `artifacts/grpo_stage36/pilot/training/RGT/fold0/audit/audit.json`

The Stage36 and Stage35 regression suite has `27 passed`.

## Formal H100 training

The two folds are independent and may be submitted as two concurrent 8-H100
jobs. Each command is queue-safe: fold0 reuses the passing local audit; fold1
runs its own one-step audit and starts formal training only after it passes.

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage36_reference_gated_tail_h100.sh \
  0 stage36_rgt_pilot_f0 0,1,2,3,4,5,6,7
```

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage36_reference_gated_tail_h100.sh \
  1 stage36_rgt_pilot_f1 0,1,2,3,4,5,6,7
```

Formal training is fixed to four epochs and 192 optimizer steps. Checkpoints
are frozen at steps `48`, `96`, `144`, and `192`. Do not extend epochs or
change loss weights after seeing pilot results.

Expected formal audit files:

- `artifacts/grpo_stage36/pilot/training/RGT/fold0/formal/audit.json`
- `artifacts/grpo_stage36/pilot/training/RGT/fold1/formal/audit.json`

## Holdout PDM evaluation

After both formal audits pass, submit the two fold evaluations. Each job uses
eight H100 GPUs to evaluate four checkpoints under two frozen noise
namespaces.

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash scripts/evaluation/run_diffusiondrive_grpo_stage36_eval_h100.sh \
  0 0,1,2,3,4,5,6,7
```

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash scripts/evaluation/run_diffusiondrive_grpo_stage36_eval_h100.sh \
  1 0,1,2,3,4,5,6,7
```

Then create the paired two-fold summary:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
/root/miniconda3/envs/navsimH100/bin/python \
  scripts/evaluation/summarize_grpo_stage36_pilot.py \
  --eval-root artifacts/grpo_stage36/pilot/eval \
  --baseline-eval-root artifacts/grpo_stage34/pilot/eval \
  --ncd-eval-root artifacts/grpo_stage35/pilot/eval \
  --bucket-manifest artifacts/grpo_stage30/manifests/buckets_6119.json \
  --output artifacts/grpo_stage36/pilot/summary.json
```

## Frozen decision rule

Stage36 advances only when one checkpoint passes every frozen gate, including:

- selected gain versus Public at least `+0.0015`;
- gain versus DPEL192 at least `+0.0005`;
- gain versus NCD192 at least `+0.00015`;
- positive fold means, namespace means, and whole-log CI lower bound;
- nonnegative candidate mean, raw oracle, public top-five, and same-anchor
  top-two deltas;
- safe-oracle gain at least `+0.0005`;
- hard-scene gain at least `+0.0045`;
- mature-scene gain at least `-0.0001`;
- no component regression below `-0.0005` and no catastrophic regression.

If raw oracle or public top-five is negative for every checkpoint, stop this
branch and do not tune the selector. If generator ceiling gates pass but
selected gain fails, freeze the generator and move selector work to a later
independent stage. Navtest remains blocked until the unchanged four-fold
promotion gate passes.
