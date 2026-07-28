# Stage31 Implementation Handoff

## Status

The frozen Stage31 selector-consistent decision-level GRPO plan has been
implemented. The frozen plan is
`GRPO_STAGE31_SELECTOR_CONSISTENT_DECISION_GRPO_PLAN_20260725.md` with SHA256
`3ff3d3bcd3120b9d73a017876f942458eb3fa16651517e4e30b27cf2fff7bb0d`.

The single-step DPF audit and all eight formal DP/DPF four-fold jobs passed on
2026-07-25. No Stage31 OOF performance result exists yet. The next action is
the eight held-out evaluation jobs.

## Optimizer-step audit result

- Audit artifact:
  `artifacts/grpo_stage31/cv/training/DPF/fold0/audit/audit.json`.
- Audited checkpoint SHA256:
  `99e6d6c0cb6aef407419356f9621c06ee3c89133d44699b1383d3bd45ae72f58`.
- Exactly 64 allowed tensors changed: 32 in decoder layer 0 and 32 in layer
  1; zero forbidden tensors changed.
- Public reference and frozen S-multi remained bitwise equal; the exact global
  bucket sampler passed.
- At the exact public restart, deployment delta, selector disagreement, and KL
  were all zero as required. DPF tie bootstrap activated 100% of scenes, with
  50% positive and 50% negative rollout advantages. Decoder gradient norm was
  finite and nonzero (`0.2176836`); safety override fraction was zero.

## Implemented scientific boundary

- Current and released-public generators sample eight complete 20-mode banks
  with the same initial, transition, and final noise.
- Frozen Stage27 S-multi independently selects the current and public bank.
- The reward is the exact selected-current minus selected-public PDMS.
- Current selected chains provide the policy log probabilities; independently
  selected public chains provide the BC teacher; KL is exact on current states.
- `DP` is a non-selectable paired-deployment ablation.
- `DPF` is the only selectable branch. Its headroom rank term is also the only
  GRPO bootstrap on exact public ties; DP ties remain zero.
- Only the registered 64 decoder tensors train. Public reference, perception,
  classification, and S-multi remain frozen.
- Stage30 four-fold manifests and `2/30/8/24` global bucket sampling are reused
  bitwise; steps are frozen at 48/96/144/192.

## Verification already completed

- Stage31 unit and gate tests: `10 passed`.
- Stage29--31 regression selection: `36 passed`.
- Python compilation passed for every modified core and Stage31 evaluation
  module.
- `bash -n` passed for all Stage31 launchers.
- Hydra resolved successfully for the audit and all eight formal branch/fold
  combinations.
- Every launcher uses the exact repository path containing
  `wangcaojun-240208020180`; referenced public checkpoint, selector,
  calibration, CV manifests, and plan hash are checked before training.

## H100 execution order

All commands start from the same exact repository path:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
```

### 1. One optimizer-step audit (completed)

Run this as one 8xH100 task:

```bash
bash ./run_stage31_audit_h100.sh
```

Completed with
`PASS Stage31 training phase=audit branch=DPF holdout=0`.

### 2. Two branches times four folds (completed)

After the audit passes, submit eight separate 8xH100 tasks. Each task receives
one distinct index:

```bash
bash ./run_stage31_formal_array_h100.sh 0
bash ./run_stage31_formal_array_h100.sh 1
bash ./run_stage31_formal_array_h100.sh 2
bash ./run_stage31_formal_array_h100.sh 3
bash ./run_stage31_formal_array_h100.sh 4
bash ./run_stage31_formal_array_h100.sh 5
bash ./run_stage31_formal_array_h100.sh 6
bash ./run_stage31_formal_array_h100.sh 7
```

Indices 0--3 are DP folds0--3; indices 4--7 are DPF folds0--3. Each job uses
all eight GPUs and trains four epochs (192 optimizer steps). Do not run several
indices simultaneously on the same physical 8-GPU node.

All eight formal audits passed. Every fold froze steps 48/96/144/192; every
checkpoint changed exactly 64 allowed decoder tensors and zero forbidden
tensors. Run directories are:

- DP fold0: `exp/stage31_cv_DP_fold0_formal_seed203100/2026.07.25.16.35.22`
- DP fold1: `exp/stage31_cv_DP_fold1_formal_seed203101/2026.07.25.16.58.55`
- DP fold2: `exp/stage31_cv_DP_fold2_formal_seed203102/2026.07.25.16.59.33`
- DP fold3: `exp/stage31_cv_DP_fold3_formal_seed203103/2026.07.25.16.59.31`
- DPF fold0: `exp/stage31_cv_DPF_fold0_formal_seed203100/2026.07.25.17.00.15`
- DPF fold1: `exp/stage31_cv_DPF_fold1_formal_seed203101/2026.07.25.17.00.58`
- DPF fold2: `exp/stage31_cv_DPF_fold2_formal_seed203102/2026.07.25.17.01.46`
- DPF fold3: `exp/stage31_cv_DPF_fold3_formal_seed203103/2026.07.25.17.02.46`

OOF evaluation preflight passed for both branches on all four folds and for
the public-reference cells. The actual 72 held-out evaluation artifacts are
under `artifacts/grpo_stage31/cv/eval`.

### 4. Frozen OOF result (completed; stopped before confirmation)

The frozen gate report is
`artifacts/grpo_stage31/cv/report.json`. Artifact provenance and all numeric
checks were evaluated successfully, but no selectable DPF checkpoint passed
the complete gate, so fold4/fold5 confirmation and NavTest were intentionally
not launched.

| branch | step | pooled delta | 95% CI lower | hard-scene delta | mature delta | candidate mean | fallback oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| DPF | 48 | +0.000203 | -0.000301 | +0.001507 | -0.000050 | +0.000432 | +0.000676 |
| DPF | 96 | +0.000581 | +0.000056 | +0.002851 | +0.000094 | +0.000663 | +0.001046 |
| DPF | 144 | +0.000538 | +0.000020 | +0.003069 | +0.000088 | +0.000778 | +0.001061 |
| DPF | 192 | **+0.000918** | **+0.000262** | **+0.005098** | **+0.000042** | **+0.000882** | **+0.001388** |

DP is a non-selectable ablation; its best pooled delta was only `+0.000117`
and its whole-log confidence intervals remained non-positive. DPF192 is the
best Stage31 checkpoint and is positive on all four folds, both namespaces,
and all guarded components, but it misses the frozen pooled `+0.003`, hard
scene `+0.010`, and fallback-oracle `+0.004` requirements. This is a real
modest generalization gain, not a gate or provenance failure; the protocol
therefore stops here without changing thresholds after seeing the result.

### 3. Held-out OOF evaluation

After all formal jobs pass their checkpoint audits, submit the same eight-index
layout using:

```bash
bash ./run_stage31_eval_array_h100.sh INDEX
```

Run `INDEX=0,1,...,7` as eight separate 8xH100 tasks. DP evaluates eight
checkpoint/namespace cells in one wave. DPF evaluates those eight cells plus
the two public reference cells in a second wave.

### 5. Frozen OOF gate (completed)

After all eight evaluation jobs pass, run once in the repository:

```bash
bash ./run_stage31_cv_gate.sh
```

The report was written to
`artifacts/grpo_stage31/cv/report.json`. It exits nonzero intentionally when no
DPF step passes; that is the current result. No fold4/fold5 confirmation script
should be launched from this report.

## Expected artifact roots

- Training audits:
  `artifacts/grpo_stage31/cv/training/{DP,DPF}/fold{0..3}/{audit,formal}`
- OOF evaluations:
  `artifacts/grpo_stage31/cv/eval/{DP,DPF}/fold{0..3}`
- H100 logs: `artifacts/grpo_stage31/h100_logs`
- Gate report: `artifacts/grpo_stage31/cv/report.json`

Never delete or overwrite a partial Stage31 artifact directory to make a
launcher continue. Inspect the corresponding log first; the launchers fail
closed on existing output by design.
