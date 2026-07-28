# Stage35 Formal Training Result (2026-07-27)

## Outcome

Stage35 NCD (Nested Counterfactual Deployment Credit) formal training completed
successfully on both pilot folds. The implementation and checkpoint-integrity
audits passed, so the checkpoints are eligible for out-of-fold PDM evaluation.

This is a **training/audit pass**, not yet evidence of a PDMS improvement.

## Formal runs

| Fold | Optimizer steps | Checkpoints | Audit |
|---|---:|---:|---|
| 0 | 192 | 48 / 96 / 144 / 192 | PASS |
| 1 | 192 | 48 / 96 / 144 / 192 | PASS |

Audit artifacts:

- `artifacts/grpo_stage35/pilot/training/NCD/fold0/formal/audit.json`
- `artifacts/grpo_stage35/pilot/training/NCD/fold1/formal/audit.json`

Integrity checks passed on both folds:

- active decoder gradients were finite;
- 64 allowed decoder tensors changed and no forbidden tensor changed;
- frozen reference and selector behavior remained bitwise identical;
- all frozen gradients were zero;
- checkpoint cadence and optimizer boundaries were correct.

## Counterfactual signal

| Fold | Counterfactual nonzero fraction | Non-owner influence | Required |
|---|---:|---:|---:|
| 0 | 0.5775 | 0.5000 | >= 0.10 / >= 0.25 |
| 1 | 0.5481 | 0.4688 | >= 0.10 / >= 0.25 |

The intended Stage35 signal is therefore active and materially stronger than
the audit minimum.

## Training-side diagnostic

The mean deployment delta was mixed:

- fold 0: `-0.002510`
- fold 1: `+0.000821`

This does not constitute the final estimator and must not be reported as PDMS.
It does show that there was no obvious collapse, while also warning against
promoting the final checkpoint without out-of-fold evaluation. All four
checkpoint times must be evaluated because the best generalizing point may be
earlier than step 192.

## Formal OOF PDM evaluation

Both commands passed real-path preflight for all 16 cells
(2 folds x 4 checkpoints x 2 noise namespaces). The evaluation output
directory is currently empty.

Run the following as two independent 8-H100 jobs:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./scripts/evaluation/run_diffusiondrive_grpo_stage35_eval_h100.sh 0
```

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./scripts/evaluation/run_diffusiondrive_grpo_stage35_eval_h100.sh 1
```

The evaluation compares Stage35 NCD against the frozen Stage34 public-88.1 and
DPEL controls. No selector retuning or navtest promotion is allowed before the
OOF summary passes the frozen gates.

