# Stage27 Phase3 Implementation and Runbook

## Status

Phase3 follows the frozen
`GRPO_STAGE27_STAGE26_ALIGNED_EXECUTION_PLAN_20260724.md`. The frozen plan is
unchanged, and training restarts from the public-88.1 checkpoint rather than
continuing a Stage26 generator.

Implemented and locally verified on 2026-07-24:

- formal mode `stage27_public_diffgrpo_selected_set`;
- a Stage27-only public checkpoint SHA guard, leaving the Stage16--26 SHA
  constant unchanged;
- folds0--4 manifest with 5096 tokens from 757 whole logs;
- frozen Phase3 input audit;
- H100 training entry, checkpoint freezer, and weight/gradient auditor.

The frozen input audit is:

`artifacts/grpo_stage27/audits/generator_phase3_inputs.json`

Its SHA256 is:

`ea3a9d173d37bd22e3b91d6a4ea3f857edf94afbba63e9dcfbe94febbe98ee89`

## One-step H100 gate

The audit uses eight H100s, batch size 1 per rank, gradient accumulation 8,
and exactly one optimizer step. It verifies:

- current/reference both start from public-88.1;
- the reference remains bitwise equal to the public decoder;
- only 64 registered decoder tensors change, 32 in each decoder layer;
- the Stage24/25 selector remains bitwise equal to selected S-multi;
- perception, selector, classification, and paired-risk gradients are zero;
- reward, advantage, log-probability, loss, gradient, and optimizer state are
  finite;
- the two 32-tensor optimizer groups use learning rates `1e-6` and `1e-7`.

Run:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage27_public88_generator_phase3_h100.sh audit
```

Expected output:

`artifacts/grpo_stage27/generator/phase3/audit/audit.json`

## Provisional two-epoch training

The formal branch is blocked unless the one-step audit is present and passing.
After independently reviewing the audit, run:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage27_public88_generator_phase3_h100.sh formal
```

This branch is fixed to folds0--4, 5096 scenes, two epochs, 80 optimizer steps
per epoch. It freezes both epoch-1/step-80 and epoch-2/step-160 checkpoints.
No epoch extension and no NavTest are authorized before the fold5 Phase4 gate.

## Local validation

- Python and shell syntax checks passed.
- `git diff --check` passed.
- Stage23/25/27 unit tests: `21 passed`.
- Frozen input re-audit passed.
- The checkpoint freezer replayed a completed Stage26 two-epoch run and
  correctly resolved step 80 and step 160.
