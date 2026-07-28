# Stage28 Implementation and H100 Runbook

## Current status

Stage28 implementation is complete through the A/B/C pilot training and
fold4 selection gate. The method remains centered on GRPO improvement over
the public 88.1 DiffusionDrive checkpoint; NavTest is not inspected during
method or epoch selection.

Implemented branches:

- A: unchanged Stage27 group-relative objective with safe S-multi;
- B: absolute paired-uplift objective with safe S-multi;
- C: absolute paired-uplift objective with training-only S-public.

Every fold4 evaluation is forced to use safe S-multi. S-public cannot run in
model eval mode and requires an exact SHA-locked training authorization.

## Completed verification

- Stage20--28 regression suite: `64 passed`.
- Stage28 objective tests cover all-worse, positive, neutral,
  mature-negative, safety override, non-finite, and shape failures.
- A, B, and C each passed a real one-optimizer-step 8x RTX 4090 audit.
- Every audit changed exactly 64 decoder tensors, 32 per refinement layer.
- Frozen reference, perception, classification, and selector gradients were
  exactly zero.
- The branch-specific training selector remained bitwise frozen.
- All reward, advantage, log-probability, KL, and gradient diagnostics were
  finite.

Frozen pilot input:

`artifacts/grpo_stage28/pilot/inputs.json`

SHA256:

`489dd645c718fc43de1790e181e9473b36a76aa03c673ac41e22f5d6f4419d9c`

## Submit three independent 8x H100 jobs

Each command below is one independent eight-GPU task. They should be
submitted in parallel, not run sequentially inside one allocation.

Branch A:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage28_pilot_branch_h100.sh formal A
```

Branch B:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage28_pilot_branch_h100.sh formal B
```

Branch C:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage28_pilot_branch_h100.sh formal C
```

Each job trains four epochs on the same ordered 4,075 folds0--3 tokens and
freezes checkpoints at optimizer steps 64, 128, 192, and 256. A successful
job ends with both `Checkpoint freeze:` and `Audit:` paths. Expected running
time is roughly 1.5--2.5 hours per branch on 8x H100, with all three branches
finishing in that wall-clock range when submitted concurrently.

## Prepare and run fold4 after all three succeed

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./prepare_stage28_fold4.sh
bash ./run_stage28_fold4_8gpu.sh
```

The first command refuses to proceed unless all three four-epoch checkpoint
audits pass. The second runs 26 cells (public baseline plus 12 checkpoints,
each under two noise namespaces) in four waves across eight GPUs, then applies
the frozen paper-level statistical gate automatically.

The final report will be:

`artifacts/grpo_stage28/pilot/fold4/report.json`

If no checkpoint passes, the script exits nonzero and Stage28 stops before
fold5 and NavTest. If one passes, the report freezes the selected branch and
epoch for the folds0--4 retraining stage.
