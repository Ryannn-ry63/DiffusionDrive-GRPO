# Stage32 formal training audit and SCF normalization correction

Date: 2026-07-26

## Completed runs

All four original 12-epoch pilot runs reached global step 576 and contain the
planned step-192, step-384, and step-576 checkpoints.  DPEL changed exactly the
64 permitted decoder tensors, changed no frozen tensor, and all checkpoint
tensors are finite.

The DPEL runs remain valid:

- `stage32_pilot_DPEL_fold0_formal_seed203200_envfix`
- `stage32_pilot_DPEL_fold1_formal_seed203201_envfix`

The original SCF runs are retained only as diagnostics and must not enter the
Stage32 pilot comparison or a paper claim:

- `stage32_pilot_SCF_fold0_formal_seed203200_envfix`
- `stage32_pilot_SCF_fold1_formal_seed203201_envfix`

## Why the original SCF runs are invalid

The SCF objective constructed `all_count`, but the BC and KL numerators were
not divided by it.  With 40 replay chains and five diffusion transitions, the
BC term was therefore summed across roughly 200 valid entries instead of being
mean-normalized.  TensorBoard exposed the error: DPEL BC loss stayed near
`-0.10`, while the original SCF BC loss ranged from roughly `-4` to `-21` and
dominated the frontier policy term.

The correction divides per-scene BC and KL sums by `all_count` before averaging
across valid scenes.  The corrected objective revision is frozen as
`mean_normalized_bc_kl_v2` in Hydra and the agent's formal-mode validation.

## Correction validation

- Stage30/31/32 focused regression suite: 25 tests passed.
- A new constant-input test verifies that BC and KL losses are independent of
  replay width and denoising-step count.
- Corrected 8-GPU SCF optimizer-step run exited successfully:
  `stage32_pilot_SCF_fold0_audit_seed203200_normfix`.
- Corrected one-step losses were: total `-0.10844`, policy `-0.00663`, BC
  `-0.10180`, KL `0.0`; decoder gradient norm was finite and positive.
- Formal SCF Hydra preflight passed after freezing the objective revision.

## Required rerun

Only SCF folds 0 and 1 must be repeated.  DPEL must not be retrained.

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage32_pilot_h100.sh formal SCF 0 stage32_pilot_SCF_fold0_formal_seed203200_normfix
bash ./run_stage32_pilot_h100.sh formal SCF 1 stage32_pilot_SCF_fold1_formal_seed203201_normfix
```

After both corrected SCF runs finish, freeze steps 192/384/576 for corrected
SCF and the existing DPEL runs, then perform the preregistered two-fold paired
pilot evaluation.  Do not use training-batch deltas as OOF PDMS evidence.
