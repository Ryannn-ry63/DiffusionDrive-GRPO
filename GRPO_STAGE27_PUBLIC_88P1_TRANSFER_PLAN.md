# Stage27: GRPO Transfer to the Actual Public 88.1 Baseline

## Status and correction

- Preregistered on 2026-07-24 before any Stage27 generator training or
  Stage27 method result is inspected.
- Stage16--26 did **not** initialize from the released 88.1 checkpoint. Their
  locked reference was the local epoch-19 checkpoint with SHA256
  `59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d`.
  Its Stage26 system-A score on the current 12,146-scene NavTest protocol is
  `0.849164256599452`.
- Stage27 is an independent transfer experiment. It must not overwrite,
  relabel, or invalidate Stage26.

## Immutable public baseline

The only Stage27 initialization and frozen reference checkpoint is:

`/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS`

Its immutable identity is:

| Field | Value |
|---|---|
| SHA256 | `008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b` |
| checkpoint epoch | `99` |
| global step | `16700` |
| model tensors | `763` |

The checkpoint may be loaded directly. It does not require supervised
fine-tuning before GRPO. Every Stage27 GRPO job must initialize both the
current policy and the frozen reference from this exact file.

## Test-set discipline

NavTest must not be used to choose a selector, threshold, training duration,
or generator checkpoint.

The released checkpoint's public-schedule NavTest score (`P`) may be
reproduced immediately because it is a baseline measurement, not a Stage27
method choice. All selector compatibility decisions and all generator
checkpoint decisions must use the existing whole-log-disjoint trainval
folds. The final Stage27 NavTest decomposition is run only after the method is
frozen.

## Phase 0: finish Stage26

The already-running Stage26 system-C NavTest is allowed to finish unchanged.
Its primary comparison remains `C-B`, with the same frozen Stage26 selector.
Stage27 work may run on other machines but must use distinct experiment and
artifact directories.

## Phase 1: parallel public-baseline characterization

The following jobs are independent and may run concurrently:

1. Reproduce `P`: the public checkpoint with the released short diffusion
   schedule `[8, 0]` on all 12,146 NavTest scenes.
2. On the 4,075-token folds0--3 selector-fit manifest, collect all 20
   candidates and their six PDM components under namespaces `-1`,
   `20260811`, and `20260812`. These are three one-GPU jobs.
3. On the 1,021-token fold4 development manifest, evaluate the frozen Stage26
   selector on the public generator under namespaces `20260821` and
   `20260822`. These are two one-GPU jobs.

The five trainval jobs are development measurements. They do not authorize a
final NavTest method by themselves.

## Phase 2: selector transfer gate

For each fold4 namespace, compare the frozen Stage26 selector with the
generator-logit fallback using paired per-token rewards. Pool uncertainty by
whole log, not by individual scene.

The existing selector may be reused only if all conditions hold:

- pooled selected-minus-fallback mean is positive;
- whole-log bootstrap 95% CI lower bound is positive;
- neither namespace has negative mean gain;
- collision, drivable-area compliance, TTC, comfort, and direction component
  means each regress by no more than `0.0005`;
- no catastrophic switched scene loses more than `0.05` aggregate PDMS;
- OOD rejection and switch diagnostics are finite and non-degenerate.

If this gate fails, Stage27 must train and calibrate a new selector using only
the three public-baseline folds0--3 banks and fold4 development data. The
Stage26 calibration thresholds must not be adjusted by inspecting NavTest.

## Phase 3: Stage27 code and one-step GRPO audit

Add a Stage27-specific formal training mode and SHA guard. Do not replace the
Stage16--26 formal-base constant globally.

Before formal training, an eight-GPU one-step audit must prove:

- current and reference policies both load the public SHA;
- the reference policy is bitwise frozen;
- gradients are finite and positive only in the registered decoder scope;
- perception, reference, and selector parameters have exact-zero gradients;
- reward, group advantage, log probability, and optimizer update are finite.

## Phase 4: development generator training

Train on folds0--4 (5,096 rewardable scenes) for exactly two epochs initially:

- eight-GPU DDP;
- batch size `1` per device;
- gradient accumulation `8`;
- learning rate `1e-6`;
- `diffgrpo_selected_set`;
- group size `8`;
- full schedule `[32, 24, 16, 8, 0]`;
- the selector that passed Phase 2, frozen throughout generator training.

Save both epoch-1 and epoch-2 checkpoints. Do not extend to dozens of epochs.
Training duration may be extended to at most four epochs only after a
preregistered trainval comparison demonstrates continuing held-out
improvement without safety-tail regression.

## Phase 5: held-out generator gate

Evaluate the public generator baseline (`B`) and each Stage27 generator
checkpoint (`C1`, `C2`) with the same frozen selector, tokens, and paired
noise. Select between epoch 1 and epoch 2 using trainval only.

The formal generator gate requires:

- `C-B >= +0.005` pooled aggregate PDMS;
- whole-log bootstrap 95% CI lower bound greater than zero;
- positive mean gain in both locked namespaces;
- no material collision or TTC regression;
- no increase in catastrophic-loss frequency.

Failure stops full retraining. It must not trigger blind epoch extension.

## Phase 6: all-scene retraining and final NavTest

After the development gate passes, restart from the original public
checkpoint and train on all 6,119 rewardable scenes for the selected fixed
duration. Audit the resulting checkpoint against the public reference.

Freeze the final system before NavTest. The final decomposition is:

| System | Generator | Selector | Schedule |
|---|---|---|---|
| `P` | public 88.1 | generator logits | `[8, 0]` |
| `P0` | public 88.1 | generator logits | `[32, 24, 16, 8, 0]` |
| `B` | public 88.1 | frozen Stage27 selector | `[32, 24, 16, 8, 0]` |
| `C` | public-initialized Stage27 GRPO | same frozen selector | `[32, 24, 16, 8, 0]` |

Primary claims:

- `C-B`: pure GRPO generator contribution;
- `B-P0`: selector contribution;
- `P0-P`: schedule contribution;
- `C-P`: total Stage27 system contribution over the actual public baseline.

Report paired scene counts, whole-log bootstrap confidence intervals, all six
PDM components, switch counts, worst losses, and checkpoint/config hashes.
