# Stage28: Public-88.1 Paired-Uplift GRPO

## Decision status

This plan is frozen on 2026-07-24 after the completed Stage27 Phase4 fold5
diagnosis and before any Stage28 code change, training, or evaluation.

Stage28 has one scientific objective: preserve the severe-failure repairs
observed in Stage27 while turning the median, trimmed mean, and whole-log
distribution of public-88.1 `C-B` positive. It does not extend the Stage27
checkpoint into a final model and it does not inspect NavTest while making
method decisions.

## Evidence motivating the change

- Stage26 produced a statistically significant `+1.1675` NavTest PDMS-point
  GRPO contribution from a weaker local epoch-19 base.
- Stage27 used the same 5,096 training tokens in the same order, the same
  160 optimizer steps, learning rate, selected-set objective, group size,
  schedule, and decoder scope.
- Stage27 gradient clipping was inactive: the maximum observed decoder
  gradient norm was approximately `0.229`, below the configured threshold
  `1.0`.
- Relative parameter movement in Stage26 and Stage27 was comparable, so the
  Stage27 result is not explained by a missing optimizer update.
- The public checkpoint has fewer low-score and zero-score scenes. On the
  common folds0--3 banks, its selected reward is about `0.842` versus `0.804`
  for the local base, while both oracle rewards are about `0.940`.
- The oracle gap therefore contracts from about `0.133` to `0.098`.
- Stage26 training current-minus-reference reward averaged `+0.00571`;
  Stage27 averaged only `+0.00167`.
- Stage27 epoch 2 produces `C-B=+0.00295066`, but its token median is
  `-0.0000377`, 10% trimmed mean is `-0.0000600`, and four large repairs
  contribute more than the complete mean.
- The public-only selector exposes `+0.01242` candidate-level gain but fails
  safety and catastrophic gates. The deployable multi-domain selector is safe
  but exposes only `+0.00305` gain with a `2.35%` switch rate.

The diagnosis is therefore twofold:

1. the Stage27 objective can reward the best current rollout even when it is
   slightly worse than the paired public reference;
2. safe deployment selection is too conservative to serve simultaneously as
   the only training exploration policy.

## Immutable public inputs

- Public checkpoint:
  `/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS`
- Public checkpoint SHA256:
  `008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b`
- Deployable S-multi selector SHA256:
  `023d7b6b77bb8fa2dc3e779849f3716688abf0796b019416494c80b29615b691`
- Deployable S-multi calibration SHA256:
  `fec2a10573e913e4452f8eea92e6334fcd5e6f1104e6244876770973a1f41990`
- Training-only S-public selector SHA256:
  `9a7a202830f0dcd87c5aeac2f91a3b2dc3958584cc965ecf4b7048f56ceca91b`
- Training-only S-public calibration SHA256:
  `ab1931fc96c00288c32b8d2fa484c2114dd5cd66ef1ccf05693d8fba8885660a`
- Pilot training population: the 4,075 ordered folds0--3 tokens from 606
  whole logs.
- Pilot validation population: fold4, 1,021 tokens.
- Confirmation population: fold5, 1,023 tokens from 151 whole logs.

The S-public selector is explicitly `training_only` and
`deployment_forbidden`. Its calibration failed the Stage27 deployment gate
and can be loaded only through a Stage28 SHA-locked exploration authorization.
Every evaluation and deployment uses S-multi or its later safe closure
successor.

## Paired-uplift objective

Add two Stage28-only formal modes without changing historical modes:

- `stage28_public_paired_uplift_multi`
- `stage28_public_paired_uplift_explore`

Both modes use common-random current/reference trajectories and compute
`delta = current_reward - reference_reward`.

The frozen advantage rules are:

- positive only when `delta > +0.002` and no safety component regresses;
- regular negative when `delta < -0.002`;
- for mature reference reward `>=0.75`, negative when
  `delta < -0.0005`;
- any collision, drivable, or TTC regression receives advantage `-2`;
- relative delta evidence scale `0.1`;
- advantage clip `2.0`;
- denoising-step discount `0.6`;
- BC weight `0.1`;
- regular/mature exact-KL weights `0.1/0.5`;
- no current-reward bootstrap advantage for neutral or all-worse groups.

The objective must log positive, regular-negative, mature-negative, safety
override, neutral, active-scene, exact-KL, and paired-delta diagnostics.

The multi mode selects training chains with S-multi. The explore mode selects
training chains with S-public, but exact PDM paired guards determine gradient
sign. Selectors, perception, classification, and reference policy remain
frozen; only the registered 64 decoder tensors may change.

## Three-branch pilot

Run three independent seed-0 jobs from the original public checkpoint on
folds0--3. Each job trains exactly four epochs and saves epochs 1--4:

| Branch | Training selector | Objective |
|---|---|---|
| A | S-multi | unchanged Stage27 selected-set |
| B | S-multi | Stage28 paired-uplift |
| C | S-public training-only | Stage28 paired-uplift |

All other hyperparameters are identical: eight GPUs, batch size 1 per device,
gradient accumulation 8, FP32, learning rate `1e-6`, group size 8, full
schedule `[32,24,16,8,0]`, layer-0 LR multiplier `0.1`, and public reference.

Branch A distinguishes insufficient duration from an objective failure.
Branch B isolates the paired-uplift objective. Branch C tests whether guarded
training exploration unlocks headroom that the safe deployment selector does
not expose.

## Fold4 selection gate

Evaluate the public baseline and all 12 pilot checkpoints with the same
S-multi deployment selector under namespaces `20261111` and `20261112`.

A checkpoint is eligible only if:

- pooled `C-B >= +0.005`;
- whole-log bootstrap 95% CI lower bound is strictly positive;
- both namespace means are positive;
- 10% trimmed mean is non-negative;
- wins exceed losses after exact ties are removed;
- hard-scene (`B<0.75`) mean gain is at least `+0.010`;
- mature-scene mean gain is at least `-0.0005`;
- collision, drivable, TTC, comfort, and direction component deltas are each
  at least `-0.0005`;
- catastrophic-regression one-sided 95% upper bound is at most `0.005`;
- all provenance and finite-value checks pass.

Eligible checkpoints are ranked lexicographically by:

1. larger whole-log CI lower bound;
2. larger 10% trimmed mean;
3. larger pooled mean;
4. fewer catastrophic regressions;
5. earlier epoch.

If no checkpoint is eligible, Stage28 stops before fold5 and NavTest.

## Confirmation, selector closure, and final training

Retrain the selected method from the original public checkpoint on folds0--4
for the fixed selected epoch. Do not continue a pilot checkpoint.

Evaluate the retrained checkpoint once on fold5 under namespaces `20261121`,
`20261122`, and `20261123`, applying the complete fold4 gate. Failure stops
Stage28.

After a fold5 pass:

1. collect three folds0--3 candidate banks from the selected generator;
2. add them to the public and historical multi-generator selector banks;
3. train a fresh safe Stage24/25 closure selector;
4. calibrate it on fold4 under namespaces `20261113` and `20261114`;
5. require selector gain at least `+0.003`, positive whole-log CI, zero
   catastrophic switches, passing component guards, and preservation of the
   full Stage28 generator gate.

Only after closure passes, restart from the public checkpoint on all 6,119
rewardable scenes using the frozen method and epoch. Freeze and audit every
model/config SHA before NavTest.

The final NavTest systems are:

- `P`: frozen public 88.1041 result;
- `P0`: public generator with full schedule;
- `B`: public generator with final safe Stage28 selector;
- `C`: Stage28 generator with the same selector.

The primary method claim is `C-B`; the complete-system claim is `C-P`.
Success requires `C-B >= +0.005` with positive whole-log CI and `C-P > 0`.
The stretch target is `C-B >= +0.010`. No method, epoch, or selector change is
allowed after NavTest inspection.

## Verification and compute

- Unit tests must cover all-positive, all-worse, neutral, mature-negative,
  safety-regression, non-finite, and shape-error objective cases.
- Exploration authorization must fail on hash drift and must be impossible to
  use in deployment/inference modes.
- One-step audits must show finite rewards/losses, active decoder gradients,
  and exact-zero gradients in every frozen module.
- Checkpoint audits must allow changes only in the registered 64 decoder
  tensors and must verify the correct training selector role.
- Evaluation artifacts must verify token order, noise namespace,
  checkpoint/selector/calibration SHA, all 20 candidates, and zero failures.

Use the local 8x RTX 4090 node for implementation audits and parallel
evaluation. Run the three four-epoch pilot branches as independent parallel
8x H100 jobs when available. Do not run dozens of epochs.
