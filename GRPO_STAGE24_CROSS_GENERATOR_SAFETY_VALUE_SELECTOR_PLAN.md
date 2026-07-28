# Stage24: Cross-Generator Safety-Value Selector and GRPO Closure

## Status

- Preregistered before Stage24 implementation or result inspection.
- The user selected the strict paper-validation protocol.
- Stage16/19/21 checkpoints and fold4 are development evidence only. Fold5
  remains protected until the complete Stage24 selector and generator recipe is
  frozen.

## Objective

Stage21 fold4 candidate artifacts show that roughly 90% of scenes contain a
safe candidate at least `+0.005` better than the generator's reference top-1,
with safe-oracle mean headroom near `+0.12`. Stage23 nevertheless switched only
14 of 2,042 records. Stage24 therefore treats cross-generator selection,
calibration alignment, recall, and correlated false confidence as the primary
bottlenecks. Fresh GRPO training is forbidden until the selector-only gates
pass.

The primary GRPO quantity remains the paired PDMS difference under one frozen
selector:

`GRPO effect = C - B`.

`B - B0` is the selector contribution and must be reported separately.

## Locked data protocol

The selector candidate bank contains all 20 trajectories, reference logits,
six PDM components, aggregate PDMS, token/log provenance, noise namespace,
generator domain, checkpoint SHA256, and evaluator/config provenance.
Generator domain is used for balanced sampling and audit only; it is never a
model input.

- Development selector training: Stage21 folds 0--2 (3,056 scenes), generator
  domains official base, Stage16 epoch8, Stage19 epoch8, and Stage21 epoch8;
  namespaces `default`, `20260811`, and `20260812`.
- Development selector calibration: fold3 (1,019 scenes), the identical four
  generator domains and three namespaces. The complete deployment ensemble,
  not one held-out member, is calibrated.
- Frozen diagnostic: consumed fold4 (1,021 scenes), all four domains and unseen
  namespaces `20260821` and `20260822`.
- Protected validation: fold5 (1,023 scenes), namespaces `default`, `20260823`,
  `20260824`, and `20260825`. Fold5 is opened once only after the final selector
  and generator recipe is frozen.

All split boundaries are complete logs. No frame/token split may cross folds.

## Locked selector definition

### Architecture

- A separate eight-member Stage24 deep ensemble preserves the Stage23 module
  unchanged for historical reproduction.
- Member `m` uses a deterministic seed and an 80% whole-log sub-bag of folds
  0--2. Each member owns its full trajectory/context encoder.
- Every trajectory is encoded from geometry/kinematics, trajectory-aligned BEV,
  agents, ego/status, and its reference logit. Two permutation-equivariant
  set-attention layers compare all 20 candidates jointly.
- Absolute mode ID and generator/checkpoint identity are forbidden inputs.
- The generator's own reference-logit top1 is the fallback. Heads predict three
  safety-event risks, any-unsafe risk, six component deltas, and PDMS delta
  relative to that fallback. The selector embedding is exposed for an OOD
  audit.

### Loss and schedule

The fixed loss is:

`2 L_any_unsafe + L_safety_components + 2 L_false_safe_hinge + L_PDMS_delta + 0.5 L_component_delta + L_safe_pair_rank`.

- Safety labels use compliance `< 0.999` as unsafe. Positive weights are the
  train-bank negative/positive ratio clipped to `[1, 50]`; focal gamma is 2.
- For every set, the four predicted-highest truly unsafe challengers are hard
  negatives in the false-safe hinge.
- Value ranking uses only truly safe pairs with absolute PDMS gap at least
  `0.005`.
- Train exactly 12 epochs. A deterministic 12-way cycle exposes each token once
  to each of four generator domains times three namespaces. Use the final epoch;
  no calibration-based checkpoint selection is allowed.

### Deployment rule and calibration

- Use the full eight-member mean and standard deviation with `z=1.96`.
- `risk_UCB = mean + z * std`.
- `delta_LCB = mean - z * std - q95_conformal_residual`.
- Fit a diagonal Mahalanobis embedding model on training candidates and reject
  challenger embeddings above the fold3 99.5th-percentile distance.
- A challenger is eligible only if it is in-distribution, its joint risk UCB is
  below the calibrated threshold, each predicted safety compliance is no worse
  than fallback by more than `0.001`, and its PDMS delta LCB is positive.
- Choose the eligible challenger with maximum delta LCB; otherwise keep the
  same generator's fallback. No base generator, PDM oracle, future information,
  evaluation feedback, or token blacklist is available at inference.

Fold3 calibration must achieve selector gain at least `+0.005`, a strictly
positive bootstrap confidence interval, non-negative gain in every generator
domain and namespace, switch rate at least 2%, safety-component mean regression
no worse than `0.0005`, and a one-sided catastrophic-switch Clopper--Pearson
upper bound at most `0.0025`. Failure stops before fold4.

## Gates and GRPO closure

### Selector-only fold4 gate

The frozen development selector must achieve pooled gain at least `+0.003`, a
strictly positive confidence interval, Stage21-domain gain at least `+0.003`,
non-negative gain in every generator domain and namespace, safety-component
mean regression no worse than `0.001`, and catastrophic upper bound at most
`0.005`. The known Stage23 catastrophic artifact must fall back through learned
logic, never a token special case.

### Fresh generator gate

After the selector gate passes, train a fresh generator from the official base
with the existing `diffgrpo_selected_set` objective on folds 0--3. Keep `G=8`,
one selected trajectory per complete 20-mode set, the Stage23 paired-base guard,
exact KL, BC weight, diffusion discount, 64 trainable decoder tensors, and
learning rates. Candidate epochs remain exactly 2, 4, 6, and 8.

Under the identical frozen selector, fold4 `C - B` must be positive in every
namespace, pooled at least `+0.005` with positive confidence interval, at least
`+0.010` on hard scenes, at least `-0.002` on mature scenes, no worse than
`0.001` in each safety component, catastrophic upper bound at most `0.005`, and
fresh-generator candidate OOD rate at most 5%.

### Final selector and protected fold5

After a fresh generator passes fold4, add its selected checkpoint as the fifth
candidate-bank domain. Retrain the exact selector recipe on folds 0--3,
calibrate the full ensemble on consumed fold4, and re-run the complete fold4
`C - B` gate. Then retrain the generator from official base on folds 0--4 for
exactly the selected epoch and evaluate fold5 once.

A fold5 gain of `+0.005` is minimally useful, `+0.010` is the paper target, and
`+0.015` is the desired target. A fold5 failure forbids threshold tuning, extra
epochs, dev-confirm, and NavTest within Stage24.

Only a fold5 pass authorizes full 6,119-scene training on external 8x H100
(preferred) or 8x RTX 4090 compute, for exactly the selected epoch rather than
an arbitrary multi-dozen-epoch extension.

## Attribution and required audits

- `B0`: official base plus official selector.
- `B`: official base plus Stage24 selector.
- `C`: Stage24 GRPO plus Stage24 selector.
- `D`: Stage24 GRPO plus official selector, diagnostic only.

Report `C-B` as GRPO, `B-B0` as selector, and `C-B0` as total system gain.

Required tests cover candidate permutation equivariance, trajectory dependence,
safety/value separation, OOD fallback, catastrophic fallback, full-ensemble
calibration, whole-log isolation, domain/namespace balance, artifact SHA/shape,
no PDM/base-generator inference call, one selected chain per `G=8` set, frozen
selector SHA, selected-chain-only gradients, exact KL/base guard, and Stage13--
Stage23 configuration regression. U8, U32, and U64 smoke tiers must pass before
formal training.

## Stop rules

- Do not use fold4 for a paper claim or fold5 for tuning.
- Do not repair a failed selector with additional generator epochs.
- Do not add a base-generator fallback, online PDM scorer, domain identity input,
  token blacklist, or hidden namespace filtering.
- Do not open fold5 until development selector, fresh generator, final selector,
  and all hashes/configs are frozen.

## Execution record

Implementation and results are appended below without changing the locked
definitions or thresholds above.

### 2026-07-22 implementation freeze and pre-training gates

- Selector-train folds0--2 manifest: 3,056 tokens from 454 complete logs;
  ordered-token SHA256
  `8af3a9631dabf8bc5821ff6cf05102bcd96be65dba5e0267e71d1a68bd8c1b7a`.
- Full-ensemble calibration fold3 manifest: 1,019 tokens from 152 complete
  logs; ordered-token SHA256
  `37e3844d1b631a6f45cd45753faf9a50092d0433973f87890a4b543d9dfd1d20`.
  The train and calibration log sets are disjoint. Fold5 remains unopened.
- All twelve official-base/Stage16/Stage19/Stage21 by three-namespace
  candidate banks completed with zero evaluator failures. The fail-closed bank
  audit is `artifacts/grpo_stage24/candidate_bank_audit.json` and passed all
  wrapper/source SHA, generator checkpoint, token/log, and all-20 shape checks.
  Across 36,672 domain/namespace scene records, fallback PDMS is 0.8173545,
  safe-challenger oracle headroom is +0.1245002, 90.8786% have at least +0.005
  safe headroom, and 88.3181% have at least +0.010 safe headroom.
- The frozen implementation contains the eight independent selector members,
  safety/value/ranking loss, deterministic whole-log sub-bags, same-generator
  fallback, full-ensemble calibration, OOD rejection, candidate-bank builder
  and audit, fold3 calibrator, and strict fold4 gate. Model inference has no
  PDM scorer or base-generator fallback path.
- Stage20--24 regression suite: 42/42 passed. Stage24-specific suite: 13/13
  passed, including Hydra/dataclass schema identity and whole-log isolation.
- Final U8 real-training audit:
  `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage24_selector_audit_seed24024_v3/2026.07.22.15.24.17`.
  Every step had positive Stage24 gradient; decoder, perception, Stage23,
  value-selector, and paired-risk gradients were exactly zero.
- Formal fixed 12-epoch selector training started from the official base at
  `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage24_selector_formal_seed24024/2026.07.22.15.25.37`.
  The final epoch is mandatory; no checkpoint is selected using calibration.

### 2026-07-22 formal selector and fold3 calibration result

- Formal training completed all 12 locked epochs and 18,336 optimizer updates.
  The mandatory final checkpoint is
  `lightning_logs/version_0/checkpoints/grpo-11-18336.ckpt`, SHA256
  `59a35466e1b59332596bf3f89b5f1a6bcd2c82af89cc0d340b7415a6e05d11df`.
  Selector loss decreased from 0.746181 at epoch 0 to 0.427663 at epoch 11.
  All 18,336 Stage24 gradient observations were finite and positive; the
  Stage23 selector, generator decoder, perception trunk, value selector, and
  paired-risk module had exactly zero gradient.
- Fold3 calibration collection completed all twelve domain/namespace
  combinations: 12,228 scene records, 1,019 per artifact, zero failures. Every
  artifact is bound to the final selector SHA and the fold3 ordered-token SHA.
- The locked calibration failed before fold4, as required by the stop rule.
  Its immutable result is
  `artifacts/grpo_stage24/stage24_selector_calibration_fold3.json`: selected
  gain 0, switch rate 0, and `passed=false`. Fold4 and fold5 remain unopened.
- The failure is localized to the uncalibrated component-delta hard gate, not
  value recall or OOD rejection. Across 232,332 non-fallback candidates,
  `delta_LCB > 0` retained 22,312 (9.60%) and the OOD gate retained 231,170
  (99.50%), but `safety_delta_LCB >= -0.001` retained zero. Its maximum was
  -0.00527. The same fold contains 115,281 truly-safe positive-delta
  candidates. The reproducible waterfall and tolerance sweep are stored in
  `artifacts/grpo_stage24/stage24_selector_fold3_diagnosis.json`.
- Relaxing that single auxiliary gate post hoc is not accepted as a Stage24
  pass. The diagnostic sweep does show that the learned value head contains
  useful signal: at safety tolerance 0.05 the fold3 mean gain is +0.00583 with
  positive gain in every domain/namespace, and at tolerance 0.10 it is
  +0.01509. Those settings violate the preregistered safety-component and
  catastrophic bounds, so they are evidence for the next safety redesign, not
  deployment results.
