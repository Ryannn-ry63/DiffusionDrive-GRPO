# Stage25: Relative-Harm Safety Gate for the Cross-Generator Selector

## Status and boundary

- Preregistered after the locked Stage24 fold3 failure and before Stage25 code,
  training, or fold4 inspection.
- Stage24 is permanently recorded as failed: its raw component-delta LCB gate
  admitted zero challengers. Post-hoc Stage24 tolerance sweeps are diagnostic
  only and cannot become deployment thresholds.
- Fold3 is now development/calibration data. Fold4 remains an unseen frozen
  diagnostic and fold5 remains protected. No fresh GRPO generator training is
  allowed until this selector-only stage passes.

## Hypothesis

Stage24 learned useful relative value but used two mismatched safety signals:
absolute candidate unsafety classification and a continuous component-delta
regressor. The deployment failure is caused by the latter's small negative
bias, while relaxing it exposes unsafe switches. Stage25 replaces that hard
regression gate with direct binary prediction of the exact relative-harm event
that deployment must avoid.

## Frozen data and initialization

- Reuse the SHA-audited Stage24 folds0--2 candidate banks: four generator
  domains by three namespaces, 3,056 tokens from 454 complete logs.
- Reuse only the mandatory Stage24 final selector checkpoint, SHA256
  `59a35466e1b59332596bf3f89b5f1a6bcd2c82af89cc0d340b7415a6e05d11df`.
  Its eight member encoders and value heads are frozen.
- Add one independent two-layer relative-harm MLP per ensemble member on that
  member's frozen trajectory/context embedding. Generator identity, namespace,
  token, PDM labels, and future information remain forbidden inputs.
- Member sub-bags remain the deterministic Stage24 80% whole-log sub-bags.

## Labels, loss, and schedule

For safety components collision, drivable-area compliance, and TTC:

`harm_j = candidate_j < fallback_j - 0.0005`.

The joint catastrophe label is true when any `harm_j` is true or aggregate
PDMS delta is at most `-0.5`. These labels match the Stage24/25 gate rather
than predicting absolute unsafety.

Each member predicts three harm logits and one joint-catastrophe logit. The
fixed loss is:

`3 L_relative_harm + 4 L_catastrophe + 4 L_high_value_false_safe`.

- The first two terms use focal binary loss with gamma 2 and train-bank
  negative/positive weights clipped to `[1, 200]`.
- For every scene/member, the four truly harmful non-fallback candidates with
  highest frozen Stage24 delta prediction are hard negatives. Their maximum
  predicted harm/catastrophe probability is pushed to at least 0.90.
- Train only the new MLPs for exactly 12 epochs using the same deterministic
  12-combination bank cycle, so every token reaches every domain/namespace
  combination exactly once. Use the final epoch only; no
  checkpoint or epoch selection from fold3 is permitted.

## Deployment and calibration

- Retain the frozen Stage24 value LCB, q95 value residual, and diagonal OOD
  rejection.
- Define relative-harm risk per member as the maximum of its three harm
  probabilities and joint-catastrophe probability. Use the eight-member
  `mean + 1.96 std` risk UCB.
- A challenger is eligible iff value LCB is positive, OOD passes, and
  relative-harm risk UCB is below the single fold3-calibrated threshold.
  The Stage24 raw component-delta LCB is diagnostic only and is not a gate.
- Select the eligible maximum value LCB or retain the same-generator fallback.
  No PDM evaluator is available inside inference.

Fold3 calibration scans only the risk threshold. It must achieve:

- mean gain at least `+0.005` and whole-log bootstrap 95% CI lower bound > 0;
- non-negative gain in every generator-domain/namespace group;
- switch rate at least 2%;
- each safety-component mean delta at least `-0.0005`;
- catastrophic-switch one-sided 95% Clopper--Pearson upper bound <= `0.0025`.

Failure stops Stage25 before fold4 and generator training.

## Unseen fold4 gate

After a fold3 pass, freeze all selector and calibration hashes and evaluate the
four generator domains under namespaces `20260821` and `20260822` on fold4.
Requirements are pooled gain >= `+0.003`, positive whole-log CI, Stage21
domain gain >= `+0.003`, every domain/namespace non-negative, each safety
component >= `-0.001`, and catastrophic upper bound <= `0.005`.

Only a fold4 pass resumes the fresh-GRPO-generator closure defined in Stage24.
Fold4 cannot be used to tune Stage25, and fold5 stays unopened.

## Required audits

- Exact relative-harm and catastrophe label boundary tests.
- Independent eight-member heads and whole-log sub-bag isolation.
- Frozen Stage24 selector/value/decoder/perception gradients exactly zero;
  Stage25 head gradients finite and positive.
- Candidate permutation equivariance and trajectory dependence.
- Calibration/checkpoint SHA fail-closed behavior.
- Inference contains no PDM scorer, base-generator fallback, domain input,
  namespace filtering, or token blacklist.
- Stage20--24 regression remains green.

## Execution record

Results will be appended here without altering the locked protocol above.

### 2026-07-22 implementation freeze

- Added eight independent member-wise relative-harm MLPs over detached
  Stage24 embeddings, direct three-component harm and joint-catastrophe labels,
  imbalance-weighted focal losses, selection-aware hard false-safe mining, and
  the relative-risk UCB deployment rule.
- Stage24 value, OOD moments, candidate-bank loading, whole-log sub-bags, and
  same-generator fallback are reused without changing the Stage24 historical
  selector. Stage25 training exposes only the new MLP parameters.
- Added fail-closed Stage25 checkpoint/calibration loading, external-PDM-only
  evaluator collection, a single-threshold calibrator, and formal training and
  evaluation launchers.
- Stage25/24 focused tests passed 17/17. Stage20--25 regression passed 46/46;
  Python compilation, shell syntax, and `git diff --check` passed.
- The U8 real-training audit has not run: its GPU execution permission request
  was denied and the sandboxed fallback cannot start because of the environment
  mount restriction. No Stage25 optimizer step or fold3 inspection occurred.

### 2026-07-23 formal execution and locked fold3 pass

- The U8 audit completed 8/8 optimizer steps. Every Stage25 gradient was finite
  and positive; decoder, perception, Stage24 value/selector, Stage23, and paired
  risk gradients were exactly zero.
- Formal training completed the mandatory 12 epochs (18,336 Stage25 updates).
  The final checkpoint is
  `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage25_selector_formal_seed25025/2026.07.23.02.45.45/lightning_logs/version_0/checkpoints/grpo-11-18336.ckpt`,
  SHA256 `f66c5b8f734fad40ac8b08c603b76a36bc97e869a106de301c667340cecbde63`.
  Its epoch loss decreased from `1.556262` to `0.936040`; all 18,336 formal
  Stage25 gradient audits were finite and positive and all frozen gradients
  remained exactly zero.
- Locked fold3 used 12,228 scenes from all four generator domains under all
  three preregistered namespaces. The calibration is
  `artifacts/grpo_stage25/stage25_selector_calibration_fold3.json`, SHA256
  `beeeedb00b159e96de9ceaeaa8cf0d9c3a6e2c9672ed505c851566b25dd75814`.
- Fold3 passed every gate at risk threshold `0.6461921291291606`: mean gain
  `+0.0057313`, whole-log 95% CI `[+0.0037763, +0.0079403]`, 12/12
  domain/namespace means positive, switch rate `3.4756%`, safety-component
  means `[0, +0.0006542, -0.0004907]`, and catastrophic CP upper bound
  `0.0011801`.
- The threshold sweep was changed from a redundant brute-force replay to an
  exactly equivalent incremental breakpoint sweep after the former produced no
  output. An explicit boundary/tie/fallback test matches the brute-force choice
  at every synthetic threshold; the selected formal threshold is additionally
  rematerialized and checked against the incremental mean before bootstrap.

### 2026-07-23 frozen unseen fold4 pass

- With all checkpoint and calibration hashes frozen, fold4 evaluated 8,168
  scenes: four generator domains under namespaces `20260821` and `20260822`.
  The immutable gate result is `artifacts/grpo_stage25/fold4/gate.json`.
- Fold4 passed every preregistered requirement: pooled gain `+0.0047268`,
  whole-log 95% CI `[+0.0030403, +0.0065076]`, Stage21-domain gain
  `+0.0046563`, and all 8/8 domain/namespace means positive.
- Safety-component means were `[0, -0.0003673, 0]`. Three catastrophic
  switches occurred in 8,168 scenes, giving a one-sided 95% CP upper bound of
  `0.0009490`; both remain inside the frozen fold4 limits.
- `stop_before_generator_training` is therefore false. Fold5 remains unopened.
  Stage25 is now allowed to resume the fresh-GRPO-generator closure defined in
  Stage24; fold4 must not be reused for selector tuning.

### 2026-07-23 fresh-generator implementation and pre-training audits

- The existing `diffgrpo_selected_set` trace had retained a Stage23-only
  selector call. It now dispatches fail-closed between the historical Stage23
  selector and the frozen Stage25 `trajectory_relative_harm_v3` selector. The
  Stage25 path uses the frozen Stage24 value/embedding ensemble, Stage25
  relative-harm ensemble, calibrated value residual, risk threshold, and OOD
  state while sampling, then replays only one selected chain from each `G=8`
  complete 20-mode set with gradients.
- Formal configuration and model construction accept only those two selector
  sources. Stage25 selected-set training requires non-empty selector checkpoint
  and calibration paths, exact calibration values, and matching checkpoint,
  calibration, and passed-fold4-gate hashes.
- Single-GPU U8, U32, and U64 one-step audits passed. Decoder gradient norms
  were respectively `0.3709450`, `0.2259482`, and `0.1474568`; perception,
  value, Stage23, Stage24, Stage25, and paired-risk gradients were exactly zero
  at every tier. GRPO and BC losses were finite.
- An eight-RTX-4090 DDP one-step audit also passed at
  `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage25_generator_ddp8_audit/2026.07.23.08.11.06`.
  All eight ranks completed their barrier, wrote `grpo-00-1.ckpt`, and exited
  cleanly. The synchronized decoder gradient was `0.1857190`; every frozen
  module gradient remained exactly zero.
- The external launcher is
  `scripts/training/run_diffusiondrive_grpo_stage25_generator.sh`. It supports
  an 8-GPU DDP audit and the locked 8-epoch folds0--3 development run, detects
  the external `navsimH100` Python environment, and honors
  `NAVSIM_DEVKIT_ROOT`, `NAVSIM_EXP_ROOT`, `GRPO_BASE_CHECKPOINT`, and
  `GRPO_TRAIN_CACHE`. The only admissible generator candidates remain epochs
  2, 4, 6, and 8.
- Before inspecting any fresh-generator fold4 result, the epoch-selection rule
  is fixed as: apply every Stage24 fresh-generator gate independently to epochs
  2, 4, 6, and 8, then choose the eligible epoch with the highest pooled paired
  `C-B`; an exact tie chooses the earlier epoch. If no epoch passes every gate,
  Stage25 stops before final-selector training. Candidate OOD rate is the
  fraction of all 20 fresh-generator candidates whose frozen Stage24 embedding
  distance exceeds the calibrated OOD threshold.

### 2026-07-23 fresh-generator fold4 C-B pass and epoch lock

- Formal 8-GPU training completed all eight epochs on the locked 4,075-token
  folds0--3 manifest. Candidate checkpoints 2/4/6/8 were finite, had the
  expected global steps 128/256/384/512, and retained all 626 Stage24/25
  selector tensors exactly. Their SHA256 values are recorded in
  `artifacts/grpo_stage25/generator_fold4/gate.json`.
- All eight epoch/namespace evaluation cells completed 1,021/1,021 fold4
  scenes with zero failures under the identical frozen Stage25 selector and
  calibration. The strict paired gate passed and locked epoch 2, checkpoint
  `grpo-01-128.ckpt`, SHA256
  `8a273860a365b2ca0eb6f97fddd8d381714cf7f4a30ceb08267aed1756413989`.
- Epoch 2 achieved pooled `C-B = +0.0101724` with whole-log 95% CI
  `[+0.0036014, +0.0184337]`. Namespace gains were `+0.0113820` and
  `+0.0089628`; hard-scene gain was `+0.0465019`, mature-scene change was
  `-0.0014212`, and all three safety-component mean changes were positive.
  Three catastrophic pairs gave CP upper `0.0037927`; candidate OOD rate was
  `0.0047013`. Every locked gate passed.
- Epochs 4/6/8 had larger pooled means (`+0.0103466`, `+0.0120721`, and
  `+0.0134707`) but are ineligible: each exceeded the mature-scene regression
  limit and catastrophic CP upper bound. They cannot be rescued by extra
  training or post-hoc threshold changes.
- This is the first frozen-selector, same-token, same-noise, whole-log
  significant paper-scale attribution of generator-only GRPO improvement in
  this closure. Fold5 remains unopened. The next authorized action is the
  final-selector retraining and consumed-fold4 revalidation defined in Stage24.
