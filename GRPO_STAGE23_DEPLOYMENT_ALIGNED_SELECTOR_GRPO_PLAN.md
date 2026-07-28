# Stage23: Deployment-Aligned Trajectory Selector and Set-Level GRPO

## Status

- Preregistered before Stage23 implementation or result inspection.
- Stage23 is a new experiment. Stage21/Stage22 checkpoints and folds 0--4 are development evidence only.
- The primary claim uses the official DiffusionDrive base checkpoint, a fresh Stage23 GRPO generator, and one deployable learned selector. It does not use a base-generator fallback or an online PDM oracle.

## Objective

Stage16--Stage21 showed that GRPO can improve the candidate set, but the gain did not reliably survive cross-log selection. Stage23 therefore aligns training with deployment: the selector sees complete trajectories and predicts PDM components, while GRPO optimizes the trajectory selected from a complete 20-mode set.

The primary quantity is the paired PDMS difference between the fresh Stage23 generator and the official base generator under the same Stage23 selector:

`GRPO effect = C - B`.

Selector gain is reported separately as `B - B0`; generator and selector gains must not be conflated.

## Locked system definition

### Inference graph

1. One generator produces all 20 complete trajectories.
2. A trajectory-aware selector scores all 20 candidates jointly.
3. The selector keeps the reference top-1 candidate unless a challenger passes the calibrated safety gate and its conservative improvement estimate is positive.
4. No frozen base generator, PDM scorer, labels, future observations, or evaluation feedback are available at inference.
5. The inference configuration must declare `inference_selector_source=trajectory_oof`.

### Trajectory-aware selector

- Five independently initialized selector members are trained with whole-log bagging.
- Each member excludes one deterministic 20% bucket of complete logs. Predictions used to fit calibration thresholds must be out-of-fold for that log.
- At inference the five members are ensembled. Their mean is the prediction and their disagreement contributes to the conservative lower bound.
- Inputs are candidate trajectory geometry and kinematics, BEV features sampled/cross-attended along that trajectory, agent/ego/status context, reference classification logit, and mode embedding.
- Outputs are collision, drivable-area compliance, TTC, progress, comfort, direction compliance, and aggregate PDMS estimates for every mode.
- Safety classification uses focal BCE with gamma 2 and negative/positive class weights `[1, 50]`.
- Continuous components and PDMS use Huber regression.
- Ranking uses all candidate pairs with absolute target gap at least `0.01`, weighted by the target gap.
- Calibration uses out-of-fold residuals only. The joint unsafe-switch threshold must have a one-sided Clopper--Pearson upper bound no larger than `0.005`.
- A challenger is eligible only when safety passes and the 95% lower confidence bound of its predicted improvement over the reference is greater than zero.

### Set-level GRPO

- Algorithm name: `diffgrpo_selected_set`.
- For each scene, sample `G=8` complete 20-mode candidate sets.
- The frozen Stage23 selector selects exactly one trajectory from every set.
- PDM is evaluated only on the eight selected trajectories, followed by within-scene group z-score normalization.
- Gradients flow only through the selected denoising chains; selector parameters and all unselected chains are frozen/detached.
- A same-noise official-base trajectory is used during training only as a paired absolute guard, never at inference.
- Safety regression or `candidate - base < -0.01` assigns advantage `-2`. On mature base scenes (`base PDMS >= 0.75`), `candidate - base < -0.002` also assigns `-2`.
- Exact denoising-policy KL coefficient is `0.1`, raised to `0.5` for mature-negative samples.
- The trainable set remains the 64 Stage21 regression/diffusion-decoder tensors. Classification, selector, perception, and reference policy parameters are frozen.
- Learning rates are `1e-7` for decoder layer 0 and `1e-6` for layer 1. BC weight is `0.1`, diffusion discount is `0.6`, optimizer states and reward arithmetic are FP32, optimizer is AdamW, and global batch size is 64.
- Candidate epochs are locked to 2, 4, 6, and 8. No later epoch may be introduced after inspecting validation results.

## Data protocol

### Candidate bank and selector development

- Build the official-base candidate bank from existing whole-log folds 0--3.
- Use three fixed noise namespaces: `default`, `20260811`, and `20260812`.
- Store all 20 trajectories, reference logits, per-candidate PDM components, aggregate PDMS, token, log identifier, namespace, generator/checkpoint hash, and feature/config provenance.
- Whole logs, not individual frames, define all split and bagging boundaries.

### Stage21 diagnostic-only gate

Before any fresh Stage23 GRPO training, train the selector only on the official-base bank and evaluate it on the consumed Stage21 epoch-8 fold-4 candidates. This diagnostic cannot support a final paper claim.

All conditions must pass:

- paired selector gain at least `+0.003`;
- paired confidence interval strictly above zero;
- every fixed namespace has non-negative mean gain;
- collision, TTC, and drivable-area rates are each no worse by more than `0.001`;
- one-sided catastrophic-switch upper bound no larger than `0.005`.

Failure stops Stage23 before generator training and triggers selector redesign. It must not be compensated by extra epochs.

### Fresh fold-4 generator gate

If the selector diagnostic passes, train a fresh generator from the official base using folds 0--3. Choose among epochs 2/4/6/8 on fold 4 with namespaces `default` and `20260813`.

All conditions must pass for `C - B` under the identical frozen selector:

- positive mean in every namespace;
- pooled mean at least `+0.005` with confidence interval strictly above zero;
- hard-scene mean at least `+0.010`;
- mature-scene mean at least `-0.002`;
- collision, TTC, and drivable-area rates each no worse by more than `0.001`;
- catastrophic-switch upper bound no larger than `0.005`.

### Protected fold-5 gate

After selecting the epoch, retrain once on folds 0--4 and evaluate fold 5 with namespaces `default`, `20260814`, `20260815`, and `20260816`. The same fold-4 thresholds apply. Fold 5 is opened once for this locked recipe.

Only a fold-5 pass authorizes full-data training on all 6,119 scenes, dev-confirm evaluation, and NavTest submission.

## Final attribution table

- `A`: official base, short evaluation, official selector.
- `B0`: official base, full evaluation, official selector.
- `B`: official base, full evaluation, Stage23 selector.
- `C`: Stage23 GRPO, full evaluation, Stage23 selector.
- `D`: Stage23 GRPO, full evaluation, official selector (diagnostic only).

Report `C - B` as the primary GRPO effect and `B - B0` as the selector effect. A gain of `+0.005` is useful, `+0.010` is paper-level, and `+0.015` is the desired target. No NavTest claim is permitted without rows B and C evaluated with identical token sets, noise protocol, selector hash, and evaluator configuration.

## Required implementation and audits

- Permutation equivariance over the 20-mode set.
- Demonstrable dependence on trajectory inputs.
- Whole-log OOF isolation and deterministic bag membership.
- OOF-only calibration and Clopper--Pearson safety audit.
- Static/runtime proof that inference cannot call PDM or a base generator.
- Exactly eight sets per GRPO scene and exactly one selected chain per set.
- Common-noise paired-base guard, exact KL audit, frozen-module audit, and selected-chain gradient audit.
- Identical selector checkpoint SHA for B and C.
- Regression coverage for legacy Stage13--Stage22 configurations.
- Unit/smoke tiers U8, U32, and U64 before long training.

## Compute escalation

Local compute is used for implementation, unit tests, selector diagnostic, and short smoke jobs. External 8x H100 or 8x 4090 training is requested only after the selector-only diagnostic passes and the fresh Stage23 recipe survives local U64/fold-4 smoke validation. A distributed run must use the frozen manifest, config hashes, and checkpoint candidates recorded here.

## Stop rules

- Do not extend Stage21 or Stage22 checkpoints for the primary experiment.
- Do not run more generator epochs to repair a failed selector diagnostic.
- Do not tune thresholds on fold 5, dev-confirm, or NavTest.
- Do not hide a negative namespace, safety component, mature subset, or catastrophic-tail result behind a pooled mean.
- Do not replace the primary no-fallback system with an oracle or base-generator fallback. Fallback is allowed only as a labeled ablation.

## Execution record

Implementation and results will be appended below without altering the locked definitions or thresholds above.

### 2026-07-22 implementation freeze

- Added a separate five-member `Stage23TrajectorySelector`; the Stage15 top-2
  selector remains unchanged for historical reproducibility.
- Each member owns its complete trajectory/context encoder.  Training masks
  exclude exactly the deterministic bucket assigned to the complete log.
- Added all-20 conservative inference, trajectory/mode permutation tests,
  focal unsafe-event supervision, six-component/PDMS Huber losses, and
  gap-weighted all-pair ranking.
- `trajectory_oof` inference executes one generator and no PDM/base-generator
  call.  PDM used by the fixed-set evaluator is explicitly outside the model
  inference graph.
- Frozen selector fit manifest: 4,075 tokens from 606 whole logs in Stage21
  folds 0--3; ordered token SHA256
  `a1e8a6d8c89c4325830abf125672c962542ac9ec481587bb9e1a1ed8c392c6cc`.
- Selector training is fixed at eight epochs with learning rate `1e-4`, FP32,
  batch size 2 (three bank namespaces per scene, effective selector batch 6).
  The epoch-8 checkpoint is used; fold4 is not used to choose a selector epoch.
- Candidate-bank generation began on three local RTX 4090 GPUs for namespaces
  `default`, `20260811`, and `20260812`.  Generator training remains forbidden
  until OOF calibration and the Stage21 fold4 diagnostic pass.

### 2026-07-22 selector diagnostic result

- Completed and shape-audited all three official-base banks: 4,075 tokens,
  20 trajectories per token, and zero evaluation failures.  Their SHA256
  values are `52418e39...f39bc`, `ec7be9e8...915c5`, and
  `f143179c...a3c2` for default, `20260811`, and `20260812` respectively.
- Trained the fixed eight-epoch selector.  Total epoch loss decreased
  monotonically from `0.614210` to `0.401909`; the locked epoch-8 checkpoint
  is `grpo-07-16304.ckpt`, SHA256 `ec89accd...bcf4c`.
- Whole-log single-held-out-member calibration used 12,225 scene-namespace
  records.  It selected residual margin `0.2168758482` and safety threshold
  `0.2811534405`; OOF calibration gain was `+0.0050280` with catastrophic
  Clopper--Pearson upper bound `0.003233 < 0.005`.
- The consumed Stage21 epoch-8 fold-4 diagnostic **failed the preregistered
  selector gate**.  Across 2,042 paired records, pooled gain was only
  `+0.0007056`, bootstrap CI `[-0.0007801, +0.0023642]`.  Namespace gains
  were `+0.0000529` and `+0.0013584`.  Safety-mean, per-namespace nonnegative,
  and catastrophic-upper-bound checks passed, but the `+0.003` gain and
  strictly-positive-CI checks failed.
- Generator training was therefore not started.  Fourteen total switches
  exposed a repeated cross-generator hard negative: token
  `d6d624b818c05333` was incorrectly admitted in both namespaces, including
  one `0.79896 -> 0.0` collision/TTC failure.  Low five-member disagreement
  on this failure indicates correlated-member overconfidence plus candidate
  distribution shift, not insufficient selector epochs.
- The threshold calibrator was changed from an intractable nested scan to an
  exactly equivalent descending eligibility-event sweep.  The candidate
  threshold set and all preregistered pass criteria were unchanged.
