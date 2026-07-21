# Stage19: Selected-Anchor DiffGRPO

Date: 2026-07-21
Status: stopped by the preregistered internal-test safety gate; protected sets unopened

## 1. Scientific objective and attribution

Stage19 keeps one primary claim: GRPO must directly improve the DiffusionDrive
generator and therefore the final trajectory chosen by an unchanged frozen
selector.  The four locked evaluation cells are:

- A: frozen base, original `[8, 0]` schedule;
- B: frozen base, full `[32, 24, 16, 8, 0]` schedule;
- C: Stage19 GRPO generator, full schedule, frozen reference selector;
- D: C plus the already frozen Stage17 base fallback.

`C-B` is the isolated GRPO contribution, `D-C` is the fallback contribution,
and `D-A` is the total deployed system gain.  Stage19 does not jointly train or
tune a selector, verifier, reward model, or oracle at inference.

## 2. Locked method

Stage16 grouped one rollout from each of 20 non-exchangeable anchor modes.
Stage19 instead implements ReCogDrive-style group-relative optimization under
the deployed DiffusionDrive condition:

1. The frozen full-chain base policy and its selector determine one deployment
   anchor for every training token using order-independent token-hash noise.
2. That same anchor is repeated for `G=8` stochastic full-chain rollouts.
3. Raw PDM rewards are standardized only within those eight same-token,
   same-anchor rollouts.
4. The advantage is applied to all five denoising actions with `gamma=0.6`.
5. One frozen-base teacher chain for the selected anchor supplies BC with weight
   `0.1`.
6. Only the regression/shared diffusion decoder is trainable.  Perception,
   classification, reference policy, value selector, and Stage17 risk head stay
   frozen.
7. Inference remains a single 20-mode rollout followed by the frozen reference
   selector.  It performs no multi-sample or oracle selection.

The new route is named `diffgrpo_selected_anchor`; legacy Stage16 behavior is
preserved unchanged.  Locked constants are FP32, AdamW, LR `1e-6`, group size
8, raw PDMS, BC `0.1`, gamma `0.6`, truncation 32, timesteps
`[32,24,16,8,0]`, scheduler steps 125, and at most 10 development epochs.
No LR, group-size, BC, gamma, schedule, or reward-shaping scan is permitted.

## 3. Data protocol

Map the 6,119 rewardable navtrain tokens to logs and deterministically assign
whole logs, seed `20260722`, to approximately 70/15/15 fit/calibration/test
partitions while minimizing token-count error.  Logs may not overlap.  Each
manifest records token, log, frozen selected mode, token/log hashes, base
checkpoint SHA, schedule, and source provenance.

The consumed dev-select3072 set is a stress test only and cannot select epochs
or tune thresholds.  Dev-confirm1024 and navtest remain unopened until their
registered gates.

## 4. Execution and gates

### Local audit

Run seed0 for 8 steps and resume to 32 steps on one GPU.  Require exact
same-anchor G=8 grouping, finite rewards/log-probabilities/losses/gradients,
nonzero allowed decoder gradients, zero gradient and weight change in every
frozen module, active BC, deterministic resume, and legacy Stage16/17
compatibility.

### Seed0 development

Train fit on 8 GPUs for 10 epochs, per-device batch 1, accumulation 8, retaining
epochs 1/2/4/6/8/10.  On log-calibration evaluate default noise and namespace
`20260723`.  An eligible checkpoint requires both `C-B>0`, two-noise mean
`C-B>=+0.003`, log-cluster bootstrap lower bound above zero, and aggregate
collision/drivable/TTC deltas nonnegative.  Select maximum two-noise mean;
within 0.0005 select the earlier epoch.

Open the internal log-test once with default and namespaces
`20260723/20260724/20260725`.  Default `C-B` and four-noise mean must each be
at least +0.005, every namespace must be positive, the clustered CI lower bound
must exceed zero, and aggregate safety deltas must be nonnegative.

### Dev-select stress and deployment branch

Evaluate only the locked seed0 checkpoint.  C requires `C-B>=+0.005`, token
CI lower bound above zero, `C-A>=+0.010`, and nonnegative safety deltas.
Then evaluate the frozen Stage17 epoch16 adapter at threshold
`0.7683204412460327`.  Use D only if `D-C>=0` and D-vs-B safety deltas are
nonnegative; otherwise lock C.  Do not retrain Stage17 or adjust the threshold.

### Formal seed0 and navtest

After passing stress, retrain seed0 from the original base on all 6,119
navtrain tokens for the locked epoch.  On the first and only dev-confirm opening,
the locked deployed cell E (C or D) requires `E-B>=+0.005`, `E-A>=+0.010`,
both token CI lower bounds above zero, and nonnegative safety deltas.  If E is D,
also require `D-C>=0`.  Only then generate the seed0 navtest submission; the
minimum target is +0.005 and the paper target is +0.010.

Seeds1/2 use the already locked configuration and epoch after seed0 navtest to
complete three-seed evidence.  They may not change the submitted method.

## 5. Tests and stop rules

Tests cover selected-mode manifests, log-disjoint assignment, same-anchor group
semantics, invalid/zero-variance reward groups, five-step replay, teacher BC,
frozen boundaries, DDP effective batch, resume/RNG continuity, noise namespace
order/shard invariance, artifact provenance, legacy checkpoints, and frozen
Stage17 reuse.

If C fails internal-test or dev-select, Stage19 stops; D cannot hide
`C-B<=0`.  If C passes but D fails, C proceeds alone.  Tail metrics are always
reported but are nonblocking under the registered average-primary objective.

## 6. Execution record

- 2026-07-21: plan recorded before Stage19 implementation, training, or result
  inspection.
- 2026-07-21: generated frozen-base full-chain selected modes for all 6,119
  navtrain tokens; constructed log-disjoint fit/calibration/test partitions of
  4,283/918/918 tokens with zero log overlap.
- 2026-07-21: selected-anchor route and provenance-locked evaluation wrappers
  passed 26 focused Stage16/17/19 tests.  U8 and resumed U32 audits passed:
  64 allowed decoder tensors changed, classification/reference/non-decoder
  mismatch counts were zero, and all required gradient/reward/log-probability
  diagnostics were finite and active.
- 2026-07-21: completed seed0 8-GPU development for 10 epochs / 670 optimizer
  steps.  All 670 metric steps and the final checkpoint passed the structural,
  frozen-boundary, finite-value, and active-gradient audits.
- 2026-07-21: calibration selected epoch 8 (`grpo-07-536.ckpt`).  Default
  `C-B=+0.014934`; two-noise mean `C-B=+0.016709`; log-cluster 95% CI
  `[+0.007842,+0.025614]`; collision/drivable/TTC deltas were all nonnegative.
- 2026-07-21: the single internal log-test opening produced positive `C-B` in
  all four namespaces (`+0.016394/+0.014029/+0.017190/+0.015317`), four-noise
  mean `+0.015732`, and log-cluster CI `[+0.007150,+0.024463]`.  The locked gate
  nevertheless failed because aggregate collision changed by `-0.0004085`:
  two collision degradations versus one partial improvement across 3,672
  token-noise pairs.  Per the pre-registered stop rule, dev-select,
  dev-confirm, Stage17-D, formal retraining, and navtest were not opened.
