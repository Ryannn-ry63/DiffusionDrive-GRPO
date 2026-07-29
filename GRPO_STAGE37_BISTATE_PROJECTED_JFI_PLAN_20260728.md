# Stage37: Bi-State Projected Deployment GRPO and JFI Selection

Date: 2026-07-28

Baseline: released `diffusiondrive_navsim_88p1_PDMS`

## Frozen diagnosis

Stage36 RGT-NCD produced a statistically positive two-fold OOF selected
trajectory gain over the released public checkpoint:

- selected gain: `+0.001824692`;
- whole-log 95% bootstrap CI: `[+0.000692022, +0.003212392]`;
- candidate-mean gain: `+0.002200048`;
- hard-scene gain: `+0.008653745`;
- mature-scene gain: `-0.000058743`.

It did not pass promotion because the public top-five modes regressed by
`-0.001064840`. In addition, the frozen Stage25 selector left substantial
headroom: the raw oracle exceeded the deployed result by about `0.0913`, while
the safe-eligible oracle exceeded it by only `0.000036`. The current
conjunction of separate risk and reward-delta confidence bounds therefore
admits only about `1.03` candidates per scene including fallback and rejects
nearly every raw-oracle candidate.

Stage37 treats these as two independent research questions:

1. can GRPO improve deployable trajectory generation while preserving the
   strong public model in function space;
2. can a selector estimate the *joint* event that a candidate is both safe
   and materially better than fallback, instead of intersecting separately
   calibrated marginal tests.

The two contributions are evaluated factorially. A selector gain is not
reported as a GRPO gain, and selector improvements may not conceal a failed
generator constraint.

## Contribution A: Bi-State Projected Deployment GRPO

The Stage37 generator training mode is
`stage37_bistate_projected_deployment_grpo` (BPD-GRPO).

It is initialized independently from the released public checkpoint. The
frozen reference is the same public checkpoint. Perception, classification,
selector, and all non-registered decoder parameters remain frozen. The exact
Stage35 nested counterfactual deployment credit is retained. The Stage36
same-anchor tail term remains an auxiliary objective and is not claimed as a
new contribution.

For every microbatch, Stage37 builds two differentiable objectives over the
registered 64 diffusion-decoder tensors:

- **reward objective**: nested deployed-decision GRPO plus safe same-anchor
  headroom expansion;
- **preservation objective**: paired public-top-five consistency evaluated at
  both the current denoising state and the exact public reference state.

The reward and preservation gradients are accumulated separately across all
eight gradient-accumulation microbatches and all DDP ranks. Immediately before
the optimizer step:

1. let `g_r` be the synchronized reward gradient and `g_p` the synchronized
   preservation gradient;
2. if `dot(g_r, g_p) < 0`, replace `g_r` with
   `g_r - dot(g_r,g_p) / (||g_p||^2 + eps) * g_p`;
3. if the synchronized current-batch public-top-five paired PDM delta is below
   `-1e-4`, add `0.25 * g_p`;
4. write the resulting gradient only to the registered 64 decoder tensors.

All top-five public modes participate in preservation, including modes that
have not yet crossed the regression tolerance. Candidate rewards and
component scores remain detached from the gradient graph. Logs must expose
the pre-projection dot product and cosine, conflict indicator, projection
norm, preservation-gradient norm, recovery activation, paired top-five PDM
delta, and the final finite-gradient check.

This contribution is specifically the combination of:

- frozen-selector counterfactual deployment credit;
- bi-state functional preservation at current and reference denoising states;
- PDM-triggered gradient projection and recovery.

Stage37 does not introduce multiplicative diffusion noise, and same-anchor
grouping is not presented as the novelty.

### Frozen generator constants

- banks per scene: `8`;
- modes per bank: `20`;
- selector-active width: `4`;
- public frontier width: `5`;
- same-anchor elite count: `2`;
- learning rate: `1e-6`;
- optimizer epochs: `4`;
- optimizer steps per epoch: `48`;
- gradient accumulation: `8`;
- freeze steps: `48, 96, 144, 192`;
- global bucket composition: `[2, 30, 8, 24]`;
- NCD coefficient: `0.25`;
- same-anchor tail coefficient: `0.25`;
- frontier auxiliary coefficient: `0.25`;
- advantage scale: `0.002`;
- advantage clip: `2.0`;
- public-frontier regression tolerance: `1e-4`;
- projection recovery coefficient: `0.25`;
- projection epsilon: `1e-12`;
- mature-scene positive exploration multiplier: `0.0`;
- BC weight: `0.1`;
- denoising schedule: `[32, 24, 16, 8, 0]`;
- trainable boundary: the registered 64 diffusion-decoder tensors only.

## Contribution B: Joint Feasible Improvement selector

The Stage37 selector source is `joint_feasible_improvement_v1` (JFI). It
freezes the Stage24 trajectory/context encoder and trains eight independent
heads. Each head predicts:

- the probability that a candidate is jointly safe and improves reward by at
  least `+0.005` relative to the scene fallback;
- reward-delta quantiles `q10`, `q50`, and `q90`.

The positive joint label requires all of:

- candidate PDM minus fallback PDM at least `+0.005`;
- collision, drivable-area compliance, and TTC each no worse than fallback by
  more than `0.0005`;
- no catastrophic PDM regression of `-0.5` or worse.

The frozen loss weights are `4:1:1` for joint focal loss, safe quantile loss,
and within-scene pair ranking. The learning rate is `1e-4`, batch size is `2`,
and training lasts `12` epochs.

The eight heads are trained from the pre-Stage37 multi-domain candidate banks
for Public, DPEL192, NCD192, and RGT192. Stage37 generator outputs may be added
only to calibration after generator training; they may not update JFI head
weights.

### Frozen cross-fitting

- holdout fold 0: folds 2 and 3 train, fold 1 calibrates, fold 0 tests;
- holdout fold 1: folds 2 and 3 train, fold 0 calibrates, fold 1 tests.

Calibration scans the joint-probability threshold and optional lower-quantile
floor with safety as the lexicographic first objective. A deployable
calibration must satisfy:

- one-sided 95% catastrophic-rate upper bound at most `0.005`;
- collision, drivable, and TTC deltas each at least `-0.0005`;
- average candidate pool including fallback between `2` and `4`;
- at most four candidates per scene.

At inference, candidates passing the calibrated joint threshold and the
existing frozen OOD guard are ranked by predicted `q10`. The public fallback
is selected when no candidate passes. Calibration fails closed when no
threshold satisfies every safety and pool-size requirement.

## Factorial pilot

For holdout folds 0 and 1 and evaluation namespaces `20261511` and
`20261512`, Stage37 evaluates the preregistered `2 x 2`:

1. Public generator + Stage25 selector;
2. Stage37 generator + Stage25 selector;
3. Public generator + JFI selector;
4. Stage37 generator + JFI selector.

Stage35 NCD192 and Stage36 RGT192 remain frozen historical controls. Every
result records checkpoint SHA256, selector SHA256, calibration SHA256,
manifest/fold SHA256, evaluation namespace, and code-plan SHA256.

## Promotion gates

### Generator-only gate

Using the frozen Stage25 selector, a Stage37 checkpoint passes only if:

- selected gain versus Public is at least `+0.0020`;
- whole-log bootstrap 95% CI lower bound is strictly positive;
- both holdout-fold means and both namespace means are strictly positive;
- public-top-five, candidate-mean, raw-oracle, and safe-oracle deltas are all
  nonnegative;
- hard-scene gain is at least `+0.008`;
- mature-scene gain is at least `-0.0001`;
- selected gain versus Stage36 RGT192 is nonnegative.

### Full-system gate

The Stage37 generator plus JFI selector passes only if:

- gain versus Public plus Stage25 is at least `+0.005`;
- generator effect under the same JFI selector,
  `Stage37+JFI - Public+JFI`, is at least `+0.001`;
- whole-log bootstrap 95% CI lower bound is strictly positive;
- both holdout-fold means and both namespace means are strictly positive;
- collision, drivable, and TTC deltas are each at least `-0.0005`;
- wins exceed losses, 10% trimmed mean is nonnegative, and catastrophic
  regression count is zero;
- calibrated mean candidate-pool size is between `2` and `4`.

The stretch target is `+0.009`, corresponding to approximately `89.0` PDM
from the released `88.1` baseline. It is not a pass/fail requirement.

If the full-system gain is large but the same-selector generator effect is
below `+0.001`, the result is reported only as a selector improvement and not
as evidence that GRPO improved DiffusionDrive.

## Audit, stop, and routing rules

Before formal training, a real eight-GPU one-step audit must prove:

- the public checkpoint initializes both trainable policy and frozen
  reference exactly;
- the 64-tensor decoder boundary is exact;
- reward and preservation gradients are finite and nonzero;
- accumulation and DDP synchronization are complete before projection;
- a negative gradient dot product produces the registered orthogonal
  projection;
- a top-five delta below `-1e-4` activates exactly `0.25` recovery;
- positive reward exploration is zero on mature scenes;
- selector, perception, classification, reference, and unregistered
  parameters have zero gradients and zero checkpoint changes;
- every trace, reward, advantage, component score, and provenance field is
  finite.

The pilot stops without threshold or loss-weight tuning if public-top-five
gain is negative at all four checkpoints. If the generator passes but JFI
fails, the generator is frozen and later selector work starts in a new stage.
If JFI passes while the generator fails, only the selector result is retained.
If no JFI calibration satisfies safety and pool-size constraints, deployment
falls back to Stage25.

No folds 2/3 expansion and no navtest evaluation are permitted until both
generator-only and full-system pilot gates pass unchanged.
