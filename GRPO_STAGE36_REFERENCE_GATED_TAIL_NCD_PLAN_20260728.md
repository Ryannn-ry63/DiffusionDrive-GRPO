# Stage36: Reference-Gated Tail NCD-GRPO

Date: 2026-07-28

Baseline: released `diffusiondrive_navsim_88p1_PDMS`

## Scientific objective

Stage35 NCD-GRPO produced a statistically positive two-fold OOF selected
trajectory gain over the released public checkpoint:

- selected gain: `+0.001342`;
- whole-log 95% bootstrap CI: `[+0.000399, +0.002521]`;
- hard-scene gain: `+0.005226`;
- mature-scene gain: `+0.000271`.

It nevertheless reduced the raw oracle by `-0.000167` and the reward of the
public model's original top-five modes by `-0.000906`. Stage36 therefore keeps
NCD's deployment counterfactual credit unchanged and adds a training-only,
reference-gated upper-tail objective. The selector remains frozen.

The method is Reference-Gated Tail NCD-GRPO (RGT-NCD):

1. retain the exact Stage35 outer deployed-decision and inner
   counterfactual-deployment advantages;
2. identify the public top-five reward modes independently inside every
   sampled bank;
3. when a paired current top-five mode regresses by more than `1e-4`, replay
   the exact public denoising chain as a reference-retention teacher;
4. within each anchor only, identify the top two current rollouts among the
   eight banks and reward them only when they exceed the public same-anchor
   top-two mean by at least `0.001` without a safety regression;
5. deduplicate all `(bank, mode)` chain replays before policy, KL, and teacher
   likelihood evaluation.

This differs from vanilla DiffGRPO and intra/inter-anchor GRPO. The claimed
contribution is the combination of frozen-selector counterfactual deployment
credit with a paired public-frontier constraint. Stage36 does not introduce
multiplicative exploration noise or train a selector.

## Frozen method

Stage36 is initialized independently from the fold-matched Stage32 DPEL192
checkpoint. Stage35 NCD192 is a frozen comparison, not the initializer. This
avoids inheriting its public-frontier regression and keeps the causal
attribution:

`Public -> DPEL192 -> NCD-GRPO -> RGT-NCD`.

The frozen constants are:

- banks per scene: `8`;
- modes per bank: `20`;
- NCD selector-active width: `4`;
- public frontier width per bank: `5`;
- same-anchor elite count: `2`;
- public-frontier regression tolerance: `1e-4`;
- tail improvement margin: `0.001`;
- tail/reference normalization scale: `0.002`;
- tail policy coefficient: `0.25`;
- adaptive retention coefficient:
  `0.25 * clip((public-current-1e-4)/0.002, 0, 2)`;
- NCD counterfactual coefficient: `0.25`;
- mature-scene positive multiplier: `0.25`;
- safety regression advantage: `-2.0`;
- BC weight: `0.1`;
- reference KL weight: `0.1`;
- safety KL weight: `0.5`;
- denoising schedule: `[32, 24, 16, 8, 0]`;
- learning rate: `1e-6`;
- trainable boundary: the registered 64 decoder tensors only;
- four epochs, 48 optimizer steps per epoch;
- freeze steps: `48, 96, 144, 192`.

All candidate rewards and components are computed outside the gradient graph.
No cross-anchor sample may enter a tail advantage group. Retention remains
fully active on mature scenes; only positive exploration and positive NCD
credit are attenuated there.

## Signal and integrity gate

Before formal training, a one-step audit must prove:

- all Stage35 NCD trace invariants still pass;
- public top-five masks contain exactly five finite modes per bank;
- current and public top-two tail groups contain exactly two rollouts per
  anchor;
- at least 1% of public-frontier entries activate retention;
- at least 0.5% of anchor entries activate safe tail expansion;
- a tail-positive entry always beats the paired public anchor threshold;
- a retention-positive entry always corresponds to a public top-five
  regression;
- replay deduplication preserves the unreduced loss and log-prob values;
- rewards, advantages, log-probs, component scores, and coefficients are
  finite;
- reference, selector, perception, classification, and all unregistered
  parameters have zero gradients and zero checkpoint changes.

Failure stops Stage36 before formal training.

## Pilot and promotion gates

Train fold0/fold1 independently and evaluate Public, fold-matched DPEL192,
fold-matched NCD192, and RGT-NCD steps 48/96/144/192 under evaluation noise
namespaces `20261511` and `20261512`.

A pilot checkpoint is promotable only if all checks pass:

- selected gain versus Public at least `+0.0015`;
- selected gain versus DPEL192 at least `+0.0005`;
- selected gain versus NCD192 at least `+0.00015`;
- both fold means and both namespace means strictly positive;
- whole-log bootstrap 95% CI lower bound strictly positive;
- candidate mean and raw oracle deltas versus Public nonnegative;
- public-top-five and same-anchor top-two tail deltas nonnegative;
- safe-deployable oracle gain versus Public at least `+0.0005`;
- hard-scene selected gain versus Public at least `+0.0045`;
- mature-scene selected gain versus Public at least `-0.0001`;
- collision, drivable, and TTC deltas each at least `-0.0005`;
- trimmed mean nonnegative, wins greater than losses, and zero catastrophic
  regressions.

Only a passing pilot expands unchanged to folds2/3. Four-fold promotion keeps
the prior frozen requirements of at least `+0.003` versus Public and `+0.001`
versus DPEL192, with no negative fold, namespace, frontier, mature, or safety
gate. Navtest remains blocked until four-fold promotion.

## Stop and routing rules

- If raw oracle or public top-five remains negative for every checkpoint,
  reject RGT-NCD and do not tune the selector.
- If the generator ceiling gates pass but selected gain fails, freeze the
  generator and create a later independent selector stage.
- If selected gain passes but a ceiling gate fails, continue generator work;
  selector changes may not hide the failed ceiling.
- No loss-weight scan, post-hoc threshold change, or navtest probing is
  permitted after pilot results are observed.

