# Stage29 pre-fold4 training-signal gate amendment (2026-07-25)

## Status and timing

This amendment is frozen after both H and HC completed their preregistered
four-epoch folds0--3 training and before any Stage29 fold4 evaluation artifact
or report existed. The user authorized correcting the confirmed audit error.
No fold4 PDMS result has been generated or inspected.

The original Stage29 plan remains immutable with SHA256
`d59f9161f2d9c2d4e99d261db788206dad19412b9569c47a7703b37da121c28b`.
This document is a narrow amendment to its training-signal diagnostic only.

## Confirmed error

The original formal health rule required average
`policy_active_scene_fraction` in `[0.15, 0.95]`. The metric is implemented as
`active.any(dim=-1).float().mean()`: a scene is active when at least one of its
eight paired rollouts has a nonzero advantage. It is not the fraction of
individual rollouts with a nonzero advantage.

Stage29 deliberately defines a dense sign-preserving paired objective. With
roughly 69--72% of individual rollouts non-neutral, almost every eight-rollout
scene is expected to contain at least one active rollout. Therefore a 0.95
upper bound rejects the intended objective even when positive, negative, and
neutral rollouts are all well represented. It does not diagnose collapse.

The immutable completed formal runs showed:

- H: active scene 0.984375, positive rollout 0.342041015625, negative rollout
  0.35040283203125, neutral rollout 0.30755615234375.
- HC: active scene 0.98486328125, positive rollout 0.4144287109375, negative
  rollout 0.30902099609375, neutral rollout 0.27655029296875.

Both runs completed global steps `[64, 128, 192, 256]`, wrote all four epoch
checkpoints, retained finite metrics, and reached the health check only after
checkpoint, gradient, reference, and selector audits succeeded. Their frozen
checkpoint records are:

- H freeze SHA256:
  `00799206d602098af3b5fc52fef9ecf89b9d1218e92824fc3077883de68d7129`.
- HC freeze SHA256:
  `7676c51709bf37e6e06439c62da5a7430de0a8f16518e477d29d5b651f278fc4`.

## Frozen correction

Replace only the erroneous active-scene interval check:

- old: `0.15 <= policy_active_scene_fraction_mean <= 0.95`;
- corrected: `policy_active_scene_fraction_mean >= 0.15`.

Retain without change:

- `positive_fraction_mean >= 0.01`;
- `neutral_fraction_mean <= 0.85`;
- all fractions finite and within `[0, 1]`;
- exactly 64 changed decoder tensors and zero forbidden changes;
- finite rewards, advantages, log probabilities, and optimizer states;
- frozen public reference and S-multi selector bitwise equality;
- every training constant, checkpoint, manifest, selector, and SHA;
- every fold4 selected/candidate/oracle, CI, namespace, hard/mature, component,
  catastrophe, and provenance gate.

No loss, regularization coefficient, training epoch, learning rate, checkpoint,
selector, evaluation namespace, or PDMS threshold is changed. H and HC must not
be retrained. Their immutable completed checkpoints are re-audited under this
corrected diagnostic, and only passing re-audits may enter the still-unseen
fold4 grid.
