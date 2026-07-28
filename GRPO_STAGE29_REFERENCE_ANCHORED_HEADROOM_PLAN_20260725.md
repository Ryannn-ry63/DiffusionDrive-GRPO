# Stage29: Reference-Anchored Headroom GRPO on the Public 88.1 Baseline

## Decision status

This plan is frozen on 2026-07-25 after the completed Stage28 fold4 gate and
before any Stage29 code change, training, or evaluation. Stage29 keeps the
scientific question fixed: can GRPO improve the released 88.1041-PDMS
DiffusionDrive checkpoint without sacrificing mature scenes or creating tail
failures?

Stage29 does not continue A3 or any other Stage28 checkpoint. Every trainable
generator restarts from the released public checkpoint. The primary paper
system remains a single GRPO generator; an exact two-generator fallback is a
secondary ablation only.

## Evidence and scope

- Stage26 established a statistically significant `+1.1675` NavTest PDMS-point
  generator contribution over the local 84.916 baseline.
- Stage27 and Stage28 retained positive public-baseline signals of about
  `+0.25`--`+0.30` points, but the gains were not statistically stable and
  regressed mature scenes.
- Stage28 branches B/C made more than 96% of paired rollouts neutral and learned
  essentially no uplift.
- Stage28 A3 improved hard scenes by about `+1.83` points, but its mature-scene
  delta was negative and it contained two catastrophic regressions.
- Even an oracle choice between the final A3 trajectory and the exact public
  trajectory has only `+0.004116` raw-PDMS fold4 headroom. Selector/fallback
  work alone therefore cannot meet the `+0.005` target; the generator must
  create a broader improvement first.

## Stage29 objective

For each valid common-random current/reference group of eight selected chains:

1. `delta_i = current_reward_i - reference_reward_i`.
2. `q` is the mean valid reference reward for the scene.
3. `headroom = clamp((0.90 - q) / 0.15, 0, 1)`.
4. `z_reward` is the group-normalized current reward with epsilon `1e-3`,
   clipped to `[-2, 2]`.
5. `paired_scale = max(std(delta), 0.002)`.
6. `paired_i = clip(delta_i / paired_scale, -2, 2)`.
7. `raw_adv_i = paired_i + 0.5 * headroom * z_reward_i`.
8. Preserve paired sign: positive delta cannot receive negative advantage,
   negative delta cannot receive positive advantage, and exact delta uses only
   the headroom-weighted group term.
9. Any collision, drivable-area, or TTC regression relative to the paired
   reference receives advantage `-2`.
10. Clip final advantages to `[-2, 2]`.

Policy loss is normalized within each active scene before averaging across
active scenes. The denoising-step discount remains `0.6`.

Two and only two new pilot branches are allowed:

- `H`: the hybrid advantage above, fixed BC weight `0.1`, regular KL weight
  `0.1`, and safety-override KL weight `0.5`.
- `HC`: the primary method, with
  `BC=0.05+0.05*(1-headroom)` and
  `KL=0.05+0.45*(1-headroom)`; safety overrides use KL `0.5`.

Both branches train only the registered 64 decoder tensors, use the frozen
Stage27 S-multi selector, and keep the public reference, selector, perception,
and classification modules frozen. S-public is forbidden.

## Pilot and fold4 selection

Run H and HC as independent seed-0 8xH100 jobs on the ordered 4,075 folds0--3
tokens. Both use batch size 1 per GPU, gradient accumulation 8, FP32, learning
rate `1e-6`, group size 8, schedule `[32,24,16,8,0]`, and exactly four epochs.
Save epochs 1--4. Do not extend a failed branch.

Before fold4, require finite metrics, exactly 64 changed decoder tensors, zero
forbidden gradients, average active-scene fraction in `[0.15,0.95]`, positive
advantage fraction at least `0.01`, and neutral fraction at most `0.85`.

Evaluate public P, historical A3, H1--H4, and HC1--HC4 under S-multi with
namespaces `20261211` and `20261212`. A Stage29 checkpoint is eligible only if:

- pooled selected `C-P >= +0.003`;
- whole-log bootstrap 95% CI lower bound is positive;
- both namespace means are positive;
- 10% trimmed mean is nonnegative and wins exceed losses;
- hard-scene gain is at least `+0.010`;
- mature-scene gain is at least `-0.0002`;
- candidate-mean delta is positive and oracle delta is at least `-0.0005`;
- every PDMS component delta is at least `-0.0005`;
- catastrophic-regression count is zero;
- all finite-value, token, checkpoint, selector, and SHA checks pass.

Record `+0.005` as the strong gate. Rank eligible checkpoints by whole-log CI
lower bound, trimmed mean, pooled mean, then earlier epoch. Historical A3 is a
control and cannot be selected. If no Stage29 checkpoint passes, stop without
changing objective constants after inspecting fold4.

## Confirmation, selector closure, and NavTest

Retrain the selected branch from the public checkpoint on folds0--4 for the
fixed selected epoch. Confirm once on fold5 under namespaces
`20261221`, `20261222`, and `20261223`, applying the complete generator gate.
Failure stops Stage29.

After a fold5 pass, collect three all-20 candidate banks for the public and
Stage29 generators on folds0--3. Retrain a Stage24/25 eight-member
cross-generator selector using the new and historical multi-domain banks.
Calibrate on fold4 namespaces `20261213` and `20261214`, then replay unchanged
on fold5. The old S-multi selector and the new closure selector are the only
eligible choices. Both must preserve zero catastrophes, component guards, and
`C-B >= +0.003` with positive whole-log CI. Rank by C-B CI lower bound and mean;
within `0.0005`, retain old S-multi.

Freeze method, epoch, selector, configs, manifests, and all SHA256 values.
Restart from the public checkpoint and train once on all 6,119 rewardable
scenes. The one-shot NavTest systems are:

- `P`: released public 88.1041 system;
- `P0`: public generator, full schedule, original logits;
- `B`: public generator, full schedule, final selector;
- `C`: Stage29 generator, full schedule, the same final selector.

The primary claim is `C-B`; the complete-system claim is `C-P`. Success requires
`C-B >= +0.005`, a positive whole-log CI lower bound, and `C-P > 0`. The stretch
target is `C-B >= +0.010`. No method or threshold changes are allowed after
NavTest inspection.

## Secondary two-generator ablation

Only after the single-generator system is frozen, train an eight-member
cross-generator router on folds0--3. It defaults to exact public system B and
admits C only when predicted delta LCB is positive, all safety-risk UCBs pass,
and OOD distance passes. Calibrate once on fold4 and confirm on fold5. It must
be nonnegative versus C, at least `+0.003` versus B, and have zero catastrophes.
Failure does not block the primary single-generator result and cannot change it.

## Required verification

- Unit tests cover hard/mature ties, all-better, all-worse, small signed deltas,
  sign preservation, safety overrides, headroom endpoints, conditional BC/KL,
  non-finite inputs, and shape failures.
- Regression tests prove Stage23--28 routes are unchanged.
- Real optimizer-step audits prove exactly 64 decoder tensors change and all
  forbidden modules remain bitwise frozen.
- Gate tests cover namespace, CI, trimmed mean, hard/mature buckets, candidate,
  oracle, component, catastrophe, and provenance failures.
- Every report stores per-token, per-log, per-namespace, bucket, component,
  bootstrap, checkpoint, selector, and manifest provenance and can be replayed
  independently from immutable JSON artifacts.
