# Stage31: Selector-Consistent Decision-Level GRPO

## Decision status

This plan is frozen on 2026-07-25 after the completed Stage30 four-fold OOF
stop and before any Stage31 code, training, or evaluation. A pre-implementation
feasibility review made one explicit correction before code existed: DPF exact
ties retain only the deployment-reward rank term so an exact public restart has
a non-zero GRPO bootstrap signal; DP exact ties remain zero. Stage31 keeps the
paper question fixed: can GRPO improve the released 0.88104106465-PDMS
DiffusionDrive system when the training reward is aligned exactly with the
trajectory deployed by the frozen Stage27 S-multi selector?

The paper-primary system remains one generator and the unchanged frozen
S-multi selector. A selector retrained after generator confirmation is a
secondary closure ablation only and cannot replace or redefine the primary
result.

## Motivation

Stage29 used eight stochastic 20-mode banks and a frozen selector, but its
public trajectory was gathered at the mode selected from the current bank.
The released public generator did not independently select its own deployed
mode. Stage30 instead rewarded all 20 modes and lifted low-ranked candidates
while slightly damaging the public frontier. Its best diagnostic checkpoint,
COV144, obtained `+0.000723` pooled selected gain and positive confidence
interval but only `+0.000011` candidate-max gain.

Stage31 changes credit assignment, not the DiffusionDrive architecture or the
GRPO research direction. It compares the exact deployed decisions of the
current and public systems under the same frozen selector.

## Common-random deployed-decision trace

For every scene, sample eight stochastic sets. Each set contains a complete
20-mode current bank and a complete 20-mode frozen-public bank. Corresponding
current/public modes share initial noise, transition noise, and final noise.

Run the frozen S-multi selector independently on both banks:

- `current_selected_mode = S(current_bank)`;
- `public_selected_mode = S(public_bank)`.

The modes are allowed to differ. Score the two selected trajectories, replay
only the eight current selected chains for policy log probabilities, replay
the eight public selected chains as the BC teacher, and compute exact
current/reference KL on the current selected states. Per-set reward is

`delta_i = PDMS(current_bank[current_selected_mode])
         - PDMS(public_bank[public_selected_mode])`.

The trace must expose both selected-mode tensors, their disagreement rate,
both deployed trajectories, common-noise diagnostics, sampled/replayed chain
counts, and selector provenance. The public reference, selector, perception,
classification, and all non-registered generator tensors remain frozen.

## Objectives

Run exactly two branches.

`DP` is the non-selectable deployment-paired GRPO ablation:

1. Group-normalize the eight valid deployment deltas with epsilon/floor
   `0.002`, clipping to `[-2, 2]`.
2. Preserve absolute deployment sign: a positive delta cannot receive
   negative advantage, a negative delta cannot receive positive advantage,
   and an exact tie receives zero.

`DPF` is the only selectable paper-primary branch:

1. Start with the same normalized deployment-delta term.
2. Let `q` be the mean valid public deployed reward and define
   `headroom = clamp((0.90 - q) / 0.15, 0, 1)`.
3. Add `0.5 * headroom * z(current_deployed_reward)`, where the reward z-score
   is computed within the eight current deployed samples and clipped to
   `[-2, 2]`.
4. Apply absolute deployment-sign preservation after combining the terms:
   positive deltas cannot receive negative credit and negative deltas cannot
   receive positive credit. An exact delta tie receives the rank term alone.
   This is required to bootstrap GRPO from an exactly identical public
   checkpoint/reference pair.

For both branches, any collision, drivable-area, or TTC regression against the
paired public deployed trajectory forces advantage `-2`. Final advantages are
clipped to `[-2, 2]`. Policy loss is normalized within each active scene;
denoising discount is `0.6`. BC weight and exact KL weight are both `0.1`;
safety overrides use KL weight `0.5`.

Both branches restart from the released public checkpoint, train only the
registered 64 decoder tensors, use main LR `1e-6`, layer-0 LR multiplier
`0.1`, FP32, gradient clip `1.0`, no weight decay, and old-policy sync interval
32. Inference remains a 20-mode bank with schedule `[32,24,16,8,0]` and the
unchanged S-multi selector.

## Sampling and four-fold OOF

Reuse the immutable Stage30 public difficulty labels and deterministic global
batch composition `2/30/8/24` for
severe/recoverable/boundary/mature scenes. Each 8-GPU accumulation window is
one global batch of 64 disjoint scenes.

Run eight independent 8xH100 jobs:

- DP and DPF holdout 0 train folds 1/2/3;
- DP and DPF holdout 1 train folds 0/2/3;
- DP and DPF holdout 2 train folds 0/1/3;
- DP and DPF holdout 3 train folds 0/1/2.

Each job trains four epochs, exactly 192 optimizer steps, and saves steps
48/96/144/192. Evaluate every checkpoint on its held-out fold under namespaces
`20261411` and `20261412`.

A DPF step is eligible only if:

- pooled selected delta is at least `+0.003`;
- whole-log bootstrap 95% CI lower bound is positive;
- both namespaces are positive and every fold is at least `-0.0005`;
- hard-scene gain is at least `+0.010`;
- mature mean is at least `-0.0001` and no mature fold is below `-0.0005`;
- candidate-mean and candidate-max deltas are each at least `-0.0005`;
- exact selected public/current fallback oracle is at least `+0.004`;
- trimmed mean is nonnegative and wins exceed losses;
- all component deltas are at least `-0.0005`;
- catastrophic-regression count is zero;
- all finite-value and provenance checks pass.

Rank eligible DPF steps by whole-log CI lower bound, worst-fold mean, pooled
mean, then earlier step. DP is always reported as an ablation and cannot be
promoted. If no DPF step passes, Stage31 stops without changing constants,
gates, or branch roles.

## Confirmation, selector closure, and NavTest

Retrain the selected DPF epoch from public on folds0--3 and confirm on fold4
under namespaces `20261421`, `20261422`, and `20261423`. If it passes the full
gate, retrain from public on folds0--4 and confirm on fold5 under namespaces
`20261431`, `20261432`, and `20261433`.

Only after both generator confirmations may a new cross-generator selector be
trained from Stage31 and historical candidate banks. It is calibrated on
fold4 and replayed unchanged on fold5. This closure selector is an ablation
only; the frozen S-multi result remains primary.

After confirmation, restart from public and train once on all 6,119 rewardable
scenes for the frozen epoch count. The one-shot NavTest systems are:

- `P`: released public 0.88104106465 system;
- `B`: public generator plus frozen S-multi;
- `C`: Stage31 DPF generator plus the same frozen S-multi;
- optional `Cprime`: Stage31 generator plus the closure selector, ablation only.

The GRPO claim is `C-B`; the complete-system claim is `C-P`. NavTest success
requires `C-B >= +0.005`, positive whole-log CI lower, `C-P > 0`, zero
catastrophic regressions, and component guards. The stretch target is
`C-B >= +0.010`. No method or threshold changes are allowed after NavTest
inspection.

## Required verification and execution safety

- Unit tests cover independent selector decisions, different selected anchors,
  common noise, public-selected BC chains, sign preservation, safety override,
  headroom endpoints, invalid shapes, and non-finite values.
- Regression tests prove every Stage23--30 route is unchanged.
- A real optimizer-step audit proves exactly 64 decoder tensors change and all
  frozen modules remain bitwise equal.
- Sampler tests prove the exact global bucket composition, rank disjointness,
  branch-identical ordering, and deterministic resume.
- All plans, manifests, checkpoints, selectors, results, reports, namespaces,
  and SHA256 values are frozen and replayable.
- H100 wrappers use the absolute repository path, preflight every referenced
  file, run `bash -n`, and expose one top-level launcher for the eight jobs.
