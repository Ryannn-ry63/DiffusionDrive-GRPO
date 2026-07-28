# Stage30: Mode-Coverage Constrained GRPO

## Decision status

This plan is frozen on 2026-07-25 after the completed Stage29 fold4 stop and
before any Stage30 implementation, training, or evaluation. The paper-primary
system remains one DiffusionDrive generator with the frozen passing Stage27
S-multi selector. Learned routing, a second generator, LoRA, inference fallback,
selector retraining, schedule changes, and post-hoc gate changes are forbidden
for the primary result.

Stage29 established a real but insufficient public-baseline gain: H3 achieved
`+0.001695` with a positive whole-log CI lower bound; HC4 achieved the largest
mean `+0.001741`, while its exact public/current selected fallback oracle was
only `+0.002212`. Approximately 20% hard scenes improved, but the approximately
80% mature scenes had small negative transfer. Stage30 must raise hard-scene
candidate headroom while functionally preserving mature scenes.

## Frozen systems and public difficulty labels

Collect the released public 88.1041 checkpoint with the exact frozen S-multi
selector over all 6,119 rewardable tokens under namespaces `20261301`,
`20261302`, and `20261303`. Freeze every selected PDMS value and all six
components before Stage30 training.

For each token:

- `severe`: any namespace has a collision, drivable, or TTC failure, or the
  three-namespace selected-PDMS mean is below `0.50`;
- `recoverable`: not severe and the maximum selected PDMS is below `0.75`;
- `boundary`: the minimum selected PDMS is below `0.75` and the maximum is at
  least `0.75`;
- `mature`: the minimum selected PDMS is at least `0.75`.

Every optimizer step is a deterministic global batch of 64 scenes with
`2/30/8/24` severe/recoverable/boundary/mature slots. Eight DDP ranks each
consume eight disjoint microbatches per accumulation window. The sampler is
rank-aware, restart-deterministic, and identical between ablation and primary
branches.

## Twenty-mode common-random trace

Replace Stage29's eight independently sampled 20-mode sets followed by eight
selected-chain rewards with one complete 20-mode set. For all 20 anchors:

- sample current and frozen-public chains with identical initial, transition,
  and final noise;
- compute current and paired-public PDMS and six components for all 40
  trajectories;
- replay all 20 current actions for policy log probabilities;
- replay paired public actions for behavior cloning;
- compute exact current/reference mean KL at every denoising step.

The frozen selector is not updated and is not used to discard modes during
training. Inference remains the unchanged 20-mode candidate bank, schedule
`[32,24,16,8,0]`, and S-multi selector.

## Objectives

Common constants are group size 20, paired-scale floor `0.002`, advantage clip
`[-2,2]`, denoising discount `0.6`, and a top tail consisting of the five
highest current rewards.

`COV` is the fully cross-validated but non-selectable coverage-only ablation.
Its advantage is sign-preserving paired delta plus `0.5 * max(reward_z, 0)` on
top-five modes. Collision, drivable, or TTC regression forces advantage `-2`.
BC and exact KL weights are both `0.1`.

`MCC` is the only selectable primary branch:

- severe/recoverable: use the COV advantage with BC/KL `0.05/0.05`;
- boundary: top-tail weight `0.25`, multiply negative advantages by two,
  require delta at least `+0.001` for a positive update, and use BC/KL
  `0.2/0.5`;
- mature: multiply advantages for delta below `-0.0002` by four; make
  `[-0.0002,+0.002)` policy-neutral; allow positive advantage only for delta at
  least `+0.002` with no regression in any of the six components; use BC/KL
  `0.5/1.0` on all modes and all denoising steps.

Safety regression always forces `-2`. Policy losses are normalized separately
over hard, boundary, and mature active scenes before summation.

Both branches restart from the released public checkpoint, train exactly the
existing 64 decoder tensors, use layer-0 LR multiplier `0.1`, main LR `1e-6`,
FP32, gradient clip `1.0`, old-policy sync interval 32, and no weight decay.

## Four-fold cross-validation

Run eight independent 8xH100 jobs, all concurrently if resources allow:

- holdout 0 trains folds 1/2/3;
- holdout 1 trains folds 0/2/3;
- holdout 2 trains folds 0/1/3;
- holdout 3 trains folds 0/1/2;
- repeat for COV and MCC.

Each job runs four equal epochs, exactly 192 optimizer steps, and saves steps
48/96/144/192. Evaluate every checkpoint on its held-out fold under namespaces
`20261311` and `20261312`. Select one common MCC step across all folds; COV is
never selectable.

An MCC step is eligible only when the pooled OOF selected delta is at least
`+0.003`, whole-log bootstrap CI lower is positive, both namespaces and all
four folds are positive (no fold below `-0.0005`), hard gain is at least
`+0.015`, mature mean is at least `-0.0001` (no fold below `-0.0005`),
candidate-mean delta is positive, candidate-max oracle delta is at least
`+0.0005`, exact selected fallback oracle is at least `+0.004`, trimmed mean is
nonnegative, wins exceed losses, all components are at least `-0.0005`,
catastrophic count is zero, and all provenance is complete. Rank eligible
steps by CI lower, mature mean, pooled mean, then earlier step.

## Confirmation and NavTest

Retrain the selected MCC epoch from public on folds0-3 and confirm on fold4
under `20261321/20261322/20261323`. If it passes the complete gate, retrain on
folds0-4 and confirm on fold5 under `20261331/20261332/20261333`. Failure at
either stage stops without tuning.

After both confirmations, restart from public and train once on all 6,119
rewardable scenes for the selected epoch count. The only primary NavTest
systems are official public `P`, public generator plus frozen S-multi `B`, and
Stage30 MCC generator plus the same selector `C`. Success requires
`C-B >= +0.005`, positive whole-log CI lower, `C-P > 0`, zero catastrophes,
and component guards. The stretch target is `C-B >= +0.010`.

## Required verification

- Unit tests cover group-20 common noise, all-mode replay, top-five selection,
  sign preservation, all buckets and margins, safety override, finite/shape
  failures, and mature preservation-only gradients.
- Sampler tests prove exact global `2/30/8/24` composition, rank disjointness,
  branch-identical schedules, epoch determinism, and resume determinism.
- A real optimizer-step audit proves exactly 64 changed decoder tensors and
  bitwise-frozen public reference, selector, perception, and classification.
- Stage23--29 routes and tests remain unchanged.
- Every plan, manifest, checkpoint, result, selection, and report is
  SHA256-locked; formal scripts refuse overwrite and namespace drift.

