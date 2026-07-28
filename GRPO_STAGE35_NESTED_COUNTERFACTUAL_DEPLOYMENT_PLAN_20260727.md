# Stage35: Nested Counterfactual Deployment GRPO

Date: 2026-07-27

Baseline: released `diffusiondrive_navsim_88p1_PDMS`

## Scientific objective

Stage35 addresses the deployment-credit gap in an anchored
generator--selector system.  Candidate reward, candidate oracle reward, and
the reward of the trajectory finally deployed by a frozen selector are not
interchangeable.  Stage30--34 showed that optimizing the wrong candidate
subset can raise low-ranked modes while damaging the public frontier.

The proposed method is Nested Counterfactual Deployment GRPO (NCD-GRPO):

1. sample eight exchangeable complete 20-mode banks from one scene;
2. retain Stage31 DPEL's outer GRPO over the eight frozen-selector deployment
   decisions;
3. for active modes, replace one candidate with a same-anchor rollout from
   another bank while holding the other 19 candidates fixed;
4. rerun the frozen selector and use the selected-PDM change as that chain's
   counterfactual deployment contribution;
5. normalize counterfactual advantages only among stochastic rollouts of the
   same anchor, never across different driving intentions.

Same-anchor grouping is a correctness requirement, not the claimed
contribution.  The contribution is counterfactual credit from the deployed
decision back to individual diffusion chains.  The selector, perception,
classification branch, and reference generator remain frozen, and inference
uses one generator without PDM or counterfactual computation.

## Frozen pilot method

Stage35 is a second GRPO phase initialized from the fold-matched Stage31
DPEL192 checkpoint.  Formal attribution remains
`Public -> DPEL192 -> NCD-GRPO`.

- group size: eight complete banks;
- modes per bank: 20;
- active set: union of current-selector top two and reference-selector top
  two, with at most four unique modes per bank;
- counterfactual donor baseline: mean selected reward over the seven hybrid
  banks obtained by same-anchor donor replacement;
- outer advantage: unchanged Stage31 DPF deployed-decision advantage;
- inner advantage: same-anchor standardization of counterfactual deployment
  contributions across the eight banks;
- total inner coefficient: `0.25`;
- safety regression advantage: `-2.0`;
- mature-scene positive multiplier: `0.25`;
- BC weight: `0.1`;
- reference KL weight: `0.1`;
- safety KL weight: `0.5`;
- denoising schedule: `[32, 24, 16, 8, 0]`;
- learning rate: `1e-6`;
- trainable boundary: the registered 64 decoder tensors only;
- four epochs, 48 optimizer steps per epoch;
- freeze steps: `48, 96, 144, 192`.

All 160 trajectories may be PDM-scored without gradient.  Gradient-bearing
chain replay is restricted to the active modes and must be micro-batched.

## Signal gate before training

Run a no-update diagnostic on training-fold tokens before an optimizer audit.
It must prove:

- complete finite candidate, selector, reward, component, and provenance
  traces;
- at least 10% of active `(scene, bank, mode)` entries have absolute
  counterfactual contribution greater than `1e-4`;
- at least 25% of scenes contain a non-selected active mode with nonzero
  deployment contribution;
- every normalized group contains only a single anchor identity.

Failure stops NCD-GRPO before formal training.  It does not authorize a
post-hoc soft-selector reward.

## Pilot and promotion gates

Train independent fold0/fold1 models and evaluate Public, fold-matched
DPEL192, and NCD steps 48/96/144/192 under two frozen noise namespaces.

A checkpoint is promotable only when all checks pass:

- selected gain versus Public at least `+0.0015`;
- selected gain versus DPEL192 at least `+0.0005`;
- both fold means and both namespace means strictly positive;
- whole-log bootstrap 95% CI lower bound strictly positive;
- candidate mean, raw oracle, and safe-deployable oracle no worse than
  DPEL192;
- mature selected delta versus DPEL192 at least `-0.0001`;
- collision, drivable, and TTC deltas versus DPEL192 each at least
  `-0.0005`;
- trimmed mean nonnegative, wins greater than losses, and zero catastrophic
  regressions.

Only a passing pilot may expand to four-fold OOF.  Four-fold promotion
requires at least `+0.003` versus Public and `+0.001` versus DPEL192, a
positive whole-log CI lower bound, and no negative fold or safety gate.
NavTest remains blocked until four-fold promotion; its target is `+0.005`
normalized PDMS versus the released Public system.

## Required audits and ablations

- hybrid-bank replacement changes exactly one candidate;
- no selected trajectory change implies exactly zero hard counterfactual
  contribution;
- donor permutation leaves the seven-donor mean invariant;
- no cross-anchor samples enter an inner advantage group;
- outer credit touches only deployed chains and inner credit only active
  chains;
- micro-batched and non-micro-batched log-prob/reward results agree;
- reference, selector, perception, classification, and all unregistered
  parameters have zero gradients and zero checkpoint changes;
- checkpoint, manifest, selector, calibration, noise, schedule, and plan
  hashes fail closed.

The fixed ablations are Public, DPEL192, same-anchor intrinsic GRPO, DPEL plus
counterfactual credit, and complete NCD-GRPO.  Do not scan loss weights after
holdout evaluation.
