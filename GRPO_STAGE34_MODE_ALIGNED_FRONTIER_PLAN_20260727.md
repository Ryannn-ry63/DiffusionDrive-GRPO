# Stage34: Mode-Aligned Deployable-Frontier GRPO

Date: 2026-07-27

Baseline/reference: released `diffusiondrive_navsim_88p1_PDMS`

## Scientific objective

Stage34 must establish a generator-only improvement with the deployment
selector held fixed.  It does not jointly train or replace the selector.

Stage33 is retained as a selected-only negative ablation.  Its current/public
deployment rewards could come from different anchor modes, and its formal
audit recorded `candidate_level_trace=false`.  Stage34 instead assigns every
policy update to the same anchor mode under current/public common random
numbers.

The primary method is Mode-Aligned Frontier GRPO (MAF-GRPO):

1. sample one complete 20-mode current bank and one mode-aligned public bank;
2. reuse identical initial, transition, and final diffusion noise;
3. replay all 20 current/public chains;
4. compute same-mode paired PDM deltas;
5. optimize only deployment owners, the public top-five frontier, and safe
   current headroom modes;
6. preserve public top modes asymmetrically and never weaken safety or mature
   scene regression credit.

This keeps GRPO on the diffusion denoising-chain log probability.  PDM is used
only as a training/evaluation reward and is never required at inference.

## Frozen objective

For mode `m`, `d_m = R_current(m) - R_public(m)`.  The robust scene scale is
`max(mean(abs(d_m)) over the active set, 0.002)`.  Pair advantages use
`d_m / scale` without subtracting a cross-mode mean.

The active set is the union of:

- current Stage25 selected mode;
- public Stage25 selected mode;
- public PDM top five;
- current top two modes which pass the frozen Stage25 risk/OOD/value gate and
  exceed the current selected reward by more than `0.001`.

Additional credit and protection:

- current selected mode receives `0.5 * deployment_delta / scale`, where
  `deployment_delta = R_current(selected_current) -
  R_public(selected_public)`;
- safe current headroom receives `0.25 * relu(headroom - 0.001) / scale`;
- negative public top-five credit is multiplied by `1.5`;
- negative public top-one credit is multiplied by `2.0`;
- collision, drivable, or TTC regression overrides the advantage to `-2.0`;
- mature-scene positive advantages are multiplied by `0.25`, while negative
  advantages remain unchanged;
- final advantages are clipped to `[-2, 2]`.

Regularization and rollout constants remain:

- BC weight `0.1`;
- reference KL weight `0.1`;
- safety KL weight `0.5`;
- diffusion step discount `0.6`;
- schedule `[32, 24, 16, 8, 0]`;
- learning rate `1e-6`;
- four epochs, 48 optimizer steps per epoch;
- only the registered 64 decoder tensors may change.

## Pilot protocol

Train independent fold0/fold1 models from the released public checkpoint.
Save steps `48, 96, 144, 192`.  Evaluate public, DPEL192, and Stage34 with the
same frozen Stage27 S-multi selector and noise namespaces
`20261511, 20261512`.

Pilot promotion requires:

- pooled selected gain at least `+0.0015`;
- both fold means and both namespace means strictly positive;
- whole-log bootstrap 95% CI lower bound above zero;
- paired gain over DPEL192 at least `+0.0005`;
- safe-deployable oracle gain at least `+0.0005`;
- raw oracle and public-top-five mean deltas non-negative;
- mature selected gain at least `-0.0001`;
- collision, drivable, and TTC deltas each at least `-0.0005`;
- trimmed mean non-negative, wins greater than losses, and zero catastrophic
  regressions.

Failure of all checkpoints to improve the safe-deployable oracle stops
Stage34.  It does not authorize more epochs or a selector change.

## Formal boundary

Only a passing pilot may expand to four-fold OOF.  Formal evaluation adds a
mode-aligned-pair-only ablation which disables deployment/headroom credit.
NavTest remains blocked until four-fold selected gain is at least `+0.003`,
the whole-log CI lower bound is positive, every fold is non-negative, and all
safety/ceiling gates pass.  The target is `+0.005` normalized PDMS.

