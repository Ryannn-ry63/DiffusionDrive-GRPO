# Stage27: Stage26-Aligned GRPO on the Public 88.1 Baseline

## Decision status

This execution plan was frozen on 2026-07-24 after completing the preregistered
Stage27 Phase-1 baseline and transfer diagnostics, and before training any
Stage27 selector or generator.

Stage27 keeps the successful Stage26 scientific structure while changing the
initialization/reference policy to the actual released checkpoint. It does
not continue the Stage26 generator checkpoint.

## Verified starting point

The released checkpoint is:

`/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS`

Its locked SHA256 is:

`008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b`

The full 12,146-scene NavTest reproduction completed with zero failures:

| Metric | Value |
|---|---:|
| normalized PDMS | `0.8810410646527626` |
| PDMS points | `88.10410646527626` |
| no-at-fault collision | `0.9814753828420879` |
| drivable-area compliance | `0.9631977605796147` |
| ego progress | `0.8224774512803593` |
| TTC within bound | `0.9476370821669685` |
| comfort | `0.9992590153136836` |
| driving-direction compliance | `0.9810637246830232` |

The CSV SHA256 is
`4f4a600035d09c502b210fe52803f30aeb4aded8bdaa258f28f72617d0c87eda`.
This `P` result is frozen and must not be rerun for model selection.

## What Stage26 established

On the local epoch-19 baseline, Stage26 obtained:

- selector contribution `B-B0 = +0.1308` PDMS points;
- pure GRPO generator contribution `C-B = +1.1675` PDMS points;
- whole-log 95% CI for `C-B`: `[+0.8084, +1.5552]` points;
- total `C-A = +1.2998` PDMS points.

The Stage26 final score was `86.2162`, so it remains `1.8879` points below the
actual public baseline. Stage26 proves the GRPO mechanism on the weaker base;
Stage27 must prove transfer to the stronger base.

## Phase-1 transfer diagnosis

Three complete public-base candidate banks were collected on the 4,075-token
folds0--3 selector-fit manifest:

| Namespace | selected reward | oracle reward | oracle gap | SHA256 |
|---|---:|---:|---:|---|
| `-1` | `0.84174344` | `0.93995142` | `0.09820799` | `21f5f83b9b524c4938e480461a1b82c6203442d70a9e3cd6cce98d1184c85d27` |
| `20260811` | `0.84116239` | `0.93983195` | `0.09866956` | `a8999f1811fc1fb0b77fc111923919e5cdea4b357fb7861b7a8de09944a303a2` |
| `20260812` | `0.84246419` | `0.93949988` | `0.09703569` | `f44bb86f09f9fcb7a48417df76dd6804ed61fb35c44ddc4f4f8c1ac55f57ed98` |

Thus the strong baseline still produces diverse candidates with substantial
oracle headroom. The generator itself is not the only bottleneck.

The frozen Stage26 selector was then evaluated on all 1,021 fold4 tokens under
namespaces `20260821` and `20260822`. It made zero switches in both cells.
This is a safe fallback but a failed transfer.

The failure is not caused by the OOD-distance gate:

- all `19,399` non-fallback modes per namespace pass the OOD threshold;
- approximately `1,637` modes pass the old risk threshold;
- approximately `2,800` modes pass the old positive-value threshold;
- zero modes pass risk and positive-value gates simultaneously.

Therefore Stage27 must retrain the selector heads on public-domain labels and
recalibrate them. Merely loosening one threshold is prohibited.

## The Stage26 mechanism to preserve

| Successful Stage26 element | Stage27 counterpart |
|---|---|
| frozen base/reference identity | frozen public-88.1 identity |
| all 20 candidates with exact PDM labels | the three completed public banks |
| cross-generator Stage24/25 selector | public-adapted cross-generator selector |
| conservative risk, value, and OOD gates | same architecture, newly calibrated |
| `diffgrpo_selected_set` | unchanged objective and hyperparameters |
| two generator epochs | exactly two initial Stage27 epochs |
| paired noise and whole-log confidence | unchanged statistical protocol |
| add fresh generator to selector domain | mandatory Stage27 selector closure |
| restart from base for all-scene training | restart from public checkpoint |

No new GRPO reward, policy objective, decoder scope, or schedule is introduced.
This makes the base checkpoint the principal experimental change.

## Phase 2: public selector adaptation

Train two selector branches in parallel. Both use the public checkpoint for
the frozen perception/trajectory context and train fresh selector heads; the
old selector is not warm-started because its risk and value predictions are
structurally misaligned on the public domain.

### Branch S-public: domain-specific control

- banks: the three public-base folds0--3 banks;
- Stage24 safety/value ensemble: exactly 3 epochs;
- Stage25 relative-harm ensemble: exactly 3 epochs;
- one epoch exposes exactly one namespace bank;
- seeds: `27024` and `27025`;
- architecture, losses, batch size, learning rate, and ensemble size are
  unchanged from Stage26.

### Branch S-multi: primary cross-generator branch

- banks: the 15 frozen Stage26 banks followed by the three public-base banks;
- Stage24 safety/value ensemble: exactly 18 epochs;
- Stage25 relative-harm ensemble: exactly 18 epochs;
- one epoch exposes exactly one generator/namespace bank;
- the last three epochs are the public-domain adaptation tail;
- seeds: `27124` and `27125`;
- architecture and optimization are unchanged from Stage26.

This pair separates public-domain fit from cross-generator robustness without
changing the selector design.

### Calibration and branch selection

For each branch, collect fold4 calibration predictions for the public
generator under namespaces `20260821` and `20260822`. Calibrate the residual
margin, risk threshold, and OOD state using only these trainval records.

A branch passes only if:

- pooled selected-minus-fallback gain is at least `+0.003`;
- the whole-log bootstrap 95% CI lower bound is strictly positive;
- both namespace cells have non-negative mean gain;
- switch rate lies in `[0.01, 0.15]`;
- collision, drivable, TTC, comfort, and direction mean deltas are each at
  least `-0.0005`;
- no switched token loses `0.5` or more PDMS;
- every calibration/OOD diagnostic is finite.

If both branches pass, select the branch with the larger whole-log bootstrap
95% CI lower bound. A difference smaller than `0.0005` is a tie and resolves
to S-multi because generator-domain robustness is the Stage26-proven design.
If neither passes, stop before generator training.

## Phase 3: provisional public-base GRPO

Introduce a Stage27-specific formal training mode and public SHA guard. Do not
replace the historical Stage16--26 base constant.

First run an eight-H100 one-step audit. It must prove:

- current policy and reference both load the public SHA;
- the frozen reference is bitwise unchanged;
- only the registered diffusion decoder tensors receive finite gradients;
- the selector, perception trunk, and reference receive exact-zero gradients;
- rewards, group advantages, log probabilities, and optimizer update are
  finite.

Then train a provisional generator:

- initialization/reference: public-88.1 checkpoint;
- data: folds0--4, 5,096 rewardable scenes;
- selected branch from Phase 2, frozen;
- eight-H100 DDP;
- exactly 2 epochs initially;
- learning rate `1e-6`;
- batch size `1` per GPU and gradient accumulation `8`;
- `diffgrpo_selected_set`, group size `8`;
- BC weight `0.1`, step discount `0.6`;
- regular/mature paired KL weights `0.1/0.5`;
- full schedule `[32, 24, 16, 8, 0]`;
- all remaining Stage26 generator settings unchanged.

Save both epoch-1 and epoch-2 checkpoints.

## Phase 4: provisional generator gate

Use fold5 as Stage27 development validation, not as a final paper test. It was
already consumed by Stage26 and therefore is not described as protected.
Use newly locked noise namespaces, shared identically by all compared systems.

Compare:

- `B`: public generator plus the selected frozen selector;
- `C1`: Stage27 epoch-1 generator plus the same selector;
- `C2`: Stage27 epoch-2 generator plus the same selector.

Choose the best epoch using the following lexicographic rule:

1. whole-log bootstrap 95% CI lower bound of `C-B`;
2. mean `C-B`;
3. lower catastrophic-regression count;
4. earlier epoch.

The provisional generator passes only if:

- pooled `C-B >= +0.005`;
- whole-log 95% CI lower bound is positive;
- both locked noise cells have positive mean gain;
- no-at-fault collision and TTC deltas are each at least `-0.0005`;
- catastrophic regression rate does not exceed `0.005`;
- no NaN, invalid reward, or provenance mismatch occurs.

Failure stops Stage27. It does not authorize blind epoch extension.

## Phase 5: final selector closure

After a provisional generator passes:

1. collect its folds0--3 candidate banks under the same three selector-fit
   namespaces;
2. train fresh final Stage24 and Stage25 ensembles on 21 banks:
   15 Stage26 domains, 3 public-base banks, and 3 provisional-generator banks;
3. use exactly 21 epochs per selector phase, one exposure per bank;
4. collect fold4 calibration records for both the public and provisional
   generators under namespaces `20260821` and `20260822`;
5. calibrate one shared deployment rule over all four cells.

The closure selector must satisfy the Phase-2 safety constraints in all four
cells, have pooled gain at least `+0.003`, and preserve a positive
provisional-generator `C-B` with whole-log CI lower bound above zero.

This step is mandatory. It is the direct Stage27 analogue of the successful
Stage26 decision to include the fresh generator in the final selector domain.

## Phase 6: all-scene generator retraining

After closure:

- restart from the original public checkpoint, never from the provisional
  generator;
- train on all 6,119 rewardable scenes;
- use the frozen closure selector;
- use the duration selected in Phase 4, expected to be epoch 1 or 2;
- use the otherwise identical Stage27 GRPO configuration;
- audit the final checkpoint against the public reference.

No extra epochs may be added after observing NavTest.

## Phase 7: final NavTest

The public `P` score is already frozen. After every Stage27 method choice is
frozen, run exactly:

| System | Generator | Selector | Schedule |
|---|---|---|---|
| `P0` | public 88.1 | generator logits | `[32, 24, 16, 8, 0]` |
| `B` | public 88.1 | final Stage27 selector | `[32, 24, 16, 8, 0]` |
| `C` | all-scene Stage27 GRPO | same selector | `[32, 24, 16, 8, 0]` |

Report:

- `P0-P`: schedule contribution;
- `B-P0`: selector contribution;
- `C-B`: pure GRPO contribution and primary method claim;
- `C-P`: complete system improvement over the actual 88.1 baseline.

A successful paper claim requires both `C-B > 0` and `C-P > 0`, with
whole-log bootstrap confidence intervals, component deltas, wins/ties/losses,
and catastrophic tails reported. No Stage27 design decision may be changed
after these final NavTest values are inspected.

## Compute layout

- one 8-H100 node can run S-public and S-multi selector branches concurrently
  on separate GPUs; selector phases themselves remain single-GPU;
- provisional and all-scene GRPO each use all eight H100s;
- three candidate-bank namespaces run concurrently on three GPUs;
- fold4/fold5 evaluation cells run concurrently on the remaining GPUs when
  dependencies permit.

The next implementation unit is Phase 2 only. Generator training is not
authorized until one selector branch passes the frozen transfer gate.
