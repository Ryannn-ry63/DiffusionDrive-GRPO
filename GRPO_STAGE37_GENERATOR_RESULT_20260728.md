# Stage37 generator pilot result (2026-07-28)

## Decision

Stage37 BPD generator training and evaluation completed successfully, but the
frozen generator promotion gate failed. No checkpoint was selected. JFI
calibration, the 2x2 factorial pilot, folds-2/3 expansion, full-6119 training,
and NavTest must not run from this branch.

The authoritative machine-readable result is
`artifacts/grpo_stage37/pilot/generator_summary.json`. The expected
`generator_selection.json` was not created.

## Infrastructure and mechanism status

- Both held-out folds completed formal training through optimizer steps 48,
  96, 144, and 192.
- All 16 held-out evaluation cells completed with zero failed scenarios.
- Both formal audits passed the 64-tensor decoder boundary, two-state
  frontier, post-accumulation projection, finite diagnostics, and
  mature-positive-exploration-zero checks.
- Projection conflicts occurred on 51.04% of fold-0 and 46.35% of fold-1
  optimizer steps, so the projected update was materially exercised.
- The eight JFI banks and isolated JFI selector training also passed, but JFI
  is intentionally not promoted around a failed generator.

## Cross-fitted generator results

All deltas below are relative to the frozen public 0.881 checkpoint under the
same Stage25 deployment selector.

| step | selected | candidate mean | raw oracle | public top-5 | hard | mature | CI95 lower | vs Stage36 RGT192 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 | +0.000247 | +0.000420 | +0.000042 | -0.000806 | +0.001641 | -0.000138 | -0.000246 | -0.001578 |
| 96 | -0.000062 | +0.000380 | +0.000051 | -0.002050 | +0.001849 | -0.000589 | -0.000733 | -0.001887 |
| 144 | +0.000270 | +0.000945 | +0.000071 | -0.001083 | +0.002192 | -0.000260 | -0.000241 | -0.001555 |
| 192 | +0.000314 | +0.001520 | +0.000104 | -0.000874 | +0.002603 | -0.000317 | -0.000208 | -0.001511 |

At step 192, every fold/namespace selected delta was positive:

- fold 0, namespace 20261511: +0.000267;
- fold 0, namespace 20261512: +0.000098;
- fold 1, namespace 20261511: +0.000116;
- fold 1, namespace 20261512: +0.000776.

This is weak cross-domain generalization, not a paper-level improvement. The
whole-log CI still crosses zero and the magnitude is substantially below the
frozen +0.002 generator gate.

## Diagnosis

The generator learned to improve the average candidate pool, especially TTC,
but did not create meaningfully better best trajectories:

- step-192 candidate mean increased by +0.001520 while raw oracle increased by
  only +0.000104;
- public top-5 candidate quality was negative at every checkpoint;
- step-192 TTC increased by about +0.001470, but progress decreased by about
  -0.000177;
- mature-scene performance decreased by -0.000317;
- reward/logit Spearman correlation also decreased as training progressed.

Therefore the dominant failure is generator ceiling and mode allocation, with
a secondary selector-conversion issue. The current objective raises weaker
tail modes and improves candidate averages, but it does not preserve and
extend the already strong public modes. A perfect selector cannot turn a raw
oracle gain of +0.000104 into the required +0.005 full-system gain.

## Frozen stop rationale

Longer continuation of the same checkpoint is not authorized. From step 48 to
192, candidate mean and hard-scene gain increased, but mature performance and
public top-5 quality remained negative, the CI never became positive, and all
steps remained worse than Stage36 RGT192. The pre-registered
`stop_for_negative_public_top5_all_steps` condition is true.

The next stage must change how GRPO allocates improvement across modes and
protects strong public trajectories. It must not be presented as a selector
threshold tuning problem or solved by running JFI on this failed generator.
