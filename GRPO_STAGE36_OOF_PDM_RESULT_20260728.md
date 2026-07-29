# Stage36 Two-Fold OOF PDM Result

Date: 2026-07-28

## Frozen experiment

Stage36 evaluates Reference-Gated Tail NCD-GRPO (RGT-NCD) on two independent
holdout folds and two frozen evaluation-noise namespaces. Both formal training
audits passed for 192 optimizer steps. All 16 PDM evaluation jobs completed
with zero failed scenarios.

The frozen result artifact is:

`artifacts/grpo_stage36/pilot/summary.json`

## Best checkpoint

Step 192 is the best checkpoint by selected-trajectory gain.

- selected gain versus Public: `+0.001824692`;
- whole-log bootstrap 95% CI: `[+0.000692022, +0.003212392]`;
- fold0/fold1 gains: `+0.001208484 / +0.002439694`;
- namespace gains: `+0.001056093 / +0.002593291`;
- selected gain versus DPEL192: `+0.000952701`;
- selected gain versus Stage35 NCD192: `+0.000482284`;
- candidate-mean gain: `+0.002200048`;
- raw-oracle gain: `+0.000238212`;
- safe-deployable-oracle gain: `+0.001829290`;
- hard-scene selected gain: `+0.008653745`;
- mature-scene selected gain: `-0.000058743`;
- public-top-five delta: `-0.001064840`;
- collision/drivable/TTC deltas:
  `0 / +0.001715686 / +0.001470588`;
- wins/losses/ties: `2682 / 1203 / 195`;
- catastrophic regressions: `0`.

## Comparison with Stage35

Stage36 improves Stage35 NCD192 on the main selected mean by `+0.000482284`.
It also changes the raw-oracle result from `-0.000166758` to `+0.000238212`,
raises safe-oracle gain from `+0.001346816` to `+0.001829290`, and raises
hard-scene gain from `+0.005226282` to `+0.008653745`.

The Stage36 confidence interval versus Public is strictly positive, and its
gain versus DPEL192 also has a strictly positive whole-log confidence
interval. The gain versus Stage35 NCD192 is positive in mean but its
confidence interval still crosses zero.

## Frozen decision

No checkpoint passes every predeclared Stage36 gate. Steps 96, 144, and 192
pass all gates except public-top-five preservation. The public-top-five delta
is negative at every checkpoint:

- step48: `-0.001083469`;
- step96: `-0.002299422`;
- step144: `-0.001355746`;
- step192: `-0.001064840`.

Therefore `pilot_passed=false` and the predeclared Stage36 stop rule is
triggered solely by the public-top-five condition. This result must not be
post-hoc promoted by deleting that gate, and Stage36 must not proceed to
navtest or selector tuning.

## Mechanistic conclusion

The RGT-NCD generator does increase the overall candidate distribution,
selected trajectory, raw oracle, safe oracle, and hard-scene performance. The
remaining failure is narrower: sparse public-chain teacher forcing does not
preserve the reward of the public model's exact old top-five modes.

The implemented retention gradient has the correct sign. Its limitation is
surrogate mismatch: it increases current-policy likelihood on frozen public
denoising states only after a sampled regression, but does not constrain the
trajectory produced under the current model's own shifted denoising state
distribution. A subsequent independent stage should address this functional
state-distribution drift with a genuinely constrained update, rather than
retuning Stage36 weights or modifying the selector.
