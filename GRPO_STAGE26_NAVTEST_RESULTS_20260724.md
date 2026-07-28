# Stage26 Full NavTest Results (2026-07-24)

## Scope

Stage26 completed the locked 12,146-scene NavTest evaluation for systems
`A`, `B0`, `B`, and `C`. All four CSV files contain exactly 12,146 valid,
unique tokens drawn from 136 whole logs.

These systems use the local epoch-19 checkpoint with SHA256
`59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d`
as their base/reference. This checkpoint is not the released public 88.1
checkpoint. The independent Stage27 `P` reproduction subsequently measured
the released checkpoint under the same NavTest protocol.

## Systems

| System | Generator | Selector | Diffusion schedule |
|---|---|---|---|
| `A` | local epoch-19 base | generator logits | `[8, 0]` |
| `B0` | local epoch-19 base | generator logits | `[32, 24, 16, 8, 0]` |
| `B` | local epoch-19 base | frozen Stage26 selector | `[32, 24, 16, 8, 0]` |
| `C` | full-6,119 Stage26 GRPO generator | same frozen selector | `[32, 24, 16, 8, 0]` |

The primary GRPO attribution is `C-B`. The selector attribution is `B-B0`.
The total Stage26 method attribution under the full schedule is `C-B0`.

## Aggregate results

| System | normalized PDMS | PDMS points |
|---|---:|---:|
| `A` | `0.8491642566` | `84.9164` |
| `B0` | `0.8491789383` | `84.9179` |
| `B` | `0.8504868980` | `85.0487` |
| `C` | `0.8621622464` | `86.2162` |

Paired whole-log bootstrap results use 10,000 samples and seed `26026`.

| Comparison | mean delta | PDMS-point delta | whole-log 95% CI |
|---|---:|---:|---:|
| `B0-A` | `+0.0000146817` | `+0.0015` | `[-0.1624, +0.1696]` |
| `B-B0` | `+0.0013079596` | `+0.1308` | `[+0.0598, +0.2025]` |
| `C-B` | `+0.0116753485` | `+1.1675` | `[+0.8084, +1.5552]` |
| `C-B0` | `+0.0129833081` | `+1.2983` | `[+0.9318, +1.6857]` |
| `C-A` | `+0.0129979898` | `+1.2998` | `[+0.9227, +1.7030]` |

The schedule change alone is indistinguishable from zero. The frozen selector
adds a small but statistically positive gain. The GRPO generator provides the
dominant gain: `+1.1675` PDMS points over the same-selector base system, with
a whole-log confidence interval strictly above zero.

## Released-checkpoint context

The actual released checkpoint reproduced `0.8810410647`, or `88.1041` PDMS
points, with zero failed scenarios. Stage26 system `C` is therefore
`-1.8879` points below the released checkpoint, with whole-log 95% CI
`[-2.4917, -1.3116]`.

This comparison does not negate the paired Stage26 GRPO attribution, because
Stage26 and the released checkpoint have different initializations. It does
show why Stage27 must restart both current and reference policies from the
released checkpoint instead of extending Stage26.

## Primary `C-B` component attribution

| Component | mean delta |
|---|---:|
| no-at-fault collision | `-0.0001234974` |
| drivable-area compliance | `+0.0110324387` |
| ego progress | `+0.0101998700` |
| TTC within bound | `+0.0024699490` |
| comfort | `-0.0000823316` |
| driving-direction compliance | `-0.0006174872` |

The aggregate improvement is driven mainly by drivable-area compliance and
progress, with a smaller TTC improvement. Collision and comfort changes are
near zero in the mean; direction has a small negative mean change.

## Paired tails

For `C-B`:

- wins: `6,868`;
- exact ties: `4,573`;
- losses: `705`;
- regressions at or below `-0.5`: `43`;
- worst paired delta: `-1.0`;
- best paired delta: `+1.0`.

Therefore Stage26 establishes a statistically significant mean GRPO gain on
the full NavTest, but it does not eliminate rare severe regressions. Stage27
must retain an explicit paired-tail audit and must not trade the public 88.1
baseline's safety for aggregate progress.

## Frozen artifacts

| System | CSV SHA256 |
|---|---|
| `A` | `a718dcfc42b9e4fe71cc8e42c3422b9d0b2cc9fe16523ec9fd44d4f403995140` |
| `B0` | `c9503f83d14aff43c8ec584026b67c909caad76f4b7c8229eaae945168551d0f` |
| `B` | `e8d1806e99710db8d1b3c5cb6f15d8f941b3fc2739ddcbec3629b447b682a2da` |
| `C` | `24d8012aa836895a86321bd3869c7f5e0455c5375e0666728c993ebd0ae49b87` |

System `C` CSV:

`/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage26_navtest_pdms_C/2026.07.24.03.09.03/2026.07.24.03.27.23.csv`

Locked Stage26 generator SHA256:

`1018f1b1a1cfcb69c27b97c2fbd30099a20a62e55fc02897f6588ca546abad6d`

Locked Stage26 selector SHA256:

`77ee768c5f469292d81049192d48797f2f3fae4f80029ae6853dd9d9c6de4207`

## Claim boundary

Stage26 proves that the implemented GRPO generator improves the local
84.916-PDMS baseline by approximately 1.17 points when the selector is held
fixed, and that the complete Stage26 system improves that local baseline by
approximately 1.30 points.

It does not prove improvement over the released
`diffusiondrive_navsim_88p1_PDMS` checkpoint. That stronger-baseline transfer
is the preregistered Stage27 experiment.
