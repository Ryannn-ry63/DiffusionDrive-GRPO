# Stage30 OOF Result and Stage31 Direction

## Frozen result

Stage30 completed all eight formal 8-GPU jobs and all 72 held-out evaluation
artifacts. The immutable four-fold OOF report is:

- report: `artifacts/grpo_stage30/cv/report.json`
- report SHA256: `d8878e294aeb9290c1152224b15855ed3051a7c32403e467fb220bfb58c22dab`
- CV freeze SHA256: `4e04ba444d9bbad7fe54bb3853aa9da28f49cd4a2c232e4bf7d0d1e1968e7b29`
- result: `passed=false`, `selected_step=null`, `eligible_steps=[]`

The frozen plan therefore stops before fold4 confirmation. No Stage30
checkpoint may be promoted to confirmation, all-6119 retraining, or NavTest.

## OOF summary

| system | pooled selected delta | whole-log 95% CI | hard delta | mature delta | candidate max delta | fallback oracle | catastrophes |
|---|---:|---:|---:|---:|---:|---:|---:|
| COV48 | +0.000293 | [-0.000032, +0.000708] | +0.001533 | +0.000004 | -0.000006 | +0.000571 | 0 |
| COV96 | +0.000539 | [+0.000036, +0.001103] | +0.002939 | +0.000089 | +0.000000 | +0.000905 | 1 |
| COV144 | **+0.000723** | **[+0.000237, +0.001300]** | **+0.003194** | **+0.000184** | +0.000011 | **+0.001024** | 0 |
| COV192 | +0.000700 | [+0.000165, +0.001282] | +0.003239 | +0.000194 | -0.000001 | +0.001006 | 1 |
| MCC48 | +0.000076 | [-0.000279, +0.000448] | +0.001242 | -0.000232 | -0.000056 | +0.000449 | 1 |
| MCC96 | +0.000363 | [-0.000069, +0.000823] | +0.002165 | +0.000017 | -0.000342 | +0.000667 | 1 |
| MCC144 | **+0.000359** | **[+0.000040, +0.000743]** | +0.002133 | -0.000003 | -0.000252 | +0.000620 | 0 |
| MCC192 | +0.000433 | [-0.000048, +0.000964] | +0.002551 | -0.000042 | -0.000052 | +0.000802 | 1 |

COV144 is the strongest diagnostic checkpoint but is non-selectable by the
frozen plan. MCC144 is the cleanest primary checkpoint. Neither approaches the
pre-registered `+0.003` pooled, `+0.015` hard, `+0.0005` candidate-max, and
`+0.004` fallback-oracle requirements. COV144 also has a negative fold1 mean;
MCC144 has a negative fold3 mean.

## Root-cause diagnosis

The failure is not explained by mature-scene damage alone. COV144 preserves
mature scenes and has a positive whole-log CI, but its policy updates the wrong
part of the candidate set. Sorting each scene's modes by the frozen public
candidate reward gives these mean paired changes:

- public rank 1: `-0.001817`
- public ranks 2--5: `-0.000407`
- public ranks 6--10: `+0.000239`
- public ranks 11--20: `+0.002876`

Thus the all-mode objective raises low-ranked modes while slightly damaging the
frontier. This explains the combination of positive candidate-mean delta
`+0.001326` and essentially zero candidate-max delta `+0.000011`.

The frozen public bank also already contains large unused headroom:

| bucket | public selected | public oracle | selector regret | COV144 selected delta | COV144 oracle delta |
|---|---:|---:|---:|---:|---:|
| severe | 0.4604 | 0.9564 | 0.4960 | +0.03486 | -0.00261 |
| recoverable | 0.6775 | 0.8819 | 0.2044 | +0.00003 | +0.00043 |
| boundary | 0.7563 | 0.9240 | 0.1677 | +0.00097 | +0.00028 |
| mature | 0.8911 | 0.9520 | 0.0609 | +0.00018 | -0.00004 |

The primary bottleneck is converting an already strong candidate frontier into
the deployed trajectory. Stage30's selector-independent coverage loss does not
do this. MCC's stronger preservation terms reduce useful learning further:
COV is consistently stronger than MCC, while COV already preserves mature
scenes sufficiently.

## Consequences

1. Do not train Stage30 for additional epochs. COV peaks at epoch 3, candidate
   max remains flat, and epoch 4 introduces a catastrophe.
2. Do not relax the frozen gate or promote COV144 post hoc.
3. Do not interpret the result as GRPO having no signal. COV144 is positive and
   statistically significant, and severe selected scenes gain `+0.03486`.
   The credit assignment is misaligned with deployed selection.
4. Do not spend the next stage lifting all 20 anchors uniformly.

## Recommended Stage31 direction (not yet frozen)

Stage31 should be a selector-consistent frontier GRPO experiment, still starting
from the released public checkpoint and retaining one generator at inference.
Its design objective is to update trajectories that can affect the deployed
decision:

- freeze the deployment selector while computing its selected anchor and a
  small observable frontier set;
- spend GRPO rollouts on the selected/frontier anchors rather than one rollout
  on every anchor;
- make positive policy credit depend on deployment utility or crossing the
  frozen public frontier, not merely improving a low-ranked mode;
- retain common-random public pairing and explicit negative credit for public
  top/frontier regressions;
- preserve mature scenes with the empirically sufficient COV regularization,
  avoiding MCC's overly strong blanket constraint;
- cross-validate the objective before any confirmation or NavTest;
- report both `public + same selector` and `GRPO + same selector` so any final
  gain remains attributable to GRPO rather than a selector-only replacement.

Before freezing Stage31, run a small offline design audit on the existing 72
artifacts to choose the frontier definition and quantify attainable selected
gain without using held-out labels at inference.
