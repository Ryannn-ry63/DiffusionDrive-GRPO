# Stage27 Phase4 Fold5 Result (2026-07-24)

## Scope and protocol

- Baseline: public DiffusionDrive `88.1` checkpoint.
- Generator candidates: frozen Stage27 Phase3 epoch 1 and epoch 2 checkpoints.
- Selector: frozen Stage27 S-multi selector and calibration.
- Evaluation split: protected fold5, 1023 tokens from 151 whole logs.
- Noise namespaces: `20261011` and `20261012`.
- Execution hardware: local 8 x NVIDIA GeForce RTX 4090; six independent FP32 inference cells.
- Frozen inputs SHA256: `f22cb655e8acfbc41a752acb99772ecabdcdd7ad9788953f4e14e6063067fe4b`.

The first local attempt was interrupted by loss of the controlling terminal at
768/1023. It had produced no result artifact. The empty-output-directory
restart guard was corrected without changing any frozen model, manifest,
noise namespace, metric, or gate. The complete rerun produced all six
artifacts.

## Frozen gate result

The frozen Phase4 gate did **not** pass. Selector closure and NavTest remain
blocked.

| System | Mean score | Mean gain vs B | Whole-log 95% CI |
| --- | ---: | ---: | ---: |
| B (public generator + selector) | 0.844654865 | - | - |
| C1 (epoch 1 + selector) | 0.846298779 | +0.001643914 | [-0.000949030, +0.005080894] |
| C2 (epoch 2 + selector) | 0.847605527 | +0.002950662 | [-0.000376684, +0.007273823] |

The frozen selection rule chooses C2 / epoch 2 provisionally:

- checkpoint: `grpo-01-160.ckpt`
- checkpoint SHA256:
  `aa4c287814ec25fdb73fafe797251a45d2ac6fc8aad82bfb8ce39f285f4baae1`
- namespace gains: `+0.002817170` and `+0.003084155`
- collision mean delta: `0.0`
- TTC mean delta: `-0.000488759` (inside the frozen `-0.0005` tolerance)
- catastrophic regressions: `1/2046` (`0.0489%`)

Failed frozen checks:

1. pooled mean gain is below `+0.005`;
2. whole-log CI lower bound is not strictly positive.

All provenance, two-namespace positivity, collision, TTC, and catastrophic-rate
checks passed.

## Independent replay

The checker was independently replayed from the six immutable JSON artifacts
to `/tmp/stage27_phase4_independent_recheck.json`. It reproduced the selected
epoch, all means, confidence intervals, component deltas, and gate decisions.
The expected exit code was 2 because the scientific gate failed.

## Diagnostic interpretation

The positive mean is real but strongly long-tailed:

- averaged over the two namespaces, 405 tokens improve, 585 regress, and 33 tie;
- token median delta is `-0.000037700`;
- 10% trimmed mean delta is `-0.000059959`;
- four large baseline-failure repairs contribute `+0.003244478` to the overall
  mean;
- after excluding those four only for diagnosis, the remaining-token mean is
  `-0.000294969`;
- 62 whole logs have positive mean delta and 88 have negative mean delta.

The selector is not the main source of the Stage27 gain:

- C2 switches on 24/1023 samples per namespace;
- switched-sample mean delta is approximately zero;
- the 999 fallback samples contribute about `+0.0030` mean delta;
- only 24-26 selected modes differ between B and C2, and their mean delta is
  negative.

Therefore Stage27 demonstrates that GRPO can repair several severe failures of
the public generator, but it does not yet demonstrate a broad distributional
uplift of the public 88.1 baseline. The correct next step is not NavTest and not
an unbounded extension of the same training. A new stage must preserve the
large repaired cases while shifting the median and whole-log distribution
positive.

## Immutable artifacts

- Formal report:
  `artifacts/grpo_stage27/generator/phase4/report.json`
- Complete execution logs:
  `artifacts/grpo_stage27/h100_logs/20260724T154747Z_phase4_fold5`
- Interrupted audit logs (not used by the report):
  `artifacts/grpo_stage27/h100_logs/20260724T150917Z_phase4_fold5`
- Cell artifacts:
  `artifacts/grpo_stage27/generator/phase4/fold5`

