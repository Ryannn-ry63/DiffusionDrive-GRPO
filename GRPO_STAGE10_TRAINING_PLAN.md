# DiffusionDrive GRPO Stage 10

Stage 10 keeps both current-policy refinement layers trainable while the complete
two-layer base reference, PPO old policy, perception stack, and generation
classification branches remain frozen. It tests whether adaptive functional KL
control and a lower layer-0 learning rate retain Stage-9 generation gains without
the observed full-epoch drift and selected-token tail failures.

Registered constants are seed 0, batch 2, layer-1 LR `1e-6`, PPO clip `0.2`, old
sync 32, raw PDMS, uniform scene/mode weighting, group-zscore generation advantage,
and no imitation, shaping, priority sampling, selector loss, or hard projection.
The adaptive KL coefficient starts at `0.1`, is bounded to `[0.1, 100]`, targets
`1e-4`, uses a 32-update rolling window checked every 8 updates, and doubles/halves
outside `[2/3, 3/2]` of target. Two consecutive checked windows above `2.5e-4`
save `kl-stop-step-*.ckpt` and stop training.

The only seed-0 U128 variants are `equal_lr` (layer-0 multiplier `1.0`) and
`layer0_01` (layer-0 multiplier `0.1`). Both decoder layers must have finite,
non-zero gradients and weight changes. Evaluation proceeds through fixed-256 and
fixed-1024 using `scripts/evaluation/check_grpo_stage10_gate.py`. If both pass,
the larger fixed-1024 selected mean wins; if neither passes, Stage 10 stops without
another hyperparameter scan. The winner alone may resume continuously to U512 and
then epoch 1 before dev-select, seed 1/2, dev-confirm, and one paired full-navtest.

Existing Stage-9 generation checkpoints at steps 128/512/1024/2048 are evaluated
post hoc only to establish the utility/KL curve and cannot be selected as a formal
Stage-10 result.

## Execution record (2026-07-19)

Focused unit tests, Python compilation, Bash syntax, and `git diff --check` passed.
Both U8 routes completed with finite, non-zero gradients and weight changes in
both current-policy decoder layers. Frozen-reference, old-policy, perception,
and generation-classification maximum weight changes were all exactly zero.
At U8, equal-LR rolling KL was `1.21783e-5`; layer0-0.1 rolling KL was
`4.88905e-7`.

The equal-LR U128 route stopped at update 88 as registered. Its rolling KL was
`2.62861e-4` at update 80 and `2.86968e-4` at update 88, producing two
consecutive hard-limit violations and `kl-stop-step-88.ckpt`. The adaptive
coefficient reached `3.2`. This route is rejected without performance
evaluation.

The layer0-0.1 route completed U128 with rolling KL `3.3324e-5`, coefficient
`0.1`, and no hard-limit violation. Its fixed-256 gate passed: selected
`+0.001146`, CI `[-0.000056, +0.003433]`, candidate mean `+0.001118`, oracle
`-0.001749`, worst token `-0.01512`, collision/drivable `0`, and TTC
`+0.003906`.

Its fixed-1024 gate failed despite positive mean metrics: selected `+0.001821`,
CI `[-0.001173, +0.005044]`, candidate mean `+0.001369`, oracle `+0.000877`,
collision `0`, drivable/TTC `+0.001953`, and safety-pass bucket `-0.000927`.
The registered failures were the CI lower bound and worst-token delta
`-0.884697`. Per registration, Stage 10 stops here without U512, extra seeds,
or another hyperparameter scan.

The worst token is `85a62c96f5455f87`. The base selected mode 3 with reward
`0.884697`, and mode 3 was also the base and candidate oracle. Stage 10 switched
selection to mode 12 with reward zero because its drivable component was zero.
The base selector margin was only `0.003530` (`0.003067` after training), which
identifies a low-margin discrete mode switch rather than broad candidate-quality
or functional-KL collapse.

The Stage-9 post-hoc curve confirms that mean utility and tail risk separate
early:

| update | selected delta | CI lower | worst token | rolling KL |
| ---: | ---: | ---: | ---: | ---: |
| 128 | +0.002274 | -0.000764 | -0.884697 | 0.000360468 |
| 512 | +0.002277 | -0.003417 | -0.867256 | 0.00152633 |
| 1024 | +0.002265 | -0.004622 | -0.867256 | 0.00435751 |
| 2048 | +0.005082 | -0.004051 | -1.000000 | 0.010865 |

Detailed artifacts are under `artifacts/grpo_stage10/`. The next experiment
should target the discrete selected-token tail failure directly while retaining
the layer0-0.1 drift control; simply training longer is not supported by these
results.
