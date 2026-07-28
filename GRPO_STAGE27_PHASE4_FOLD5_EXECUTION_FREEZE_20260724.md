# Stage27 Phase4 Fold5 Execution Freeze

## Decision status

This Phase4 execution record was frozen on 2026-07-24 after the two-epoch
Phase3 run passed checkpoint/gradient audit and before any Stage27 generator
was evaluated on fold5.

Phase4 follows the rules in
`GRPO_STAGE27_STAGE26_ALIGNED_EXECUTION_PLAN_20260724.md`. It does not use
NavTest and does not change the generator objective, selector, thresholds,
schedule, or epoch count.

## Locked inputs

- public generator/reference SHA256:
  `008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b`;
- selected S-multi selector SHA256:
  `023d7b6b77bb8fa2dc3e779849f3716688abf0796b019416494c80b29615b691`;
- calibration SHA256:
  `fec2a10573e913e4452f8eea92e6334fcd5e6f1104e6244876770973a1f41990`;
- Phase3 formal audit SHA256:
  `68b903d0c12fc5d99a040c176d4773afcdc009a95cec212c962217714dd7cc5a`;
- epoch-1/step-80 generator SHA256:
  `665e0fda0526e8ffe33d1931f97e48816cf72acd85ba93d4c04dbae90c84d3b3`;
- epoch-2/step-160 generator SHA256:
  `aa4c287814ec25fdb73fafe797251a45d2ac6fc8aad82bfb8ce39f285f4baae1`;
- protected fold5 manifest SHA256:
  `2424fe88319f81650eec1b8e2dbe265143d5d6d7cc8d113e104caf8d150cf2d4`;
- fold5 size: 1023 scenes from 151 whole logs.

## Locked paired grid

The two fresh evaluation noise namespaces are:

- `20261011`;
- `20261012`.

Each namespace is shared exactly by all three systems:

- `B`: public generator plus the frozen S-multi selector;
- `C1`: epoch-1 generator plus the same selector;
- `C2`: epoch-2 generator plus the same selector.

This gives six artifacts and 2046 paired observations for each generator
comparison. The full schedule is fixed at `[32, 24, 16, 8, 0]`.

## Selection and gate

For each epoch compute `C-B` with a whole-log bootstrap 95% confidence
interval. Select one epoch lexicographically by:

1. larger confidence-interval lower bound;
2. larger pooled mean gain;
3. fewer catastrophic regressions (`delta <= -0.5`);
4. earlier epoch.

After selection, the provisional generator passes only if all are true:

- pooled `C-B >= +0.005`;
- whole-log 95% CI lower bound is strictly positive;
- both namespace mean gains are strictly positive;
- collision mean delta is at least `-0.0005`;
- TTC mean delta is at least `-0.0005`;
- catastrophic-regression rate is at most `0.005`;
- every artifact is complete, finite, correctly paired, and provenance exact.

Failure stops Stage27 and does not authorize extra epochs. Passing authorizes
Phase5 selector closure; it does not authorize NavTest.
