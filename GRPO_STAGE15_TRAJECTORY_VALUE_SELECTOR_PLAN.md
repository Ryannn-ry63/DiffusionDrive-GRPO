# DiffusionDrive GRPO Stage 15: Safety-Conservative Top-2 Trajectory-Value Selector

Date: 2026-07-20
Status: preregistered implementation plan
Scope: deployment selection alignment for the existing DiffusionDrive base and
Stage-9 formal U128 GRPO generators.

## 1. Objective and attribution boundary

Stage 15 tests whether an explicitly trajectory-conditioned, scene-conditioned
value selector can turn useful generated candidates into higher deployed raw PDMS.
The frozen base/reference classification selector remains the fallback and defines
the two candidate anchors that may be reranked.

The value selector may use detached PDM reward and component labels during
training. It may not use ground-truth trajectory regression, imitation loss, or an
online PDM scorer at inference. It is an auxiliary reward-aligned selector, not a
pure-GRPO result. Formal evaluation must separate:

1. base generator + frozen reference selector;
2. Stage-9 GRPO generator + frozen reference selector;
3. base generator + value top-2 selector;
4. Stage-9 GRPO generator + value top-2 selector.

The GRPO contribution under the new selector is row 4 minus row 3. Total row-4
improvement over row 1 must not be wholly attributed to GRPO.

## 2. Locked selector design

The selector re-encodes each final generated trajectory, samples BEV features along
its eight poses, and combines them with agents, ego-query, and status context. Three
bootstrap heads predict the six existing PDM components:

- `NO_COLLISION`;
- `DRIVABLE_AREA`;
- `PROGRESS`;
- `TTC`;
- `COMFORTABLE`;
- `DRIVING_DIRECTION`.

The predicted aggregate follows the scorer's multiplicative/weighted composition.
Each head is trained with deterministic per-token 80% bootstrap inclusion. The
locked loss is:

```text
L = L_safety_BCE + L_component_SmoothL1
  + L_PDMS_SmoothL1 + L_top2_pairwise
```

The pairwise term applies only to the frozen-reference selector's top two modes and
only when the detached raw-PDMS gap is at least `0.01`.

The deployed selector considers only those two modes. The frozen-reference top-1 is
the fallback. The top-2 challenger may replace it only when its predicted safety
lower bounds for collision, drivable area, and TTC are at least `0.9`, are not below
the fallback bounds, and its predicted PDMS advantage exceeds the calibrated
margin. Otherwise the fallback is returned.

## 3. Data and training protocol

Use only the 6,119 rewardable navtrain tokens. Sort by a fixed SHA-based key with
seed `20260721` and split exactly:

- selector-train: 4,283 tokens;
- selector-calibration: 918 tokens;
- selector-test: 918 tokens.

All manifests record ordered token SHA and must have zero overlap with fixed-1024,
dev-select, dev-confirm, and navtest. The Stage-9 formal seed-0 U128 decoder and the
frozen base decoder generate 20 candidates each from the same scene and diffusion
noise, giving 40 labeled candidates per scene. Stage-14 candidates are excluded
from training.

Freeze perception, both generators, the original classification head, and the
regression branches. Train only the value selector with AdamW, learning rate
`1e-4`, weight decay `1e-4`, batch size 2, FP32, and at most four epochs. Select one
checkpoint using selector-calibration loss only. Do not use selector-test, fixed,
or dev metrics for checkpoint choice.

Calibration may choose exactly one deployment margin. Among observed predicted
advantage thresholds, choose the calibration-PDMS maximizer satisfying: collision,
drivable, and TTC means do not decline; worst switched-token delta is greater than
`-0.5`; harmful-switch total loss is at most 25% of beneficial-switch total gain.
If no margin satisfies these rules, stop Stage 15.

## 4. Gates

### A. Implementation and smoke

- value-selector gradients are finite and nonzero;
- perception, generator, original classification, and regression gradients are zero;
- default/current/reference inference remains numerically unchanged;
- requesting value inference without a trained selector and calibration fails early.

### B. Locked selector-test

On the 918 unseen selector-test tokens, value top-2 versus frozen-reference fallback
must have selected delta `>=+0.002`, paired CI95 lower bound `>0`, switch rate
`>=1%`, nonnegative collision/drivable/TTC component deltas, worst switched-token
delta `>-0.5`, and beneficial-switch gain greater than harmful-switch loss.

### C. Fixed sets

After a fixed-256 safety screen, run the four attribution cells on fixed-1024.
Require row 4 minus row 1 `>=+0.003`, positive value-selector gains on both base and
GRPO generators, row 4 minus row 3 `>=0`, and no collision/drivable/TTC decline.

### D. Independent dev-select

On the locked 3,072-token dev-select manifest require:

- row 4 minus row 1 `>=+0.005` with paired CI95 lower bound `>0`;
- row 4 minus row 3 `>=+0.0005` with paired CI95 lower bound `>0`;
- positive selector gains for both generators;
- nonnegative collision, drivable, TTC, and safety-pass-bucket deltas;
- worst selected-token delta `>-0.5`.

If total system gain passes but the same-selector GRPO increment fails, report a
selector improvement only and do not attribute the total gain to GRPO.

### E. Generalization and final evaluation

Do not retrain the selector for Stage-9 formal seeds 1/2. All three generators must
be positive relative to base + value on dev-select before unlocking dev-confirm.
Dev-confirm requires total gain `>=+0.003`, positive GRPO increment, and nonnegative
safety components. Only then prepare external-cluster scripts and one formal
full-navtest sequence. The paper-level engineering target remains three-seed mean
absolute PDMS delta `>=+0.005`, with seed-stratified and token-clustered CI lower
bounds above zero.

## 5. Interfaces, tests, and stop rules

Add compatible configuration for `grpo_training_mode=value_selector` and
`inference_selector_source=value_top2`, plus explicit selector/calibration
provenance in checkpoints and evaluator artifacts. Evaluation records fallback and
challenger modes, switch decisions, predicted components/uncertainty, true switch
gain, checkpoint SHA, calibration SHA, and token SHA.

Tests cover final-trajectory dependence, BEV sampling shapes, component composition,
bootstrap reproducibility, top-2/tie/invalid/fallback behavior, frozen gradient
boundaries, absence of metric-cache use at inference, old-checkpoint compatibility,
manifest isolation, calibration, and artifact pairing. Focused pytest, Python/Bash
syntax, checkpoint audit, and `git diff --check` must pass.

Do not scan learning rate, head count, top-k, loss weights, or safety thresholds. Do
not inspect dev-confirm or full-navtest early. Do not use external 8-GPU resources
until the independent dev-select gate passes and the user supplies the cluster
script requirements.

## 6. Execution record

### 2026-07-20 — implementation, manifests, and U8 audit

- Implemented final-trajectory encoding, trajectory-aligned BEV sampling,
  agent/ego/status context, three bootstrap component heads, exact PDM composition,
  top-2 ranking, conservative inference, independent adapter loading, evaluator
  diagnostics, calibration, manifest, runner, and checkpoint-audit tools.
- Locked manifests: train `4,283` / SHA
  `158db13ac32b3796df74c97fda67201d8d170a36b3ca4be61dcacfffe1a51d92`;
  calibration `918` / SHA
  `88684e137a1e6aeaec6afec07f1cb85590777ae471de498b02db80ebb8dfe3de`;
  test `918` / SHA
  `49d84f7a06f53cd7ee65ccaf776f521805e5823582569e93fd0bd4548f82f06c`.
  The builder verified zero overlap with fixed-1024, dev-select, and dev-confirm.
- U8 experiment:
  `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage15_value_top2_seed0_audit/2026.07.20.15.38.38`.
  All 40 candidates were valid. Epoch mean value-selector gradient was
  `2.5252471`; decoder and perception gradients and decoder weight changes were
  exactly zero.
- Independent adapter loading and value-top2 evaluation completed on eight
  calibration tokens with zero failures. The selector checkpoint SHA is
  `f904d02ddb5f3553c9b528e1e2076c29d0d9e1e4e97313b22ace2b4b0351e1e4`.
  U8 produced no switch and is implementation evidence only.
- Focused Stage-11/13/14/15 regression set: `36 passed`. Python/Bash syntax and
  `git diff --check` passed.
- Formal four-epoch seed-0 training started from the locked Stage-9 U128 generator
  at `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage15_value_top2_seed0_formal/2026.07.20.15.42.28`.

### 2026-07-21 — network-interruption recovery

- Epoch checkpoints at global steps `2142`, `4284`, and `6426` completed. Their
  selector-calibration losses were respectively `0.897656`, `0.876108`, and
  `0.860891`; checkpoint choice remains loss-only.
- The connection/process died during the fourth epoch at observed step `7536`,
  before an epoch checkpoint existed. Those partial updates are discarded.
- Recovery is locked to the structurally audited epoch-boundary checkpoint
  `grpo-02-6426.ckpt`. Lightning restores model, optimizer, scheduler, epoch,
  global-step, and RNG state, then runs only the fourth epoch to step `8568`.

### 2026-07-21 — calibration selection and selector-test result

- The recovered fourth epoch completed at `grpo-03-8568.ckpt`. Its structural
  audit passed: update/global step `8568`, 38 finite selector tensors, zero
  generator mismatch, and zero reference mismatch.
- Selector-calibration losses for epochs 1–4 were `0.897656`, `0.876108`,
  `0.860891`, and `0.876006`. The loss-only rule therefore locked epoch 3
  (`grpo-02-6426.ckpt`, SHA
  `f3c6a3015ad8eef3759002b7d504ca10b2d4f9419b4b1314c4b5b7fd86fc1eb0`).
- The one allowed calibration selected margin `0.024751663208007812`. On the
  calibration split it switched 18/918 tokens, gained `+0.0006804682`, had
  zero mean decline in the three safety components, no harmful switch, and a
  worst switch gain of `+0.00976437`.
- The locked 918-token selector-test gate **failed**. Value top-2 versus the
  frozen selector gained only `+0.0005946055`, with paired CI95
  `[-0.0001756457, +0.0011843845]`. Switch rate was 30/918 (`3.27%`), but TTC
  declined `-0.0010893246`. The three formal failures were: delta below
  `+0.002`, CI lower bound not above zero, and safety decline. Fixed-1024 and
  dev-select remain locked and were not run.
- Post-mortem: frozen top-2 oracle headroom on selector-test was still
  `+0.0116513120`, so candidate quality was not the bottleneck. The predicted
  safety rule made only 112/918 scenes eligible; even a true-reward oracle over
  those eligible challengers could gain only `+0.0010635137`. One catastrophic
  TTC false-safe switch predicted challenger TTC lower bound `0.9309233` while
  true TTC was `0`, causing `-0.2614738` token loss. The next selector stage
  must address safety calibration/rare-event supervision and avoid treating
  bootstrap standard deviation as a reliable safety confidence bound.
