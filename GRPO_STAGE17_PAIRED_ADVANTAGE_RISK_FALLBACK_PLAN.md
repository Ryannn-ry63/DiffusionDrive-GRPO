# DiffusionDrive GRPO Stage 17: Paired Advantage-Risk Fallback

Date: 2026-07-21
Status: preregistered implementation and execution plan
Scope: improve Stage-16 deployed PDMS while eliminating rare same-mode tail failures.

## 1. Objective and attribution

Stage 16 established a statistically significant full-chain GRPO contribution of
`+0.0202407170` raw PDMS on its one-shot selector-test, but failed the locked
worst-token gate because a reference-selected mode changed from a safe base
trajectory to a collision/TTC-invalid GRPO trajectory.  Stage 17 freezes the
Stage-16 epoch-8 generator and adds a same-mode paired fallback.  It must improve,
not merely preserve, the average result whenever the learned decision is reliable.

The existing paired artifacts show useful headroom.  On calibration, the exact
same-mode oracle improves full-schedule base by `+0.0218901624`, which is
`+0.0047891375` above Stage 16.  On the already observed selector-test, the same
diagnostic values are `+0.0240263409` and `+0.0037856239`.  The selector-test is
diagnostic only and may not be used for Stage-17 fitting, selection, calibration,
or gates.

Formal evaluation contains four cells: A is base weights with `[8,0]`; B is base
weights with `[32,24,16,8,0]`; C is Stage-16 weights with the full schedule and
reference selector; D is C with the Stage-17 paired fallback.  D-B is the deployed
GRPO contribution, D-C is the Stage-17 incremental effect, and D-A is total system
improvement.

## 2. Locked method

Both the frozen Stage-16 and base decoders run the full schedule from shared
initial noise.  Frozen base/reference classification logits select mode `m`.
Stage 17 compares only `GRPO[m]` and `base[m]`; it never reranks across modes.  If
the paired fallback score exceeds one calibrated threshold, deployment returns the
exact base trajectory for mode `m`; otherwise it returns the GRPO trajectory.

The paired head encodes both eight-pose trajectories, samples frozen BEV features
along each trajectory, and fuses agents, ego-query, and status context with the
current/base representation, difference, and absolute difference.  It predicts
six binary events: GRPO reward below base, raw-PDMS delta at most `-0.1`, raw-PDMS
delta at most `-0.5`, and relative collision, drivable-area, or TTC regression.
An auxiliary SmoothL1 output predicts raw-PDMS delta but is not used directly at
inference.  The fallback score is the maximum of the six sigmoid probabilities.
Inference may not use metric caches or future labels.

The locked generator checkpoint is
`grpo-07-536.ckpt`, SHA
`3a7641d4ac2a9d644eda4cb945d1902d4ac4bebfdad09b156676dc0cbed94e23`.
The locked base/reference SHA is
`59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d`.
The schedule is `[32,24,16,8,0]`, truncation 32, and 125 scheduler inference
steps.

## 3. Data, optimization, and selection

Use the 4,283-token Stage-15 train manifest and all 20 same-mode pairs.  Split
tokens deterministically by `SHA256("stage17:20260721:" + token)` into 3,427
risk-fit and 856 risk-validation tokens.  Stage-15 calibration-918 is reserved for
one threshold calibration after checkpoint selection.

Freeze perception, both generators, classification, regression, and the Stage-15
value selector.  Train only the paired head in FP32 with AdamW, LR `1e-4`, weight
decay `1e-4`, weighted BCE for the six events, and auxiliary delta-loss weight
`0.25`.  Positive weights are computed on risk-fit and capped at 100.  The
reference-selected mode receives four times the loss weight of other modes.

Run local U8/U32 and a two-epoch pilot first.  Formal distributed training uses
eight GPUs, per-device batch 1, gradient accumulation 4, effective batch 32, and
at most 12 epochs with checkpoints 1/2/4/6/8/10/12.  Select by
`0.5 * macro-AUPRC + 0.5 * minimum-label-AUPRC` on risk-validation; within 0.005
choose the earlier epoch.  Before calibration is opened, extend to epoch 24 only
if epoch 12 is selected and improves the selection metric over epoch 10 by at
least 0.005; then also inspect epochs 16/20/24.

On calibration, select the single score threshold maximizing D PDMS subject to:
D-C at least `+0.001`; D-B no lower than C-B; fallback rate in `[0.01,0.35]`;
nonnegative collision/drivable/TTC deltas versus B; no admitted `delta<=-0.5`;
worst D-B token greater than `-0.5`; bottom-1% D-B CVaR at least `-0.1`; and a
one-sided 95% Clopper-Pearson upper bound on admitted catastrophic rate no greater
than `0.005`.  Within 0.0005 PDMS choose the lower fallback rate.  If no threshold
is eligible, stop.

## 4. Gates and external training

Seed-0 fixed1024 requires D-B at least `+0.010` with CI lower above zero, D-A at
least `+0.015` with CI lower above zero, D-C nonnegative, nonnegative safety
deltas, worst D-B greater than `-0.5`, bottom-1% CVaR at least `-0.1` and better
than C-B, and fallback rate no greater than 35%.  Failure stops without tuning.

Seed-0 dev-select3072 uses the same artifacts and additionally requires D-C at
least `+0.001` with CI lower nonnegative.  Only then train Stage-16 seeds 1/2 on
the external eight-GPU system for exactly eight epochs.  All seeds reuse the same
Stage-17 adapter and threshold.  Three-seed dev-select requires every D-B positive,
every D-C nonnegative, no safety/tail failure, mean D-B at least `+0.010`, and a
positive seed-stratified CI lower bound.

The untouched dev-confirm1024 is opened once after those gates.  It requires
three-seed mean D-B at least `+0.008`, D-A at least `+0.012`, positive D-C,
positive token-clustered and seed-stratified CI lower bounds, and unchanged safety
and tail gates.  The final navtest minimum is mean D-B `+0.005`; the paper target
is `+0.010`, with D-C nonnegative and both CI lower bounds above zero.

## 5. Interfaces, tests, and stop rules

Add compatible `paired_tail_risk_selector` training and inference modes, adapter
and calibration paths, threshold, manifests, generator/reference SHA provenance,
and evaluator diagnostics.  Tests cover shared-noise same-mode pairing, labels,
weighted loss, exact fallback, absence of inference metric-cache access, frozen
boundaries, resume/DDP batch accounting, artifact pairing, and legacy Stage-9--16
compatibility.

Do not use geometric-distance thresholds: calibration already shows catastrophic
changes at only 2--4 cm displacement.  Do not inspect fixed/dev to tune Stage 17.
Do not extend the generator for tens of epochs: Stage 16 peaked at epoch 8 and was
flat by epoch 10.  External long training is triggered only by the locked
risk-validation rule above.

## 6. Execution record

- 2026-07-21: this plan was recorded before any Stage-17 implementation or run.
- 2026-07-21: implemented the paired six-event risk/delta head, exact same-mode
  base fallback, frozen-boundary training route, manifest filtering, evaluator
  provenance, checkpoint/metric audits, calibration/gate tooling, and protocol
  tests.
- 2026-07-21: deterministically split the locked 4,283-token source into 3,427
  risk-fit and 856 risk-validation tokens. Risk-fit contains 68,540 valid mode
  pairs; capped positive weights are
  `[2.9781763306053746, 100, 100, 100, 100, 100]`.
- 2026-07-21: U8 and resumed U32 audits passed. Only all 38 paired-head tensors
  changed; Stage-16 generator/reference mismatches and forbidden-module gradient
  norms were zero, with no non-finite adapter tensors or scalar metrics.
- 2026-07-21: the eight-GPU two-epoch pilot completed at global steps 108/216.
  Both checkpoints passed the frozen-boundary audit. On all 17,120 independent
  risk-validation pairs, the locked selection metric increased from `0.0760610`
  at epoch 1 to `0.0976564` at epoch 2; all six event AUPRCs improved, so epoch 2
  was selected for continuation.
- 2026-07-21: resumed the selected epoch-2 checkpoint on eight RTX 4090 GPUs for
  the locked formal schedule with `max_epochs=12`, meaning ten additional epochs
  and twelve total, while preserving optimizer and scheduler state. Calibration,
  fixed/dev, and selector-test remain unopened.
- 2026-07-21: the 12-epoch formal run completed at global step 1,296. The final
  checkpoint and all 1,080 resumed steps passed frozen-boundary, finite-metric,
  and step-continuity audits. Risk-validation evaluated the preregistered epochs
  1/2/4/6/8/10/12 over the same 856 tokens and 17,120 valid pairs. Epoch 12 was
  selected with metric `0.1456741813`; epoch 10 scored `0.1248158434`, so the
  locked extension margin was `+0.0208583380`, exceeding `+0.005`.
- 2026-07-21: the selector therefore set `extend_to_epoch24=true`. Resumed the
  audited epoch-12 checkpoint with optimizer/scheduler state for twelve additional
  epochs and will inspect only epochs 16/20/24 before opening calibration.
- 2026-07-21: the epoch-24 continuation completed and passed all checkpoint and
  1,296-step metric audits. The locked comparison selected epoch 16 with metric
  `0.2388469161`, versus `0.1456741813` at epoch 12, `0.1371949102` at epoch 20,
  and `0.1606179301` at epoch 24. The selected SHA is
  `c8f898f21b7e22cf172bdab4b6e4f99bb1537b85c7bf0ed916d23d600ffecbdb`.
- 2026-07-21: the one-shot calibration-918 selected threshold `0.7683204412`.
  It passed with D-B `+0.0201373`, D-C `+0.0010588`, fallback rate `20.92%`,
  zero admitted catastrophic cases, worst D-B `-0.00236`, bottom-1% CVaR
  `-0.00147`, and nonnegative collision/drivable/TTC deltas.
- 2026-07-21: fixed1024 confirmed strong average gains but failed the preregistered
  absolute tail gates. The original four-artifact gate measured D-B `+0.0202487`
  with 95% CI `[+0.0117967,+0.0291022]`, D-A `+0.0280647`, D-C `+0.0013011`,
  and nonnegative safety means, but worst D-B was `-0.88684` and bottom-1% CVaR
  was `-0.28381`. Exact internally paired shared-noise diagnostics improve the
  averages to D-B `+0.0216623` and D-C `+0.0019324`, but retain three confident
  catastrophic false negatives (scores `0.0453/0.0673/0.1126`), worst `-0.82974`,
  and CVaR `-0.23057`. Per the stop rule, do not tune on fixed1024 and do not open
  dev-select; a new stage must address out-of-distribution tail detection.
