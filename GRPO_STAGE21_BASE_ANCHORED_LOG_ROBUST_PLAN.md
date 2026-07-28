# Stage21: Base-Anchored Log-Robust Selected-Anchor GRPO

Date registered: 2026-07-21
Status: preregistered before implementation, training, or Stage21 result inspection

## 1. Objective and evidence

Stage20 showed that Stage19 retained large gains on difficult validation scenes
but damaged mature scenes: base PDMS below 0.5 improved by roughly 0.06, while
base PDMS above 0.75 declined by roughly 0.008--0.012.  Stage21 therefore keeps
same-selected-anchor G=8 GRPO but requires every reinforced rollout to beat the
frozen base and explicitly preserves mature scenes.  Inference remains one
DiffusionDrive generator followed by the frozen reference selector; there is no
fallback, oracle, learned selector, or extra sampling.

All Stage21 training starts from the registered strong base, SHA256
`59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d`.
Stage19 weights are not continued.

## 2. Locked data protocol

Generate one second frozen-base full-schedule artifact on all 6,119 navtrain
tokens with noise namespace `20260728`.  Combine it with the existing default
artifact to record per token: selected mode, mean base reward, mean base
collision/drivable/TTC, log identity, checkpoint/schedule SHA, and deterministic
weight metadata.

Assign all 908 navtrain logs to six folds with seed `20260728`, balancing token
count, base-reward bins `[0,.5),[.5,.75),[.75,.9),[.9,1]`, safety failures, and
selected-mode counts.  Folds 0--3 are fit, fold 4 is calibration, and fold 5 is
an internal one-shot test.  Logs never cross folds.  Calibration/test each must
contain at least 130 logs; token imbalance may not exceed 15%, hard-scene share
may not differ by more than two percentage points, and selected-mode total
variation may not exceed 0.05.

Stage20 dev-select is consumed development stress only.  It cannot fit weights,
select epochs, or tune constants.  Dev-confirm1024 and navtest remain protected.

## 3. Locked objective

For every selected-anchor G=8 group, compute group z-score `z_i`, frozen-base
mean reward `b`, and `delta_i=r_i-b`.  The dead zone is 0.01, absolute scale is
0.10, and advantage magnitude is capped at 2.0:

- any collision/drivable/TTC regression relative to base gives advantage -2;
- `delta_i>0.01` permits only positive advantage, combining positive group
  evidence and base-relative magnitude equally;
- `delta_i<-0.01` permits only negative advantage, combining negative group
  evidence and base-relative magnitude equally;
- candidates in the dead zone receive zero.

Thus a group winner that remains worse than base is never reinforced.

Scene policy weight is inverse log frequency times
`1+2*clip((0.75-b)/0.75,0,1)`, clipped to `[0.25,4]` and normalized.  Safety
penalties retain weight at least one.  Per-scene BC is 0.1 through base reward
0.75 and rises linearly to 0.3 at reward 1.0.  Both regression decoder layers
remain trainable, with layer-0 LR `1e-7` and layer-1 LR `1e-6`; classification,
perception, reference, and selector weights remain frozen.

The route is `diffgrpo_selected_anchor_base_preserve`.  G=8, gamma 0.6, full
`[32,24,16,8,0]` schedule, 125 scheduler steps, FP32, and all other Stage19
DiffGRPO mechanics remain fixed.  No margin, scale, LR, weight, BC, group-size,
schedule, or reward scan is allowed.

## 4. Execution and gates

Run U8 and resumed U32 single-GPU audits first.  Require exact advantage signs,
active policy/BC gradients, finite values, correct layer LRs, and zero forbidden
gradient/weight changes.

Development seed0 uses folds 0--3 on eight GPUs for at most eight epochs, saving
epochs 2/4/6/8.  Evaluate fold 4 under default and namespace `20260729`.
An epoch is eligible only when C-B is at least +0.005 with log-cluster CI lower
above zero, base<0.75 gain is at least +0.010, base>=0.75 delta is at least
-0.002, safety has neither material (>0.002) nor significant regression, and
the catastrophic-rate one-sided 95% upper bound is at most 0.005.  Select the
highest C-B; within 0.001 choose the earlier epoch.

Open fold 5 once under default plus namespaces 20260729/30/31.  Every namespace
must have C-B positive, four-noise mean at least +0.005, log-cluster CI lower
above zero, and the same difficulty/safety/tail gates.  Then run the locked
checkpoint once on consumed dev-select3072 with default and 20260729.  It must
achieve C-B at least +0.005 with positive log-cluster CI lower and the same
mature-scene and safety guards.  Failure at either gate stops Stage21.

Only after those gates, train seed0 fresh on all 6,119 tokens for the locked
epoch.  Dev-confirm requires C-B>=+0.005, C-A>=+0.010, positive CI lower, and
unchanged difficulty/safety gates.  Full navtest labels are: positive feasibility
for C-B>0 with positive CI lower, useful gain at +0.005, and paper target at
+0.010.  Seeds1/2 begin only after seed0 reaches +0.005.

## 5. Tests and stop discipline

Tests cover better/worse/dead-zone advantages, safety override, invalid and
zero-variance groups, fold balance and zero log overlap, log/difficulty weights,
dynamic BC, layer LR groups, DDP/resume/RNG continuity, checkpoint provenance,
shard invariance, and exact Stage13--20 compatibility.

No Stage21 result may be rescued by Stage17 fallback, checkpoint soup, threshold
tuning, longer training, or opening a later protected set.  If dev-select fails,
the next independent stage may add a residual adapter/gate; Stage21 itself stops.

## 6. Execution record

- 2026-07-21: plan recorded before Stage21 implementation and result inspection.
- 2026-07-22: generated the independent namespace-20260728 frozen-base
  artifact for all 6,119 navtrain tokens with zero failures.  Merged artifact
  SHA256 is `ea149b3088d12f2c89a94ec4f7928e025178d9992c092fca09850e0aa27773fd`.
- 2026-07-22: constructed the seed-20260728 whole-log folds.  Fold token
  counts are `[1019,1021,1016,1019,1021,1023]`, log counts are
  `[152,151,151,152,151,151]`, hard-share range is `0.0021392`, maximum
  selected-mode TV is `0.0091735`, and log overlap is zero.  The locked audit
  is `artifacts/grpo_stage21/manifests/fold_audit.json`.
- 2026-07-22: Stage16--21 compatibility suite passed 46 tests.  U8 and resumed
  U32 passed finite-loss, active policy/BC gradient, exact LR-ratio, frozen
  parameter, and frozen-reference audits.  Independent U32 replay preserved
  all frozen tensors bitwise; permitted model and Adam-state maximum absolute
  differences were `7.45e-9` and `1.54e-8`, respectively, and were recorded
  explicitly as numerically equivalent rather than bitwise equal.
- 2026-07-22: started the one preregistered seed0 eight-GPU, eight-epoch
  development run on folds 0--3 from the registered strong base.
- 2026-07-22: the development run finished naturally at 8 epochs / 512
  optimizer steps.  The final checkpoint audit passed: 64 permitted tensors
  changed (32 in each decoder layer), no frozen tensor changed, the reference
  policy matched the base exactly, all 512 logged steps were finite, and the
  effective layer learning rates were `1e-7` and `1e-6`.
- 2026-07-22: completed the locked fold-4 calibration for epochs 2/4/6/8 under
  both registered noise namespaces, with 1,021/1,021 tokens and zero failures
  in every cell.  Epochs 4/6/8 produced mean C-B gains of
  `+0.0137879/+0.0149266/+0.0157736`, with positive log-cluster CI lower
  bounds `+0.0050569/+0.0060277/+0.0065387`.  Their hard-scene gains were
  `+0.0658138/+0.0680228/+0.0802350`, but mature-scene deltas were
  `-0.0038955/-0.0031205/-0.0061365` (registered minimum `-0.002`) and their
  catastrophic-rate upper bounds were `0.0064292/0.0064292/0.0089008`
  (registered maximum `0.005`).  Epoch 2 preserved the mature/tail gates but
  its mean gain was only `+0.0031760`, below the registered `+0.005` minimum.
  Consequently no checkpoint was eligible.  Stage21 stops before fold 5,
  dev-select, formal retraining, dev-confirm, or navtest; no longer training,
  threshold relaxation, checkpoint soup, or fallback rescue is performed.
