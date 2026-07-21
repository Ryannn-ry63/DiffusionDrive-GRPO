# DiffusionDrive GRPO Stage 16: Full-Chain DiffGRPO

Date: 2026-07-21
Status: preregistered implementation and execution plan
Scope: direct reward optimization of the DiffusionDrive trajectory generator.

## 1. Objective and attribution boundary

Stage 15 established that frozen-selector top-2 trajectories retain
`+0.0116513120` raw-PDMS oracle headroom, but its learned safety gate admitted
only 112/918 challengers and still produced a catastrophic TTC false-safe.  Stage
16 therefore freezes the Stage-15 selector branch and tests the primary scientific
claim directly: GRPO can improve the deployed PDMS of DiffusionDrive by changing
the generated trajectory distribution.

Formal evaluation separates three cells:

1. base weights, original `[8, 0]` schedule, frozen base selector;
2. base weights, full `[32, 24, 16, 8, 0]` schedule, frozen base selector;
3. Stage-16 GRPO weights, full schedule, frozen base selector.

Cell 3 minus cell 1 is total system improvement, cell 3 minus cell 2 is the GRPO
contribution, and cell 2 minus cell 1 is the schedule contribution.  Schedule
improvement may not be attributed to GRPO.

## 2. Locked method

Add a separate `diffgrpo_full_chain` generation algorithm while preserving all
legacy Stage-9--15 behavior.  The locked denoising schedule is
`[32, 24, 16, 8, 0]`, with scheduler inference steps 125 and stride 8.  The plan
anchors are noised at timestep 32.  The first four decisions use stochastic DDIM
transitions and the final decision uses the existing Gaussian trajectory action.
Inference uses the corresponding mean chain.

The 20 DiffusionDrive anchor modes form the per-scene GRPO group.  Valid raw PDMS
is standardized within each scene.  The scene-level advantage is propagated to
all five denoising decisions with `gamma=0.6`; log probabilities are averaged per
trajectory coordinate.  Rollouts are sampled on-policy from the current decoder.
The new algorithm does not use the legacy PPO behavior-policy ratio, old-policy
synchronization, or explicit reference KL.

A frozen base decoder samples a teacher chain from the same initial anchor noise.
The current decoder maximizes its likelihood with coefficient `0.1`.  Perception,
BEV, classification selector, Stage-15 value head, and reference decoder remain
frozen.  Deployment always uses the frozen base classification selector.

Locked training constants are FP32, AdamW, LR `1e-6`, raw PDMS, uniform 20-mode
grouping, `diffgrpo_bc_weight=0.1`, `diffgrpo_step_discount=0.6`, and mean
log-probability reduction.  Do not scan learning rate, BC coefficient, gamma,
schedule, or group size inside Stage 16.

## 3. Execution protocol and gates

### A. Full-schedule feasibility

On the existing 918-token Stage-15 calibration manifest, compare base `[8,0]`
against base `[32,24,16,8,0]`.  Full schedule requires 918/918 valid tokens,
selected delta at least `-0.005`, and collision/drivable/TTC deltas each at least
`-0.002`.  Failure stops Stage 16 without scanning another schedule.

### B. Local implementation audit

Run seed-0 U8 and U32 with batch size 2.  Require finite rewards, log probabilities,
losses, gradients, and weights; nonzero allowed decoder gradients; exactly zero
gradient and weight change in every frozen module; exact action replay; active BC;
and continuous optimizer/scheduler/RNG state after checkpoint resume.

### C. External development training

After the local audit passes, use 8 GPUs with per-device batch 1 and gradient
accumulation 8, for effective global batch 64.  Train on the locked 4,283-token
Stage-15 train manifest for at most ten epochs and retain epochs 1/2/4/6/8/10.
Evaluate only those checkpoints on calibration.  Select the highest total PDMS
checkpoint satisfying nonnegative collision/drivable/TTC means; within `0.0005`,
choose the earlier epoch.

Run the locked 918-token selector-test once.  Require total delta over original
base at least `+0.003` with CI95 lower bound above zero, GRPO delta over full-schedule
base at least `+0.002` with CI95 lower bound above zero, nonnegative safety-component
deltas, and worst-token delta greater than `-0.5`.  Failure stops fixed/dev work.

### D. Formal attribution and generalization

Retrain seed 0 from the original base on all 6,119 navtrain tokens for the locked
epoch count.  Fixed-1024 requires total delta `>=+0.003`, GRPO delta `>=+0.002`,
positive total CI lower bound, and nonnegative safety deltas.  Dev-select3072 then
requires total delta `>=+0.005`, GRPO delta `>=+0.003`, both CI lower bounds above
zero, and nonnegative safety deltas.

Only then train seeds 1/2.  Dev-confirm must pass before full navtest.  The minimum
paper target is three-seed mean raw-PDMS gain `>=+0.005`; the preferred target is
`>=+0.010`.  Seed-stratified and token-clustered CI lower bounds must both exceed
zero.

## 4. Interfaces, tests, and stop rules

Add explicit configuration and artifact provenance for the generation algorithm,
schedule, BC weight, step discount, log-probability reduction, group size, manifest
SHA, reference checkpoint SHA, and selector source.

Tests cover arbitrary decreasing schedules, DDIM transition/action replay,
discount ordering, on-policy gradient flow, teacher BC, group advantage, invalid
rewards, frozen boundaries, old-checkpoint compatibility, legacy `[8,0]` numerical
equivalence, distributed effective batch, checkpoint resume, and artifact pairing.

Do not inspect dev-confirm or navtest early.  Do not change gates after observing
results.  If Stage 16 again raises candidate/oracle quality without deployed GRPO
gain, stop and define a separate Stage 17 safety-risk/conditional-pairwise selector;
do not reactivate selector tuning inside this stage.

## 5. Execution record

- 2026-07-21: plan recorded before implementation.  No Stage-16 code or experiment
  was run before the protocol above was locked.
- 2026-07-21: implementation and 50 focused Stage-9/10/14/15/16 regression tests
  passed.  The preregistered 918-token schedule gate passed: base full-chain minus
  base original selected PDMS was `+0.0102748161` raw (paired CI95
  `[+0.0034221934, +0.0169904807]`).  Collision and drivable deltas were each
  `-0.0010893246`, TTC was unchanged, so all locked feasibility thresholds passed.
  Artifacts are under `artifacts/grpo_stage16/schedule/`.
- 2026-07-21: seed-0 U8 and resumed U8-to-U32 local audits passed.  U8/U32
  reached exact global steps 8/32 with one optimizer and one scheduler state,
  64 allowed decoder tensors changed, and classification/reference/all other
  base tensors had zero mismatches.  Every recorded reward, loss, current/BC
  log probability, and gradient was finite; both decoder layers had positive
  shared/regression gradients at every step, while classification, perception,
  and Stage-15 value-selector gradients were exactly zero.  The U32 resumed
  interval covered steps 8--31.  Machine-readable audits are under
  `artifacts/grpo_stage16/audit/`.
- 2026-07-21: completed the locked 8-GPU seed-0 development run for 10 epochs
  (effective global batch 64, 4,283 training tokens, 670 optimizer steps).  All
  670 steps and the epoch-10 checkpoint passed finite-value and frozen-boundary
  audits.  Calibration selected rewards at epochs 1/2/4/6/8/10 were
  `0.8043818 / 0.8103938 / 0.8164512 / 0.8191258 / 0.8209322 / 0.8206648`.
  The preregistered safety and 0.0005 earlier-epoch tie break selected epoch 8.
  Its calibration system-total delta was `+0.0273758409` and its isolated GRPO
  delta was `+0.0171010249` raw.
- 2026-07-21: the one-shot 918-token selector-test produced original-base
  `0.7994323960`, full-schedule-base `0.8108352506`, and epoch-8 GRPO
  `0.8310759676`.  System total was `+0.0316435715` with CI95
  `[+0.0227174207, +0.0412711030]`; isolated GRPO was `+0.0202407170` with CI95
  `[+0.0114594392, +0.0298628258]`.  Collision, drivable, and TTC system deltas
  were positive.  The gate nevertheless failed its locked worst-token rule:
  token `26cb5b136b8652c7` changed from original `0.9113325` (full base
  `0.9392080`) to GRPO `0.0`, for a system delta of `-0.9113325`.  The frozen
  selector chose mode 19, which became collision/TTC-invalid, while GRPO mode 9
  retained reward `0.9688336`.  Per protocol, Stage 16 stops here; fixed1024,
  formal retraining, dev-select, and navtest were not run.  A separate Stage 17
  must address per-mode tail-risk/fallback selection without relabeling the
  Stage-16 mean gain as a passed result.
