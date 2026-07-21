# DiffusionDrive GRPO Stage 14: Collision-Truncated Intra-Anchor GRPO

Date registered: 2026-07-20
Branch: `experiment-after-stage9`
Starting commit at registration: `db8ed812567d2754d57c8faa00d60c9fec9ce561`

## 1. Research objective

Stage 14 tests one previously unimplemented credit-assignment mechanism:

```text
K=2 intra-anchor GRPO + absolute collision truncation
```

The goal is to preserve the statistically positive within-anchor signal found in
Stage 6 while preventing the candidate-set/oracle collapse repeatedly observed from
U64 to U128 in Stages 6--8.

This is a causal bridge experiment, not a novelty claim. The truncation rule follows
the public DiffusionDriveV2 formulation
<https://arxiv.org/abs/2512.07745>. Stage 14 determines whether that missing mechanism
explains the gap between this repository's small GRPO gain and published multi-point
systems under the current DiffusionDrive checkpoint, reward implementation, and
evaluation protocol.

Stage 14 does **not** add multiplicative exploration noise, a new selector, extra
training data, imitation loss, or architecture changes. If truncation works, a later
stage may isolate multiplicative noise and then develop an original extension.

## 2. Why this is not a repeat of earlier stages

- Stage 6 `within_anchor` used a symmetric standardized advantage. A safe rollout
  with lower reward than its sibling received a negative update; its U64 oracle fell
  by `-0.002944`.
- Stage 6 `hierarchical` mixed that signal with a reference-selected absolute
  advantage. About 84% of candidates were negative; dev-select improved only
  `+0.0009843908`, while oracle was `-0.0004424095`.
- Stage 7 `anchor_hierarchical` repaired cross-intent baseline mismatch but U128
  oracle still fell by about `-0.0031` for K2 and K4.
- Stage 8 RLOO and the hard-trust route retained symmetric negative relative updates
  and still produced the same catastrophic oracle token.
- Stage 9's plain K1 group-zscore produced the strongest completed full-navtest
  evidence, but only `+0.0013528957` normalized PDMS (`+0.1353` paper points).
- Stages 11--13 showed that a frozen base selector removes some selector drift, but
  does not increase the generator gain enough for a performance claim.

No completed stage used the rule “reward relative improvements, but penalize only
absolute safety failures.”

## 3. Locked objective

For each scene and anchor, generate exactly two stochastic denoising traces. Let the
valid raw PDM scores be `r[k, 1]` and `r[k, 2]`. First compute the existing detached
within-anchor standardized advantage `A_intra` using `grpo_advantage_eps=1e-3`.
Then apply:

```text
A_stage14[k, i] = -1                         if NO_COLLISION[k, i] < 1
                  max(0, A_intra[k, i])      otherwise
```

Invalid candidates are excluded. An anchor is optimized only when it has two valid
rewards or a valid collision candidate. Collision is read from the scorer's existing
`MultiMetricIndex.NO_COLLISION` component; aggregate PDMS equal to zero is not used as
a proxy. Static-object partial collision scores (`0.5`) count as collisions.

The policy loss remains the existing clipped GRPO/PPO surrogate. The existing
generation KL to the immutable base reference remains `0.1`. Classification and
perception stay frozen. This objective is named `collision_truncated_intra_anchor`.

Required online diagnostics:

- valid-anchor fraction and within-anchor reward gap;
- positive-truncated, safe-zeroed, and collision-penalty fractions;
- collision candidates, valid candidates, and optimized candidates;
- mean/max generation KL, policy ratio, clipping fraction, and gradient norms;
- exact zero classification/perception gradients.

## 4. Locked training recipe

- immutable base and reference checkpoint:
  `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model`;
- `generation` mode, raw PDMS, uniform scene and anchor weighting;
- K=2 rollouts per anchor, batch size 2, FP32, one RTX 4090, no DDP;
- LR `1e-6`, PPO clip `0.2`, generation KL coefficient `0.1`;
- old-policy sync every 32 optimizer steps;
- both current decoder refinement layers trainable;
- truncation timestep/schedule `8 -> 0`, 125 scheduler inference steps;
- additive scheduler exploration already used by the repository; no new noise;
- seed 0 for mechanism selection; seeds 1/2 only after the independent gate;
- U8 audit and one formal U128 run from base, preserving U64/U128 checkpoints;
- no reward shaping, priority sampling, reference-centered term, RLOO term,
  selector weighting/training, projection, adaptive KL, imitation/GT loss, soup,
  conditional fallback, or hyperparameter scan.

Current-selector evaluation is the primary single-model result. Frozen-reference
selector evaluation is a paired diagnostic and may not replace a failed primary
gate. Candidate/oracle results are diagnostics and safety guards, never deployment
scores.

## 5. Implementation and unit-test gate

Before any real training:

1. Add component scores to the training prediction payload without changing legacy
   modes.
2. Implement the new advantage mode with strict K=2 and shape/finite validation.
3. Add a dedicated Stage-14 runner that rejects all recipe deviations.
4. Add tests proving:
   - safe positive advantages are retained;
   - safe negative advantages become exactly zero;
   - collision scores `0` and `0.5` become `-1` even when locally best;
   - safe ties have zero policy gradient;
   - invalid/single-valid/all-invalid groups behave fail-closed;
   - legacy objectives are numerically unchanged;
   - component scores are detached and aligned with flattened K2 candidates;
   - classification/perception gradients remain exactly zero.
5. Run the focused Stage 6--14 regression suite. Any failure stops before training.

## 6. Phase A: U8 real-batch audit

Run a separate U8 smoke from the immutable base. It passes only if:

- exactly 8 optimizer steps complete with finite loss/gradients;
- both decoder refinement layers receive nonzero finite gradients;
- classification and perception gradients are exactly zero;
- collision/positive/zero fractions are finite and sum to the optimized valid
  population;
- collision penalty is observed if the batch stream contains a collision; absence of
  a collision is recorded and checked with unit tests, not manufactured by changing
  the data;
- rolling generation KL is finite and `<= 5e-4`.

The U8 checkpoint is never a formal resume source.

## 7. Phase B: seed-0 U64/U128 development gates

Run one fresh U128 training job from base and retain both checkpoints. Evaluate U64
and U128 sequentially on the locked fixed-256 set with the current selector.

A checkpoint is eligible for fixed-1024 only if:

- selected delta `> -0.001`;
- oracle delta `>= -0.001`;
- collision, drivable, and TTC deltas each `>= -0.002`;
- worst token oracle delta `> -0.5`;
- artifact identity, token alignment, and completion checks pass.

If neither is eligible, stop Stage 14. Evaluate every eligible checkpoint on the
locked fixed-1024 set. A checkpoint remains eligible only if:

- selected delta `> 0`;
- oracle delta `>= -0.0005`;
- candidate-mean delta `>= 0`;
- collision, drivable, progress, and TTC deltas each `>= 0`;
- worst selected and oracle token deltas are each `> -0.5`.

Choose exactly one seed-0 checkpoint by highest fixed-1024 selected delta. Ties within
`0.00025` choose U64. This duration-selection rule is registered before either
artifact is generated.

## 8. Phase C: independent dev-select gate

Evaluate base then the locked seed-0 checkpoint on the untouched 3,072-token
dev-select manifest using the current selector. The mechanism passes only if:

- selected delta `>= +0.0015` and paired bootstrap CI95 lower bound `> 0`;
- oracle and candidate-mean deltas are each `>= 0`;
- collision, drivable, progress, and TTC deltas are each `>= 0`;
- worst selected and oracle token deltas are each `> -0.5`;
- all 3,072 unique tokens and identities align exactly.

Run the frozen-reference selector on the same checkpoint only after the primary
artifact is complete. It is diagnostic and cannot rescue a failed current-selector
gate.

Failure stops before extra seeds or any multiplicative-noise experiment.

## 9. Phase D: confirmation hierarchy

If Phase C passes:

1. Train fresh seeds 1 and 2 with the locked duration and recipe.
2. Require all three fixed-1024 deltas positive, mean delta `>= +0.0015`,
   seed-stratified CI95 lower bound `> 0`, mean oracle/candidate and safety component
   deltas nonnegative.
3. Evaluate the untouched 1,024-token dev-confirm split once. Require all seed deltas
   positive, seed-stratified CI lower bound `>= 0`, and nonnegative mean safety
   components.
4. Only then evaluate full-navtest once per seed and report normalized and 0--100
   PDMS units.

Result labels on full-navtest:

- feasibility replication: statistically positive three-seed mean;
- useful method effect: mean delta `>= +0.005` normalized (`+0.5` PDMS point);
- strong paper-scale effect: mean delta `>= +0.010` (`+1.0` point);
- below `+0.005`: retain as an ablation, not the main performance claim.

## 10. Stop discipline and next-stage boundary

- Gates are authoritative; do not change thresholds after seeing results.
- Operational failures may rerun the identical command after provenance recovery.
- Do not inspect dev-confirm or full-navtest early.
- Do not replace failed seeds, average checkpoints, or use frozen-selector fallback.
- Do not tune collision penalty, K, LR, KL, duration, or sampler inside Stage 14.
- If Stage 14 passes Phase C but remains below `+0.005`, Stage 15 may add only
  scale-adaptive multiplicative exploration noise as a preregistered causal ablation.
- If oracle/candidate safety still collapses, stop this route: the next plan must use
  an explicit constrained optimizer or learned value/selection mechanism rather than
  another relative-advantage variant.

## 11. Execution checklist

- [x] Hypothesis, formula, provenance, gates, and stop rules registered
- [x] Implementation and focused regression suite pass
- [x] U8 real-batch audit passes
- [x] Fresh seed-0 U128 job completes; U64/U128 checkpoint audits pass
- [x] Fixed-256 safety gate selects eligible checkpoints
- [ ] Fixed-1024 locks exactly one duration — FAILED; Stage 14 stopped
- [ ] Independent dev-select current-selector gate passes
- [ ] Frozen-reference selector diagnostic recorded
- [ ] Fresh seeds 1/2 and three-seed fixed-1024 gate pass
- [ ] Untouched dev-confirm gate passes
- [ ] Full-navtest three-seed result and paper-scale label recorded

## 12. Execution log

Append a timestamped block for every implementation check, training/evaluation job,
gate, stop, and recovery decision. Record exact command, exit code, GPU, checkpoint,
token count/SHA, primary and diagnostic selectors, selected/candidate/oracle metrics,
PDM components, tails, KL, gradient boundaries, and explicit CONTINUE/STOP decision.

### 2026-07-20 — implementation and tests

- Added component-score propagation and the
  `collision_truncated_intra_anchor` K2 objective. Safe positive within-anchor
  advantages are retained, safe non-positive advantages are zeroed, and any valid
  `NO_COLLISION < 1` candidate receives advantage `-1`.
- Added the locked Stage-14 runner, checkpoint audit, and focused objective/runner/
  callback tests.
- Stage-14 tests passed: objective/protocol `13 passed`; callback correction included.
  The focused generation/Stage9--14 subset passed `47 passed`.
- The broader selected regression command produced `108 passed, 3 failed`. All three
  failures are pre-existing selector-validation result-key assertions
  (`rank_loss`/`active_scene_fraction`) outside the changed generation path.
- Two operational failures were recovered without accepting a scientific result:
  the first attempt stopped before training on the model mode whitelist; the second
  completed 8 steps but failed at epoch-end because the legacy ablation callback
  monitored an undefined validation reward. The whitelist and save-all callback
  behavior were corrected and covered by tests before the clean rerun.

### 2026-07-20 — clean U8 real-batch audit

- Experiment:
  `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage14_collision_truncated_k2_seed0_audit/2026.07.20.13.21.34`.
- Completed exactly 8 optimizer steps from the immutable base and saved
  `grpo-00-8.ckpt`.
- Generation KL mean/max: `1.85518e-5 / 5.82107e-5`.
- Per-step valid candidates: exactly `80`. Mean positive / safe-zero / collision
  fractions: `0.453125 / 0.4484375 / 0.0984375`; the partition sums to one.
- Mean collision candidates and optimized candidates per batch: `7.875 / 44.125`.
- Mean layer-0/layer-1 regression gradient norms:
  `50.2828 / 20.1818`; classification and perception gradient norms were exactly
  zero for all 8 steps.
- Decision: U8 PASS; continue to the one registered fresh seed-0 U128 job.

### 2026-07-20 — fresh seed-0 U128 training and checkpoint audit

- Experiment:
  `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage14_collision_truncated_k2_seed0_u128/2026.07.20.13.22.47`.
- Completed exactly 128 optimizer steps from base; retained
  `grpo-step-64.ckpt` and `grpo-step-128.ckpt`.
- Final step generation KL: `0.0003802010`, below the registered `5e-4` limit.
- Both structural audits passed. Each checkpoint contains one optimizer and one
  scheduler state, 32 changed current-decoder keys in each layer, 84 exact frozen
  reference tensors, and 84 valid old-policy tensors. Last sync is 32 at U64 and 96
  at U128.
- Audit JSON:
  `artifacts/grpo_stage14/checkpoints/u64_audit.json` and
  `artifacts/grpo_stage14/checkpoints/u128_audit.json`.

### 2026-07-20 — fixed-256 current-selector safety gate

- U64 artifact: `artifacts/grpo_stage14/u64_fixed256_current.json`.
  Selected `+0.00124371`, oracle `+0.00014882`, candidate mean
  `+0.00120050`; minimum selected/oracle token deltas
  `-0.0150562 / -0.0159241`; collision/drivable/progress/TTC deltas
  `0 / 0 / +0.00016118 / +0.00390625`. U64 is eligible.
- U128 artifact: `artifacts/grpo_stage14/u128_fixed256_current.json`.
  Selected `+0.00086784`, but oracle `-0.00202808` and minimum oracle token
  `-0.78689671`; progress `-0.00047780`. U128 triggers the oracle-mean and
  catastrophic-token hard stops and is ineligible.
- Decision: evaluate only U64 on fixed-1024. Do not evaluate or select U128.

### 2026-07-20 — fixed-1024 primary gate and final decision

- Artifact: `artifacts/grpo_stage14/u64_fixed1024_current.json`; completed
  `1024/1024`, token SHA
  `fc2f4c607de003688f5fbfc1ab5f97568b3c334e4a34c3cb4ae2fe3de994df25`,
  current selector, registered `8 -> 0` schedule.
- Selected delta: `-0.0002446672`, paired CI95
  `[-0.0024779447, +0.0012616901]`; wins/ties/losses
  `609/141/274`.
- Oracle delta: `+0.0009988928`. The generator candidate-set upper bound improves,
  but the deployed current selector does not realize that gain.
- Registered gate requires selected delta `> 0`; therefore the gate fails.
- **Decision: STOP Stage 14.** Do not run dev-select, frozen-selector diagnostic,
  seeds 1/2, dev-confirm, full-navtest, multiplicative-noise Stage 15, or distributed
  formal training under this recipe.
- Interpretation: collision truncation delays the U64 candidate collapse and improves
  oracle on fixed-1024, but does not solve generator--selector credit assignment.
  The next plan must target deployable selection/value alignment rather than another
  relative-advantage or duration sweep.
