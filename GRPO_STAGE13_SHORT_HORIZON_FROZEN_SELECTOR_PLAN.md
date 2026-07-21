# DiffusionDrive GRPO Stage 13: Short-Horizon Frozen-Selector Confirmation

Date registered: 2026-07-20
Branch: `experiment-after-stage9`
Registration commit at planning time: `db8ed812567d2754d57c8faa00d60c9fec9ce561`

## 1. Objective and interpretation

Stage 13 formally tests the short-horizon candidate exposed by Stage 12:

```text
128-update generation-only GRPO + frozen base-policy selector
```

The hypothesis is not that longer training is beneficial. Stage 12 rejected that
hypothesis because no U512+ checkpoint had reliable continuation headroom. The new
hypothesis is that the useful window is near U128 and that freezing the deployed
selector can retain the generator's average improvement without the current-selector
mode-switch tail.

The existing Stage-9 seed-0 U128 checkpoint is a pilot only. It was identified after
fixed-1024 inspection and may not be included in the formal three-seed statistic.
If the pilot passes independent dev-select, formal seeds 0/1/2 are trained fresh from
the immutable DiffusionDrive base.

The reporting levels are:

- scientific success: three-seed full-navtest improvement is statistically positive
  and the mean safety components do not regress;
- new champion: the mean delta exceeds the existing generation-GRPO champion
  `+0.0013528957`;
- strong result: mean delta is at least `+0.003`;
- engineering stretch goal: mean delta is at least `+0.005`.

A significant result below `+0.003` remains valid evidence for GRPO feasibility and
must not be mislabeled as a failure of the method.

## 2. Locked provenance and method boundary

```bash
export ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
export PY=/root/miniconda3/envs/navsim/bin/python
export BASE_CKPT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model
export PILOT_U128=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage9_generation_epoch1_seed0/2026.07.18.15.58.29/lightning_logs/version_0/checkpoints/grpo-step-128.ckpt
export FIXED1024_BASE=$ROOT/artifacts/grpo_stage11/backtrack_base_reference_1024.json
export FIXED1024_TOKENS=$FIXED1024_BASE
export DEV_SELECT=$ROOT/artifacts/grpo_stage0/dev_select3072_manifest.json
export DEV_CONFIRM=$ROOT/artifacts/grpo_stage0/dev_confirm1024_manifest.json
export DEV_FEATURE_CACHE=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/grpo_dev4096_feature_cache_20260717
export DEV_METRIC_CACHE=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/metric_cache_trainval
export STAGE13_ARTIFACTS=$ROOT/artifacts/grpo_stage13
```

The formal training recipe is fixed:

- `generation_group`, raw PDMS, group-zscore advantage, uniform scene/mode weighting;
- both current decoder layers trainable; regression/shared blocks trainable;
- classification branches, perception, and full base reference frozen; old policy
  is non-trainable and synchronized from the current decoder every 32 steps;
- batch size 2, LR `1e-6`, generation KL weight `0.1`, PPO clip `0.2`;
- old-policy sync interval 32, exactly 128 optimizer updates;
- diffusion truncation/schedule `[8,0]`, 125 scheduler inference steps;
- FP32, one RTX 4090 per seed, no DDP;
- no adaptive-KL change, reward shaping, priority sampling, selector training,
  projection, imitation/GT loss, U256, or hyperparameter scan.

Deployment evaluation uses `selector_logits_source=reference`. This executes the
frozen base planning decoder to obtain a deterministic mode index and applies that
index to the GRPO-generated candidate set. It is not online PDM reranking and has no
conditional fallback. Its latency, memory, and throughput overhead must be recorded.

## 3. Implementation contract

Implement a Stage-13 runner and gate without changing the model objective:

- the runner must reject wrong modes, update counts, batch size, precision, hardware,
  resume provenance, DDP, base checkpoint, or reference checkpoint;
- U8 audits are separate non-formal jobs; each formal U128 begins fresh from
  `BASE_CKPT` and is never resumed from the pilot or an U8 checkpoint;
- the checkpoint audit must prove global step 128, one optimizer/scheduler, nonzero
  finite changes in both current decoder layers, and exact zero change for
  classification, perception, and reference state; the non-trainable old-policy
  snapshot must have the registered last-sync step and valid snapshot provenance
  (it is expected to differ from base after synchronization);
- every evaluation artifact must record generator checkpoint, selector reference
  checkpoint, selector source, token count/order/SHA, schedule, split, cache paths,
  selected/candidate/oracle rewards, PDM components, tail metrics, and completion;
- gates are fail-closed on identity mismatches, incomplete artifacts, NaN/Inf, token
  misalignment, selector source other than `reference`, or non-registered schedules.

Focused tests must cover runner rejection, base current/reference equivalence,
reference-selector detach and mode identity, frozen boundaries, gate thresholds,
artifact completeness, and the absence of imitation/GT losses.

## 4. Phase A: pilot on independent dev-select

Generate a fresh frozen-selector base artifact and evaluate `PILOT_U128` on the exact
3,072-token dev-select manifest, sequentially on one RTX 4090.

The pilot passes only if all conditions hold:

- selected raw-PDMS delta `>= +0.0015`;
- paired bootstrap CI95 lower bound `> 0`;
- collision, drivable, progress, and TTC deltas are each `>= 0`;
- worst selected-token delta is `> -0.5`;
- every artifact contains exactly 3,072 unique aligned tokens.

Candidate mean, oracle, regret, selector agreement, and bottom-tail statistics are
diagnostic only. A negative undeployed oracle token is not a hard stop.

Failure stops Stage 13 before any new training. Success locks the recipe permanently.

## 5. Phase B: fresh formal seed 0

Run a seed-0 U8 audit, then start a separate fresh seed-0 U128 job from `BASE_CKPT`.
The U8 checkpoint is not a resume source for the formal job.

Evaluate frozen-selector fixed-1024. The seed-0 gate requires:

- selected delta `>= +0.002` and paired CI95 lower bound `> 0`;
- collision, drivable, progress, and TTC deltas each `>= 0`;
- worst selected-token delta `> -0.5`;
- rolling generation KL finite and `<= 5e-4`;
- exact registered checkpoint/configuration/artifact identities.

Failure stops Stage 13 and no replacement seed is allowed.

## 6. Phase C: fresh seeds 1 and 2

After seed 0 passes, run independent U8 audits and fresh U128 training for seeds 1/2.
Training may use separate RTX 4090s concurrently; all evaluations run sequentially.

The three-seed fixed-1024 gate requires:

- every seed selected delta `> 0`;
- three-seed mean delta `>= +0.002`;
- seed-stratified CI95 lower bound `> 0`;
- at least two single-seed CI95 lower bounds `> 0`;
- mean collision/drivable/progress/TTC deltas each `>= 0`;
- every seed worst selected-token delta `> -0.5`;
- every rolling generation KL finite and `<= 5e-4`.

## 7. Phase D: untouched confirmation

Only after the three-seed fixed-1024 gate passes, evaluate dev-confirm in the exact
order base, seed 0, seed 1, seed 2.

The gate requires all three seed deltas positive, seed-stratified CI lower bound
`>= 0`, mean safety/progress component deltas nonnegative, no selected catastrophic
tail below `-0.5`, and exactly 1,024 unique aligned records per artifact.

Failure stops without loading, summarizing, or evaluating full-navtest candidates.

## 8. Phase E: one formal full-navtest

After dev-confirm passes, build/revalidate the locked 12,146-token manifest with SHA
`1a9c72551d5daa80242c30f713c1a41189a3bf3d063972e83e960707990dc72f`.
Resolve and record the real navtest feature and metric cache paths. Run base, seed 0,
seed 1, and seed 2 sequentially; each must complete `12,146/12,146`.

Scientific success requires:

- all three seed selected deltas positive;
- at least two single-seed paired CI95 lower bounds `> 0`;
- seed-stratified and token-clustered CI95 lower bounds `> 0`;
- mean collision, drivable, progress, and TTC deltas each `>= 0`;
- no artifact identity, token, cache, or completion mismatch.

Report absolute PDMS, delta versus base, delta versus the existing `+0.0013528957`
champion, all PDM components, and buckets for base `<0.5`, each safety failure, and
all-safety-pass scenes. Full-navtest is not used for model selection or reruns.

## 9. Stop discipline and prohibited actions

- Gates are authoritative only when the command exit code and output JSON agree.
- Exit `1` with a valid `passed=false` JSON is a scientific stop, not an invitation
  to change thresholds, seeds, LR, KL, or duration.
- Operational failures may rerun the exact registered command after artifact identity
  recovery; valid completed JSON/checkpoints must not be overwritten without a log.
- Do not include the pilot checkpoint in the formal three-seed aggregate.
- Do not replace failed seeds, inspect dev-confirm early, or run multiple full-navtest
  jobs concurrently.
- Do not claim selector learning: the paper attribution is GRPO generator improvement
  under a frozen base selector.

## 10. Formal execution checklist

- [x] Stage-13 hypothesis, method boundary, gates, and stop conditions registered
- [x] Branch, worktree, environment, input identities, caches, and RTX 4090s verified
- [x] Stage-13 runner/gate/tests implemented and focused verification passed
- [x] Base current/reference selector equivalence passed
- [x] Pilot base and U128 dev-select artifacts completed sequentially
- [x] Pilot dev-select gate executed: **FAILED; Stage 13 stopped before training**
- [ ] Fresh seed-0 U8 audit and independent U128 training passed
- [ ] Fresh seed-0 frozen-selector fixed-1024 gate passed
- [ ] Fresh seeds 1/2 audits and independent U128 training passed
- [ ] Three-seed frozen-selector fixed-1024 gate passed
- [ ] Dev-confirm remained unopened until authorized; formal gate passed
- [ ] Frozen-selector inference cost recorded
- [ ] Full-navtest base and three seeds completed sequentially, 12,146 each
- [ ] Scientific/new-champion/strong/stretch result classification recorded

## 11. Execution log

Append one timestamped block for every preflight, implementation verification,
evaluation, training job, gate, stop, and recovery decision. Record exact commands,
absolute paths, GPU identity, exit codes, checkpoint/config provenance, token count/SHA,
selector source, schedule/caches, selected/candidate/oracle metrics, CI, PDM components,
tail metrics, KL, gate failures, and the explicit CONTINUE/STOP decision.

### 2026-07-20 10:44:50 UTC — implementation and preflight

- Corrected the registered old-policy boundary after inspecting the real checkpoint:
  it is non-trainable but synchronized every 32 steps. The pilot U128 checkpoint has
  `_old_policy_last_sync_step=96`, as expected at checkpoint global step 128.
- Added `scripts/training/run_diffusiondrive_grpo_stage13.sh`,
  `scripts/training/check_grpo_stage13_checkpoint.py`,
  `scripts/evaluation/check_grpo_stage13_gate.py`, and
  `test_grpo_stage13_protocol.py`. Evaluation summaries now record the frozen
  reference checkpoint, feature/metric caches, tokens file, requested limit,
  completion, and failure count.
- Static checks passed: `bash -n`, Python compilation, and `git diff --check` all
  exited 0. Focused test command
  `/root/miniconda3/envs/navsim/bin/python -m pytest -q
  test_grpo_stage13_protocol.py test_grpo_stage9_training.py` passed `10/10`.
- Real pilot checkpoint audit exited 0 and wrote
  `artifacts/grpo_stage13/pilot_checkpoint_audit.json`: global step 128, one
  optimizer, one scheduler, 32 changed generation keys in each decoder layer,
  84/84 exact frozen-reference tensors, 84 valid old-policy tensors, last sync 96,
  and no failures.
- Base current/reference equivalence from the fresh 2026-07-20 run was revalidated:
  256 tokens, mode agreement 1.0, trajectory max error 0, PDMS difference 0,
  `passed=true`.
- All eight devices are NVIDIA GeForce RTX 4090 and idle (13–21 MiB used, 0%
  utilization). Dev feature cache exists (4.4 GiB); trainval metric cache exists
  (21 GiB). Dev-select contains exactly 3,072 unique ordered tokens with SHA
  `1d5e2c1ae12f8e5b65edfa9976166b7a24cb77df87fe67b154bff9431a491299`;
  dev-confirm remains unopened for evaluation and has registered SHA
  `c98144d252415f74ccdf041a6f610b9a9bedee6adbba91027a7bb48a2df5554d`.
- Decision: **CONTINUE** to the sequential pilot dev-select base and U128 evaluation.

### 2026-07-20 11:59:05 UTC — pilot dev-select and authoritative stop

- Ran the base and historical Stage-9 seed-0 U128 evaluations sequentially on GPU 0
  with the exact 3,072-token dev-select manifest, frozen base selector, `[8,0]` /
  125-step schedule, registered dev feature cache, and trainval metric cache. Both
  commands exited 0, produced 3,072/3,072 unique aligned records, reported zero
  failures, and matched token SHA
  `1d5e2c1ae12f8e5b65edfa9976166b7a24cb77df87fe67b154bff9431a491299`.
- Base selected PDMS was `0.7372025089037683`; pilot selected PDMS was
  `0.7386953106858224`. The authoritative paired selected delta was
  `+0.0014928017820542057`, with CI95
  `[+0.00045141016489651524, +0.0027213993717547656]`.
- Diagnostic deltas: candidate mean `+0.0009476864973597306`, oracle mean
  `+0.000008970499038696289`. Core component deltas were collision
  `+0.0013020833333333333`, drivable `+0.0003255208333333333`, progress
  `+0.0018870545800382388`, and TTC `+0.0003255208333333333`. Worst selected-token
  delta was `-0.2943311333656311`, safely above the registered `-0.5` boundary.
- Gate command wrote `artifacts/grpo_stage13/pilot/gate.json` and exited 1 with
  `passed=false`. The sole failure was `selected delta is < +0.0015`: the observed
  mean missed the preregistered threshold by `0.0000071982179457943`. Artifact
  identity, CI, all four components, and tail conditions passed.
- Per the registered fail-closed discipline, no U8 audit or fresh seed-0 U128 job was
  launched, thresholds were not rounded or changed, and dev-confirm remained unused.
  Process inspection found no residual training/evaluation jobs; all eight RTX 4090s
  were idle at 13–15 MiB and 0% utilization.
- Decision: **STOP STAGE 13**. This is a threshold stop, not evidence of a negative
  GRPO effect: the independent pilot estimate is positive and statistically
  significant, but it does not authorize the preregistered formal three-seed phase.
