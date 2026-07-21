# DiffusionDrive GRPO Stage 12 Handoff

Date prepared: 2026-07-19
Branch: `experiment-after-stage9`
Required starting commit: `0c45475` (`Implement Stage 10-11 generation GRPO validation`)
Stage-12 implementation commit: `24d6287` (`Add Stage 12 frozen-selector continuation protocol`)

## 0. New session: start here

This document is the complete operational handoff. A new session must read the
entire document before running anything, then continue from the first unchecked
item in Section 13. Do not infer progress from an old terminal, process name, or
partially written output file.

### Exact current state at handoff

- Stage 12 code and this protocol are implemented in the worktree.
- The focused Stage 9-12 regression suite passed: `53 passed`.
- Python compilation, Bash syntax checks, and `git diff --check` passed.
- The real Stage-10 U128 Lightning checkpoint passed resume preflight: global step
  `128`, one optimizer, one scheduler, adaptive-KL coefficient `0.1`, rolling KL
  `3.332438564e-05`, and `should_stop=false`.
- The manifest builder was checked against the historical navtest CSV. It found
  exactly `12,146` scenario tokens after excluding the CSV's `token=average` row,
  with SHA `1a9c72551d5daa80242c30f713c1a41189a3bf3d063972e83e960707990dc72f`.
- No formal Stage-12 diagnostic evaluation, U256 training, dev evaluation, or
  full-navtest job has been run. There are no Stage-12 results to interpret yet.
- Therefore tomorrow starts at **Phase A**, after re-running the environment,
  resume-preflight, and base-equivalence checks. It does not start at Phase B.

`0c45475` is the immutable committed Stage-10/11 experimental base. The Stage-12
implementation was committed separately as `24d6287` and pushed to
`origin/experiment-after-stage9`. First inspect `git status -sb`, `git log -2`, and
`git diff`; preserve unrelated untracked history and record the actual checked-out
commit in the run log. Do not reset or discard the worktree to make it look clean.

### Tomorrow's first actions, in order

1. Read this entire file, including all stop conditions and Section 11 recovery
   rules.
2. Run `git status -sb`, `git rev-parse HEAD`, and inspect the Stage-12 diff. Confirm
   the branch is `experiment-after-stage9`; preserve unrelated untracked history.
3. Export the locked variables in Section 2 and inspect available RTX 4090 GPUs.
4. Re-run the Stage-10 U128 checkpoint preflight and base selector equivalence in
   Section 2. A stale successful result from today does not replace these checks.
5. Run the four Stage-9 frozen-selector fixed-1024 evaluations in Section 3,
   sequentially, in the registered order U128, U512, U1024, U2048.
6. Run the diagnostic gate once. If it exits nonzero, record the failure and stop.
   Only a passing `diagnostic/gate.json` authorizes Phase B training.

Copy-paste prompt for the next session:

> Read `GRPO_STAGE12_FROZEN_SELECTOR_CONTINUATION_HANDOFF.md` completely. Inspect
> the branch, worktree, checkpoints, and existing artifacts without discarding
> changes. Do not alter any registered threshold or configuration. Re-run the
> environment, resume-preflight, and base-equivalence checks, then begin at the
> first unchecked formal-execution item. Execute phases in order and stop
> immediately at any failed gate. Record every command, artifact, metric, exit
> code, and continue/stop decision back into the handoff document.

### Stage-12 files and responsibilities

- `scripts/evaluation/check_grpo_stage12_gate.py`: the sole authority for all
  registered diagnostic, fixed-1024, dev, and full-navtest pass/fail decisions.
- `scripts/training/check_grpo_stage12_resume.py`: fail-closed inspection of full
  Lightning trainer state, optimizer/scheduler state, adaptive-KL state, and
  hard-stop status before a continuation.
- `scripts/training/run_diffusiondrive_grpo_stage12.sh`: registered training entry
  point; enforces RTX 4090 hardware, allowed seed/target combinations, and exact
  resume provenance.
- `scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh`: registered
  evaluation entry point; always delegates with selector source `reference`.
- `scripts/evaluation/build_grpo_stage12_navtest_manifest.py`: creates the locked
  12,146-token navtest manifest and excludes the aggregate `average` row.
- `test_grpo_stage12_protocol.py`: focused tests for gates, checkpoint preflight,
  reference-selector enforcement, and navtest token handling.
- `scripts/evaluation/evaluate_grpo_schedule.py`: extended to support `test` split;
  it writes checkpoint, selector source, schedule, token SHA, selected/candidate/
  oracle rewards, and safety components into artifacts.
- `scripts/evaluation/run_diffusiondrive_grpo_stage11_eval.sh`: extended to pass
  `train|val|test`; Stage 12 wraps this rather than duplicating evaluator logic.

### Actions that are explicitly prohibited

- Do not run `git add .`; the worktree contains many unrelated historical plans,
  artifacts, and checkpoints. Stage only an explicit reviewed file list.
- Do not start any U256 training until the Phase-A diagnostic gate passes.
- Do not edit gate thresholds, seeds, LR, KL settings, rollout/scheduler settings,
  batch size, precision, token sets, or evaluation order after seeing results.
- Do not select any Stage-9 checkpoint as the formal model. Stage-9 U128/U512/
  U1024/U2048 exist only to diagnose whether longer training has headroom.
- Do not add loss terms, reward shaping, selector training, trust projection,
  priority sampling, extra LR/KL sweeps, U384, U512, or another fallback.
- Do not train a seed with DDP, do not use H100 checkpoints in the formal set, and
  do not mix hardware provenance. Formal jobs are single RTX 4090, batch 2, FP32.
- Do not replace a failed seed, skip a gate, inspect dev-confirm early, or inspect
  navtest before dev-confirm passes.
- Do not run full-navtest candidates in parallel. Run base, seed 0, seed 1, seed 2
  sequentially, and require 12,146 successful scenarios in each artifact.
- Do not overwrite a completed JSON or checkpoint until Section 11 has established
  that it is invalid or mismatched and the reason has been recorded.

## 1. Objective and registered hypothesis

Stage 11 established a statistically positive frozen-selector result on fixed-1024:

- selected PDMS delta: `+0.0026683249743655324`;
- 95% paired CI: `[+0.0004596310332999565, +0.0055847586205345565]`;
- candidate mean delta: `+0.0013691200183529872`;
- oracle delta: `+0.0008774464367888868`;
- collision/drivable/TTC: `0 / +0.0029296875 / +0.001953125`;
- worst token: `-0.03775966167449951`;
- rolling KL: `3.3324e-5`.

It failed only because selected delta was below the registered `+0.003` gate. Stage 12
does not reinterpret that failure. It registers a new hypothesis: the Stage-10 U128
route may be under-trained because rolling KL remains well below the `1e-4` target;
one continuous U128-to-U256 continuation may add candidate utility while the frozen
base selector prevents current-selector tail failures.

The only formal variable is total optimizer updates, `128 -> 256`. Do not add or tune
losses, reward shaping, selector training, projection, priority sampling, LR, KL
parameters, rollout count, mode weighting, or model structure. There is no U384/U512
fallback.

## 2. Locked provenance

```bash
export ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
export PY=/root/miniconda3/envs/navsim/bin/python
export BASE_CKPT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model
export STAGE10_U128_CKPT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage10_layer0_01_u128/2026.07.19.04.43.02/lightning_logs/version_0/checkpoints/grpo-step-128.ckpt
export FIXED1024_BASE=$ROOT/artifacts/grpo_stage11/backtrack_base_reference_1024.json
export STAGE10_U128_REF=$ROOT/artifacts/grpo_stage11/backtrack_candidate_reference_1024.json
export FIXED1024_TOKENS=$FIXED1024_BASE
export STAGE12_ARTIFACTS=$ROOT/artifacts/grpo_stage12
mkdir -p "$STAGE12_ARTIFACTS"
cd "$ROOT"
```

Required fixed-1024 token SHA:
`fc2f4c607de003688f5fbfc1ab5f97568b3c334e4a34c3cb4ae2fe3de994df25`.

All formal jobs use one RTX 4090, batch size 2, FP32. A seed is never trained with
DDP. Seeds 1 and 2 may occupy separate 4090s only after seed 0 passes. Do not mix
4090 and H100 checkpoints in the three-seed result.

Before any work:

```bash
git status -sb
git rev-parse HEAD
nvidia-smi --query-gpu=index,name,memory.free --format=csv
$PY scripts/training/check_grpo_stage12_resume.py \
  --checkpoint "$STAGE10_U128_CKPT" --expected-global-step 128
bash scripts/evaluation/run_diffusiondrive_grpo_stage11_base_equivalence.sh 256 0
```

The base-equivalence gate must report mode agreement `1.0`, trajectory max error
`<=1e-6`, and selected PDMS difference exactly `0`.

## 3. Phase A: zero-training checkpoint curve

These Stage-9 checkpoints are diagnostic only and can never become the formal model:

```bash
export S9_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage9_generation_epoch1_seed0/2026.07.18.15.58.29/lightning_logs/version_0/checkpoints
export S9_U128=$S9_DIR/grpo-step-128.ckpt
export S9_U512=$S9_DIR/grpo-step-512.ckpt
export S9_U1024=$S9_DIR/grpo-step-1024.ckpt
export S9_U2048=$S9_DIR/grpo-step-2048.ckpt
mkdir -p "$STAGE12_ARTIFACTS/diagnostic"
```

Run the following four commands sequentially on GPU 0:

```bash
bash scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh "$S9_U128" \
  "$STAGE12_ARTIFACTS/diagnostic/stage9_u128_reference_1024.json" \
  "$FIXED1024_TOKENS" 1024 "$FIXED1024_BASE" default default val 0
bash scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh "$S9_U512" \
  "$STAGE12_ARTIFACTS/diagnostic/stage9_u512_reference_1024.json" \
  "$FIXED1024_TOKENS" 1024 "$FIXED1024_BASE" default default val 0
bash scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh "$S9_U1024" \
  "$STAGE12_ARTIFACTS/diagnostic/stage9_u1024_reference_1024.json" \
  "$FIXED1024_TOKENS" 1024 "$FIXED1024_BASE" default default val 0
bash scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh "$S9_U2048" \
  "$STAGE12_ARTIFACTS/diagnostic/stage9_u2048_reference_1024.json" \
  "$FIXED1024_TOKENS" 1024 "$FIXED1024_BASE" default default val 0
```

Apply the registered diagnostic gate:

```bash
$PY scripts/evaluation/check_grpo_stage12_gate.py --gate diagnostic \
  --baseline-artifacts "$FIXED1024_BASE" \
  --stage10-u128-artifact "$STAGE10_U128_REF" \
  --candidate-artifacts \
    "$STAGE12_ARTIFACTS/diagnostic/stage9_u128_reference_1024.json" \
    "$STAGE12_ARTIFACTS/diagnostic/stage9_u512_reference_1024.json" \
    "$STAGE12_ARTIFACTS/diagnostic/stage9_u1024_reference_1024.json" \
    "$STAGE12_ARTIFACTS/diagnostic/stage9_u2048_reference_1024.json" \
  --candidate-steps 128 512 1024 2048 \
  --output "$STAGE12_ARTIFACTS/diagnostic/gate.json"
```

At least one U512+ checkpoint must pass every diagnostic condition: selected delta
`>=+0.003`, CI lower bound `>0`, at least `+0.0005` above Stage-10 U128, candidate
mean no lower than Stage-10 U128, oracle nonnegative, safety components no worse
than `-0.001`, and worst token `>-0.5`. If the command exits 1, stop Stage 12. Do
not train U256 and do not relax the diagnostic.

## 4. Phase B: seed-0 U128-to-U256 continuation

Only run this phase after `diagnostic/gate.json` passes.

```bash
bash scripts/training/run_diffusiondrive_grpo_stage12.sh \
  0 256 stage12_layer0_01_u256_seed0 "$STAGE10_U128_CKPT" 0
```

This restores the full trainer state. It must not initialize from U128 weights as a
new job. The runner verifies global step 128, optimizer state, LR scheduler state,
adaptive-KL buffers, and that the checkpoint is not already hard-stopped.

After training, set `SEED0_U256` to the generated `grpo-step-256.ckpt` and run:

```bash
export SEED0_U256=/absolute/path/to/stage12_seed0/grpo-step-256.ckpt
$PY scripts/training/check_grpo_stage12_resume.py \
  --checkpoint "$SEED0_U256" --expected-global-step 256
```

Record `adaptive_kl_rolling_mean` from this output as `SEED0_KL`. If the checkpoint
is a `kl-stop-step-*.ckpt`, `adaptive_kl_should_stop=true`, or KL exceeds `2.5e-4`,
the seed fails immediately.

Run fixed-256 only as a runtime smoke, then fixed-1024:

```bash
bash scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh "$SEED0_U256" \
  "$STAGE12_ARTIFACTS/seed0_u256_reference_256.json" \
  "$FIXED1024_TOKENS" 256 \
  "$ROOT/artifacts/grpo_stage11/base_equivalence/base_reference_256.json" \
  default default val 0
bash scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh "$SEED0_U256" \
  "$STAGE12_ARTIFACTS/seed0_u256_reference_1024.json" \
  "$FIXED1024_TOKENS" 1024 "$FIXED1024_BASE" default default val 0
$PY scripts/evaluation/check_grpo_stage12_gate.py --gate seed0-fixed1024 \
  --baseline-artifacts "$FIXED1024_BASE" \
  --stage10-u128-artifact "$STAGE10_U128_REF" \
  --candidate-artifacts "$STAGE12_ARTIFACTS/seed0_u256_reference_1024.json" \
  --generation-kl-rolling-means "$SEED0_KL" \
  --output "$STAGE12_ARTIFACTS/seed0_fixed1024_gate.json"
```

The gate requires `>=+0.003` versus base, CI lower `>0`, `>=+0.0005` versus U128,
U256-vs-U128 CI lower `>=0`, nonnegative candidate/oracle, candidate mean no lower
than U128, safety thresholds, worst token `>-0.5`, and KL `<=2.5e-4`. Failure ends
Stage 12. There is no second duration or checkpoint selection.

## 5. Phase C: independent seeds

Only after seed 0 passes, run U8 audits for seeds 1 and 2. These may run concurrently
on GPUs 1 and 2 in separate shells:

```bash
bash scripts/training/run_diffusiondrive_grpo_stage12.sh \
  1 8 stage12_layer0_01_u8_seed1 none 1
bash scripts/training/run_diffusiondrive_grpo_stage12.sh \
  2 8 stage12_layer0_01_u8_seed2 none 2
```

Both current decoder layers must have finite nonzero gradients and weight changes.
Reference, old policy, perception, and classification changes must be exactly zero.
Any failure stops that seed; do not replace it.

Set `SEED1_U8` and `SEED2_U8` to the audited checkpoints, then continue each same
trainer state to U256:

```bash
bash scripts/training/run_diffusiondrive_grpo_stage12.sh \
  1 256 stage12_layer0_01_u256_seed1 "$SEED1_U8" 1
bash scripts/training/run_diffusiondrive_grpo_stage12.sh \
  2 256 stage12_layer0_01_u256_seed2 "$SEED2_U8" 2
```

Extract each rolling KL with the resume checker. Evaluate the three U256 checkpoints
sequentially on fixed-1024 with the Stage-12 evaluation wrapper. Then run:

```bash
$PY scripts/evaluation/check_grpo_stage12_gate.py --gate three-seed-fixed1024 \
  --baseline-artifacts "$FIXED1024_BASE" \
  --candidate-artifacts \
    "$STAGE12_ARTIFACTS/seed0_u256_reference_1024.json" \
    "$STAGE12_ARTIFACTS/seed1_u256_reference_1024.json" \
    "$STAGE12_ARTIFACTS/seed2_u256_reference_1024.json" \
  --generation-kl-rolling-means "$SEED0_KL" "$SEED1_KL" "$SEED2_KL" \
  --output "$STAGE12_ARTIFACTS/three_seed_fixed1024_gate.json"
```

All seed deltas must be positive, mean delta `>=+0.003`, seed-stratified CI lower
`>0`, candidate/oracle means nonnegative, safety component means nonnegative, and
each KL finite and `<=2.5e-4`.

## 6. Phase D: dev-select and dev-confirm

Use these locked manifests and artifact names:

```bash
export DEV_SELECT=$ROOT/artifacts/grpo_stage0/dev_select3072_manifest.json
export DEV_CONFIRM=$ROOT/artifacts/grpo_stage0/dev_confirm1024_manifest.json
export DEV_SELECT_BASE=$STAGE12_ARTIFACTS/dev_select_base_reference_3072.json
export DEV_CONFIRM_BASE=$STAGE12_ARTIFACTS/dev_confirm_base_reference_1024.json
```

Only after the three-seed fixed-1024 gate passes, evaluate dev-select in the exact
order base, seed 0, seed 1, seed 2:

```bash
bash scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh "$BASE_CKPT" \
  "$DEV_SELECT_BASE" "$DEV_SELECT" 3072 none default default val 0
bash scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh "$SEED0_U256" \
  "$STAGE12_ARTIFACTS/dev_select_seed0_reference_3072.json" \
  "$DEV_SELECT" 3072 "$DEV_SELECT_BASE" default default val 0
bash scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh "$SEED1_U256" \
  "$STAGE12_ARTIFACTS/dev_select_seed1_reference_3072.json" \
  "$DEV_SELECT" 3072 "$DEV_SELECT_BASE" default default val 0
bash scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh "$SEED2_U256" \
  "$STAGE12_ARTIFACTS/dev_select_seed2_reference_3072.json" \
  "$DEV_SELECT" 3072 "$DEV_SELECT_BASE" default default val 0

$PY scripts/evaluation/check_grpo_stage12_gate.py --gate dev-select \
  --baseline-artifacts "$DEV_SELECT_BASE" \
  --candidate-artifacts \
    "$STAGE12_ARTIFACTS/dev_select_seed0_reference_3072.json" \
    "$STAGE12_ARTIFACTS/dev_select_seed1_reference_3072.json" \
    "$STAGE12_ARTIFACTS/dev_select_seed2_reference_3072.json" \
  --output "$STAGE12_ARTIFACTS/dev_select_gate.json"
```

Dev-select requires 3072 tokens per seed, three-seed mean `>=+0.003`,
seed-stratified CI lower bound `>0`, every seed positive, and mean
collision/drivable/TTC deltas nonnegative. If it fails, stop and do not open,
summarize, or evaluate dev-confirm.

Only after dev-select passes, run dev-confirm in the same order:

```bash
bash scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh "$BASE_CKPT" \
  "$DEV_CONFIRM_BASE" "$DEV_CONFIRM" 1024 none default default val 0
bash scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh "$SEED0_U256" \
  "$STAGE12_ARTIFACTS/dev_confirm_seed0_reference_1024.json" \
  "$DEV_CONFIRM" 1024 "$DEV_CONFIRM_BASE" default default val 0
bash scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh "$SEED1_U256" \
  "$STAGE12_ARTIFACTS/dev_confirm_seed1_reference_1024.json" \
  "$DEV_CONFIRM" 1024 "$DEV_CONFIRM_BASE" default default val 0
bash scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh "$SEED2_U256" \
  "$STAGE12_ARTIFACTS/dev_confirm_seed2_reference_1024.json" \
  "$DEV_CONFIRM" 1024 "$DEV_CONFIRM_BASE" default default val 0

$PY scripts/evaluation/check_grpo_stage12_gate.py --gate dev-confirm \
  --baseline-artifacts "$DEV_CONFIRM_BASE" \
  --candidate-artifacts \
    "$STAGE12_ARTIFACTS/dev_confirm_seed0_reference_1024.json" \
    "$STAGE12_ARTIFACTS/dev_confirm_seed1_reference_1024.json" \
    "$STAGE12_ARTIFACTS/dev_confirm_seed2_reference_1024.json" \
  --output "$STAGE12_ARTIFACTS/dev_confirm_gate.json"
```

Dev-confirm requires 1024 tokens per seed, three-seed mean `>0`,
seed-stratified CI lower bound `>=0`, every seed positive, and mean
collision/drivable/TTC deltas nonnegative. If it fails, stop without loading or
evaluating navtest.

## 7. Phase E: full navtest

Only after dev-confirm passes, lock the historical 12,146-token order:

```bash
$PY scripts/evaluation/build_grpo_stage12_navtest_manifest.py \
  --score-csv /inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/paired_full_navtest_base_20260715/2026.07.15.09.04.23/2026.07.15.09.24.58.csv \
  --output "$STAGE12_ARTIFACTS/navtest_manifest_12146.json"
```

The manifest command must report 12,146 tokens and SHA
`1a9c72551d5daa80242c30f713c1a41189a3bf3d063972e83e960707990dc72f`.

Resolve and record the real feature-cache path before continuing. The placeholder
below is intentionally invalid; do not launch until it has been replaced:

```bash
export NAVTEST_TOKENS=$STAGE12_ARTIFACTS/navtest_manifest_12146.json
export NAVTEST_BASE=$STAGE12_ARTIFACTS/navtest_base_reference_12146.json
export NAVTEST_FEATURE_CACHE=/absolute/path/to/navtest_feature_cache
export NAVTEST_METRIC_CACHE=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/metric_cache
test -d "$NAVTEST_FEATURE_CACHE"
test -d "$NAVTEST_METRIC_CACHE"
```

Generate a new frozen-selector base artifact unless every reuse condition in
Section 11 is documented. Then run base and all candidate seeds sequentially; wait
for each command to exit before starting the next:

```bash
bash scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh "$BASE_CKPT" \
  "$NAVTEST_BASE" "$NAVTEST_TOKENS" 12146 none \
  "$NAVTEST_FEATURE_CACHE" "$NAVTEST_METRIC_CACHE" test 0
bash scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh "$SEED0_U256" \
  "$STAGE12_ARTIFACTS/navtest_seed0_reference_12146.json" \
  "$NAVTEST_TOKENS" 12146 "$NAVTEST_BASE" \
  "$NAVTEST_FEATURE_CACHE" "$NAVTEST_METRIC_CACHE" test 0
bash scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh "$SEED1_U256" \
  "$STAGE12_ARTIFACTS/navtest_seed1_reference_12146.json" \
  "$NAVTEST_TOKENS" 12146 "$NAVTEST_BASE" \
  "$NAVTEST_FEATURE_CACHE" "$NAVTEST_METRIC_CACHE" test 0
bash scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh "$SEED2_U256" \
  "$STAGE12_ARTIFACTS/navtest_seed2_reference_12146.json" \
  "$NAVTEST_TOKENS" 12146 "$NAVTEST_BASE" \
  "$NAVTEST_FEATURE_CACHE" "$NAVTEST_METRIC_CACHE" test 0

$PY scripts/evaluation/check_grpo_stage12_gate.py --gate full-navtest \
  --baseline-artifacts "$NAVTEST_BASE" \
  --candidate-artifacts \
    "$STAGE12_ARTIFACTS/navtest_seed0_reference_12146.json" \
    "$STAGE12_ARTIFACTS/navtest_seed1_reference_12146.json" \
    "$STAGE12_ARTIFACTS/navtest_seed2_reference_12146.json" \
  --output "$STAGE12_ARTIFACTS/full_navtest_gate.json"
```

The wrapper arguments above lock limit `12146`, log split `test`, frozen reference
selector, batch 2, rollout schedule `8 -> 0`, 125 scheduler inference steps, and
paired deterministic evaluation. Never run multiple full jobs concurrently.

Apply `--gate full-navtest` to one base artifact and the three candidate artifacts.
Success requires mean absolute PDMS delta `>=+0.005`, all seeds positive, at least
two single-seed CI lower bounds `>0`, seed-stratified and token-clustered CI lower
bounds `>0`, collision/drivable/progress/TTC means nonnegative, and exactly
`12,146/12,146` successful records for every artifact.

A statistically significant result below `+0.005` may be recorded as the new
champion, but Stage 12 remains failed.

## 8. Verification before execution

Run the complete focused suite before Phase A if code has changed since handoff:

```bash
$PY -m pytest -q \
  test_grpo_stage9_training.py \
  test_grpo_stage10_training.py \
  test_grpo_stage11_selector.py \
  test_grpo_stage12_protocol.py \
  test_generation_grpo_objective.py \
  test_diffusion_grpo_probability.py
$PY -m py_compile \
  scripts/evaluation/check_grpo_stage12_gate.py \
  scripts/evaluation/build_grpo_stage12_navtest_manifest.py \
  scripts/training/check_grpo_stage12_resume.py \
  test_grpo_stage12_protocol.py
bash -n \
  scripts/evaluation/run_diffusiondrive_grpo_stage11_eval.sh \
  scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh \
  scripts/training/run_diffusiondrive_grpo_stage12.sh
git diff --check
find scripts -name '*.rej' -o -name '*.orig'
```

Expected focused regression result at handoff is `53 passed`; the syntax, compile,
and diff checks produce no errors, and the final `find` produces no output. A later
test-count increase is acceptable if every test passes. A decrease or skipped test
must be explained before execution.

## 9. Verified implementation evidence

These checks were completed while preparing Stage 12. They prove the implementation
is runnable, but they do not authorize skipping tomorrow's formal preflight or any
experimental gate:

- [x] Stage-12 gate, resume checker, training runner, evaluation wrapper, manifest
  builder, protocol tests, and handoff implemented.
- [x] Focused Stage 9-12 regression suite passed: `53 passed`.
- [x] Python compilation, Bash syntax, and `git diff --check` passed.
- [x] Real Stage-10 U128 checkpoint inspected successfully at global step 128 with
  optimizer, scheduler, and live adaptive-KL state present.
- [x] Historical navtest CSV parsed to 12,146 unique non-average tokens with the
  registered SHA.
- [x] Runner rejection checks covered invalid seed, invalid seed-0 target, and
  invalid seed-1/2 U256 resume provenance.
- [x] No `.rej` or `.orig` files remained after implementation.
- [x] Confirmed that no formal Stage-12 evaluation or training result exists yet.

## 10. Exit codes and stop discipline

A gate is passed only when the gate command exits `0` and its output JSON contains
`"passed": true`. Exit `1` with a valid gate JSON is a scientific gate failure:
record all failure reasons and stop Stage 12 at once. Do not tune or rerun for a
different random outcome.

An exception, missing file, invalid argument, unavailable GPU/cache, interrupted
process, corrupt JSON, or exit `2` is an operational failure, not evidence for or
against the hypothesis. Diagnose the infrastructure issue, preserve the partial
logs, and rerun the exact registered command only after the input identity checks
below pass. Operational recovery never permits a threshold or configuration change.

At every phase boundary, read the gate JSON directly. Do not rely on a terminal
summary or on whether a checkpoint filename exists. Update Sections 12 and 13 before
starting the next phase.

## 11. Interruption and artifact recovery

After a disconnect, first run `ps` and `nvidia-smi` to determine whether the job
is still active. Do not launch a duplicate job. If it is active, reconnect to its
log/session or wait for it to exit. If it is not active, inspect outputs before
deciding whether anything needs to run again.

A completed evaluation JSON may be reused only when all of the following match the
registered command:

- exact resolved checkpoint path, and for a candidate the intended seed/step;
- exact token count and token-set SHA, in the same order as its paired baseline;
- `selector_logits_source=reference`;
- exact schedule `8 -> 0`, 125 inference steps, deterministic-noise protocol, and
  the same evaluator code/commit;
- exact log split, feature cache, metric cache, baseline artifact, and successful
  record count required by that phase.

The gate script checks token alignment, SHA, count, and selector source, but the
operator must still audit checkpoint path, schedule, caches, code revision, and
job completion. A JSON that merely exists is not automatically complete. Validate
that it parses, that `summary.num_tokens == len(records)`, that every token is
unique, and that its summary matches the command recorded in Section 12.

Never rerun or overwrite a valid completed artifact without first recording why it
cannot be reused. Preserve corrupt or partial outputs under a clearly marked path
before retrying the same command. Never merge records from two attempts.

Training continuation is allowed only from a full Lightning `.ckpt`, never an
`eval_model`, extracted state dict, or manually merged weights. Before resuming,
run `check_grpo_stage12_resume.py` with the exact expected global step. It must
report one optimizer, one scheduler, complete finite adaptive-KL state, and
`adaptive_kl_should_stop=false`. For seed 0 the only resume source is the registered
Stage-10 U128 checkpoint. For seeds 1/2, U256 may resume only from that same seed's
audited U8 checkpoint. A `kl-stop-step-*.ckpt` is a failed seed and must not resume.

A completed gate JSON may be reused only after every input artifact passes the same
identity audit and the gate code/thresholds are unchanged. If any identity field is
missing or inconsistent, regenerate one paired base artifact and all affected
candidate artifacts using the locked command; do not combine old and new pairs.

## 12. Result recording template

Append one block below for every preflight, evaluation, training job, and gate.
Record absolute paths and exact floating-point values rather than rounded prose.

```markdown
### YYYY-MM-DD HH:MM UTC - <phase/job>

- Git branch / commit / dirty diff: ...
- Exact command: `...`
- GPU index / model: ...
- Exit code and completion state: ...
- Input checkpoint: ...
- Resume checkpoint and expected global step: ...
- Output checkpoint or artifact: ...
- Baseline artifact: ...
- Token count / token-set SHA / log split: ...
- Selector source / schedule / caches: ...
- Selected PDMS delta: ...
- Candidate-mean delta / oracle delta: ...
- Paired or seed-stratified CI95: [...]
- Token-clustered CI95, when required: [...]
- Collision / drivable / progress / TTC deltas: ...
- Safety-pass bucket delta / worst-token delta: ...
- Adaptive-KL coefficient / rolling mean / hard-stop: ...
- Gate JSON / gate exit code / failure reasons: ...
- Decision: CONTINUE to <next phase> | STOP Stage 12
- Recovery notes or reused-artifact identity proof: ...
```

For diagnostic, also record all four checkpoint rows, each U128-relative delta, and
the exact `eligible_stage9_steps`. For multi-seed gates, record each seed delta and
single-seed CI in addition to the aggregate. A significant navtest result below
`+0.005` must be labeled `new champion candidate; Stage 12 engineering target
failed`, not passed.

## 13. Formal execution checklist

The unchecked state below is intentional: formal execution has not started.

- [x] Re-read this entire handoff in the new session
- [x] Branch, commit, worktree diff, environment, and RTX 4090 availability verified
- [x] Stage-10 U128 resume preflight re-run and recorded
- [x] Base selector equivalence re-run and passed
- [x] Stage-9 U128/U512/U1024/U2048 diagnostic artifacts completed sequentially
- [ ] Diagnostic gate passed and explicitly authorized U256
  - Formal run completed 2026-07-20: gate exited `1`, `passed=false`, and Stage 12 stopped; U256 was not authorized.
- [ ] Seed-0 U128-to-U256 continuation completed without KL hard-stop
- [ ] Seed-0 fixed-256 runtime smoke completed
- [ ] Seed-0 fixed-1024 gate passed
- [ ] Seeds 1/2 U8 gradient, weight-change, and freeze audits passed
- [ ] Seeds 1/2 resumed from their own U8 checkpoints to U256 without hard-stop
- [ ] Three-seed fixed-1024 gate passed
- [ ] Dev-select base and three candidate artifacts completed; gate passed
- [ ] Dev-confirm was not inspected early; base/candidates completed; gate passed
- [ ] Navtest manifest and explicit cache paths revalidated
- [ ] Full-navtest base, seed 0, seed 1, seed 2 completed sequentially, 12,146 each
- [ ] Full-navtest gate output and final `+0.005` engineering decision recorded

## 14. Execution log

No formal Stage-12 entries at handoff. Append Section-12 blocks here during the run.

### 2026-07-20 08:37 UTC - new-session provenance and environment preflight

- Git branch / commit / dirty diff: `experiment-after-stage9` / `db8ed812567d2754d57c8faa00d60c9fec9ce561`; no tracked worktree diff before logging, historical untracked files preserved. HEAD is the pushed documentation-only follow-up to Stage-12 implementation commit `24d6287`.
- Exact command: `git status -sb; git rev-parse --abbrev-ref HEAD; git rev-parse HEAD; git log -2 --oneline; nvidia-smi --query-gpu=index,name,memory.free,memory.used,utilization.gpu --format=csv`
- GPU index / model: GPUs 0-7, all `NVIDIA GeForce RTX 4090`; GPU 0 had 48,619 MiB free and 0% utilization, GPUs 1-7 had 48,622-48,627 MiB free and 0% utilization.
- Exit code and completion state: Git and GPU checks exited `0`; no active Stage-12, GRPO training, or schedule-evaluation process was found. Two initial concurrent read-only commands hit `bwrap: Failed to make / slave: Permission denied`; both were rerun sequentially/outside the failed sandbox and completed.
- Decision: CONTINUE to Stage-10 U128 resume preflight.
- Recovery notes or reused-artifact identity proof: `artifacts/grpo_stage12` did not exist before this run. Locked base, Stage-10 U128, and all four Stage-9 diagnostic checkpoints exist. No code changed after `24d6287`; `db8ed81` changes only this handoff, so the handoff's passing 53-test implementation verification remains applicable.

### 2026-07-20 08:37 UTC - Stage-10 U128 resume preflight

- Git branch / commit / dirty diff: `experiment-after-stage9` / `db8ed812567d2754d57c8faa00d60c9fec9ce561`; only this formal execution log/checklist became a tracked worktree modification.
- Exact command: `/root/miniconda3/envs/navsim/bin/python scripts/training/check_grpo_stage12_resume.py --checkpoint /inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage10_layer0_01_u128/2026.07.19.04.43.02/lightning_logs/version_0/checkpoints/grpo-step-128.ckpt --expected-global-step 128`
- GPU index / model: CPU checkpoint inspection; formal evaluation GPU remains GPU 0 / `NVIDIA GeForce RTX 4090`.
- Exit code and completion state: exit `0`, `passed=true`.
- Input checkpoint: `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage10_layer0_01_u128/2026.07.19.04.43.02/lightning_logs/version_0/checkpoints/grpo-step-128.ckpt`
- Resume checkpoint and expected global step: same checkpoint; expected `128`, observed `128`.
- Adaptive-KL coefficient / rolling mean / hard-stop: `0.1` / `3.3324385640298715e-05` / `false`.
- Gate JSON / gate exit code / failure reasons: resume checker stdout; exit `0`; `optimizer_state_count=1`, `lr_scheduler_state_count=1`, failures `[]`.
- Decision: CONTINUE to fresh base selector equivalence.
- Recovery notes or reused-artifact identity proof: none; the formal preflight was rerun in this session as required.

### 2026-07-20 08:46 UTC - base selector equivalence

- Git branch / commit / dirty diff: `experiment-after-stage9` / `db8ed812567d2754d57c8faa00d60c9fec9ce561`; tracked diff is this execution log/checklist only; historical untracked files preserved.
- Exact command: `bash scripts/evaluation/run_diffusiondrive_grpo_stage11_base_equivalence.sh 256 0`
- GPU index / model: GPU 0 / `NVIDIA GeForce RTX 4090`.
- Exit code and completion state: initial sandbox launch exited `1` before running with `bwrap: Failed to make / slave: Permission denied`; exact command rerun outside the failed sandbox exited `0`; both evaluator jobs completed `256/256` and gate completed.
- Input checkpoint: `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model`
- Output checkpoint or artifact: `artifacts/grpo_stage11/base_equivalence/base_current_256.json`; `artifacts/grpo_stage11/base_equivalence/base_reference_256.json`.
- Baseline artifact: `artifacts/grpo_stage11/base_equivalence/base_current_256.json`.
- Token count / token-set SHA / log split: both artifacts `256`, `d3c9f56e66c13bead6950141acc3a6bf44cafc98a27c7b49bd8126cc405e6113`, `val`; direct audit found `summary.num_tokens == len(records) == unique_tokens == 256` in both.
- Selector source / schedule / caches: baseline `current`, candidate `reference`; `8 -> 0`, 125 inference steps; default feature and metric caches.
- Selected PDMS delta: `0.0`.
- Gate JSON / gate exit code / failure reasons: `artifacts/grpo_stage11/base_equivalence/gate_256.json`; exit `0`; `passed=true`, mode agreement `1.0`, trajectory max error `0.0`, failures `[]`.
- Decision: CONTINUE to Phase-A Stage-9 U128 diagnostic evaluation.
- Recovery notes or reused-artifact identity proof: the complete prior July-19 artifacts were preserved before the protocol-required fresh rerun under `artifacts/grpo_stage11/base_equivalence/pre_stage12_session_20260720_0837/`; no records were merged across attempts.

### 2026-07-20 08:53 UTC - Phase A Stage-9 U128 frozen-selector fixed-1024

- Git branch / commit / dirty diff: `experiment-after-stage9` / `db8ed812567d2754d57c8faa00d60c9fec9ce561`; tracked diff is this execution log/checklist only.
- Exact command: `bash scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh /inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage9_generation_epoch1_seed0/2026.07.18.15.58.29/lightning_logs/version_0/checkpoints/grpo-step-128.ckpt /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage12/diagnostic/stage9_u128_reference_1024.json /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage11/backtrack_base_reference_1024.json 1024 /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage11/backtrack_base_reference_1024.json default default val 0`
- GPU index / model: GPU 0 / `NVIDIA GeForce RTX 4090`.
- Exit code and completion state: initial sandbox launch exited `1` before running due the known `bwrap` mount failure; exact command rerun outside that sandbox exited `0`, completed `1024/1024`.
- Input checkpoint: `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage9_generation_epoch1_seed0/2026.07.18.15.58.29/lightning_logs/version_0/checkpoints/grpo-step-128.ckpt`
- Output checkpoint or artifact: `/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage12/diagnostic/stage9_u128_reference_1024.json`
- Baseline artifact: `/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage11/backtrack_base_reference_1024.json`
- Token count / token-set SHA / log split: `1024`, `fc2f4c607de003688f5fbfc1ab5f97568b3c334e4a34c3cb4ae2fe3de994df25`, `val`; direct audit found `summary.num_tokens == len(records) == unique_tokens == 1024`.
- Selector source / schedule / caches: `reference`; `8 -> 0`, 125 inference steps; default feature and metric caches.
- Selected PDMS delta: `0.0030900143610779196`.
- Candidate-mean delta / oracle delta: `0.0017440853662265` / `0.0004205663572065532`.
- Paired or seed-stratified CI95: `[0.0007709306919423399, 0.00612509441707516]`.
- Collision / drivable / progress / TTC deltas: `0.0 / 0.001953125 / 0.0014444317876041168 / 0.0048828125`.
- Safety-pass bucket delta / worst-token delta: `-6.7307424228803244e-06` / `-0.057477355003356934`.
- Gate JSON / gate exit code / failure reasons: deferred until all four registered diagnostic checkpoints complete.
- Decision: CONTINUE to Phase-A Stage-9 U512 diagnostic evaluation.
- Recovery notes or reused-artifact identity proof: newly generated artifact; no prior Stage-12 artifact existed and no records were reused or merged.

### 2026-07-20 08:58 UTC - Phase A Stage-9 U512 frozen-selector fixed-1024

- Git branch / commit / dirty diff: `experiment-after-stage9` / `db8ed812567d2754d57c8faa00d60c9fec9ce561`; tracked diff is this execution log/checklist only.
- Exact command: `bash scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh /inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage9_generation_epoch1_seed0/2026.07.18.15.58.29/lightning_logs/version_0/checkpoints/grpo-step-512.ckpt /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage12/diagnostic/stage9_u512_reference_1024.json /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage11/backtrack_base_reference_1024.json 1024 /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage11/backtrack_base_reference_1024.json default default val 0`
- GPU index / model: GPU 0 / `NVIDIA GeForce RTX 4090`.
- Exit code and completion state: exit `0`, completed `1024/1024`.
- Input checkpoint: `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage9_generation_epoch1_seed0/2026.07.18.15.58.29/lightning_logs/version_0/checkpoints/grpo-step-512.ckpt`
- Output checkpoint or artifact: `/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage12/diagnostic/stage9_u512_reference_1024.json`
- Baseline artifact: `/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage11/backtrack_base_reference_1024.json`
- Token count / token-set SHA / log split: `1024`, `fc2f4c607de003688f5fbfc1ab5f97568b3c334e4a34c3cb4ae2fe3de994df25`, `val`; direct audit found `summary.num_tokens == len(records) == unique_tokens == 1024`.
- Selector source / schedule / caches: `reference`; `8 -> 0`, 125 inference steps; default feature and metric caches.
- Selected PDMS delta: `0.0037984511873219162`.
- Candidate-mean delta / oracle delta: `0.0037482837396964896` / `-0.00033859445829875767`.
- Paired or seed-stratified CI95: `[-0.0014479406512691638, 0.009217475495825056]`.
- Collision / drivable / progress / TTC deltas: `-0.0009765625 / 0.00390625 / 0.0014768991404707776 / 0.00390625`.
- Safety-pass bucket delta / worst-token delta: `-0.00274491230998419` / `-0.8505653142929077`.
- Gate JSON / gate exit code / failure reasons: deferred until all four registered diagnostic checkpoints complete.
- Decision: CONTINUE to Phase-A Stage-9 U1024 diagnostic evaluation; no checkpoint is evaluated in isolation for the registered gate.
- Recovery notes or reused-artifact identity proof: newly generated complete artifact; no records were reused or merged.

### 2026-07-20 09:03 UTC - Phase A Stage-9 U1024 frozen-selector fixed-1024

- Git branch / commit / dirty diff: `experiment-after-stage9` / `db8ed812567d2754d57c8faa00d60c9fec9ce561`; tracked diff is this execution log/checklist only.
- Exact command: `bash scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh /inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage9_generation_epoch1_seed0/2026.07.18.15.58.29/lightning_logs/version_0/checkpoints/grpo-step-1024.ckpt /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage12/diagnostic/stage9_u1024_reference_1024.json /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage11/backtrack_base_reference_1024.json 1024 /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage11/backtrack_base_reference_1024.json default default val 0`
- GPU index / model: GPU 0 / `NVIDIA GeForce RTX 4090`.
- Exit code and completion state: exit `0`, completed `1024/1024`.
- Input checkpoint: `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage9_generation_epoch1_seed0/2026.07.18.15.58.29/lightning_logs/version_0/checkpoints/grpo-step-1024.ckpt`
- Output checkpoint or artifact: `/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage12/diagnostic/stage9_u1024_reference_1024.json`
- Baseline artifact: `/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage11/backtrack_base_reference_1024.json`
- Token count / token-set SHA / log split: `1024`, `fc2f4c607de003688f5fbfc1ab5f97568b3c334e4a34c3cb4ae2fe3de994df25`, `val`; direct audit found `summary.num_tokens == len(records) == unique_tokens == 1024`.
- Selector source / schedule / caches: `reference`; `8 -> 0`, 125 inference steps; default feature and metric caches.
- Selected PDMS delta: `0.003897509363014251`.
- Candidate-mean delta / oracle delta: `0.006558172317454591` / `-0.0004785839410033077`.
- Paired or seed-stratified CI95: `[-0.00262141449493356, 0.010576725391001672]`.
- Collision / drivable / progress / TTC deltas: `-0.001953125 / 0.0048828125 / 0.002133579962901422 / 0.00390625`.
- Safety-pass bucket delta / worst-token delta: `-0.005200242079728473` / `-0.8599309325218201`.
- Gate JSON / gate exit code / failure reasons: deferred until all four registered diagnostic checkpoints complete.
- Decision: CONTINUE to Phase-A Stage-9 U2048 diagnostic evaluation; no checkpoint is evaluated in isolation for the registered gate.
- Recovery notes or reused-artifact identity proof: newly generated complete artifact; no records were reused or merged.

### 2026-07-20 09:08 UTC - Phase A Stage-9 U2048 frozen-selector fixed-1024

- Git branch / commit / dirty diff: `experiment-after-stage9` / `db8ed812567d2754d57c8faa00d60c9fec9ce561`; tracked diff is this execution log/checklist only.
- Exact command: `bash scripts/evaluation/run_diffusiondrive_grpo_stage12_eval.sh /inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage9_generation_epoch1_seed0/2026.07.18.15.58.29/lightning_logs/version_0/checkpoints/grpo-step-2048.ckpt /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage12/diagnostic/stage9_u2048_reference_1024.json /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage11/backtrack_base_reference_1024.json 1024 /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage11/backtrack_base_reference_1024.json default default val 0`
- GPU index / model: GPU 0 / `NVIDIA GeForce RTX 4090`.
- Exit code and completion state: exit `0`, completed `1024/1024`.
- Input checkpoint: `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage9_generation_epoch1_seed0/2026.07.18.15.58.29/lightning_logs/version_0/checkpoints/grpo-step-2048.ckpt`
- Output checkpoint or artifact: `/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage12/diagnostic/stage9_u2048_reference_1024.json`
- Baseline artifact: `/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage11/backtrack_base_reference_1024.json`
- Token count / token-set SHA / log split: `1024`, `fc2f4c607de003688f5fbfc1ab5f97568b3c334e4a34c3cb4ae2fe3de994df25`, `val`; direct audit found `summary.num_tokens == len(records) == unique_tokens == 1024`. A final four-artifact audit confirmed the same count, SHA, selector source, and registered checkpoint identity for every diagnostic artifact.
- Selector source / schedule / caches: `reference`; `8 -> 0`, 125 inference steps; default feature and metric caches.
- Selected PDMS delta: `0.006810275051975623`.
- Candidate-mean delta / oracle delta: `0.008875055245880503` / `-0.0016392350080423057`.
- Paired or seed-stratified CI95: `[-0.002037906461919192, 0.015922316799697]`.
- Collision / drivable / progress / TTC deltas: `-0.0009765625 / 0.0078125 / 0.004839187000470702 / 0.00390625`.
- Safety-pass bucket delta / worst-token delta: `-0.00929519573671628` / `-1.0`.
- Gate JSON / gate exit code / failure reasons: deferred to the immediately following registered diagnostic gate.
- Decision: CONTINUE to Phase-A diagnostic gate only; U256 training remains unauthorized.
- Recovery notes or reused-artifact identity proof: newly generated complete artifact; all four evaluations ran sequentially in registered order U128, U512, U1024, U2048, with no records reused or merged.

### 2026-07-20 09:09 UTC - Phase A diagnostic gate

- Git branch / commit / dirty diff: `experiment-after-stage9` / `db8ed812567d2754d57c8faa00d60c9fec9ce561`; tracked diff is this execution log/checklist only; historical untracked files preserved.
- Exact command: `/root/miniconda3/envs/navsim/bin/python scripts/evaluation/check_grpo_stage12_gate.py --gate diagnostic --baseline-artifacts /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage11/backtrack_base_reference_1024.json --stage10-u128-artifact /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage11/backtrack_candidate_reference_1024.json --candidate-artifacts /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage12/diagnostic/stage9_u128_reference_1024.json /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage12/diagnostic/stage9_u512_reference_1024.json /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage12/diagnostic/stage9_u1024_reference_1024.json /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage12/diagnostic/stage9_u2048_reference_1024.json --candidate-steps 128 512 1024 2048 --output /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage12/diagnostic/gate.json`
- GPU index / model: CPU gate computation after four sequential GPU-0 / `NVIDIA GeForce RTX 4090` evaluations.
- Exit code and completion state: exit `1`; valid scientific gate failure, output JSON parsed successfully.
- Input checkpoint: Stage-9 U128/U512/U1024/U2048 artifacts from the four locked checkpoints; Stage-10 comparison artifact from the registered U128 checkpoint.
- Output checkpoint or artifact: `/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage12/diagnostic/gate.json`
- Baseline artifact: `/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/artifacts/grpo_stage11/backtrack_base_reference_1024.json`
- Token count / token-set SHA / log split: gate reports `1024`; all audited input artifacts have SHA `fc2f4c607de003688f5fbfc1ab5f97568b3c334e4a34c3cb4ae2fe3de994df25`, split `val`, and `1024/1024` unique aligned tokens.
- Selector source / schedule / caches: all candidates `reference`; `8 -> 0`, 125 inference steps; default feature and metric caches.
- Stage-10 U128 registered row: selected delta `0.0026683249743655324`, gate CI95 `[0.00047456770917051475, 0.005617414171138079]`, candidate delta `0.0013691200183529872`, oracle delta `0.0008774464367888868`, worst token `-0.03775966167449951`.
- Stage-9 U128 row: selected delta `0.0030900143610779196`, gate CI95 `[0.0008036552339035553, 0.006081308615830493]`, U128-relative delta `0.0004216893867123872`, candidate/oracle `0.0017440853662265 / 0.0004205663572065532`, worst token `-0.057477355003356934`; ineligible because it is below U512+ and improvement over Stage-10 U128 is `<+0.0005`.
- Stage-9 U512 row: selected delta `0.0037984511873219162`, gate CI95 `[-0.0013423811971733808, 0.009296068496041697]`, U128-relative delta `0.0011301262129563838`, candidate/oracle `0.0037482837396964896 / -0.00033859445829875767`, worst token `-0.8505653142929077`; ineligible because CI lower is not `>0`, oracle is negative, and worst token is not `>-0.5`.
- Stage-9 U1024 row: selected delta `0.003897509363014251`, gate CI95 `[-0.0027464356608106755, 0.010519266568007878]`, U128-relative delta `0.0012291843886487186`, candidate/oracle `0.006558172317454591 / -0.0004785839410033077`, worst token `-0.8599309325218201`; ineligible because CI lower is not `>0`, oracle is negative, collision delta is `-0.001953125 < -0.001`, and worst token is not `>-0.5`.
- Stage-9 U2048 row: selected delta `0.006810275051975623`, gate CI95 `[-0.0020775724668055775, 0.01585598153978935]`, U128-relative delta `0.00414195007761009`, candidate/oracle `0.008875055245880503 / -0.0016392350080423057`, worst token `-1.0`; ineligible because CI lower is not `>0`, oracle is negative, and worst token is not `>-0.5`.
- Gate JSON / gate exit code / failure reasons: `artifacts/grpo_stage12/diagnostic/gate.json`; exit `1`; `passed=false`; exact `eligible_stage9_steps=[]`; exact aggregate failure `no Stage-9 U512+ checkpoint demonstrates continuation headroom`.
- Decision: STOP Stage 12. Do not train U256 and do not run seed, dev, or navtest phases.
- Recovery notes or reused-artifact identity proof: direct post-gate JSON audit confirmed `gate=stage12_diagnostic`, `passed=false`, `num_tokens=1024`, four checkpoint rows, and the empty eligible list. Post-stop process audit found no residual training or evaluation job; all eight RTX 4090s were at 0% utilization and GPU 0 had 48,619 MiB free.
