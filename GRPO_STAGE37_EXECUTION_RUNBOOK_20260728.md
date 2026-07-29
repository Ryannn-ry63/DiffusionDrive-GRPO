# Stage37 execution runbook (2026-07-28)

This runbook executes the frozen design in
`GRPO_STAGE37_BISTATE_PROJECTED_JFI_PLAN_20260728.md` (SHA256
`4dc469dd0be4d2579796ff70ae7a09ed9fd8e4c92449cd057e42463d268bb5d8`).
It must not be used to bypass either pilot gate.

## Fixed scope

- Public reference: the official 0.881 checkpoint, SHA256
  `008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b`.
- Pilot/test folds: 0 and 1 only.
- JFI selector training folds: 2 and 3 only; its frozen token manifest contains
  2035 frames from 303 logs and has SHA256
  `d782a5c527e578d92816c8bc7adb5e98ffd9f095f36620af2457f478b5f2cead`.
- Generator candidates: optimizer steps 48, 96, 144, and 192.
- Evaluation namespaces: 20261511 and 20261512.
- No folds 2/3 expansion, full-6119 retraining, or NavTest launch is allowed
  until both the generator and full-system gates pass.

## Implementation audit already completed

The real eight-GPU fold-0 audit passed at one optimizer step:

- audit: `artifacts/grpo_stage37/generator/fold0/audit/audit.json`
- checkpoint freeze:
  `artifacts/grpo_stage37/generator/fold0/audit/checkpoints.json`
- 64/64 permitted decoder tensors changed, 32 in each decoder layer;
- accumulated microbatch count was exactly 8;
- reward/preservation cosine was -0.1722369, so the conflict projection was
  exercised rather than merely configured;
- mature-positive exploration was zero and every projection diagnostic was
  finite.

This is an implementation/mechanism result, not a PDMS result.

## Phase 1: jobs that can be queued in parallel

Run every command from the exact repository path:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
```

Queue these as independent jobs. The fold-0 generator job safely reuses the
passed audit above and starts formal training; fold 1 performs audit then
formal training.

```bash
bash ./run_stage37_bistate_generator_h100.sh 0 stage37_bpd_chain_f0
```

```bash
bash ./run_stage37_bistate_generator_h100.sh 1 stage37_bpd_chain_f1
```

Build the eight frozen JFI candidate banks (one bank per GPU):

```bash
bash ./run_stage37_jfi_banks_h100.sh
```

This fourth job may be queued immediately. It waits for the bank freeze and
then performs the JFI audit and formal selector training; no manual handoff is
needed:

```bash
bash ./run_stage37_jfi_after_banks_h100.sh stage37_jfi_chain 0
```

Phase 1 is complete only when all of the following exist and report `passed`:

- `artifacts/grpo_stage37/generator/fold0/formal/audit.json`
- `artifacts/grpo_stage37/generator/fold1/formal/audit.json`
- `artifacts/grpo_stage37/jfi/banks/freeze.json`
- `artifacts/grpo_stage37/jfi/training/formal/audit.json`

## Phase 2: generator ceiling gate

After both formal generator jobs pass, evaluate the four saved steps on each
held-out fold. Each command occupies eight GPUs with one step/namespace cell
per GPU and the two folds may run concurrently.

```bash
bash ./run_stage37_generator_eval_h100.sh 0
```

```bash
bash ./run_stage37_generator_eval_h100.sh 1
```

After both evaluation jobs pass, run once on any repository node:

```bash
bash ./scripts/evaluation/finalize_grpo_stage37_generator.sh
```

The finalizer creates `artifacts/grpo_stage37/pilot/generator_selection.json`
only if a single cross-fold step satisfies every frozen generator gate. A
nonzero exit is a scientific stop, not an infrastructure failure. Do not
continue to calibration if this selection file was not created.

The most important generator requirements are selected gain >= 0.002, a
strictly positive whole-log CI lower bound, both folds and both namespaces
positive, hard-scene gain >= 0.008, mature-scene gain >= -0.0001, and no
regression relative to the Stage36 RGT192 ceiling.

## Phase 3: cross-fitted JFI calibration

This phase requires both the selected generator and the trained JFI selector.
Collect eight calibration cells in one eight-GPU job:

```bash
bash ./run_stage37_jfi_calibration_collect_h100.sh
```

Then calibrate each held-out fold once:

```bash
bash ./scripts/evaluation/finalize_grpo_stage37_jfi_calibration.sh
```

Calibration fixes the inherited Stage25 OOD threshold, a joint feasibility
LCB, and a q10 improvement floor. It chooses at most three challenger modes
plus the public fallback. The calibrated mean pool size must remain in [2, 4]
and the one-sided 95% upper confidence bound on catastrophic switches must be
at most 0.005.

## Phase 4: frozen 2x2 attribution and final pilot gate

Run the two missing JFI cells for both folds and both namespaces:

```bash
bash ./run_stage37_factorial_h100.sh
```

The summarizer reuses the already frozen Public+Stage25 and BPD+Stage25 cells,
so the final table is exactly:

| generator | selector |
|---|---|
| Public | Stage25 |
| BPD | Stage25 |
| Public | JFI |
| BPD | JFI |

Finalize once:

```bash
bash ./scripts/evaluation/finalize_grpo_stage37_factorial.sh
```

Promotion requires full-system gain >= 0.005, generator effect under the same
JFI selector >= 0.001, a strictly positive whole-log CI lower bound, every
fold and namespace positive, no component regression beyond 0.0005, more wins
than losses, nonnegative trimmed-10% mean, zero catastrophic switches, and a
deployment cap of four candidates including fallback.

The authoritative report is
`artifacts/grpo_stage37/pilot/factorial_summary.json`. Only a passing report
authorizes planning the full-6119 retrain and NavTest deployment. A failing
report freezes the diagnosis by factorial cell and must not be converted into
a NavTest claim.
