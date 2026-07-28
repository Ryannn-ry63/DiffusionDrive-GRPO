# Stage26: Final Selector Closure Before Protected Fold5

> Post-run identity correction (2026-07-24): references below to
> `official_base` denote the locked local epoch-19 checkpoint with SHA256
> `59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d`.
> They do not denote the released epoch-99
> `diffusiondrive_navsim_88p1_PDMS` checkpoint. The historical Stage26 plan is
> otherwise preserved unchanged; transfer to the actual public checkpoint is
> registered separately as Stage27.

## Status and purpose

- Preregistered on 2026-07-23 after the frozen Stage25 fresh-generator fold4
  gate passed and before any Stage26 candidate-bank generation, selector
  training, fold4 recalibration, or fold5 inspection.
- Stage25 has locked generator epoch 2 with paired fold4
  `C-B = +0.0101724`, whole-log 95% CI
  `[+0.0036014, +0.0184337]`.
- Stage26 has one purpose: make the selector's training distribution include
  that successful fresh GRPO generator, then prove on the already-consumed
  fold4 that this final selector preserves the generator-only improvement.
- Fold4 is calibration/development data in this stage and cannot support the
  final paper claim. Fold5 remains protected and unopened until every Stage26
  artifact, threshold, checkpoint, and training duration below is frozen.

## Immutable inputs

The five generator domains are:

| Domain | Checkpoint SHA256 |
|---|---|
| `official_base` | `59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d` |
| `stage16_epoch8` | `3a7641d4ac2a9d644eda4cb945d1902d4ac4bebfdad09b156676dc0cbed94e23` |
| `stage19_epoch8` | `f0bc08a76d5c3aa27f2f19fd46dcf569406bdd41ff5d9af0078fa75e12986c4b` |
| `stage21_epoch8` | `7a8d850280aa043ef41734fc9854e5fe8f4c2bb6376cf2d7274c1511d06a9492` |
| `stage25_epoch2` | `8a273860a365b2ca0eb6f97fddd8d381714cf7f4a30ceb08267aed1756413989` |

The Stage25 epoch-2 path is fixed as:

`/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage25_generator_development8_seed0/2026.07.23.08.31.19/lightning_logs/version_0/checkpoints/grpo-01-128.ckpt`.

The selector-fit population is the existing complete-log folds0--3 manifest:
4,075 tokens. The consumed calibration population is the existing complete-log
fold4 manifest: 1,021 tokens. Their file SHA256 values are respectively
`cefe6cdc470f9e14a5244ba422d7ff7265eed7769d4147189d705c1df1604f7f`
and
`8e0f55b18e1faf3e3b12788390abf041ab3049b7e72ff55b0968ebcfd71f4a23`.
Their token and log sets must remain disjoint.

## Candidate-bank construction

The final selector-fit bank is the Cartesian product of:

- five generator domains above;
- namespaces `default`, `20260811`, and `20260812`;
- all 4,075 folds0--3 tokens;
- all 20 candidate trajectories, reference logits, aggregate PDMS labels, and
  six PDM components for every token.

Existing SHA-audited records may be reused without recomputation:

- the official-base 4,075-token banks;
- the 3,056-token folds0--2 prefixes for Stage16/19/21.

For Stage16/19/21, generate only the missing 1,019-token fold3 suffixes and
merge them with the matching prefixes in the frozen folds0--3 manifest order.
Generate complete 4,075-token banks for Stage25 epoch 2. Every merged bank must
fail closed on namespace, generator SHA, domain, unique token order, log
provenance, zero evaluator failures, and all-20 tensor shapes. No PDM label is
available to selector inference; these labels are training/evaluation targets
only.

## Final selector training

The architecture, labels, losses, deployment logic, and optimizer
hyperparameters remain exactly the locked Stage24 value ensemble plus Stage25
relative-harm ensemble. No new head, feature, loss weight, safety tolerance,
token rule, generator identity input, or base-generator inference fallback is
allowed.

Training is sequential and uses final checkpoints only:

1. Train a new eight-member Stage24 safety/value ensemble from the official
   base checkpoint on all 15 fit banks.
2. Initialize a new Stage25 relative-harm ensemble from that mandatory final
   Stage24 checkpoint, freeze the complete Stage24 ensemble, and train only the
   Stage25 heads on the same 15 banks.

Each phase trains exactly 15 epochs. The current deterministic bank scheduler
uses one generator/namespace combination per epoch, so 15 epochs is exactly
one exposure to every `5 domains x 3 namespaces` combination. It is not an
epoch extension or a checkpoint sweep. With batch size 2, the locked
`stage24_selector_steps_per_epoch` is 2,038. Stage24 uses seed `26024`;
Stage25 uses seed `26025`. The final epoch is mandatory in both phases and no
intermediate checkpoint may be evaluated for selection.

Before each formal phase, an eight-step audit must show finite positive
gradients only in the intended trainable selector module and exact-zero
gradients in the generator, perception trunk, historical selectors, and every
frozen selector module. Candidate-bank cycling must be audited at the epoch
boundaries `0`, `14`, and `15`.

## Consumed-fold4 calibration

After both final selector checkpoints are frozen, collect the five generator
domains under the two already-used namespaces `20260821` and `20260822` on all
1,021 fold4 tokens. This creates ten fixed calibration artifacts.

Calibrate only the single Stage25 relative-harm risk threshold. Retain the
locked Stage24/25 definitions of value LCB, q95 conformal value residual,
diagonal embedding OOD rejection, `z=1.96`, and same-generator fallback.
The selected threshold must satisfy on the ten pooled cells:

- selector gain at least `+0.005`;
- whole-log bootstrap 95% CI lower bound strictly positive;
- non-negative gain in every generator-domain/namespace cell;
- switch rate at least 2%;
- each safety-component mean delta at least `-0.0005`;
- catastrophic-switch one-sided 95% Clopper--Pearson upper bound at most
  `0.0025`.

Failure stops Stage26. Thresholds, architectures, epochs, losses, or candidate
domains cannot be changed after inspecting this result.

## Final fold4 generator-attribution revalidation

Using the one calibrated final selector for both systems, re-evaluate:

- `B`: official base generator plus final selector;
- `C`: locked Stage25 epoch-2 generator plus final selector.

Use identical fold4 tokens and paired noise for namespaces `20260821` and
`20260822`. Reuse the exact frozen Stage25 fresh-generator gate:

- `C-B` positive in both namespaces;
- pooled `C-B >= +0.005` with whole-log 95% CI lower bound positive;
- hard-scene `C-B >= +0.010`;
- mature-scene `C-B >= -0.002`;
- each safety-component mean `C-B >= -0.001`;
- catastrophic-pair one-sided 95% upper bound `<= 0.005`;
- Stage25-generator candidate OOD rate `<= 0.05`.

The expected target is preservation of the already observed approximately
`+0.010` generator-only gain. This is a pass/fail revalidation, not permission
to select a more favorable selector checkpoint or threshold.

## Protected fold5 protocol

Only if both consumed-fold4 gates pass:

1. Freeze the final Stage24 checkpoint SHA, final Stage25 checkpoint SHA,
   calibration SHA, candidate-bank audit SHA, and all configs.
2. Retrain the generator once from the official base on folds0--4 for exactly
   two epochs with the unchanged Stage25 `diffgrpo_selected_set` objective and
   the final frozen selector.
3. Use the mandatory epoch-2 checkpoint; there is no epoch selection.
4. Open fold5 once and evaluate paired `B` versus `C` under namespaces
   `default`, `20260823`, `20260824`, and `20260825`.

Interpret generator-only fold5 `C-B` as:

- below `+0.005`: Stage26 fails;
- at least `+0.005`: minimally useful;
- at least `+0.010`: paper target;
- at least `+0.015`: desired target.

The final report must also separate `B-B0` (selector contribution) and
`C-B0` (total system contribution), where `B0` is official base plus official
selector. A fold5 failure forbids threshold tuning, extra epochs, a second
fold5 attempt, or NavTest within this closure.

## NavTest boundary

Only a protected-fold5 pass authorizes full 6,119-scene training on external
8x H100 (preferred) or 8x RTX 4090 for exactly two epochs, followed by the
official NavTest/PDMS evaluation. Multi-dozen-epoch training is explicitly
forbidden because Stage25 showed that later epochs increase mean development
gain while violating mature-scene and tail-risk gates.

## Execution record

Results are appended here without changing the preregistered protocol above.

### 2026-07-23: candidate-bank and selector closure

- All 15 candidate-bank artifacts passed the full-bank audit: five generator
  domains times three namespaces, 4,075 records per artifact and 61,125
  domain/namespace records in total. Audit SHA256:
  `caab2f6adf620dfa166a4bdbeae8bbe4e350c26b00c603bde2cd5892fcab4066`.
- The deterministic bank-cycle audit passed with 2,038 steps per epoch and
  30,570 updates over the locked 15 epochs. Audit SHA256:
  `d1a23adcc770c038865cb864ded0742ae3c4ed92d31d6a65b917f9a4c64a34ca`.
- Final Stage24 selector checkpoint SHA256:
  `6a90a1c5ab74bc980e53da800f3fab7da045011b3045f7ea23fd82698b534a7e`.
  Its epoch loss decreased from `0.72368` to `0.41189`; Stage24 gradients
  were positive on all 30,570 steps and every frozen-module gradient was zero.
- Final Stage25 selector checkpoint SHA256:
  `77ee768c5f469292d81049192d48797f2f3fae4f80029ae6853dd9d9c6de4207`.
  Its epoch loss decreased from `1.39757` to `0.88614`; Stage25 gradients
  were positive on all 30,570 steps and every frozen-module gradient was zero.

### 2026-07-23: consumed-fold4 gates

- Final-selector calibration passed. Mean selector gain was `+0.00732187`,
  whole-log bootstrap 95% CI was `[+0.00526294, +0.00948703]`, all ten
  generator-domain/namespace cells were positive, switch rate was `4.72%`,
  and all safety/tail gates passed. Calibration SHA256:
  `0c14b8f9531d64028bdf4d71f35b34418ab31ad4ae11ccc6b3c60a4325f53a0d`.
- The fixed Stage25 epoch-2 `C-B` generator gate passed with pooled gain
  `+0.01059044`, whole-log 95% CI `[+0.00400007, +0.01883401]`, namespace
  gains `+0.01204268` and `+0.00913820`, hard-scene gain `+0.04986276`,
  mature-scene gain `-0.00128142`, and all safety/tail/OOD gates passing.
  Gate SHA256:
  `3436534cdca9e8aefd2b4410b25a1198be3c267bcd2dc7dd7c4efdd6fae6a5a2`.
- Stage26 final-selector closure passed. Protected fold5 remains unopened
  pending the mandatory final generator checkpoint.

### 2026-07-23: final-generator preparation and DDP audit

- The folds0--4 whole-log-isolated manifest contains 5,096 tokens from 757
  logs. Manifest SHA256:
  `1ed85e5fc3d46505e715c297d776dc73801f61dabf3ff32456afa8f709b22705`.
- The 8-GPU audit completed normally at `max_steps=1`: all eight NCCL ranks
  registered, the diffusion decoder gradient norm was `0.17855069`, and the
  perception, value-selector, Stage23 selector, Stage24 selector, Stage25
  selector, and paired-risk gradient norms were exactly zero. Audit run:
  `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/`
  `stage26_generator_final_ddp8_audit/2026.07.23.15.04.27`.
- The formal external entry point is `run_stage26_generator_folds0_4_h100.sh`.
  It is locked to the official base, final selector, folds0--4 manifest, eight
  GPUs, and exactly two epochs. Fold5 has not been evaluated.

### 2026-07-23: final generator frozen and fold5 authorized

- External 8x H100 training completed at the mandatory epoch-2 checkpoint
  `grpo-01-160.ckpt`. Its SHA256 is
  `c0baa2c42023727195f7f28e83e1e9532783e572ddbc8fc60d9ed7a25409ec2b`.
- The checkpoint reports `epoch=1` and `global_step=160`; Hydra records exactly
  two epochs, eight DDP devices, 5,096 folds0--4 tokens, the official base,
  and the frozen final selector.
- All 160 logged diffusion-decoder gradient norms were positive
  (minimum `0.10557979`, mean `0.16537645`, maximum `0.20356491`).
  Perception, value selector, Stage23 selector, Stage24 selector, Stage25
  selector, paired-risk, and classification gradients were exactly zero on
  all 160 steps.
- The fail-closed audit is frozen at
  `artifacts/grpo_stage26/final_generator_audit.json`. This audit authorizes
  the single protected-fold5 evaluation; no fold5 result had been inspected
  when the checkpoint and evaluation code were frozen.

### 2026-07-23: one-shot protected fold5 passed

- The one-shot fold5 evaluation completed on all 1,023 tokens from 151 whole
  logs under `default`, `20260823`, `20260824`, and `20260825`, with 4,092
  paired observations. All eight raw B/C artifacts passed the frozen
  checkpoint, selector, calibration, schedule, token-order, and completeness
  checks.
- Mean system scores were `B0=0.80783112`, `B=0.81472443`, and
  `C=0.82332731`.
- The selector contribution `B-B0` was `+0.00689332`, with whole-log 95% CI
  `[+0.00500502, +0.00905035]`.
- The GRPO generator contribution `C-B` was `+0.00860288`, with whole-log 95%
  CI `[+0.00427241, +0.01314559]`. All four namespace gains were positive:
  `+0.01286716`, `+0.00570520`, `+0.00787156`, and `+0.00796758`.
- The total system contribution `C-B0` was `+0.01549619`, with whole-log 95%
  CI `[+0.01074572, +0.02046724]`, reaching the preregistered desired target.
- Generator-only hard-scene gain was `+0.04095181`; mature-scene gain was
  `-0.00109769`. Mean collision, drivable, and TTC deltas were respectively
  `+0.00061095`, `+0.00757576`, and `+0.00268817`. Candidate OOD rate was
  `0.00478983`.
- The frozen report is `artifacts/grpo_stage26/fold5/final_report.json`, SHA256
  `80b951729b4d6041a0d7d02b14e18de1846fac013bdc98357acca4e3dc525e3f`.
  It records `passed=true` and `stop_before_navtest=false`; Stage26 therefore
  authorizes full 6,119-scene training for exactly two epochs and subsequent
  official NavTest/PDMS evaluation.

### 2026-07-23: full-6119 NavTest training prepared

- The six folds were merged into 6,119 unique training scenes from 908
  whole logs. Manifest SHA256:
  `1763ca12bfd1bf480a1ed6a47cc71f95ef31123517e500551dae24292620ef68`.
- The 8-GPU one-step audit passed with diffusion-decoder gradient norm
  `0.17532101`; perception, value selector, Stage23 selector, Stage24 selector,
  Stage25 selector, and paired-risk gradients were exactly zero.
- The locked external entry point is `run_stage26_navtest_generator_h100.sh`.
  It retrains from the official base on all 6,119 scenes for exactly two
  epochs; no epoch selection or extension is allowed.

### 2026-07-24: full-6119 generator frozen and NavTest authorized

- External 8x H100 training completed in
  `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/`
  `stage26_navtest_generator_full6119_seed0/2026.07.23.17.12.16`.
  The mandatory final checkpoint is `grpo-01-192.ckpt`, with SHA256
  `1018f1b1a1cfcb69c27b97c2fbd30099a20a62e55fc02897f6588ca546abad6d`.
  It is byte-identical to `last.ckpt`.
- The checkpoint reports `epoch=1` and `global_step=192`. Hydra records the
  locked 6,119-token whole-log manifest, exactly two epochs, eight DDP devices,
  the official base, the final selector, and the unchanged full-chain
  `diffgrpo_selected_set` objective.
- A direct tensor audit against the official base matched all 763 official
  tensors. Exactly 699 were bitwise unchanged; the only 64 changed tensors
  were the 32 tensors in each of diffusion-decoder layers 0 and 1. No official
  perception, BEV, agent, classification, or other frozen tensor changed.
- The fail-closed checkpoint audit is
  `artifacts/grpo_stage26/navtest_generator_checkpoint_audit.json`.
  It records `passed=true` and `navtest_authorized=true`.
- NavTest will report four systems to avoid attributing a sampling-schedule
  change to GRPO: `A` reproduces the public 88.1 setup with the shared
  token-deterministic evaluation noise; `B0` is the official generator and
  official selector under the Stage26 full-chain schedule; `B` adds the frozen
  Stage26 selector to `B0`; `C` uses the full-6,119 Stage26 GRPO generator and
  the same frozen selector. The primary GRPO attribution is paired `C-B`,
  selector attribution is `B-B0`, total Stage26 attribution is `C-B0`, and
  `C-A` is reported separately as the end-to-end change from the public setup.
