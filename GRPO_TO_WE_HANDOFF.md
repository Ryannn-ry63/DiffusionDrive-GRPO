# DiffusionDrive GRPO Handoff

Last updated: 2026-07-15

This document is the source of truth for continuing the current work in a new
Codex session. Read it before changing code or launching experiments.

## 1. Goal and scope

The current goal is to improve DiffusionDrive planning performance with GRPO.
The GRPO result must be attributable to reinforcement learning itself, so the
GRPO stage must not include imitation loss or supervised diffusion loss.

WorldEngine (WE) integration is the following stage. Do not redesign the whole
WE-to-DiffusionDrive interface yet. First establish whether the new GRPO method
improves DiffusionDrive, then port the validated implementation into WE in the
same way that the original DiffusionDrive implementation was ported.

## 2. Repository snapshot

- Repository: `/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive`
- Current branch: `grpo-selection-v2`
- Starting point requested by the user:
  `newbaselinegrpo` commit `2d783740d993e6a6729f8b388c3806494cd4214b`
- Working tree was clean immediately before this handoff file was created.
- GitHub remote used for saving the work:
  `https://github.com/Ryannn-ry63/DiffusionDrive-GRPO.git`

Recent commits:

```text
dbca5ef Record generation GRPO tuning results
adb5d33 Implement generation-aware DiffusionDrive GRPO
29118b2 Document next-stage generative GRPO plan
3a00a21 Implement and validate classification GRPO v2
2d78374 Original newbaselinegrpo starting commit
```

At the beginning of a new session, run:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
git status --short
git branch --show-current
git log --oneline -5
```

Preserve any new user changes if the working tree is no longer clean.

## 3. User decisions and hard constraints

1. GRPO training must not use imitation loss. Otherwise an improvement cannot
   be cleanly attributed to GRPO.
2. Freeze the perception stack. Train only the later planning components,
   principally the DiT blocks/planning head.
3. Run DiffusionDrive experiments in the `navsim` conda environment, not the
   current `algengine` environment.
4. Prefer short oracle/smoke experiments before committing to long single-GPU
   training.
5. For the later WE port, keep the existing WE baselines intact and add a
   separate GRPO head/config.
6. For WE reward computation, the selected direction is true online PDM
   scoring of dynamically generated candidates, not a nearest-neighbor lookup
   into the fixed 8192 trajectory bank.

## 4. What has been implemented in DiffusionDrive

The main GRPO implementation is in:

- `navsim/agents/diffusiondrive/diffusion_grpo.py`

The implementation now includes:

- Stochastic DDIM transitions and action log-probability computation.
- Replay of the same sampled state/action under old, current, and reference
  policies. This makes the PPO ratio and KL comparison meaningful.
- Classification/selection GRPO, generation GRPO, and joint modes.
- PPO clipping and a Gaussian transition KL term.
- A fixed diffusion schedule using 8 inference steps, timesteps `[8, 0]`, and
  the training horizon expected by the model.
- Removal of the earlier hard-coded test-time scheduling mismatch.
- Explicit freezing modes plus gradient auditing to verify which parameters
  receive gradients.
- Evaluation scripts for common-token comparisons and schedule checks.
- No imitation loss or ground-truth trajectory regression in the GRPO
  objective.

Relevant tests include:

- `test_diffusion_grpo_probability.py`
- `test_diffusion_schedule.py`
- `test_generation_grpo_objective.py`
- `test_grpo_objective_v3.py`
- `test_grpo_training.py`

The focused GRPO test suite passed 13/13 tests after the latest implementation.

Useful design and experiment documents already in this repository:

- `grpo_logic_clarification.md`
- `GRPO_SELECTION_V2_PLAN.md`
- `GRPO_NEXT_STAGE_WE_ALIGNMENT_PLAN.md`
- `GRPO_IMPLEMENTATION_SUMMARY.md`
- `GRPO_TRAINING_GUIDE.md`
- `GRPO_QUICK_REFERENCE.txt`

Read the implementation and these documents before making architectural
changes. The code is more authoritative than an older planning note if they
disagree.

## 5. Checkpoints and current evidence

Base DiffusionDrive checkpoint:

```text
/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model
```

Best generation-GRPO checkpoint so far (experiment D, 128 updates):

```text
/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/grpo_stage2_D_gen_128/2026.07.15.05.10.31/lightning_logs/version_0/checkpoints/grpo-00-128.ckpt
```

On the fixed common-1024 proxy evaluation:

| Model | Selected score | Oracle score |
| --- | ---: | ---: |
| Base checkpoint | 0.757878 | 0.917801 |
| Generation GRPO D, 128 | 0.761178 | 0.918610 |
| Difference | +0.003299 | +0.000809 |

The paired bootstrap 95% confidence interval for the selected-score difference
was approximately `[+0.000782, +0.006601]`. This is a real positive proxy
signal, but it is small and remains below the provisional `+0.005` engineering
gate.

Other observed results:

| Experiment | Selected | Oracle | Interpretation |
| --- | ---: | ---: | --- |
| D generation, 256 | 0.757323 | 0.919837 | Longer training regressed selection |
| E joint, 128 | 0.759181 | not primary | Below D-128 |
| E joint, 256 | 0.756787 | not primary | Regressed |
| gen lr=2e-6, std=0.05 | 0.760274 | not primary | Positive but below D-128 |
| gen lr=3e-6, std=0.05 | 0.759107 | not primary | Below D-128 |
| gen lr=2e-6, std=0.10 | 0.760559 | not primary | Positive but below D-128 |
| sequential selector S1 | 0.760423 | not primary | Below D-128 |
| sequential selector S2 | 0.759229 | not primary | Below D-128 |
| sequential selector S3 | 0.760391 | not primary | Below D-128 |

Pure classification variants B1/B2/B3 did not beat the base reliably.
Generation-only GRPO is the best direction currently tested. Joint training
and additional selector tuning did not improve on D-128.

Historical reference only:

```text
navtest PDMS = 0.8487464442665378
```

The current best GRPO checkpoint has not yet been evaluated through the same
full navtest pipeline against a paired base checkpoint. Therefore, do not claim
that GRPO has exceeded 0.84 yet. The current result is only a common-1024 proxy
improvement.

## 6. Immediate next action in DiffusionDrive

The first task in the new session is a full paired navtest evaluation:

1. Evaluate the base checkpoint and D-128 checkpoint with exactly the same
   navtest tokens, metric cache, diffusion schedule, worker setup, and relevant
   random-seed policy.
2. Save aggregate scores and per-token results for both checkpoints.
3. Report overall PDMS, selected trajectory score, oracle score if available,
   failure-subset performance, and paired differences.
4. Compare D-128 to the newly measured paired base. Treat the historical
   `0.848746` result only as a reference unless its protocol is identical.
5. If D-128 improves the paired full navtest result, proceed to the WE port.
6. If it does not, diagnose the common-1024/navtest mismatch before spending
   time on WE integration or a long multi-GPU run.

A full many-epoch retraining is not the immediate next step. First evaluate the
existing best short-run checkpoint. A long multi-GPU training should be
justified only after the paired full evaluation confirms that the short-run
signal transfers to navtest.

## 7. WorldEngine context for the following stage

WE repository:

```text
/inspire/hdd/global_user/wangcaojun-240208020180/nry/Worldengine-Diffusion
```

Last inspected WE state:

- Branch: `main`
- Commit: `aa64610` (`0525basemodel`)
- Existing untracked/dirty items included `DiffusionDrive-main/` and
  `meeting_note.md`. Preserve them.

Existing WE DiffusionDrive baseline files:

```text
projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_planning_head.py
projects/AlgEngine/configs/navformer/e2e_diffusiondrive.py
```

Existing WE DiffusionDrive checkpoint:

```text
/inspire/hdd/global_user/wangcaojun-240208020180/nry/Worldengine-Diffusion/experiments/navformer/e2e_diffusiondrive/epoch_8.pth
```

Existing detector checkpoint:

```text
/inspire/hdd/global_user/wangcaojun-240208020180/nry/Worldengine-Diffusion/data/alg_engine/ckpts/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth
```

The existing WE planning head adapts NavFormer BEV features to DiffusionDrive,
generates status/query features, converts an 8-step representation to a
40-step trajectory, and can recompute candidate scores at test time. Its
current training objective is supervised diffusion loss and is not yet the
desired pure GRPO training path.

NavFormer already contains online single-trajectory PDM recomputation methods,
including `_forward_test_recompute_pdm`, `_score_generated_trajectory`, and an
LRU-style cache. Training metadata includes `metric_cache_path`. However, the
current training path mainly passes precomputed `gt_pdm_score` values for a
fixed 8192-trajectory bank, which is not sufficient for true GRPO over newly
generated trajectories.

## 8. Proposed WE port after DiffusionDrive validation

Port the validated code as a parallel implementation:

1. Add
   `projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_planning_head.py`.
2. Add a registered `DiffusionGRPOPlanningHead` class.
3. Add
   `projects/AlgEngine/configs/navformer/e2e_diffusiondrive_grpo.py`.
4. Do not overwrite the existing `DiffusionPlanningHead` or baseline config.
5. Reuse the existing BEV-to-DiffusionDrive adapter and port the validated
   stochastic transition, log-probability, old/reference replay, generation
   GRPO, PPO clipping, and KL logic.
6. Keep imitation/supervised diffusion loss out of the GRPO objective.
7. Freeze the perception stack and train the planning DiT blocks/head.
8. Make the smallest NavFormer detector change needed to conditionally pass
   `img_metas`/metric context to heads that declare something like
   `requires_metric_context = True`.
9. Score all dynamically generated GRPO candidates with online PDM in no-grad
   CPU code, backed by an LRU cache. Do not use nearest-8192 matching as the
   formal reward.
10. Start from the WE epoch-8 DiffusionDrive checkpoint as the policy/reference
    initialization.

Recommended initial WE smoke configuration, subject to matching the actual
ported config names:

```text
mode: generation-only GRPO
dynamic candidates per scene: 20
learning rate: 1e-6
DDIM eta: 1.0
final transition std: 0.05
generation KL coefficient: 0.1
perception: frozen
planning DiT/head: trainable
```

The senior student's suggestion not to freeze the GRPO task is best understood
as allowing the generative planning policy to change, rather than training only
a detached classifier. It does not require unfreezing the expensive perception
backbone. The present evidence supports this interpretation: pure
classification GRPO was weak, while generation GRPO changed the diffusion
policy and produced the best proxy result.

## 9. Validation gates for the WE port

Before a long run:

1. Config import and model construction succeed.
2. Ported probability/KL tests pass.
3. One-batch online-PDM reward smoke test succeeds for dynamic candidates.
4. Gradient audit confirms that perception is frozen and the intended
   planning DiT/head parameters receive gradients.
5. Existing WE baseline config still builds and evaluates unchanged.
6. Run a very short smoke (about 8 batches), then a 128-update experiment.

Suggested promotion gates:

- Common-set selected-score gain at least `+0.005`.
- Failure-subset gain at least `+0.02`.
- Oracle-score drop no worse than `-0.005`.
- Reproduce the direction with two seeds.
- Final paired full evaluation must beat the WE baseline before claiming a
  performance improvement.

## 10. Suggested prompt for a new session

If the new session has access to the same filesystem, provide this file path
and use a prompt like:

```text
Please first read
/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/GRPO_TO_WE_HANDOFF.md
in full. Then inspect git status, the current branch, the recent commits, and
the referenced GRPO implementation. Continue from section 6, performing the
paired full navtest evaluation in the navsim environment. Preserve existing
user changes and do not start the WE port until the DiffusionDrive evaluation
result is established.
```

If filesystem access is not shared, paste the full contents of this file into
the new session and separately provide any required repository access.
