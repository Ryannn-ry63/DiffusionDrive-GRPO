# DiffusionDrive GRPO Phase 4: Magnitude-Aware Anchor RLOO

## Recovery status

- Status: Phase 4 complete; U128 failed fixed-256 oracle/catastrophic hard stops
- Started: 2026-07-18 (UTC)
- Objective: add the preregistered `generation_advantage_mode=anchor_rloo` objective, preserve all legacy objectives numerically, validate it, then run the gated K2 experiment if the environment is ready.
- Fixed experiment: seed 0, K=2, batch=2, 128 updates, LR `1e-6`, generation KL `0.1`, old-policy sync 32, uniform sampling/mode weighting, raw PDMS.
- Frozen hyperparameters: margin `0.01`, scale `0.20`, clip `1.0`; anchor margin `0.01`, anchor scale `0.10`, anchor clip `2.0`; RLOO/anchor mixing weights `0.5/0.5`.

## Recovery procedure

After an interruption, inspect this document, `git status --short`, active training/evaluation processes, and the latest checkpoint/artifact before running anything. Do not repeat completed training or evaluation.

## Execution log

- 2026-07-18 UTC: created this recovery document before Phase 4 implementation.
- Audited the dirty worktree and preserved existing Phase 3 changes. No training/evaluation process was active; all eight RTX 4090 GPUs were idle.
- Implemented detached `compute_anchor_rloo_advantages`: K2 leave-one-out baseline, margin `0.01`, scale `0.20`, clip `1.0`, no group standardization.
- Added `anchor_rloo` with matching fixed-reference anchor reward, `0.5/0.5` mixing, raw-PDMS routing, K2 enforcement, and independent KL over every valid transition.
- Added gap mean/P50/P90, dead-zone/clip/active fractions, final advantage signs, reference delta/coverage, ratio/KL, and decoder/perception gradient logs.
- Added a runtime hard failure if generation-only training produces a nonzero classification or perception gradient.
- Runner now enforces raw PDMS, K2, batch 2, uniform sampling/mode weighting, KL `0.1`, LR `1e-6`, sync 32, and fixed RLOO hyperparameters. `UPDATES` remains selectable for U8 smoke then U128.
- Focused command: `/root/miniconda3/envs/navsim/bin/python -m pytest -q test_generation_grpo_objective.py test_grpo_objective_v3.py test_diffusion_grpo_probability.py` -> `31 passed in 4.00s`.
- Static commands passed: touched-file `py_compile`; runner `bash -n`; `git diff --check`; no `.orig`/`.rej` files remain.
- Two focused-test failures found during development (RLOO-local initialization and exact-boundary test ordering) were fixed before the final green run.

## Required validation

- [x] K2 leave-one-out gaps match hand calculation.
- [x] RLOO dead zone, saturation, invalid/single-valid/all-tie behavior.
- [x] Rewards, reference rewards, and advantages are detached.
- [x] Anchor and RLOO terms preserve rollout ordering.
- [x] PPO ratio is one when current and old policies match.
- [x] Generation KL retains gradients when policy advantage is zero.
- [x] Classification and perception gradients are exactly zero in synthetic audit and in U8/U128 real-batch training.
- [x] Legacy objective numerical compatibility (`group_zscore`, `hierarchical`, `anchor_hierarchical`).
- [x] Focused pytest.
- [x] `py_compile`.
- [x] Shell syntax checks.
- [x] `git diff --check`.

## Experiment and gate log

- Initial U8 command failed before agent construction because the three dataclass fields were absent from Hydra's structured agent YAML. Added the same fixed values to `diffusiondrive_agent.yaml`; no update/checkpoint was produced by the failed attempt.
- U8 command: `bash scripts/training/run_diffusiondrive_grpo_ablation.sh D 0 0.0 4 0 0.1 0.001 0.1 uniform 2 anchor_rloo none 8 2`.
- U8 experiment: `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/grpo_d_seed0_skl0.0_gkl0.1_eps0.001_dense0.1_mweightuniform_bs2_advanchor_rloo_k2_uniform_u8/2026.07.18.11.12.05`.
- U8 completed exactly 8 optimizer updates; checkpoints: `lightning_logs/version_0/checkpoints/grpo-00-8.ckpt` and `last.ckpt`.
- U8 TensorBoard epoch metrics:
  - RLOO absolute gap mean/P50/P90: `0.234707/0.151688/0.588089`;
  - dead-zone/clip/active fractions: `0.134375/0.303125/0.865625`;
  - final positive/negative advantage fractions: `0.334375/0.600000`;
  - reference delta mean/std: `-0.120312/0.284127`; reference-anchor coverage: `1.0`;
  - generation KL: `2.24925e-7`; ratio mean: `0.999986`;
  - decoder/shared/regression grad norms: `12.5192/3.15507/12.1036`;
  - classification/perception grad norms: `0/0`.
- U8 decision: passed; advance to the preregistered seed-0 U128 run from frozen base.
- U128 command: `bash scripts/training/run_diffusiondrive_grpo_ablation.sh D 0 0.0 4 0 0.1 0.001 0.1 uniform 2 anchor_rloo none 128 2`.
- U128 experiment: `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/grpo_d_seed0_skl0.0_gkl0.1_eps0.001_dense0.1_mweightuniform_bs2_advanchor_rloo_k2_uniform_u128/2026.07.18.11.16.16`.
- U128 completed exactly 128 optimizer updates from frozen base. Preserved `grpo-step-64.ckpt`, `grpo-step-128.ckpt`, `grpo-00-128.ckpt`, and `last.ckpt`.
- U128 TensorBoard epoch metrics:
  - RLOO absolute gap mean/P50/P90: `0.265443/0.129490/0.735884`;
  - dead-zone/clip/active fractions: `0.239258/0.362695/0.760742`;
  - final positive/negative advantage fractions: `0.300391/0.573926`;
  - reference delta mean/std: `-0.177478/0.350770`; reference-anchor coverage: `1.0`;
  - generation KL: `1.40386e-5`; ratio mean: `0.999978`;
  - decoder/shared/regression grad norms: `14.4369/3.73665/13.9252`;
  - classification/perception grad norms: `0/0`.
- U64 is retained for diagnosis only and is not eligible for model selection. Next action: U128 fixed-256 hard gate.
- U64 training diagnostic (TensorBoard step 63): KL `8.88272e-6`, ratio `1.00032`, positive/negative final advantage `0.3125/0.6375`, reference coverage `1.0`, RLOO gap mean/P50/P90 `0.260185/0.126619/0.765959`, dead-zone/clip/active `0.275/0.400/0.725`. No U64 gate was used for selection.

- U128 fixed-256 command: `bash scripts/evaluation/run_diffusiondrive_grpo_gate.sh <grpo-step-128.ckpt> artifacts/grpo_stage8/anchor_rloo_k2_u128_fixed256.json 256 0`.
- Artifact: `artifacts/grpo_stage8/anchor_rloo_k2_u128_fixed256.json`; paired token SHA `d3c9f56e66c13bead6950141acc3a6bf44cafc98a27c7b49bd8126cc405e6113`.
- U128 fixed-256 selected delta: `+0.00103725`, CI95 `[-0.000249006,+0.00336073]`.
- U128 fixed-256 oracle delta: `-0.00323310`, below the preregistered `-0.001` hard-stop threshold.
- Worst single-token oracle delta: `-0.786897` on token `2db6398553cc5bfb` (base `0.786897` -> candidate `0.0`), also triggering the catastrophic `<-0.5` hard stop.
- Candidate-mean delta: `+0.000768426`; baseline-bottom-1% selected delta: `0.0`; worst selected 1% mean delta: `-0.0140262`.
- The legacy fixed baseline artifact lacks per-record safety components, so paired collision/drivable/TTC deltas cannot be reconstructed. This does not affect the stop: two independent oracle rules already fail.
- Stop decision: Phase 4 fails fixed-256. Do not run fixed-1024, dev-select, dev-confirm, seed 1/2, or full-navtest. U64 remains diagnostic-only and cannot replace U128.
- Route decision: because catastrophic oracle collapse remains, stop the RLOO/Step-Tree route; the next phase must investigate hard trust projection.

Record every command, checkpoint, TensorBoard metric, fixed/dev artifact, and stop decision below before advancing to the next gate.
