# DiffusionDrive GRPO Phase 5: Hard Functional Trust Projection

## Recovery contract

Phase 5 addresses rare candidate-set collapse rather than average PPO instability. The
carrier objective is the previously positive K2 `hierarchical` generation objective.
Anchor-RLOO, Step-Tree, selector training, imitation, shaping, priority sampling, and
oracle/PDM signals are not part of the trust calibration or online projection.

The current and old denoising means are projected independently for every scene,
anchor, and diffusion step around the frozen base-policy mean:

```text
z = (mu_current - mu_base) / sigma_step
d = sqrt(mean(z^2))
alpha = min(1, radius_step / (d + 1e-8))
mu_projected = mu_base + alpha * (mu_current - mu_base)
```

The two registered radii are the transition-step and final-step P99 normalized RMS
displacements of the formal D128 champion from the base policy on the 6,119-token
train-only manifest. Calibration seed is `20260719`; current and base use the same
initial diffusion state. The resulting JSON records the distribution quantiles,
formula version, schedule, seed, ordered manifest SHA256, and base/D128 checkpoint
SHA256 values. The builder makes the artifact read-only. Runtime loads the radii from
that artifact and rejects missing references, provenance/schedule mismatches,
non-finite values, shape mismatches, or a post-projection excess above `1e-6`.

`generation_trust_projection_mode=none` is the exact legacy path.
`generation_trust_projection_mode=reference_mean_ball` is always active in both GRPO
training and inference. Frozen reference means and sampled action targets are detached.
Current and old policies are projected before action/log-prob evaluation; generation KL
uses projected current means against the unprojected frozen base means.

## Preregistered execution

1. Collect D128/base raw displacements on exactly the train-only manifest with
   `scripts/evaluation/evaluate_grpo_schedule.py --collect-trust-calibration`.
2. Build the immutable P99 artifact with
   `scripts/evaluation/build_generation_trust_calibration.py`.
3. Evaluate the D128 champion with projection on fixed-256 and run the compatibility
   gate. Stop on selected/oracle delta below `-0.0005`, any token oracle delta below
   `-0.5`, incomplete reference coverage, or a projection bound violation.
4. Run the fixed Phase-5 K2 hierarchical recipe: seed 0, batch 2, LR `1e-6`, KL `0.1`,
   old sync 32, uniform sampling/weighting, raw PDMS, U8 smoke then U128; U64 is
   diagnostic only.
5. Apply the fixed-256 hard stop (`selected/oracle < -0.001`, collision/drivable/TTC
   `< -0.002`, token oracle `< -0.5`, coverage/bound failure), then fixed-1024 and
   dev-select gates recorded in the preregistration. Only a dev-select mean delta at
   least `+0.003` authorizes seed 1/2 and dev-confirm.

If the always-on projection still permits catastrophic oracle collapse, stop the GRPO
generation route. Do not change the percentile/radius and do not select U64 after the
fact.

## 2026-07-18 execution record

Phase 5 was executed through its preregistered fixed-256 hard stop.

- Raw train-only collection:
  `artifacts/grpo_stage5/d128_train6119_trust_collection.json`; all 6,119 ordered
  tokens matched SHA256
  `c7edaae00ad527b2cfe3c4391794dc70798a7e5bd69faa24c4b603822c75fde0`,
  reference coverage was 100%, and each step contained 122,380 anchor records.
- Immutable calibration:
  `artifacts/grpo_stage5/generation_trust_calibration_d128_p99.json` (mode 0444).
  The transition radius is `0.005229807957075536` and the final radius is
  `0.02303805343806742`. The artifact binds base SHA
  `59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d`
  and D128 SHA
  `eaeff0efb6e8dcce2e6eb9c915f49962b55a3a66324216e2f253b67967f47c70`.
- D128 projected compatibility artifact:
  `artifacts/grpo_stage5/d128_projected_fixed256.json`. The compatibility gate
  passed: selected delta `+0.0000002743`, oracle delta `+0.0000000731`, minimum
  token oracle delta `-0.0000008345`, 100% coverage, and no projection excess.
- The first U8 attempt correctly failed closed on a non-finite gradient at exact
  `current == base`. The mathematically equivalent RMS implementation was changed
  from explicit `sqrt(mean(z^2))` to `vector_norm(z) / sqrt(numel(z))`, giving a
  finite zero-displacement backward without changing forward values, radii, or the
  artifact. A focused exact-reference backward regression test was added.
- The repeated U8 smoke passed: both reference coverages were 100%; transition/final
  post maxima were `0.00162650` and `0.00120007`; all gradients were finite; and
  classification/perception gradient norms were exactly zero.
- Formal U128 checkpoint:
  `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/grpo_d_seed0_skl0.0_gkl0.1_eps0.001_dense0.1_mweightuniform_bs2_advhierarchical_k2_uniform_u128/2026.07.18.12.47.46/lightning_logs/version_0/checkpoints/grpo-00-128.ckpt`.
  Training maintained 100% coverage, finite KL/ratio/gradient metrics, and zero
  classification/perception gradients.
- Paired fixed-256 artifacts:
  `artifacts/grpo_stage5/base_fixed256_phase5.json` and
  `artifacts/grpo_stage5/phase5_k2_hierarchical_u128_fixed256.json`.
  Selected delta was `+0.0010475407`, candidate-mean delta `+0.0004500390`, and
  oracle delta `-0.0019340955`. Collision/drivable deltas were zero and TTC delta
  was `+0.00390625`. Projection coverage was 100%, with transition/final post maxima
  `0.0052300645` and `0.0230380930`, both within the registered radius plus `1e-6`.
- The fixed-256 gate failed on oracle mean `< -0.001` and one catastrophic token:
  token `2db6398553cc5bfb` changed from base oracle `0.7868967056` to `0`, for delta
  `-0.7868967056`. Excluding that single token, oracle mean delta was
  `+0.0011441892`. None of this token's 20 anchors triggered projection at either
  step: its transition/final pre-distance maxima were `0.0040658438` and
  `0.0194049217`, already inside the registered radii. The failure is therefore an
  insufficiency of the D128-P99 functional envelope for this rare state, not missing
  reference coverage or a post-projection bound violation.

This is the registered terminal result for the current GRPO generation route. Do not
run fixed-1024, dev-select, dev-confirm, seed 1/2, a post-hoc U64 selection, or a new
radius/percentile scan from this branch.
