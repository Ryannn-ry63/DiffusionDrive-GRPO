# Stage29 implementation and H100 runbook (2026-07-25)

## Frozen identity

- Plan: `GRPO_STAGE29_REFERENCE_ANCHORED_HEADROOM_PLAN_20260725.md`
- Plan SHA256: `d59f9161f2d9c2d4e99d261db788206dad19412b9569c47a7703b37da121c28b`
- Pilot inputs: `artifacts/grpo_stage29/pilot/inputs.json`
- Pilot-input SHA256: `775e8ac63c8b398e91809e8e4d7222a724f29552fc397dd3f69a215a8fc4d0f7`
- Public reference SHA256: `008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b`
- Training selector: frozen passing S-multi only.

## Implemented scope

Stage29 adds reference-anchored headroom GRPO and exactly two public-baseline
branches: fixed-regularization H and headroom-conditioned HC. Both restart from
the released 88.1041 checkpoint, train only the registered 64 decoder tensors,
and leave reference, selector, perception, and classification frozen.

The fold4 grid is fixed to P, historical non-selectable A3, H1--H4, and
HC1--HC4 under namespaces 20261211 and 20261212. The checker applies all frozen
selected, candidate, oracle, CI, namespace, trimmed-mean, hard/mature,
component, and zero-catastrophe gates. A3 can never be selected.

## Verification completed

- Stage29 target, audit-amendment, and fold4-grid tests: 20 passed.
- Stage23--29 critical regression suite: 62 passed (three benign PyTorch warnings).
- H one-step real optimizer audit: passed.
  - Audit: `artifacts/grpo_stage29/pilot/training/H/audit/audit.json`
  - active-scene fraction 0.875; positive fraction 0.53125; neutral 0.125.
  - exactly 64 changed decoder tensors; zero forbidden gradients.
- HC one-step real optimizer audit: passed.
  - Audit: `artifacts/grpo_stage29/pilot/training/HC/audit/audit.json`
  - active-scene fraction 0.875; positive fraction 0.53125; neutral 0.125.
  - exactly 64 changed decoder tensors; zero forbidden gradients.

## Formal training status

Both independent 8xH100 formal jobs completed all four epochs and global step
256. Their initial wrapper failures occurred after training, in the checkpoint
audit, because the original signal gate incorrectly imposed an upper bound on
the fraction of scenes containing at least one active rollout. With eight
rollouts and dense signed advantages, that scene-level statistic is naturally
close to one and is not a collapse indicator.

The correction was frozen before any fold4 scoring:

- Amendment: `GRPO_STAGE29_PREFOLD4_SIGNAL_GATE_AMENDMENT_20260725.md`
- Amendment SHA256:
  `941eef5f370f90ad9e5b9c7ed7a1dbaa12a971f30e79a211c446098f576f3bd9`
- The invalid `active_scene_fraction <= 0.95` condition was removed.
- Finite/unit-interval checks and the frozen lower activity, positive, and
  neutral-fraction checks remain unchanged.
- No loss, reward weight, epoch count, checkpoint, selector, namespace, or
  PDMS/fold4 acceptance gate changed.

The already-trained checkpoints were then re-audited without retraining:

- H formal: passed; active 0.984375, positive 0.342041, neutral 0.307556.
  Audit SHA256:
  `ebc04358f99d84af75790a2da362bf2ee537d62d5052b2764212aa609282d5d2`
- HC formal: passed; active 0.984863, positive 0.414429, neutral 0.276550.
  Audit SHA256:
  `d9315c7ed94fbac3ad61711923eac9d2aa8325033b545c1d481affda4372868c`
- Both branches changed exactly the allowed 64 decoder tensors, changed zero
  forbidden tensors, retained finite metrics, and kept the public reference
  and selector frozen.

Do not rerun H or HC formal training.

## Next job: frozen fold4 evaluation

Fold4 inputs are already frozen:

- Input: `artifacts/grpo_stage29/pilot/fold4/inputs.json`
- Input SHA256:
  `b75de0ca35f5b60ff34ed93438b9f3b274eb725c3c074883f295e8dc6960d771`
- Systems: P, non-selectable historical control A3, H1--H4, HC1--HC4.
- Namespaces: 20261211 and 20261212.
- All 20 cells passed strict read-only startup preflight.

Run only:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage29_fold4_8gpu.sh
```

The fold4 job evaluates 20 cells in three waves on eight GPUs. Its final report
is `artifacts/grpo_stage29/pilot/fold4/report.json`. A nonzero exit from the
last command is an intended scientific stop when no H/HC checkpoint passes;
it is not permission to tune constants after looking at fold4.

Do not rerun `prepare_stage29_fold4.sh`: the frozen input already exists and the
preparation script intentionally refuses overwrite.
