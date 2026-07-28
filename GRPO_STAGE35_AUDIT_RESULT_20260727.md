# Stage35 audit result

Date: 2026-07-27

## Initial failure

Both 8-H100 audit trainings completed one optimizer step and produced
valid checkpoints. Post-training audit initially failed because Hydra
preserved `0.10` in `overrides.yaml`, while the frozen contract rendered
the numerically identical value as `0.1`. The original audit compared
the two strings instead of their parsed numeric values.

The audit comparison now:

- accepts JSON-equivalent numeric, Boolean, and list renderings;
- still rejects changed numeric values, paths, hashes, and symbolic
  strings.

The launcher also uses canonical `0.1` and `0.9` renderings for future
runs. Dedicated regression tests cover equivalent and genuinely drifted
overrides.

## Recovered fold0 audit

- checkpoint SHA256:
  `08cd475b0de5deaf944242affcedd6ff58bf93612b495a412bebe3707a0a3652`
- changed allowed tensors: `64`
- changed forbidden tensors: `0`
- layer split: `32 + 32`
- counterfactual nonzero fraction: `0.3125`
- required nonzero fraction: `0.10`
- non-selected active-mode influence scene fraction: `1.0`
- required influence scene fraction: `0.25`
- result: `PASS`

Artifact:

```text
artifacts/grpo_stage35/pilot/training/NCD/fold0/audit/audit.json
```

## Recovered fold1 audit

- checkpoint SHA256:
  `5382a21e68d49cd0e44d86016819f8cb8a9af3a6e1bfc43674f05f5ceef58b64`
- changed allowed tensors: `64`
- changed forbidden tensors: `0`
- layer split: `32 + 32`
- counterfactual nonzero fraction: `0.50`
- required nonzero fraction: `0.10`
- non-selected active-mode influence scene fraction: `1.0`
- required influence scene fraction: `0.25`
- result: `PASS`

Artifact:

```text
artifacts/grpo_stage35/pilot/training/NCD/fold1/audit/audit.json
```

Both audits additionally passed finite reward/log-probability checks,
active-gradient checks, zero frozen-gradient checks, exact optimizer
boundary checks, bitwise public-reference and selector freezes, and
exact `320 sampled / 64 replayed / 224 counterfactual selector` trace
accounting.

## Formal handoff

Both formal experiment paths are unused. The original commands now
detect and reuse the frozen PASS audit JSON, skip audit training, and
start formal directly:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage35_nested_counterfactual_h100.sh 0
```

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage35_nested_counterfactual_h100.sh 1
```

The reuse-to-formal path and both fold-specific Hydra configurations
were executed in preflight-only mode and passed.
