# Stage27 Phase3 H100 Audit Result

The eight-H100 audit completed on 2026-07-24 and was independently replayed
from the frozen checkpoint and TensorBoard event.

- status: `passed=true`;
- run: `stage27_generator_phase3_audit_seed0/2026.07.24.13.30.04`;
- optimizer steps: 1;
- audit SHA256:
  `b8fbaadaf766455eb1aa8bfccf2642ec9c15941d67f60fb83a86b5d41e666734`;
- checkpoint-freeze SHA256:
  `9e825e4cc4ff3cd488bee787472d7c8025b990497b1b8877141aa17a38d4eba0`;
- step-1 checkpoint SHA256:
  `41fa035a0787bf5f80210fda2efd7e966d9601f85a018622836ab384e4893973`;
- public tensors: 763;
- changed tensors: 64 permitted, 0 forbidden, 32 per decoder layer;
- frozen reference mismatches: 0;
- frozen selector mismatches: 0;
- all prohibited gradients: exact zero;
- reward, advantage, log-probability, gradient, and optimizer states: finite.

The H100 audit gate authorizes the frozen two-epoch provisional run:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage27_public88_generator_phase3_h100.sh formal
```

No epoch extension or NavTest is authorized before the fold5 Phase4 gate.
