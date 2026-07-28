# Stage32 formal 结果与 OOF 评测交接（2026-07-26）

## 结论

修正后的 Stage32 SCF formal 已成功完成；此前 `_envfix` 版本因 BC/KL 未按
`all_count` 归一化而作废，不能用于论文结果。修正版使用
`agent.config.stage32_scf_objective_revision=mean_normalized_bc_kl_v2`。

两条 SCF 和两条 DPEL control 均完成 12 epochs、576 optimizer steps，并通过
Stage32 checkpoint/provenance audit：

- 只改变 64 个允许的 `diff_decoder` 参数（每层 32 个），无 frozen tensor drift；
- frozen reference、Stage25 selector、optimizer boundary 均通过；
- decoder 梯度非零且 frozen-module 梯度为零；
- reward/advantage/log-prob 和 TensorBoard step 序列全部有限；
- SCF 修正版 BC step 均约 `-0.10`，KL 约 `1e-5`，不再出现原始版本的
  `-4` 到 `-21` 放大。

这还不是最终 PDMS 结论。必须先完成两折 OOF 的公共基线、DPEL control 和
SCF 对照，再决定是否进入四折 paper gate / NavTest。

## 冻结与审计产物

冻结清单位于：

```text
artifacts/grpo_stage32/pilot/training/DPEL/fold0/formal/checkpoints.json
artifacts/grpo_stage32/pilot/training/DPEL/fold1/formal/checkpoints.json
artifacts/grpo_stage32/pilot/training/SCF/fold0/formal/checkpoints.json
artifacts/grpo_stage32/pilot/training/SCF/fold1/formal/checkpoints.json
```

每个清单选择 `global_step=192,384,576`（对应 epoch 3、7、11），审计文件为
同目录下的 `audit.json`。其中 SCF 的 formal checkpoint 来源为：

```text
exp/stage32_pilot_SCF_fold0_formal_seed203200_normfix/2026.07.26.07.43.25/lightning_logs/version_0/checkpoints/grpo-03-192.ckpt
exp/stage32_pilot_SCF_fold0_formal_seed203200_normfix/2026.07.26.07.43.25/lightning_logs/version_0/checkpoints/grpo-07-384.ckpt
exp/stage32_pilot_SCF_fold0_formal_seed203200_normfix/2026.07.26.07.43.25/lightning_logs/version_0/checkpoints/grpo-11-576.ckpt
exp/stage32_pilot_SCF_fold1_formal_seed203201_normfix/2026.07.26.07.44.19/lightning_logs/version_0/checkpoints/grpo-03-192.ckpt
exp/stage32_pilot_SCF_fold1_formal_seed203201_normfix/2026.07.26.07.44.19/lightning_logs/version_0/checkpoints/grpo-07-384.ckpt
exp/stage32_pilot_SCF_fold1_formal_seed203201_normfix/2026.07.26.07.44.19/lightning_logs/version_0/checkpoints/grpo-11-576.ckpt
```

## OOF 评测

每个 branch/fold 使用 8 张 GPU，8 个 cell 会并行评测：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage32_pilot_eval_array_h100.sh 0  # DPEL fold0
bash ./run_stage32_pilot_eval_array_h100.sh 1  # DPEL fold1
bash ./run_stage32_pilot_eval_array_h100.sh 2  # SCF fold0
bash ./run_stage32_pilot_eval_array_h100.sh 3  # SCF fold1
```

每个 branch/fold 产生：

- `P_ns20261411.json`, `P_ns20261412.json`：88.1 public reference；
- `DPEL{192,384,576}_ns*.json` 或 `SCF{192,384,576}_ns*.json`；
- 每个 JSON 都验证 checkpoint SHA、selector/calibration SHA、noise namespace、
  token 顺序、roll schedule 和 generator domain。

评测输出目录：

```text
artifacts/grpo_stage32/pilot/eval/<DPEL|SCF>/fold{0,1}/
```

通过 OOF gate 后才允许扩展到四折正式 NavTest；在 gate 之前不要把任意单折或
单一 noise 的最好 checkpoint 宣称为最终提升。
