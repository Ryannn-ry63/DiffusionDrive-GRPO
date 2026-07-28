# Stage33 formal training result（2026-07-26）

## 1. 完成状态

fold0 和 fold1 的 chain 都完整执行了：

`audit training -> checkpoint audit -> formal training`

两个 audit 报告均通过，formal 均完成 4 个 epoch / 192 optimizer steps。

| fold | audit report | formal checkpoint | audit changed tensors |
|---|---|---|---|
| 0 | `artifacts/grpo_stage33/audit/stage33_cdc_chain_f0_fix1_audit.json` | `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage33_cdc_chain_f0_fix1_formal/2026.07.26.13.03.56/lightning_logs/version_0/checkpoints/last.ckpt` | 64 = 32(layer0)+32(layer1) |
| 1 | `artifacts/grpo_stage33/audit/stage33_cdc_chain_f1_fix1_audit.json` | `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage33_cdc_chain_f1_fix1_formal/2026.07.26.13.17.41/lightning_logs/version_0/checkpoints/last.ckpt` | 64 = 32(layer0)+32(layer1) |

每个 formal 目录还保留了 `grpo-step-48/96/144/192.ckpt`，可用于选择 checkpoint。

## 2. 训练信号

TensorBoard 末 epoch 的关键数值：

| fold | deployment credit mean | headroom credit mean | mature delta | catastrophic fraction | policy active |
|---|---:|---:|---:|---:|---:|
| 0 | +0.000085 | +0.004163 | +0.0000013 | 0.0 | 0.9427 |
| 1 | +0.001540 | +0.004969 | -0.0000463 | 0.0 | 0.9505 |

这说明 CDC smoke 目标产生了有限但有效的训练信号，且没有灾难性样本爆炸；fold1 的 deployment credit 明显强于 fold0。两 fold 的 selector switch rate 约为 0.017–0.022，仍在 Stage25 安全范围内。

## 3. 重要限制

当前 checkpoint audit 中：

`candidate_level_trace = false`

因此这批结果是 **selected common-noise CDC smoke**，不是最终 candidate-level CDC-GRPO。它可以用于检查 generator 训练边界和 PDM 评估趋势，但不能直接宣称已经完成最终方法的全部创新验证。

## 4. 下一步

先对 `step=48/96/144/192` 和 public baseline 各跑两个 common-noise namespace 的 holdout PDM，使用：

`scripts/evaluation/run_diffusiondrive_grpo_stage33_cdc_eval_cell.sh`

所有 cell 完成后，用：

`scripts/evaluation/summarize_grpo_stage33_cdc_pdm.py`

选择满足 pooled gain、mature safety、catastrophic count 和 whole-log CI 门槛的 checkpoint，再决定是否进入 navtest。
