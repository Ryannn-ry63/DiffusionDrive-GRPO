# Stage39 H100 执行手册

本手册只描述执行顺序；冻结研究设计仍以
`GRPO_STAGE39_40_NON_DESTRUCTIVE_CHALLENGER_PLAN_20260729.md` 为准。
所有命令均使用固定仓库绝对路径，并在真正启动 GPU 前按 `AGENTS.md`
只关闭 `gpu-occupy`。

## 0. Phase0：不训练，验证额外候选预算

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage39_phase0_h100.sh
```

通过标志：

```text
artifacts/grpo_stage39/phase0/audit.json
```

若该 gate 失败，不提交后续训练；先检查其中的
`safe_positive_fraction` 与 `strict_safe_oracle_gain`。

### 2026-07-29 Phase0 consistency repair

第一次 Phase0 的四个 cell 均完成，target/headroom gate 通过，但
`public20_bitwise_exact=false`。离线逐 token 比较确认：P20 与 P40 前 20 条
trajectory/logit 完全一致，差异来自 PDM scorer 会用整个 proposal 集合的
最大 raw progress 归一化 progress component；把 40 条一起评分会反向改变前 20 条
的 reward/component。Stage39 evaluation 现将
public20 与 extra20 各自按独立 20-mode bank 评分，再拼接结果；这保持候选集合
不变，同时使 public20 成为严格反事实 control。

旧的失败产物已可恢复地隔离到：

```text
artifacts/grpo_stage39/phase0/repair_20260729T080650Z/
```

当前四个 output 已清空，直接重跑本节开头的普通 Phase0 命令即可。以后若再次
留下 `passed=false` 的 audit，可显式使用：

```bash
STAGE39_PHASE0_REPAIR=1 bash ./run_stage39_phase0_h100.sh
```

该模式只会把失败 audit 与四个 Phase0 control JSON 移入带时间戳的隔离目录，
不会删除或覆盖训练 checkpoint。

## 1. Phase1+2：三个独立 pilot 分支

三个任务各自需要一台 `8×H100` 实例，可同时排队。若 Phase0 尚未完成，
脚本会等待通过的 audit，不会提前开始训练。

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage39_pilot_train_h100.sh BC
```

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage39_pilot_train_h100.sh STD
```

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage39_pilot_train_h100.sh SET
```

每个任务先完成一个真实 optimizer-step audit，再在 fold2 训练到固定
`48/96/192` steps。三个分支分别写入：

```text
artifacts/grpo_stage39/pilot/training/{BC,STD,SET}/fold2/checkpoints.json
```

## 2. Pilot 评测与唯一全局 step

三个训练任务成功后，运行一个 `8×H100` 评测任务：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage39_pilot_eval_h100.sh
```

该任务复用 Phase0 的 Public20/Public40 controls，评测其余 18 个 cell，
并只选择一次全局 step：

```text
artifacts/grpo_stage39/pilot/summary.json
artifacts/grpo_stage39/pilot/selection.json
```

## 3. Formal OOF 训练

`selection.json` 通过后，共有 `3 branches × 3 folds = 9` 个独立任务；
每个任务需要 `8×H100`，可按资源并行排队：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage39_formal_train_h100.sh BC 0
bash ./run_stage39_formal_train_h100.sh BC 1
bash ./run_stage39_formal_train_h100.sh BC 3
bash ./run_stage39_formal_train_h100.sh STD 0
bash ./run_stage39_formal_train_h100.sh STD 1
bash ./run_stage39_formal_train_h100.sh STD 3
bash ./run_stage39_formal_train_h100.sh SET 0
bash ./run_stage39_formal_train_h100.sh SET 1
bash ./run_stage39_formal_train_h100.sh SET 3
```

每个任务严格使用 pilot 的同一个 selected step，不允许 formal 重选。

## 4. Formal OOF 评测和硬 gate

三个 fold 各运行一个 `8×H100` 任务；它会等待本 fold 的三分支 checkpoint：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage39_formal_eval_h100.sh 0
```

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage39_formal_eval_h100.sh 1
```

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage39_formal_eval_h100.sh 3
```

三个任务成功后，在任意 CPU 环境汇总：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash scripts/evaluation/finalize_grpo_stage39.sh
```

唯一进入 Stage40 的凭证：

```text
artifacts/grpo_stage39/formal/gate.json
```

只有其中 `passed=true` 才训练 Stage40 selector；否则按冻结计划停止或仅报告
sampling/GRPO ablation，不进入 NavTest performance claim。
