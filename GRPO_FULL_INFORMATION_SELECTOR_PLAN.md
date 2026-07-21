# DiffusionDrive Full-Information Selector GRPO 计划与结果

日期：2026-07-17
目标：修复 categorical selector GRPO 对低旧概率、高 PDMS mode 梯度不足的问题，
在不使用 GT、imitation 或 supervised diffusion loss 的前提下，提高部署 selector
实际选中轨迹的 raw PDMS。

## 1. 已确认的问题

- 现有 selector objective 对每个 mode 使用 `old_policy_probability * PPO surrogate`。
  即使 20 个候选的 PDMS 全部已知，低旧概率高奖励 mode 仍被压低。
- epsilon 0.1/0.3 的 S64 更新 cosine 为 0.999993，更新差异仅为 S0.1 更新范数的
  0.37%；fixed-1024 部署 argmax 完全相同。
- selector 训练的 ratio mean 约为 1、clip fraction 为 0；现有探索平滑没有形成
  足以跨过 selector margin 的更新。
- selector reference gate 在约 89% 训练场景激活，并等价于 selection-regret gate；
  它没有解决 mode 梯度覆盖问题。
- generation scene gate 只判断 oracle 是否超过 reference selected。一旦激活，仍会
  更新低于 reference 的候选，因此暂不继续 generation/alternating。

## 2. 唯一核心改动

新增 `selection_behavior_weighting`：

- `old_policy`：保持现有 objective，作为兼容默认值；
- `uniform_valid`：所有 valid modes 使用 `1 / num_valid` 作为 PPO surrogate 的
  behavior measure。

`uniform_valid` 仍保留：

- group-relative PDMS advantage；
- current/old policy ratio；
- PPO clipping；
- fixed-reference KL；
- invalid-mode mask；
- raw PDMS 正式评价。

它属于 full-information / counterfactual categorical GRPO，不是 listwise
reward distillation；不构造 soft label，不加入 GT 或 imitation loss。

首轮不同时修改 gate、reward shaping、generator 或网络结构。full-information
behavior 已提供完整 mode 覆盖，因此 `selection_exploration_floor=0`，不再扫描 epsilon。

## 3. 评测诊断

评测 artifact 增加：

- `selected_mode`、`oracle_mode`；
- selector top-1/top-2 logit margin；
- selected/oracle mode probability；
- valid candidate raw PDMS 和 selector probability 向量。

当提供带上述字段的 selector baseline artifact 时，额外报告：

- mode switch count/rate；
- switched-token selected PDMS delta；
- beneficial/harmful/neutral switch 数量；
- selector margin 和 oracle probability 的配对变化。

先对输入 D-128 生成 fixed-256/fixed-1024 诊断 artifact。S checkpoint 的总性能
仍相对正式 base 计算；同时必须报告相对 D-128 的增量，不能把 generator 已有收益
归因于 selector。

## 4. Seed-0 训练与门控

初始化：已通过 full-navtest 的 batch-size-2 D-128 seed-0 checkpoint。

S64 唯一配置：

- selector-only，最后 classification branch 以外全部冻结；
- `selection_behavior_weighting=uniform_valid`；
- `selection_exploration_floor=0`；
- raw PDMS；reference-headroom scene gate；
- categorical reference KL 0.01；
- batch size 2，LR 1e-6，old sync 32，64 optimizer updates；
- fixed reference 始终为正式 DiffusionDrive base。

门控：

1. 真实 batch 必须满足 classification grad > 0、shared/regression grad = 0；
   current=old 时 ratio=1，invalid mode 无梯度。
2. fixed-256 若相对 D-128 无任何 mode switch、selected delta < 0、oracle 改变超过
   1e-6 或出现 NaN，则停止。
3. 只有 fixed-256 未退化才跑 fixed-1024。
4. fixed-1024 若相对 D-128 selected delta <= 0、switched-token mean delta <= 0，
   或 safety component 退化，则停止。
5. 仅当 S64 相对 D-128 为正但总 delta 尚未到 +0.005，且 mode switch rate >=1%、
   switched-token mean delta >0 时，允许从 S64 追加一次 S64，形成 S128。
6. S128 是本路线最大 selector 预算；不扫描 LR、epsilon、KL 或 shaping。
7. 只有 total fixed-1024 delta >=+0.005、paired CI 下界 >0、oracle 不变、regret
   降低且 safety-pass bucket 不退化，才运行 dev-4096。
8. dev-4096 同样达到 +0.005 后才补 seed 1/2；本轮不直接运行 full-navtest。

## 5. 测试与停止后的结论

必须测试旧行为兼容、uniform valid 归一化、all-invalid/single-valid、低概率高奖励
mode 梯度、PPO ratio/clipping、reference KL、checkpoint stage 初始化及评测 mode-switch
配对。完成后运行 focused pytest、py_compile、bash syntax 和 `git diff --check`。

如果 S64/S128 仍不能产生有益 mode switch，则结论不是“训练不够久”，而是最后
classification branch 的可学习表示或 reward 泛化不足；下一步应转向 selector
representation/calibration 诊断，而不是继续 categorical PPO 超参扫描。

## 6. 2026-07-17 执行结果

### 6.1 实现与验证

已实现 `selection_behavior_weighting={old_policy,uniform_valid}`；默认
`old_policy` 保持旧配置兼容。`uniform_valid` 只替换 PPO surrogate 的 behavior
measure，保留 group-relative raw-PDMS advantage、current/old ratio、clipping、
reference KL 和 invalid-mode mask。

评测 artifact 已增加 selected/oracle mode、selector margin、selected/oracle
probability、20 个候选 reward/probability，以及配对 mode-switch 和 safety-pass
诊断。训练 runner 支持显式传入 behavior weighting。

focused objective 测试共 `24 passed`；`py_compile`、shell syntax 和
`git diff --check` 均通过。8-update GPU smoke 的 classification gradient 非零，
shared/regression gradient 严格为零。

### 6.2 正式 S64

输入 D128：

```text
/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/
grpo_d_seed0_skl0.0_gkl0.1_eps0.001_dense0.1_mweightuniform_bs2/
2026.07.16.14.53.31/lightning_logs/version_0/checkpoints/grpo-00-128.ckpt
```

输出 S64：

```text
/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/
grpo_fullinfo_selector_s64_seed0_20260717/2026.07.17.10.06.45/
lightning_logs/version_0/checkpoints/grpo-00-64.ckpt
```

正式训练审计：

- classification grad norm：`6.373736`；
- shared/regression grad norm：严格 `0.0/0.0`；
- ratio mean：`0.998524`，clip fraction：`0.0`；
- active scene fraction：`0.890625`；
- oracle headroom / selection regret：约 `0.153095`。

### 6.3 严格配对结果

fixed-256 相对 D128：

- selected delta：`+0.0000849147`，CI `[0,+0.0002547442]`；
- oracle delta：`0.0`；
- mode switches：`1/256`（`0.390625%`）；
- 唯一切换为 beneficial，PDMS `+0.02173817`；
- safety-pass bucket delta：`+0.0000945138`。

fixed-1024 相对 D128：

- selected delta：`+0.0000212287`，CI `[0,+0.0000636860]`；
- oracle delta：`0.0`；
- mode switches：`1/1024`（`0.09765625%`）；
- 唯一切换仍为同一个 beneficial token，PDMS `+0.02173817`；
- safety components 全部不变，safety-pass bucket delta `+0.0000239407`。

Artifacts：

- `artifacts/grpo_stage3/d128_fullinfo_diagnostic_fixed256.json`；
- `artifacts/grpo_stage3/d128_fullinfo_s64_fixed256.json`；
- `artifacts/grpo_stage3/d128_fullinfo_diagnostic_fixed1024.json`；
- `artifacts/grpo_stage3/d128_fullinfo_s64_fixed1024.json`。

### 6.4 诊断与停止决定

在 fixed-1024 上，S64 相对 D128 的 selector probability L1 变化均值仅
`0.002460`，margin 绝对变化均值 `0.002243`；D128 top1-top2 margin 均值
`0.423317`、中位数 `0.293122`。current classification branch 参数相对更新范数
仅 `6.47e-5`，不足以跨过绝大多数既有决策边界。

这说明 full-information weighting 确实消除了低旧概率 mode 的显式 behavior-weight
抑制，且产生的唯一部署切换方向正确；但在当前冻结表示、S64 预算和预注册超参下，
它几乎只改变概率校准，不能形成足够多的 argmax 切换。

mode-switch rate `0.0977%` 明显低于预设的 `1%` S128 晋级线，因此按计划：

- 停止本路线，不追加 S128；
- 不扫描 LR、epsilon、KL 或 shaping；
- 不运行 dev-4096、seed 1/2 或 full-navtest；
- 结果作为“full-information categorical GRPO 仍受 selector 决策边界/表示限制”的
  负向消融保留。
