# DiffusionDrive GRPO + PDMS Preference Ranking 计划

日期：2026-07-17
目标：在不修改 generator、不新增 selector 网络、不使用 GT/imitation 的前提下，
用全部 20 个候选的 raw PDMS preference 直接训练现有 classification head，冲击
full-navtest absolute PDMS `+0.005`。

## 1. 动机与核心假设

full-information selector GRPO S64 已验证 objective、冻结边界和梯度方向正确，但
fixed-1024 仅产生 `1/1024` 次部署 mode switch，相对 D128 只提升
`+0.00002123`。selector probability L1 变化均值 `0.00246`，远小于 D128
top1/top2 margin 中位数 `0.29312`。

fixed-1024 中，只在当前 selector top-2 候选内做 oracle 重排就有约 `+0.01729`
的平均上限，因此问题不是候选不足，而是现有 GRPO 无法可靠地跨越 selector 决策
边界。下一步加入 raw-PDMS pairwise preference ranking，同时保留 GRPO 和 reference
KL。

## 2. 唯一机制改动

对每个场景的 valid ordered pair `(i,j)`：

- 只保留 `r_i-r_j > 0.01`；
- pair weight 为 `clamp((r_i-r_j-0.01)/0.10, 0, 1)`；
- pair loss 为 `softplus(0.20-(logit_i-logit_j))`；
- 先按每场景 pair-weight sum 归一化，再用现有 reference-headroom scene weight；
- reward 全程 detach，正式训练只使用 raw PDMS。

总目标：

```text
L = policy_weight * L_GRPO
  + rank_weight * L_pairwise_rank
  + 0.01 * KL(current || reference)
```

generator、shared decoder、regression branch 和 perception 全部冻结，只更新现有最后
classification head。

## 3. 接口与兼容性

新增配置，默认关闭 ranking，保持旧 checkpoint/config 行为：

- `selection_rank_loss_weight=0.0`；
- `selection_rank_reward_gap=0.01`；
- `selection_rank_reward_scale=0.10`；
- `selection_rank_logit_margin=0.20`。

新增日志：rank loss、active-pair scene fraction、有效 pair count/weight、pair reward
gap，以及 GRPO/ranking classification gradient 诊断。训练 runner 必须显式记录 LR、
policy/rank weight 和输入 checkpoint。

## 4. Seed-0 消融

三组都从同一 D128 seed-0 checkpoint 初始化，seed、sampler、batch size 2、S64、
LR `1e-5`、uniform-valid GRPO、reference-headroom gate、KL `0.01`、old sync 32
完全一致：

| 实验 | policy weight | rank weight |
|---|---:|---:|
| GRPO-LR control | 1 | 0 |
| Rank-only ablation | 0 | 1 |
| GRPO+Rank primary | 1 | 1 |

三组先跑 fixed-256。晋级 fixed-1024 必须满足 selected delta `>0`、oracle delta
绝对值 `<1e-6`、mode-switch rate `>=1%`、beneficial switch 总收益大于 harmful
switch 总损失，且 collision/drivable/TTC 不退化。

Hybrid 的 fixed-1024 主要门槛：相对 D128 至少 `+0.0031`，使总提升达到约
`+0.005`；相对 GRPO-LR control 为正；paired CI 下界 `>0`；oracle 不变、regret
降低、safety-pass bucket 不退化。

只有 hybrid 相对 D128 `>=+0.001`、CI 下界 `>0`、mode-switch `>=5%`，但总提升
尚未达到 `+0.005` 时，允许唯一一次 S64 continuation，形成最大 S128。本阶段不扫
更多 LR、rank weight、margin、reward gap、KL、epsilon 或 shaping。

## 5. 后续门控与停止条件

fixed-1024 总提升达到 `+0.005` 后才运行独立 dev-4096；dev 同样通过后补 seed
1/2，最终才运行 full-navtest。正式成功标准沿用三-seed mean `>=+0.005`、至少两个
单-seed CI 下界 `>0`、seed-stratified/token-clustered CI 下界 `>0`，且
collision/drivable/TTC 不退化。

必须测试 invalid/all-invalid/single-valid、reward tie、gap threshold、pair
normalization、permutation invariance、正确梯度方向和 rank-weight-zero 兼容性；完成
focused pytest、py_compile、bash syntax、`git diff --check` 和真实 batch 冻结边界
审计。

若三组在 fixed-1024 都无法产生足够的有益 mode switch，则结论为冻结轨迹特征不足
以泛化 PDMS preference；下一阶段才考虑 selector adapter，不通过延长 schedule 或
事后超参扫描解释失败。

## 6. 2026-07-17 执行结果

### 6.1 实现与验证

已实现上述 pairwise objective、四个兼容配置、训练指标和 runner 参数。ranking
训练阶段强制读取 `raw_rewards`；仅在 validation rollout 且
`grpo_reward_mode=pdms` 时允许使用 `rewards` 作为 raw-PDMS fallback，防止 shaping
被误接入。

focused tests 共 `34 passed`；Python/Bash syntax 和 `git diff --check` 通过。
三组 8-update smoke 的 classification grad norm 分别为：

- GRPO-only：`5.600057`；
- rank-only：`7.560764`；
- hybrid：`9.848840`。

三组 shared/regression grad norm 均严格为 `0`。ranking 每个场景平均约 172 个
有效 preference pairs，真实 reward 信号密集且梯度生效。

### 6.2 正式 S64 checkpoints

```text
/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/
grpo_rank_ablation_grpo_lr1e5_s64_seed0_20260717/2026.07.17.11.42.34/
lightning_logs/version_0/checkpoints/grpo-00-64.ckpt

/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/
grpo_rank_ablation_rankonly_lr1e5_s64_seed0_20260717/2026.07.17.11.45.01/
lightning_logs/version_0/checkpoints/grpo-00-64.ckpt

/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/
grpo_rank_ablation_hybrid_lr1e5_s64_seed0_20260717/2026.07.17.11.47.26/
lightning_logs/version_0/checkpoints/grpo-00-64.ckpt
```

正式训练 classification grad norm 为 `6.249191 / 7.556180 / 11.565546`；三组
shared/regression 仍严格为 `0`，active scene fraction 均为 `0.890625`。

### 6.3 Fixed-256

| 配置 | vs D128 | mode switches | beneficial / harmful | oracle delta |
|---|---:|---:|---:|---:|
| GRPO-only | `+0.000125281` | `3/256` | `2/0` | `0` |
| rank-only | `+0.000265961` | `7/256` | `6/0` | `0` |
| hybrid | `+0.000260014` | `6/256` | `5/0` | `0` |

三组 safety components 均未退化，全部按预注册规则晋级 fixed-1024。

### 6.4 Fixed-1024 与停止决定

| 配置 | vs D128 | vs 正式 base | mode switches | beneficial / harmful |
|---|---:|---:|---:|---:|
| GRPO-only | `+0.000117607` | `+0.002021914` | `12/1024` | `10/1` |
| rank-only | `-0.000473159` | `+0.001431148` | `24/1024` | `21/2` |
| hybrid | `-0.000468153` | `+0.001436155` | `24/1024` | `20/2` |

所有配置 oracle delta 均严格为 `0`。GRPO-only 的 relative-base CI 为
`[-0.001017,+0.005214]`；rank-only/hybrid CI 下界也小于 0，均未达到 `+0.005`。

rank-only 与 hybrid 的主要失败来自同一个 token `b0ad1a8107ad54dc`：D128 选择
mode 6，PDMS `0.695360`、collision `1`；两者都切换到 mode 13，PDMS `0`、
collision `0`，单 token delta `-0.695360`。另一个 harmful token
`582e330653095d1b` 仅因 progress 轻微降低而退化 `-0.001623`。

这说明 ranking 把 mode-switch rate 从 GRPO 的约 `1.17%` 提高到 `2.34%`，且绝大
多数切换方向正确，但冻结特征上的 held-out 泛化会产生极少量灾难性 safety switch；
现有 GRPO 项没有抑制该共同失效模式。按照预注册门控：

- hybrid relative D 小于 0，停止 S128；
- collision 均值退化 `-0.0009765625`，不运行 dev-4096；
- 不补 seeds、不运行 full-navtest；
- 不事后扫描 LR、rank weight、margin、gap、KL 或 shaping。

本路线作为清晰消融保留：直接 reward ranking 能增加有益选择，但在没有显式
safety-conservative 机制或更可靠 selector representation 时，平均收益会被极少量
灾难切换支配。
