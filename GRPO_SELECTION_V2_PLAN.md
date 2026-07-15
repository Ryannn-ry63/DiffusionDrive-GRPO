# DiffusionDrive 分类式 GRPO 性能提升方案

## 1. 目标与约束

- 本阶段仅优化 DiffusionDrive 在 NAVSIM 上的分类式 mode-selection GRPO，不设计 WorldEngine 接入，不改为去噪级 GRPO。
- 初始模型和 reference policy 使用：
  `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model`。
- 该 checkpoint 的历史 navtest PDMS 为 `0.8487464442665378`。实施后先重跑配对基线 `B`，验收标准为 `PDMS >= B + 0.01`。
- 主实验不使用 imitation loss、trajectory L1 loss 或 reward regression loss，使性能变化可归因于 GRPO。
- 冻结感知 backbone、BEV/agent 等其他模块，按实验约定解冻整个 `diff_decoder`。
- 所有 Python 运行、测试和训练显式使用 `/root/miniconda3/envs/navsim/bin/python`，不使用当前 `algengine` 环境。

## 2. 建立可信基线与 Oracle 上限

1. 通过 Hydra 显式传入 checkpoint，不依赖 Python 默认参数。checkpoint 为空、文件不存在或核心权重缺失时直接报错。
2. 输出 missing/unexpected keys 并校验 backbone、`diff_decoder`、分类和回归分支确实来自 0.8487 checkpoint。
3. 评测阶段按 scenario token 生成稳定噪声种子，消除 Ray 顺序和随机去噪对模型比较的干扰。
4. 在固定 1024 个 trainval holdout token 上计算：
   - argmax mode reward；
   - 20 个 mode 的 oracle max reward；
   - selection regret；
   - top-1 oracle 命中率；
   - mode reward 均值、标准差和无差异场景比例。
5. 如果 oracle 相对当前 argmax 的平均提升不足 `0.015`，停止完整分类 GRPO 训练，保留诊断结果，不自动转向生成级方案。

## 3. GRPO 公式与数据流修正

### 3.1 训练/推理对齐

- 训练 rollout 与 `forward_test` 对齐：使用相同的两步 truncated-DDIM 链，并仅在最终候选轨迹和最终 `poses_cls` 上计算 reward 与 GRPO。
- current、old 和 reference policy 共享同一批感知特征、初始噪声和 timestep。
- 关闭 decoder dropout 以保证 current/old/reference log-prob 可比，但不关闭梯度。
- reference policy 始终固定在 0.8487 checkpoint；old policy 从 current policy 初始化，每 32 个 optimizer step 同步一次。

### 3.2 Group-relative advantage

对每个场景的 20 个 PDM rewards：

```text
mean = rewards.mean()
std = rewards.std(unbiased=False)
advantage = (reward - mean) / max(std, 1e-3)
```

- advantage 必须 `detach`。
- reward 标准差低于 `1e-3` 的场景不产生 policy gradient。
- 不对不同 batch 的 advantage 做全局分位数裁剪。

### 3.3 Clipped GRPO/PPO objective

```text
ratio = exp(logp_current - logp_old)
surr1 = ratio * advantage
surr2 = clamp(ratio, 0.8, 1.2) * advantage
policy_loss = -mean_batch(sum_mode(p_old * min(surr1, surr2)))
```

- 20 个 mode 已全量枚举，因此按 `p_old` 对 mode 加权，不再错误地对结构化 anchors 均匀平均。
- old policy 只用于 PPO ratio；reference policy 只用于 KL，两者不混用。
- KL 使用精确 categorical `KL(p_current || p_reference)`。
- 总损失固定为 `policy_loss + beta * KL`。

## 4. Reward 正确性与训练可观测性

- 删除跨 batch 的 `_cached_rewards` 复用，`reward_compute_interval` 固定为 1。metric cache 仍可按 token lazy-load。
- 不再将异常轨迹静默赋值 `0.5`；reward scorer 返回 validity mask，无效 mode 不参与 advantage 和 loss。
- 记录缺失 token、轨迹转换失败、模拟失败和有效 mode 比例。
- TensorBoard 增加 reward mean/std、selected/oracle reward、selection regret、policy loss、KL、entropy、ratio、clip fraction、零优势场景比例和 `diff_decoder` 梯度范数。
- validation checkpoint 按固定 holdout 的 `selected_reward` 保存，不再按 trajectory L1 loss 保存。

## 5. 实验阶段与验收

### 5.1 基础测试

- 单 batch 检查 checkpoint 加载、冻结范围、两步 rollout、reward shape、validity mask 和反向传播。
- 数值单测覆盖 advantage 零均值、正负 advantage clipping、相同策略时 KL=0，以及无效 mode 不参与 loss。
- 重跑原 checkpoint 的 navtest，得到配对基线 `B`。

### 5.2 Smoke

- 8 个 train batch、2 个 validation batch。
- 默认参数：`lr=1e-5`、`beta=0.01`、clip `0.2`、old sync 32 steps。
- 要求无 NaN/Inf、reward 有效率接近 100%、policy gradient 非零，且 KL/entropy 不塌缩。

### 5.3 三组短训

固定 5% 训练数据、3 epochs：

| 实验 | learning rate | KL beta |
|---|---:|---:|
| A | `1e-5` | `0.01` |
| B | `3e-5` | `0.01` |
| C | `1e-5` | `0.05` |

- 以 holdout selected reward 提升为主排序。
- oracle reward 相对基线下降不得超过 `0.01`，KL 不得超过 `0.05`。
- 如果三组都不能降低 selection regret，停止完整训练。

### 5.4 完整验证

- 选择短训最优的 1–2 组，在完整训练集上最多训练 5 epochs。
- 对最优 checkpoint 运行完整 navtest PDMS。
- `PDMS >= B + 0.01` 时判定阶段目标完成；仅有小幅正增益时保留结果，但不视为通过。


## 6. 实际执行记录（2026-07-14 至 2026-07-15）

### 6.1 数据与实现修正

- 全部 Python、训练和验证均显式使用
  `/root/miniconda3/envs/navsim/bin/python`。
- 原训练缓存与 `metric_cache_trainval` 只有 7,457 个目录 token 重合；
  按 navtrain/navval log split 和完整 feature/target 检查后，实际可训练 6,119 条、
  可验证 1,338 条。训练入口现已强制过滤到可评分 token。
- 修复了空 checkpoint、核心权重缺失、跨 batch reward 复用、异常 reward 填 0.5、
  训练/推理 rollout 不一致、reference/old 混用、validation 随机噪声等问题。
- current checkpoint 与固定 base reference checkpoint 已拆分。加载训练后 checkpoint
  做验证时，reference 仍固定为 0.8487 base，不再错误地复制 current 导致 KL=0。
- smoke 使用 NAVSIM GPU 实际跑通；reward valid mode/group 均为 100%，
  `diff_decoder` 梯度非零，且无 imitation loss。

### 6.2 Oracle 诊断

固定 1,024 个 validation token 的 base 结果：

| 指标 | 数值 |
|---|---:|
| selected reward | 0.744802 |
| oracle reward | 0.923022 |
| selection regret | 0.178219 |
| oracle hit rate | 0.072266 |
| valid mode fraction | 1.000000 |
| valid group fraction | 0.985352 |

oracle gap 远高于 0.015，说明候选轨迹中存在理论选择空间。

### 6.3 短训结果

128-token 严格配对 base：selected `0.820174`，oracle `0.936557`，
regret `0.116383`。

| 实验 | 设置 | 最好 selected | 对配对 base | 最好/对应 val KL | 结论 |
|---|---|---:|---:|---:|---|
| A | lr=1e-5, beta=0.01, AMP | 0.821171 | +0.000997 | KL 持续升至 0.580 | 淘汰；出现 1 次非有限梯度 |
| B | lr=3e-5, beta=0.01, FP32 | 0.810431 | -0.009743 | 0.593854 | 淘汰 |
| C | lr=1e-5, beta=0.05, FP32 | 0.818981 | -0.001193 | 0.203358 | 淘汰 |
| D | lr=3e-6, beta=0.5, FP32 | 0.823999 | +0.003825 | 0.010459 | 小 holdout 合格，进入 1024 验证 |

D 最佳 checkpoint：
`/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/grpo_short_D_fp32_lr3e6_kl05/2026.07.15.02.43.52/lightning_logs/version_0/checkpoints/grpo-01-610.ckpt`。

### 6.4 1024-token 最终筛选

| 模型 | selected | oracle | regret | 相对 base selected |
|---|---:|---:|---:|---:|
| base | 0.744802 | 0.923022 | 0.178219 | 0 |
| D best | 0.743763 | 0.923258 | 0.179495 | -0.001040 |

结论：D 在 128-token holdout 的小幅提升没有泛化到 1,024 token。按本方案的
停止门槛，不启动完整训练和 navtest PDMS；当前阶段尚未达到性能提升目标。

该结果表明瓶颈已从“公式/缓存实现错误”转为“分类概率目标与部署 argmax 的泛化不一致”。
下一轮若继续保持纯 GRPO，优先研究更稳健的探索/熵约束、固定大 holdout 早停，
以及避免低概率高 reward mode 几乎得不到梯度的问题；在这些诊断通过前不扩大训练规模。
