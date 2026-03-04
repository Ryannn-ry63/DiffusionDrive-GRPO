# DiffusionDrive GRPO 训练改造说明

## 改造概述

本次改造实现了基于 Group Relative Policy Optimization (GRPO) 的轨迹规划训练方法，主要包括以下几个方面：

## 1. 可训练参数配置

### 实现位置
- `navsim/agents/diffusiondrive/transfuser_agent.py` (L57-68)

### 实现内容
- **冻结策略**：冻结除 `diff_decoder` 之外的所有模型参数
  - 冻结 backbone（图像和激光雷达编码器）
  - 冻结 BEV 特征提取
  - 冻结其他感知模块
- **可训练模块**：只训练 `diff_decoder`（即 DiT block / CustomTransformerDecoder）

```python
# 1. 冻结整个模型
self._transfuser_model.requires_grad_(False)

# 2. 只解冻 diff_decoder
self._transfuser_model._trajectory_head.diff_decoder.requires_grad_(True)
```

## 2. 参考策略（Reference Policy）

### 实现位置
- `navsim/agents/diffusiondrive/transfuser_agent.py` (L65-70)
- `navsim/agents/diffusiondrive/transfuser_model_v2.py` (L518-521)

### 实现内容
- 在加载预训练权重后，**deepcopy** `diff_decoder` 作为参考策略
- 冻结参考策略的所有参数（`requires_grad=False`）
- 设置为评估模式（`eval()`），禁用 dropout 等训练特性

```python
ref_policy = copy.deepcopy(self._transfuser_model._trajectory_head.diff_decoder)
ref_policy.requires_grad_(False)
ref_policy.eval()
self._transfuser_model._trajectory_head.set_ref_policy(ref_policy)
```

## 3. 轨迹奖励计算

### 实现位置
- `navsim/agents/diffusiondrive/transfuser_model_v2.py` 
  - `TrajectoryHead.forward_train()` (L545-596)
  - `TrajectoryHead.reward_fn()` (L689-736)

### 实现内容

#### 3.1 PDM Score 作为奖励
使用 NavSim 的 PDM (Planning-oriented Dynamic Model) 评分作为轨迹质量奖励：
- **Progress**: 任务进度（沿参考路线前进）
- **TTC (Time-to-Collision)**: 碰撞时间安全性
- **Comfort**: 驾驶舒适度（加速度、急转等）

#### 3.2 奖励计算流程
1. 从 `metric_cache` 加载场景信息（只加载 unique tokens，避免重复 IO）
2. 对每条预测轨迹调用 `pdm_score()` 计算奖励
3. 返回形状为 `[batch_size, num_modes]` 的奖励张量

```python
rewards = self.reward_fn(final_poses_reg, tokens_expanded, metric_cache)
# rewards shape: [bs, num_modes]
```

#### 3.3 双模型推理
在训练时，需要同时使用当前策略和参考策略进行推理：

```python
# 当前策略（可训练）
poses_reg_list, poses_cls_list = self.diff_decoder(...)
final_poses_cls = poses_cls_list[-1]  # [bs, num_modes]

# 参考策略（冻结）
with torch.no_grad():
    ref_poses_reg_list, ref_poses_cls_list = self.ref_policy(...)
    final_ref_poses_cls = ref_poses_cls_list[-1]  # [bs, num_modes]
```

## 4. GRPO 损失计算

### 实现位置
- `navsim/agents/diffusiondrive/transfuser_loss.py`
  - `transfuser_loss()` (L10-56)
  - `compute_grpo_loss3()` (L365-395)

### 实现内容

#### 4.1 组内相对优势（Group-Relative Advantage）
对于每个样本的 K 条轨迹，计算组内归一化的优势函数：

```python
# 1. 组内标准化
mean_r = rewards.mean(dim=1, keepdim=True)  # [bs, 1]
std_r = rewards.std(dim=1, keepdim=True)     # [bs, 1]
advantages = (rewards - mean_r) / (std_r + 1e-8)  # [bs, num_modes]
```

#### 4.2 对数概率比（Log Probability Ratio）
计算当前策略相对于参考策略的对数概率比：

```python
current_probs = F.softmax(current_poses_cls, dim=-1)  # [bs, num_modes]
ref_probs = F.softmax(ref_poses_cls, dim=-1)          # [bs, num_modes]
log_ratios = torch.log(current_probs) - torch.log(ref_probs)
```

#### 4.3 GRPO 损失
策略梯度损失，鼓励高奖励轨迹的概率增加：

```python
grpo_loss = -torch.mean(log_ratios * advantages)
```

## 5. KL 散度正则化

### 实现位置
- `navsim/agents/diffusiondrive/transfuser_model_v2.py` (L565-570)
- `navsim/agents/diffusiondrive/transfuser_loss.py` (L29-30)
- `navsim/agents/diffusiondrive/transfuser_config.py` (L88)

### 实现内容
添加 KL 散度作为正则化项，防止策略偏离参考策略过远：

```python
kl_div = F.kl_div(
    F.log_softmax(final_poses_cls, dim=-1),
    F.softmax(final_ref_poses_cls, dim=-1),
    reduction='batchmean'
)

# 总损失
total_loss = diff_loss_weight * grpo_loss + kl_loss_weight * kl_div
```

### 配置参数
- `kl_loss_weight`: 默认值 0.01（可在 `TransfuserConfig` 中调整）

## 6. 训练和验证模式

### 训练模式 (training=True)
- 计算 PDM 奖励
- 使用双模型推理（当前策略 + 参考策略）
- 计算 GRPO loss 和 KL loss
- 返回所有必需的字段用于损失计算

### 验证模式 (training=False)
- **不计算** PDM 奖励（设为 None）
- 只使用当前策略推理
- 损失函数返回零损失（不影响验证指标）

```python
# transfuser_loss.py
if "rewards" not in predictions or predictions["rewards"] is None:
    # Validation 模式
    return {
        "loss": torch.tensor(0.0, device=device),
        "grpo_loss": torch.tensor(0.0, device=device),
        "kl_loss": torch.tensor(0.0, device=device)
    }
```

## 7. 完整训练流程

### 数据流
```
Dataset (tokens) 
  ↓
AgentLightningModule (features, targets, tokens_list)
  ↓
TransfuserAgent.forward() (features, targets, tokens_list)
  ↓
V2TransfuserModel.forward() (features, targets, tokens_list)
  ↓
TrajectoryHead.forward_train()
  ├─ Current Policy (diff_decoder) → poses_cls
  ├─ Reference Policy (ref_policy) → ref_poses_cls
  └─ PDM Rewards (metric_cache + tokens) → rewards
  ↓
transfuser_loss()
  ├─ compute_grpo_loss3() → grpo_loss
  └─ KL divergence → kl_loss
  ↓
Total Loss = diff_loss_weight * grpo_loss + kl_loss_weight * kl_loss
```

### 关键配置

#### TransfuserConfig 参数
- `diff_loss_weight`: GRPO 损失权重（默认 20.0）
- `kl_loss_weight`: KL 散度权重（默认 0.01）
- `plan_anchor_path`: 轨迹锚点路径（K-means 聚类结果）

#### 训练超参数
- 轨迹模式数量: 64（`ego_fut_mode`）
- 时间步数: 8（`num_poses`）
- 时间范围: 4.0 秒
- 优势函数裁剪: 分位数裁剪（可调整）

## 8. 测试和验证

### 测试脚本
运行 `test_grpo_training.py` 来验证：

```bash
python test_grpo_training.py
```

测试内容：
1. ✓ 参数冻结正确性（只训练 diff_decoder）
2. ✓ ref_policy 正确设置（冻结 + eval 模式）
3. ✓ 前向传播完整性
4. ✓ 损失计算正确性（GRPO + KL）

## 9. 关键文件修改总结

| 文件 | 修改内容 | 说明 |
|-----|---------|------|
| `transfuser_agent.py` | 添加参数冻结和 ref_policy 创建 | 确保只训练 diff_decoder |
| `transfuser_model_v2.py` | 修改 `forward_train()`，添加 ref_policy 推理和 KL 计算 | 双模型推理 + 奖励计算 |
| `transfuser_loss.py` | 实现 `compute_grpo_loss3()` 和修改 `transfuser_loss()` | GRPO + KL 损失 |
| `transfuser_config.py` | 添加 `kl_loss_weight` 配置 | 损失权重配置 |

## 10. 注意事项

### Metric Cache
- 确保 `metric_cache_path` 正确指向缓存目录
- 路径：`/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/metric_cache`
- 如果缓存不存在，需要先运行 metric caching 脚本

### 内存优化
- 参考策略使用 `torch.no_grad()` 和 `.detach()` 避免计算图
- Unique tokens 加载避免重复读取 metric cache
- 只在训练时计算 rewards，验证时跳过

### 数值稳定性
- `compute_grpo_loss3` 使用最小干预版本
- 添加极小值保护（1e-8, 1e-12）
- 检测并清理 NaN/Inf 值

## 11. 训练命令示例

```bash
# 假设使用现有的训练脚本
python scripts/training/run_transfuser_training.sh
```

确保配置文件中：
- checkpoint_path 指向预训练权重
- plan_anchor_path 指向轨迹锚点文件
- metric_cache_path 正确配置

## 12. 预期效果

- **训练稳定性**：KL 正则化防止策略崩溃
- **计算效率**：只训练 decoder，backbone 冻结
- **奖励优化**：PDM score 直接优化规划质量
- **多模态**：保持 K=64 种轨迹选择的多样性

---

## 常见问题排查

### Q: 训练时 rewards 始终为 None
A: 检查 tokens_list 是否正确传递，metric_cache 是否存在

### Q: GRPO loss 为 NaN
A: 检查 rewards 范围，确保 PDM score 计算正常

### Q: 内存不足
A: 减少 batch_size 或 num_modes，或使用梯度累积

### Q: 验证时报错
A: 确保 forward_test 正确返回所需字段，validation 时 rewards=None

---

**完成日期**: 2026-02-14  
**版本**: v1.0  
**维护者**: GRPO Training Implementation
