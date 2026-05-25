# DiffusionDrive GRPO 现状交接说明

## 1. 任务背景

项目路径：

`F:\OneDrive\桌面\研0\Worldengine-diff\WorldEngine-main\DiffusionDrive-GRPO-experiment-0407-v2`

目标不是纯 selector，而是验证：

- IL 预训练之后，再做 GRPO / RL-style post-training
- 是否能缓解原始模仿学习的多模态不足
- 是否能在不破坏原有能力的前提下，让轨迹质量和 mode 选择都变好

当前现象：

- 训练后 PDMScore 明显骤降
- 这和“后训练至少不应把模型训崩”的预期不一致
- 所以先核查实现语义是否对，再谈调参

## 2. 当前已确认的事实

### 2.1 代码不是标准 GRPO 复现

核心文件：

- `navsim/agents/diffusiondrive/transfuser_agent.py`
- `navsim/agents/diffusiondrive/transfuser_model_v2.py`
- `navsim/agents/diffusiondrive/transfuser_loss.py`
- `navsim/agents/diffusiondrive/transfuser_config.py`

当前 `transfuser_loss.py` 的逻辑不是标准 GRPO surrogate。

它做的是：

- `reward_cls_loss`
  - 把 PDM reward softmax 成 target distribution
  - 用分类交叉熵训练 `final_poses_cls`
- `reward_reg_loss`
  - 用 reward 权重对 `final_poses_reg` 做 all-mode 回归
- `kl_loss`
  - 对 reference policy 做 KL 约束

这更像：

> reward-guided multi-task fine-tuning

而不是典型的：

> `ratio * advantage` 形式的 GRPO / PPO-style policy optimization

### 2.2 训练范围偏大

`transfuser_agent.py` 当前是：

- `self._transfuser_model.requires_grad_(False)`
- 再 `self._transfuser_model._trajectory_head.diff_decoder.requires_grad_(True)`

这意味着整个 `diff_decoder` 都在训练，包括：

- cross attention
- agent/ego attention
- FFN
- time modulation
- `plan_cls_branch`
- `plan_reg_branch`

所以它是 post-training 形态，但更新面比较大。

### 2.3 当前实现和 earlier discussion 里的 selector-only 不是一回事

之前讨论的 selector-only 更接近：

- 只优化 mode selection
- 不让轨迹生成主体漂移

但现在的项目目标其实不是纯 selector，而是：

- 希望 RL 能推动生成器变好
- 希望突破 IL 的多模态不足
- 希望轨迹坐标本身也有改进

所以不能简单冻结到只剩分类头；但也不能让训练目标和训练范围都过于发散。

### 2.4 baseline 目录有更接近 GRPO 的实现

对照目录：

`DiffusionDrive-GRPO-experiment-newbaseline`

其中 `transfuser_loss.py` 里更接近：

- `compute_grpo_loss_clipped(...)`
- `policy_loss_weight`
- `kl_loss_weight`

这说明当前实验目录的实现和标准 GRPO 已有明显语义差异。

## 3. 当前最值得怀疑的几个问题

这些只是优先怀疑点，不是最终结论。

### 3.1 loss 语义可能不对

当前 loss 把 reward 直接变成分类 target，还加了 reward-weighted regression。

风险是：

- 模型学到的是“reward-guided imitation”
- 但不是“真正沿 reward 方向优化 policy”

尤其 `reward_reg_loss` 这一项，可能和 PDM 目标不完全一致。

### 3.2 训练对象可能太大

即使 backbone 冻结了，`diff_decoder` 整体仍然会被更新。

如果 reward 噪声较大，或者 cache / token / reward 统计不稳定，decoder 主体可能被带偏，导致：

- 分类头更尖锐
- 轨迹整体质量下降
- PDMScore 崩掉

### 3.3 reward 计算链路可能有隐藏异常

`transfuser_model_v2.py` 里 reward 计算使用了：

- `tokens_list`
- `metric_cache_loader`
- lazy cache
- disk cache fallback
- `try/except Exception: pass`

这类写法很容易把错误吞掉。

需要确认：

- 是否有大量 reward fallback
- 是否很多轨迹模式拿到默认 reward
- 是否 reward 全部接近常数
- 是否某些 batch 的 reward 其实无效但仍然在训练

### 3.4 评估和训练输出是否一致

需要确认推理阶段最终选轨迹的逻辑：

- 是否仍然用 `poses_cls.argmax()`
- `trajectory` 字段是不是实际拿去算 PDMScore 的那个输出
- 训练和评估是否使用一致的 checkpoint / config

## 4. 建议不要直接照做，而是先验证的事情

### 4.1 先判断当前实现到底是不是 GRPO

先回答：

- 当前 loss 是否真的是 policy gradient / GRPO surrogate
- 是否存在 `old policy` / `ref policy` / `ratio` / `advantage` 的正确对应
- 还是现在只是 reward-supervised fine-tuning

如果它不是 GRPO，那后面所有“GRPO 训练效果不好”的讨论都要重命名。

### 4.2 再判断崩盘原因属于哪一类

优先区分三类：

1. loss 语义不对
2. 训练范围过大
3. reward / metric cache 链路有问题

不要一开始就默认是超参数。

### 4.3 再判断“多模态不足”是否真的被改善

建议做一个最小验证：

- pretrained checkpoint 的 PDMScore
- 当前训练 checkpoint 的 PDMScore
- oracle best-of-modes PDMScore

这三者能帮助判断：

- 是 selector 变差了
- 还是候选轨迹整体变差了
- 还是 reward 计算出了问题

## 5. 需要重点阅读的文件

优先级建议：

1. `navsim/agents/diffusiondrive/transfuser_loss.py`
2. `navsim/agents/diffusiondrive/transfuser_agent.py`
3. `navsim/agents/diffusiondrive/transfuser_model_v2.py`
4. `navsim/agents/diffusiondrive/transfuser_config.py`

然后对比：

- `DiffusionDrive-GRPO-experiment-newbaseline/navsim/agents/diffusiondrive/transfuser_loss.py`
- `DiffusionDrive-GRPO-experiment-newbaseline/navsim/agents/diffusiondrive/transfuser_config.py`
- `DiffusionDrive-GRPO-experiment-newbaseline/navsim/agents/diffusiondrive/transfuser_agent.py`

## 6. 可以自己判断的几个问题

1. 当前实现是不是真的 GRPO
2. 还是更像 reward-guided fine-tuning
3. 当前 PDMScore 崩盘的主因更像哪类
   - loss 定义
   - 训练范围
   - reward 链路
   - checkpoint / eval mismatch
4. 如果要保留“后训练 + 改善多模态”的目标，最小改动应该是什么
5. 是否需要先把训练目标语义拉回标准 GRPO，再谈调参

## 7. 当前倾向，但不是最终结论

- 这套代码不是严格 GRPO 复现
- PDMScore 降得太厉害，说明实现层面大概率有逻辑偏差
- 但这个偏差不一定意味着“后训练”方向错了
- 更可能是目标函数和训练对象没有对齐

## 8. 最终希望得到的结论

请明确回答：

- 当前实现是否符合 GRPO 语义
- 当前 PDMScore 降低的最可能原因
- 需要怎么改，才能既保留 post-training，又不破坏原模型能力
- 如果要继续做实验，建议先改哪一处
