# GRPO 中 cls logp 与 denoising logp 的语义整理

## 0. 汇报结论

当前代码的 RLFT 更准确地说是 **score-head / mode-selection RLFT**：

- 模型先输出一组候选轨迹，例如 64 条 `poses_reg`。
- `poses_cls` 给这 64 条候选轨迹打分。
- PDM reward 给每条候选轨迹算质量分。
- GRPO/PG loss 会让高 reward 轨迹对应的 `poses_cls` 概率变高，低 reward 轨迹对应的概率变低。

所以它最直接提升的是：

> 模型从已有候选轨迹里选出高质量轨迹的能力。

它可能提升最终输出质量，因为推理时通常会用 `poses_cls.argmax()` 选最终轨迹。但它不等价于“直接提升每条候选轨迹的坐标生成质量”。如果候选轨迹集合里本来就有好轨迹，这个方法会有效；如果 64 条候选轨迹整体都不好，单靠 `cls logp` 很难稳定地教模型“轨迹坐标应该往哪里改”。

DiffusionDriveV2 的思路更接近 **generator-level RLFT**：它把 diffusion 每一步 denoising transition 当成 policy action，用每一步的 Gaussian likelihood 计算 logp。这样 reward 可以更直接地优化“轨迹是怎么被生成出来的”，而不只是优化“生成完以后选哪条”。

---

## 1. 当前代码的微调流程

当前实现分成：

1. 先用 DiffusionDrive 原始训练流程训练出一个 pretrained 模型。
2. GRPO 代码加载这个 pretrained。
3. 冻结大部分网络，只解冻 `diff_decoder`。
4. deepcopy 一份 frozen `diff_decoder` 作为 `ref_policy`。
5. current policy 和 ref policy 都 forward，得到 current/ref 的 `poses_cls`。
6. 对 current policy 输出的每条候选轨迹算 PDM reward。
7. 用 group advantage 和 `poses_cls` 的 logp 做 RL loss。

当前代码里比较关键的实现位置：

- `DiffusionDrive-GRPO-main/navsim/agents/diffusiondrive/transfuser_agent.py`
  - 加载 pretrained。
  - `self._transfuser_model.requires_grad_(False)` 冻结整体模型。
  - `self._transfuser_model._trajectory_head.diff_decoder.requires_grad_(True)` 只训练 `diff_decoder`。
  - `copy.deepcopy(...diff_decoder)` 得到 frozen `ref_policy`。

- `DiffusionDrive-GRPO-main/navsim/agents/diffusiondrive/transfuser_model_v2.py`
  - current `diff_decoder` forward 得到 `final_poses_reg` 和 `final_poses_cls`。
  - ref `diff_decoder` forward 得到 `final_ref_poses_cls`。
  - `reward_fn(final_poses_reg, ...)` 对每条候选轨迹算 PDM reward。

- `DiffusionDrive-GRPO-main/navsim/agents/diffusiondrive/transfuser_loss.py`
  - 根据 reward 做组内标准化 advantage。
  - 根据 current/ref `poses_cls` 算 log ratio。
  - 用 `log_ratio * advantage` 形成 policy loss。

---

## 2. poses_cls 作为 logp 的语义

当前模型每个样本会输出多条候选轨迹：

```text
poses_reg: [batch_size, num_modes, future_steps, 3]
poses_cls: [batch_size, num_modes]
```

如果把 action 定义成：

```text
action = 选择第 i 条 mode 轨迹
```

那么可以把 `poses_cls` 看作离散 policy 的 logits：

```python
logp = F.log_softmax(poses_cls, dim=-1)
```

此时：

```python
logp[:, i]
```

表示模型选择第 `i` 条候选轨迹的 log probability。

这件事在语义上是成立的，但要注意：pretrained 里的 `cls` 原本不是 RL 训练出来的 policy。它更可能来自 imitation learning 中的 mode classification / trajectory scoring 监督，含义更接近：

```text
哪个 mode 更接近 GT / 更像训练数据里的目标轨迹。
```

进入 RLFT 后，我们重新赋予它一个新的解释：

```text
哪个 mode 更值得被最终选中，因为它的 PDM reward 更高。
```

所以 `cls` 不是完全没依据，但它的语义发生了迁移：

```text
pretrained 阶段: GT-nearest / imitation selector
RLFT 阶段: reward-preferred / PDM selector
```

这也是为什么它可以作为一个冷启动，但严格说不是天然的、完整的 trajectory generation policy。

---

## 3. 当前 GRPO 组内竞争到底优化了什么

假设某个样本输出 64 条轨迹：

```text
traj_1, traj_2, ..., traj_64
```

PDM reward 分别是：

```text
r_1, r_2, ..., r_64
```

组内标准化后得到 advantage：

```text
A_i = (r_i - mean(r)) / std(r)
```

如果第 10 条轨迹 reward 高，那么：

```text
A_10 > 0
```

如果第 23 条轨迹 reward 低，那么：

```text
A_23 < 0
```

用 policy-gradient 风格写，loss 可以理解成：

```python
loss = -mean(logp_current_i * A_i)
```

那么优化方向是：

- 高 reward 的 mode：增大它的 `logp`，也就是增大对应 `poses_cls`。
- 低 reward 的 mode：减小它的 `logp`，也就是压低对应 `poses_cls`。

因此，“GRPO 组内竞争，好的轨迹会更好”在当前代码里更准确的说法是：

> 组内 reward 更高的轨迹，会在 score head 上获得更高选择概率。

这直接训练的是 **mode 排序 / mode 选择能力**。

---

## 4. 梯度路径：为什么主要提升选择能力

当前代码的 reward 是由轨迹坐标算出来的：

```text
reward_i = PDM(poses_reg_i)
```

但是 policy loss 使用的是：

```text
logp_i = log_softmax(poses_cls)_i
```

所以核心计算关系是：

```text
poses_reg_i  -> PDM reward_i -> advantage_i
poses_cls_i  -> logp_i
loss         = - logp_i * advantage_i
```

这里最关键的是：PDM reward 一般是不可微的。在当前代码中，轨迹会经过 `detach()`、CPU、numpy、PDM scorer 等外部计算。这样 reward/advantage 对神经网络来说更像一个常数权重。

因此反向传播时，主要梯度路径是：

```text
loss
  -> logp_current
  -> poses_cls
  -> cls branch
  -> diff_decoder 共享特征
```

而不是：

```text
loss
  -> PDM reward
  -> poses_reg 的 x/y/heading 坐标
  -> reg branch
```

也就是说，当前 loss 会明确告诉模型：

```text
这 64 条候选轨迹里，第 i 条的 reward 更高，你以后应该给它更高分。
```

但它不会直接告诉模型：

```text
第 i 条轨迹的第 3 个点 x 应该往左移 0.2m；
第 5 个点 heading 应该减小 0.1rad；
这样 PDM 才会变高。
```

原因是 PDM reward 不是一个可微分的坐标级 loss。它只能作为 advantage 权重调节 `poses_cls` 的概率。

---

## 5. 为什么轨迹质量仍然可能提升

虽然当前方法直接优化的是选择能力，但最终输出质量仍然可能提升，原因有两个。

### 5.1 推理阶段依赖 score head 选最终轨迹

如果推理时最终轨迹是：

```python
mode_idx = poses_cls.argmax(dim=-1)
best_traj = poses_reg[mode_idx]
```

那么 score head 排序变准以后，最终输出自然会变好。

例如 RLFT 前：

```text
64 条候选里第 10 条 PDM 最高，但 cls 最高的是第 3 条。
最终输出第 3 条，质量一般。
```

RLFT 后：

```text
第 10 条因为 reward 高，cls 被训练得更高。
最终输出第 10 条，质量提升。
```

这种提升是真实的，但它来自：

```text
选得更准。
```

不是直接来自：

```text
每条候选轨迹本身都生成得更好。
```

### 5.2 diff_decoder 有共享特征，可能间接影响 reg

当前只解冻 `diff_decoder`，而 `poses_cls` 和 `poses_reg` 都来自 `diff_decoder` 内部的轨迹特征。

简化看，结构类似：

```text
diff_decoder shared feature
  -> cls branch -> poses_cls
  -> reg branch -> poses_reg
```

当 loss 从 `poses_cls` 回传时，它会更新：

- cls branch 的参数。
- cls branch 前面的 shared decoder feature。
- 产生 shared feature 的 DiT / attention / FFN 参数。

因为 reg branch 也使用这些 shared feature，所以 shared feature 改变后，`poses_reg` 也可能被间接改变。

但这个影响不够直接，因为监督信号并没有明确作用在 `poses_reg` 坐标上。它更像是：

```text
为了让 score head 更容易识别高 reward mode，shared feature 被调整；
shared feature 的变化可能顺带改变 reg 输出。
```

所以可以说：

> 当前方法对轨迹生成质量可能有间接影响，但最主要、最稳定的收益应该来自最终轨迹选择能力提升。


> 当前实现可以作为 mode-selection RLFT baseline。它确实可能提升最终输出 PDM，因为它让模型更倾向选组内高 reward 轨迹。但如果目标是系统性提升候选轨迹的生成质量，就需要把 logp 放到轨迹生成过程本身，比如 diffusion denoising step。

---

## 7. DiffusionDriveV2：为什么 denoising logp 更直接

DiffusionDriveV2 的核心区别是：它不是等轨迹全部生成完以后，用 score head 给 64 条轨迹打分；而是把 diffusion 的每一步去噪过程当成 policy。

普通 diffusion denoising 可以抽象为：

```text
x_T -> x_{T-1} -> ... -> x_1 -> x_0
```

其中：

```text
x_t: 第 t 步的 noisy trajectory
x_0: 最终 clean trajectory
```

每一步模型根据当前 noisy trajectory 和场景条件，预测一个分布：

```text
p_theta(x_{t-1} | x_t, condition)
```

通常这个分布可以写成 Gaussian：

```text
x_{t-1} ~ Normal(mu_theta(x_t, condition, t), sigma_t)
```

于是这一步的 logp 可以解析计算：

```text
logp_t = log Normal(x_{t-1}; mu_theta, sigma_t)
```

整条 denoising path 的 logp 是：

```text
logp_total = sum_t logp_t
```

最后得到的 `x_0` 拿去算 PDM reward：

```text
reward = PDM(x_0)
```

然后用 policy gradient：

```text
loss = - reward_advantage * logp_total
```

这样优化的含义就是：

```text
如果某条 denoising path 最终生成了高 reward 轨迹，
就提高这条生成路径的概率；
如果某条 denoising path 最终生成了低 reward 轨迹，
就降低这条生成路径的概率。
```

---

## 8. 为什么 V2 更能提升 reg / trajectory generation

当前 `cls logp` 方案中，action 是：

```text
选择哪个 mode
```

V2 denoising logp 方案中，action 是：

```text
每一步从 x_t 生成 x_{t-1} 的采样动作
```

这两个 action 定义完全不同。

当前方案：

```text
state = scene feature
action = mode index
policy = softmax(poses_cls)
reward = PDM(poses_reg_i)
```

优化重点：

```text
哪个 mode 应该被选中。
```

V2 方案：

```text
state = scene feature + current noisy trajectory x_t + timestep t
action = next denoised trajectory sample x_{t-1}
policy = Gaussian distribution predicted by diffusion decoder
reward = PDM(final x_0)
```

优化重点：

```text
模型应该怎样一步步去噪，才能生成高 reward 轨迹。
```

这就是为什么 V2 对 `reg / trajectory generation` 更直接。

它的梯度路径可以理解成：

```text
loss
  -> logp_t
  -> Gaussian mean mu_theta
  -> diffusion decoder
  -> denoising trajectory generation
```

而 `mu_theta` 本身就是模型每一步用来生成轨迹坐标的核心输出。所以 reward 高低会直接影响：

```text
模型以后更倾向生成什么样的 x_{t-1}
```

进一步影响最终：

```text
x_0 / poses_reg 的轨迹质量
```

这比当前 `poses_cls` 方案更直接。

---

## 9. 两种方案的对比

| 对比项 | 当前 cls logp 方案 | DiffusionDriveV2 denoising logp 方案 |
|---|---|---|
| action 定义 | 选择第几个 mode | 每一步去噪采样 `x_t -> x_{t-1}` |
| policy 分布 | `softmax(poses_cls)` | Gaussian denoising transition |
| logp 来源 | score head logits | diffusion step likelihood |
| reward 来源 | 每条候选轨迹的 PDM | 最终生成轨迹的 PDM |
| 直接优化对象 | mode ranking / trajectory selection | denoising generator / trajectory generation |
| 对最终输出质量 | 可以提升，依赖候选集合中已有好轨迹 | 更直接提升生成高 reward 轨迹的概率 |
| 实现复杂度 | 较低 | 较高，需要记录 denoising chain、mean、sigma、old logp |

---

## 10. 当前方案的合理定位

当前方案不是完全错误，而是需要明确定位：

```text
它是一个基于 score head 的 mode-selection RLFT baseline。
```

它适合验证：

```text
如果 DiffusionDrive 已经能生成多条合理候选轨迹，
那么用 PDM reward 做组内竞争，
能否让模型更稳定地选出其中 PDM 更高的轨迹。
```

它不完全适合证明：

```text
RLFT 已经直接优化了 diffusion generator 的轨迹坐标生成质量。
```

如果后续想更接近 DiffusionDriveV2，需要考虑：

1. 在 denoising rollout 中记录每一步 `x_t`、`x_{t-1}`。
2. 记录模型预测的 Gaussian mean / variance。
3. 计算每一步 transition logp。
4. 保存或构造 old logp。
5. 用最终 PDM reward 对整条 denoising chain 做 policy gradient / GRPO。
6. ref policy 主要用于 KL regularization，而不是简单替代 old policy。

---


建议重点问下面几个问题。

### 问题 1：当前目标到底是 selection 还是 generation？

> 我现在的实现是把 64 条候选轨迹的 mode selection 当作 action，用 `log_softmax(poses_cls)` 做 logp。这样主要优化的是 score head 选择高 PDM mode 的能力。这个目标是否符合我们这阶段想做的 RLFT？

### 问题 2：pretrained cls 的语义是否足够？

> 原版 DiffusionDrive pretrained 里的 `cls` 更像 imitation 阶段的 mode score / GT-nearest selector。我现在把它解释成 RL policy `pi(mode|state)`，这个语义是否可以接受？

### 问题 3：好的轨迹会更好，该怎么理解？

> 当前代码里 reward 是由 `poses_reg` 算的，但 loss 的 logp 来自 `poses_cls`。所以高 reward 轨迹会被赋予更高选择概率，但轨迹坐标本身没有被直接可微地优化。我们是否认为这种 indirect improvement 已经足够？

### 问题 4：是否需要 V2 那种 denoising-step logp？

> 如果目标是直接提升 diffusion generator 生成轨迹的质量，是不是应该把 logp 放到每一步 denoising transition 上，而不是只用最终 score head？

### 问题 5：ref policy 和 old policy 是否应该区分？

> 当前 frozen pretrained model 被我用作 ref policy。它适合做 KL reference，但如果要严格做 PPO/GRPO ratio，denominator 是否应该是 rollout 时的 old policy logp，而不是固定 pretrained ref？

---

## 12. 可以用来汇报的一句话版本

当前代码把 GRPO 用在 `poses_cls` 上，本质是在做 64 条候选轨迹的 mode-selection RLFT：PDM 高的轨迹会在 score head 上获得更高概率，因此最终 argmax 选出的轨迹可能更好。但由于 reward 虽然由 `poses_reg` 算出，梯度却主要经过 `poses_cls` 回传，所以它对轨迹坐标生成质量的提升是间接的。DiffusionDriveV2 把 logp 放到 diffusion denoising step，是把每一步轨迹生成动作本身纳入 policy，因此 reward 能更直接地优化 generator 和最终轨迹质量。

