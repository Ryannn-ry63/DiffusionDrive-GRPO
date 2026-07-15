 GRPO 策略选择的语义澄清

## 1. 先说结论


1. 原始 DiffusionDrive 的训练重点，确实更偏向“把距离 GT 最近的那条 mode 学好”。
2. 你现在把 `poses_cls` 当作 `logp` 做 GRPO，主要优化的是“选轨迹的能力”，而不是直接优化“生成轨迹的能力”。
3. 但这不意味着它完全不会影响轨迹质量。它对轨迹质量的提升，更多是**间接的**。
4. 如果目标是更直接地提升轨迹生成质量，DiffusionDriveV2 那种把 logp 放进 denoising 过程里的做法，语义上更贴近“优化生成器本身”。

---

## 2. 原始 DiffusionDrive 的痛点

DiffusionDrive 在监督训练时，核心问题是：

> 每个样本最终通常只强化一条最接近 GT 的轨迹 mode。

更准确地说：

- 训练信号会把“最接近 GT 的 mode”拉高。
- 其他 mode 虽然不是完全没有梯度，但通常是被压低、被边缘化。
- 结果是模型更像在学“哪个 mode 最像 GT”，而不是让所有 mode 都朝着高质量、多样性方向共同进化。

所以你说的“其他轨迹像陪跑”这个理解，方向是对的，但最好改成：

> 其他轨迹不是完全没训练，而是没有被作为高质量候选去正向强化，更多是被当作非目标 mode 处理。

这意味着 pretrained DiffusionDrive 的强项通常是：

- mode ranking
- nearest-mode selection
- 生成一组还不错的候选轨迹

但不一定是：

- 每条候选轨迹都尽量变得更优
- 让所有 mode 都朝 PDM 高分方向持续优化

---

## 3. 你现在的 GRPO 在优化什么

你现在的写法是把 `poses_cls` 当作离散 policy 的 logits：

logp = log_softmax(poses_cls)

然后对每条候选轨迹算 reward：

reward = PDMScore(poses_reg)
```

再用组内 advantage 做 policy gradient / GRPO 风格优化。

这时 action 的语义其实是：

```text
action = 选择第几条候选轨迹
```

所以它直接优化的是：

> 模型从候选轨迹里选出高 PDM 轨迹的能力，而没有直接优化轨迹。

原因：PDM reward 一般是不可微的。在当前代码中，轨迹会经过 `detach()`、CPU、numpy、PDM scorer 等外部计算。这样 reward/advantage 对神经网络来说更像一个常数权重。

> 他的反向传播时，主要梯度路径是：

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

## 4. 为什么它还是可能提升 PDMScore

虽然它主要在训选择能力，但 PDMScore 还是可能提升，原因有两类。

### 4.1 直接提升：选得更准

如果候选集中本来就有一条pdmscore更好的轨迹但是距离gt较远，但原模型没把它排到前面，那么 GRPO 之后：

- reward 高的轨迹会得到更高 `poses_cls`
- 最终 argmax 更容易选到它

这类提升最直接，也最稳定。

### 4.2 轨迹质量间接提升：共享特征被改了

你的 `poses_cls` 和 `poses_reg` 不是完全独立的，它们共享前面的 decoder 特征。

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

---

## 5. 困惑

你担心的是：

> baseline 已经有 88.1 分，参数基本都在为“最近 GT 的那条轨迹”服务，其他轨迹会不会已经很差了？如果候选集里本来就没有更好的轨迹，那 GRPO 最后还是只能在那条最近轨迹上反复打转。

这个担心是合理的

原因是：

1. 如果 pretrained 模型已经很偏向 nearest mode，那么其他 mode 的质量可能确实不足。
2. 你的 GRPO 如果只做 `poses_cls` 级别的 selection learning，它不能凭空“造”出一个新的高质量 mode。
3. 它只能更好地排序已有候选。

所以选择式 GRPO 的上限受一个东西限制：

> 候选集合里本来有没有更好的轨迹。

如果没有，那它很可能只是在：

- 更认真地选那条最近轨迹
- 或者在相邻几个模式之间轻微重新排序

而不是显著提升轨迹生成上限。

这也是为什么你说“最后还是在训距离 GT 最近的那条”这个担忧，并不是空穴来风。

---

## 6. 为什么 DiffusionDriveV2 更直接

DiffusionDriveV2 的关键差别是，它不是等轨迹全生成完以后再做 mode selection，而是把 logp 放进**去噪的每一步**。

可以理解成：

```text
current method: 先生成多条轨迹，再学会选哪条
V2 method: 学会每一步该怎么去噪，才能把轨迹生成好
```

在 V2 里，action 不是“选 mode”，而是：

```text
x_t -> x_{t-1}
```

也就是每一步去噪采样。

于是 logp 对应的是：

```text
这一小步去噪动作的概率
```

reward 是对最终轨迹算的。

这样 reward 的梯度会更直接地作用到：

- denoising transition
- diffusion generator
- 轨迹生成过程本身

所以它更像是在训练：

> 轨迹怎么生成出来，才能更高 PDM。

而不是只在训练：

> 生成完以后，哪条更值得选。

这就是它比 `poses_cls` 方案更直接的原因。

---

## 7. 你现在这套方案的准确定位

如果用一句话概括，你现在的方案更像：

> 基于 score head 的 mode-selection RLFT baseline。

它适合验证：

- 现有候选轨迹里是否已经包含更好的 PDM 轨迹
- 重新排序是否能提高最终输出

它不适合直接证明：

- 模型生成器本身被显著强化了
- 所有候选轨迹的质量都被系统性提升了

所以你后面和学长讨论时，最好把两个目标分开：

1. **selection improvement**：更会选
2. **generation improvement**：更会生成

当前 `poses_cls`-based GRPO 更偏第 1 个。
DiffusionDriveV2 更偏第 2 个。

---

## 8.

> DiffusionDrive 原始监督训练的痛点在于，它主要强化的是离 GT 最近的那条 mode，其他轨迹没有被作为高质量候选去正向优化，所以 pretrained 模型更强的是 mode ranking，而不是所有候选轨迹的生成质量。
> 我现在把 `poses_cls` 当作 logp 做 GRPO，本质上是在训练模型更准确地选择已有候选轨迹，所以它首先提升的是 selection 能力。它对轨迹质量的提升是间接的，主要来自共享 decoder 特征被更新后，reg 分支可能跟着变化，以及最终 argmax 更容易选到高 PDM 的轨迹。
> 但如果候选集合里本来没有更好的轨迹，selection-only 的 GRPO 上限就会受限，可能还是围绕最近 GT 的那条轨迹打转。相比之下，DiffusionDriveV2 把 logp 放到 denoising step 上，直接优化轨迹生成过程，因此对轨迹质量的提升会更直接。
