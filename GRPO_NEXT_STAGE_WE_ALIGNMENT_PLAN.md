# DiffusionDrive GRPO 下一阶段方案：从纯分类选择走向生成—选择联合优化

## 1. 本轮判断

当前 `poses_cls`-based GRPO 应保留为选择式基线，不再作为最终主路线。下一阶段的主路线应当是：

- 不使用 imitation loss、trajectory L1 loss 或 GT regression loss；
- 冻结图像/LiDAR backbone、BEV、检测等感知模块；
- 不把 DiffusionDrive 的生成过程冻结成固定候选集；
- 对去噪生成和最终 mode selection 分别定义可计算的策略概率，再用同一组 PDM reward 做联合 GRPO；
- 使用固定 pretrained reference policy 和 old/behavior policy，通过 PPO clip 与双层 KL 代替 imitation loss 提供稳定性。

因此，“生成”和“选择”可以在损失、日志和消融实验中分开，但不应在最终方法里完全拆成“固定生成器 + 独立分类器”。后者本质上会退化为第三个固定词表/候选集的选线模型，不能体现 DiffusionDrive 的生成能力。

## 2. 从 WorldEngine 代码得到的证据

### 2.1 两类选线实现的共同本质

WorldEngine 中可看到两个相关维度：

1. VADv2/HydraMDP 配置使用 `TrajScoringHead`，在固定的 8192 条轨迹词表上进行上下文相关的轨迹打分；HydraMDP 还增加 reward-shaping 分量。
2. 后训练使用 `TrajScoringHeadRL`，在同一个固定词表上做 reward-guided RLFT，并通过 LoRA 更新评分网络。

这两类方法的共同点是：

- `vocab` 本身是 `requires_grad=False` 的固定轨迹集合；
- 训练的并非只有最后一个标量分类层，而是轨迹编码、状态编码、BEV cross-attention、transformer 和打分 heads，RLFT 时主要以 LoRA 形式适配；
- RLFT 配置冻结 image backbone、image neck 和 BEV encoder，说明“冻结感知、训练规划任务”是已有实验采用的边界；
- `TrajScoringHeadRL.loss()` 仍会计算 `loss.imi`，配置中的 `orig_IL=True` 和 `hard_case_no_imi=True` 只是在 hard/synthetic case 上屏蔽部分 imitation，而不是全程纯 RL。

所以 WE 的结果能证明“稀有失败数据 + reward-guided planning post-training”有效，但不能直接证明当前无 imitation 的分类 GRPO 公式有效，也不能直接复制其稳定性结论。

### 2.2 学长“不冻结 GRPO 任务”的合理初衷

结合代码，较合理的解释不是“连感知 backbone 一起全量解冻”，而是：

- 感知表示可以冻结；
- 接收 reward 的规划策略必须有足够容量适配；
- 对 DiffusionDrive 来说，规划策略不只是 `poses_cls`，还包括以 20 条 anchor 为起点的两步 truncated denoising、回归 refinement 和 mode ranking；
- 如果只训练分类器，新增的 diffusion 模型实验就会退化为已有两类选线模型的重复，无法回答“reward 能否提高连续轨迹生成质量”。

这个初衷是合理的，我同意将“纯分类冻结方案”降级为对照实验。

## 3. 原始 DiffusionDrive 为什么不适合完全拆开

DiffusionDrive 的 20 个 mode 并不是固定词表中的 20 条不变轨迹。每个 mode 都经历：

```text
plan anchor + truncated noise
    -> anchor/time encoding
    -> DiT attention/refinement
    -> poses_reg 更新候选几何
    -> DDIM 更新扩散状态
    -> poses_cls 对最终候选排序
```

`poses_reg` 和 `poses_cls` 使用同一份 decoder feature，再分别进入 regression branch 和 classification branch。回归结果还会成为下一层 refinement 和下一步 DDIM 的输入。因此候选生成质量、候选语义和最终排名是耦合的。

当前实现的问题在于：

- PDM reward 经 CPU/NumPy/scorer 后已不可微；
- GRPO log-prob 只有 `log_softmax(poses_cls)`；
- loss 对 regression branch 没有直接梯度；
- 解冻整个 `diff_decoder` 时，共享 attention/FFN 会因分类梯度改变，导致轨迹几何间接漂移，但这种漂移没有生成级 reward 方向约束；
- current/old/reference 各自生成了略有不同的候选轨迹，却把相同 anchor index 当作完全相同的离散 action 做 PPO ratio，策略更新后这一 action 语义并不严格稳定；
- 当前 exact categorical objective 按 `p_old` 加权，base policy 很尖锐时，低概率但高 reward 的 mode 几乎得不到梯度；同时 entropy 只记录、没有进入 loss，所以很难改变部署时的 argmax。

这解释了现有结果：1024-token 上 oracle 仍约为 `0.923`，但 selected reward 从 `0.744802` 变为 `0.743763`。候选上限没有明显变坏，分类更新却没有形成可泛化的 argmax 改进。

## 4. 下一阶段的核心目标

把联合策略写成两个可检查的部分：

```text
π_joint = π_gen(denoising transitions | scene, anchor)
          × π_sel(mode | final generated candidates, scene)
```

对每个场景的 20 条最终候选计算 PDM reward，并得到组内标准化 advantage。然后分别优化：

```text
L_total = λ_gen * L_GRPO_gen
        + λ_sel * L_GRPO_sel
        + β_gen * KL_gen(current || reference)
        + β_sel * KL_sel(current || reference)
        - α * H_sel
```

其中不包含 imitation loss 或 GT trajectory loss。

### 4.1 生成级策略概率

当前推理使用确定性 DDIM step，`eta=0` 时 transition 是零方差映射，不能定义稳定的 PPO log-prob。训练采样阶段需要显式使用带方差的 transition：

```text
p_theta(x_{t-1} | x_t, scene, anchor)
    = Normal(mu_theta, sigma_t^2 I)
```

实施要求：

- behavior/old policy 采样并保存每一步相同的 `x_t`、`x_{t-1}`、noise、old log-prob；
- current 和 fixed reference 必须在这些完全相同的 state/action transition 上重新计算 log-prob，而不是分别自由生成后比较结果；
- reward 只在最终轨迹上计算，并作为该候选全部 denoising steps 的 advantage；
- 对时间点和轨迹维度归一化 log-prob/KL，避免 20×8×2 维相加后尺度过大；
- 最后一个零方差 step 应排除、设置明确的 `sigma_min`，或改用有合法方差的训练 scheduler；
- 部署评测仍使用原始两步 deterministic DDIM，不强制引入推理随机性。

需要先审计当前 schedule：初始样本在 `t=8` 加噪，但第一步按 `t=10` 调 scheduler。原代码虽然训练和评测沿用了这一设置，但它不满足严格 transition likelihood 的状态定义。生成 GRPO 前必须通过配对消融确定使用 `add_noise(t=10) + [10, 0]`，还是 `add_noise(t=8) + [8, 0]`。

### 4.2 选择级策略概率

选择项继续使用 categorical GRPO，但改为真正与 behavior rollout 对齐：

- selection logits 在 behavior 生成出的同一组最终候选上比较 current/old/reference；
- old policy 只负责 ratio，fixed pretrained policy 只负责 KL；
- 加小幅 entropy bonus 或 temperature exploration，避免高 reward 低概率 mode 永远没有更新机会；
- selected reward、oracle reward、regret、top-1 hit、rank correlation 与 expected reward 同时记录；
- 不能只根据 expected reward 保存 checkpoint，最终部署指标仍是 deterministic argmax。

## 5. 冻结范围

### 5.1 主实验

冻结：

- sensor backbone；
- BEV/agent/perception heads；
- 共享 query decoder 和检测相关模块（第一阶段先固定）；
- 固定 `plan_anchor`；
- fixed reference policy 和 rollout 期间的 old policy。

训练：

- `TrajectoryHead.plan_anchor_encoder`；
- `TrajectoryHead.time_mlp`；
- `TrajectoryHead.diff_decoder` 的 attention、FFN、time modulation；
- 每层 `task_decoder` 的 regression 和 classification branches。

建议使用分组学习率，而不是所有规划参数同一个 LR：

| 参数组 | 初始 LR |
|---|---:|
| classification branches | `3e-6` |
| regression branches | `1e-6` |
| decoder shared blocks | `1e-6` |
| anchor encoder / time MLP | `5e-7` |

如果全量 planning head 不稳定，优先改成 planning-head LoRA，而不是重新退化为只训练分类头。

### 5.2 必须保留的对照组

| ID | 生成部分 | 选择部分 | 目的 |
|---|---|---|---|
| A | frozen | frozen | 0.848746 配对基线 |
| B | frozen | categorical GRPO | 纯选择上限；对应已有选线范式 |
| C | shared decoder 可训练 | categorical GRPO only | 当前实现；已有 1024-token 负结果 |
| D | generative GRPO | frozen/base selector | 判断 reward 是否能直接提高候选 oracle |
| E | generative GRPO | categorical GRPO | 推荐主方案；生成—选择联合优化 |

其中 B 很重要，但它是科学对照，不是最终方案。C 用现有实验结果即可，不重复大规模训练。

## 6. 数据与评测策略

WE 的实验显示 common-log post-training 可能损害 closed-loop，而 rare logs/rare rollouts 的收益明显更大。因此下一阶段虽然仍只在 DiffusionDrive/NAVSIM 中完成，不做 WE 接入，但训练数据应改为 failure-aware sampling：

- 用 base checkpoint 在 navtrain 中挖掘低 selected reward、高 selection regret、collision/off-road 的 rare token；
- 训练 batch 初始采用 `75% rare + 25% common`，全部只使用 RL reward，不在 common 样本上加入 imitation；
- validation 固定为 common holdout 与 rare holdout 两套，严禁只根据 128-token 小集合选模型；
- 最终同时报告 navtest common PDMS 和 navtest_failures/rare PDMS；common 作为退化防线，rare 作为优化主信号。

## 7. 实施顺序与停止门槛

### Stage 0：可信诊断与 schedule 审计

1. 固定 1024 common + 一套固定 rare holdout。
2. 对 base 记录 selected/oracle/regret、候选多样性、reward-rank correlation、分类熵。
3. 配对比较 `[t=8 -> 8,0]`、`[t=10 -> 10,0]` 与现有 `[t=8 -> 10,0]`，只选择不降低 base PDMS/oracle 的合法 schedule。
4. 建立按模块的梯度检查，确认 `L_gen` 对 regression branch 有非零梯度，`L_sel` 对 classification branch 有非零梯度。

Stage 0 不通过时不训练。

### Stage 1：纯选择对照 B

- 真正冻结 regression branch 和 shared generator，只训练 classification branches；
- 加 entropy/temperature 小网格；
- 用 1024 common + rare holdout 判断纯 selection 是否能稳定降低 regret。

若 B 仍无提升，就确认现有候选虽有 oracle gap，但静态排序泛化不足，不再扩大纯分类训练。

### Stage 2：生成级最小闭环 D

- 实现 stochastic behavior rollout、transition log-prob、old/current/reference replay；
- 只启用 `L_GRPO_gen + β_gen KL_gen`；
- 先跑 8 batch smoke，再跑 128/512 token 短训；
- 成功标准是 oracle reward 和候选均值提高，而不是只看 selected reward。

### Stage 3：联合方案 E

- 加入 selection GRPO、entropy 和 categorical KL；
- 从小 LR 开始，使用 adaptive KL coefficient；
- 每个候选只使用同一条 final reward，避免额外 reward regression/监督信号；
- common 与 rare holdout 同步早停。

### Stage 4：完整 NAVSIM 验收

只有满足以下门槛才进入完整训练与 navtest：

- fixed 1024 common selected reward 至少 `+0.005`；
- rare holdout selected reward 至少 `+0.02`；
- common oracle 不下降超过 `0.005`；
- selection regret 不增大；
- generation/categorical KL 均在目标区间，轨迹多样性不塌缩；
- 两个不同 seed 的提升方向一致。

最终阶段目标仍是：相对历史 navtest PDMS `0.8487464442665378` 提升至少 `0.01`，即达到 `>= 0.858746`；同时 rare PDMS 必须提升，不能以牺牲 rare/闭环潜力换 common 小涨。

## 8. 近期最小工作包

下一次开始改代码时，按以下顺序执行：

1. 为 rollout 增加 transition trace 数据结构和按 step 的 log-prob/KL 单测。
2. 修正/确定合法的两步 noise schedule，并重跑配对 base。
3. 把 pure-selection、generation-only、joint 三种 objective 做成显式 config，而不是继续在一个 forward 中混杂。
4. 实现 planning-head 参数分组、梯度审计和 adaptive KL。
5. 从现有 metric cache 构建 rare/common 固定 token 清单。
6. 依次运行 B、D、E 的 smoke 和门控短训；所有 Python 均使用 `navsim` 环境。

在这六步完成前，不建议直接延长现有分类 GRPO 的 epoch，也不建议先跑完整 navtest。
## 9. 2026-07-15 执行记录

### 9.1 Schedule 与纯选择对照

固定、排序后的 1024-token common holdout 结果：

| 配置 | selected | oracle | regret | candidate mean |
|---|---:|---:|---:|---:|
| 历史不一致 schedule：t8 + [10,0] / 1000 | 0.744300 | 0.920899 | 0.176600 | 0.547196 |
| 合法 t10 + [10,0] / 100 | 0.756833 | 0.915626 | 0.158794 | 0.665828 |
| 合法 t8 + [8,0] / 125（采用） | **0.757878** | **0.917801** | 0.159923 | 0.660489 |

评测代码已去除 forward_test 中初始 timestep 的硬编码，之后 summary 中的
truncation timestep 与实际加噪 timestep 一致。当前 D/E 实验均使用未受该问题影响的
t8 + [8,0] / 125。

Stage 1 只训练最后 classification branch（132,865 个参数），三个配置在相同 1024
holdout 上的 selected 分别为 0.755853、0.755975、0.755958，均低于
0.757878 base。纯分类方案 B 因此停止扩训。

### 9.2 生成与联合策略闭环

已完成：

- stochastic DDIM behavior action；
- old/current/reference 在同一 state/action trace 上重算 log-prob；
- 两步 generation PPO clip 与 fixed-reference Gaussian KL；
- generation、joint 显式训练模式；
- 无 imitation loss、无 GT trajectory regression loss；
- 感知全冻结，只解冻 DiT planning decoder；
- 12 个 schedule/objective/probability 单测全部通过。

真实一 batch 梯度审计：

| 模式 | shared grad | regression grad | classification grad |
|---|---:|---:|---:|
| generation-only D | 6.807 | 26.217 | **0.000** |
| joint E | 7.204 | 26.217 | 1.255 |

这证明生成 reward 对 regression branch 有直接非零梯度，D 的 selector 同时保持严格冻结。

### 9.3 128-token 门控短训与严格 1024 结果

两组均训练 2 epoch、每 epoch 128 step，LR 1e-6：

| checkpoint | selected | 相对 base | oracle | 相对 base | candidate mean |
|---|---:|---:|---:|---:|---:|
| base t8 | 0.757878 | — | 0.917801 | — | 0.660489 |
| D epoch0 / 128 step | **0.761178** | **+0.003299** | 0.918610 | +0.000809 | 0.660757 |
| E epoch0 / 128 step | 0.759181 | +0.001302 | 0.918590 | +0.000789 | 0.660731 |
| D epoch1 / 256 step | 0.757323 | -0.000555 | **0.919837** | **+0.002036** | **0.661787** |
| E epoch1 / 256 step | 0.756787 | -0.001091 | 0.919126 | +0.001325 | 0.661690 |

D epoch1 的 oracle 配对 bootstrap 95% CI 为 [+0.000181, +0.004502]，首次得到
候选上限的正向信号；但 selected 未过 +0.005 门槛。D epoch0 的 selected 最好，
而继续训练提高 oracle、降低 selected，说明生成分布逐渐与 selector 失配，不能直接延长 epoch。

### 9.4 下一决策

当前不进入完整 navtest。下一小阶段优先级：

1. 保留 D epoch0 为当前 best common checkpoint；
2. 建立固定 rare/common token 清单与 failure-aware sampling；
3. 以 128 step 早停为中心，仅小网格 generation LR / final action std / KL；
4. joint selector 降低 LR 或延迟启用，先让 generation 获得稳定 oracle 增益，再做短程 selector 对齐；
5. 只有 fixed common selected >= +0.005、rare selected >= +0.02 且两个 seed 同向，才进入 512/full/navtest。
### 9.5 Common-only generation 小网格与停止结论

在 D epoch0 周围追加 128-step generation-only 小网格：

| 配置 | selected | 相对 base | oracle | 相对 base |
|---|---:|---:|---:|---:|
| LR 2e-6 / final std .05 | 0.760274 | +0.002396 | 0.919161 | +0.001360 |
| LR 3e-6 / final std .05 | 0.759107 | +0.001228 | 0.920028 | +0.002226 |
| LR 2e-6 / final std .10 | 0.760559 | +0.002681 | 0.918346 | +0.000545 |

更强更新继续体现“oracle 上升、selected 跟不上”的趋势，没有超过 LR 1e-6 的 D epoch0。

随后从 best D epoch0 出发，只训练 132K classification 参数做 sequential selector alignment：

| 配置 | selected | 相对 base |
|---|---:|---:|
| S1：temp 1.0 / entropy 0 / LR 3e-6 | 0.760423 | +0.002545 |
| S2：temp 1.5 / entropy .01 / LR 1e-5 | 0.759229 | +0.001351 |
| S3：temp 1.5 / entropy .005 / LR 3e-6 | 0.760391 | +0.002513 |

三组都没有超过未做 selector alignment 的 D epoch0。best D epoch0 的 selected 配对
bootstrap 95% CI 为 [+0.000782, +0.006601]，说明 common proxy 上方向可信；但
+0.003299 仍低于 +0.005 工程门槛，也不能外推为 navtest 0.848746 已提升。

因此停止 common-only 超参搜索。下一次实验必须先完成 rare/failure-aware token
构建与固定 rare holdout；在此之前不跑完整 navtest，也不继续堆 classification 超参。
