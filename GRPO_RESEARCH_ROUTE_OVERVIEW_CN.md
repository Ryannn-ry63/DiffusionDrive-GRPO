# DiffusionDrive GRPO 研究路线总览

> 文档定位：这是一份跨 Stage 的研究路线说明，用于回答“整个项目为什么这样做、每一阶段解决了什么问题、目前得到了什么结论”。它不是某一次实验的运行手册，也不是单个 Stage 的结果记录。

## 1. 一句话概括

这条研究路线从“用 GRPO 重新排序 DiffusionDrive 已有的候选轨迹”出发，逐步转向“在 diffusion 去噪过程中直接优化轨迹生成”，随后集中解决生成收益不稳定、少量场景灾难性退化，以及新旧轨迹之间难以可靠选择的问题。

当前最准确的结论是：

> Full-chain diffusion GRPO 已经证明能够在受控数据上生成 PDMScore 更高的候选轨迹，但这种收益尚未在更大、独立的数据集上稳定复现，并且存在少量严重回退。当前重点已经从“能否生成更好的轨迹”转向“如何可靠识别改善轨迹，同时保护原本表现良好的场景”。

## 2. 最初的问题是什么

原始 DiffusionDrive 会生成多条候选轨迹，并通过 `poses_cls` 为这些轨迹打分。监督训练通常重点强化离 GT 最近的 mode，因此模型比较擅长：

- 生成一组候选轨迹；
- 判断哪一条轨迹更接近监督目标；
- 将最接近 GT 的轨迹作为最终输出。

但离 GT 最近不一定等于 PDMScore 最高。项目最初提出的问题是：

> 能否直接使用 PDMScore 作为 reward，让模型选择或者生成驾驶质量更好的轨迹？

最早的方案把 `poses_cls` 视作离散策略的 logits：

```text
logp = log_softmax(poses_cls)
action = 选择第几个候选 mode
reward = 对该候选轨迹计算的 PDMScore
```

这种方法直接优化的是轨迹选择能力，而不是轨迹坐标本身。PDM evaluator 不可微，reward 经过 detach、CPU 和外部 scorer 后，不存在下面这条梯度路径：

```text
PDM reward -> 轨迹坐标 -> regression branch
```

真实的主要梯度路径是：

```text
GRPO loss -> poses_cls -> classification branch -> shared decoder features
```

因此，早期 selection GRPO 能学习“已有候选中哪条更好”，但不能直接告诉 diffusion generator 怎样修改轨迹。这构成了整条研究路线的第一个转折点。

## 3. 整体演化路线

```text
固定评测基线与数据切分
        ↓
Selection GRPO：学习重新排序已有候选
        ↓
发现 selection-only 上限有限
        ↓
Generation GRPO：优化 diffusion 去噪动作
        ↓
从最后一步扩展到 full-chain denoising
        ↓
候选轨迹平均质量明显提高
        ↓
发现 selector 不可靠且存在尾部灾难样本
        ↓
Value selector / risk selector / fallback
        ↓
Selected anchor / base preserve / paired residual
        ↓
Candidate bank + 保守的集成 selector
```

## 4. Stage 0–4：建立基线并验证 Selection GRPO

### Stage 0：固定实验和评测基础

主要工作：

- 建立固定的 256、1024、3072、4096 场景评测集；
- 固定 token 顺序和 manifest；
- 对齐旧评测流程与新评测流程；
- 计算原始模型 baseline；
- 建立困难场景 priority manifest；
- 检查训练 cache 和评测数据是否一致。

这一阶段的目的不是提高模型，而是排除以下混淆因素：

- 不同实验实际评测了不同场景；
- 随机 diffusion noise 不一致；
- token 顺序变化；
- evaluator 配置变化；
- 数据集或 cache 不一致。

### Stage 1：初步超参数搜索与 Model Soup

尝试内容：

- 不同 batch size；
- 不同训练步数；
- 不同 checkpoint；
- 多个 checkpoint 的参数平均，即 model soup；
- 在 fixed-256 与 holdout-1024 上复核。

得到的认识：小规模评测中可以观察到轻微提升，但提升幅度较小，在更大数据集上不够稳定。

### Stage 2：拆分 Selection 与 Generation 目标

开始明确区分：

- `selector`：优化 `poses_cls`，提高候选排序能力；
- `generation`：优化 diffusion 去噪动作，改变轨迹生成；
- `decoupled`：分别控制 selection objective 与 generation objective。

同时加入 PDM 分项诊断：

- collision；
- drivable；
- progress；
- TTC；
- 后续阶段中的 comfort 和 direction。

这一改动使实验不再只看总分，因为总分提升可能掩盖碰撞或 TTC 退化。

### Stage 3：补充完整场景信息

尝试让 reward 和训练目标使用更完整的 evaluator 信息，并比较 small fixed set 与更大 holdout set。

主要认识：在 256 个场景上可见的改善，扩大到 1024 个场景后经常明显收缩。

### Stage 4：加入 Ranking Loss

比较：

- 纯 GRPO；
- 纯 ranking loss；
- GRPO 与 ranking loss 混合。

Ranking loss 显式要求高 reward 轨迹的分类分数高于低 reward 轨迹。结果仍然主要表现为小规模略有提升、大规模不稳定，进一步说明 selection-only 的上限有限。

## 5. Stage 5–8：改进采样与 Advantage 估计

### Stage 5：Priority Sampling 与 Trust Projection

尝试内容：

- 增加困难场景的训练采样比例；
- 与 uniform sampling 对照；
- 收集训练集上的轨迹变化分布；
- 使用分位数校准 trust projection；
- 限制新策略相对 base/reference 的偏移。

结果表明 priority sampling 没有带来稳定收益，部分配置明显退化。可能原因是困难场景 reward 方差更大，过度采样会改变整体训练分布。

### Stage 6：同一 Mode 多次 Rollout

此前每个 mode 只有一个 diffusion rollout，无法区分：

- mode 本身更好；
- 某一次 noise 恰好更好。

因此引入：

- `rollouts_per_mode=2`；
- within-anchor advantage；
- hierarchical advantage。

Hierarchical advantage 同时比较：

1. 同一个 mode 在不同 noise 下的表现；
2. 不同 mode 之间的表现。

部分 fixed-1024 结果达到约 `+0.00307`，但在更大的 development selection set 上仍未形成稳定收益。

### Stage 7：Anchor Hierarchical Advantage

继续加强 anchor 设计：

- 每个 mode 使用 2 或 4 个 rollout；
- 比较多个训练步数；
- 使用 base/reference 输出作为稳定参照。

目的是减弱纯 group z-score 对偶然 diffusion noise 的敏感性。

### Stage 8：Anchor RLOO

使用 leave-one-out baseline：评价某个 rollout 时，以同组其他 rollout 的平均 reward 作为 baseline。

该方法降低了 advantage 估计中的部分偏差，但实际提升仍较小，没有解决大规模泛化问题。

## 6. Stage 9–14：从“更会选”转向“更会生成”

### Stage 9：Selection 与 Generation 的正面对比

Stage 9 注册了两个独立目标：

- `selector_group`；
- `generation_group`。

实验协议规定 selector gate 失败后再启动 generation 实验，以避免混淆两条路线。

代表性 fixed-1024 结果：

- selector：约 `+0.00399`；
- generation：约 `+0.00904`。

这一阶段给出了重要证据：直接优化 diffusion generation 比只训练候选排序更有潜力。

### Stage 10：短链 Generation GRPO

主要配置：

- `generation_group_adaptive`；
- diffusion rollout timesteps 为 `[8, 0]`；
- 学习率约 `1e-6`；
- 关闭额外 selector objective；
- 关闭 priority sampling；
- 比较 decoder 梯度范围与不同层的学习率倍数。

小规模 fixed-256 gate 可以通过，但 fixed-1024 提升约为 `+0.00182`，未达到正式门槛。

### Stage 11：Checkpoint 回退与 Base 等价性检查

主要检查：

- base 与 reference checkpoint 是否等价；
- 训练早期 checkpoint 是否优于最终 checkpoint；
- 是否存在训练过头；
- 不同评测会话是否能够复现相同 baseline。

回退候选在 fixed-1024 上约有 `+0.00267`，仍未通过正式门槛。

### Stage 12：冻结 Selector 并延长 Generation 训练

主要设计：

- 推理时冻结 selector；
- 从 generation checkpoint 继续训练；
- 检查 U512、U1024、U2048 等训练时刻；
- 使用多个 seed；
- 扩大 dev 和 navtest 验证规模。

部分 checkpoint 相对 base 出现 `+0.003` 到 `+0.0068` 的结果，但没有同时满足：

- 相对 base 显著提升；
- 相对 Stage 10 继续提高；
- candidate/oracle 不下降；
- 安全分项不退化；
- 多 seed 稳定。

因此 Stage 12 的正式诊断没有通过。

### Stage 13：扩大 Decoder 更新范围

开始尝试更深的 decoder 更新和更完整的 diffusion schedule。Pilot 在 3072 场景上的 selected delta 为约 `+0.0014928`，门槛为 `+0.0015`，非常接近但仍判定失败。

### Stage 14：Collision-Truncated Intra-Anchor

引入：

```text
generation_advantage_mode = collision_truncated_intra_anchor
```

设计目的：

- 同一 mode 生成两个 rollout；
- 主要进行 anchor 内比较；
- 对碰撞相关的极端 reward 做截断或特殊处理；
- 避免少量极端碰撞样本产生过大的梯度。

训练与 checkpoint 健康检查通过，但结果仍表现为小规模正向、更大规模不稳定。

## 7. Stage 15–18：生成器改善后，解决轨迹选择和尾部风险

### Stage 15：Value Selector

训练新的 value selector：

- 3 个 value heads；
- top-2 候选；
- bootstrap 训练；
- 根据候选轨迹 reward 学习轨迹价值。

独立 selector test 中，value selector 相对 frozen selector 的提升约为 `+0.00059`，置信区间跨 0，并伴随 TTC 下降，因此未通过 gate。

这说明即使候选中出现了更好的轨迹，普通 value selector 也未必能稳定选择它。

### Stage 16：Full-Chain Diffusion GRPO

这是路线中的关键实验之一。

主要改动：

- 使用 `diffgrpo_full_chain`；
- rollout timesteps 从 `[8, 0]` 扩大到 `[32, 24, 16, 8, 0]`；
- 优化多个 denoising transitions；
- 更新所有 decoder layers；
- 推理时继续使用 reference selector，以隔离 generation 的真实贡献。

Development calibration 选择 epoch 8：

- system total delta：约 `+0.02738`；
- 其中 GRPO generation contribution：约 `+0.01710`；
- collision、drivable、TTC 在 calibration 上未下降。

这证明 full-chain diffusion GRPO 能明显提高受控数据上的候选生成质量。

但 Stage 16 训练出的 selector 没有通过独立 selector test。问题从“能不能生成好轨迹”转变为“能不能稳定选中好轨迹”。

### Stage 17：Paired Tail-Risk Selector

不再只预测候选的平均价值，而是训练风险模型识别：

- base 是否更好；
- 新轨迹是否损失超过 0.1 或 0.5；
- collision 是否退化；
- drivable 是否退化；
- TTC 是否退化。

当风险超过阈值时，系统 fallback 到 base 轨迹。风险模型最终选择 epoch 16。

Fixed-1024 上：

- 相对 base 平均提升约 `+0.02025`；
- CI95 明确大于 0；
- safety 分项均值没有下降。

但正式 gate 仍失败，因为：

- 最坏 token 下降约 `-0.887`；
- bottom 1% CVaR 过差。

这揭示了后半程最核心的矛盾：平均收益很好，但少数场景可能发生灾难性回退。

### Stage 18：Average-Primary Gate

Stage 18 主要调整评测协议，而不是训练新的生成器。

定义：

- A：原始完整系统；
- B：base generation；
- C：GRPO generation；
- D：风险 selector 在 B/C 之间选择的结果。

Fixed-1024 diagnostic 可以通过，但在 3072 dev-select 上，D 相对 B 仅约 `+0.00056`，置信区间跨 0，正式 gate 失败。

这说明 Stage 17 在 fixed-1024 上的明显提升没有稳定迁移到更大的 development set。

## 8. Stage 19–22：围绕已选 Mode 优化并保护 Base

### Stage 19：Selected-Anchor GRPO

使用：

```text
diffgrpo_selected_anchor
```

不再平均优化所有 mode，而是围绕 reference selector 实际选中的 mode 进行生成优化，使训练目标更接近最终系统输出。

Calibration 选择 epoch 8：

- 两个 noise namespace 平均提升约 `+0.01671`；
- calibration safety 分项为正。

但 internal test 中 collision 平均约下降 `-0.00041`，因此严格 gate 失败。

### Stage 20：更大 Dev-Select 迁移验证

Stage 20 主要验证 Stage 19，而不是提出新的训练算法。

在 3072 场景、两个 noise namespace 上：

- GRPO 相对 base 约 `+0.00038`；
- 置信区间跨 0；
- collision 和 TTC 显著下降；
- 灾难样本比例超过门槛。

这说明 Stage 19 calibration 上的大幅提升存在选择偏差，换到更大场景集后不能复现。

### Stage 21：Selected Anchor + Base Preserve

使用：

```text
diffgrpo_selected_anchor_base_preserve
```

加入：

- base trajectory preservation；
- behavior cloning 正则；
- base margin；
- 对原本高分场景的保护；
- safety regression tolerance。

结果呈现清晰的训练强度权衡：

- epoch 2 较保守，平均提升约 `+0.00318`，低于 `+0.005` 门槛；
- epoch 4–8 平均提升可达约 `+0.013` 到 `+0.016`；
- 训练越久，对原本 `base >= 0.75` 的场景破坏越严重；
- 灾难样本随训练增加。

最终没有 epoch 同时满足收益、高分场景保护和灾难率门槛。

### Stage 22：Paired Residual + LoRA

使用：

```text
diffgrpo_paired_residual
```

主要改动：

- rank-8 LoRA，仅学习 residual；
- 同时训练正向改善和负向保护；
- 高分成熟场景使用更强 KL；
- 使用 paired positive/negative margin；
- 梯度裁剪；
- 锁定官方 base checkpoint SHA；
- 使用多个 fold，降低数据选择泄漏风险。

结果过于保守：epoch 1–4 的平均 C-B 约在 `-0.0003` 到 `-0.0001`，几乎没有学到有效提升。

Stage 21 与 Stage 22 共同说明：

> 约束较弱时，模型能够提高低分场景，但会破坏部分高分场景；约束过强时，可以限制模型漂移，但 GRPO 几乎学不动。

## 9. Stage 23：Candidate Bank 与集成 Selector

Stage 23 已有训练和评测代码，但当前仓库中尚未看到完整的正式 gate 结果，因此应视为正在进行的方向。

主要思路：

- 从多个 diffusion noise namespace 收集候选轨迹；
- 建立 candidate bank；
- 使用 5 个 selector members 做集成；
- 每个场景比较 20 个 modes；
- 使用 pairwise reward gap；
- 使用 focal loss；
- 对不安全正样本设置较大的损失权重。

这一阶段不再优先修改 generation，而是尝试回答：

> 已经能够生成更好候选的前提下，能否训练一个更强、更保守、更稳定的 selector？

## 10. 已经得到的核心研究结论

### 10.1 Selection GRPO 有效，但上限有限

Selection GRPO 可以重新排序已有候选，却不能直接创造更好的轨迹。如果候选集合中不存在更好的轨迹，再好的 selector 也无法提高上限。

### 10.2 Generation GRPO 比 Selection GRPO 更有潜力

Stage 9 的正面对比支持这一结论。把策略梯度放入 diffusion denoising transition 后，reward 能更直接地影响轨迹生成过程。

### 10.3 Full-Chain 优于只优化最后一步

从 `[8, 0]` 扩展到 `[32, 24, 16, 8, 0]` 后，Stage 16 在受控 calibration 上获得了明显更大的 generation gain。

### 10.4 平均 PDM 提升不代表系统可靠

Stage 17 平均提升约 `+0.020`，但最坏样本下降接近 `-0.887`。自动驾驶场景不能只看平均值，还需要关注：

- 最坏 token；
- bottom 1% CVaR；
- catastrophic rate；
- collision、drivable、TTC 分项；
- 置信区间。

### 10.5 Selector 是一个独立瓶颈

更好的候选被生成出来，不代表系统能够识别它。Stage 15、17、23 分别使用 value、risk 和 ensemble selector 处理这一问题。

### 10.6 Calibration 提升容易过拟合

Stage 19 calibration 有约 `+0.0167`，到了 Stage 20 dev-select 只剩约 `+0.00038`。因此 checkpoint selection 和正式测试必须使用严格隔离的数据切分。

### 10.7 提高低分场景与保护高分场景存在冲突

Stage 21 表明，训练越充分，低分场景改善越大，但原本表现良好的场景和灾难率可能恶化。

### 10.8 过强的保护会让模型学不动

Stage 22 的 LoRA、paired residual 和 KL 保护成功限制了模型变化，但同时几乎消除了可测的正向收益。

## 11. 当前状态

目前不能直接声称“GRPO 已经稳定提高 DiffusionDrive 正式成绩”。更准确的状态是：

### 已经证实

- generation GRPO 比 selection-only 更有潜力；
- full-chain diffusion GRPO 能在受控数据上明显改善候选生成；
- 候选集中确实可能存在比 base 输出更高分的轨迹；
- selector、tail risk 和数据泛化是当前主要瓶颈。

### 尚未证实

- 在严格独立的大规模测试集上稳定提高最终 PDMS；
- 多 seed 下稳定复现；
- 在提高平均分的同时控制灾难性回退；
- 新 selector 可以可靠选择 GRPO 改善轨迹；
- Stage 23 candidate-bank selector 已经通过正式 gate。

## 12. 各 Stage 的角色速查

| Stage | 核心任务 | 主要认识/状态 |
|---|---|---|
| 0 | 固定基线、manifest、cache 和 evaluator | 建立可信实验基础 |
| 1 | 超参数与 model soup | 小规模有轻微收益 |
| 2 | 拆分 selector/generation objective | 明确两种优化语义 |
| 3 | 完整场景信息 | 小规模收益难扩展 |
| 4 | Ranking loss | Selection-only 仍不稳定 |
| 5 | Priority sampling、trust projection | 困难样本过采样无稳定收益 |
| 6 | 多 rollout、hierarchical advantage | 降低 noise 混淆 |
| 7 | Anchor hierarchical | 加强 reference anchor |
| 8 | Anchor RLOO | 降低 advantage 偏差 |
| 9 | Selector vs generation | Generation 明显更有潜力 |
| 10 | 短链 generation GRPO | 小规模通过，大规模不足 |
| 11 | 回退与 base 等价检查 | 排查 checkpoint/评测混淆 |
| 12 | 冻结 selector、延长训练 | 局部提升但多条件未通过 |
| 13 | 扩大更新范围 | Pilot 几乎擦线通过 |
| 14 | Collision-truncated intra-anchor | 控制极端碰撞梯度 |
| 15 | Value selector | 独立测试失败 |
| 16 | Full-chain diffusion GRPO | 受控数据上 generation gain 明显 |
| 17 | Paired tail-risk selector | 平均提升大，但尾部风险失败 |
| 18 | Average-primary gate | 大 dev set 未稳定复现 |
| 19 | Selected-anchor GRPO | Calibration 好，内部安全 gate 失败 |
| 20 | 大规模迁移验证 | 收益消失且安全退化 |
| 21 | Base preserve | 学得动，但高分场景受损 |
| 22 | Paired residual + LoRA | 保护较强，但基本学不动 |
| 23 | Candidate bank + ensemble selector | 已实现，正式结果待完成 |

## 13. 常用术语

- **Base/Reference**：原始监督训练的 DiffusionDrive checkpoint，或者训练期间固定的参考策略。
- **Candidate**：模型生成的候选轨迹。
- **Selected reward**：按照当前 selector 选择轨迹后得到的 reward。
- **Oracle reward**：假设知道所有候选真实 reward，直接选择最高者时的上界。
- **Selection GRPO**：把“选择哪个 mode”作为 action。
- **Generation GRPO / DiffGRPO**：把 diffusion 去噪 transition 作为 action，直接影响生成过程。
- **Anchor**：用于稳定 advantage 或限制策略漂移的 base/reference 轨迹。
- **Fallback**：风险较高时放弃新轨迹，退回 base 输出。
- **Noise namespace**：固定 diffusion 随机性的标识，用于 paired evaluation 和复现。
- **Gate**：实验预先规定的通过标准，不只是检查代码能否运行。
- **CI95**：95% 置信区间，用于判断平均提升是否可能只是抽样波动。
- **CVaR**：关注分布尾部最差样本的风险指标。
- **Catastrophic regression**：新策略相对 base 出现非常大的单场景下降。

## 14. 阅读仓库时的建议顺序

1. 先阅读本文件，理解整条研究路线。
2. 阅读 `grpo_logic_clarification.md`，理解 selection 与 generation 的语义区别。
3. 查看 `scripts/training/run_diffusiondrive_grpo_stage*.sh`，了解各阶段实际训练配置。
4. 查看 `scripts/evaluation/check_grpo_stage*_gate.py`，了解每个阶段的通过标准。
5. 查看 `artifacts/grpo_stage*/` 中的 `gate.json`、`selection.json` 和 audit 文件，区分“代码运行成功”和“研究假设通过”。

需要特别注意：

> `audit passed` 通常只说明 checkpoint、参数更新、训练指标或文件结构正常，不等于模型效果通过正式评测。判断实验是否真正成功，应优先查看正式 `gate.json` 或 checkpoint `selection.json`。
