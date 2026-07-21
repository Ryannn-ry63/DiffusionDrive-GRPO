# DiffusionDrive 独立 GRPO 强化与消融计划

日期：2026-07-16
分支：`grpo-selection-v2`

## 1. 目标

本阶段完全不涉及 WorldEngine，只在 DiffusionDrive/NAVSIM 中验证 GRPO 的独立作用：

- 不使用 imitation loss、GT trajectory regression 或 supervised diffusion loss；
- 保持感知模块冻结，只更新规划 diffusion decoder 的生成路径；
- 最终只用严格配对的原始 full-navtest PDMS 判断是否提升；
- 要求 3 个独立训练 seed 同向，至少 2 个 seed 的逐 token bootstrap 95% CI 不跨 0，且 seed-stratified 总体 CI 下界大于 0。

不强制要求 `+0.005` 或 `+0.01`，但结论必须可复现、统计可信。

## 2. 已有消融与下一步

已有 A--E 结论保留：

| 组别 | 方法 | 已有结论 |
|---|---|---|
| A | frozen base | paired full-navtest PDMS `0.8491642572` |
| B | frozen generator + categorical GRPO | 未稳定超过 base |
| C | shared decoder + categorical GRPO | 1024-token 结果为负 |
| D | generation-only GRPO + 原始 PDMS reward | 当前最佳；full navtest `+0.0007060281` |
| E | generation + selection joint GRPO | 弱于 D，延长训练后回退 |

下一轮围绕 D 做递进消融：

- **D**：保持现有 generation-only GRPO，补齐训练 seed 1、2。
- **D+**：在 D 上增加 frozen-reference selector consistency KL，抑制 shared decoder 更新造成的 selector 漂移；不增加 selection policy gradient，不解冻 classification branch。
- **D+S**：在 D+ 上增加 PDM component tie-break reward；原始 PDMS 保持主排序，PDM 分项只打破同分或极近分候选。

## 3. 实现方案

### 3.1 D+ selector consistency

- 新增 `selector_consistency_kl_weight`，默认 `0.0`，保证旧 checkpoint/config 行为兼容。
- generation rollout 在同一 behavior state/action trace 上输出 current 与 fixed-reference selector logits。
- consistency 使用 `KL(reference || current)`；classification branch 参数保持冻结，但允许 KL 约束 shared decoder feature。
- generation 模式中的 selection policy-gradient 权重继续为 0。
- seed-0 第一轮快筛权重 `0.05`、`0.2`；若约束项相对 generation
  objective 尺度过小，则根据实际 loss 比例校准，而不是机械延长训练。

### 3.2 D+S reward shaping

- 新增 `grpo_reward_mode: pdms | pdm_tiebreak`，默认 `pdms`。
- scorer 同时返回 aggregate PDMS 与 collision、drivable、progress、TTC、comfort、direction 分项。
- secondary score 为安全分项均值与官方 weighted-metric 均值的等权组合。
- 每个场景根据不同 PDMS 值的最小正间隔计算 tie-break 系数：最大 `1e-3`，且不超过最小正间隔的四分之一。
- `reward = PDMS + epsilon * secondary`，因此不会反转不同 aggregate PDMS 的主要排序。
- 训练可使用 shaped reward；验证、checkpoint 比较与最终 full navtest 始终使用原始 PDMS。

## 4. 实验与门控

1. 从 fixed common-1024 中固化 256-token 快筛子集，两者均与训练 token 隔离。
2. 所有 seed 固定 DataLoader、diffusion noise 和优化器随机性；评测继续使用 token-derived deterministic noise。
3. checkpoint 先跑 fixed-256；selected 高于 paired base 且 oracle 不低于 base `-0.002` 才进入 fixed-1024。
4. fixed-1024 要求相对 base 的 selected paired bootstrap CI 下界大于 0、oracle 不显著下降、selection regret 不恶化。
5. D+S 只有在 fixed-1024 point estimate 优于 D+ 时才补跑 seed 1、2。
6. 最优方案的 3 个 seed 进入 12,146-token full navtest，最终报告 PDMS 增益均值、标准差、置信区间和全部 PDM 分项。

## 5. 测试与审计

- current/reference 相同时 consistency KL 为 0；发生偏移时为正且有限。
- consistency 对 shared/regression 路径有梯度，classification branch 参数梯度严格为 0。
- generation 模式不产生 selection policy gradient。
- `pdms` reward 模式与当前实现数值一致。
- tie-break reward 不反转不同 PDMS 的排序，并正确处理全零、全同分、无效候选和缺失分项。
- smoke 中 reward、advantage、ratio、KL、梯度均有限；perception 与 classification branch 保持冻结。

## 6. 明确不做

- 本阶段不做 WorldEngine 迁移或 WE 实验。
- 不做 rare/failure-aware sampling；困难场景只作为 full-navtest 附加分析。
- 不新增 selector 网络，不改变 DiffusionDrive 主体架构。
- 不重复 B/C/E，除非新结果与已有结论冲突。

## 7. 2026-07-16 第一轮 D+ 结果与下一步

第一轮使用 seed 0、raw PDMS reward，比较
`selector_consistency_kl_weight=0.05/0.2`。两者在 fixed-256 和
fixed-1024 上几乎完全重合，说明该权重区间没有形成有效约束。

| checkpoint | KL weight | fixed-256 selected delta | fixed-1024 selected delta | fixed-1024 95% CI | fixed-1024 oracle delta |
|---|---:|---:|---:|---:|---:|
| 128 step | 0.05 | +0.001353 | +0.003299 | [+0.000802, +0.006617] | +0.000809 |
| 384 step | 0.05 | +0.005124 | -0.000951 | [-0.006621, +0.004651] | +0.002655 |
| 384 step | 0.20 | +0.005125 | -0.000951 | [-0.006621, +0.004651] | +0.002656 |

结论：256-token 快筛对少量大幅变化样本过于敏感，不能替代 fixed-1024；
384 step 的 oracle 提升但 selected 回退，仍是 generator--selector mismatch。
训练到 384 step 时，未加权 selector consistency KL 约为
`4.1e-5`，而 generation policy loss 绝对值约为 `1.7e-4`；原权重只让
consistency 项达到后者约 1%/5%。下一轮使用 `2.0` 与 `10.0`，分别形成
与 policy objective 同量级和明显更强的约束，仍保持 classification branch
冻结、selection policy gradient 为零。先训练到 384 step 并执行相同
fixed-256/fixed-1024 gate；D+ 未通过前不启动 D+S。

第二轮完成后，384-step fixed-1024 结果如下：

| selector KL weight | selected delta | 95% CI | oracle delta | selection regret |
|---:|---:|---:|---:|---:|
| 0.05 | -0.000951 | [-0.006621, +0.004651] | +0.002655 | 0.163530 |
| 0.20 | -0.000951 | [-0.006621, +0.004651] | +0.002656 | 0.163530 |
| 2.0 | -0.001671 | [-0.007185, +0.003793] | +0.002654 | 0.164248 |
| 10.0 | -0.001448 | [-0.006953, +0.004018] | +0.002304 | 0.163675 |

`10.0` 已把 384-step 未加权 consistency KL 从约 `4.1e-5` 压到
`1.6e-5`，其加权 loss 超过 generation policy loss 的量级，但仍未改善
selected PDMS。因此停止扫描 selector consistency；它不是延长 D 有效训练窗口
的主要手段。保留 128-step D（fixed-1024 `+0.003299`，CI 下界
`+0.000802`）作为当前可靠点。

下一步校准已有的 generation KL trust region。384 step 时未加权
generation KL 约为 `2.3e-4`，当前权重 `0.1` 只贡献约 `2.3e-5`。
比较 `generation_kl_loss_weight=1.0` 与 `5.0`，令约束分别与 policy
objective 同量级和占主导；selector consistency 设为 0，reward 保持 raw
PDMS。仍按 128/256/384 checkpoint、fixed-256 到 fixed-1024 的顺序门控。

generation KL 的 384-step fixed-1024 结果为：

| generation KL weight | selected delta | 95% CI | oracle delta | selection regret |
|---:|---:|---:|---:|---:|
| 0.1 | -0.000951 | [-0.006621, +0.004651] | +0.002655 | 0.163530 |
| 1.0 | -0.001739 | [-0.007242, +0.003737] | +0.002534 | 0.164196 |
| 5.0 | -0.001150 | [-0.006364, +0.004133] | +0.002500 | 0.163573 |

generation trust region 同样未把 oracle 增长转化为 selected 增长，因此停止继续
扩大 KL。下一步进入 reward 层面的 D+S 消融，但不再依赖未通过 gate 的 D+：
使用原始 D 设置（selector consistency `0`、generation KL `0.1`），只把
训练奖励切换为 order-preserving PDM component tie-break。这样可以直接回答
shaping 相对 raw PDMS 是否改善 GRPO 的学习信号；评测仍只使用原始 PDMS。

第一轮 D+S 使用 `pdm_tiebreak_max_epsilon=0.001`。128-step fixed-1024
为 `+0.00329914`，CI `[+0.00080178,+0.00661729]`，与 raw D-128
数值上几乎完全一致；256/384-step 也同样重合。训练中的实际平均 epsilon
仅约 `0.00041--0.00048`，shaped reward std 相比 raw reward std 只增加
约 `1e-4`，经组内 advantage 标准化后不足以改变优化轨迹。

因此做一次有边界的 shaping 强度校准：全局上限提高到 `0.05`，但每场景
仍强制 `epsilon <= min_positive_PDMS_gap / 4`，所以不会反转任何不同 raw
PDMS 候选的排序。先只训练 128 step；只有相对 raw D 产生有意义差异才继续
256/384 step。

强 tie-break `max_epsilon=0.05` 的实际平均 epsilon 仍只有约
`0.00109`（主要受 gap/4 约束），128-step fixed-256 继续与 raw D
重合。因此保序 tie-break 作为训练 shaping 的强度不足，停止继续扫描。

新增独立 `pdm_dense` reward mode：

`reward = raw_PDMS + lambda * (no_collision * drivable) * secondary_components`

其中 `secondary_components` 沿用安全分项均值与官方 weighted-metric
均值的等权组合。该模式允许安全候选间发生重排，但 collision 或 drivable
任一 gate 为 0 时不提供 dense bonus，避免把 aggregate PDMS 为 0 的明显
不安全候选抬高。第一轮只测试 `lambda=0.1`、seed 0、128 step；评测仍为
raw PDMS，只有 fixed-1024 优于 raw D-128 才继续训练或增加 seed。

`lambda=0.1` 在 128 step 的 fixed-256 为 `+0.00135218`，与 raw
D-128 分数接近；但 shaped reward std 为 `0.33436`、raw reward std 为
`0.30193`，且相对 tie-break checkpoint 有 64 个浮点 tensor 发生变化，
最大绝对参数差约 `7.1e-6`。因此它不是数值等价实验。并行运行
128-step fixed-1024，并从头训练到 384 step，重点检查 dense shaping 是否
改变 raw D 在 256/384 step 的 generator--selector mismatch；只有通过
fixed-1024 才进入多 seed。

dense-128 fixed-1024 为 `+0.00329820`，dense-256 fixed-1024 为
`-0.00055766`（oracle `+0.00203459`），分别复现 raw D-128/D-256。
因此 reward shaping 已产生参数差异，但没有改变 generator--selector mismatch；
停止扫描 shaping 权重且不补 dense-384。

下一项直接修改 generation objective 的 mode 聚合方式。原 D 对所有 mode
等权计算 generation policy loss，这更接近优化候选集/oracle，而部署只使用
selector 选中的候选。新增 `selector_softmax` 与 `selector_top1` 权重：
权重来自当前 selector logits 但完全 detach，classification branch 继续冻结，
不产生 selection policy gradient；generation KL 仍约束全部 mode。先比较
seed 0、raw PDMS、128 step，目标是把纯 generation GRPO 更新集中到实际会被
部署 selector 选中的 mode。

mode-weighting 的 128-step 结果：

| weighting | fixed-256 selected delta | fixed-1024 selected delta | fixed-1024 95% CI | oracle delta | regret |
|---|---:|---:|---:|---:|---:|
| uniform D | +0.001353 | +0.003299 | [+0.000802, +0.006617] | +0.000809 | 0.157433 |
| selector softmax | +0.001106 | +0.001950 | [+0.000229, +0.004483] | +0.000195 | 0.158168 |
| selector top1 | -0.002612 | -- | -- | -- | -- |

softmax weighting 证明部署对齐的 generation objective 可以稳定超过 base，但仍弱于
uniform D-128；top1 过于激进。停止继续扫描 weighting/temperature。当前 seed0
最优方案仍为 raw PDMS、uniform mode、128 step 的 D。下一步补齐 D-128 的
train seed 1、2，并在同一 fixed-1024 上要求三个 seed 同向；只有跨 seed
结果成立才进入 full-navtest。

D-128 三个训练 seed 在 fixed-1024 上均同向：`+0.003299`、
`+0.000272`、`+0.001557`。seed-stratified 平均为 `+0.001709`，
CI `[+0.000441,+0.003192]`，但只有 seed0 的单-seed CI 下界大于 0，
未满足事前稳定性标准。下一项只做训练方差控制：保持 raw PDMS、uniform、
128 optimizer update 与其余超参不变，把 batch size 从 1 提到 2，比较三个
训练 seed。目标是收紧 seed 间与逐 token 不确定性，而不是增加训练步数。

batch-size 2 的 fixed-1024 三 seed selected delta 为 `+0.001904`、
`+0.000749`、`+0.001488`，oracle delta 分别为 `+0.000868`、
`+0.001288`、`+0.000287`，全部同向。seed 标准差由 batch-size 1 的
`0.001519` 降到 `0.000585`；seed-stratified CI 为
`[-0.000015,+0.002831]`。虽然 1024-token 总体下界仍极轻微跨 0，但该设置
已达到进入最终大样本评测的工程条件：方向一致、oracle 不退、方差显著降低。
按与 2026-07-15 完全相同的 ray-distributed full-navtest 协议顺序评测三个
seed，复用严格配对的 base CSV；最终标准仍是至少两个单-seed CI 下界大于 0
且 seed-stratified 总体 CI 下界大于 0。

## 8. Batch-size 2 full-navtest（完成）

评测协议与 2026-07-15 的 paired full-navtest 完全一致：`navtest` 的
12,146 个 token、32 个 Ray CPU worker、相同 metric cache、相同
token-derived deterministic diffusion noise，并严格复用 baseline CSV
逐 token 配对。训练设置为 raw PDMS reward、uniform mode weighting、
batch size 2、128 optimizer updates；三个训练 seed 顺序评测，避免多个
Ray job 争用 GPU/CPU。

seed 0 已完成 12,146/12,146，失败数为 0：baseline mean
`0.8491642572`，candidate mean `0.8506098457`，paired delta
`+0.0014455885`，10,000 次 paired bootstrap 95% CI
`[+0.0006516939,+0.0022742746]`，`P(Delta<=0)=0.0001`。逐 token
wins/ties/losses 为 `2277/4644/5225`。PDM 分项平均变化为：collision
`+0.0002058291`、drivable `+0.0012349745`、progress
`+0.0012572039`、TTC `+0.0006586531`、comfort `0`、direction
`+0.0000411658`，主要安全与进度指标均同向。seed 0 单-seed CI 已通过
最终门槛，现继续 seed 1，随后在结果为正时评测 seed 2。

seed 1 同样完成 12,146/12,146，失败数为 0：candidate mean
`0.8505070109`，paired delta `+0.0013427537`，95% CI
`[+0.0005503296,+0.0021685488]`，10,000 次 bootstrap 中没有一次
`Delta<=0`。wins/ties/losses 为 `6250/4645/1251`；collision、drivable、
progress、TTC、comfort、direction 分项变化依次为 `+0.0002058291`、
`+0.0010703112`、`+0.0012574156`、`+0.0004116582`、`0`、
`+0.0001234974`。seed 0、1 的单-seed CI 下界均大于 0，事前标准中的
“至少两个单-seed CI 为正”已经满足；继续评测 seed 2 以完成三 seed
同向性及 seed-stratified 总体 CI。

seed 2 完成 12,146/12,146，失败数为 0：candidate mean
`0.8504346022`，paired delta `+0.0012703450`，95% CI
`[+0.0005862105,+0.0020173883]`，`P(Delta<=0)=0`。wins/ties/losses
为 `6175/4652/1319`；collision、drivable、progress、TTC、comfort、
direction 分项变化依次为 `+0.0002058291`、`+0.0010703112`、
`+0.0011988890`、`+0.0003293265`、`0`、`+0.0001234974`。

最终汇总如下（单-seed CI 均用相同 seed `20260716` 的 10,000 次逐 token
paired bootstrap）：

| train seed | candidate PDMS | paired delta | 95% CI | pass |
|---:|---:|---:|---:|:---:|
| 0 | 0.850609846 | +0.001445588 | [+0.000651694, +0.002274275] | yes |
| 1 | 0.850507011 | +0.001342754 | [+0.000550330, +0.002168549] | yes |
| 2 | 0.850434602 | +0.001270345 | [+0.000586211, +0.002017388] | yes |

三 seed delta 平均为 `+0.0013528957`，sample std 为 `0.0000880609`，
相对 baseline 是约 `+0.1593%`。事前定义的 seed-stratified bootstrap
95% CI 为 `[+0.0009151099,+0.0018116218]`，10,000 次中无 `Delta<=0`。
由于三个训练 seed 复用相同 navtest token，其逐 token delta 相关系数为
`0.71--0.82`；额外使用共同重采样 token 的保守 cluster bootstrap，CI
仍为 `[+0.0006650846,+0.0021075112]`，结论不变。三 seed 平均 PDM 分项
delta 为 collision `+0.0002058291`、drivable `+0.0011251990`、progress
`+0.0012378362`、TTC `+0.0004665459`、comfort `0`、direction
`+0.0000960536`。

结论：raw PDMS、uniform mode weighting、batch size 2、128 optimizer
updates 的 generation-only GRPO 满足全部事前标准：3 个训练 seed 同向，
3/3 单-seed CI 下界大于 0（要求至少 2/3），seed-stratified 与更保守的
token-clustered 总体 CI 下界均大于 0，且 collision、drivable、progress、
TTC 等主要分项没有均值退化。该配置成为当前 DiffusionDrive GRPO 的推荐
训练设置；shaping 保留为合法接口，但现有 dense/tie-break 实验没有优于它。
