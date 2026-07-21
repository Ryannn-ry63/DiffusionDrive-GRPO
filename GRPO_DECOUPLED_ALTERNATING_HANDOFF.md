# DiffusionDrive 解耦交替 GRPO 交接文档

最后更新：2026-07-17
仓库：`/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive`
当前分支：`grpo-selection-v2`
下一阶段目标：将严格配对的 full-navtest 三-seed平均 absolute PDMS 提升推进到 `>= +0.005`

> 这是新会话继续 DiffusionDrive GRPO 工作的最新 source of truth。
> `GRPO_TO_WE_HANDOFF.md` 记录的是 2026-07-15 的历史状态，其中“尚未完成
> full-navtest”等结论已经过时。新会话必须优先阅读本文和
> `GRPO_DIFFUSIONDRIVE_ABLATION_PLAN.md`，再参考旧文档了解历史背景。

## 0. 新会话开始时先做什么

先只读审计工作区，不要立即改代码或启动训练：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
git status --short
git branch --show-current
git log --oneline -5
```

随后依次阅读：

1. `GRPO_DECOUPLED_ALTERNATING_HANDOFF.md`（本文，最新执行入口）；
2. `GRPO_DIFFUSIONDRIVE_ABLATION_PLAN.md`（完整消融过程和最终数值）；
3. `GRPO_TO_WE_HANDOFF.md`（旧阶段背景，不能覆盖本文的新结论）；
4. 当前实现，以代码为最终权威：
   - `navsim/agents/diffusiondrive/diffusion_grpo.py`；
   - `navsim/agents/diffusiondrive/transfuser_model_v2.py`；
   - `navsim/agents/diffusiondrive/transfuser_loss.py`；
   - `navsim/agents/diffusiondrive/transfuser_agent.py`；
   - `navsim/agents/diffusiondrive/transfuser_config.py`。

当前工作区包含本轮尚未提交的代码、脚本、测试和用户文档。它们不是垃圾文件。
禁止执行 `git reset --hard`、`git checkout --`、清理 untracked 文件或覆盖用户改动。
除非用户明确要求，不要自行 commit、push 或切换分支。

## 1. 项目初衷与研究边界

### 1.1 初衷

最核心的问题是：**GRPO 能否在不依赖 imitation/supervised trajectory loss 的情况下，
独立改善 DiffusionDrive 的规划性能？**

因此，GRPO 阶段必须满足：

- 不使用 imitation loss；
- 不使用 GT trajectory regression；
- 不使用 supervised diffusion loss；
- policy reward 来自动态生成候选的 PDM/PDMS 评估；
- 感知模块冻结，只优化后端规划策略；
- 正式结论只看 selector 实际选中轨迹的 raw PDMS；
- oracle 只能用于诊断候选上限，不能当作可部署性能。

这个初衷已经被当前实验初步验证：纯 generation GRPO 在三独立训练 seed 上，
full-navtest 均显著超过严格配对 baseline。下一阶段不是重新证明“是否可行”，
而是沿同一条 GRPO 主线，把提升从约 `+0.00135` 推进到 `+0.005`。

### 1.2 不违背初衷的允许范围

允许：

- PDMS 或 PDM component 作为 GRPO reward；
- reward shaping，只要正式评测仍使用 raw PDMS，并明确报告为 shaping 消融；
- 将 DiffusionDrive 的生成和候选选择视为两个策略，分别用 group-relative
  PDM reward 做 GRPO；
- 采用 reference gate、探索平滑、分阶段/交替更新等 RL 优化手段；
- 对多个纯 GRPO seed 做单模型权重平均，作为附加稳定化实验。

不允许：

- 把 imitation/GT loss 混入 GRPO 阶段后仍声称提升来自纯 GRPO；
- 新增一个未经消融的 selector 网络并把收益归因于 GRPO；
- 用 online oracle/PDM reranking 代替部署 selector 后声称是模型性能；
- 在 full-navtest 上反复调参；
- 当前阶段修改或迁移 WorldEngine。

### 1.3 架构与实验约束

- 保持 DiffusionDrive 主体架构和现有 20 个 trajectory modes；
- 不新增 selector 网络，下一阶段只使用已有 classification branch；
- perception stack 保持冻结；
- 使用 `navsim` conda 环境；
- 先做短程 seed-0 gate，再补多 seed；
- full-navtest 使用相同 token、metric cache、diffusion noise policy 和 worker 协议；
- 多个 32-worker Ray full-navtest 必须顺序运行，避免 GPU/CPU 争用。

## 2. 当前工作区与已经实现的能力

当前 `git status --short` 显示本轮 GRPO 修改尚在工作区中，主要包括：

- `diffusion_grpo.py`：stochastic DDIM、Gaussian log-prob/KL、PDM tie-break 和
  dense shaping；
- `transfuser_model_v2.py`：current/old/reference replay、动态候选 PDM reward、
  generation trace、selector logits 暴露；
- `transfuser_loss.py`：categorical GRPO、generation GRPO、PPO clipping、
  selector consistency、mode weighting；
- `transfuser_agent.py`：训练模式冻结、reference/old policy 初始化和梯度边界；
- `transfuser_config.py` 及 YAML：GRPO/reward/weighting/checkpoint 配置；
- `evaluate_grpo_schedule.py`：固定 token 评测、逐 token artifact、paired bootstrap；
- 训练和评测 runner；
- generation objective、categorical objective、tie-break/dense reward 测试。

当前实现已经具备：

1. stochastic DDIM transition 和 action log-prob；
2. old/current/reference 在同一 sampled state/action trace 上重算概率；
3. categorical selection GRPO、generation GRPO 和 naive joint 模式；
4. PPO clipping、fixed-reference categorical KL、Gaussian generation KL；
5. `pdms`、`pdm_tiebreak`、`pdm_dense` reward mode；
6. generation mode weighting：`uniform`、`selector_softmax`、`selector_top1`；
7. token-derived deterministic evaluation noise；
8. fixed-256、fixed-1024、full-navtest 严格配对分析；
9. checkpoint top-k 可配置和冻结分支梯度审计。

最近记录的验证包括：

- 最新 generation/objective/tie-break 针对性测试：16 passed；
- 更早的完整 focused GRPO suite：20 passed；
- Python compile、runner `bash -n`、`git diff --check` 均通过；
- 真实训练 smoke 中 classification branch gradient 在 generation-only 模式严格为 0。

新会话修改算法后必须重新运行相关测试，不能只依赖上述历史通过记录。

## 3. 已完成实验与 A--E 消融

### 3.1 A--E 主消融

| 组别 | 方法 | 结论 |
|---|---|---|
| A | frozen DiffusionDrive base | full-navtest paired baseline `0.8491642572` |
| B | frozen generator + categorical selection GRPO | 未稳定超过 base |
| C | shared decoder + categorical selection GRPO | fixed-1024 为负 |
| D | generation-only GRPO + raw PDMS | 当前唯一跨三 seed、full-navtest 稳定正向的方法 |
| E | generation + selection naive joint GRPO | 128 step 弱于 D，延长训练后回退 |

E 不能原样重跑。当前 `joint` 模式会同时解冻整个 diffusion decoder 和
classification branch，并用同一个 LR 更新；selection loss 还会穿过 shared
decoder。真实 batch 梯度审计表明 generation 与 selector 确实同时收到梯度，
但这种耦合更新没有改善部署 selected PDMS。

### 3.2 D 的关键早期结果

fixed common-1024：

| checkpoint | selected delta | oracle delta | 解释 |
|---|---:|---:|---|
| D-128 | `+0.003299` | `+0.000809` | selected 最佳 |
| D-256 | `-0.000555` | `+0.002036` | oracle 上升但 selector 未兑现 |
| D-384 | 约 `-0.000951` | 约 `+0.002655` | generator–selector mismatch 继续扩大 |

这是下一阶段最重要的方法动机：候选生成质量仍能提高，但部署 selector
没有跟上，单纯延长 generation GRPO 会让 selected 性能回退。

### 3.3 已经停止的方向

#### Selector consistency KL

测试权重 `0.05`、`0.2`、`2.0`、`10.0`。大权重确实压低 current/reference
selector KL，但 384-step selected 仍为负。结论：仅保持 selector 不漂移，
无法把更高 oracle 转化为更高 selected。

#### Generation KL

测试 `0.1`、`1.0`、`5.0`。更强 trust region 没有解决 selected 回退。
不继续扩大 generation KL。

#### Order-preserving tie-break shaping

`max_epsilon=0.001` 和 `0.05` 都几乎复现 raw D。最小 PDMS gap/4 的保序约束
使实际 shaping 太弱，经组内 advantage 标准化后无法显著改变优化。

#### Dense PDM shaping

`pdm_dense` 确实改变了 reward 统计和模型参数，但 dense-128/dense-256
仍复现 raw D 的走势，没有解决 selector mismatch。shaping 接口保留，
但不要继续无目的扫权重。

#### Selector-derived generation weighting

| weighting | fixed-1024 selected delta | 结论 |
|---|---:|---|
| uniform | `+0.003299` | 当前最好 |
| selector softmax | `+0.001950` | 稳定正向，但弱于 uniform |
| selector top1 | fixed-256 `-0.002612` | 过于激进，停止 |

#### Sequential selector alignment

从早期 D-128 checkpoint 只训练最后 classification branch 的 S1/S2/S3
均未超过未对齐的 D-128。旧 categorical PPO 按 old selector 概率加权，
低概率但高 reward 的候选获得的有效梯度较弱；这正是下一阶段引入
full-support selector policy 的原因，而不是重复原有 sequential selector。

### 3.4 Batch size 2 方差控制

保持 raw PDMS、uniform、128 optimizer updates，其余超参不变，把 batch size
从 1 提到 2：

- fixed-1024 三 seed selected delta：`+0.001904`、`+0.000749`、`+0.001488`；
- 三 seed oracle delta 全部为正；
- seed 标准差从 `0.001519` 降到 `0.000585`。

因此 batch size 2 成为当前推荐默认值。它主要改善稳定性，不足以单独把均值
推到 `+0.005`。

## 4. 已完成的最终 full-navtest 证据

### 4.1 配对 baseline

Base checkpoint：

```text
/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model
```

Base CSV：

```text
/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/paired_full_navtest_base_20260715/2026.07.15.09.04.23/2026.07.15.09.24.58.csv
```

Baseline raw PDMS：`0.8491642572`。

### 4.2 三个已验证 GRPO checkpoints

共同配置：

- `grpo_training_mode=generation`；
- raw PDMS；
- uniform mode weighting；
- batch size 2；
- 128 optimizer updates；
- LR `1e-6`；
- generation KL weight `0.1`；
- diffusion schedule `[8,0]`；
- final action std `0.05`。

Checkpoints：

```text
seed 0:
/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/grpo_d_seed0_skl0.0_gkl0.1_eps0.001_dense0.1_mweightuniform_bs2/2026.07.16.14.53.31/lightning_logs/version_0/checkpoints/grpo-00-128.ckpt

seed 1:
/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/grpo_d_seed1_skl0.0_gkl0.1_eps0.001_dense0.1_mweightuniform_bs2/2026.07.16.14.53.31/lightning_logs/version_0/checkpoints/grpo-00-128.ckpt

seed 2:
/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/grpo_d_seed2_skl0.0_gkl0.1_eps0.001_dense0.1_mweightuniform_bs2/2026.07.16.14.53.31/lightning_logs/version_0/checkpoints/grpo-00-128.ckpt
```

### 4.3 Full-navtest 结果

每个 checkpoint 都完成 12,146/12,146 scenarios，0 failure：

| train seed | candidate PDMS | paired delta | paired bootstrap 95% CI |
|---:|---:|---:|---:|
| 0 | `0.8506098457` | `+0.0014455885` | `[+0.0006516939,+0.0022742746]` |
| 1 | `0.8505070109` | `+0.0013427537` | `[+0.0005503296,+0.0021685488]` |
| 2 | `0.8504346022` | `+0.0012703450` | `[+0.0005862105,+0.0020173883]` |

汇总：

- 三 seed delta 平均：`+0.0013528957`；
- sample std：`0.0000880609`；
- 相对 baseline：约 `+0.1593%`；
- seed-stratified 95% CI：`[+0.0009151099,+0.0018116218]`；
- 更保守的 token-clustered CI：`[+0.0006650846,+0.0021075112]`；
- 三个单-seed CI 下界全部大于 0。

三 seed 平均 PDM component delta：

- collision：`+0.0002058291`；
- drivable：`+0.0011251990`；
- progress：`+0.0012378362`；
- TTC：`+0.0004665459`；
- comfort：`0`；
- direction：`+0.0000960536`。

正式结论：**纯 generation GRPO 已经在 DiffusionDrive 上实现可复现、统计显著的
selected raw PDMS 提升。** 不应说“没有效果”，但也必须承认当前 absolute
delta 只有约 `+0.00135`，距离下一阶段 `+0.005` 目标仍有明显差距。

## 5. 新诊断：为什么提升仍然不够

### 5.1 Generator–selector mismatch

更长 generation 训练提高 oracle、candidate mean，却让 selected 回退。
当前主要瓶颈不是完全生成不出好候选，而是现有 selector 没有稳定选择更好的候选。

fixed-1024 base 的 selected 约 `0.757878`，oracle 约 `0.917801`，
selection regret 很大。oracle 使用未来 PDM 信息，不能直接部署，这个 gap
并非全部可学习；但它说明候选集合仍包含大量未被 selector 兑现的上限。

### 5.2 收益集中在少量失败场景

对三 seed full-navtest delta 取均值后按 baseline 分桶：

| baseline bucket | token 数 | 平均 delta |
|---|---:|---:|
| score < 0.5 | 1,060 | `+0.0191626` |
| 0.5 <= score < 0.8 | 601 | `+0.0022007` |
| 0.8 <= score < 0.95 | 5,150 | `-0.0007927` |
| score >= 0.95 | 5,335 | `-0.0002100` |
| collision fail | 235 | `+0.0097872` |
| drivable fail | 812 | `+0.0213455` |
| TTC fail | 646 | `+0.0074101` |
| all safety pass | 10,759 | `-0.0004827` |

解释：

- 总体提升主要来自少数安全失败被大幅修复；
- 大量正常高分场景发生微小退化；
- current group-relative advantage 只比较组内候选，即使某场景所有新候选都
  不如 reference，也会强化“组内相对最好”的候选；
- 继续均匀延长训练很可能扩大这种 trade-off。

因此下一阶段需要 reference-improvement gate：只有存在相对固定 reference
的实际 headroom 时才进行 policy update，正常场景主要由 KL 保真。

### 5.3 Selector GRPO 的低概率候选覆盖不足

现有 categorical objective 使用 old selector probability 作为 mode weight。
即使所有 20 个候选 reward 都已知，当前低概率但高 PDMS 的 mode 仍得到很小权重。
下一阶段通过给 selector policy 加 uniform valid-mode probability floor，
在不改变部署 argmax 的情况下增加这些候选的梯度覆盖。

### 5.4 Model soup 的可行性

三个成功 seed 的 4,227,138 个 trainable decoder 参数更新：

- 两两 cosine similarity 约 `0.23--0.31`；
- uniform mean update 与各 seed cosine 约 `0.70--0.74`；
- mean update 保留约 72% 的平均更新范数。

方向不是高度一致，但也不是互相抵消。uniform model soup 值得先做一次低成本
fixed-dev 评测；它是附加稳定化手段，不替代三-seed训练证据。

## 6. 下一方法：Decoupled Alternating GRPO

论文和实现建议统一使用以下逻辑：

```text
DiffusionDrive baseline
  -> generation-only GRPO (D，已证明可行)
  -> naive joint GRPO (E，梯度耦合失败)
  -> reference-gated generation GRPO
  -> full-support selector GRPO
  -> decoupled alternating generation/selection GRPO
```

这不是突然加入无关的监督模块，而是把 DiffusionDrive 的两个决策拆成两个
reward-driven policies：

- generation policy：stochastic diffusion denoising actions；
- selection policy：20 个候选上的 categorical action。

两者都只使用 group-relative PDM reward，不使用 imitation/GT。

### 6.1 显式 selector-only 模式

新增清晰的 selector-only 训练入口，可作为现有 `classification_head` 的兼容别名
或新 `grpo_training_mode=selector`：

- perception 冻结；
- plan anchor/context 冻结；
- shared decoder 和 regression branch 冻结；
- 只更新最后现有 `plan_cls_branch`；
- 不新增网络；
- generator 输出必须 detach，selector loss 不得回传到 generation/shared。

### 6.2 Full-support categorical GRPO

对于 valid modes，定义均匀分布 `U_valid`，并对 current/old selector policy
同时平滑：

```text
pi_tilde = (1 - epsilon) * pi + epsilon * U_valid
```

要求：

- PPO behavior weight 使用 smoothed old policy；
- PPO ratio 使用 smoothed current/old policy；
- current=old 时 ratio 必须严格为 1；
- invalid modes 必须正确 mask 并重新归一化；
- 部署仍使用原始 selector logits/argmax，uniform mixing 不进入推理；
- `selection_exploration_floor=0` 保持旧行为兼容。

第一轮只比较：

- `epsilon=0.1`；
- `epsilon=0.3`。

不做更大超参网格。选择标准是 fixed-1024/dev-4096 selected PDMS、regret、
oracle hit、entropy 和 paired CI，而不是训练 loss。

### 6.3 Reference-improvement gate

generation 阶段额外无梯度运行 fixed-reference policy，在相同 scene/noise policy
下得到 reference 部署 selector 所选轨迹，并动态计算 raw PDMS `r_ref`。

对当前 group：

```text
h = max_i(r_i) - r_ref
w = clamp((h - 0.01) / 0.05, 0, 1)
```

行为：

- generation/selector per-scene policy loss 乘 `w`；
- `w=0` 时该场景不产生 policy gradient，只保留对应 reference KL；
- reward、headroom 和 gate weight 必须 detach；
- 正式 checkpoint/evaluation 仍只看 raw PDMS；
- 日志同时记录 gated 和 ungated raw metrics。

selector-only 阶段 generator 固定，可使用当前候选 group 中 reference selector
所选 reward 计算 selector headroom。generation 阶段则需要真正额外评分
fixed-reference selected trajectory，不能用 oracle 或当前 selector selected
reward 冒充 reference baseline。

新增配置建议：

- `selection_exploration_floor: float = 0.0`；
- `grpo_scene_weight_mode: uniform | reference_headroom`；
- `grpo_reference_gate_margin: float = 0.01`；
- `grpo_reference_gate_scale: float = 0.05`。

新增日志：

- `reference_selected_reward`；
- `oracle_headroom`；
- `grpo_scene_weight`；
- `active_scene_fraction`；
- generation/shared/classification gradient norms；
- safe/failure bucket raw PDMS（评测侧）。

### 6.4 解耦交替训练

不用一个 optimizer step 同时更新 generation 和 selector。正式 schedule：

```text
G128 -> S64 -> G64 -> S64
```

默认设置：

- batch size 2；
- raw PDMS；
- generation mode weighting uniform；
- G LR `1e-6`；
- S LR `1e-6`；
- generation KL weight `0.1`；
- selector categorical KL 使用现有 `kl_loss_weight=0.01`；
- old policy sync interval 32；
- diffusion schedule `[8,0]`；
- final std `0.05`。

冻结要求：

- G 阶段 classification parameter grad 必须严格为 0；
- S 阶段 generation/shared/regression grad 必须严格为 0；
- 每个阶段用输入 checkpoint 初始化 current 和 old policy；
- fixed reference 始终来自正式 base checkpoint；
- 每个阶段单独保存 checkpoint、optimizer 设置和 token manifest；
- 每个阶段结束立即评测，不能等所有阶段训练完才检查。

## 7. 决策完整的实验顺序

### Phase 0：冻结新 dev 集

在任何新训练前，从**纯 val split** 构建固定 dev-4096。这里不能只看某个
cache 目录的表面 token 数量，也不能在 val 现成交集不足时直接从 train 池补齐。

#### 2026-07-17 cache 审计结论

对默认 `training_cache`、`metric_cache_trainval` 和官方 train/val log split
逐 token 核对后的数量如下：

| 状态 | train | val |
|---|---:|---:|
| 已有 DiffusionDrive feature/target | 85,109 | 18,179 |
| 已有 PDM metric cache | 41,084 | 6,866 |
| 两种 cache 的现成交集，可立即用于 GRPO/评测 | 6,119 | 1,338 |

现有 fixed-1024 全部属于上述 1,338 个 val 交集，因此现成且未被 fixed-1024
占用的 val token 只有 314 个。这个事实**不表示训练数据总量只有 7,457**；
`7,457 = 6,119 + 1,338` 只是两种 cache 的当前交集。

metric cache 中另有 5,528 个纯 val token 已经可以计算 PDM reward，只缺
DiffusionDrive feature/target。排除 fixed-1024 后，纯 val 候选总数为
`6,866 - 1,024 = 5,842`，足够构建与 fixed-1024 完全独立的 dev-4096。

#### 正确的 dev-4096 构建协议

1. 从 6,866 个带 metric cache 的纯 val token 中排除 fixed-1024，得到 5,842
   个候选；不得从 6,119 个 rewardable train token 中补 token。
2. 使用固定 seed 的 hash 排序，并尽量按 log 分层，从完整 5,842 候选中选择
   4,096 个；不要简单拼接“现成 314 + 缺 cache 3,782”，以免 cache 来源造成
   隐含采样偏差。
3. 只为所选 dev token 中缺失的样本补建 DiffusionDrive feature/target cache。
   优先写入独立 dev cache；如果复用默认 cache，也必须依靠 val log 边界和显式
   token manifest 防止训练误采样。
4. `run_training.py` 的 train/val dataset 分别由 `cfg.train_logs`、`cfg.val_logs`
   构建；必须保留该边界，并增加 manifest 交集断言。补建纯 val cache 不得改变
   现有 6,119-token rewardable train pool。
5. dev manifest 必须记录 token、所属 log、split、选择 seed/算法、数量和 SHA；
   evaluator 必须通过 `--tokens-file` 使用该 manifest，不能依赖默认字典序截断。
6. 构建完成后验证 dev-4096 与 train、fixed-1024、navtest 的 token 交集均为 0，
   并验证每个 dev token 同时存在 feature、target 和 metric cache。
7. 后续所有新方法调参只用 fixed-256、fixed-1024 和 dev-4096；不再用
   full-navtest 选择 epsilon、gate 或 stage 数。

禁止采用 `314 val + 3,782 train` 的 fallback。该方案会把 rewardable train pool
从 6,119 缩到 2,337，显著改变训练分布，使新结果无法与既有 D-128 公平比较。
如果缺失的纯 val feature/target 无法生成，应停止并记录数据依赖问题；不得静默
改用 train token。只有在明确修订整个实验协议并重跑所有对照时，才能考虑改变
训练池或缩小 dev 集。

### Phase 1：Uniform model soup

使用三个已经通过 full-navtest 的 batch-size-2 seed：

- 只平均 current `diff_decoder` 浮点权重；
- backbone 和其他相同权重保持 base/current一致；
- 不平均 `old_policy`、`ref_policy` snapshot；
- 重新从 base checkpoint 初始化 reference；
- 输出一个单模型 soup checkpoint。

评测 fixed-256、fixed-1024、dev-4096。只有 fixed-1024 和 dev-4096
selected delta 都达到 `+0.005` 且 CI 下界大于 0，才允许做一次 final
full-navtest。soup 是附加实验，不用于替代多 seed 显著性证明。

### Phase 2：Seed-0 核心消融

按以下顺序，后一项只有在前一项代码和梯度审计通过后启动：

| ID | 初始化 | 训练 | 目的 |
|---|---|---|---|
| D | base | 现有 G128 | 已验证基线，不重复无意义训练 |
| D+Gate | base | reference-gated G128 | 验证正常场景保真 |
| D+S0.1 | best G128 | selector-only S64，epsilon 0.1 | 验证低概率 mode 覆盖 |
| D+S0.3 | best G128 | selector-only S64，epsilon 0.3 | 有边界的探索强度对比 |
| Decoupled-GS | best gated G | `G128 -> S64 -> G64 -> S64` | 主方法 |

如果 selector S64 的训练尚未收敛但 dev 指标持续提高，可延长到 S128；如果 S64
已经回退，不得仅凭训练 loss 继续延长。

### Phase 3：评测门控

fixed-256 只负责快速淘汰：

- selected delta `<=0`，直接停止；
- oracle delta < `-0.002`，直接停止；
- NaN、invalid reward、冻结梯度越界，直接停止。

fixed-1024 和 dev-4096 才能晋级。主门槛：

- selected delta `>= +0.005`；
- paired bootstrap CI 下界 > 0；
- oracle 不低于对应 baseline；
- selection regret 降低；
- safety-pass bucket delta `>=0`；
- collision/drivable/TTC 均值无退化。

如果结果达到 `+0.003` 且显著，但未到 `+0.005`，保留 checkpoint 和分析，
但不能提前启动 full-navtest；继续执行下面预先规定的有限升级。

### Phase 4：有限升级与停止条件

记录 `active_scene_fraction = mean(w>0)`。

如果 active fraction < 25%：

- 只使用 navtrain token 构建 headroom priority manifest；
- 不得使用 navtest 或 dev token；
- batch 固定为 50% uniform + 50% high-headroom；
- 重跑一次 gated G128；
- manifest 和 SHA 必须保存。

如果 active fraction >= 25% 但仍未达到 `+0.005`：

- 最多追加一轮 `G64 -> S64`；
- 不再扫描 shaping、KL 或更多 epsilon。

连续两个 stage 的 dev-4096 selected 提升都 < `0.0005` 时停止此路线，
形成负向结论，不用无限延长训练。

### Phase 5：多 seed 与最终 full-navtest

只有 seed-0 在 fixed-1024 和 dev-4096 都通过 `+0.005` 门槛后：

1. 完整训练 seed 1、2；
2. 使用相同 token manifests、stage schedule、hyperparameters；
3. 要求三个 seed selected delta 同向；
4. 至少两个单-seed paired bootstrap CI 下界 > 0；
5. seed-stratified CI 下界 > 0；
6. 再锁定唯一配置进入 full-navtest。

最终 full-navtest 成功标准：

- 三-seed平均 absolute PDMS delta `>= +0.005`；
- 至少两个单-seed CI 下界 > 0；
- seed-stratified CI 和 token-clustered CI 下界都 > 0；
- collision、drivable、TTC 均值不退化；
- safety-pass bucket 不再出现系统性负向；
- 每个 seed 12,146/12,146 success，0 failure。

## 8. 必须补充的测试

### 8.1 Full-support selector policy

- epsilon=0 与旧实现数值兼容；
- valid mode 概率和为 1；
- invalid modes 概率为 0；
- current=old 时 ratio=1；
- mixing 不改变原始 logits argmax；
- 低 old-prob、高 advantage mode 获得非零有限梯度；
- epsilon 非法值（<0 或 >=1）明确报错；
- 全 invalid、单 valid、zero-advantage group 安全处理。

### 8.2 Reference gate

- 所有候选都不优于 reference 时，policy gradient 为 0；
- 存在 headroom 时梯度有限且非零；
- margin/scale 边界正确；
- gate weight 和 reference reward detach；
- reference rollout 与 PDM scoring 无梯度；
- extra reference candidate 不被误计入 generation action log-prob；
- raw/shaped reward 日志不混淆。

### 8.3 冻结与 stage 切换

- G 阶段 classification grad=0；
- S 阶段 shared/regression/generation grad=0；
- old policy 从 stage input checkpoint 初始化；
- fixed reference 从正式 base 初始化；
- checkpoint reload 后 training mode 和 requires_grad 正确恢复；
- optimizer 只包含当前 stage 允许的参数。

### 8.4 评测与数据

- dev-4096 与 train/common/navtest token 交集为 0；
- dev-4096 的每个 token 都属于配置中的 `val_logs`，且不属于 fixed-1024；
- dev manifest 从排除 fixed-1024 后的完整 5,842-token val metric 候选池中按固定
  hash/分层规则生成，不能按 cache 是否现成选择；
- 每个 dev token 同时具备 feature、target 和 metric cache；
- 补建 dev cache 前后 rewardable train pool 均为 6,119，训练 token manifest
  不发生变化；
- token SHA 可复现；
- fixed evaluator 保存 selected、oracle、regret、entropy、headroom；
- paired artifact 一一按 token join，禁止按 CSV 行号配对；
- bootstrap seed 和样本数固定并记录；
- full-navtest 同时报告 seed-stratified 与 token-clustered CI。

### 8.5 完成后的检查

至少运行：

- focused GRPO pytest；
- 新 selector/gate/stage tests；
- `python -m py_compile`；
- runner `bash -n`；
- `git diff --check`；
- 一批真实数据 smoke 和梯度审计。

## 9. 论文中的可声明结论与不可声明结论

### 当前已经可以声明

- 在不使用 imitation/GT trajectory loss 的情况下，generation GRPO 能够在
  DiffusionDrive 上带来三-seed、full-navtest、统计显著的 selected PDMS 提升；
- batch size 2 提高了训练稳定性；
- reward shaping、简单 selector consistency、单纯加强 KL 和 naive joint
  没有解决 generator–selector mismatch；
- 当前收益主要体现在部分安全失败场景。

### 当前不能声明

- 不能把 oracle score 当作部署性能；
- 不能声称当前已达到 `+0.005`；
- 不能声称 shaping 优于 raw PDMS；
- 不能把未来 reward-ranking/online reranking 的收益全部归因于纯 GRPO；
- 不能使用 navtest 调参后仍把它描述为未见测试集；
- 不能因为单 seed 或 fixed-256 正向就声称方法稳定。

### 下一阶段预期论文消融

建议最终表格至少包含：

1. Base；
2. D generation-only GRPO；
3. E naive joint；
4. D + reference gate；
5. D + full-support selector GRPO；
6. decoupled alternating GS；
7. optional uniform model soup；
8. tie-break/dense shaping 的负向消融。

这样可以清楚说明：GRPO 本身可行，简单 joint 为什么失败，以及解耦、探索覆盖和
reference-conservative update 分别贡献了什么。

## 10. 明确不要做的事情

- 不再重复 B/C/E 的原始配置；
- 不继续无边界扫描 selector consistency、generation KL 或 shaping weight；
- 不用 fixed-256 作为晋级 full-navtest 的唯一证据；
- 不并行运行多个 full-navtest；
- 不在 navtest 上选择 epsilon、gate threshold 或 alternating stage 数；
- 不改变主体架构或新增 selector 网络；
- 不加入 imitation/supervised diffusion loss；
- 不迁移 WorldEngine；
- 不破坏或清理当前 dirty worktree；
- 不因现成 val cache 交集不足而从 train 池切走 3,782 个 token；
- 不在没有逐 token paired artifact 的情况下比较两个 aggregate score。

## 11. 新会话第一轮实际执行清单

1. 审计 git/worktree 和所有现有文件；
2. 阅读本文、最新 ablation plan 和当前代码；
3. 把本文计划追加/同步到实验日志，避免结果只存在聊天上下文；
4. 按 Phase 0 的 cache 审计结果生成纯 val dev-4096 manifest，选择性补建缺失的
   feature/target，并完成 split/cache/交集测试；
5. 实现并测试 uniform model soup builder，完成低成本 gate；
6. 实现 selector policy smoothing helper 和单元测试；
7. 实现 reference selected trajectory scoring、scene gate 和日志；
8. 增加显式 selector-only runner 与冻结审计；
9. seed-0 按 D+Gate、S0.1、S0.3、Decoupled-GS 顺序执行；
10. 每个 stage 更新交接文档和结果表，再决定是否进入下一 gate。

如果新会话发现代码实际状态与本文冲突，以当前代码、checkpoint artifact 和 CSV
为准，但必须先解释冲突并更新本文，不能静默采用旧文档结论。

## 12. 2026-07-17 实际执行记录

本节记录已经真正落盘或运行的工作，不是待办计划。后续会话不得重复构建或把它们
误写为“尚未开始”。

### 12.1 Phase 0：纯 val dev-4096 已完成

- 新增 `scripts/evaluation/build_grpo_dev_manifest.py`，按 val log 分层并使用固定
  SHA256 排序选样；新增对应测试 `test_grpo_dev_manifest.py`。
- manifest：`artifacts/grpo_stage0/dev4096_manifest.json`；
- ordered token SHA256：
  `1feab82ea71cce9c44bd50c0ae0e36abe6c2259ed1e2bf6f40d8c653c0071a46`；
- 候选为排除 fixed-1024 后的 5,842 个纯 val metric-ready token，最终选 4,096；
- 选择时 214 个已有 feature/target、3,882 个缺失；缺失项随后全部补建成功；
- 独立 feature cache：
  `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/grpo_dev4096_feature_cache_20260717`；
- 审计 artifact：`artifacts/grpo_stage0/dev4096_cache_audit.json`；
- 最终 4,096/4,096 同时具备 feature、target、metric cache；missing=0；
- 与 rewardable train、fixed-1024、navtest token 的交集均为 0；
- rewardable train pool 前后保持 6,119，没有从 train 切走 token。

`run_dataset_caching.py` 已支持 JSON manifest、`tokens_limit` 和基于 record 中
`log_name` 的日志过滤，避免为少量 token 扫描全部日志。

### 12.2 Phase 1：uniform model soup 已完成并按门控淘汰

- builder：`scripts/evaluation/build_grpo_model_soup.py`；
- checkpoint：`artifacts/grpo_stage1/uniform_bs2_d128_soup.ckpt`；
- metadata：`artifacts/grpo_stage1/uniform_bs2_d128_soup.json`；
- 仅平均三个成功 seed 的 current `diff_decoder`：84 个浮点 tensor、
  4,227,138 个参数；未平均 old/reference snapshot；
- output SHA256：
  `8cad4748e0fe407428b03414098c84d14b02a18c08a0135b32679b540db2d2d7`；
- fixed-256 artifact：`artifacts/grpo_stage1/soup_fixed256.json`；
- fixed-256 selected delta `+0.0011888912`，CI 下界 `+0.0000040756`，但 oracle
  delta `-0.0027583467`；
- 因 oracle `< -0.002` 触发预设淘汰条件，未继续 fixed-1024/dev/full-navtest。

该结果作为低成本负向消融保留，不再重复 soup 扫描。

### 12.3 解耦算法实现与测试

已实现：

- 显式 `grpo_training_mode=selector`，只解冻最后 classification branch；
- valid-mode full-support selector policy：current/old/reference 同规则平滑，
  PPO ratio 和 behavior weight 使用平滑 policy，部署 argmax 不变；
- fixed-reference selected raw PDMS 动态评分和 reference-headroom scene gate；
- gate 只乘 policy loss，reference KL 不乘 gate；
- `reference_selected_reward`、`oracle_headroom`、`active_scene_fraction` 和三组
  decoder gradient norm 日志；
- 评测 artifact 保存 selector 所选轨迹的六项 PDM component，并在新 baseline
  artifact 可用时计算 safety-pass bucket delta；
- 通用阶段 runner：
  `scripts/training/run_diffusiondrive_grpo_decoupled_stage.sh`；
- dev-4096 runner：
  `scripts/evaluation/run_diffusiondrive_grpo_dev_gate.sh`。

新增 selector/gate objective 测试与既有 generation/objective 测试合跑为
`21 passed`。component artifact 已通过 2-token GPU evaluation smoke。

### 12.4 真实 batch 梯度边界审计

gated-G 8-batch smoke：

- active scene fraction：`0.4375`；
- mean oracle headroom：`0.0722683`；
- shared grad norm：`4.63153`；
- regression grad norm：`18.76228`；
- classification grad norm：严格 `0.0`。

selector-S（从已验证 D-128、epsilon=0.1）8-batch smoke：

- trainable parameters：132,865，仅最后 classification branch；
- active scene fraction：`0.9375`；
- mean oracle headroom：`0.141462`；
- shared grad norm：严格 `0.0`；
- regression grad norm：严格 `0.0`；
- classification grad norm：`1.04582`；
- total diff-decoder grad norm 与 classification grad norm 相同。

因此 G/S 两侧都已覆盖“允许组非零、冻结组严格为零”的真实数据反向路径。

### 12.5 Seed-0 核心消融与停止决策

#### D+Gate：从 base 训练 reference-gated G128

checkpoint：

```text
/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/
grpo_d_gate_seed0_bs2_g128_20260717/2026.07.17.05.18.59/
lightning_logs/version_0/checkpoints/grpo-00-128.ckpt
```

checkpoint 内 `global_step=128`，128 个 validation batch 已完成。网络中断发生在
正式 checkpoint 写完之后、`last.ckpt` 收尾之前，不影响该 checkpoint 的完整性。
记录到的所有 classification grad 均严格为 0。

fixed-256：

- selected delta：`+0.0010585070`；
- paired CI：`[-0.0002539898,+0.0033947753]`；
- oracle delta：`-0.0030656906`；
- artifact：`artifacts/grpo_stage2/d_gate_seed0_g128_fixed256.json`。

oracle delta 低于预设 `-0.002` 淘汰线，因此停止 D+Gate，不跑 fixed-1024、
dev-4096 或 full-navtest。

#### Full-support selector S64：epsilon 0.1 与 0.3

两条都从此前已验证的 raw D-128 seed-0 checkpoint 初始化，只更新 132,865 个
classification 参数。fixed-256 两者部署结果完全相同：

- selected delta：`+0.0012665123`；
- CI：`[+0.0000411296,+0.0035750048]`；
- oracle delta：`-0.0016528177`。

fixed-1024 两者仍得到完全相同的部署 argmax 和 raw PDMS：

- selected delta：`+0.0019419930`；
- CI：`[-0.0009790358,+0.0052484025]`；
- oracle delta：`+0.0008679464`；
- selection regret：`0.1588488651`；
- artifacts：
  `artifacts/grpo_stage2/d_s01_seed0_s64_fixed1024.json`、
  `artifacts/grpo_stage2/d_s03_seed0_s64_fixed1024.json`。

相对输入 D-128 的 fixed-1024 delta（约 `+0.001904`），selector S64 只增加约
`+0.000038`；未达到 `+0.005` 且 CI 下界小于 0，因此不跑 dev-4096。epsilon
确实改变训练 policy/entropy，但在 1,024 个固定 token 上没有改变部署 argmax；
当前证据不支持扩大 epsilon 网格。

#### Decoupled alternating：D128 -> S64 -> gated G64 -> S64

由于从 base 的 gated-G 已按 oracle 门控淘汰，交替实验采用不改变首阶段预算的
保守初始化：把此前已验证 D-128 视为首个 G128，接 epsilon=0.1 S64，然后执行
gated G64 和最后一个 epsilon=0.1 S64。fixed reference 始终为正式 base。

中间 gated G64 fixed-256：

- selected delta：`+0.0011974834`；
- CI：`[-0.0000876798,+0.0035397389]`；
- oracle delta：`-0.0018412187`。

完整 alternating checkpoint：

```text
/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/
grpo_decoupled_seed0_d128_s01_g64_s01_20260717/2026.07.17.07.49.59/
lightning_logs/version_0/checkpoints/grpo-00-64.ckpt
```

完整方法 fixed-1024：

- selected delta：`+0.0016838466`；
- CI：`[-0.0013362367,+0.0050003713]`；
- oracle delta：`+0.0008299450`；
- selection regret：`0.1590690102`；
- artifact：
  `artifacts/grpo_stage2/decoupled_d128_s01_g64_s01_fixed1024.json`。

完整 alternating 弱于单次 selector S64 的 `+0.0019419930`，也未达到
`+0.005`、CI 下界不大于 0。按照预先规定的门控和停止原则：

- 不运行 dev-4096；
- 不补 seed 1/2；
- 不运行 full-navtest；
- 不再追加 G64/S64、扫描更多 epsilon、KL 或 shaping 权重。

本轮结论是：reference gate、full-support selector 和严格 G/S 解耦的工程行为均
验证正确，但在既定 seed-0 门控上没有把原始 D 的约 `+0.002` 提升扩大到
`+0.005`。这些结果应作为有价值的负向消融保留；下一轮若继续冲击性能，需要提出
新的、预先可检验的机制假设，而不是延长当前 schedule。

## 13. Full-information selector GRPO 执行结果（2026-07-17）

为直接解决旧 categorical PPO 使用 old-policy probability 加权、从而压低
低概率高 PDMS mode 梯度的问题，新增 `selection_behavior_weighting`：

- `old_policy` 为完全兼容的默认行为；
- `uniform_valid` 在所有 valid mode 上使用均匀 behavior measure，同时保留
  group-relative PDMS advantage、current/old ratio、PPO clipping、reference KL
  和 raw-PDMS 正式评价。

该方法是 full-information/counterfactual categorical GRPO，不使用 GT、imitation、
soft label 或新增 selector 网络。详细预注册计划与结果见
`GRPO_FULL_INFORMATION_SELECTOR_PLAN.md`。

工程验证：focused tests `24 passed`；真实 8-update smoke 和正式 S64 均满足只更新
classification branch。正式 S64 checkpoint：

```text
/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/
grpo_fullinfo_selector_s64_seed0_20260717/2026.07.17.10.06.45/
lightning_logs/version_0/checkpoints/grpo-00-64.ckpt
```

正式训练中 classification grad norm `6.373736`，shared/regression grad norm 均严格
为 `0`，active scene fraction `0.890625`，说明训练和 gate 都实际生效。

相对同一 D128 的严格配对结果：

| split | selected delta | oracle delta | mode switches | switched mean |
|---|---:|---:|---:|---:|
| fixed-256 | `+0.0000849147` | `0.0` | `1/256` | `+0.02173817` |
| fixed-1024 | `+0.0000212287` | `0.0` | `1/1024` | `+0.02173817` |

fixed-1024 中 safety components 全部不变，safety-pass bucket delta 为
`+0.0000239407`。唯一切换有益，但切换率只有 `0.09765625%`。selector probability
L1 变化均值 `0.002460`，margin 绝对变化均值 `0.002243`，而 D128 margin 中位数为
`0.293122`；current classification 参数相对更新范数只有 `6.47e-5`。

因此该机制解决了 objective 中的显式 mode-coverage 问题，却没有解决冻结 selector
表示下的大决策边界。它未达到预设 `1%` mode-switch 晋级线，故停止 S128、dev、
额外 seeds 和 full-navtest，也不事后扫描 LR/epsilon/KL/shaping。下一阶段若继续，
应先设计 selector representation/calibration 的可检验机制，而不是继续延长当前
categorical GRPO schedule。

## 14. GRPO + PDMS preference ranking（2026-07-17）

完整计划、公式和 artifacts 见 `GRPO_PREFERENCE_RANKING_PLAN.md`。本阶段保持
generator 和 shared decoder 冻结，不新增网络，用 raw PDMS 构造 reward-gap weighted
pairwise ranking，并完成同 LR `1e-5` 的 GRPO-only、rank-only、hybrid 三组 S64。

实现后 focused tests `34 passed`。真实 smoke 和正式训练中只有 classification
branch 有梯度；hybrid 正式 classification grad norm `11.565546`，shared/regression
均严格为 `0`。

fixed-256 三组均为正且无 harmful safety switch；fixed-1024 结果为：

| 配置 | vs D128 | vs base | mode switches | collision delta |
|---|---:|---:|---:|---:|
| GRPO-only | `+0.000117607` | `+0.002021914` | `12/1024` | `0` |
| rank-only | `-0.000473159` | `+0.001431148` | `24/1024` | `-0.0009765625` |
| hybrid | `-0.000468153` | `+0.001436155` | `24/1024` | `-0.0009765625` |

所有 oracle delta 为 `0`。ranking 将切换率提高约一倍，rank-only 有 21 次 beneficial
和 2 次 harmful，hybrid 有 20 次 beneficial 和 2 次 harmful；但两者都在 token
`b0ad1a8107ad54dc` 从安全 mode 6 切到 collision-fail mode 13，单 token PDMS
下降 `-0.695360`，压过所有小幅正收益。GRPO 项没有避免这一共同失败。

Artifacts 位于 `artifacts/grpo_stage4/`，分别包含三组 fixed-256/fixed-1024 严格
配对结果。由于 hybrid relative D 为负且 collision 退化，本路线按门控停止 S128、
dev、额外 seeds 和 full-navtest。下一机制必须显式解决 held-out catastrophic safety
switch；不能继续通过扩大 ranking 权重或延长 schedule 冲击 argmax。
