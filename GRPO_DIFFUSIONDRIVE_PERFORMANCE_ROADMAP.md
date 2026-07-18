# DiffusionDrive GRPO 性能提升执行路线

日期：2026-07-18
状态：执行中的 source of truth
范围：只研究 GRPO 在 DiffusionDrive 中的性能提升；本阶段不修改 WorldEngine，也不实现自回归模型迁移。

## 1. 研究目标

核心问题是：**在 GRPO 阶段不使用 imitation loss、GT trajectory regression 或 supervised diffusion loss，仅使用在线 PDM reward，能否稳定提高 DiffusionDrive 的部署 selected raw PDMS。**

当前已经成立的结论是：generation-only GRPO 在三个独立训练 seed、完整 navtest 和严格配对统计下均为正，证明 GRPO 在 DiffusionDrive 上具备可行性。下一阶段不再重复回答“有没有效果”，而是把提升从当前约 `+0.00135` 推进到工程目标 `>=+0.005`。

正式成功标准：

- 三 seed full-navtest mean absolute PDMS delta `>=+0.005`；
- 至少两个单-seed paired bootstrap CI 下界大于 0；
- seed-stratified 与 token-clustered CI 下界大于 0；
- collision、drivable、TTC 均值不得退化；
- 正式结论只使用 selector 实际选中轨迹的 raw PDMS，不使用 oracle reranking；
- perception 始终冻结，GRPO 阶段不加入 imitation/GT loss。

## 2. 当前真实基线与“后续效果下降”的解释

必须区分 proxy best 和 formal champion：

| 版本 | fixed-1024 | full-navtest | 定位 |
|---|---:|---:|---|
| GitHub D128，batch 1，seed 0 | `+0.003299` | `+0.000706` | 单 seed proxy 峰值 |
| 本地 D128，batch 2，seed 0 | `+0.001904` | `+0.001446` | 单 seed 正式结果更好 |
| 本地 D128，batch 2，三 seed | 均值约 `+0.00138` | `+0.0013528957` | 当前 formal champion |
| 后续 hybrid ranking | `+0.001436` vs base | 未晋级 | selector challenger 失败 |

因此，后续主版本没有在真实 full-navtest 上弱于 GitHub 的 `+0.003299` checkpoint。下降主要来自：

1. `+0.003299` 是反复使用的 fixed-1024、单 seed 最好点，存在选择偏差；
2. fixed-1024 的逐 token delta 是重尾分布，少数接近 `+1/-1` 的安全事件足以移动均值；
3. batch 1 与 batch 2 在 fixed-1024 上的逐 token delta 相关系数约 `0.54`，有 `481/1024` 个 token 方向相反；
4. batch size 2 降低了训练 seed 方差，并在 full-navtest 上优于旧 batch-size-1 checkpoint；
5. 真正负向的是 reference gate、alternating、full-information selector 和 ranking 分支，而不是 generation-only GRPO champion。

当前 champion 固定为三条 batch-size-2 generation-only D128：

- raw PDMS reward；
- uniform mode weighting；
- batch size 2；
- 128 optimizer updates；
- LR `1e-6`；
- generation KL `0.1`；
- diffusion schedule `[8,0]`；
- final action std `0.05`。

任何新实验都是 challenger；未通过完整门控前不得覆盖 champion。

## 3. 关键瓶颈

### 3.1 Group-relative advantage 的绝对退化问题

现有 objective 对同一场景的候选 reward 做组内标准化。如果所有新候选都比 fixed reference 差，组内相对最好者仍获得正 advantage。这与 full-navtest 诊断一致：收益集中在少量失败场景，而大量 safety-pass 高分场景有微小退化。

### 3.2 20 个 mode 不是同分布 rollout

当前把 20 个不同 anchor mode 当作 GRPO group。它们既包含 anchor 先验差异，也包含 stochastic denoising 质量差异，credit assignment 不够干净。后续需要在同一 anchor 内生成多个 rollout，区分“anchor 好坏”和“denoising action 好坏”。

### 3.3 Generator–selector mismatch

D128 之后继续 generation 训练时，oracle/candidate mean 上升而 selected 回退，说明 generator 仍能产生更好候选，但 selector 无法兑现。已经验证：

- selector consistency KL 无法解决；
- generation KL 扫描无效；
- raw/tie-break/dense shaping 走势相同；
- softmax mode weighting 弱于 uniform，top1 失败；
- selector-only full-information GRPO 只切换 `1/1024`；
- pairwise ranking 增加有益切换，但一个 `-0.695360` collision switch 吞掉全部收益。

因此近期不再扫描 selector LR、epsilon、ranking weight、KL 或 shaping。先改善 generation GRPO 的绝对 credit assignment 和训练数据分布。

## 4. Phase 0：版本、数据和兼容性审计

### 4.1 固化实验身份

为 GitHub proxy best 与本地 formal champion 保存统一 ledger：

- checkpoint path 与 SHA256；
- parent/reference checkpoint；
- Git commit 与工作区 diff 标识；
- 完整 Hydra overrides；
- seed、batch size、optimizer updates；
- train/fixed/dev/navtest token SHA；
- raw CSV、JSON artifact 与 paired bootstrap 结果。

命名固定为：

- `proxy_best_bs1_d128`；
- `formal_champion_bs2_d128_seed{0,1,2}`。

### 4.2 Legacy 数值兼容

在当前代码与 GitHub generation-GRPO commit 间，用相同 cached batch 和 RNG 比较：

- generated trajectories；
- current/old/reference log-prob；
- raw PDMS 与 valid mask；
- group advantage、policy loss、KL；
- generation/shared/classification gradient。

兼容配置为：raw PDMS、uniform mode、uniform scene weight、rank weight 0、exploration floor 0。前向最大绝对误差要求 `<=1e-6`，loss/gradient 相对误差要求 `<=1e-5`。如果不通过，先修复兼容性，禁止启动新训练。

### 4.3 验证集纪律

- fixed-256：只淘汰明显错误；
- fixed-1024：已多次使用，只做方向与故障诊断；
- 现有独立 dev-4096 按固定 SHA 分为 dev-select 3,072 与 dev-confirm 1,024；
- dev-confirm 在唯一配置锁定前不得查看；
- full-navtest 不用于调参，只评测最终唯一配置。

## 5. Phase 1：Reference-Centered Generation GRPO

### 5.1 Objective

对每个场景先用 fixed base reference policy 得到部署 selected raw PDMS `r_ref`。对当前候选 `i`：

```text
delta_i = r_i - r_ref

A_ref_i = clip(
    sign(delta_i) * max(abs(delta_i) - 0.01, 0) / 0.10,
    -2,
    2
)
```

行为要求：

- 比 reference 好超过 `0.01` 才强化；
- 比 reference 差超过 `0.01` 时产生负 advantage；
- margin 内 policy gradient 为 0；
- 不再使用失败的 scene-level binary/linear gate；
- generation KL `0.1` 对全部 valid candidate/step 生效，即使 policy advantage 为 0；
- reference scoring 和 reward 全程 detach/no-grad；
- selector/perception 冻结，classification gradient 严格为 0。

新增配置：

- `generation_advantage_mode=group_zscore|reference_centered|within_anchor|hierarchical`；
- `reference_advantage_margin=0.01`；
- `reference_advantage_scale=0.10`；
- `reference_advantage_clip=2.0`；
- `grpo_rollouts_per_mode=1`。

默认 `group_zscore`、`grpo_rollouts_per_mode=1` 必须完全复现现有 champion 行为。

### 5.2 Failure-aware sampler

只从 navtrain 的 6,119 个 rewardable token 构建 priority manifest。满足任一条件即为 priority：

- base selected raw PDMS `<0.5`；
- collision、drivable 或 TTC fail；
- base candidate oracle headroom `>0.05`。

训练 batch 固定为 `50% uniform + 50% priority`。priority manifest 必须与 fixed-1024、dev-4096、navtest token 零交集。common replay 不得低于 50%，避免正常场景遗忘。

### 5.3 Seed-0 因果消融

全部使用 batch size 2、LR `1e-6`、generation KL `0.1`、128 updates：

| 实验 | advantage | sampler |
|---|---|---|
| D-control | group z-score | uniform |
| D-priority | group z-score | 50/50 |
| RA-uniform | reference-centered | uniform |
| RA-priority | reference-centered | 50/50，主实验 |

保存 update 64/128。只有当 update 128 在 dev-select 相比 update 64 再提升 `>=0.0005`，且 safety components 不退化时，允许唯一一次延长到 192。不得继续扫描 LR、KL、margin 或 scale。

## 6. Phase 2：Hierarchical Multi-Sample GRPO

如果 Phase 1 未达到最终目标但 reference-centered 方向为正，则令每个 anchor 采样 `K=2` 条 stochastic denoising trace，形成 `[B,20,2,T]`：

```text
A_final = 0.5 * clip(A_within_anchor, -2, 2)
        + 0.5 * A_ref
```

- `A_within_anchor` 只比较同一 anchor 的两个 rollout；
- `A_ref` 保证绝对性能不低于 fixed reference；
- terminal raw PDMS 广播到该 rollout 的全部 denoising transitions；
- current/old/reference 必须在相同 trace 上回放；
- mode aggregation 保持 uniform；
- 只比较 within-anchor-only 与 hierarchical 两组 seed-0，不增加其它网格。

如果 `K=2` 只提高 oracle 而 selected 增量低于 oracle 增量的 50%，才重新进入 selector 表示研究；否则继续 generation 路线。

## 7. 统一门控与停止规则

### 7.1 fixed-256 淘汰

相对输入 champion 满足任一条件即停止：

- selected delta `<-0.001`；
- oracle delta `<-0.002`；
- collision、drivable、TTC 任一 delta `<-0.002`。

### 7.2 dev 晋级

补 seed 1/2 前要求：

- dev-select 相对正式 base 总 delta `>=+0.003`；
- 相对 champion 增量 `>=+0.0005`；
- paired CI 下界大于 0；
- safety-pass bucket delta `>=0`；
- collision、drivable、TTC 不低于 champion；
- bottom-1% delta/CVaR 不比 champion 恶化；
- dev-confirm 同向且满足相同安全约束。

补齐三 seed 后要求：

- 三 seed delta 同向；
- 至少两个单-seed CI 下界大于 0；
- seed-stratified CI 下界大于 0；
- 三 seed safety components 均不退化。

只有唯一配置通过后运行 full-navtest。高于当前三 seed mean `+0.0013528957` 且安全不退化即可成为新 champion；达到 `+0.005` 才完成本阶段工程目标。

### 7.3 路线停止

- 连续两个机制阶段相对 champion 增量 `<0.0005`，停止该路线；
- 不因 fixed-256 正向直接运行 full-navtest；
- 不重复已完成的 selector epsilon、ranking LR、KL、dense shaping 网格；
- 不同时运行多个 32-worker full-navtest；
- 不使用 oracle 或 online PDM reranking 作为部署结果。

## 8. 工程测试与日志

必须测试：

- 默认 `group_zscore/K=1` 与旧 objective 数值兼容；
- reference-centered advantage 的正、负、margin、clip；
- 所有候选都差于 reference 时不存在正向强化；
- policy advantage 为 0 时 generation KL 仍工作；
- invalid/all-invalid/single-valid candidate 正确处理；
- reward、reference rollout、PDM scorer 全程无梯度；
- current=old 时 PPO ratio 为 1；
- G 阶段 generation/shared gradient 非零、classification gradient 严格为 0；
- `K=2` rollout shape、trace replay 和 within-anchor normalization 正确；
- sampler 的 priority/uniform 比例与 token 隔离正确；
- evaluator保存 checkpoint SHA、配置、token清单/SHA、逐 token结果和 paired bootstrap；
- focused pytest、`py_compile`、bash syntax、`git diff --check` 全部通过。

新增训练日志：

- reference selected reward；
- candidate-reference delta mean/std；
- improved/worse/within-margin candidate fraction；
- positive/negative advantage fraction；
- policy-active scene fraction；
- generation ratio、clip fraction、KL；
- selected/oracle/regret；
- bottom-tail/CVaR；
- generation/shared/classification gradient norm。

## 9. 执行顺序

1. 完成 Phase 0 ledger、dev split 和 legacy 数值审计；
2. 实现 reference-centered advantage、配置、日志和单测；
3. 实现 navtrain priority manifest 与 50/50 sampler；
4. 跑 D-control、D-priority、RA-uniform、RA-priority smoke；
5. 按 fixed-256 → fixed-1024 → dev-select → dev-confirm 门控；
6. 胜出配置补 seed 1/2，再运行唯一 full-navtest；
7. 若未达到 `+0.005` 但 RA 有稳定增量，进入 K=2 hierarchical GRPO；
8. 每完成一个阶段更新本文的 checkpoint、artifact、统计结果和停止决策。

## 10. 最终论文消融链

```text
DiffusionDrive baseline
→ generation-only GRPO D（已证明显著正向）
→ batch-size 2 方差控制（当前 champion）
→ reference-centered GRPO
→ failure-aware reference-centered GRPO
→ hierarchical multi-sample GRPO
```

selector consistency、generation KL、tie-break/dense shaping、selector weighting、naive joint、decoupled alternating、full-information selector 和 pairwise ranking 作为已完成的负向或边界消融保留，不再重复消耗主实验预算。

## 11. 2026-07-18 执行记录

### 11.1 已完成工程

- 实现 `reference_centered` generation advantage，默认 `group_zscore` 保持旧行为；
- 固定 reference deployed reward 全程无梯度，policy advantage 为零时 KL 仍有效；
- 新增 reference delta、正负 advantage、margin 和 active scene 日志；
- 新增 `50% uniform + 50% priority` batch sampler；uniform 指全量 navtrain，而不是 non-priority 子集；
- 新增 train/val 可选 evaluator、priority manifest builder、dev-select/dev-confirm split；
- runner 支持 advantage、priority manifest 和可选 `UPDATES`，默认仍为 128；
- focused pytest 51 项通过，`py_compile`、bash syntax、`git diff --check` 通过。

### 11.2 冻结数据 artifact

- navtrain base：6,119 tokens，SHA `c7edaae00ad527b2cfe3c4391794dc70798a7e5bd69faa24c4b603822c75fde0`；
- base selected `0.7932127823`，oracle `0.9402665652`，regret `0.1470537830`；
- priority：4,541 tokens，common：1,578 tokens，SHA `36d8c7e735a86e0c3e0d7d4d95eea29549fddb009d5a4bcfd25bb4ddf290f0a9`；
- priority reasons：high-headroom 4,515、low reward 266、drivable fail 218、TTC fail 130、collision fail 30；
- priority 与 fixed-1024、dev-4096 overlap 均为 0；
- dev-select 3,072 与 dev-confirm 1,024 overlap 为 0。

### 11.3 真实训练 smoke

- RA-uniform U8：classification grad `0`，shared `9.07`，regression `31.94`；正/负 advantage `9.06%/83.44%`；
- RA-priority U8：classification grad `0`，shared `9.43`，regression `39.63`；正/负 advantage `11.56%/83.75%`；
- fixed-256 paired delta：RA-uniform `+0.00005469`，RA-priority `+0.00000048`；两者 CI 均跨 0，oracle 基本不变；
- 结论：实现与冻结边界通过，U8 未见破坏，但不能作为性能晋级证据；priority 增强了正 advantage 占比，也提高了梯度强度。

### 11.4 Phase 1 完整结果与停止决策

- D-priority U8 在 fixed-256 为 -0.003685，且 drivable 退化，说明简单 failure-aware oversampling 会改变训练分布并放大破坏性更新；
- RA-uniform U64/U128 在 fixed-256 分别约为 +0.000229/+0.000999，但 U128 oracle 下降 -0.003220；
- RA-priority U64/U128 在 fixed-256 分别约为 +0.00112/+0.00104，但 oracle 分别下降约 -0.00340/-0.00359；
- RA-uniform U64 扩大到 fixed-1024 后 selected 为 -0.001381，CI 跨 0；
- reference-centered 与 priority sampler 均未形成可泛化增益，Phase 1 停止；不进入 dev-select，不继续扫描 margin、scale、LR 或 KL。

该结果说明：把每条候选直接与 reference deployed reward 比较，会使大量候选长期携带负 advantage；它能抑制明显坏轨迹，但也削弱候选多样性和 oracle，不能替代同一 anchor 内的低方差相对比较。

### 11.5 Phase 2：K=2 hierarchical GRPO

已完成：

- 每个 anchor 两条独立 stochastic trace，形成 40 条候选；
- current/old/reference 在相同 trace 上回放；
- within-anchor 与 hierarchical 两种 advantage；
- hierarchical 使用 0.5 * within-anchor + 0.5 * reference-centered；
- step 64/128 checkpoint、真实 batch 梯度审计与 K=1 兼容测试；
- focused pytest 55 项通过；classification gradient 严格为 0。

seed-0 结果：

| 配置 | split | selected delta | oracle delta | 结论 |
|---|---|---:|---:|---|
| K2 within U64 | fixed-256 | +0.001220 | -0.002944 | oracle 退化，停止 |
| K2 hierarchical U64 | fixed-256 | +0.001134 | -0.000023 | 晋级 fixed-1024 |
| K2 hierarchical U64 | fixed-1024 | +0.000516 | +0.000068 | 保留 |
| K2 hierarchical U128 | fixed-256 | +0.001047 | -0.001932 | fixed-256 方差较大 |
| K2 hierarchical U128 | fixed-1024 | +0.003070 | +0.000723 | 晋级 dev-select |

独立 dev 数据使用专用 4,096-token val feature cache，而不是训练 cache：

- dev-select：3,072 tokens，SHA 1d5e2c1ae12f8e5b65edfa9976166b7a24cb77df87fe67b154bff9431a491299；
- dev-confirm：1,024 tokens，SHA c98144d252415f74ccdf041a6f610b9a9bedee6adbba91027a7bb48a2df5554d；
- 两者与 navtrain、fixed-1024、navtest 均零交集；
- dev-confirm 尚未查看。

K2 hierarchical U128 在 dev-select 的正式配对结果：

- 相对 base selected：+0.0009843908，95% CI [+0.0002693424, +0.0019379290]；
- 相对 base oracle：-0.0004424095，95% CI [-0.0015521693, +0.0005757713]；
- 相对旧 D128 champion selected：+0.0005653033，95% CI [+0.0000496109, +0.0012611681]；
- 相对旧 champion oracle：-0.0000477904，CI 跨 0；
- wins/ties/losses（相对 base）：1364/805/903；
- safety-pass bucket：+0.0002960929；
- base <0.5 bucket（340 tokens）：+0.0056902662；
- collision/drivable/TTC：分别 +0.0003255208/+0.0003255208/+0.0006510417；
- 配对 artifact：artifacts/grpo_stage6/k2_hierarchical_u128_dev_select3072_comparison.json。

### 11.6 当前结论与下一阶段边界

K2 hierarchical 是目前第一个在独立 dev-select 上显著优于 base、同时也显著优于旧 D128 champion 的新机制，证明同一 anchor 内多采样的 GRPO 相对信号有效。但它没有通过预先锁定的总增量 >=+0.003 门槛：

- **机制结论：通过**。相对 base/champion 的 CI 下界均大于 0，安全指标不退化；
- **性能晋级：未通过**。总体仅 +0.000984，不运行 dev-confirm，不补 seed 1/2，不运行 full-navtest；
- fixed-1024 的 +0.003070 未在独立 dev-select 复现，不能称为最终 +0.003 模型；
- 增益集中于少量低分难例，正常场景增益很小；oracle 未提升，继续单纯增加 update 不具备足够依据。

下一阶段必须围绕“扩大 K2 的有效覆盖面，同时保护 common 场景”设计单一新机制。不得直接延长 U192/U256，不得重新开启已失败的 priority oversampling、reference margin、selector weighting 或 shaping 网格。新机制在写入本路线图、明确因果消融和停止门槛之前不得启动训练。
