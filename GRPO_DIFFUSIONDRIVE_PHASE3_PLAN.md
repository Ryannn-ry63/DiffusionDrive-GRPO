# DiffusionDrive GRPO Phase 3 执行与断网恢复文档

日期：2026-07-18
状态：执行前已锁定；本文是 Phase 3 的 source of truth
分支：grpo-selection-v2
范围：只研究 GRPO 对 DiffusionDrive 扩散生成策略的提升；不涉及 WorldEngine、难例数据集或自回归模型。

## 1. 研究目标与结论口径

本阶段继续回答：不使用 imitation loss、GT trajectory regression 或 supervised diffusion loss，只使用 online raw PDMS reward，能否稳定提升 DiffusionDrive 的扩散去噪策略。

采用双门槛：

- 科学可行性：跨 seed、统计显著、安全不退化即可证明 GRPO 对 DiffusionDrive 这类条件扩散规划模型有效；
- 工程目标：三 seed full-navtest mean absolute PDMS delta 达到 +0.005。

已有证据不作废：

- generation-only D128、batch size 2 在三 seed full-navtest 的平均增量为 +0.0013528957；
- K2 hierarchical U128 在独立 dev-select 3072 上相对 base 为 +0.0009843908，95% CI [+0.0002693424,+0.0019379290]；
- 同一 K2 模型相对旧 D128 seed-0 champion 为 +0.0005653033，95% CI [+0.0000496109,+0.0012611681]；
- 这些结果已经证明 GRPO 可行，但尚未达到 +0.005 的工程目标。

正式评价只认 selector 实际选中轨迹的 raw PDMS。oracle 只用于诊断，不作为部署结果。

## 2. 当前问题与 Phase 3 初衷

当前 hierarchical objective 使用 fixed reference selector 最终选中轨迹的 reward，作为所有 20 anchors × 2 rollouts 的共同绝对基准。该 selected reference 通常是较强轨迹，因此约 84% 当前候选携带负 advantage。

这会带来两个问题：

1. 不同 anchor 的结构差异没有被消除，低先验 anchor 即使相对自身 reference 有改善，仍可能受到负向压制；
2. 候选集合被整体收缩，selected 虽小幅上升，但 oracle 在 dev-select 上没有提升。

Phase 3 用同场景、同 initial diffusion state、同 anchor 的 fixed-reference deterministic trajectory 作为绝对 baseline。baseline 在 sampled action 产生前确定，只依赖状态与 anchor，不依赖当前 stochastic action，因此符合 policy-gradient baseline 的约束。

## 3. 核心方法：Anchor-Conditioned Hierarchical GRPO

对场景 b、anchor m、rollout k：

    r_bmk = 当前 stochastic rollout 的 raw PDMS

    b_bm = fixed reference policy 从相同 initial diffusion state
           对 anchor m 无梯度 deterministic rollout 得到的 raw PDMS

    A_within_bmk = 同一 anchor 的 K 个 rollout 内做 group-relative normalization

    delta_bmk = r_bmk - b_bm

    A_anchor_bmk =
        clip(sign(delta) * max(abs(delta) - 0.01, 0) / 0.10, -2, 2)

    A_final =
        0.5 * clip(A_within, -2, 2)
        + 0.5 * A_anchor

行为约束：

- reference policy、reference rollout、PDM scoring 与 baseline 全程 no-grad/detach；
- reference anchor baseline shape 为 [B,20]，按 anchor-major 顺序扩展到 [B,20*K]；
- reference anchor invalid 时，该 anchor 的 absolute term 为 0，within-anchor term仍可工作；
- terminal raw PDMS 广播到该 rollout 的两个 denoising transitions；
- generation KL 0.1 对所有 valid transition 生效，即使 policy advantage 为 0；
- perception 与 selector classification branch 始终冻结；
- mode weighting 与 scene sampling 均保持 uniform；
- 不使用 shaping、priority sampling、selector policy gradient 或 oracle reranking。

新增接口：

- generation_advantage_mode 新增 anchor_hierarchical；
- grpo_rollouts_per_mode 在 anchor_hierarchical 下支持 2 和 4；
- 旧 group_zscore、reference_centered、within_anchor、hierarchical 数值兼容；
- 新日志包括 reference_anchor_reward、anchor_delta_mean/std、anchor valid fraction、正/负/死区 advantage fraction、within-anchor有效组比例和 reward gap。

## 4. 预注册实验

所有新模型从同一个 frozen DiffusionDrive base checkpoint 初始化，不从 K2 checkpoint 继续训练，保证因果比较。

固定超参：

- raw PDMS；
- generation-only；
- LR 1e-6；
- generation KL 0.1；
- diffusion schedule [8,0]；
- final action std 0.05；
- uniform sampler；
- uniform mode weighting；
- 128 optimizer updates；
- seed 0 用于机制选择。

实验：

| 名称 | advantage | K | batch策略 | 定位 |
|---|---|---:|---|---|
| H-Global-K2 | 现有 hierarchical | 2 | batch 2 | 已完成对照 |
| H-Anchor-K2 | anchor_hierarchical | 2 | batch 2 | 只隔离 baseline 变化 |
| H-Anchor-K4 | anchor_hierarchical | 4 | microbatch 1，gradient accumulation 2 | 检验 group size scaling |

K4 训练语义：

- 256 forward batches 对应 128 optimizer updates；
- old policy 每 64 forward batches 同步，对应每 32 optimizer updates；
- 有效 scene batch size 仍为 2；
- checkpoint 保存 optimizer step 64/128；
- 不扫描 K、混合权重、margin、scale、LR 或 KL。

## 5. 执行顺序与门控

### 5.1 工程实现

1. 扩展 reference rollout，返回并评分全部 20 个 anchor trajectories；
2. 实现 anchor_hierarchical objective 与 K=2/K=4 通用 reshape；
3. 扩展 config、YAML、runner 和训练日志；
4. 加入单元测试、静态检查和真实 batch 梯度审计。

### 5.2 seed-0 训练

每个 challenger 固定保存 U64/U128：

1. U8 real-batch smoke；
2. U64 只用于安全早停和趋势诊断，不作为事后择优点；
3. U128 先跑 fixed-256；
4. 通过后跑 fixed-1024；
5. 两个 challenger 都通过 fixed gates 后，才进入 dev-select 3072。

fixed-256 硬停止条件沿用旧路线：

- selected delta < -0.001；
- oracle delta < -0.002；
- collision、drivable、TTC 任一 delta < -0.002。

dev-select winner 先满足安全约束，再按 selected delta 选择；若 K2/K4 差距小于 0.00025，选择计算成本更低的 K2。

### 5.3 科学晋级

seed 0 必须同时满足：

- 相对 base selected paired CI 下界 > 0；
- 相对现有 H-Global-K2 增量 >= +0.0005，且 paired CI 下界 > 0；
- oracle 相对 base不下降；
- safety-pass bucket delta >= 0；
- collision、drivable、TTC 均不下降；
- baseline-bottom-1% delta >= 0；
- worst-1% delta tail 不差于现有 K2。

通过后锁死 K 与配置，补 seed 1、2。强化版科学结论要求：

- 三 seed 相对 base 同向；
- 至少两个单-seed CI 下界 > 0；
- seed-stratified CI 下界 > 0；
- 三 seed safety components 均不退化。

### 5.4 工程晋级

只有三 seed dev-select mean delta >= +0.003 才查看未使用的 dev-confirm 1024。

dev-confirm 要求：

- 三 seed 总体同向；
- seed-stratified CI 下界 > 0；
- safety components 和 tail 不退化。

通过后才运行唯一一次三 seed full-navtest。最终成功标准：

- 三 seed mean delta >= +0.005；
- 至少两个单-seed CI 下界 > 0；
- seed-stratified 与 token-clustered CI 下界 > 0；
- collision、drivable、TTC 均不退化。

若 Phase 3 显著优于现有 K2、但低于 +0.003，则记录为科学正结果，不运行 full-navtest。

## 6. Phase 4 预设：Step-Tree GRPO

只有 Phase 3 证明 anchor baseline 有效、但仍低于 +0.003 时，才进入逐 denoising-step credit assignment：

- timestep 8 采样两个 first-step action；
- 每个 first-step state 在 timestep 0 再采样两个 final action；
- 每个 anchor 共四个 leaf trajectories；
- final-step advantage 在同一 parent 的两个 leaves 内计算；
- first-step advantage 使用两个 branch 的平均 terminal reward计算；
- 不再把同一个 terminal advantage 无差别广播给两个 denoising steps；
- 与 flat K4 使用相同 leaf 数、scene batch 和 optimizer updates；
- 只运行 branching=2×2，不做树宽、树深或权重网格。

## 7. 必须测试

- anchor_hierarchical 的 K2/K4 shape、anchor-major ordering 和 invalid mask；
- current 与 reference 相同时 anchor advantage 为 0；
- reference baseline 不依赖 sampled action；
- reward/reference rollout/PDM scorer 全程无梯度；
- reference anchor 到 K rollouts 的 repeat 映射正确；
- K2 旧 hierarchical 与 K1 legacy objective 数值兼容；
- current=old 时 PPO ratio 为 1；
- advantage 为 0 时 generation KL 仍工作；
- generation/shared gradient 非零；
- perception/classification gradient严格为 0；
- K4 accumulation 后 optimizer updates、old-policy sync 和 checkpoint step正确；
- evaluator保存 checkpoint SHA、token清单/SHA、逐 token结果和 paired bootstrap；
- focused pytest、py_compile、bash syntax、git diff --check 全部通过。

## 8. 数据与断网恢复规则

- navtrain rewardable tokens：6119；
- dev-select：3072，SHA 1d5e2c1ae12f8e5b65edfa9976166b7a24cb77df87fe67b154bff9431a491299；
- dev-confirm：1024，SHA c98144d252415f74ccdf041a6f610b9a9bedee6adbba91027a7bb48a2df5554d；
- dev-confirm 当前未查看；
- full-navtest 不用于调参；
- 所有 checkpoint、artifact、命令、结果和停止决定追加到本文执行记录；
- 网络中断后，新会话先阅读本文和 GRPO_DIFFUSIONDRIVE_PERFORMANCE_ROADMAP.md，再检查 git status、当前进程与最后 artifact，不重复已完成训练。

## 9. 执行记录

### 2026-07-18：执行前审计

- 已确认当前 K2 rollout 为 anchor-major flatten；
- 当前 hierarchical 只允许 K=2；
- 当前 absolute baseline 为 reference selected reward，shape [B]；
- fixed reference decoder 已冻结，current/old/reference 在同一 sampled action trace 上回放；
- 当前 runner 以 limit_train_batches 表示 forward batches，K4 accumulation 必须显式换算 optimizer updates；
- 8 张 RTX 4090 每张约 49GB 可用；
- 尚未修改 Phase 3 代码，下一恢复点是实现 reference-anchor reward 数据流。

### 2026-07-18：工程实现与 U8 smoke

已完成：

- generation_advantage_mode 新增 anchor_hierarchical；
- fixed reference 从相同 initial diffusion state 无梯度生成并评分全部 20 个 anchors；
- reference anchor rewards 以 anchor-major 顺序扩展到 K rollouts；
- anchor_hierarchical 支持 K=2/K=4，旧 hierarchical 仍严格限定 K=2；
- K4 runner 固定 batch1、gradient accumulation2、old-policy forward sync64；
- runner 使用 max_steps 锁定真实 optimizer updates；
- 72 项 focused tests、py_compile、bash syntax、git diff --check 通过。

H-Anchor-K2 U8：

- experiment：grpo_d_seed0_skl0.0_gkl0.1_eps0.001_dense0.1_mweightuniform_bs2_advanchor_hierarchical_k2_uniform_u8/2026.07.18.09.53.23；
- 8 optimizer steps 正常完成；
- positive/negative advantage：38.13%/55.94%；
- reference anchor coverage：100%；
- within-anchor valid fraction：90.63%；
- diff-decoder/shared/regression grad norm：13.39/3.51/12.91；
- classification grad norm：0。

H-Anchor-K4 U8：

- experiment：grpo_d_seed0_skl0.0_gkl0.1_eps0.001_dense0.1_mweightuniform_bs1_advanchor_hierarchical_k4_uniform_u8/2026.07.18.09.54.31；
- max_steps=8、accumulate=2、16 forward batches、old sync64 均正确；
- positive/negative advantage：36.88%/56.56%；
- reference anchor coverage：100%；
- within-anchor valid fraction：91.87%；
- diff-decoder/shared/regression grad norm：10.69/2.89/10.27；
- classification grad norm：0。

结论：新 baseline 将旧 hierarchical 约 84% 的负 advantage 降到约 56%，且冻结边界、K4 accumulation 与 PDM reference scoring 全部通过。下一恢复点：并行训练 seed0 H-Anchor-K2/K4 U128，保留 step64/128 checkpoints。

### 2026-07-18：seed-0 U128 与 fixed-256 门控

正式训练均从 frozen base 初始化并正常到达 128 optimizer steps：

- H-Anchor-K2：batch2，step64/128 checkpoints 完整；
- H-Anchor-K4：batch1、accumulate2，step64/128 checkpoints 完整；
- 两者 reference anchor coverage 均为 100%，classification grad 始终为 0；
- K2 训练期正/负 advantage 平均约 33.90%/54.02%；
- K4 训练期正/负 advantage 平均约 33.46%/57.07%。

fixed-256 严格配对结果：

| 配置 | selected delta | 95% CI | oracle delta | 决策 |
|---|---:|---:|---:|---|
| H-Anchor-K2 U64 | +0.001116 | [-0.000174,+0.003509] | +0.000094 | 仅诊断，未显著 |
| H-Anchor-K2 U128 | +0.001107 | [-0.000153,+0.003422] | -0.003161 | oracle 硬停止 |
| H-Anchor-K4 U64 | +0.001137 | [+0.000013,+0.003376] | -0.000001 | 仅诊断，安全 |
| H-Anchor-K4 U128 | +0.004607 | [-0.000171,+0.012995] | -0.003123 | oracle 硬停止 |

预注册 fixed-256 规则规定 oracle delta < -0.002 必须停止。两个 U128 均触发该条件，因此：

- 不运行 fixed-1024；
- 不运行 dev-select 或 dev-confirm；
- 不补 seed 1/2；
- 不以 U64 事后替代 U128 作为 winner；
- 不进入 full-navtest。

诊断结论：

- anchor-conditioned baseline 成功把绝大多数候选负 advantage 降到约 54%--57%，说明 baseline 修正本身有效；
- 但 64→128 updates 后两种 K 都出现约 -0.0031 的 oracle 回退，说明主要剩余问题是持续 group-relative 更新损害候选集合上界；
- K4 U128 的 +0.004607 selected 点估计由少量 selected 安全事件驱动，CI 跨 0，不能抵消 oracle 硬停止；
- Phase 3 完成了机制实现和预注册实验，但未通过科学晋级门槛；
- Phase 4 的启动前提是 Phase 3 先证明 anchor baseline 在正式门控中有效，本次未满足，因此当前不启动 Step-Tree GRPO。

当前恢复点：Phase 3 已按门控停止。下一会话应先基于“U64安全、U128 oracle下降”的更新时序问题规划新阶段，不得直接运行 fixed-1024/dev/full-navtest。
