# Stage39–40：Non-Destructive Challenger GRPO

日期：2026-07-29  
状态：冻结计划；实现、训练和选参的唯一 source of truth  
官方基线：DiffusionDrive public 88.1，NavTest PDMS `0.8810410646527626`

## 1. 研究问题与停止规则

Stage38 ESCR 在 shared decoder 上达到正但不足的 selected gain，且无法修复
public-top-5 退化。更关键的是，正 challenger credit 仅占约 `0.169%`，而已有两组
fresh-noise public banks 的 40-candidate control 显示 strict-safe oracle 增量约
`+0.011`、union top-5 增量约 `+0.024`。因此 Stage39 不再继续修补同一个 decoder，
而是验证一个可证伪的新假设：

> 冻结且永久保留 official public 20 candidates；让独立 challenger decoder
> 只承担 candidate support expansion，并用 public-frontier set-marginal GRPO
> 学习。这样探索梯度不能破坏 public fallback。

Stage39 必须用 `Public20 / Public40-extra / Challenger-BC /
Challenger-StdGRPO / Challenger-SetGRPO` 完成归因。若 GRPO 分支不能稳定优于
相同架构、参数量和计算预算的 BC control，则 DiffusionDrive 主线停止，不能把
sampling gain 写成 GRPO gain，也不能进入 selector 或 NavTest。

## 2. 冻结边界

- public/reference checkpoint：
  `/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS`
- SHA256：
  `008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b`
- public decoder、classification head、backbone、perception 与已有 selectors
  全部冻结。
- challenger 是 public decoder 的完整物理副本，必须从 official public
  checkpoint 初始化；不得从 Stage36/37/38 checkpoint 初始化。
- public 与 challenger 不允许参数或 storage alias。
- 仅 challenger 的两个 regression decoder layers 可训练，共 64 个 tensor；
  challenger classification head 严格冻结。
- public 始终输出原 20 candidates；challenger 输出额外 20 candidates；
  union origin mask 固定为 `[public × 20, challenger × 20]`。
- public 使用原 evaluation namespace，challenger 使用独立确定性 namespace；
  不使用 multiplicative noise，不复刻 DiffusionDriveV2 的 intra/inter-anchor
  truncated GRPO。

## 3. Stage39 归因矩阵

1. `Public20`：official public，一组 20 candidates。
2. `Public40-extra`：同一个 frozen public decoder 独立采样两组，每组 20；
   仅估计额外 sampling/PDM budget 的贡献。
3. `Challenger-BC`：完整 challenger decoder；用 frozen public diffusion chains
   做行为克隆，仍执行相同 candidate rollout/PDM budget。
4. `Challenger-StdGRPO`：每个 anchor 的 8 个 challenger rollouts 做 PDM
   group z-score，clip 到 `[-2, 2]`。
5. `Challenger-SetGRPO`：使用 public top-5 frontier 的绝对 set marginal credit。

归因固定为：

```text
sampling effect      = Public40-extra - Public20
architecture/BC      = Challenger-BC - Public40-extra
standard GRPO effect = Challenger-StdGRPO - Challenger-BC
set-credit effect    = Challenger-SetGRPO - Challenger-StdGRPO
selector effect      = same generator 下 new selector - frozen/fallback selector
```

所有训练分支使用相同数据、full chain `[32,24,16,8,0]`、`G=8`、rollout/PDM
budget、optimizer steps、LR `1e-6`、layer-0 multiplier `0.1`、clip `1.0`、
weight decay `0`。BC 分支只忽略 detached reward，不减少 reward 计算。

## 4. Stage39 objectives

### 4.1 Standard GRPO

对每个 scene、anchor `m`，在 8 个 challenger rollouts 上：

```text
A_std(g,m) = clip((r(g,m) - mean_g r(g,m)) /
                  max(std_g r(g,m), 1e-4), -2, 2)
```

无效轨迹为 `-2`；advantage、reward、validity 全 detach。

### 4.2 Public-anchored set-marginal GRPO

对每个 scene/group：

```text
E_g = top5(valid public candidates)
U5(S) = mean(top5 rewards in S)
delta(g,m) = max(0, U5(E_g union challenger(g,m)) - U5(E_g))
```

若 challenger 无效，或 collision/drivable/TTC 任一低于 public elite set
对应安全下界超过 `1e-6`，则 `A=-2`。否则：

```text
delta > 0.001 : A = clip((delta - 0.001) / 0.02, 0, 2)
else          : A = 0
```

Loss：

```text
L_bc  = public-chain diffusion behavior cloning
L_std = -logp_challenger * A_std * step_discount + 0.1 * KL(challenger||public)
L_set = -logp_challenger * A_set * step_discount + 0.1 * KL(challenger||public)
```

`step_discount=0.6`。loss 先按 scene 归一化，再做 batch mean。

## 5. Stage39 执行阶段

### Phase 0：离线 target audit

复用冻结的两组 fresh-noise public banks，只验证 target 和预算，不训练：

- safe positive set-marginal fraction 必须在 `[0.05, 0.40]`；
- extra-sampling strict-safe oracle 必须 `>= +0.005`；
- public20 records、trajectory 与 score 必须 bitwise/数值完全一致；
- target 不得含 NaN/Inf，安全 veto 必须覆盖人工构造的 unsafe candidate。

### Phase 1：真实 optimizer-step audit

`BC / StdGRPO / SetGRPO` 三个分支各跑一个真实 8-GPU optimizer step。必须验证：

- public decoder 和 public output bitwise 不变；
- 仅 challenger regression decoder 的 64 个 tensor 改变；
- challenger 两层均有有限非零梯度；
- classification/backbone/selectors 均无梯度且 checkpoint bitwise 不变；
- public/challenger noise 独立且相同 namespace 重跑确定性一致；
- DDP sampler、global bucket composition `[2,30,8,24]` 与 accumulation `8`
  完全一致。

### Phase 2：pilot fold2

- train：`train_except_fold2`；
- fixed checkpoints：48/96/192 optimizer steps；
- evaluation namespaces：`20261711/20261712`；
- 三个训练分支同预算并行；
- 只允许用 fold2 pilot 选择一个全局 step，禁止逐分支或逐 namespace 选 step。

### Phase 3：formal OOF

- holdout folds：0/1/3；
- 使用 pilot 冻结的唯一 step，从 official public 独立训练三个分支；
- evaluation namespaces：`20261721/20261722`；
- 不得用 formal 结果重选 step、margin 或 objective。

进入 Stage40 的硬 gate：

- public20 bitwise exact；
- `StdGRPO - BC` 两个 namespace 均为正且 whole-log bootstrap CI 下界 `>0`；
- Set union strict-safe oracle vs Public20 `>= +0.012`；
- Set vs Public40-extra `>= +0.001`；
- Set vs BC `>= +0.001`；
- Set vs StdGRPO `>= +0.0005`；
- 每个 formal fold/namespace 均为正；
- catastrophic candidate rate 不高于 Public40-extra `+0.001`。

若 GRPO 优于 BC 但不优于 Public40-extra，只能作为 positive GRPO ablation，
不能进入 selector/NavTest performance claim。若 GRPO 不优于 BC，冻结 sampling-only
结论并停止 DiffusionDrive 主线。

## 6. Stage40：Candidate-Count-Agnostic Hierarchical Selector

只有 Stage39 formal gate 通过才允许训练 Stage40。

1. frozen Stage25 只在 public20 内得到 public fallback；
2. 四成员 ensemble 对每个 challenger 相对 fallback 独立预测：
   PDMS delta、六个 component delta、harm probability、catastrophe probability；
3. selector 输入不得含 generator/branch ID；
4. 只有 improvement lower confidence bound `> margin` 且三个 safety component
   lower bounds均不劣于 `-0.0005` 时才允许切换；
5. 未通过、OOD 或无有效 challenger 时必须 bitwise 返回 public fallback。

Nested OOF 固定为：

```text
test fold0: fit folds2/3, calibrate fold1
test fold1: fit folds2/3, calibrate fold0
test fold3: fit folds0/1, calibrate fold2
```

校准网格固定：

```text
improvement margin: {0, 0.001, 0.002, 0.005}
risk threshold:     {0.05, 0.10, 0.20}
confidence z:       {1.0, 1.64, 1.96}
```

只在 calibration fold 上、满足 catastrophes=0、collision/drivable/TTC
均 `>= -0.0005`、switch rate 在 `[0.01,0.20]` 的组合中选择。

Stage40 OOF gate：

- SetGRPO + new selector vs Public20 selected gain `>= +0.005`；
- 论文目标 `>= +0.010`，bootstrap CI 下界 `>0`；
- 每个 fold/namespace 为正；
- 同一 selector 下 vs BC `>= +0.001`、vs extra `>= +0.001`、
  vs StdGRPO `>= +0.0005`；
- 至少兑现 union strict-safe oracle headroom 的 40%；
- mature `>= -0.0001`，collision/drivable/TTC 各 `>= -0.0005`，
  catastrophes=0，fallback bitwise exact。

## 7. NavTest

只有 Stage40 OOF gate 通过后，固定一次运行：

1. Public20；
2. Public40-extra + same selector；
3. Challenger-BC + same selector；
4. Challenger-StdGRPO + same selector；
5. Challenger-SetGRPO + same selector。

NavTest 不得选择任何参数。相对 official `0.8810410646527626`：

- 最低通过：`>= 0.8860410646527626`（`+0.005`）；
- 论文目标：`>= 0.8910410646527626`（`+0.010`）。

## 8. 产物与 fail-closed 约定

- Stage39：`artifacts/grpo_stage39/`
- Stage40：`artifacts/grpo_stage40/`
- 所有 wrapper 必须导出 NAVSIM 环境、使用仓库绝对路径、先执行
  `--cfg job --resolve` preflight、拒绝覆盖、记录 plan/checkpoint/manifest SHA256。
- Stage40 wrapper 在 Stage39 selection/gate artifact 缺失时必须拒绝运行。
- 本机任何 GPU 任务开始前必须按 `AGENTS.md` 仅关闭 `gpu-occupy` 并确认显存释放；
  任务结束后不得自动重启。

