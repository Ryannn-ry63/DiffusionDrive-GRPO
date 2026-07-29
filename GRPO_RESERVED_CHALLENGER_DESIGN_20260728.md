# Reserved design：Non-Destructive Challenger GRPO

日期：2026-07-28  
状态：仅记录；Stage38 期间未授权实现、训练或占用 stage 编号

## 1. 启动条件

只有 Stage38 出现以下任一冻结结论时，才允许把本设计转成新的正式 Stage 计划：

- `Fail`；
- `Positive ablation`，即 ESCR 已稳定修复共享 decoder，但 union safe-oracle
  仍小于 `+0.005`。

Stage38 `Mainline pass` 时不得启动 Challenger。

## 2. 核心假设

强 public 88.1 generator 的共享参数同时承担“保持成熟轨迹”和“探索新轨迹”，
使 GRPO 的探索梯度持续伤害 public elite modes。Challenger 将两者结构性解耦：

- public generator 完全冻结，始终输出原 20 条 candidates；
- 新增独立、轻量 challenger policy，只负责产生增量 candidates；
- GRPO 只更新 challenger；
- 推理候选集合为 public candidates 与 challenger candidates 的 union；
- selector 独立训练，并且始终可 fallback 到原 public trajectory。

该设计的价值是 public selected trajectory 可按构造 bitwise 保留，同时扩大
generator support。代价是它改变了架构和候选预算，不能把全部提升归因于 vanilla
GRPO。

## 3. 必做归因矩阵

正式计划必须保持相同数据、参数量、候选数、PDM budget 和 selector，至少包含：

1. `Public-20`：原始基线；
2. `Public-extra-sampling`：匹配 union candidate/PDM 计算预算；
3. `Challenger-BC`：相同新分支，但不使用 RL；
4. `Challenger-StdGRPO`：标准 group-relative objective；
5. `Challenger-SetGRPO`：public-anchored set-marginal objective；
6. generator 通过后，才做 frozen selector 与新 selector 的 2×2 factorial。

归因定义固定为：

```text
GRPO effect = Challenger-StdGRPO - Challenger-BC
set-credit effect = Challenger-SetGRPO - Challenger-StdGRPO
architecture/sampling effect = Challenger-BC - Public-extra-sampling
selector effect = same generator 下 new selector - frozen selector
```

## 4. 最低可行性 gate

Challenger 的首轮只能是一折、固定预算的 generator kill-test。进入 formal 前
必须满足：

- frozen public 20 candidates 与原模型 bitwise equal；
- public fallback score 与安全组件严格不退化；
- union safe-oracle delta `>= +0.005`；
- 两个 fresh namespaces 均为正；
- challenger positive marginal scene fraction 足够且无 catastrophic candidate；
- `Challenger-StdGRPO > Challenger-BC`，否则不能声称 GRPO 有效。

未达到 union safe-oracle `+0.005` 时立即停止，不得先训练 selector“救分”。

## 5. 创新与重合度约束

冻结 base + residual/challenger 本身不是新的研究贡献。未来可主张的组合必须是：

> Non-destructive candidate augmentation with public-anchored set-marginal
> diffusion GRPO and an independently audited safe selector.

正式立项前必须再次审查与 PlannerRFT、Policy Decorator、residual RL、
DiffusionDriveV2 和 set-level policy optimization 的重合。不得照搬
DiffusionDriveV2 的 multiplicative-noise 与 intra/inter-anchor truncated GRPO；
不得把 dual-branch 本身包装成独创。

## 6. 与 Stage38 的隔离规则

- 不复用 Stage38 实验目录、namespace、selection artifact 或 post-hoc threshold；
- 可以把 public checkpoint 作为共同 reference，但不得把失败的 ESCR checkpoint
  当作 challenger initializer；
- 独立 plan SHA、objective revision、training mode、artifact root 和 H100 wrapper；
- 只有新的正式计划冻结后才能实现。

