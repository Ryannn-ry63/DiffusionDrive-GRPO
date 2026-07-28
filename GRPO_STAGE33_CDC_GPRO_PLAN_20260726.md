# Stage33：Counterfactual Deployment-Credit GRPO（CDC-GRPO）

日期：2026-07-26  
基线：官方 `diffusiondrive_navsim_88p1_PDMS`（commit/checkpoint 由运行脚本固定）  
前一阶段：Stage32 selector-aware frontier

## 1. 目标与边界

Stage32 的 OOF 结果已经说明 selector 还有可用的选择增益，但 candidate ceiling 基本不动，说明“生成器没有把足够高的上限交给 selector”是当前瓶颈。Stage33 不把训练目标改成“离线 PDM 模仿”，也不把 selector 的分类分数冒充 diffusion log-prob；主线仍然是 **GRPO 作用于 diffusion generator 的完整去噪链**。

Stage33 的核心问题定义为：

> 如果某个生成候选在真正部署的 selector 下被选中，它相对公共/参考 generator 的 PDM 变化应当获得 credit；如果候选没有被部署但具有明显 headroom，则只获得受 selector 概率约束的弱 credit，避免 generator 追逐 selector 当前不可实现的 oracle。

这产生一个与候选排序解耦的 generator 目标：**Counterfactual Deployment-Credit GRPO（CDC-GRPO）**。

## 2. 与已有论文的差异（避免“照搬”）

| 工作 | 主要做法 | Stage33 明确不复现的部分 | Stage33 的独立贡献 |
|---|---|---|---|
| DiffusionDriveV2 | intra-anchor GRPO、inter-anchor truncated GRPO、噪声尺度控制 | 不把 anchor/truncated GRPO 当主方法，不采用其噪声策略作为必要组件 | 用公共/当前 generator 的 common-noise 配对部署差值、selector regret/headroom 和 detached selector probability 构造 deployment credit |
| GTRS | 面向 proposal scorer 的词汇 dropout、跨 proposal 泛化和 refinement | 不直接复制 vocabulary dropout 或 refinement head | 独立 selector 仅作 PDM-calibrated relative-harm reranker，带 Stage25 risk/OOD fallback；先离线 C2 验证 |
| DriveSuprim | coarse-to-fine hard-negative scorer | 不把 coarse-to-fine scorer 作为唯一 selector | 将“成熟场景不受伤害”写成显式 deployment safety gate，并把 selector 只作为可替换接口 |
| CCTR 等校准方法 | context-aware calibration | 不使用额外的场景分类器作为奖励替代 | 对每个 candidate 的相对伤害、灾难概率和 headroom 做可解释校准，并保留 PDM 只在训练/评估端出现 |

因此论文主线应写成：**GRPO generator 的 credit 分配由部署结果决定；selector 是独立的安全决策层，二者通过 common-noise counterfactual 配对而不是共享 PDM 分类头耦合。**

## 3. 三个必要的因果实验

1. **C0 / Public control**：官方 88.1 generator + 已冻结 Stage25 selector。
2. **C1 / CDC generator**：Stage33 CDC-GRPO generator + 冻结 Stage25 selector。若 C1 优于 C0，这是 generator 主贡献。
3. **C2 / External selector**：官方 88.1 generator + Stage33 external reranker + Stage25 fallback。若 C2 优于 C0，这是 selector 独立贡献。
4. **C3 / Full**：CDC generator + external reranker。仅在 C1、C2 各自通过 OOF gate 后运行。
5. **Ablation（可选）**：C1 + anchor/truncated control，作为 DiffusionDriveV2 风格的对照，不能替代 CDC 主线。

所有比较固定 token split、common-noise seed、PDM 版本、候选数 20、GRPO group size 8，并报告 mean、trimmed mean、95% bootstrap CI、成熟/困难场景分桶和 catastrophic count。

## 4. CDC-GRPO 目标

对 scene `i` 和 candidate `m`：

* `deployment_delta = PDM(current_i,m) - PDM(public_i,m)`；
* `selected_credit` 只给实际 selector 选中的 mode；
* `headroom_credit = relu(PDM(current_i,m) - PDM(current_i,selected) - margin)`；
* `soft_credit = selector_probability_i,m.detach() * headroom_credit`；
* `catastrophe_penalty` 对碰撞/越界等灾难 component 使用单侧惩罚；
* 在 anchor 内做标准化，再进入 PPO/GRPO ratio 和 reference-KL 约束。

没有 candidate-level labels 时必须 fail-closed，不能偷偷把当前已选 reward 当作 oracle。第一版训练模式可以复用 Stage32 的 selected-set trace 做 smoke；正式 C1 需要 candidate-level common-noise trace。

## 5. selector v0

`stage33_external_reranker_v1` 是独立 PyTorch 模块，输入 candidate trajectory、reference logits、selector probability，以及存在时的 Stage24/25 harm/risk/OOD 诊断；输出 mean、uncertainty、catastrophe logit。推理时不读取 PDM，使用 `mean - z * std`，并在 Stage25 eligible mask 或低置信度时回退到 Stage25/base mode。

第一版只做离线 C2，不改变 DiffusionDrive 的 trajectory head。训练脚本从已有 candidate bank 读取 PDM labels，按 log/token 做 deterministic split；评估脚本输出 selected/oracle/regret/fallback/catastrophe 和 per-domain 指标。

## 6. 通过/停止门槛

* Generator C1：pooled gain `>= +0.003`，whole-log CI lower `> 0`，mature gain `>= -0.0002`，catastrophic count = 0，且 candidate ceiling 不下降。
* Selector C2：相同安全门槛，且 switch rate `0.01–0.15`，fallback rate 不超过 0.30。
* 任一门槛失败，冻结代码和 checkpoint，先分析 bucket/domain，而不是直接增加 epoch。
* 只有 C1/C2 同时通过，才申请 8×H100/4090 做 C3 正式训练和最终 navtest。

## 7. 运行产物

* `navsim/agents/diffusiondrive/stage33_external_reranker.py`
* `navsim/agents/diffusiondrive/stage33_cdc_grpo.py`
* `scripts/evaluation/train_grpo_stage33_reranker.py`
* `scripts/evaluation/evaluate_grpo_stage33_reranker.py`
* `scripts/training/run_diffusiondrive_grpo_stage33_cdc_pilot.sh`
* 对应单测和 checkpoint audit

本文件是 Stage33 的冻结计划；任何修改训练信号、selector fallback 或基线 checkpoint 的行为都必须新建 amendment，并重新计算计划 SHA。
