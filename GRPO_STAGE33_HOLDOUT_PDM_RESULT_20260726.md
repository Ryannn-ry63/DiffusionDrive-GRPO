# Stage33 holdout PDM result（2026-07-26）

## 1. 结论

Stage33 selected-common-noise CDC smoke **没有通过 PDM gate，不进入 navtest**。

两个 fold、两个 common-noise namespace、四个 checkpoint 的 catastrophic selector switch 均为 0，但所有 checkpoint 的跨-fold selected PDMS 平均增益均为负，且每个单-fold 95% CI 都跨过 0。

## 2. Selected PDMS

| step | fold0 gain vs public | fold1 gain vs public | 两 fold 简单平均 |
|---:|---:|---:|---:|
| 48 | -0.000478 | +0.000019 | -0.000229 |
| 96 | +0.000172 | -0.000445 | -0.000137 |
| 144 | -0.000286 | -0.000450 | -0.000368 |
| 192 | -0.000662 | -0.000388 | -0.000525 |

最好单点为 fold0/step96 的 `+0.000172`，但：

* fold1 同 step 为 `-0.000445`；
* fold0/step96 的 10% trimmed mean 为 `-0.000157`；
* wins/losses 为 `635/1298`；
* paired 95% CI 为 `[-0.001125, +0.001469]`；
* candidate mean 和 oracle ceiling 分别下降 `-0.000463`、`-0.000244`。

因此该正均值由少数尾部正样本驱动，不是稳定泛化提升。

## 3. Generator/selector 诊断

* Stage25 selector switch rate 基本与 public 相同（约 1.8%–2.2%），说明这轮差异主要来自 generator，而不是 selector 改变选择策略。
* 除少数 noise/fold 外，candidate mean 与 oracle reward 均下降或近似持平，说明 generator ceiling 没有提高。
* fold1/step144 的 wins 多于 losses、trimmed mean 为正，但均值为负，说明少量尾部损失吞掉了多数小收益。
* 虽然 selector 内部 catastrophic switch count 为 0，system-vs-public 的 paired delta 仍出现极少数 `<= -0.5` 的尾部退化，严格安全门槛仍失败。
* checkpoint audit 明确记录 `candidate_level_trace=false`；本阶段只是 selected-pair CDC smoke，不能代表完整 candidate-level CDC-GRPO。

## 4. 决策

1. 不选择任何 Stage33 smoke checkpoint 进入 navtest。
2. 不继续增加 epoch；step 越长整体越差，继续训练没有证据支持。
3. 下一阶段必须实现真正的 candidate-level common-noise CDC：
   * 对 20 个 candidate 计算 current/public paired reward；
   * credit 给 deployed mode，同时给 selector-probability-weighted headroom；
   * 加入 per-candidate catastrophe mask 与 mature-scene cap；
   * selector 保持冻结，先单独验证 generator ceiling；
   * 只有 candidate/oracle ceiling 和 selected PDMS 同时跨 fold 为正，才启用 external reranker 和 navtest。

聚合原始结果：`artifacts/grpo_stage33/pdm_summary.json`
