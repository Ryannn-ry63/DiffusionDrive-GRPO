# Stage18：平均 PDMS 优先的独立验证计划

日期：2026-07-21
状态：**seed0 dev-select3072 门禁未通过，已按预注册规则停止**

## 1. 目标与边界

Stage16 已证明 GRPO generator 能显著提高候选轨迹质量；Stage17 用冻结的 paired-risk verifier 和 base fallback 将候选优势转成最终 selected trajectory，但在 fixed1024 上仍有 3 个低分 false negative。Stage18 不重新训练或调参，而是对已冻结方法做平均 PDMS 优先的独立验证。

本阶段锁定：

- Stage16 generator：`grpo-07-536.ckpt`
- Stage17 verifier：epoch16，`grpo-15-1728.ckpt`
- paired-risk threshold：`0.7683204412460327`
- 采样 schedule：`[32, 24, 16, 8, 0]`
- truncation steps：`32`
- base model、随机种子和评估实现均保持不变

Stage17 fixed1024 已被分析，因此只作为 development diagnostic；其严格尾部失败必须在论文中披露，但不再作为 Stage18 平均指标的阻断条件。

## 2. 同噪声统计定义

为避免独立运行带来的 diffusion noise 不匹配，Stage18 的 B/C/D 必须从同一个 Stage17 paired-risk artifact 派生：

- A：原始短 schedule 输出，来自独立冻结的 original artifact
- B：paired artifact 内的 `base_reward/base_components`
- C：paired artifact 内的 `current_reward/current_components`
- D：paired artifact 最终的 hybrid selected reward/components

门禁脚本必须 fail closed，并检查 token 集、checkpoint SHA、threshold、calibration provenance、schedule、fallback 选择一致性以及 selected trajectory 与被选分支的一致性。

## 3. Seed0 dev-select3072 主门禁

仅在门禁实现及 fixed1024 回归通过后，运行未见过的 `dev_select3072_manifest.json`。通过条件预注册为：

- `mean(D-B) >= +0.015`
- token bootstrap 95% CI 的 `D-B` 下界 `> 0`
- `mean(D-A) >= +0.020`
- token bootstrap 95% CI 的 `D-A` 下界 `> 0`
- `mean(D-C) >= 0`
- collision、drivable、TTC 三项平均差值相对 B 均非负
- fallback rate `<= 35%`

worst delta、bottom-1% mean、catastrophic-admission count/rate 继续记录，但在平均 PDMS 主线中仅作为非阻断诊断，不允许据此事后修改 threshold。

若 seed0 主门禁失败，立即停止本阶段，不在 dev-select 上调 threshold；后续另立模型阶段，采用按 log 分组、multi-noise 和 verifier ensemble 重新训练。

## 4. 多种子复现

seed0 主门禁通过后，使用外部 8×H100/4090 环境训练 Stage16 generator 的 seed1、seed2；每个 seed 固定训练 8 epoch，并复用同一个 Stage17 verifier 与 threshold。通过条件：

- 每个 seed 的 `D-B > 0`
- 三种子平均 `D-B >= +0.010`
- seed-stratified bootstrap 95% CI 下界 `> 0`
- 聚合 `D-C >= 0`
- 聚合 collision、drivable、TTC 差值均非负

不根据中间 seed 结果增加 epoch 或改变超参数。

## 5. Dev-confirm1024 与 navtest

多种子通过后才打开独立 `dev_confirm1024_manifest.json`：

- 三种子平均 `D-B >= +0.008`
- 三种子平均 `D-A >= +0.012`
- 聚合 `D-C >= 0`
- token bootstrap 与 seed-stratified bootstrap 的 `D-B` CI 下界均 `> 0`
- 聚合 safety component 差值均非负

dev-confirm 通过后，navtest 仅提交冻结的 seed0 D：

- 最低成功目标：`D-B >= +0.005`
- 论文目标：`D-B >= +0.010`

## 6. 执行顺序

1. 实现并测试 Stage18 同噪声平均门禁。
2. 在已消费 fixed1024 上做结构和数值回归，不作模型选择。
3. 审计 fixed1024、dev-select3072、dev-confirm1024 token 互斥。
4. 运行 seed0 dev-select3072 并一次性判门。
5. 仅在通过时准备外部 seed1/2 八卡训练脚本。
6. 多种子通过后再执行 dev-confirm；最后才进入 navtest。

## 7. 执行记录

- 2026-07-21：在查看 dev-select3072 结果前写入本预注册文档。
- 2026-07-21：实现同噪声 Stage18 门禁；fixed1024 diagnostic 精确复现
  `D-B=+0.021662`、`D-C=+0.001932`、fallback `18.07%`，并保留 3 个
  admitted catastrophic 和 worst `-0.829745` 的非阻断披露。
- 2026-07-21：审计确认 fixed1024、dev-select3072、dev-confirm1024 两两
  零 token 重叠；最终相关测试共 12 项通过。
- 2026-07-21：首次 dev-select 启动在前向前安全失败，原因是 wrapper 默认指向
  `training_cache`，而独立 4096 集的冻结配套 cache 是
  `grpo_dev4096_feature_cache_20260717`。该 cache 包含全部 3072 个目标 token；
  未生成部分结果，也未改变任何阈值或门禁。随后仅修正 cache provenance 原样重跑。
- 2026-07-21：8 个 shard 均完成 384/384、零失败，按 manifest 顺序合并后一次性
  执行预注册门禁。结果：`D-B=+0.000561`，95% CI
  `[-0.002852,+0.004016]`；`D-A=+0.009411`，95% CI
  `[+0.005528,+0.013190]`；`D-C=+0.001104`；`C-B=-0.000543`；
  fallback `9.60%`。collision `-0.002279`、drivable `+0.001953`、TTC
  `-0.002930`（均为 D-B）。因此 D-B、D-A 与 safety 门禁失败。
- 2026-07-21：遵守预注册停止规则：不训练 seed1/2，不打开 dev-confirm，不进入
  navtest，也不在 dev-select 上重新选择 threshold。Stage18 的正式结论是 Stage16
  generator 的平均优势未跨日志泛化；Stage17 虽有 `D-C=+0.001104` 的恢复量，
  但不能把一个 `C-B<0` 的 generator 转成论文级提升。

## 8. 后续阶段的约束性结论

下一阶段必须作为新的模型阶段重新预注册，而不能继续修改 Stage18：generator 训练与
verifier 训练均按 log 分组切分，generator 使用 multi-noise reward，verifier 使用
跨日志 OOF/ensemble 校准。Stage18 dev-select 只能作为 development 诊断集；新的模型
选择需要另建内部 train/validation，最终仍保留 dev-confirm 与 navtest 未打开。
