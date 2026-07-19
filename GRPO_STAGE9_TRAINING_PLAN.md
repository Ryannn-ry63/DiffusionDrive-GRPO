# DiffusionDrive GRPO 完整训练纠正计划

## 总结

原始方向没有错：从 PDMS `0.84+` 的 DiffusionDrive checkpoint 初始化 current policy，保留 frozen reference 计算 KL，不加入 imitation loss，通过 GRPO 更新 planning diffusion decoder。

当前实现存在两个关键偏差：

- 两层 decoder 之间使用了 `detach()`，历史所有实验实际只更新第二层约 198 万参数，第一层没有梯度；
- 128 updates、batch 2 只覆盖约 256/6,119 scenes，是机制 pilot，不是充分训练。

后续回到学长原定顺序：先纠正 selector GRPO；selector 充分训练仍失败，才进入 generation GRPO。Phase 3–5 的 reference advantage、RLOO 和 hard projection 不再进入正式训练。

## 核心训练纠正

- 明确三套策略：
  - `current`：从正式 base 初始化，可训练；
  - `old`：current 的定期快照，仅用于 PPO ratio；
  - `reference`：完整两层 base decoder 的冻结副本，仅用于 KL。
- frozen reference 不参与 reward、advantage、候选选择或 shaping。
- 保持 perception、上游 Transfuser、anchor encoder 和输入特征冻结；保留部署一致的 `[8,0]` 两步 diffusion schedule。
- 新增 `grpo_decoder_gradient_scope=last_layer|all_layers`：
  - legacy 默认保留 `last_layer`，保证历史实验可复现；
  - 新正式 runner 强制 `all_layers`；
  - current policy 的两层 refinement 之间取消 stop-gradient，但前向数值保持不变；
  - old/reference 始终无梯度。
- 增加逐 decoder layer 的 shared-attention、FFN、time modulation、regression、classification 梯度范数及权重变化日志。

### Selector GRPO

- 作为第一正式路线。
- 同一 scene 的原始 20 个 candidate 构成 group，使用 raw online PDMS 做组内标准化 advantage。
- 两层 planning decoder 都设为可训练；selector policy 使用 categorical PPO。
- 损失固定为：
  - selector GRPO policy loss；
  - frozen-reference categorical KL，权重 `0.01`；
  - current/reference denoising-mean KL，权重 `0.1`，防止 shared decoder 更新破坏生成轨迹。
- 不使用 entropy bonus、priority sampling、reference reward advantage 或 imitation loss。

### Generation GRPO

- 仅在 selector 路线失败后启动。
- 沿用用户选择的原始 20-candidate scene group，不同时引入 K4。
- selector classification branches 冻结；两层 decoder core 与 regression branches 可训练。
- 损失固定为纯 group-zscore generation PPO loss 加 frozen-reference Gaussian KL `0.1`。
- 不使用 hierarchical、anchor baseline、RLOO、hard projection、oracle signal 或 selector weighting。

固定公共超参：seed 0、batch 2、LR `1e-6`、PPO clip `0.2`、old sync 32、uniform sampling、raw PDMS。

## 实验执行

1. 建立独立恢复文档和 stage-9 artifact 目录，保留现有 Phase 1–5 代码与结果，不覆盖历史 checkpoint。
2. 完成 one-batch gradient audit：
   - selector：第一层 shared/regression、第二层 shared/classification/regression 均有有限梯度；
   - generation：两层 shared/regression 均有有限梯度，classification 为零；
   - perception、reference、old policy 梯度严格为零。
3. 每条新路线先跑 U8 smoke，再跑完整一轮：
   - 6,119 个 train scenes；
   - batch 2，约 3,060 optimizer updates；
   - 保存 step 128/512/1024/2048 和 epoch-1 checkpoint。
4. Selector epoch-1 晋级到三轮的条件：
   - fixed-1024 selected delta `>0`；
   - selected CI 下界 `>-0.0005`；
   - collision/drivable/TTC 各自 delta `>=-0.001`；
   - 无 selected token delta `<-0.5`；
   - generation KL 有限且 rolling mean `<=2.5e-4`。
5. Selector 未通过即停止 selector，并按相同协议启动 generation；通过则从 epoch-1 checkpoint 连续恢复到 epoch 3，不重置 optimizer、old policy 或数据顺序。
6. Epoch-3 在 dev-select 上要求：
   - 相对 base selected CI 下界 `>0`；
   - safety components 不显著为负；
   - invalid candidate rate 不增加；
   - reference KL 不超过注册上限。
7. seed 0 通过后锁死路线与配置，补 seed 1/2；只有三 seed 同向才查看未使用的 dev-confirm，随后进行 paired full-navtest。
8. 科学可行性标准为三 seed selected mean CI 下界 `>0`；工程目标仍为 full-navtest absolute PDMS delta `>=+0.005`。

Oracle 和 candidate mean 继续记录，但只作为生成覆盖诊断，不再因单个未部署 oracle token直接否定训练；正式判断以 selector 实际部署的 selected raw PDMS 和安全分项为准。

## 接口与测试

- 新增完整 decoder 梯度范围配置和 selector generation-KL 权重。
- 正式 runner 只允许 `selector_group` 和 `generation_group` 两种干净目标，并支持 epoch-1 到 epoch-3 的严格恢复。
- 测试覆盖：
  - legacy stop-gradient 数值兼容；
  - full-gradient 模式前向完全一致；
  - current=old 时 PPO ratio 为 1；
  - current=reference 时两类 KL 为 0；
  - 两层实际获得预期梯度并在 optimizer step 后都发生权重变化；
  - reference checkpoint SHA 和参数始终不变；
  - selector/generation 冻结边界正确；
  - resume 前后 global step、optimizer、old-policy sync 和数据顺序连续；
  - loss 中不存在 imitation、GT regression 或 supervised diffusion 项；
  - focused pytest、`py_compile`、Bash syntax 和 `git diff --check` 全部通过。

## 固定假设

- 正式 base 继续使用 SHA256 为 `59a8de...19d` 的 DiffusionDrive checkpoint。
- `[8,0]` 是原模型真实部署的两步 refinement 路径，不扩展 diffusion horizon。
- frozen base 是独立的完整 reference decoder 副本，不是把 current decoder 的某一层冻结后当作 base。
- 当前 Phase 5 calibration、投影半径和 catastrophic oracle gate 不进入新训练。

## 当前执行状态（2026-07-18）

- [x] 独立 stage-9 文档与 `artifacts/grpo_stage9/` 目录。
- [x] `last_layer|all_layers` 梯度范围配置；legacy 默认 `last_layer`，正式 runner 强制 `all_layers`。
- [x] `current`、optimizer-step 同步的 `old`、独立完整 frozen `reference` 三策略结构。
- [x] `selector_group` 的 raw-PDMS categorical PPO、categorical KL `0.01`、denoising-mean KL `0.1`。
- [x] `generation_group` 的 group-zscore generation PPO、Gaussian KL `0.1` 和 classification 冻结。
- [x] 正式配置 fail-closed，排除 entropy、priority、reference advantage、RLOO、hard projection 等漂移。
- [x] 逐层梯度范数及权重变化日志。
- [x] Lightning checkpoint resume 接口与持久化 old-policy sync step。
- [x] `py_compile`、Bash syntax、`git diff --check` 以及现有 focused objective/probability tests。
- [x] 增补 decoder-gradient、reference immutability、freeze boundary 和 old-policy sync-state resume 专项单测；完整数据顺序恢复仍随 epoch-1→3 实跑审计。
- [x] 使用正式 checkpoint 完成 selector/generation one-batch gradient audit。
- [x] 运行 selector U8 smoke。
- [x] 运行 selector 完整 epoch 1（3,060 updates）并执行 fixed-1024 晋级门槛；结果未通过。
- [x] 根据 selector gate 停止 selector epoch-3，启动 generation 路线。
- [ ] seed 1/2、dev-confirm 和 paired full-navtest。

纠正后的代码、专项审计、两条路线的 U8、完整 epoch-1 与 fixed-1024 评估均已执行；两条路线都触发预注册停止边界，因此 epoch 3、seed 1/2、dev-confirm 与 full-navtest 按计划不再启动。

### 2026-07-18 实跑审计补充

- focused Stage-9 tests `5 passed`；连同 generation objective tests 为 `27 passed`。
- selector U1：layer-0 shared/regression 与 layer-1 shared/regression/classification 梯度非零；两层 optimizer 后最大权重变化均为 `3.5762787e-7`；perception 梯度为零，reference/old 相对 base 逐 tensor 最大差为零，ratio `0.9999998`。
- generation U1：两层 shared/regression 梯度非零，两个 classification branch 与 perception 梯度均为零；两层 optimizer 后最大权重变化均为 `3.5762787e-7`；reference/old 相对 base 最大差为零，ratio `1.0`。
- selector U8：valid-group fraction `1.0`，categorical KL `2.0219e-5`，denoising-mean KL `1.6837e-5`，clip fraction `0`；两层预期梯度有限且非零，perception 梯度仍为零。
- 正式 checkpoint 构造和 U8 暴露并修复了三个运行接线问题：YAML 漏注册 clip ratio、validation 错误计算 PPO、Stage-9 checkpoint callback 错误监控不存在的 validation reward。修复后 U1/U8 均干净退出并保存 checkpoint。
- selector epoch-1 fixed-1024：selected delta `+0.003988`，95% CI `[-0.005478, +0.013453]`，oracle delta `+0.000374`；collision/TTC 均为 `-0.002930`，最差 selected token `-0.947532`，最后 128-step generation-KL 均值 `0.021628`。均值有正信号，但 CI、安全、tail 与 KL 均未通过预注册 gate，因此不恢复到 epoch 3。
- generation U8：valid-group fraction `1.0`，Gaussian KL `1.2178e-5`，clip fraction `0`；两层 shared/regression 梯度有限且非零，classification/perception 梯度为零。generation epoch-1 随后完整执行。
- generation epoch-1 fixed-1024：selected delta `+0.009044`，95% CI `[-0.000833, +0.018945]`，oracle delta `-0.004414`；collision `-0.000977`、drivable `+0.009766`、TTC `+0.004883`，selector mode-switch rate 仅 `3.32%`。最差 selected token 为 `-1.0`，最后 128-step Gaussian-KL 均值 `0.014303`。均值和多数安全分项显示更强正信号，但 CI、tail 与 KL 未通过固定 gate，因此不进入 epoch 3、seed 1/2 或 dev-confirm。

Stage-9 在预注册停止边界结束。结论不是“组内竞争无效”：selector 与 generation 的 fixed-1024 selected point estimate 分别为 `+0.003988` 和 `+0.009044`；结论是当前固定 KL/PPO 配置无法同时控制极端 token 与完整 epoch 的函数漂移，尚不足以证明工程可行性。

最终验证：focused pytest `51 passed`，相关模块 `py_compile`、Stage-9 runner `bash -n` 与 `git diff --check` 均通过；两个 epoch-end checkpoint 的 frozen reference 与 perception 相对 base 最大逐 tensor 差均为 `0`，generation classification branch 最大差为 `0`。
