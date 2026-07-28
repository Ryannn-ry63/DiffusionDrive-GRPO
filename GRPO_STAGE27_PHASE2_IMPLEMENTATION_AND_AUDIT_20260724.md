# Stage27 Phase 2 实现与短程审计记录

日期：2026-07-24

## 目标

Stage27 以公开 `88.1041` PDMS checkpoint 为唯一 generator/reference，复现
Stage26 已验证的两级 selector 路径：

1. Stage24 学习候选安全性与价值。
2. Stage25 学习相对 fallback 的伤害风险。
3. 在独立 fold4、两个未参与训练的 noise namespace 上校准和选择分支。
4. 只有 selector gate 通过后，才允许进入 Stage27 GRPO generator 训练。

公开 checkpoint：

`/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS`

SHA256：

`008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b`

## 已实现分支

- `S-public`：只使用 3 个 public-88.1 candidate banks；Stage24/25 各训练
  3 epochs。
- `S-multi`：先使用 15 个 Stage26 跨 generator banks，再使用 3 个
  public-88.1 banks；Stage24/25 各训练 18 epochs。

两条分支并行训练，彼此不共享 selector 参数。generator/trunk 始终冻结。

## 2026-07-24 短程 audit 结果

运行命令：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage27_public88_selector_branches_h100.sh audit
```

结果：

- `S-public`: PASS
- `S-multi`: PASS
- 两条 Stage24 均为 8 updates，embedding count 均为 320。
- 两条 Stage25 均为 8 updates。
- 公版 checkpoint 的 763 个原始 tensors 在 Stage24/25 后全部 bitwise
  unchanged。
- Stage25 训练后，Stage24 selector 与 embedding moments 全部 bitwise
  preserved。

日志：

`artifacts/grpo_stage27/h100_logs/20260724T055238Z_selector_audit`

冻结审计：

- `artifacts/grpo_stage27/selectors/public/audit/frozen_selector.json`
- `artifacts/grpo_stage27/selectors/multi/audit/frozen_selector.json`

## H100 正式执行顺序

### 1. 并行训练两个正式 selector 分支

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage27_public88_selector_branches_h100.sh formal
```

该命令在 GPU 0/1 并行运行 `S-public` 与 `S-multi`，并自动完成 Stage24、
Stage25 checkpoint 冻结和逐 tensor 审计。


#### 2026-07-24 正式训练结果

状态：两条分支均 PASS。

- `S-public`
  - Stage24: epoch 2 / global step 6114
  - Stage25: epoch 2 / global step 6114
  - Stage24 embedding count: 244500
  - Stage24 SHA256:
    `e36291f90bec9a370e3252bcff7cdf9f25a0397d925e0f06dac005b6982c619b`
  - Stage25 SHA256:
    `9a7a202830f0dcd87c5aeac2f91a3b2dc3958584cc965ecf4b7048f56ceca91b`
- `S-multi`
  - Stage24: epoch 17 / global step 36684
  - Stage25: epoch 17 / global step 36684
  - Stage24 embedding count: 1467000
  - Stage24 SHA256:
    `c7364dd32e3397f2500e6ce274384852f5b881716b931cc595aa656a7fd78cfb`
  - Stage25 SHA256:
    `023d7b6b77bb8fa2dc3e779849f3716688abf0796b019416494c80b29615b691`

两条分支的 763 个 public checkpoint tensors 全部 bitwise unchanged；
Stage25 训练后 Stage24 selector 与 embedding moments 全部 bitwise
preserved。

正式日志：

`artifacts/grpo_stage27/h100_logs/20260724T080335Z_selector_formal`
### 2. fold4 原始收集、独立校准和预注册分支选择

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage27_public88_selector_fold4_h100.sh collect
```

该命令在 GPU 0-3 并行评估两个分支 × 两个 noise namespaces，随后自动
执行校准。硬 gate：

- pooled candidate-level gain `>= 0.003`
- whole-log bootstrap 95% CI lower bound `> 0`
- 两个 namespace 分别 nonnegative
- switch rate 位于 `[0.01, 0.15]`
- collision/drivable/TTC/comfort/direction 五项均值下降不超过 `0.0005`
- switched trajectory 中 `delta <= -0.5` 的数量为 0

若两个分支都通过，优先选择 whole-log CI lower bound 更大的分支；差值
小于 `0.0005` 时，按预注册规则选择 `S-multi`。

#### 2026-07-24 fold4 校准结果

四个收集任务全部完成，每个任务 1021 tokens、0 failures。自动选择结果：
`S-multi`。

- `S-multi`：PASS
  - pooled mean gain: `+0.0030522449`
  - whole-log bootstrap 95% CI:
    `[+0.0019085286, +0.0041896120]`
  - namespace 20260821: `+0.0030302780`
  - namespace 20260822: `+0.0030742119`
  - switch count/rate: `48 / 2042`, `2.3506%`
  - catastrophic count: `0`
  - guard component mean deltas:
    `[0, 0, 0, -0.0004897160, 0]`
- `S-public`：FAIL
  - pooled mean gain: `+0.0124235182`
  - whole-log bootstrap 95% CI:
    `[+0.0085820617, +0.0161323326]`
  - switch count/rate: `271 / 2042`, `13.2713%`
  - catastrophic count: `7`
  - 失败原因：五项 guard component 下降超过门槛，且灾难切换不为零。

该结果说明只使用 public bank 的 selector 更激进但尾部风险不可接受；跨
generator 的 `S-multi` 将增益压缩为保守、跨 namespace 稳定且统计显著的
安全增益。此处为 fold4 candidate-level selector 增益，不是最终 NavTest
PDMS。

选择文件：

`artifacts/grpo_stage27/selectors/selection.json`

### 3. 用已选择阈值重新部署并执行运行时 gate

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
bash ./run_stage27_public88_selector_fold4_h100.sh deploy
```

只有最终生成的
`artifacts/grpo_stage27/selectors/selector_gate.json`
中 `passed=true`，才开始设计和训练 Stage27 generator。不得使用 NavTest
结果反向调整 selector 阈值。

#### 2026-07-24 deploy 运行时 gate

状态：PASS，`stop_before_generator_training=false`。

- count: 2042
- pooled mean gain: `+0.0030522449`
- whole-log bootstrap 95% CI:
  `[+0.0019085286, +0.0041896120]`
- switch count/rate: `48`, `2.3506%`
- harmful/catastrophic switches: `0 / 0`
- 离线与运行时 mean gain、switch count 均精确一致。
- 两个 deploy artifacts 均为 1021 tokens、0 failures。

最终 gate：

`artifacts/grpo_stage27/selectors/selector_gate.json`

SHA256：

`1b442569dddafb82ea0ad0d07b34c6ded1f93151851c556a346c0dd8b98ca636`

## 当前结论

Stage27 Phase 2 已完成正式训练、独立 fold4 校准、预注册分支选择和 deploy
运行时重放，并通过全部统计与安全 gate。它证明 `S-multi` 能在公开 88.1
generator 的候选集上提供保守、统计显著、零伤害切换的 selector 增益。

该增益仍是 fold4 candidate-level 结果，而不是最终 NavTest PDMS。当前已经
获准进入 generator 设计；`S-public` 仅保留为独立研究分支，不插入冻结主线。
