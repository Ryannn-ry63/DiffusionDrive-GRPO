# Stage22：同噪声配对约束残差 GRPO（预注册执行计划）

记录时间：2026-07-22（UTC）  
状态：在任何 Stage22 代码修改、训练或结果查看之前锁定。

## 1. 研究目标与不可偏离的主线

Stage22 继续验证同一个论文命题：**GRPO 能否在官方 DiffusionDrive 强基线（公开 PDMS 约 88.1）上，提升最终被执行的单条轨迹质量。**

Stage21 已经在 fold4 上得到最高约 `C-B = +0.01577` 的平均候选收益，但同时出现 mature 场景退化和约 0.9% 的灾难性尾部事件。Stage22 的目标不是用更保守的模型把平均收益压回零，而是在保留 Stage21 约 `+0.015` 收益潜力的同时消除跨日志尾部灾难。

因此锁定以下原则：

1. 只训练并最终部署一个生成器；不使用推理期双模型、base fallback 或额外 selector 来制造提升。
2. 最终归因仍为 `C-B`：同一官方强基线、同一 full diffusion schedule 下，Stage22 GRPO 生成器相对基线的净贡献。
3. 安全约束不能替代性能提升。任一正式 gate 必须同时满足平均 `C-B >= +0.005`；最终希望复现/超过 Stage21 的 `+0.015`，而不是接受“更安全但没有收益”。
4. 不继续延长 Stage21 checkpoint；Stage22 从官方强基线重新训练，避免把旧的跨日志漂移带入新实验。

官方强基线 checkpoint：

`/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model`

锁定 SHA256：

`59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d`

## 2. Stage21 失败模式与 Stage22 假设

Stage21 的 current rollout 与 frozen-base 参考均值来自不同随机扩散噪声，reward difference 混入了较大的采样方差；同时只用单条随机 BC teacher 约束更新。尾部诊断显示，若干 mature 场景中仅 0.75–1.32 cm 的轨迹变化就跨过 PDM 的碰撞、TTC 或 drivable 边界。因此仅靠欧氏残差阈值或训练后 fallback 无法可靠解决问题。

Stage22 的假设是：

- 对 current 和 frozen base 使用完全相同的初始 anchor noise、每一步 DDIM transition noise 与 final noise，可把 reward 信号变成低方差的 paired delta；
- 使用 frozen base 的 8 条链做精确同方差 mean-KL，而不是单一样本 BC，可在 mature/安全边界附近提供稳定约束；
- 仅在两层 decoder 的最终 regression linear 上训练低秩 residual adapter，可限制跨日志漂移，同时保留对困难场景的有效轨迹修正。

## 3. 锁定算法：`diffgrpo_paired_residual`

### 3.1 同噪声 paired rollout

每个 selected anchor 采样组大小 `G=8`。对每条 current chain，同时运行 frozen-base chain，并强制复用：

- 相同 initial anchor noise；
- 相同每一步 DDIM transition noise；
- 相同 final transition noise；
- 相同 scheduler、timestep 与 `eta`。

为 current/base 共 16 条轨迹计算 PDM 总分与 component scores，并保留逐 pair 结果。实现必须经过“相同权重 + 相同噪声时 current/base action、reward delta 为零”的测试。

### 3.2 paired-delta advantage

对第 `g` 个 pair：

`d_g = R_current,g - R_base,g`

先在每个 group 内对 `d` 做 z-score。优势规则锁定为：

- 正例：`d > +0.01`；
- 普通负例：`d < -0.01`；
- mature pair（`R_base >= 0.75`）负例：`d < -0.002`；
- current 相对同 pair 的 base 出现 collision、drivable 或 TTC component 变差：直接设为 `-2`；
- 正优势：`0.5 * max(z, 0) + 0.5 * min(|d| / 0.1, 2)`；
- 负优势使用对称形式；mature 负优势再乘 `2`，最终绝对值裁剪到 `2`；
- 未超过对应阈值的 pair 优势为零。

训练样本权重只使用 inverse-log-frequency，裁剪到 `[0.5, 2.0]` 后在当前训练 manifest 内归一化。删除 difficulty upweight；安全/mature 负例的最终权重下限为 `1.0`。

### 3.3 精确 reference-mean KL

在 8 条 frozen-base chain 的 state 上比较 current 与 frozen reference 的预测 mean，计算相同标准差、按 action dimension 归一化的精确高斯 KL：

- 普通 pair 系数 `0.1`；
- mature pair（base reward `>=0.75`）系数 `0.5`；
- KL 对 G=8 pair 求均值；
- reference 路径必须完全 `no_grad`，且不得更新任何 reference tensor。

### 3.4 LoRA residual 的唯一可训练参数

仅包装两个 decoder layer 的：

`task_decoder.plan_reg_branch[-1]`

锁定：

- rank `r=8`；
- alpha `=8`，scale `alpha/r=1`；
- adapter dropout `0`；
- A 正常小随机初始化，B 零初始化，保证训练起点严格等于官方 base；
- 原 regression weight/bias、其余 decoder、classification、perception、selector、reference 全冻结；
- AdamW：LR `3e-5`，weight decay `0`，gradient clip `1.0`。

训练 checkpoint 必须可以合并为普通单模型：

`W_merged = W_base + (alpha/r) * B @ A`

所有 calibration、transfer 与最终评测只允许使用合并后的普通 generator checkpoint。合并前后输出最大绝对误差必须 `<=1e-6`。

### 3.5 采样与训练常数

- full schedule：`[32, 24, 16, 8, 0]`
- scheduler train timesteps：`125`
- group size：`8`
- PPO/GRPO clip gamma：`0.6`
- 数值：FP32
- global batch：`64`
- 8 GPU 时：device batch `1`、accumulate grad batches `8`
- checkpoint candidates：epoch `1/2/3/4`
- 最多 `4 epochs`，不因结果接近门槛而延长训练

## 4. 数据隔离与阶段划分

Stage22 创建独立 manifest/provenance，复用 Stage21 的 whole-log folds，但不使用 Stage21 的 offline base reward 作为训练目标。

| 用途 | whole-log folds | 数量 | 是否消耗 |
|---|---:|---:|---|
| checkpoint-selection fit | 0–2 | 3056 | 训练集 |
| inner calibration | 3 | 1019 | 选 epoch 后消耗 |
| locked retrain fit | 0–3 | 4075 | 训练集 |
| known stress | 4 | 1021 | 已知压力集 |
| internal test | 5 | 1023 | 一次性 |
| formal train | 0–5 | 6119 | 最终训练 |

不同集合必须 whole-log 零重叠。inverse-log-frequency 权重在每个实际 fit manifest 内独立确定并记录；formal all-6119 重新计算。

## 5. 预注册 gates 与停止规则

所有比较均为 paired `C-B`，并报告 log-cluster bootstrap CI、hard/mature 子集、collision/drivable/TTC component 和灾难性尾部。任何 gate 失败都停止，不扫描 LR/rank/threshold、不做 checkpoint soup、不用 fallback，也不提前打开下一集合。

### 5.1 inner calibration：fold3

对 epoch 1/2/3/4 使用 default 与 namespace `20260801`：

- 两噪声合并 mean `C-B >= +0.005`；
- 每个 namespace mean 均 `>0`；
- log-cluster CI lower `>0`；
- hard（base `<0.75`）mean `>= +0.010`；
- mature（base `>=0.75`）mean `>= -0.001`；
- collision/drivable/TTC 平均退化均不得超过 `0.001`，且不得统计显著退化；
- catastrophic（`C-B <= -0.5`）Clopper–Pearson upper `<=0.005`。

满足条件者取 mean 最大；若与最佳差 `<=0.001`，选更早 epoch。

### 5.2 fresh retrain 后 known-stress：fold4

从官方 base 在 folds0–3 重新训练到锁定 epoch，fold4 使用 default 与 `20260729`：

- mean `C-B >= +0.005`，CI lower `>0`；
- hard `>=+0.010`；mature `>=-0.002`；
- 相同 safety 约束；catastrophic CP upper `<=0.005`。

fold4 是已消耗诊断集，不参与 epoch 选择。

### 5.3 internal-test：fold5

仅在 fold4 通过后打开一次。使用 default 与 `20260801/20260802/20260803`：

- 每个 namespace mean 均 `>0`；
- 合并 mean `>=+0.005`，CI lower `>0`；
- 相同 hard/mature/safety/tail 门槛。

### 5.4 dev-select3072 与 formal run

fold5 通过后才打开 consumed dev-select3072（default + `20260804`），使用相同 gate。通过后：

1. 从官方 base、seed0、all-6119 fresh formal train 到锁定 epoch；
2. 先跑 dev-confirm1024；
3. 通过后才进入 NavTest PDMS。

最终归因：

- A：官方 base，short schedule `[8,0]`
- B：官方 base，full schedule `[32,24,16,8,0]`
- C：Stage22 merged GRPO，full schedule
- `C-B`：GRPO 的净贡献
- `C-A`：完整系统相对常用短 schedule 的总贡献

NavTest `C-B >=+0.005` 视为有效；`>=+0.010` 为论文目标。若 seed0 NavTest `>=+0.005`，再使用外部 8×H100/4090 跑 seed1/2；在此之前不消耗外部多卡资源做长 epoch。

## 6. 强制审计与测试顺序

代码完成后必须依次通过：

1. common-noise bundle 完全一致；同权重下 action/delta 为零；
2. paired reward、advantage 符号、mature threshold/weight、safety override；
3. G=8 exact KL 数值；
4. 只有 adapter A/B 有梯度，base/reference 全冻结；
5. LoRA merge equivalence `<=1e-6`；
6. DDP resume 与 RNG 可复现；
7. Stage13–21 legacy compatibility；
8. U8 单卡 smoke、resumed U32、8-GPU U64，并审计 finite、trainable tensor 集合、LR 与 resume 连续性。

只有全部通过，才允许启动 development 训练与上述逐级 gates。

## 7. 成功与失败的解释边界

- 若 Stage22 达到 `>=+0.005` 且清除 Stage21 的 mature/tail 失败，则说明同噪声 paired GRPO + constrained residual 能把生成器收益转化为可泛化的单模型收益。
- 若安全改善但 mean `<+0.005`，不视为论文成功，也不声称优于 Stage21。
- 若 mean 保持在约 `+0.015` 且通过所有 safety/tail gate，这是首要理想结果。
- 若任一预注册 gate 失败，Stage22 到此停止，保留未打开的测试集，不通过事后修改门槛制造成功。

## 8. 开发前算法勘误：零梯度对称性破缺

记录时间：2026-07-22（UTC），发生在任何 Stage22 development 训练和 fold3 结果之前。

U8 首次真实运行证明，原第 3.2 节存在一个数学退化点：LoRA-B 零初始化使 current policy 严格等于 frozen base；完全同噪声又使每个 current/base action、reward delta 和 exact KL 严格为零。因此原 paired-delta-only objective 在第一个 optimizer step 的梯度严格为零，LoRA-B 在 8 步后仍逐元素为零。若不修正，任何 epoch 数都只能复现 base，不可能验证 GRPO 或保持 Stage21 的 +0.015 潜力。

在不查看任何 development/fold3 分数的前提下，锁定以下最小勘误：

1. 原 paired-delta advantage、安全 override、mature 规则、exact KL、LoRA 零残差初始化及全部 gates 保持不变。
2. 对尚未触发 paired 正/负规则且没有 safety regression 的 pair（即 paired delta 位于对应 dead zone），加入标准 GRPO 组内探索优势：先在 8 条 current reward 内做 valid z-score，再裁剪到 [-1,1]，乘固定系数 0.25。
3. 一旦 pair 达到 d>+0.01、普通 d<-0.01、mature d<-0.002 或 safety override，原 paired advantage 完全接管，不与探索项相加。
4. 探索项只负责从严格相同的 base 打破对称性；其系数不扫描、不延长 epoch、不因结果修改。
5. 新的实现名称/manifest provenance 更新为 paired_delta_bootstrap_exact_kl_lora_v2；数据划分和测试集仍保持封闭。

这一勘误仍是纯 GRPO：探索信号来自同一 selected anchor 的 8 条 on-policy rollout 的相对 PDM reward；frozen base pair 继续承担绝对收益、安全边界和 KL 约束。它不引入监督标签、Stage21 checkpoint、推理 fallback 或额外模型。

## 9. 执行结果与停止决定（2026-07-22 UTC）

### 9.1 训练和实现审计

- v2 U8 单卡、断点续训 U32、8-GPU DDP U64 均完成；U64 明确达到 global step 64。
- U64 审计通过：仅 4 个 LoRA A/B 张量可训练，官方 base 与 frozen reference mismatch 均为 0，LR=3e-5、weight decay=0、gradient clip=1.0，64 个记录步全部 finite，禁止模块梯度均为 0。审计见 `artifacts/grpo_stage22/u64_ddp_audit.json`。
- Stage13--22 共 53 项回归测试通过；Python 编译与 calibration shell 语法检查通过。
- development 在 folds0--2 的 3056 tokens 上完成预注册的 4 epochs，共 192 global steps；候选 checkpoint 为 step 48/96/144/192，没有延长 epoch 或修改超参数。
- step192 最终审计通过：仅 4 个 LoRA 张量、base/reference mismatch=0、192 个指标步 finite、禁止模块梯度=0。审计见 `artifacts/grpo_stage22/development_step192_audit.json`。
- 四个 LoRA checkpoint 均成功合并为普通单 generator，remaining LoRA keys 为空；每个 fold3 单元均完成 1019/1019 tokens 且 0 failure。

### 9.2 fold3 inner-calibration gate

双 namespace（default/-1 与 20260801）的 paired `C-B` 结果：

| epoch | default | 20260801 | 合并 mean | log-cluster 95% CI |
|---:|---:|---:|---:|---:|
| 1 | -0.000223 | -0.000352 | -0.000287 | [-0.001431, +0.000522] |
| 2 | -0.000527 | +0.000209 | -0.000159 | [-0.001412, +0.000918] |
| 3 | -0.000569 | +0.000366 | -0.000101 | [-0.001374, +0.000970] |
| 4 | -0.000557 | +0.000291 | -0.000133 | [-0.001433, +0.000948] |

四个候选均失败，`selected_epoch=null`。共同失败项为：至少一个 namespace 不为正、合并 mean 低于 +0.005、CI lower 不为正、base<0.75 的 hard gain 低于 +0.010。catastrophic CP upper=0.002326 仍满足尾部门槛，collision 没有退化，但这些安全结果不能替代平均 PDMS 增益。

按照第 5.1 和第 7 节的预注册停止规则：Stage22 在此停止，不打开 fold4/fold5/dev-select/NavTest，不延长 epoch，也不把训练期 reward 当成 PDMS 成功。Stage22 的可靠结论是 constrained paired residual 成功保持了 base 附近的稳定性，却没有保住或复现先前约 +0.015 的潜在增益；问题仍位于“更好 generator candidates 如何经冻结 selector 转化成最终 selected trajectory”。正式 gate 记录见 `artifacts/grpo_stage22/calibration_v2/selection.json`。
