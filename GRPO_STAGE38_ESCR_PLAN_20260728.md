# Stage38：Elite-Set Counterfactual Repair GRPO（ESCR）

日期：2026-07-28  
状态：冻结计划；实现和训练前的唯一 source of truth  
主线：ESCR  
后备方案：见 `GRPO_RESERVED_CHALLENGER_DESIGN_20260728.md`，Stage38 内禁止实现或混用

## 1. 决策与目标

Stage38 不从 Stage37 继续训练，也不立刻更换为 Challenger 架构。它回到目前最强、
且跨折统计显著的 Stage36 RGT192，并只解决 Stage36 已经定位的单一问题：

> 如何在保留 frozen public elite trajectories 的同时，让 GRPO 只强化能进入
> public elite set、并带来绝对 PDM 增益的新轨迹。

Stage38 的目标分为三层：

1. **最低机制目标**：修复 Stage36 的 public-top-5 退化，而不丢掉 Stage36 已有
   的 selected gain。
2. **可发表正结果目标**：在相同冻结 selector 下，OOF selected gain
   `>= +0.002`、bootstrap CI 下界严格大于 0、public-top-5 不退化。
3. **论文级上限目标**：public/current union safe-oracle headroom
   `>= +0.005`。只有达到这一条，才值得新开 selector 阶段继续兑现上限。

Stage38 不是一次无止境调参。最多训练到 96 个新增 optimizer steps；失败后冻结
结果，不扫描 LR、margin、KL、loss weight，也不延长 checkpoint。

## 2. 已冻结的证据与研究假设

### 2.1 为什么从 Stage36 出发

Stage36 RGT192 的两折 OOF 结果为：

- selected gain：`+0.001824692`；
- bootstrap 95% CI：`[+0.000692022, +0.003212392]`；
- hard-scene gain：`+0.008653745`；
- mature-scene gain：`-0.000058743`；
- safe-deployable-oracle gain：`+0.001829290`；
- public-top-5 delta：`-0.001064840`；
- catastrophic regression：`0`。

这说明 Stage36 已经得到真实 GRPO 增益，失败面集中在 elite retention。Stage38
必须使用 fold-matched Stage36 RGT192 作为 initializer，public 88.1 checkpoint
仍作为不可变 reference。Stage37 checkpoint 不得作为 initializer。

### 2.2 Stage37 给出的反例

Stage37 step192 把 candidate mean 提高到 `+0.001520`，但 raw oracle 只有
`+0.000104`，selected 只有 `+0.000314`，public-top-5 与 mature scenes 仍为负。
因此：

- 只抬高平均候选没有意义；
- 全局梯度投影不能替代正确的 upper-tail credit；
- 不能继续使用“组内相对较好即可获得正 advantage”的目标；
- selector 训练必须等 generator 的 union oracle gate 通过后再做。

### 2.3 Stage38 的可证伪假设

对 frozen public top-5 做逐 mode 的反事实替换，可以同时产生：

- public elite mode 退化时的绝对负 credit；
- 非 elite current mode 真正进入 public top-5 时的绝对正 credit；
- 所有 current candidates 都更差时，不会出现“相对最好但仍为正”的错误信号。

如果该目标在 96 个新增 optimizer steps 内仍不能同时修复 top-5 并保住 selected
gain，则共享 decoder 上继续做 retention 修补不再是主线，触发 Challenger 后备方案。

## 3. 冻结实验身份

### 3.1 Checkpoint 与数据

- Public/reference：
  `/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS`
- Public SHA256：
  `008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b`
- Initializer：只使用 freeze 中 basename 为 `grpo-step-192.ckpt` 的记录，
  不得把同为 global-step 192 的 `last.ckpt` 当作等价输入：
  - fold0：
    `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage36_rgt_pilot_f0_formal/2026.07.28.09.38.46/lightning_logs/version_0/checkpoints/grpo-step-192.ckpt`
    ，SHA256 `2a2302da415e38aa654c172858d3f89f8112f774a6215d89b9c150112f0dab33`；
  - fold1：
    `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage36_rgt_pilot_f1_formal/2026.07.28.09.56.24/lightning_logs/version_0/checkpoints/grpo-step-192.ckpt`
    ，SHA256 `0bafe817ebfe98032b3d4e9499eb78d547efcf7230ddc3c83aa4b1bd4e4e7a9b`。
- 训练集：沿用 Stage30 CV freeze 的 `train_except_fold{0,1}`，不得改变 token。
- Bucket manifest：沿用 Stage30 frozen `buckets_6119.json`。
- Fold：先 fold0 kill-test，通过后才允许 fold1。
- 训练 seed：fold0/fold1 固定为 `203800/203801`。
- 新评测 noise namespaces：`20261611/20261612`；不得用 Stage36/37 namespace
  选参。

### 3.2 Generator、selector 与训练边界

- `G=8` stochastic replicas，`M=20` anchors，full diffusion chain
  `[32,24,16,8,0]`。
- 训练 batch size 1，8-GPU DDP，gradient accumulation 8。
- LR `1e-6`，decoder all layers，layer-0 LR multiplier `0.1`。
- perception、backbone、classification head、所有 selector 参数严格冻结。
- 训练和所有 OOF 比较统一使用 frozen Stage25 S-multi selector：
  - checkpoint：
    `/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage27_selector_multi_stage25_formal_seed27125/2026.07.24.09.28.29/lightning_logs/version_0/checkpoints/grpo-17-36684.ckpt`；
  - checkpoint SHA256：
    `023d7b6b77bb8fa2dc3e779849f3716688abf0796b019416494c80b29615b691`；
  - calibration：`artifacts/grpo_stage27/selectors/multi/calibration.json`；
  - calibration SHA256：
    `fec2a10573e913e4452f8eea92e6334fcd5e6f1104e6244876770973a1f41990`。
- Stage37 JFI selector、calibration 和 checkpoint 不得进入 Stage38。
- PDM reward/component、public rollout、mode 选择和 ESCR advantage 全部 detach。
- 不使用 GT trajectory、imitation loss 或 supervised diffusion loss；public-policy
  BC/KL 只作为 trust-region regularizer。

## 4. ESCR 目标函数

### 4.1 Frozen public elite set

对每个 scene 和 stochastic group `g`，使用 public bank 的有效 PDM reward
确定 top-5 mode：

```text
E_g = stable_top5({(m, r_pub[g,m]) | valid_pub[g,m]})
U5(S) = sum(top5_valid_rewards(S))
```

排序并列时按 mode index 升序，保证所有 rank、GPU 和重跑完全一致。public
有效候选少于 5 个时，该 group 不产生 policy credit，并记入 invalid diagnostic。

### 4.2 单 mode 的 public-anchored substitution credit

对 current mode `m` 构造一个仅改变该 mode 的反事实集合：

```text
若 m 属于 E_g：
    S(g,m) = E_g 中 public mode m 替换成 current mode m
否则：
    S(g,m) = top5(E_g 并入 current mode m)

delta_set(g,m) = U5(S(g,m)) - U5(E_g)
```

这个定义具有三项必要性质：

- elite current trajectory 退化时 `delta_set < 0`；
- 非 elite candidate 只有超过 public 第五名并进入集合时才可能 `> 0`；
- 当全部 current candidates 都弱于 public elite set 时，不存在伪正 advantage。

current elite trajectory 无效时直接视为最大负 credit；无效的 non-elite
challenger 不参与训练。

### 4.3 Safety veto 与 advantage

Safety component 固定为 collision、drivable、TTC。current mode 相比同 anchor
public mode 任一 safety component 退化超过 `1e-6` 时，advantage 强制为 `-2`。
其余样本使用非对称 deadband：

```text
if delta_set > +0.001:
    A = (delta_set - 0.001) / 0.002
elif delta_set < -0.0001:
    A = (delta_set + 0.0001) / 0.002
else:
    A = 0

A = clip(A, -2, 2)
```

Mature bucket 的正 advantage 乘 `0.25`，负 advantage 保持不变。不得再做
group z-score、跨 anchor 标准化、Stage35 NCD 混合或 Stage37 PCGrad 投影。

### 4.4 Compact replay pool

每个 group 的 differentiable replay pool 最大为 8 个 mode，按下列固定顺序
去重：

1. public top-5 elite modes，全部保留；
2. `delta_set` 最大且超过正 margin 的最多两个 non-elite challengers；
3. frozen Stage25 selector 的 current selected mode（若尚未出现）。

未填满的 slot 标记 invalid，不用其他 mode 补齐。这样同时覆盖 elite repair、
upper-tail expansion 和实际 deployment mode，又不扩大为全 20-mode replay。

### 4.5 Loss

```text
L_policy = scene_normalized_mean(
    - logp_current * A_ESCR * step_discount
)
L_elite_bc = public-chain BC on all five public elite modes
L_active_kl = exact current/reference diffusion KL on active replay modes

L_total = L_policy + 0.1 * L_elite_bc + L_active_kl
```

- diffusion step discount 固定为 `0.6`；
- ordinary active KL coefficient 为 `0.1`；
- safety-vetoed mode 的 KL coefficient 为 `0.5`；
- 每个分支先按 scene 归一化，再做 batch mean，避免 active 数量改变 loss scale；
- 不使用额外 gradient projection；梯度裁剪保持 `1.0`。

## 5. 实现接口与产物

### 5.1 新增训练接口

新增且只允许以下身份：

```text
grpo_training_mode = stage38_elite_set_counterfactual_repair_grpo
generation_policy_algorithm = diffgrpo_elite_set_counterfactual_repair
stage38_objective_revision = elite_set_counterfactual_repair_v1
```

配置项固定为：

```text
stage38_elite_width = 5
stage38_positive_challenger_width = 2
stage38_active_pool_width = 8
stage38_positive_margin = 0.001
stage38_negative_tolerance = 0.0001
stage38_advantage_scale = 0.002
stage38_advantage_clip = 2.0
stage38_mature_positive_multiplier = 0.25
stage38_bc_weight = 0.1
stage38_kl_weight = 0.1
stage38_safety_kl_weight = 0.5
stage38_step_discount = 0.6
stage38_optimizer_steps_per_epoch = 48
stage38_gradient_accumulation = 8
stage38_global_bucket_composition = [2,30,8,24]
```

实现时增加独立的 target/objective、trace、loss bridge 和 contract；不得修改
Stage36/37 既有语义。Hydra YAML 必须声明全部 Stage38 key，禁止依赖 `+key`
绕过 struct 检查。

### 5.2 Artifact 与脚本约定

- 训练产物：
  `artifacts/grpo_stage38/pilot/training/ESCR/fold{0,1}/{audit,formal}/`
- 评测产物：
  `artifacts/grpo_stage38/pilot/eval/fold{0,1}/`
- 汇总：
  `artifacts/grpo_stage38/pilot/summary.json`
- 选择结果仅在 formal gate 通过时生成：
  `artifacts/grpo_stage38/pilot/selection.json`
- H100 wrapper 必须：
  - 自行导出 NAVSIM 所需环境变量；
  - 使用仓库绝对路径验证所有输入；
  - 先运行 `--cfg job --resolve` preflight；
  - audit 通过后才启动 formal；
  - 拒绝覆盖已有 experiment/artifact；
  - 记录 checkpoint、plan、manifest、selector、calibration 的 SHA256。

实现阶段必须先按仓库 `AGENTS.md` 自动关闭并确认释放 `gpu-occupy`，然后才能
启动任何本机 GPU audit、训练或评测；训练结束后不得自动重启占卡程序。

## 6. 执行阶段与硬性门控

### Phase 0：离线 target/replay 验证

在已有 Stage36/37 held-out candidate JSON 上计算 ESCR target，不训练模型。
必须同时通过：

- 所有 target finite，public/current provenance 完整；
- public=current 时 `delta_set`、advantage、union gain bitwise 为 0；
- synthetic elite improvement/elite regression/non-elite entry/non-entry 四种情况
  的符号完全正确；
- safety regression 始终得到 `-2`；
- positive non-elite fraction 在 `[0.02, 0.30]`；
- negative elite fraction 在 `[0.01, 0.40]`；
- mode permutation（无 reward ties 时）不改变 set utility。

失败时只允许修复公式或实现错误；不得用 held-out selected score调 margin/scale。

### Phase 1：一 optimizer-step audit

先在 fold0 运行一个真实 optimizer step。必须通过：

- 精确 `G=8/M=20/K=5/active<=8`；
- current/public 使用相同 anchor 与 noise pairing；
- rewards、components、target、public policy 均无梯度；
- decoder active gradients 非零且 finite；
- perception、backbone、classification、Stage25、Stage37 JFI 梯度严格为 0；
- positive challenger 与 negative elite 信号均非零；
- DDP 八 rank 的 global bucket composition 和 accumulation 均一致；
- optimizer step 后 public reference 输出 bitwise 不变。

audit 未通过时 formal 不得启动。

### Phase 2：fold0 kill-test

从 fold0 的 Stage36 RGT192 开始，训练新增 96 optimizer steps，在
step `24/48/96` 保存 checkpoint。用 fresh namespaces 同时评测：

- Public；
- frozen Stage36 RGT192；
- ESCR24、ESCR48、ESCR96。

至少一个 ESCR checkpoint 必须满足：

- selected delta vs Public `>= +0.0015`；
- selected delta vs RGT192 `>= -0.00025`；
- 两个 namespace 的 selected delta 均严格为正；
- public-top-5 delta `>= -0.0005`；
- public-top-5 相比 RGT192 改善 `>= +0.0005`；
- union safe-oracle delta `>= +0.001`；
- mature delta `>= -0.0002`；
- collision/drivable/TTC 各自 delta `>= -0.0005`；
- catastrophic delta `<= -0.5` 的数量为 0。

没有 checkpoint 通过即停止 Stage38，不运行 fold1。

### Phase 3：两折 OOF formal

fold0 kill-test 通过后，以同样设置训练 fold1，并对两折、两个 namespaces 做
严格配对汇总。checkpoint 先按下列“安全正结果 gate”筛选：

- pooled selected delta vs Public `>= +0.002`；
- whole-log bootstrap 95% CI 下界 `> 0`；
- 两个 fold mean 与两个 namespace mean 全部 `> 0`；
- pooled selected delta vs Stage36 RGT192 `>= 0`；
- public-top-5 delta `>= 0`；
- candidate mean、current raw oracle、union top-5 utility 均非负；
- union safe-oracle delta `>= +0.002`；
- hard-scene selected gain `>= +0.008`；
- mature-scene selected gain `>= -0.0001`；
- collision/drivable/TTC 各自 `>= -0.0005`；
- trimmed-10% mean `>= 0`，wins `>` losses，catastrophic count 为 0；
- 所有 artifact finite、provenance 和 SHA 完整。

通过者按以下顺序唯一选择：

1. selected delta 最大；
2. 若差值小于 `1e-4`，union safe-oracle 更大者优先；
3. 仍并列时选择更早的 optimizer step。

### Phase 4：结果分流

只允许三种结论：

1. **Mainline pass**：安全正结果 gate 全过，且 union safe-oracle
   `>= +0.005`。冻结 checkpoint；下一独立 Stage 才能训练 selector，并做
   generator × selector factorial。
2. **Positive ablation**：安全正结果 gate 全过，但 union safe-oracle
   `< +0.005`。允许用 frozen Stage25 selector 做一次锁定 NavTest，作为
   “GRPO 可稳定提升但 ceiling 不足”的消融；禁止 selector 调参，并触发
   Challenger 后备方案。
3. **Fail**：安全正结果 gate 未全过。不得 NavTest、不得 selector、不得延长
   或扫参；保存负结果并触发 Challenger 后备方案。

## 7. 明确禁止项

- 不从 Stage37 checkpoint 继续；
- 不同时训练 ESCR 与 Challenger；
- 不改 20 anchors、noise schedule、LR、KL、margin、scale 或 bucket ratio；
- 不用 old namespaces、fold0 或 NavTest 做 post-hoc 选参；
- generator gate 前不训练/校准 selector；
- 不删除 public-top-5、mature、component 或 CI gate；
- 不因“趋势还在上升”而超过 96 个新增 optimizer steps；
- 未通过 gate 的 checkpoint 不得改名为 champion。

## 8. 论文归因与创新边界

Stage38 保持 DiffusionDrive 架构、candidate count、public reference 和 frozen
selector 不变，因此 `ESCR - Stage36` 只归因于 GRPO credit/repair 目标。论文
主张应限定为：

> Public-anchored elite-set substitution credit prevents relative-advantage
> false positives and repairs strong-mode forgetting during diffusion-policy
> GRPO.

不能声称“首次提出 set-level GRPO”。SGRPO 等工作已经使用 set-level
leave-one-out contribution；Stage38 的区别是：

- frozen public driving elite set，而非生成集合内部 diversity；
- same-anchor trajectory substitution，而非跨 intention 比较；
- safety-vetoed absolute PDM marginal credit；
- 在共享 diffusion decoder 中同时处理 elite retention 与 new-mode entry。

Stage38 也不使用 DiffusionDriveV2 的 scale-adaptive multiplicative noise、
intra-anchor GRPO + inter-anchor truncated GRPO 组合。后续论文必须把这些方法列为
related work，并用 Stage36 与 ESCR 的同初始化/同数据/同 selector 对照支持归因。

