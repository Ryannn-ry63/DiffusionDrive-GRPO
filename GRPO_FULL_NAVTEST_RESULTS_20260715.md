# GRPO paired full navtest evaluation

评测日期：2026-07-15
仓库分支：`grpo-selection-v2`

## 结论

Generation-GRPO D-128 在当前代码和严格配对的完整 navtest 上取得了小幅正提升：PDMS 从 `0.8491642572` 提升到 `0.8498702853`，差值 `+0.0007060281`。

这个提升的 paired bootstrap 95% 置信区间为 `[+0.0000438298, +0.0014030718]`，因此在本次固定 token 配对下方向为正。但增益明显低于 provisional `+0.005` engineering gate，不能描述为大幅提升。

收益主要集中在 base 失败场景：

- base score `< 0.5` 的 1,060 个 token：差值 `+0.0132967`
- base score bottom decile 的 1,387 个 token：差值 `+0.0107664`
- base score 为 0 的 994 个 token：D-128 平均 `0.0133384`

因此，当前最合理的判断是：短程 generation-GRPO 的正向 proxy 信号确实迁移到了 full navtest，但总体收益很小，主要改善困难/失败场景；是否值得进入 WE port，需要通过 WE smoke 和 paired baseline 验证，不能仅凭 `+0.0007` 宣称显著性能突破。

## 严格配对协议

Base 和 D-128 均使用：

- `train_test_split=navtest`
- 完全相同的 12,146 个 navtest token
- 完全相同的 metric cache：`/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/metric_cache`
- `ray_distributed`，32 CPU workers
- 相同 simulator、scorer 和 Hydra agent 配置
- diffusion roll timesteps：`[8, 0]`
- scheduler inference steps：`125`
- 按 token 的 SHA-256 派生确定性 diffusion noise
- 相同 reference checkpoint：base checkpoint

两份结果均为 `12,146/12,146` 成功，失败数为 0。两份 CSV 的 token 集完全相等。

标准 `run_pdm_score.py` 管线的 `score` 是最终选中轨迹的 PDMS；该管线不输出 20 个候选轨迹的 oracle score，因此本次没有 oracle 全量结果。

## Checkpoints

Base：

`/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model`

D-128：

`/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/grpo_stage2_D_gen_128/2026.07.15.05.10.31/lightning_logs/version_0/checkpoints/grpo-00-128.ckpt`

## 总体和分项结果

| 指标 | Base | D-128 | 差值 |
|---|---:|---:|---:|
| PDMS / selected score | 0.8491642572 | 0.8498702853 | **+0.0007060281** |
| no-at-fault collisions | 0.9819693726 | 0.9822163675 | +0.0002469949 |
| drivable area compliance | 0.9331467150 | 0.9338876997 | +0.0007409847 |
| ego progress | 0.7858142537 | 0.7866524946 | +0.0008382410 |
| time to collision within bound | 0.9468137658 | 0.9467314342 | -0.0000823316 |
| comfort | 0.9995883418 | 0.9995883418 | 0.0000000000 |
| driving direction compliance | 0.9786349415 | 0.9787172732 | +0.0000823316 |

## Paired统计

- mean paired score difference：`+0.0007060281`
- median paired difference：`+0.0000002082`
- bootstrap seed：`20260715`
- bootstrap samples：`10,000`
- bootstrap 95% CI：`[+0.0000438298, +0.0014030718]`
- D-128 wins：`6,100`
- ties：`4,655`
- losses：`1,391`

## 结果文件

原始逐 token CSV：

- Base：[2026.07.15.09.24.58.csv](/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/paired_full_navtest_base_20260715/2026.07.15.09.04.23/2026.07.15.09.24.58.csv)
- D-128：[2026.07.15.09.47.18.csv](/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/paired_full_navtest_d128_20260715/2026.07.15.09.26.47/2026.07.15.09.47.18.csv)

配对分析产物：

- [summary.json](/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/paired_full_navtest_comparison_20260715/summary.json)
- [paired_per_token.csv](/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/paired_full_navtest_comparison_20260715/paired_per_token.csv)

2-token smoke 结果也已保存：

- [base smoke CSV](/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/paired_navtest_smoke_base/2026.07.15.09.03.40/2026.07.15.09.03.52.csv)
- [D-128 smoke CSV](/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/paired_navtest_smoke_d128/2026.07.15.09.14.26/2026.07.15.09.14.42.csv)

## 历史结果说明

历史 base navtest PDMS `0.8487464443` 来自同一个 base checkpoint，但使用的是旧代码协议，未包含当前固定 schedule 和 token-derived deterministic noise。因此它只作为历史参考，不能替代本次 paired base。

本次 paired base 为 `0.8491642572`；D-128 相对它提升 `+0.0007060281`。

## Git 和 WE 状态

DiffusionDrive：

- branch：`grpo-selection-v2`
- 与远端同名分支一致
- 代码没有被本次评测修改
- 仅有交接文档 `GRPO_TO_WE_HANDOFF.md` 作为原有未跟踪文件

WorldEngine-Diffusion：

- branch：`main`
- commit：`aa64610` (`0525basemodel`)
- 原有未跟踪项 `DiffusionDrive-main/`、`meeting_note.md` 已保留
- 现有 baseline 文件未修改

## 下一步

按照交接文档，下一步可以进入 WE 的独立 GRPO port：新增 parallel 的 GRPO planning head 和 config，保留现有 baseline 不变；先完成 config/model construction、online-PDM reward smoke、gradient audit 和约 8 batch smoke，再考虑 128-update 实验。

当前结果不支持直接启动长 multi-GPU 训练，也不支持声称已经超过 `0.84` 很多；WE 阶段仍需以 paired WE baseline evaluation 为最终判断。
