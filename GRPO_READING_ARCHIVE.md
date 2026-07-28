# DiffusionDrive GRPO 阅读归档

本分支用于归档和阅读 DiffusionDrive GRPO Stage 0–36 的研究过程。

## 分支定位

- Markdown 文档是主要研究记录。
- Python、Shell、测试和配置文件作为相关源码快照保留。
- 本分支不保证能够直接运行或完整复现实验。
- 当前公共源码文件主要反映 Stage 36 完成后的工作树状态，不代表每个历史 Stage 当时的完整源码。

## 未包含的内容

为了控制 Git 仓库体积，本分支不保存以下本地生成内容：

- 数据集与缓存；
- `artifacts/` 中的逐场景评测结果、candidate bank、calibration collect 和中间输出；
- checkpoint 与模型权重；
- core dump、运行日志、缓存和临时备份。

各 Stage 的核心结论以对应的计划、runbook、audit 和结果 Markdown 为准。

## 建议阅读顺序

1. `GRPO_RESEARCH_ROUTE_OVERVIEW_CN.md`
2. 各 Stage 的计划与结果 Markdown
3. `scripts/training/` 下的训练入口
4. `scripts/evaluation/` 下的评测、gate、audit 和 summary 脚本
5. `navsim/agents/diffusiondrive/` 下的模型与 GRPO 实现
6. 顶层 `run_stage*.sh` 和 `test_grpo_stage*.py`
