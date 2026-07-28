# Stage32 pilot OOF 结果（2026-07-26）

## 结论

两折、两种 noise namespace、每折 2,019 个 token 的 OOF 评测全部完成并通过
provenance 校验。结果不是“SCF 失败到低于 public”，而是：

1. DPEL control（Stage31 objective 延长到 12 epochs）稳定地提升了 public；
2. 修正归一化后的 SCF 也提升了 public，但没有超过 DPEL；
3. 因此当前不应把 SCF checkpoint 送入最终 NavTest，也不应把它宣称为
   selector-aware frontier 的额外收益。

完整机器可读汇总：

```text
artifacts/grpo_stage32/pilot/oof_summary.json
```

## Pooled paired selected-reward delta

delta 均为同一 fold、同一 noise、同一 token 相对 88.1 public 的
`selected_reward` 差值；SCF-DPEL 为额外的同 token 差值。

| checkpoint | DPEL vs public | SCF vs public | SCF vs DPEL |
| --- | ---: | ---: | ---: |
| step 192 | +0.001464 | +0.001161 | -0.000303 |
| step 384 | +0.001141 | +0.000925 | -0.000216 |
| step 576 | +0.001106 | +0.001016 | -0.000090 |

DPEL 的 95% whole-log bootstrap CI 下界分别为 `+0.000282`,
`+0.000156`、`+0.000035`；SCF 的下界分别为 `+0.000048`,
`+0.000113`、`+0.000102`。也就是说 SCF 相对 public 的小幅正增益是可见的，
但 SCF 相对 DPEL 在三个 step 都是负的，且 CI 都跨过 0。

SCF 的最佳 pooled selected delta 是 step 192（+0.001161），仍低于预先记录的
pilot gate（至少 +0.0015，且相对 DPEL 至少 +0.0005）。SCF 三个 step 的
hard-scene delta 分别为 +0.00413、+0.00182、+0.00253，也没有达到 +0.005。

## 解释

这轮实验把两个问题分离出来了：

- generator 只更新 64 个 decoder 参数、训练信号和归一化现在是健康的；
- DPEL 的“部署选择概率/风险约束”已经能把 generator 的小幅改善稳定转成
  selected trajectory；
- SCF 的 frontier 候选确实提高了候选侧信号（例如 candidate delta），但在
  最终 selector 上没有把这部分上限转成额外 selected gain，且相对 DPEL 有
  轻微损失。

所以这不是路径或训练崩溃，而是当前 frontier credit 的信用分配和 selector
转化仍不够有效。`pilot_screening` 在三个 step 均为 false，不能进入最终
四折 paper gate / NavTest。

## 当前保留的可靠 checkpoint

当前 Stage32 最可靠的开发候选是 DPEL control 的 step 192 或 384；step 576
仍为正但边际更小。最终选择要以四折 OOF 重新确认，不能仅凭这两折决定论文
checkpoint。

SCF 修正版 checkpoint 仍保留用于消融和方法分析：

```text
artifacts/grpo_stage32/pilot/training/SCF/fold0/formal/
artifacts/grpo_stage32/pilot/training/SCF/fold1/formal/
```
