# Stage32 Selector-aware Frontier GRPO

Date: 2026-07-26

Stage31 DPF-192 produced a reproducible positive OOF deployment delta, but the
full candidate frontier stayed flat. Stage32 therefore keeps the Stage31
selector-consistent objective as a long-training control (`DPEL`) and adds a
generator-only frontier-credit route (`SCF`). The selector, public 88.1
checkpoint, calibration, bucket sampler, and evaluation protocol remain frozen.

## Frozen pilot

- Start from the official public 88.1 checkpoint.
- Run branches `DPEL` and `SCF` on folds 0 and 1.
- Train 12 epochs; save/evaluate epochs 4, 8, and 12 (48 optimizer steps per
  epoch under the frozen Stage31 sampler).
- Use group size 8, full 20-mode banks, common-noise replay, FP32 reward and
  advantage arithmetic, 64 decoder tensors, and accumulation 8.
- Do not retrain or fine-tune the selector.

## SCF frontier credit

For every current/public bank, exclude fallback and deployed selected modes,
rank the remaining modes by frozen `delta_lcb`, and retain the top four modes
passing `risk_ucb <= min(1, risk_threshold + 0.10)` and the calibrated OOD
threshold. Strict Stage25 `eligible` is deliberately not required because its
positive challenger rate is too sparse.

The training trace replays selected plus the four-mode frontier pool under the
current policy and the public reference with common noise. Exact PDM rewards
are computed for all replayed chains. Stage31 selected-deployment credit is
unchanged. A frontier owner is the best valid member of each
selected-plus-pool group; a non-selected owner must exceed selected by at
least 0.001. The owner receives a frontier credit with coefficient 0.5,
capped at 0.25 of the selected credit in mature scenes. Unsafe or OOD modes
are masked from frontier credit, while selected safety overrides remain
unchanged. Public BC and exact KL are applied to all valid replayed chains.

## Pilot screening (not a paper claim)

Use one common checkpoint across both folds. SCF must satisfy all:

- selected pooled gain versus public >= 0.0015;
- paired SCF minus DPEL gain >= 0.0005 with whole-log CI lower bound > 0;
- hard-scene gain >= 0.005 and mature-scene gain >= -0.0002;
- candidate-max gain >= 0.0002 and not below DPEL;
- fallback-oracle gain >= 0.002;
- every pilot fold >= -0.0005, no catastrophes, and all provenance/frozen
  selector/finite-gradient checks pass.

If no common checkpoint passes, do not run full OOF. If a checkpoint passes,
freeze that epoch and run the unchanged four-fold paper gate before fold4 or
NavTest.

