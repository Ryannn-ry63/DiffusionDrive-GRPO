# Stage20: PDMS-First Formal Training and Full Navtest

Date registered: 2026-07-21
Status: stopped by the preregistered dev-select transfer gate; protected sets unopened

## 1. Objective and attribution

Stage20 asks one primary question: does the Stage19 selected-anchor GRPO method
improve the final frozen-selector trajectory on full NAVSIM navtest?  PDMS is the
primary endpoint.  Safety components remain mandatory disclosures, but only a
statistically supported or material safety regression blocks navtest.

The locked cells are:

- A: frozen strong base, original `[8,0]` schedule;
- B: frozen strong base, full `[32,24,16,8,0]` schedule and reference selector;
- C: selected-anchor GRPO, full schedule and the same reference selector.

`C-B` is the isolated GRPO contribution and `C-A` is the total deployed gain.
Stage17 fallback is excluded so the primary result remains attributable to GRPO.

## 2. Locked provenance and method

- base checkpoint SHA256:
  `59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d`;
- Stage19 development checkpoint: epoch 8 `grpo-07-536.ckpt`, SHA256
  `f0bc08a76d5c3aa27f2f19fd46dcf569406bdd41ff5d9af0078fa75e12986c4b`;
- development method: same-selected-anchor G=8, raw-PDMS group z-score,
  BC weight 0.1, step discount 0.6, LR 1e-6, FP32, AdamW, and only the shared
  regression/diffusion decoder trainable;
- full schedule: truncation 32, timesteps `[32,24,16,8,0]`, 125 scheduler steps;
- evaluation noise namespaces: default `-1` and `20260726`;
- formal training manifest: 6,119 tokens, ordered SHA256
  `c7edaae00ad527b2cfe3c4391794dc70798a7e5bd69faa24c4b603822c75fde0`;
- dev-select3072 ordered SHA256:
  `1d5e2c1ae12f8e5b65edfa9976166b7a24cb77df87fe67b154bff9431a491299`;
- dev-confirm1024 ordered SHA256:
  `c98144d252415f74ccdf041a6f610b9a9bedee6adbba91027a7bb48a2df5554d`.

The two collision degradations already seen on the consumed Stage19 internal
test are diagnostic only.  They may not be used for training, threshold tuning,
epoch selection, or sample reweighting.

## 3. Gates and execution

### Development stress

Evaluate locked development C and A/B on dev-select3072 under both namespaces.
Continue only if every namespace has `C-B>0`, the two-noise mean `C-B>=+0.008`,
the log-cluster bootstrap 95% lower bound is above zero, and mean `C-A>=+0.010`.

A safety component blocks only if its pooled mean is below `-0.002` or its
log-cluster bootstrap 95% upper bound is below zero.  The one-sided 95%
Clopper-Pearson upper bound for `C-B<=-0.5` must not exceed 0.005.

### Formal seed0

If the stress gate passes, train fresh from base on all 6,119 navtrain tokens
for exactly eight epochs.  Use eight GPUs, per-device batch one, accumulation
eight, seed zero, and the locked Stage19 recipe.  No intermediate checkpoint is
evaluated or selected.  The final checkpoint must have exactly the permitted
decoder changes, immutable classification/perception/reference weights, and
finite active metrics and gradients.

### Independent confirmation

Open dev-confirm1024 exactly once for the formal checkpoint under both noise
namespaces.  Continue only if every namespace has `C-B>0`, the two-noise mean
`C-B>=+0.005`, its log-cluster CI lower bound is above zero, mean
`C-A>=+0.010`, and the same material/significant safety guards pass.  Results
cannot change the method, epoch, LR, reward, or schedule.

### Full navtest

After paired smoke tests, run A/B/C on all 12,146 navtest tokens through the
standard `run_pdm_score.py` pipeline with identical caches, simulator/scorer,
token-derived noise, and worker configuration.  Report absolute PDMS, `C-B`,
`C-A`, paired and log-cluster intervals, wins/ties/losses, all score components,
worst/bottom-1% behavior, and catastrophic counts.

The result labels are: statistically positive feasibility when `C-B>0` and its
CI lower bound exceeds zero; useful gain at `C-B>=+0.005`; paper target at
`C-B>=+0.010`.  A failed or inconclusive navtest is consumed and cannot be used
for post-hoc Stage20 tuning.

## 4. Implementation and tests

Stage20 adds evaluation, merge/gate, audit, and full-navtest wrappers only; it
does not change model inference or training semantics.  Tests must cover SHA,
token order/count, cache/schedule/selector/noise provenance, log-cluster
bootstrap, safety and catastrophic guards, shard order invariance, and legacy
Stage13--19 compatibility.  Full navtest requires successful 2-token and
64-token A/B/C smoke runs first.

If seed0 achieves at least +0.005 on full navtest, seeds 1/2 will later be
trained for the same locked eight epochs on an external eight-GPU system.

## 5. Execution record

- 2026-07-21: plan registered before Stage20 code changes, dev-select evaluation,
  formal training, dev-confirm opening, or navtest opening.
- 2026-07-21: implemented the fail-closed A/B/C two-noise gate with checkpoint,
  schedule, selector, cache, token-order, and noise provenance checks; added
  log-cluster bootstrap and material/significant safety guards.  The focused
  Stage16--20 regression suite passed 35 tests.
- 2026-07-21: completed all six dev-select3072 cells (A/B/C under default and
  namespace 20260726) as eight deterministic shards per cell, with 3,072/3,072
  tokens and zero failures in every merged artifact.
- 2026-07-21: the one-shot gate failed.  C-B was positive but negligible in
  both namespaces (`+0.0005962` and `+0.0001655`); the two-noise mean was
  `+0.0003808` with log-cluster 95% CI `[-0.0031387,+0.0038950]`.  C-A was
  `+0.0098490`.  Collision changed by `-0.0033366` with CI entirely below
  zero, TTC by `-0.0066732` with CI entirely below zero, and 55/6,144 pairs
  had C-B at most -0.5 (one-sided 95% upper rate `0.01119`).
- 2026-07-21: per the registered stop rule, formal eight-epoch training was not
  started and dev-confirm1024 and navtest remain unopened.  Stage20 establishes
  that the Stage19 navtrain internal gain does not transfer to the independent
  validation-log distribution at a useful or statistically confirmed level.
