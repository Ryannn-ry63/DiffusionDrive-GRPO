# Stage35 OOF PDM Result (2026-07-28)

## Outcome

Both Stage35 H100 evaluation jobs completed and all 16 expected artifacts
passed provenance validation:

- 2 out-of-fold folds;
- 4 checkpoints: steps 48, 96, 144, and 192;
- 2 independent evaluation-noise namespaces.

Stage35 NCD-GRPO produces a statistically positive selected-trajectory gain
over the frozen public-88.1 checkpoint on this paired pilot. However, no
checkpoint passes the complete frozen promotion gate, so Stage35 must not yet
be promoted to navtest.

Machine-readable summary:

- `artifacts/grpo_stage35/pilot/summary.json`

## Checkpoint comparison

All deltas below are paired against the frozen public checkpoint.

| Step | Selected delta | 95% log bootstrap CI | Safe oracle delta | Candidate mean delta | Pass |
|---:|---:|---:|---:|---:|:---:|
| 48 | +0.001177 | [+0.000301, +0.002250] | +0.001181 | +0.001223 | No |
| 96 | +0.001238 | [+0.000289, +0.002385] | +0.001236 | +0.001224 | No |
| 144 | +0.001105 | [+0.000248, +0.002116] | +0.001109 | +0.001624 | No |
| 192 | **+0.001342** | **[+0.000399, +0.002521]** | **+0.001347** | **+0.001968** | No |

Step 192 is the best checkpoint by selected gain.

## Step-192 detail

Paired pilot absolute means:

| System | Selected PDM |
|---|---:|
| Public checkpoint | 0.846870 |
| Stage34 DPEL192 | 0.847742 |
| Stage35 NCD192 | **0.848213** |

These are paired two-fold pilot values, not the official 12,146-scenario
navtest score. The valid comparison is the paired delta.

Stage35 NCD192:

- versus public: `+0.001342` (about `+0.134` PDMS points);
- versus Stage34 DPEL192: `+0.000470` (about `+0.047` PDMS points);
- fold 0 / fold 1: `+0.000784` / `+0.001900`;
- noise 20261511 / 20261512: `+0.000913` / `+0.001772`;
- hard-scene delta: `+0.005226`;
- mature-scene delta: `+0.000271`;
- wins / losses / ties: `2912 / 965 / 203`;
- catastrophic regressions at or below `-0.5`: `0`;
- collision / drivable / TTC deltas:
  `0.000000 / +0.001225 / +0.001225`.

The gain over public is positive in both folds and both noise namespaces, and
its whole-log bootstrap confidence interval excludes zero.

## Why the frozen gate did not pass

For step 192:

1. The required public gain was `+0.001500`; observed gain was `+0.001342`.
   The miss is `0.000158`.
2. The required improvement over DPEL192 was `+0.000500`; observed improvement
   was `+0.000470`. The miss is `0.000030`.
3. Raw oracle delta was `-0.000167`, below the required nonnegative value.
4. Public-top-5 candidate delta was `-0.000906`, below the required
   nonnegative value.
5. The gain over DPEL192 was not statistically resolved:
   its 95% log-bootstrap interval was `[-0.000400, +0.001458]`.

All remaining step-192 safety, consistency, mature-scene, component, trimmed
mean, win/loss, and catastrophic-regression checks passed.

## Interpretation

Stage35 is not a null result. It is the clearest evidence on the public-88.1
baseline so far that the GRPO generator update can improve the final selected
trajectory consistently:

- selected PDM improves on both folds and both noise namespaces;
- hard scenes improve substantially while mature scenes remain positive;
- the improvement over public is statistically positive;
- Stage35 is numerically better than Stage34 DPEL.

The remaining limitation is specifically the generator ceiling. NCD improves
the average candidate and the safely deployable candidate set, but does not
improve the raw oracle or the public model's original top-5 modes. Thus the
current gain comes from reshaping selector-usable modes rather than creating a
clearly higher global trajectory ceiling.

## Decision

- Preserve NCD192 as the Stage35 best checkpoint.
- Do not run official navtest or claim a promoted Stage35 model.
- Do not discard the NCD objective: it produced a robust positive effect.
- The next experiment should target preservation/improvement of public top
  modes and raw-oracle headroom while retaining NCD's hard-scene and
  mature-scene behavior.

