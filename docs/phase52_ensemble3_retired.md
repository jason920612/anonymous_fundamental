# Phase 52 — 3-model rank-average ensemble (RETIRED)

## Hypothesis

Bagging across heterogeneous models (LambdaRank + Ridge + Diffusion)
should reduce idiosyncratic prediction noise via the standard
bias-variance averaging argument. With cross-sectional rank averaging
inside each quarter, the ensemble should be a strictly better signal
than any single model.

## Result

| Variant | Sharpe | MDD | Calmar |
|---|---|---|---|
| ensemble3_rmt | 0.917 | -0.364 | 0.620 |
| ensemble3_rmt + dd_target | 0.799 | **-0.249** | 0.566 |
| (reference: rmt_minvar+dd_target) | **1.084** | -0.286 | **0.801** |

**Verdict: retired.** ensemble3+dd_target *did* achieve the lowest
MDD recorded in any Phase 38-52 variant (-0.249), but at a Sharpe
cost too large to be useful (0.799 vs 1.084). Calmar 0.566 << 0.801.

## Why it failed

The three models do not have equal predictive quality on this universe:

- LambdaRank: Sharpe 1.02 (best single model, pairwise ranking loss)
- Ridge + price features: Sharpe ≈ 0.76
- Diffusion v2: Sharpe ≈ 0.79

Equal-weight rank averaging pulls the ensemble signal halfway toward
the weaker models. The bias-variance argument assumes roughly equal
quality components; with this signal asymmetry, the variance benefit
is dominated by the bias hit.

A *weighted* ensemble (higher weight on LambdaRank) would presumably
do better — but choosing those weights from the val/test backtest
is test-set fitting. We do not attempt it.

## Discipline

- Equal weights, no tuning.
- Single-pass evaluation.
- The lowest-MDD observation (-0.249) is logged but **not selected**
  as production — selecting it would mean choosing a configuration
  that under-performs Calmar by 30%, which would be an admission that
  we are optimizing MDD in isolation rather than the joint
  Sharpe/MDD trade-off the goal directive asks for.
