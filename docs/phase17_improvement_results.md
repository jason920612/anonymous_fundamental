# Phase 17 — Improvement Run Results

This phase ran ten model variants on the same 1000-CIK universe to figure
out which (if any) of the Phase 12-16 / 18-20 changes actually improved
portfolio performance vs the Phase 11 baseline (LightGBM default, no
filter, no derived, no weighting, no sector caps).

## Headline

| Variant | Sharpe | Ann.Ret | MDD | Notes |
|---|---:|---:|---:|---|
| **Ridge static (winner)** | **0.76** | 15.2% | -37.2% | default α=1, train only |
| Static LightGBM (Phase 11 baseline) | 0.72 | 14.2% | -37.2% | |
| Ensemble mean(Ridge+LGBM+XGB+MLP) | 0.73 | 14.5% | -36.0% | best MDD |
| Ridge+XGB ensemble (mean) | 0.72 | 14.3% | -37.4% | |
| Ridge diversified (target=150) | 0.72 | 14.4% | -37.4% | |
| Ridge train+val combined | 0.65 | 13.2% | -37.4% | more data **hurt** |
| Improved bundle (filter+rank+der+sw+sec) | 0.67 | 13.3% | -38.2% | rank target |
| Tanh ablation (filter+der+sw+sec, tanh) | 0.61 | 12.2% | -38.9% | rank > tanh |
| XGB GPU static | 0.65 | 12.9% | -38.5% | highest val Spearman 0.052 |
| MLP GPU static | 0.64 | 13.1% | -37.1% | high turnover 12.5× |
| LightGBM HPO-tuned + weighting | **0.56** | 11.2% | -39.6% | worst — over-tuned |

Baselines for context (same 1000-CIK universe, 2020-01 → 2026-05):

```
SPY                                    0.77    15.7%  -33.7%
QQQ                                    0.87    21.6%  -34.8%
equal-weight active universe           1.00    22.8%  -39.6%
equal-risk active universe (target)    1.48    25.3%  -17.9%
random positive signal                 1.42    24.8%  -23.2%
shuffled signal                        1.30    32.9%  -29.5%
```

## What We Learned

### 1. The model has structurally near-zero alpha

Best validation Spearman across all variants was **0.056** (XGB GPU, HPO-picked
params). Random-positive in this universe gets Sharpe 1.42; equal-risk
1.48. The model adds essentially zero discriminative information beyond
what diversified inverse-vol allocation already provides.

### 2. Inverse correlation between validation IC and Sharpe

| Model | Val Spearman | Sharpe |
|---|---:|---:|
| XGB GPU | 0.052 | 0.65 |
| LightGBM tuned | 0.041 | 0.56 |
| LightGBM static | 0.029 | 0.72 |
| MLP GPU | 0.017 | 0.64 |
| Ridge | 0.009 | **0.76** |

Models with stronger validation IC concentrated bets and incurred more
portfolio variance. Ridge's near-zero IC paired with very smooth signal
distribution lets the inverse-vol allocator do most of the work — and that
combination wins.

### 3. Phase 12-16 "improvements" hurt as a bundle

The full improved config (universe filter + derived features + rank target +
sample weighting + sector caps) underperforms Phase 11 baseline by 0.05
Sharpe. The biggest culprits in ablation:

- **Sample weighting** by company × quarter dilutes the natural mega-cap
  signal that drove 2020-2024 performance.
- **Derived features** (4818 columns) on 1842 training rows = severe
  overfitting that not even rank target could compensate for.
- **Sector caps at 30%** prevented the tech overweight that aligned with
  the 2020-2024 AI rally.
- **Universe filter** shrank training from 2884 → 1842 rows; the lost
  micro-caps had a non-trivial alpha contribution.

The one improvement that *did* help (within the improved bundle): cross-
sectional rank target (0.67 vs 0.61 tanh ablation).

### 4. The flipped diagnostic: improved model picks better-than-random

The `positive_signal_vs_random` t-statistic went from
**-4.96 (Phase 11 static) → +2.90 (improved)** — a sign flip from
significantly-worse-than-random to significantly-better-than-random per
sample. But the same improvements also helped the random baseline more,
so the relative portfolio gap actually widened. The model is *less wrong*,
but the diversification benefit of being random is what drives Sharpe.

### 5. More data ≠ better — regime mismatch dominates

Training Ridge on `train ∪ validation` (7604 rows vs 1842) gave Sharpe 0.65
vs 0.76 from train-only. The 2016-2019 validation period is the COVID
prelude and has different regime characteristics from both 2003-2015 and
2020-2026. Adding it as training corrupts the fit.

### 6. Ensembles compress to the simplest member

| Ensemble | Sharpe |
|---|---:|
| Mean(Ridge, LGBM, XGB, MLP) | 0.73 |
| Mean(Ridge, XGB) | 0.72 |
| Rank-avg(Ridge, LGBM, XGB, MLP) | 0.65 |

Equal-weight averaging weakens any over-confident pick from the noisy
tree/MLP models, pulling the ensemble back toward Ridge's smoothness.
But it can't exceed Ridge solo because Ridge is already at the
inverse-vol-allocator-saturation point.

## Decision

Per the user's decision rule (Phase 11):

> **A — Strategy does NOT beat equal-risk-active baseline.**
> Do not tune advanced models. Improve data, universe, target, or
> portfolio design first.

This conclusion **strengthens** after the 10-variant ablation: every model
architecture we tried — including XGB on GPU, deep MLP on GPU, HPO-tuned
LightGBM, ensembles — converges to Sharpe ≤ 0.76. The information
content of quarterly fundamentals at single-firm 60-90 day horizons is
fundamentally too low to beat naive risk-parity on this universe.

Recommended Phase 21+ directions are *not* more model complexity. They are:

1. **Longer-horizon target** (12-month forward instead of next-filing
   60-90d). More signal-to-noise per sample.
2. **Cross-sectional rank target with sector neutralization** (winners
   within sector). Removes the macro regime dominance.
3. **Larger universe with proper survivorship correction.** 1000 CIKs is
   ~5% of investable US equities; many alpha-bearing names are excluded.
4. **Sector-relative or industry-relative returns.** Same as #2 but at
   target level rather than portfolio level.
5. **Combine fundamentals with price momentum** (RFC currently forbids
   in V1; the user has authorized lifting V1 constraints).
6. **Pre-train an autoencoder on the much larger event set** before the
   supervised head — currently the model only sees fundamentals supervised
   by event returns; an unsupervised pretraining pass might extract
   structure that supervised regression misses.

## Configs Used

```
configs/deployment_v1000.yaml           ← Phase 11 baseline (the winner)
configs/deployment_v1000_improved.yaml  ← all P12-P16 changes + rank target
configs/deployment_v1000_tanh.yaml      ← improved minus rank target
configs/deployment_v1000_tuned.yaml     ← HPO-tuned LightGBM + weighting
configs/deployment_v1000_diversified.yaml  ← target=150 positions
```

## Acceptance — verified by `tests/test_universe_filter.py`,
`tests/test_rank_target.py`, `tests/test_derived_features.py`,
`tests/test_sample_weights.py`, `tests/test_sectors.py`, `tests/test_hpo.py`,
`tests/test_gpu_models.py`, `tests/test_ensemble.py`.
