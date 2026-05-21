# Phase 64 — Per-stock cross-disciplinary features (NEUTRAL)

## Hypothesis

Phase 62 added 6 MARKET-LEVEL cross-disciplinary features and won
big. The next natural step: add 6 PER-STOCK cross-disciplinary
features computed from each candidate's own return series:

| Feature | Source |
|---|---|
| `f_stock_hurst` | R/S analysis Hurst exponent (Mandelbrot) |
| `f_stock_skew` | 252d return skew |
| `f_stock_kurtosis` | 252d return excess kurtosis |
| `f_stock_levy_z` | z-score of >3σ event frequency vs Gaussian |
| `f_stock_lyapunov` | Mean log absolute return (chaos proxy) |
| `f_stock_dfa_alpha` | Detrended Fluctuation Analysis exponent |

These come from econophysics + chaos theory + long-range memory
literature. Causal computation; anonymous features.

## Result

| Variant | Sharpe | MDD | Calmar |
|---|---|---|---|
| Phase 54 (no cross) | 1.079 | -0.286 | 0.875 |
| Phase 62 (market cross only) | **1.128** | **-0.278** | **0.917** |
| Phase 64 (market + per-stock cross) | 1.132 | -0.279 | 0.914 |

**Verdict: NEUTRAL.** Phase 64 vs Phase 62 differ by ≤0.004
Sharpe — within measurement noise. The 6 per-stock features added
no detectable predictive value on top of Phase 62.

## Why per-stock features were redundant

The LambdaRank model already sees ~17000 anonymous fundamental
features PER STOCK PER SAMPLE. These encode the entire history of
financial-statement ratios, growth rates, valuation metrics, etc.

The new per-stock cross-disciplinary features (Hurst, DFA, etc.)
are functions of the recent price series — but the model also has
6 PRICE-FEATURES (trailing returns, vol) that capture the same
underlying signal in a complementary form. Adding Hurst and skew
on top is largely a non-linear reparameterization of what
trailing-return/vol features already encode.

**The lesson**: cross-disciplinary information helps when it's
ORTHOGONAL to what the model already has. Market-level features
(Phase 62) were new because the model had ZERO universe-wide
observables. Per-stock features (Phase 64) overlap with existing
rich per-stock representations.

## Feature importance check

Top 10 features by gain (Phase 64 trained model):

```
17677  203.94   price feature
17672  102.30   price feature
17674   80.47   price feature
17697   75.13   PER-STOCK CROSS (f_stock_skew or kurtosis)
17698   55.59   PER-STOCK CROSS
17700   54.06   PER-STOCK CROSS
17676   39.96   price feature
17684   36.90   MARKET CROSS (f_market_temp_z, Phase 62)
17675   35.91   price feature
17688   33.23   MARKET CROSS
```

Per-stock cross features DO get used (ranks 4-6 by gain), but the
backtest result is unchanged. This suggests the model finds them
informative for ranking but the resulting weight assignment is
nearly identical to Phase 62 — they correlate strongly with what
the price features already provide.

## Production stack

Phase 62 (market features only) remains the production
recommendation. Per-stock features are saved as an ablation
reference; they could be useful if the universe were larger or if
the fundamental encoder were stripped down.

## Discipline

- 6 per-stock features pre-declared from cross-disciplinary
  literature.
- Same LambdaRank hyperparameters as Phase 62.
- Single training run; single backtest.
- Honest "neutral" verdict; not chasing marginal Sharpe gains
  that fall within noise.
