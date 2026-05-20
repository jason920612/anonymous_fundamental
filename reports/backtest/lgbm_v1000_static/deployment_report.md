# Phase 11 Deployment Report — 1000-CIK Validation Run

**Date:** 2026-05-20  
**Pipeline:** SEC ingest (1000 CIKs) → yfinance prices (997 tickers) → event samples → anonymous features → LightGBM static-train → backtest + diagnostics + attribution  
**Walk-forward:** queued separately (see `walk_v1000/` once complete)

## Result in one sentence

> **The 1000-CIK LightGBM strategy underperforms every benchmark including SPY, and is statistically worse than picking random positive names (t≈-4.96). Decision Rule A applies: STOP model tuning; fix data/universe/target first.**

## Data Summary

| Stage | Count | Notes |
|---|---:|---|
| CIKs fetched | 1,000 | 50 returned 404 on companyfacts (fast-fail saved ~25 min vs Phase 10) |
| Companies with usable data | 950 | |
| Filings (10-Q + 10-K, 1994-2026) | 27,749 | resumable cache + pagination |
| Financial facts (long format) | 16,536,909 | 456 MB parquet |
| yfinance tickers cached | 999 | 3 fetch failures (delisted) |
| Price rows | 4,643,470 | 2003-01-02 → 2026-05-19 |
| Event samples built | 26,724 | one per (company, 10-Q/10-K with next filing) |
| Event samples excluded | 1,025 | mostly `missing_next_event` for tail filings |
| Anonymous feature IDs | 1,103 | min_company_count=50, periods_back=8 |
| Train rows (entry ≤ 2015-12-31) | 2,884 | heavily skewed: many CIKs post-2015 |
| Validation rows (2016-2019) | 6,657 | |
| Test rows (entry ≥ 2020-01-01) | 16,955 | |

## Validation Metrics — LightGBM (static train)

| Metric | Value | vs Phase 10 (50-CIK) |
|---|---:|---|
| MSE | 0.141 | up from 0.133 |
| MAE | 0.306 | up from 0.286 |
| Direction accuracy | **50.4%** | **down from 56.6%** |
| Spearman | 0.029 | up from -0.040 |
| Pearson | 0.014 | up from -0.005 |
| R² | -0.30 | down from -0.21 |

The 50-CIK direction accuracy of 56.6% was small-sample noise: on the
properly sized universe (1000 CIKs, 6,657 validation rows), the signal
is **indistinguishable from random**.

## Backtest 2020-01 → 2026-05 (1,603 trading days)

| Portfolio | Sharpe | Ann.Ret | Vol | MDD | Verdict |
|---|---:|---:|---:|---:|---|
| **Strategy (LightGBM)** | **0.72** | 14.2% | 19.6% | -37.2% | worst Sharpe of all |
| SPY | 0.77 | 15.7% | 20.4% | -33.7% | beats strategy |
| QQQ | 0.87 | 21.6% | 24.9% | -34.8% | |
| Equal-weight full universe | 1.06 | 21.3% | 20.1% | -34.8% | |
| **Equal-risk full universe** | **1.64** | 29.9% | 18.2% | -18.2% | best overall |
| Dollar-volume weighted | 1.19 | 27.1% | 22.8% | -34.1% | |
| Equal-weight active universe | 1.00 | 22.8% | 23.0% | -39.6% | |
| **Equal-risk active universe** | **1.48** | 25.3% | 17.1% | -17.9% | strictest apples-to-apples baseline |
| **Random positive signal** | **1.42** | 24.8% | 17.5% | -23.2% | beats strategy by 2× |
| **Shuffled signal** | **1.30** | 32.9% | 25.3% | -29.5% | shuffling helps |

The strategy is the **single worst Sharpe** in the entire table. Even
shuffling the model's own signals across companies improves Sharpe from
0.72 to 1.30.

## Signal Diagnostics (16,955 test samples)

### Decile table — no monotonic pattern
| Decile | mean pred | mean realized log ret |
|---|---:|---:|
| D1 (lowest) | -0.241 | **+0.0333** |
| D2 | -0.147 | +0.0305 |
| D5 | -0.031 | +0.0327 |
| D9 | +0.115 | **+0.0424** |
| D10 (highest) | +0.197 | +0.0283 |

**Top minus bottom: -0.50%** (negative — the model's "best" picks
underperform its "worst" picks). D9 is the best bucket, not D10.

### Time-series consistency
- Quarterly Spearman: mean **-0.008** | only **12/25 (48%)** quarters positive
- Quarterly direction accuracy: mean **49.6%** (worse than chance)

### Positive-signal vs random
- Model top-N realized return: 0.02903
- Random N positives realized return: 0.02903
- Spread: -0.000005 | **t-stat: -4.96**

A t-stat of -4.96 is *strong* statistical evidence the model is *worse*
than random — not just no-better.

## Portfolio Attribution

- avg cash weight: 0.5% | max 95.3% (early days before signal active)
- annualized turnover: **7.6×** → cost drag **1.51%/yr**
- mean Herfindahl: 0.012 | top-10 weight share: 20%
  → diversified portfolio is *not* concentrated; the model is just picking
    poorly across many names

## Comparison to Phase 10 (50-CIK)

| | Phase 10 (50-CIK) | Phase 11 (1000-CIK) |
|---|---|---|
| Universe size | 49 effective | 950 effective |
| Direction acc (validation) | 56.6% (n=691) | 50.4% (n=6,657) |
| Strategy Sharpe | 1.36 | 0.72 |
| Strategy MDD | -16.8% | -37.2% |
| Equal-risk-active Sharpe | n/a | 1.48 |
| Strategy beats SPY? | yes (+10pp/yr) | no (-1.5pp/yr) |
| Strategy beats equal-risk? | no (close: 1.36 vs 1.49) | no (decisive: 0.72 vs 1.48) |
| Strategy beats random positives? | n/a | **no, by t≈-5** |
| Strategy beats shuffled? | n/a | **no (0.72 vs 1.30)** |

Phase 10's apparent edge was concentration in mega-cap tech 2022-2026.
At fair scale (1000 names, full 2020 test window including COVID
drawdown), no edge survives.

## Decision per Phase 11 Rule

**A — STOP.**

Per the decision rule the user set:

> If Strategy does not beat equal-risk after costs:
> Do not tune advanced models. Improve data, universe, target, or portfolio design first.

Specifically:
- ❌ Sharpe vs equal-risk-active: 0.72 vs 1.48 (lose by 0.76)
- ❌ Sharpe vs SPY: 0.72 vs 0.77 (lose)
- ❌ Top-decile > bottom-decile: **no** (-0.50%)
- ❌ Beats random positives: **no**, t=-4.96
- ❌ Beats shuffled signal: **no**
- ❌ Quarterly Spearman > 0: 48% of quarters
- ✓ Concentration acceptable (HHI 0.012)
- ✓ Costs not the explanation (1.51%/yr drag; even at 0 bps, strategy loses)

**Do NOT** proceed to:
- LightGBM hyperparameter tuning
- Autoencoder
- Tabular transformer
- Conditional diffusion

## Recommended next steps (Phase 12)

The system architecture is sound — every gate caught what it was designed
to catch. The bottleneck is the **information content of the inputs**, not
the model. In priority order:

1. **Survivorship correction.** Pull CRSP delisted file or equivalent. The
   current universe excludes companies that disappeared, biasing all
   results.
2. **Target re-design.** event-to-event log return at the company level
   may be too noisy. Consider:
   - excess return vs sector or vs same-day market
   - longer-horizon target (12-month forward instead of next filing)
   - cross-sectional rank target instead of regression
3. **Universe filter.** Many of the 1000 CIKs are micro-cap / illiquid.
   Apply RFC-01 §6.1 filters at universe construction (min ADV $10M,
   min price $5, min 5yr history). Compare baselines on the *filtered*
   universe before comparing models.
4. **Sector neutralization.** Add sector metadata (FF12 or SIC-based)
   to the portfolio constraints. If raw equal-risk universe is so much
   better than the model, equal-risk *sector-neutral* could be the real
   benchmark to beat.
5. **Feature audit.** 1,103 anonymous features survived the
   min_company_count=50 filter. Many are likely sparse or redundant.
   Add coverage diagnostics and possibly aggressive feature selection
   before retraining.
6. **Decompose the test window.** Pre-COVID / COVID / post-COVID
   sub-windows would clarify whether the model has any signal in
   *some* regime, even if the average is bad.

## Caveats Disclosed

- survivorship handling: imperfect / best-effort
- price source: yfinance — research prototype quality
- test window: single 2020-01 → 2026-05 block; sub-window decomposition
  pending Phase 12
- restatements: as-of-download snapshots only

## Artifacts

```
data/processed/{companies,filings,financial_facts_long,
                prices_daily,benchmarks_daily,trading_calendar,
                event_samples,event_samples_excluded,manifest}.parquet
artifacts/feature_encoder/v1/{anonymous_feature_map.parquet,
                              scaling_stats.npz, metadata.json}
artifacts/predictions/lgbm_v1000_static.parquet
reports/backtest/lgbm_v1000_static/{daily,holdings,trades}.parquet
reports/backtest/lgbm_v1000_static/{metrics.csv,attribution.json,
                                    summary.md, deployment_report.md}
reports/diagnostics/lgbm_v1000_static/diagnostics.json

# Walk-forward outputs (when complete):
artifacts/predictions/walk_v1000.parquet
reports/backtest/walk_v1000/...
```
