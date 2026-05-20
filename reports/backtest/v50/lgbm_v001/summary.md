# Live Deployment Run — lgbm_v001 (50-CIK universe)

**Date:** 2026-05-20  
**Owner:** jason  
**Purpose:** First production-style end-to-end run on real SEC + Yahoo Finance data.  
**Model:** LightGBM (4.6.0) on anonymous SEC fundamentals.

## Universe

- 50 CIKs from `data/processed/companies.parquet` (top of `company_tickers.json` — mega-cap US equities, heavily tech-tilted).
- 1 CIK (CIK0002070829) had no companyfacts → dropped after 5 retries.
- 49 effective companies with filings + facts.

## Data

| Layer | Count |
|---|---:|
| Companies ingested | 7,993 |
| CIKs processed | 50 |
| Filings (10-Q + 10-K, 1994–2026) | 4,287 |
| Financial facts (long-format) | 1,191,496 |
| Price rows (SEC universe + SPY + QQQ) | ~270K |
| Event samples built | 3,379 |
| Excluded samples | 908 (mostly `missing_next_event` for tail filings) |
| Anonymous feature IDs | 1,534 |
| Train rows (entry ≤ 2018-12-31) | 2,123 |
| Validation rows (2019–2021) | 691 |
| Test rows (entry ≥ 2022-01-01) | 565 |

## Model — LightGBM Validation Metrics

| Metric | Value |
|---|---:|
| MSE | 0.133 |
| MAE | 0.286 |
| Direction accuracy | 56.6% |
| Pearson | -0.005 |
| Spearman | -0.04 |
| R² | -0.21 |

Interpretation: direction accuracy modestly above 50% but rank correlation
essentially zero. R² < 0 means the model underperforms predicting the mean on
validation. This is typical for fundamental signals on a 50-name universe
where firm-specific factors dominate.

## Backtest (2022-01-03 → 2026-05-19, 1,100 trading days)

| Portfolio | Ann. Return | Ann. Vol | Sharpe | Sortino | Max DD | Calmar | Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Strategy (LGBM inverse-vol)** | 22.5% | 16.5% | **1.36** | 2.35 | -16.8% | 1.34 | 6.8× |
| SPY | 12.0% | 17.7% | 0.68 | 1.12 | -24.5% | 0.49 | — |
| QQQ | 14.6% | 23.2% | 0.63 | 1.04 | -34.8% | 0.42 | — |
| Equal-weight universe | **25.0%** | 18.0% | **1.39** | 2.35 | -23.5% | 1.06 | — |
| Equal-risk universe | 22.2% | 14.9% | **1.49** | 2.48 | -18.8% | 1.18 | — |

## RFC-07 §17 Minimum Strategy Approval

| Criterion | Result |
|---|---|
| Sharpe > equal-weight universe Sharpe | ✗ (1.36 vs 1.39) |
| Sharpe > equal-risk universe Sharpe | ✗ (1.36 vs 1.49) |
| Max DD < equal-weight universe MDD | ✓ (-16.8% vs -23.5%) |
| Net return > SPY | ✓ (+10.5 pp/yr) |
| Net return > QQQ | ✓ (+7.9 pp/yr) |
| Turnover not excessive | ✓ (6.8× annualized at 10 bps = -0.7% drag) |

**Verdict (RFC-09 §16):** Case A / Case B borderline. The strategy beats market
benchmarks (SPY, QQQ) but does **not** outperform the same-universe equal-weight
or equal-risk baselines. Most of the absolute alpha comes from concentration in
mega-cap US tech during 2022–2026, not from the model.

This is exactly the failure mode the equal-weight / equal-risk benchmarks are
designed to expose. Per the RFC-09 §16 decision tree, the next move should
**not** be to deploy or to add diffusion. It should be to either:

1. expand the universe to 500–1,000 names so the universe baseline becomes a
   meaningful market-cap-neutral reference, **or**
2. improve target definition / sample quality before adding model complexity.

## Leakage Checklist

- Future financial filing: **no** (point-in-time `accepted_datetime ≤ sample.accepted_datetime`)
- Future price: **no** (scale uses `daily_returns.loc[idx < entry_date]`)
- Future returns in scale: **no** (event-vol uses strictly prior events)
- Scaler fit on training data only: **yes**
- Hyperparameters tuned on test data: **no** (default LightGBM, no HP search)
- Point-in-time universe: **disclosed** (best-effort; survivor bias not eliminated for the 50-name pre-filtered slice)
- Restatements controlled: **disclosed** (using `companyfacts.json` as-of download; no original-as-filed reconstruction)

## Artifacts

```
data/processed/{companies,filings,financial_facts_long,prices_daily,benchmarks_daily,trading_calendar,event_samples,event_samples_excluded}.parquet
data/model_inputs/features_{train,validation,test}.parquet
artifacts/feature_encoder/v1/{anonymous_feature_map.parquet,scaling_stats.npz,metadata.json}
artifacts/predictions/lgbm_v001.parquet
reports/backtest/lgbm_v001/{daily,holdings,trades}.parquet
reports/backtest/lgbm_v001/metrics.csv
```

## Reproduce

```bash
afp ingest-sec    --config configs/deployment.yaml --limit-ciks 50
afp ingest-prices --config configs/deployment.yaml --limit 50
afp build-dataset --config configs/deployment.yaml
afp train         --config configs/deployment.yaml --model-kind lightgbm --model-id lgbm_v001
afp backtest      --config configs/deployment.yaml \
                  --predictions artifacts/predictions/lgbm_v001.parquet \
                  --portfolio-id lgbm_v001
```
