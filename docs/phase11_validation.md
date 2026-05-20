# Phase 11 — Validation: 1000-CIK + Walk-Forward + Signal Diagnostics

This phase exists because the 50-CIK run (Phase 10) beat SPY/QQQ but **not** the
same-universe equal-weight or equal-risk baselines. That's the failure mode
RFC-09 §16 Case A warns about: the system's apparent alpha was driven by
universe selection (mega-cap tech 2022-2026) rather than by the model.

The Phase 11 design answers the question every model now has to answer:

> **Does the LightGBM signal beat the same universe under the same risk allocator and the same transaction costs?**

## What was built

### Resumable ingest + 404 fast-fail (`afp.data.sec_client`, `afp.data.ingest`)

- `PermanentSecError` raised on 4xx (excluding 429) so a missing CIK no longer
  burns 31s of exponential backoff per attempt.
- `run_sec_ingestion` now caches each CIK's submissions + companyfacts JSON to
  disk. Reruns parse from cache and only hit the network for new CIKs.
- A `_404_ciks.json` marker file is written so repeated runs skip known
  permanent failures.

These three changes turned a ~4.5-hour 1000-CIK run into ~45 minutes.

### Stronger baselines (`afp.backtest.benchmarks`)

| Function | Purpose |
|---|---|
| `dollar_volume_weighted_universe_returns` | Cap proxy when market-cap data is unavailable |
| `equal_weight_active_universe_returns` | Only names with at least one active prediction (apples-to-apples vs strategy) |
| `equal_risk_active_universe_returns` | Same, inverse-vol weighted — the strictest baseline |
| `random_positive_signal_returns` | N random picks from the active set + inverse-vol weighting |
| `shuffled_signal_returns` | Shuffle the model's signals across companies each day |

`build_report` now reports all of them automatically when `predictions` is passed.

### Signal diagnostics (`afp.models.diagnostics`)

| Function | Output |
|---|---|
| `decile_table` | Per-bucket mean predicted signal, mean realized log return, mean target, mean excess vs SPY |
| `top_minus_bottom_spread` | Top decile mean realized return minus bottom decile |
| `quarterly_rank_correlation` | Per-quarter Spearman of predicted_signal vs raw_log_return |
| `quarterly_direction_accuracy` | Per-quarter `sign(pred) == sign(realized)` |
| `positive_signal_vs_random` | Top-N by signal vs N random positives over `n_trials` Monte Carlo draws; reports model mean, random mean, spread, t-statistic |
| `build_diagnostic_report` | Bundles everything as a JSON-serializable dict |

CLI: `afp diagnostics --predictions <file>.parquet --split test`. Writes
`reports/diagnostics/<portfolio_id>/diagnostics.json`.

### Portfolio attribution (`afp.backtest.attribution`)

| Function | Output |
|---|---|
| `top_contributors(holdings, prices, n)` | Σ `weight_{d-1} * return_d` per name |
| `concentration(holdings)` | Per-day Herfindahl + top-5/top-10 weight share |
| `dollar_volume_exposure(holdings, prices)` | Weighted-average trailing dollar volume of held positions |
| `cash_and_turnover_summary(daily, tcost_bps)` | avg/max cash, annualized turnover, cost drag |

Wired into `afp backtest` — every run now writes `attribution.json` alongside
`metrics.csv`.

## Decision Rule (applied to walk-forward output)

```text
A. Strategy does not beat equal-risk-active after costs
   → STOP. Improve data, universe, target, or portfolio design first.
   → Do NOT tune LightGBM more, do NOT add autoencoder, do NOT add diffusion.

B. Strategy beats equal-risk-active in Sharpe but worse drawdown
   → Investigate risk concentration. Likely candidate: top contributors are
     a handful of mega-cap names.

C. Strategy beats equal-risk-active in Sharpe, max drawdown, AND rolling 12m
   excess return
   → Earned the right to try representation learning (autoencoder, transformer).

D. After autoencoder shows incremental alpha → consider diffusion.
```

Additional checks (must all hold to call the model useful):

- top decile realized return > bottom decile
- top decile realized return > universe average
- shuffled-signal portfolio underperforms strategy
- positive_signal_vs_random t-stat > 2 (signal beats random picks from the same active set)
- rolling 12-month excess vs equal-risk-active positive > 50% of the time
- top-name contribution not concentrated (Herfindahl <0.1 on average)

## Acceptance — verified by tests

- `tests/test_baselines_phase11.py` — all five new baselines produce finite
  series of the right length on a 6-name synthetic world.
- `tests/test_diagnostics.py` — decile table monotonic for a perfect signal,
  flat for noise; spread / Spearman / direction-accuracy behave as expected;
  `positive_signal_vs_random` discriminates signal from noise.
- `tests/test_attribution.py` — top-contributors ranked, Herfindahl ∈ (0,1],
  dollar-volume exposure produced, turnover/cost summary returns the right keys.
- `tests/test_sec_client.py` — 404 raises `PermanentSecError` after exactly one
  HTTP call; 429 retries; 200 returns body.

## How to reproduce

```bash
# 1. Snapshot v50 (already done in this run) — keeps the 50-CIK comparison
mv data/processed/*.parquet data/processed/v50/
mv artifacts/feature_encoder/v1 artifacts/v50/feature_encoder/v1
mv artifacts/predictions/lgbm_v001.parquet artifacts/v50/lgbm_v001.parquet
mv reports/backtest/lgbm_v001 reports/backtest/v50/lgbm_v001

# 2. Ingest 1000 CIKs (resumable; ~45 min cold, <2 min warm)
afp ingest-sec    --config configs/deployment_v1000.yaml --limit-ciks 1000
afp ingest-prices --config configs/deployment_v1000.yaml --limit 1000   # ~8-15 min

# 3. Build + train + walk-forward + backtest
afp build-dataset  --config configs/deployment_v1000.yaml
afp walk-forward   --config configs/deployment_v1000.yaml \
                   --model-kind lightgbm \
                   --frequency QS \
                   --out artifacts/predictions/walk_v1000.parquet
afp backtest       --config configs/deployment_v1000.yaml \
                   --predictions artifacts/predictions/walk_v1000.parquet \
                   --portfolio-id walk_v1000
afp diagnostics    --predictions artifacts/predictions/walk_v1000.parquet \
                   --portfolio-id walk_v1000

# Outputs:
reports/backtest/walk_v1000/{daily,holdings,trades}.parquet
reports/backtest/walk_v1000/metrics.csv          # all baselines side-by-side
reports/backtest/walk_v1000/attribution.json     # contributors + concentration + cash/turnover
reports/diagnostics/walk_v1000/diagnostics.json  # decile table, quarterly stats, random comparison
```

## Caveats Disclosed in Every Report

- **survivorship handling: imperfect / best effort.** The 1000-CIK universe is
  drawn from SEC's current `company_tickers.json`, so historically-delisted
  names are absent. Real survivorship-bias control needs the CRSP delisted file
  or equivalent.
- **price source: yfinance** — adequate for prototype; production needs
  source with verified split/dividend adjustments and historical ticker
  tracking.
- **test window length:** 1,100 trading days (2022-01 → 2026-05). Phase 12+
  should split into pre-COVID, COVID-era, and post-2022 sub-windows.
- **restatements:** we use `companyfacts.json` as-downloaded; original-as-filed
  reconstruction is a Phase 12 item.
