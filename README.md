# anonymous-fundamental-portfolio

Event-driven US equity research system using anonymous raw financial statement
data, normalized directional movement prediction, and signal-weighted risk
parity portfolio construction.

The full specification lives in `rfc/`. Each phase has matching technical
documentation in `docs/`.

## Layout

```
src/afp/             implementation packages
  data/              SEC + price + universe ingestion
  features/          anonymous feature encoder, sample builder
  targets/           ex-ante volatility scale + tanh target
  models/            baseline + neural models
  portfolio/         signal restore + allocation
  backtest/          event-driven engine + metrics
  experiments/       config + registry + validation gates
  cli/               `afp` CLI subcommands
  utils/             logging + config loaders
configs/             YAML configs (data/features/target/model/portfolio/backtest)
tests/               pytest suite (leakage, accounting, anonymity gates, CLI)
docs/                phase-level technical docs (phase1..phase10)
data/                raw + processed + model_inputs + backtest outputs
artifacts/           feature encoders, model artifacts, experiment runs
reports/             backtest reports per portfolio
rfc/                 source specification (do not edit during implementation)
```

## Quickstart — `afp predict` (recommended)

The simplest way to use the system. Takes a US stock ticker (or several),
auto-fetches SEC fundamentals + yfinance prices, and emits a human-readable
prediction.

```bash
pip install -e .

# One-time setup: fetch base universe + train the LambdaRank model (~45 min)
afp ingest-sec    --config configs/deployment_v1000.yaml --limit-ciks 1000
afp ingest-prices --config configs/deployment_v1000.yaml --limit 1000
afp build-dataset --config configs/deployment_v1000.yaml
python3 scripts/save_lambdarank_model.py

# Then daily: just predict
afp predict TSLA
afp predict AAPL MSFT NVDA META TSLA --top-n 3
afp predict --tickers-file my_watchlist.txt
```

Example output:

```
=== TSLA ===
  Last filing            10-Q on 2026-04-22 (period ending 2026-03-31)
  Days since             28 days ago
  Next filing (ESTIMATE) ~2026-07-22  (63 days from today)
                         ↑ estimated from this company's median filing cadence — SEC has not confirmed yet

  Direction              ↑ Up
  Expected return        +7.27%  (over the estimated window)
  Expected vol           ±53.3% annualized  (daily ≈ ±3.35%)
  Rank vs peers          72nd percentile (within same-quarter reporting cohort)
  Confidence             moderate
```

Full documentation: [docs/phase35_predict_tool.md](docs/phase35_predict_tool.md).

## Smoke test (synthetic data, no network)

```bash
pytest -q                              # 131 tests across phases 1..35
python -m afp.experiments.smoke_run    # synthetic-data end-to-end pipeline
```

## Research / training pipeline (real SEC + Yahoo Finance)

```bash
afp ingest-sec    --limit-ciks 1000    # SEC companies + 10-Q/10-K filings + facts
afp ingest-prices --limit 1000         # yfinance daily prices + SPY/QQQ + trading calendar
afp build-dataset                       # event_samples + targets + dense features
afp train         --model-kind ridge   # or: gbt, lightgbm, constant_zero, historical_mean
afp backtest      --predictions artifacts/predictions/ridge_v001.parquet
afp walk-forward  --frequency QS       # quarterly retrain + stitched OOS predictions
afp run-pipeline  --skip-ingest        # one-shot orchestrator
```

Docker / CI deployment runbook: [docs/phase10_deployment.md](docs/phase10_deployment.md).

Full research log: [docs/README.md](docs/README.md) (36 phases, including the
LambdaRank breakthrough, tier-2 cross-universe holdout, and final methodology
corrections).

## Version 1 constraints (RFC-00)

- long only, no leverage, no shorting
- model output `∈ [-1, 1]`; portfolio weights computed **outside** the model
- anonymous feature IDs only — no human-readable concept names enter the model
- point-in-time data; no look-ahead, no restatement leakage
- adjusted close prices for all return calculations
- transaction costs always included in reported PnL
