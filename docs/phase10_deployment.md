# Phase 10 — Production Training Deployment

Maps to **RFC-09 §13-§17** (operational maturity) and consolidates the runtime
glue that lets you go from "synthetic smoke run" to "actual research backtest
on live SEC + Yahoo Finance data".

## Components

| Layer | File | Purpose |
|---|---|---|
| Price source | `afp.data.price_client_yfinance.YFinancePriceClient` | yfinance fetch + parquet cache + offline fallback |
| CLI dispatch | `afp.cli.main:main` (entry: `afp`) | All commands callable as `afp <subcommand>` |
| Subcommands | `afp.cli.{ingest_sec,ingest_prices,build_dataset,train,backtest,walk_forward,run_pipeline}` | One module per command, individually testable |
| Walk-forward | `afp.cli.walk_forward` | Quarterly retrain + stitched OOS predictions (RFC-05 §9) |
| Image | `Dockerfile` | Python 3.12 slim + ruff + lightgbm + yfinance |
| Stack | `docker-compose.yml` | Sequenced services: ingest-sec → ingest-prices → build → train → backtest, plus opt-in walk-forward profile |
| CI | `.github/workflows/ci.yml` | Matrix lint + pytest + docker build |
| Scheduled run | `.github/workflows/scheduled-walk-forward.yml` | Weekly cron + manual dispatch, uploads predictions + reports |

## CLI Surface

```bash
afp ingest-sec    [--limit-ciks N]      # download companies/filings/facts
afp ingest-prices [--limit N]           # yfinance pull for ingested tickers + benchmarks
afp build-dataset                       # event_samples + ex-ante scale + tanh target + dense features
afp train        [--model-kind ridge|gbt|lightgbm|constant_zero|historical_mean]
afp backtest     --predictions <parquet> [--portfolio-id ID]
afp run-pipeline [--skip-ingest] [--limit-ciks N]  # ingest → build → train → backtest
afp walk-forward [--frequency QS] [--start ...] [--end ...]
```

All commands take `--config configs/default.yaml` by default and honor every
config knob (target k, periods_back, vol bounds, transaction cost bps, …).

## Walk-Forward Retraining (RFC-05 §9)

`afp walk-forward` partitions the OOS window into chunks (`--frequency QS` by
default = calendar quarter starts) and for each chunk:

1. Train on every sample with `entry_date < window_start`.
2. Predict on samples with `window_start ≤ entry_date < window_end`.
3. Stitch all OOS predictions into one parquet.

The encoder (`AnonymousFeatureEncoder`) is **not** re-fit per window — RFC-03
§13 forbids re-fitting scaler statistics on data that overlaps validation. Only
the supervised head re-trains.

Skipping windows that lack `--min-train-months` of history is logged as
`window_skipped_insufficient_train`.

## Docker

Build once:

```bash
docker build -t afp:latest .
```

Run the whole pipeline with bind-mounted volumes:

```bash
AFP_LIMIT_CIKS=100 AFP_MODEL_KIND=ridge docker compose up \
    afp-ingest-sec afp-ingest-prices afp-build afp-train afp-backtest
```

The walk-forward service is gated behind a `walk-forward` Compose profile:

```bash
docker compose --profile walk-forward up afp-walk-forward
```

Volumes:

- `./data` — persists raw and processed parquet output
- `./artifacts` — feature encoder, model predictions, experiment runs
- `./reports` — backtest summaries and CSVs

## SEC Politeness

`configs/data.yaml` ships with `User-Agent: anonymous-fundamental-portfolio research jasonya2206@gmail.com` — SEC requires an identifiable UA and rate-limits anonymous traffic. The `SecClient` throttles to 8 req/s by default and applies exponential backoff on 429/5xx (`retry_attempts=5`).

## yfinance Resilience

`YFinancePriceClient` caches each ticker's full history as `data/raw/prices_cache/<TICKER>.parquet`. On fetch failure it transparently falls back to the cache, so a transient Yahoo error never breaks the pipeline once you've done one successful pass.

## CI

`ci.yml`:

- Matrix: Python 3.10 / 3.11 / 3.12
- `ruff check src tests`
- `pytest -q --cov=afp`
- Docker build with GHA build cache

`scheduled-walk-forward.yml`:

- Weekly cron (Sat 03:00 UTC) and manual `workflow_dispatch` with model-kind / cik-limit inputs
- Runs the full pipeline end-to-end with `AFP_LIMIT_CIKS=200`
- Uploads `artifacts/predictions/` and `reports/backtest/` to the run as artifacts

## Ops Runbook

| Symptom | Diagnosis / Fix |
|---|---|
| `sec_get_retry` logs piling up | SEC rate limit. Lower `max_requests_per_second` in `configs/data.yaml` or rerun with `--limit-ciks` |
| `yfinance_empty` for a ticker | Symbol delisted / wrong ticker. Cross-check `data/processed/companies.parquet` |
| `missing_ex_ante_scale` for many samples | Insufficient price history or unusually short cycle between filings; check `data/processed/event_samples_excluded.parquet` |
| Backtest `cash_weight` always high | `min_positive_signal` too high or too few candidates; try `--config` with `min_positions: 10` |
| CI fails ruff on `src/afp/cli/_common.py` re-exports | They are intentional; `noqa: F401` is set per function |

## Tests Added in Phase 10

`tests/test_yfinance_client.py` — cache round-trip, column normalization, empty fetch  
`tests/test_cli.py` — parser recognises subcommands, end-to-end `build-dataset → train → backtest` on materialized synthetic data, walk-forward produces stitched predictions across multiple windows.

Total test count: **68** across 12 phase suites.
