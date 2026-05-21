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

# One command: ingest + features + train + save artifacts (~45 min cold,
# ~5-10 min on warm cache when only a few new filings have appeared)
afp refresh-all

# Then daily: just predict
afp predict TSLA
afp predict AAPL MSFT NVDA META TSLA --top-n 3
afp predict --tickers-file my_watchlist.txt
```

The single `afp refresh-all` command auto-runs all seven pipeline stages
(SEC ingest → yfinance ingest → build samples → cross-disciplinary
features → train Phase 62 LambdaRank → save booster + metadata →
invalidate prediction cache). It is idempotent — re-running on a fresh
day only re-fetches stale data and skips training if no new samples
appeared. See [docs/phase67_refresh_all.md](docs/phase67_refresh_all.md)
for flag reference.

### Daemon mode (24h watcher + Telegram bot)

```bash
afp daemon
```

On first launch the daemon interactively prompts for your Telegram bot
token and chat IDs (press Enter to skip), saves them to
`configs/telegram.json`, then enters its 24-hour cycle. Every cycle
fetches the current S&P 500 constituent list, detects new earnings
filings, retrains the Phase 62 model only when new data is present,
and pushes a Markdown digest (new filings + top long-side ideas) to
every subscribed chat.

For non-interactive deploys (systemd, Docker, CI) use `--no-prompt`
with either env vars (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_IDS`) or
a pre-populated `configs/telegram.json`. To re-prompt and overwrite
existing credentials use `--telegram-setup`.

Full daemon docs: [docs/phase68_daemon_telegram.md](docs/phase68_daemon_telegram.md).

If you prefer to run the stages individually:

```bash
afp ingest-sec    --config configs/deployment_v1000.yaml --limit-ciks 1000
afp ingest-prices --config configs/deployment_v1000.yaml --limit 1000
afp build-dataset --config configs/deployment_v1000.yaml
python3 scripts/train_lambdarank_cross_features.py   # Phase 62 training
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
