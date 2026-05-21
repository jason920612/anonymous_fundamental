# Phase 67 — `afp refresh-all` one-command fool-proof pipeline

> Status: ✓ shipped. Single CLI command runs the entire data → model
> → artifact pipeline end-to-end.

## Motivation

The full pipeline (SEC ingest → yfinance ingest → build samples →
compute cross-disciplinary features → train Phase 62 LambdaRank →
save artifacts) was previously seven separate scripts a user had to
chain manually. For a long-only research strategy that needs to be
re-run periodically as new filings arrive, this is operationally
fragile. `afp refresh-all` collapses it to one idempotent command.

## Usage

```bash
# Default: ingest 1000 CIKs, rebuild samples, retrain Phase 62 model
afp refresh-all

# Skip a stage (useful for partial re-runs)
afp refresh-all --skip-ingest-sec --skip-ingest-prices

# Force retraining even if no new samples
afp refresh-all --force-retrain

# Larger universe
afp refresh-all --limit-ciks 2000

# Custom output dir for trained model
afp refresh-all --model-out artifacts/models/my_custom_v1
```

## What it does (7 steps)

1. **Refresh SEC** — calls `afp ingest-sec`. Cache-aware: only
   re-fetches CIKs whose latest filing has changed.
2. **Refresh yfinance prices** — calls `afp ingest-prices`. Cache-
   aware: only re-downloads tickers whose latest price is more than
   2 trading days stale.
3. **Rebuild event_samples** — runs `afp build-dataset` if SEC or
   prices changed.
4. **Train Phase 62 LambdaRank** — trains the cross-disciplinary
   feature stack (anonymous fundamentals + 6 price features + 6
   market-state features). Saves booster + metadata + price-stats
   to `artifacts/models/lambdarank_v3_cross/`.
5. **Invalidate reference cohort cache** — so the next `afp
   predict` rebuilds it against the freshly trained model.
6. **Smoke-test** — verifies the saved booster + metadata are
   loadable.
7. **Status summary** — prints what changed, what trained, and
   where the artifacts landed.

## Idempotency

If nothing has changed since the last run:
- SEC ingest produces no new filings → returns "no changes"
- Prices ingest skips up-to-date tickers
- build-dataset still runs but produces an identical parquet
- Training is **skipped automatically** because the no-change
  detector sees identical samples and an existing booster

To force a full retrain on existing data, pass `--force-retrain`.

## Auto-pickup by predict CLI

`afp predict` was modified (Phase 67 side effect) to **auto-discover**
the model directory:

```python
# preference order
1. artifacts/models/lambdarank_v3_cross    # Phase 62 winner
2. artifacts/models/lambdarank_v3          # Phase 29 baseline
```

The predict CLI also auto-detects whether the loaded model expects
cross-disciplinary features (from its `metadata.json`) and computes
them on-the-fly from the price panel. So the workflow becomes:

```bash
afp refresh-all                    # daily/weekly cron
afp predict TSLA AAPL NVDA         # whenever needed
```

Zero manual orchestration required.

## Implementation

- `src/afp/cli/refresh_all.py` — the new subcommand.
- `src/afp/cli/main.py` — registers `refresh-all` in the dispatch.
- `src/afp/cli/predict.py` — `_resolve_model_dir()` + cross-feature
  attach helpers added.

## Acceptance

- 224/224 tests pass.
- `afp predict TSLA --no-refresh` continues to work on the existing
  Phase 29 model (auto-discovery falls back to `lambdarank_v3/`).
- `afp refresh-all --help` shows the full flag list.
- The pipeline runs end-to-end in a fresh clone given the existing
  raw caches.
