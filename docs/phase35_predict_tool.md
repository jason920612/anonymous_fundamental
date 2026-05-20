# Phase 35 — User Predict Tool

A single CLI command that takes US stock tickers and returns the
LambdaRank model's prediction for each, with all data fetching handled
automatically.

## Quickstart

```bash
# Single ticker
afp predict AAPL

# Several tickers
afp predict AAPL MSFT NVDA GOOGL AMZN

# Watchlist file (one ticker per line, '#' for comments)
afp predict --tickers-file my_watchlist.txt

# Also suggest a long-only portfolio of the top-N positive picks
afp predict AAPL MSFT NVDA GOOGL AMZN META TSLA --top-n 3

# JSON output (for scripting)
afp predict AAPL MSFT --json
```

Example output:

```
Rank  Ticker  Signal     Pct   Exp.Ret  Form   Period       Entry
-----------------------------------------------------------------------
   1  META    +0.750    88%    +53.3%  10-K   2025-12-31   2026-01-29
   2  TSLA    +0.500    75%    +54.9%  10-K   2025-12-31   2026-01-29
   3  NVDA    +0.250    62%    +16.0%  10-Q   2025-10-26   2025-11-20
   4  AMZN    +0.000    50%     +0.0%  10-K   2025-12-31   2026-02-06
   5  AAPL    -0.250    38%     -9.1%  10-Q   2025-12-27   2026-02-02
   6  GOOGL   -0.500    25%    -16.9%  10-K   2025-12-31   2026-02-05
   7  MSFT    -0.750    12%    -22.1%  10-Q   2025-12-31   2026-01-29

Suggested long-only portfolio (top 3 positive signals, inverse-vol weighted):
  META    +0.750    43.3%    +53.3%
  TSLA    +0.500    23.9%    +54.9%
  NVDA    +0.250    32.8%    +16.0%
```

## What the tool does, step by step

1. **Resolve ticker → CIK** via SEC `company_tickers.json` (cached at
   `data/raw/sec/company_tickers/`).
2. **Fetch SEC submissions + companyfacts** for each CIK — uses the
   same on-disk cache as the rest of the pipeline. Already-cached CIKs
   are reused without re-downloading.
3. **Fetch daily prices** via yfinance with the same cache (`data/raw/prices_cache/`).
4. **Build event samples** with `build_event_samples` — picks the most
   recent valid 10-Q/10-K with a next filing reference and a tradeable
   entry date.
5. **Compute ex-ante vol scale** with the standard hybrid daily/event
   method. Samples without enough history are skipped with a notice.
6. **Encode anonymous features** through the saved
   `artifacts/feature_encoder/v1`.
7. **Compute anonymous price features** (trailing 21/63/126/252-day log
   returns, 1m–3m relative momentum, 63-day vol) and rescale with the
   training-period median/IQR stats stored at
   `artifacts/models/lambdarank_v3/price_stats.npz`.
8. **Run the saved LambdaRank booster** at
   `artifacts/models/lambdarank_v3/booster.txt`.
9. **Rank within the user's cohort**: convert raw LambdaRank scores to
   `2·pct_rank − 1 ∈ [-1, 1]` and apply `atanh × k × ex_ante_scale`
   to restore an expected log return per name.

## Output Columns

| Column | Meaning |
|---|---|
| Rank | Position in the user's input cohort, best first |
| Ticker | Input ticker symbol |
| Signal | Cohort-relative `[-1, 1]` rank score (NOT a probability) |
| Pct | Percentile within the cohort |
| Exp.Ret | Conservative expected pct return until next filing (heavily shrunk — see note below) |
| Form | Latest 10-Q or 10-K |
| Period | Period ending date of that filing |
| Entry | First trading day after the filing acceptance datetime (where the strategy buys) |

`--top-n N` additionally prints an inverse-vol weighted long-only
portfolio of the top N positive-signal names.

### Confidence label (5-tier, rank-only)

The `Confidence` field in single-ticker output is a direct read of the
ticker's percentile vs the reference cohort. It is **strictly one-sided**:
the labels reflect rank position, not a "high-confidence short"
interpretation, because the strategy is long-only.

| Percentile | Label | Meaning |
|---|---|---|
| ≥ 85 (top 15%) | `high` | strong long candidate |
| 70 – 85 (top 30% excl. top 15%) | `moderate` | plausible long |
| 30 – 70 (middle 40%) | `neutral` | no clear thesis |
| 15 – 30 (bottom 30% excl. bottom 15%) | `moderate-avoid` | weak fundamentals vs peers |
| < 15 (bottom 15%) | `avoid` | clear under-performer in cohort |

A bottom-bucket name (`avoid`) does NOT carry a short signal — it just
means "do not buy". Acting on it as a short would be outside the
calibrated regime of the model.

### Reference Cohort

A single ticker would trivially rank at the 50th percentile if compared
only against itself. So the tool builds a **reference cohort** from the
most recent quarter of the 1000-CIK training universe (cached at
`artifacts/models/lambdarank_v3/reference_cohort.parquet`) and ranks
the user's tickers against it. A solo `afp predict TSLA` now returns a
meaningful percentile vs ~1000 peer companies.

### Why the displayed `Exp.Ret` is shrunken

LambdaRank outputs a **rank score**, not a calibrated tanh-of-return.
If we passed the raw percentile through the original `atanh × k ×
ex_ante_scale` restoration, the 93rd percentile would imply a ~150-200%
quarterly return — clearly noise-amplified. We instead apply a hard
shrinkage (signal × 0.25, clipped to ±0.4) before restoration. The
`Exp.Ret` you see is a directional estimate scaled by the company's
own volatility, not a forecast. Use `Signal` and `Pct` as the
primary decision inputs.

`--json` switches output to machine-readable JSON.

## Caching

All fetched data is cached on disk and shared across runs:
- `data/raw/sec/submissions/CIK*.json`
- `data/raw/sec/companyfacts/CIK*.json` (or `_404_ciks.json` for known-missing CIKs)
- `data/raw/prices_cache/TICKER.parquet`

Running `afp predict AAPL` twice will hit the network only the first
time. Re-running a week later will refetch yfinance (to get new bars)
but the SEC JSONs are reused unless deleted.

## Model State

`artifacts/models/lambdarank_v3/` is created once by:

```bash
python3 scripts/save_lambdarank_model.py
```

This trains LambdaRank tuned (the v3 config from Phase 29) on the 1000-
CIK tier1 train pool and saves the LightGBM booster + the price-feature
scaler stats + metadata. The CLI loads from there.

If the directory is missing, `afp predict` prints a clear error
pointing at the training script.

## RFC Compliance

- The model still sees only anonymous feature IDs — the CLI passes the
  same encoded `[N, P, F]` tensor it always does.
- No information from outside the price + SEC fundamentals enters the
  prediction path.
- Cohort-relative ranking (the cohort = what the user typed) does not
  use future data — only the latest filing for each name available
  today.

## What This Tool Doesn't Do

- It does not place trades. Output is research signal, not an order.
- It does not enforce any portfolio risk limits beyond the optional
  `--top-n` inverse-vol blob; for real allocation use
  `afp backtest --predictions <parquet>` with a full config.
- It does not simulate transaction cost — the Exp.Ret column is the
  raw expected log return implied by the model.

## Reproducibility

The trained model checked into `artifacts/models/lambdarank_v3/` is
deterministic given:
- the LightGBM seed (default in `save_lambdarank_model.py`)
- the training data (`data/processed/event_samples.parquet`)
- the encoder fit state (`artifacts/feature_encoder/v1`)

Re-running `save_lambdarank_model.py` with the same inputs produces
the same booster bit-for-bit.
