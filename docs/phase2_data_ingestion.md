# Phase 2 — Data Ingestion

Maps to **RFC-01** and **RFC-09 Milestone 2**.

## Modules

| Module | Responsibility |
|---|---|
| `afp.data.identifiers` | CIK normalization (10-digit zero-padded); `internal_company_id = COMP<cik>` |
| `afp.data.trading_calendar` | NYSE-style calendar built deterministically from holiday rules. `next_trading_day` / `previous_trading_day` / `trading_days_between` |
| `afp.data.sec_client` | Throttled HTTP client over `data.sec.gov` with retry + exponential backoff. Enforces `User-Agent` per SEC policy |
| `afp.data.parse_companies` | Flattens `company_tickers.json` → RFC-01 §2.1 schema |
| `afp.data.parse_submissions` | Filters a CIK's `submissions.json` to 10-Q/10-K events (RFC-01 §2.3). Excludes amendments unless explicitly enabled |
| `afp.data.parse_companyfacts` | Flattens `companyfacts.json` into the long-format facts table (RFC-01 §2.2 / RFC-02 §7.3). Raw concept names preserved here only — Phase 5 anonymizes |
| `afp.data.price_client` | `PriceClient` protocol + `CsvPriceClient`. Returns the canonical OHLC+adjusted_close+volume schema |
| `afp.data.universe` | `point_in_time_universe(as_of)` — applies liquidity / price / history filters using only data ≤ `as_of` |
| `afp.data.ingest` | Orchestrators: `run_sec_ingestion`, `run_price_ingestion`, `build_calendar` |

## Trading Calendar Implementation

Holidays are computed from rules (no static table, no network):

- New Year's, Independence Day, Christmas — fixed dates with NYSE observed-day rule (Sat → Fri, Sun → Mon).
- MLK Day, Washington Birthday, Labor Day, Thanksgiving — n-th weekday of month.
- Memorial Day — last Monday of May.
- Good Friday — derived from the Gregorian Easter algorithm.
- Juneteenth — included only for years ≥ 2021.

This means the calendar is reproducible without a market-data dependency. The
test suite verifies New Year, Christmas, Juneteenth (post-2021), and the
weekend-skip + next/previous lookup invariants.

## Point-in-Time Universe Rules (RFC-01 §6.3)

`point_in_time_universe(prices, filings, as_of, cfg)` enforces:

1. `adjusted_close[as_of] >= min_adjusted_close_price_usd`
2. `len(price_history) >= min_years_price_history * 252`
3. `trailing_252d_dollar_volume.shift(1) >= min_avg_daily_dollar_volume_usd` (one-day shift = past-only)
4. number of accepted filings on or before `as_of` ≥ `min_financial_report_events`

Survivorship: only companies with prices ending at-or-near `as_of` are kept. We
explicitly **do not** filter to "still alive at backtest end" (RFC-01 §6.3 forbids that).

## Storage Layout

```
data/raw/sec/{company_tickers,submissions,companyfacts}/
data/raw/prices/{TICKER}.csv
data/processed/companies.parquet
data/processed/filings.parquet
data/processed/financial_facts_long.parquet
data/processed/prices_daily.parquet
data/processed/benchmarks_daily.parquet
data/processed/trading_calendar.parquet
data/processed/manifest.json
```

## Failure Handling (RFC-01 §10)

`run_sec_ingestion` catches per-CIK failures, logs `cik_failed`, and continues.
The manifest records what was actually written.
`validate_price_frame` returns warnings rather than raising so that partial price
coverage does not halt the pipeline.

## Acceptance — verified by `tests/test_data_ingestion.py`

- CIK normalization (`normalize_cik(320193) == "0000320193"`).
- `parse_company_tickers` dedupes by CIK, output is 10-char CIK.
- `parse_submissions` filters to allowed forms and uniquifies filing IDs.
- `parse_company_facts` produces non-null long rows with unique `fact_id`.
- Calendar correctly excludes weekends and US holidays (incl. Juneteenth gating).
- `next_trading_day(2020-07-02) == 2020-07-06` (Independence Day observed Friday).
- `point_in_time_universe` only changes if data ≤ as_of changes — verifies the
  past-only constraint structurally.
