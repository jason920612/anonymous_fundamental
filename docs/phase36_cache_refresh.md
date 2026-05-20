# Phase 36 — Cache Freshness Auto-Refresh

The predict tool now automatically checks if the on-disk SEC and yfinance
caches are stale and refreshes them when a new filing or new trading day
becomes available. Eliminates the "predict using stale data" failure.

## What gets checked

| Source | Cache file | Freshness rule |
|---|---|---|
| SEC submissions | `data/raw/sec/submissions/CIK{cik}.json` | Always re-fetched (small, <100 KB); compared against cached payload by `acceptanceDateTime` |
| SEC companyfacts | `data/raw/sec/companyfacts/CIK{cik}.json` | Refreshed only when submissions reveal a new filing |
| yfinance prices | `data/raw/prices_cache/{TICKER}.parquet` | Refreshed if the cache's max date is older than `today - 2 trading days` (default) |

`afp predict TSLA` runs the freshness check automatically. Use
`afp predict TSLA --no-refresh` to skip it (~5s faster but may use stale data).

## How it works (`afp.data.cache_refresh.refresh_sec_for_cik`)

1. Fetch the submissions JSON (small, always cheap).
2. Compare its `max(acceptanceDateTime)` to the cache's.
3. If they differ → write the new submissions JSON, then re-fetch
   companyfacts (the expensive call) and write that too.
4. If they match → reuse cache. companyfacts is loaded from disk
   unmodified.
5. CIKs that returned 404 (no companyfacts at all) are remembered in
   `data/raw/sec/companyfacts/_404_ciks.json` and skipped on future runs.

## Price refresh (`refresh_prices_for_ticker`)

1. Read the cache's max date.
2. If max date < `today - max_age_trading_days - 4` (4-day weekend grace),
   call yfinance to re-download the full ticker history.
3. Otherwise reuse the cache, sliced to the requested date window.

`max_age_trading_days=2` by default — handles weekends and single-day
holidays without unnecessary network calls.

## Network cost summary

| Scenario | Per-ticker requests |
|---|---|
| Cold cache | 1× submissions + N× pagination + 1× companyfacts + 1× yfinance |
| Warm cache, no new filing, prices up-to-date | 1× submissions only |
| Warm cache, new filing detected | 1× submissions + 1× companyfacts |
| `--no-refresh` mode | 0 (uses whatever's on disk) |

For a 5-ticker watchlist, the typical "no new filings" run hits SEC
exactly 5 times (one submissions call each) and yfinance 0 times.

## Logged events

- `sec_submissions_refreshed` — when the submissions cache was rewritten
- `sec_facts_404_marked` — when a CIK has no companyfacts (added to skip list)
- `prices_refreshed` — when the yfinance cache was rewritten
- `predict_cache_refreshed` — aggregate counter at the end of a predict run

## Acceptance — `tests/test_cache_refresh.py`

1. `_latest_accepted` returns the lexicographic max of the
   `acceptanceDateTime` list.
2. With a stale submissions cache and a fresh new acceptance time, the
   refresh function rewrites both submissions and companyfacts.
3. With a matching cache, neither submissions nor companyfacts is
   re-downloaded.
4. With a fresh prices parquet (max date within 2 trading days), yfinance
   is NOT called.
5. With a stale prices parquet (max date 30 days ago), yfinance IS called.
