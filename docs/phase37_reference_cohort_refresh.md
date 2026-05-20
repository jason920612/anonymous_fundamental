# Phase 37 — Reference Cohort Auto-Refresh

The reference cohort (~640 same-quarter peer scores) used to compute
percentile rankings now auto-updates when the underlying training data
advances to a new quarter, eliminating the "ranked vs stale Q1 pool"
failure when the user's prediction is actually for Q2.

## Why this matters

A single-ticker query like `afp predict TSLA` only has its own raw
LambdaRank score — meaningless on its own. The system ranks it against
a cached pool. If TSLA filed in Q2 2026 but the cache is still 2026Q1
peers, the comparison is across quarters that experienced different
market regimes — distorting the percentile.

## What the refresh checks

`_reference_cohort_scores(...)` runs on every `afp predict` and rebuilds
the cohort cache (`artifacts/models/lambdarank_v3/reference_cohort.parquet`)
when ANY of the following is true:

1. **`force_rebuild=True`** — set by `--refresh-cohort` flag.
2. **Quarter advanced**: the latest quarter in `event_samples.parquet`
   is different from the cache's stored quarter.
3. **Source updated**: the mtime of `event_samples.parquet` is newer
   than the cache file.

If none triggers, the cache is loaded as-is (fast).

## Two-tier staleness

| Tier | What's stale | Auto-handled? | How to refresh |
|---|---|---|---|
| Cohort cache (`reference_cohort.parquet`) | Score pool out of sync with `event_samples.parquet` | ✓ automatic on every predict run | nothing |
| Training universe (`event_samples.parquet` itself) | The 1000-CIK ingest hasn't been re-run; latest sample is months old | ✗ requires `afp ingest-sec` + `afp build-dataset` | manual re-ingest |

When tier-2 is stale, the predict tool emits a warning:

```
WARN  reference_cohort_underlying_universe_stale
      latest_sample_date=2026-01-15  days_since=125
      hint: re-run `afp ingest-sec --config configs/deployment_v1000.yaml --limit-ciks 1000`
            then `afp build-dataset` to refresh the training universe
```

The threshold is 120 days. The warning is informational — the predict
still runs, but the user knows the comparison is being made against an
aging peer cohort.

## CLI

```bash
# default — auto-refreshes cohort cache when stale, warns if universe stale
afp predict TSLA

# Force cohort rebuild even if cache appears fresh
afp predict TSLA --refresh-cohort

# Skip ALL cache freshness checks (cohort + SEC + price). Fast but stale-prone.
afp predict TSLA --no-refresh
```

## Acceptance — `tests/test_reference_cohort_freshness.py`

1. Cache quarter matches → no rebuild.
2. Quarter advances → rebuild triggered.
3. Samples-parquet mtime > cache mtime → rebuild triggered.
4. Latest sample older than 120 days → warning condition.

## Implementation note

The refresh is **synchronous** during a normal predict call. Cohort
rebuild takes ~5-15s for 600+ same-quarter peers (encoder transform
+ booster predict). For larger universes consider running it as a
nightly cron, but the on-demand path keeps the cohort current with
zero user effort.
