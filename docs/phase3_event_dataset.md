# Phase 3 — Event Dataset

Maps to **RFC-02** and **RFC-09 Milestone 3**.

## Goal

Produce one supervised sample per `(company, filing event with a known next filing)`,
with all entry/exit dates and prices resolved using only data available at the
time of the current event.

## Pipeline (`afp.features.sample_builder.build_event_samples`)

```
filings (10-Q,10-K)
   │ dedupe same-Eastern-date (RFC-02 §12) — prefer 10-K, then latest accepted
   ▼
kept filings, dropped filings → excluded[duplicate_event_same_date]
   │ group by internal_company_id, sort by event_date ascending
   ▼
for each consecutive pair (cur, nxt):
   entry_date = calendar.next_trading_day(cur.event_date)
   exit_date  = calendar.previous_trading_day(nxt.event_date)
   entry_price / exit_price ← adjusted_close lookup
   raw_log_return = log(exit / entry)
   raw_simple_return = exit/entry - 1
   spy_log_return, qqq_log_return ← benchmark lookup
   split = train / validation / test by entry_date
```

The last filing per company yields no sample (no `next`) but is recorded in the
excluded table with reason `missing_next_event`.

## Schema (matches RFC-02 §7.6)

`sample_id, internal_company_id, cik, filing_event_id, report_event_date,
accepted_datetime, form_type, period_end_date, entry_date, entry_price,
next_filing_event_id, next_report_event_date, exit_date, exit_price,
raw_log_return, raw_simple_return, ex_ante_scale, target_normalized_signal,
spy_log_return, qqq_log_return, eligible_for_training, eligible_for_backtest,
exclusion_reason, split, holding_days`

`ex_ante_scale` and `target_normalized_signal` are NaN at this stage — they are
populated by Phase 4. `holding_days` is the inclusive trading-day count between
entry and exit, used both by Phase 4 (scale) and the backtest (annualization).

## Exclusion Reasons

A bounded vocabulary (`afp.features.sample_builder.EXCLUSION_REASONS`):

```
missing_next_event           # last filing per company
missing_entry_price          # no adjusted_close on entry_date
missing_exit_price           # no adjusted_close on exit_date
entry_after_exit             # next event arrived too quickly
duplicate_event_same_date    # two filings on the same Eastern date
insufficient_price_history   # calendar lookup fell off the start/end
... etc.
```

Each excluded candidate is logged so the data quality report (Phase 9) can
quantify drop reasons.

## Eastern-Day Boundary

`accepted_datetime` is converted to an `America/New_York` calendar date via
`afp.utils.dates.to_eastern_date` before any calendar lookup. This is the
specific operationalization of RFC-00 §5.2:

> `report_event_date = SEC filing acceptance date converted to US/Eastern calendar date`

## Acceptance — verified by `tests/test_event_samples.py`

- All required columns present.
- `entry_date > report_event_date` strictly.
- `exit_date < next_report_event_date` strictly.
- `entry_date < exit_date` strictly.
- `raw_log_return ≈ log(exit_price / entry_price)`.
- Excluded table contains `missing_next_event` per company.
- `split` boundary respects `train_end_date` and `validation_end_date`.
