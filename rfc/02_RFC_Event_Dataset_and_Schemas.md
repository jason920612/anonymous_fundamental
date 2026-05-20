# RFC-02: Event Dataset and Schemas

Status: Draft  
Phase: 2  
Purpose: Define the event-driven supervised learning dataset, database tables, row-level schemas, and point-in-time rules.

---

## 1. Objective

Create one supervised sample per company report event.

Each sample represents:

```text
company i
report event t
features available at event t
target measured from entry after event t to exit before event t+1
```

The sample is the central unit used by:

1. model training,
2. prediction,
3. portfolio construction,
4. event-level evaluation.

---

## 2. Sample Definition

For company `i` and report event `t`:

```text
sample_id = hash(internal_company_id, filing_event_id_t)
```

Required sample fields:

```text
sample_id
internal_company_id
cik
ticker_at_event
filing_event_id
report_event_date
accepted_datetime
form_type
period_end_date
entry_date
entry_price
next_filing_event_id
next_report_event_date
exit_date
exit_price
raw_log_return
raw_simple_return
ex_ante_scale
target_normalized_signal
benchmark_spy_return
benchmark_qqq_return
eligible_for_training
eligible_for_backtest
exclusion_reason
```

---

## 3. Event Date Rule

Version 1:

```text
report_event_date = date part of SEC accepted_datetime in US/Eastern
```

If SEC accepted time is after market close, Version 1 still uses:

```text
entry_date = first trading day after report_event_date
```

This conservative rule avoids using announcement-day jumps.

---

## 4. Entry Date Rule

```text
entry_date = next_trading_day(report_event_date)
```

Where:

```text
next_trading_day(d) = smallest trading calendar date greater than d
```

Examples:

| Report Event Date | Day | Entry Date |
|---|---:|---|
| 2020-05-05 | Tuesday | 2020-05-06 |
| 2020-05-08 | Friday | 2020-05-11 |
| 2020-07-03 | Market holiday | 2020-07-06 |

---

## 5. Exit Date Rule

Given next report event date:

```text
exit_date = previous_trading_day(next_report_event_date)
```

Where:

```text
previous_trading_day(d) = largest trading calendar date smaller than d
```

Examples:

| Next Report Event Date | Day | Exit Date |
|---|---:|---|
| 2020-08-05 | Wednesday | 2020-08-04 |
| 2020-08-10 | Monday | 2020-08-07 |
| 2020-11-27 | Market half day | previous trading day based on calendar |

---

## 6. Target Period Validity

A sample is valid only if:

```text
entry_date < exit_date
entry_price > 0
exit_price > 0
```

Exclude if:

```text
entry_date == exit_date
entry_date > exit_date
missing entry price
missing exit price
missing next report event
security not tradable at entry
```

---

## 7. Database Tables

### 7.1 `companies`

```sql
CREATE TABLE companies (
    internal_company_id TEXT PRIMARY KEY,
    cik TEXT NOT NULL,
    current_ticker TEXT,
    company_name TEXT,
    first_seen_date DATE,
    last_seen_date DATE,
    source TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

---

### 7.2 `filings`

```sql
CREATE TABLE filings (
    filing_event_id TEXT PRIMARY KEY,
    internal_company_id TEXT NOT NULL,
    cik TEXT NOT NULL,
    accession_number TEXT NOT NULL,
    form_type TEXT NOT NULL,
    filed_date DATE,
    accepted_datetime TIMESTAMP,
    report_period_end_date DATE,
    primary_document TEXT,
    is_amendment BOOLEAN,
    source TEXT,
    downloaded_at TIMESTAMP,
    UNIQUE(cik, accession_number)
);
```

---

### 7.3 `financial_facts_long`

```sql
CREATE TABLE financial_facts_long (
    fact_id TEXT PRIMARY KEY,
    internal_company_id TEXT NOT NULL,
    cik TEXT NOT NULL,
    taxonomy TEXT,
    concept_name TEXT NOT NULL,
    unit TEXT NOT NULL,
    value DOUBLE PRECISION,
    period_start_date DATE,
    period_end_date DATE,
    fiscal_year INTEGER,
    fiscal_period TEXT,
    form_type TEXT,
    accession_number TEXT,
    accepted_datetime TIMESTAMP,
    filed_date DATE,
    frame TEXT,
    source TEXT,
    downloaded_at TIMESTAMP
);
```

Raw `concept_name` is stored here, but this table is not used directly by model training.

---

### 7.4 `anonymous_feature_map`

```sql
CREATE TABLE anonymous_feature_map (
    anonymous_feature_id TEXT PRIMARY KEY,
    taxonomy TEXT NOT NULL,
    concept_name TEXT NOT NULL,
    unit TEXT NOT NULL,
    first_seen_at TIMESTAMP,
    active BOOLEAN,
    mapping_version TEXT,
    UNIQUE(taxonomy, concept_name, unit, mapping_version)
);
```

This table is metadata. It must not be passed to the model.

---

### 7.5 `prices_daily`

```sql
CREATE TABLE prices_daily (
    internal_security_id TEXT,
    internal_company_id TEXT,
    ticker_at_date TEXT,
    date DATE,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    adjusted_close DOUBLE PRECISION,
    volume BIGINT,
    dividend DOUBLE PRECISION,
    split_coefficient DOUBLE PRECISION,
    source TEXT,
    downloaded_at TIMESTAMP,
    PRIMARY KEY(internal_security_id, date)
);
```

---

### 7.6 `event_samples`

```sql
CREATE TABLE event_samples (
    sample_id TEXT PRIMARY KEY,
    internal_company_id TEXT NOT NULL,
    cik TEXT NOT NULL,
    ticker_at_event TEXT,
    filing_event_id TEXT NOT NULL,
    report_event_date DATE NOT NULL,
    accepted_datetime TIMESTAMP,
    form_type TEXT,
    period_end_date DATE,
    entry_date DATE,
    entry_price DOUBLE PRECISION,
    next_filing_event_id TEXT,
    next_report_event_date DATE,
    exit_date DATE,
    exit_price DOUBLE PRECISION,
    raw_log_return DOUBLE PRECISION,
    raw_simple_return DOUBLE PRECISION,
    ex_ante_scale DOUBLE PRECISION,
    target_normalized_signal DOUBLE PRECISION,
    spy_log_return DOUBLE PRECISION,
    qqq_log_return DOUBLE PRECISION,
    eligible_for_training BOOLEAN,
    eligible_for_backtest BOOLEAN,
    exclusion_reason TEXT,
    created_at TIMESTAMP
);
```

---

## 8. Sample Construction Algorithm

Pseudocode:

```python
for company in companies:
    filings = get_filings(company, forms=["10-Q", "10-K"], exclude_amendments=True)
    filings = sort_by_accepted_datetime(filings)

    for idx in range(len(filings) - 1):
        current_event = filings[idx]
        next_event = filings[idx + 1]

        report_event_date = eastern_date(current_event.accepted_datetime)
        next_report_event_date = eastern_date(next_event.accepted_datetime)

        entry_date = trading_calendar.next_trading_day(report_event_date)
        exit_date = trading_calendar.previous_trading_day(next_report_event_date)

        entry_price = adjusted_close(company, entry_date)
        exit_price = adjusted_close(company, exit_date)

        if invalid(entry_date, exit_date, entry_price, exit_price):
            write_excluded_sample(reason)
            continue

        raw_log_return = log(exit_price / entry_price)
        raw_simple_return = exit_price / entry_price - 1

        write_event_sample(...)
```

---

## 9. Benchmark Return Alignment

For each event sample:

```text
benchmark_entry_price = benchmark adjusted close on entry_date
benchmark_exit_price = benchmark adjusted close on exit_date
benchmark_log_return = log(benchmark_exit_price / benchmark_entry_price)
```

Required:

```text
spy_log_return
qqq_log_return
```

If benchmark prices are missing:

```text
sample remains valid for model training
benchmark comparison for that sample is marked missing
```

---

## 10. Event-Level Excess Returns

Store optional diagnostic targets:

```text
raw_log_return_minus_spy = raw_log_return - spy_log_return
raw_log_return_minus_qqq = raw_log_return - qqq_log_return
```

Version 1 model target remains absolute normalized directional movement unless configured otherwise.

Optional target modes:

```yaml
target_mode:
  absolute_log_return
  excess_vs_spy
  excess_vs_qqq
```

---

## 11. Point-in-Time Feature Availability

For each sample, features may use only financial facts satisfying:

```text
fact.accepted_datetime <= current_event.accepted_datetime
```

Recommended stricter condition:

```text
fact.accepted_datetime <= entry_date market open
```

Since Version 1 buys after the first post-event trading day close, using facts from the current filing is allowed.

Forbidden:

```text
fact.accepted_datetime > current_event.accepted_datetime
```

---

## 12. Handling Multiple Filings on Same Date

If multiple filings exist for same company on the same date:

1. Keep all filings in raw data.
2. For event construction, select the latest accepted filing per form priority.
3. If both 10-Q and 10-K occur on the same date, prefer 10-K.
4. Record excluded filings with reason:

```text
duplicate_event_same_date
```

---

## 13. Handling Amendments

Version 1 default:

```yaml
include_amendments_as_events: false
```

Amendments may update raw facts but must not alter historical event samples unless running a dedicated restatement experiment.

If amendments are used later:

```text
form_type in ["10-Q/A", "10-K/A"]
```

must be clearly separated from original reports.

---

## 14. Sample Split Fields

Each sample must receive a time-based split:

```text
split
```

Allowed values:

```text
train
validation
test
```

Default split:

```yaml
train: samples with entry_date < 2016-01-01
validation: 2016-01-01 <= entry_date < 2020-01-01
test: entry_date >= 2020-01-01
```

Walk-forward experiments may override this.

Do not use random split across time for final evaluation.

---

## 15. Exclusion Reasons

Allowed exclusion reasons:

```text
missing_next_event
missing_entry_price
missing_exit_price
entry_after_exit
insufficient_price_history
insufficient_financial_history
failed_liquidity_filter
security_not_common_stock
duplicate_event_same_date
invalid_adjusted_close
missing_ex_ante_scale
target_outlier_filtered
```

Each excluded candidate should be logged.

---

## 16. Output Files

This phase produces:

```text
data/processed/event_samples.parquet
data/processed/event_samples_excluded.parquet
data/processed/sample_splits.parquet
data/processed/event_benchmark_returns.parquet
```

---

## 17. Validation Tests

### 17.1 Date Ordering Test

For every valid sample:

```text
report_event_date < entry_date <= exit_date < next_report_event_date
```

The strict relationship between report event and entry date is:

```text
entry_date > report_event_date
```

---

### 17.2 Price Test

For every valid sample:

```text
entry_price > 0
exit_price > 0
```

---

### 17.3 No Future Filing Test

For every sample feature extraction window:

```text
max(feature_fact.accepted_datetime) <= sample.accepted_datetime
```

---

### 17.4 Next Event Test

For every valid sample:

```text
next_report_event_date is not null
```

---

## 18. Acceptance Criteria

Phase 2 is complete when:

1. `event_samples.parquet` exists.
2. Each valid sample has entry and exit dates.
3. Each valid sample has raw log return.
4. Each valid sample maps to one company and one current filing.
5. Excluded samples are saved with reasons.
6. Benchmarks are aligned by event period.
7. Time splits are assigned without leakage.
8. Unit tests pass for date ordering and no-future-filing logic.

---

## 19. Example Sample Row

```json
{
  "sample_id": "SAMPLE_7f3a2c",
  "internal_company_id": "COMP000001",
  "cik": "0000320193",
  "ticker_at_event": "AAPL",
  "filing_event_id": "FILING_2020Q2_AAPL",
  "report_event_date": "2020-05-01",
  "entry_date": "2020-05-04",
  "entry_price": 71.25,
  "next_report_event_date": "2020-07-31",
  "exit_date": "2020-07-30",
  "exit_price": 95.04,
  "raw_log_return": 0.2882,
  "raw_simple_return": 0.3339,
  "ex_ante_scale": null,
  "target_normalized_signal": null,
  "spy_log_return": 0.1201,
  "qqq_log_return": 0.1985,
  "eligible_for_training": true,
  "eligible_for_backtest": true,
  "exclusion_reason": null
}
```
