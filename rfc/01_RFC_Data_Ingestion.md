# RFC-01: Data Ingestion

Status: Draft  
Phase: 1  
Purpose: Define how to collect public US equity financial statements, prices, report events, identifiers, trading calendars, and benchmark data.

---

## 1. Objective

Build a reproducible data ingestion layer that collects:

1. US company identifiers.
2. SEC financial statement facts.
3. SEC filing dates.
4. Stock adjusted close prices.
5. Benchmark adjusted close prices.
6. Trading calendar.
7. Optional sector and industry metadata for portfolio constraints.

The ingestion layer must save raw data before transformation.

---

## 2. Required Data Domains

### 2.1 Company Identifier Data

Required fields:

```text
internal_company_id
cik
ticker
company_name
exchange
security_type
first_seen_date
last_seen_date
source
```

Rules:

1. `internal_company_id` is permanent inside this system.
2. Do not use ticker as the primary key.
3. Track ticker changes when available.
4. Store CIK as zero-padded 10-digit string.
5. Exclude ETFs in Version 1 unless explicitly configured otherwise.

Example:

```json
{
  "internal_company_id": "COMP000001",
  "cik": "0000320193",
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "exchange": "NASDAQ",
  "security_type": "common_stock",
  "first_seen_date": "1980-12-12",
  "last_seen_date": null,
  "source": "sec_company_tickers"
}
```

---

### 2.2 Financial Statement Facts

Required raw fields:

```text
fact_id
cik
taxonomy
concept_name
unit
value
period_start_date
period_end_date
fiscal_year
fiscal_period
form_type
filing_accession_number
accepted_datetime
filed_date
frame
source_url_or_file
downloaded_at
```

Examples of original concept names, stored in raw data but not passed to the model:

```text
us-gaap:Revenues
us-gaap:NetIncomeLoss
us-gaap:Assets
us-gaap:Liabilities
us-gaap:CashAndCashEquivalentsAtCarryingValue
```

The raw layer may store these names. The model input layer must not.

---

### 2.3 Filing Event Data

Required fields:

```text
filing_event_id
cik
accession_number
form_type
filed_date
accepted_datetime
report_period_end_date
primary_document
is_amendment
source
downloaded_at
```

Supported forms in Version 1:

```text
10-Q
10-K
```

Optional later:

```text
8-K
10-Q/A
10-K/A
earnings press release
```

Version 1 excludes amendments from event construction unless explicitly configured.

---

### 2.4 Price Data

Required fields:

```text
price_id
internal_security_id
ticker_at_date
date
open
high
low
close
adjusted_close
volume
dividend
split_coefficient
source
downloaded_at
```

Rules:

1. Use adjusted close for target and backtest returns.
2. Keep raw close for diagnostics.
3. Missing adjusted close means the sample cannot be used for target construction.
4. Store source-specific adjustment method if available.

---

### 2.5 Benchmark Data

Required benchmarks:

```text
S&P 500 proxy
Nasdaq-100 proxy
```

Recommended proxies:

```text
SPY adjusted close
QQQ adjusted close
```

If total return index data is available, store it separately and prefer total return for reporting.

Required benchmark fields:

```text
benchmark_id
symbol
date
adjusted_close
source
downloaded_at
```

---

### 2.6 Trading Calendar

Required fields:

```text
date
is_trading_day
market
session_open_time
session_close_time
timezone
```

Default market:

```text
XNYS / US equities
```

The trading calendar is required for:

1. finding the first trading day after a report event,
2. finding the trading day before the next report event,
3. aligning portfolio returns,
4. handling weekends and holidays.

---

## 3. Data Sources

### 3.1 SEC Data

Use SEC public data for:

1. company tickers,
2. submissions,
3. company facts,
4. filing metadata.

The ingestion layer must set a compliant user agent.

Example config:

```yaml
sec:
  base_url: "https://data.sec.gov"
  user_agent: "anonymous-fundamental-portfolio research contact@example.com"
  max_requests_per_second: 8
  retry:
    max_attempts: 5
    backoff_seconds: [1, 2, 4, 8, 16]
```

Do not exceed source rate limits.

---

### 3.2 Price Data Source

Version 1 may use a public or licensed price source.

The data client interface must be source-agnostic:

```python
class PriceClient:
    def get_daily_prices(self, ticker: str, start_date: str, end_date: str) -> DataFrame:
        ...
```

Minimum columns returned:

```text
date
open
high
low
close
adjusted_close
volume
```

---

### 3.3 Sector Metadata

Sector metadata is optional for Version 1.

If used, it must be used only by the portfolio module for exposure caps, not by the prediction model unless explicitly enabled.

Required fields:

```text
internal_company_id
sector_source
sector_code
sector_name
industry_code
industry_name
effective_start_date
effective_end_date
```

Prediction model input should use either:

```yaml
sector_in_model: false
```

or, if enabled later:

```text
anonymous_sector_id
```

Never pass sector names as text to the model in Version 1.

---

## 4. Raw Data Storage

All ingested raw files must be stored before cleaning.

Recommended layout:

```text
data/raw/sec/company_tickers/
data/raw/sec/submissions/
data/raw/sec/companyfacts/
data/raw/prices/daily/
data/raw/benchmarks/
data/raw/calendar/
data/raw/sectors/
```

Each downloaded file should have a metadata sidecar:

```json
{
  "source": "sec_companyfacts",
  "url": "source url",
  "downloaded_at": "2026-05-20T12:00:00Z",
  "sha256": "file hash",
  "request_headers": {
    "User-Agent": "configured user agent"
  }
}
```

---

## 5. Cleaned Tables

The ingestion phase must produce these cleaned tables:

```text
companies.parquet
securities.parquet
filings.parquet
financial_facts.parquet
prices_daily.parquet
benchmarks_daily.parquet
trading_calendar.parquet
sectors.parquet
```

---

## 6. Universe Construction

### 6.1 Default Universe Filters

A company is eligible for the model universe if:

```yaml
has_cik: true
has_common_stock_price_history: true
min_adjusted_close_price_usd: 5
min_average_daily_dollar_volume_usd: 10000000
min_financial_report_events: 12
min_years_price_history: 5
exclude_etfs: true
exclude_funds: true
exclude_preferreds: true
```

---

### 6.2 Dollar Volume

For company `i` at date `d`:

```text
dollar_volume_i,d = adjusted_close_i,d * volume_i,d
```

Eligibility uses trailing average:

```text
avg_dollar_volume_i,d = mean(dollar_volume_i,d-252 ... dollar_volume_i,d-1)
```

Must use only past data.

---

### 6.3 Point-in-Time Universe

The universe at date `d` must not depend on future survival.

Correct:

```text
eligible_i,d = has required history and liquidity as of d
```

Incorrect:

```text
include only companies that still exist at the end of the backtest
```

Version 1 may use best-effort survivorship handling, but the report must disclose limitations.

---

## 7. Ingestion Pipeline Steps

### Step 1: Download Company List

Output:

```text
companies_raw.json
companies.parquet
```

Validation:

```text
cik is not null
ticker is not null
security_type classified
```

---

### Step 2: Download SEC Submissions

For each CIK:

```text
submissions/CIK##########.json
```

Extract:

```text
10-Q and 10-K filings
accepted datetime
filed date
accession number
period end date
form type
```

---

### Step 3: Download SEC Company Facts

For each CIK:

```text
companyfacts/CIK##########.json
```

Extract all available concepts and units.

Do not filter to only known financial fields in raw layer.

---

### Step 4: Normalize Financial Facts

Convert nested source data to long format.

Each row:

```text
cik
taxonomy
concept_name
unit
value
period_start_date
period_end_date
form_type
filed_date
accepted_datetime
accession_number
frame
```

Do not calculate financial ratios in Version 1.

---

### Step 5: Download Daily Prices

For each security:

```text
prices_daily.parquet
```

Required date coverage:

```text
start_date <= earliest report event entry date - 3 years
end_date >= latest report event exit date
```

---

### Step 6: Download Benchmarks

Symbols:

```text
SPY
QQQ
```

Store adjusted close.

---

### Step 7: Build Trading Calendar

Use reliable US equity trading calendar.

Output:

```text
trading_calendar.parquet
```

---

## 8. Data Quality Checks

### 8.1 Financial Fact Checks

Required checks:

```text
no duplicate fact_id
accepted_datetime parseable
period_end_date parseable
value numeric
unit not null
concept_name not null
form_type in allowed forms
```

---

### 8.2 Price Checks

Required checks:

```text
date unique per security
adjusted_close > 0
volume >= 0
no impossible split-adjusted jumps without corresponding split metadata flagged
```

---

### 8.3 Event Checks

Required checks:

```text
each event has cik
each event has accepted_datetime
entry_date exists
next event exists
exit_date exists
entry_date < exit_date
entry_price > 0
exit_price > 0
```

---

## 9. Required Logs

Every ingestion run must save:

```text
run_id
started_at
finished_at
source
records_downloaded
records_written
errors
warnings
config_hash
code_commit_hash
```

---

## 10. Failure Handling

### 10.1 SEC Request Failure

If a request fails:

1. retry with exponential backoff;
2. if still failing, record CIK and error;
3. continue other CIKs;
4. write an error report.

---

### 10.2 Missing Price History

If company has financial data but no price data:

```text
exclude from sample construction
log reason = missing_price_history
```

---

### 10.3 Missing Next Report Event

If report event has no next report event:

```text
exclude from supervised training
log reason = missing_next_event
```

---

## 11. Acceptance Criteria

This phase is complete when:

1. At least 1,000 US companies are ingested, subject to data availability.
2. At least 10 years of price data are available where possible.
3. Financial facts are stored in long format.
4. Filing event table contains 10-Q and 10-K events.
5. Trading calendar supports next/previous trading day lookup.
6. Benchmark price data exists for SPY and QQQ or configured alternatives.
7. Data quality report is generated.
8. Raw files are preserved.
9. Cleaned parquet tables are reproducible from raw files.

---

## 12. Example CLI Commands

```bash
python -m src.data.ingest_companies --config configs/data.yaml
python -m src.data.ingest_sec_submissions --config configs/data.yaml
python -m src.data.ingest_companyfacts --config configs/data.yaml
python -m src.data.ingest_prices --config configs/data.yaml
python -m src.data.build_calendar --config configs/data.yaml
python -m src.data.validate_ingestion --config configs/data.yaml
```

---

## 13. Example Output Manifest

```json
{
  "run_id": "data_ingestion_20260520_120000",
  "companies": 4200,
  "filings": 185000,
  "financial_facts": 12500000,
  "price_rows": 8500000,
  "benchmark_rows": 6000,
  "calendar_rows": 7000,
  "warnings": [
    "315 companies missing sufficient price history",
    "74 companies missing sufficient 10-Q/10-K events"
  ]
}
```
