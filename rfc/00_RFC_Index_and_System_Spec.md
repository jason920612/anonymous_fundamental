# RFC-00: Anonymous Fundamental Portfolio Research System — Index and System Specification

Status: Draft  
Owner: Jason  
Purpose: Define the full research system for an event-driven US equity portfolio model using anonymous raw financial statement data, normalized directional movement prediction, and signal-weighted risk parity allocation.

---

## 1. System Goal

Build a research pipeline that:

1. Collects public US equity financial statement data.
2. Converts raw financial statement fields into stable anonymous feature IDs.
3. Prevents the prediction model from receiving human-readable financial field names.
4. Builds company-event samples around financial report events.
5. Predicts a bounded directional movement signal in `[-1, 1]`.
6. Restores the predicted signal into an expected directional movement using ex-ante volatility scale.
7. Constructs a long-only portfolio using signal-weighted risk parity.
8. Backtests the strategy against:
   - S&P 500 benchmark
   - Nasdaq-100 benchmark
   - equal-weight model universe baseline
   - equal-risk model universe baseline
9. Reports performance, risk, drawdown, turnover, transaction costs, and event-level prediction quality.

The first version is a research system, not a live trading system.

---

## 2. Core Research Hypothesis

The hypothesis is:

> A model can learn useful latent patterns from anonymous raw financial statement values across many US companies and use those patterns to predict the direction and normalized magnitude of price movement between one report event and the next report event.

The model should not be told that a field is revenue, net income, assets, liabilities, cash flow, EPS, debt, or any other financial concept.

The model may know that `feature_001` is the same field across all samples, but it must not receive the human-readable meaning of `feature_001`.

---

## 3. Non-Goals

This project does not attempt to:

1. Predict intraday earnings announcement jumps.
2. Use analyst estimates in the first version.
3. Use news, social media, earnings call transcripts, or alternative data in the first version.
4. Let the model directly output portfolio weights.
5. Use leverage in the first version.
6. Short stocks in the first version.
7. Optimize for tax outcomes.
8. Trade options.
9. Execute live orders.
10. Guarantee outperformance.

---

## 4. Phase Files

The RFC is split into the following implementation phases:

| File | Phase | Purpose |
|---|---:|---|
| `00_RFC_Index_and_System_Spec.md` | 0 | System goal, vocabulary, global rules |
| `01_RFC_Data_Ingestion.md` | 1 | Universe, public data sources, raw data collection |
| `02_RFC_Event_Dataset_and_Schemas.md` | 2 | Report-event sample construction and database schemas |
| `03_RFC_Anonymous_Feature_Encoding.md` | 3 | Anonymous field mapping, raw value tensor construction |
| `04_RFC_Target_Normalization.md` | 4 | Entry/exit prices, raw return, volatility scale, `[-1, 1]` target |
| `05_RFC_Modeling.md` | 5 | Baseline models, representation learning, optional diffusion model |
| `06_RFC_Portfolio_Construction.md` | 6 | Signal restoration, candidate selection, risk parity, constraints |
| `07_RFC_Backtesting_and_Evaluation.md` | 7 | Event-driven backtest engine and metrics |
| `08_RFC_Experiment_Management.md` | 8 | Reproducibility, configs, logging, model registry |
| `09_RFC_Implementation_Roadmap.md` | 9 | Milestones, acceptance tests, Claude Code task breakdown |

---

## 5. Definitions

### 5.1 Company

A tradable US-listed equity with:

- valid identifier mapping,
- sufficient public financial statement filings,
- price history,
- adjusted close series,
- acceptable liquidity.

Primary identifiers:

```text
cik
ticker
perm_id or internal_company_id
exchange
security_type
```

The system must not rely only on ticker, because tickers can change.

---

### 5.2 Report Event

A report event is a date on which a company publishes a financial report or earnings-related filing that updates the market with financial statement information.

For the first implementation, use the following priority:

1. SEC filing acceptance datetime for 10-Q and 10-K.
2. If earnings announcement dates are available from a reliable public source, store them separately.
3. If both SEC filing date and earnings date exist, keep both but explicitly define which one drives the event.

Version 1 event date:

```text
report_event_date = SEC filing acceptance date converted to US/Eastern calendar date
```

Later versions may switch to earnings announcement date if the data source is reliable.

---

### 5.3 Entry Date

For a report event at time `t`:

```text
entry_date = first valid trading day after report_event_date
```

Version 1 deliberately does not attempt to trade on the announcement day.

---

### 5.4 Exit Date

For the next report event at time `t + 1`:

```text
exit_date = trading day immediately before report_event_date_{t+1}
```

The model predicts movement from `entry_date_t` to `exit_date_t`.

---

### 5.5 Entry Price

```text
entry_price = adjusted close on entry_date
```

---

### 5.6 Exit Price

```text
exit_price = adjusted close on exit_date
```

---

### 5.7 Raw Return

Use log return:

```text
raw_return_i,t = log(exit_price_i,t / entry_price_i,t)
```

Simple return may be stored for reporting:

```text
simple_return_i,t = exit_price_i,t / entry_price_i,t - 1
```

The model target uses log return.

---

### 5.8 Ex-Ante Scale

`ex_ante_scale_i,t` is a volatility or movement scale computed using only information available no later than `entry_date_t`.

It must not use any return between `entry_date_t` and `exit_date_t`.

---

### 5.9 Normalized Directional Movement Target

```text
target_i,t = tanh(raw_return_i,t / (k * ex_ante_scale_i,t))
```

Where:

```text
target_i,t ∈ [-1, 1]
k > 0
```

Default:

```text
k = 2.5
```

---

### 5.10 Model Output

The model outputs:

```text
predicted_signal_i,t ∈ [-1, 1]
```

The prediction model must not output portfolio weights.

---

### 5.11 Restored Expected Directional Return

For portfolio construction:

```text
clipped_signal_i,t = clip(predicted_signal_i,t, -0.99, 0.99)
restored_expected_return_i,t = atanh(clipped_signal_i,t) * k * ex_ante_scale_i,t
```

---

### 5.12 Positive Signal

For long-only portfolio construction:

```text
positive_signal_i,t = max(predicted_signal_i,t, 0)
```

Negative signals mean the company is not eligible for long-only buying in Version 1.

---

## 6. Global Constraints

### 6.1 No Look-Ahead Bias

At sample time `t`, the system may only use data available on or before `entry_date_t`.

Forbidden:

- future prices,
- future financial filings,
- future restatements,
- future volatility,
- future universe membership,
- future benchmark constituents,
- future delisting information not known at the time.

---

### 6.2 No Restatement Leakage

The feature store must preserve the as-filed financial values available at the filing time.

If restated data is later downloaded, it must not overwrite historical as-of values used in past samples.

Every financial fact must store:

```text
source_filing_id
accepted_datetime
period_end_date
filed_value
downloaded_at
as_of_available_date
```

---

### 6.3 Stable Anonymous Feature IDs

The system may convert financial field names into anonymous IDs.

Allowed:

```text
us-gaap:Revenues -> feature_000123
us-gaap:NetIncomeLoss -> feature_000456
```

Forbidden model input:

```text
"Revenues"
"NetIncomeLoss"
"Assets"
"Liabilities"
"OperatingCashFlow"
```

The mapping table may exist in metadata, but the prediction model input pipeline must not pass human-readable field names.

---

### 6.4 Do Not Shuffle Feature Meanings Per Sample

The model cannot learn if `feature_001` means different things in different samples.

Correct:

```text
feature_001 always means the same original financial field internally
```

Incorrect:

```text
feature_001 means revenue for one sample and assets for another sample
```

---

### 6.5 Use Adjusted Prices

All target and portfolio return calculations must use adjusted close unless explicitly testing an alternative.

The system must handle:

- stock splits,
- dividends,
- symbol changes,
- delistings when data is available.

---

### 6.6 Model Does Not Know Human Field Meaning

The prediction model input must not include:

- financial field names,
- natural-language labels,
- statement category names,
- accounting concept descriptions,
- hand-built human financial ratios in Version 1.

Allowed:

- anonymous field ID,
- numeric value,
- missing flag,
- time offset,
- fiscal period metadata,
- filing age metadata,
- company anonymous ID if enabled,
- sector anonymous ID only if explicitly enabled.

---

### 6.7 Portfolio Module May Use Risk Metadata

The portfolio construction module may use:

- realized historical volatility,
- historical covariance,
- sector caps,
- liquidity constraints,
- benchmark membership,
- transaction cost estimates.

This metadata is not passed to the prediction model unless explicitly defined.

---

## 7. Initial Default Parameters

```yaml
universe:
  min_price_usd: 5
  min_avg_daily_dollar_volume_usd: 10000000
  min_history_years: 5
  include_etfs: false
  include_adrs: false
  include_financials: true

events:
  event_source_v1: sec_10q_10k_filing_date
  entry_rule: first_trading_day_after_report_event
  exit_rule: trading_day_before_next_report_event

target:
  return_type: log
  normalization: tanh
  k: 2.5
  scale_method: trailing_daily_vol_scaled_to_holding_period
  clip_signal_for_restore: 0.99

features:
  use_raw_financial_values: true
  use_human_financial_ratios: false
  anonymize_field_names: true
  stable_anonymous_feature_ids: true
  periods_back: 8
  include_missing_flags: true
  include_time_metadata: true

portfolio:
  long_only: true
  leverage: 1.0
  allow_cash: true
  max_single_stock_weight: 0.05
  min_positions: 30
  target_positions: 50
  max_positions: 100
  weighting: signal_weighted_inverse_vol_first_version
  transaction_cost_bps_per_trade: 10

backtest:
  train_start: "2005-01-01"
  validation_start: "2016-01-01"
  test_start: "2020-01-01"
  walk_forward_retrain_frequency: quarterly
```

---

## 8. Repository Structure

```text
anonymous-fundamental-portfolio/
  README.md
  pyproject.toml
  configs/
    default.yaml
    data.yaml
    model_baseline.yaml
    portfolio.yaml
    backtest.yaml
  data/
    raw/
    interim/
    processed/
    feature_store/
    model_inputs/
    backtest_outputs/
  docs/
    rfc/
      00_RFC_Index_and_System_Spec.md
      01_RFC_Data_Ingestion.md
      02_RFC_Event_Dataset_and_Schemas.md
      03_RFC_Anonymous_Feature_Encoding.md
      04_RFC_Target_Normalization.md
      05_RFC_Modeling.md
      06_RFC_Portfolio_Construction.md
      07_RFC_Backtesting_and_Evaluation.md
      08_RFC_Experiment_Management.md
      09_RFC_Implementation_Roadmap.md
  src/
    data/
      sec_client.py
      price_client.py
      identifiers.py
      universe.py
      trading_calendar.py
    features/
      anonymous_mapping.py
      raw_fact_matrix.py
      normalization.py
      sample_builder.py
    targets/
      event_returns.py
      volatility_scale.py
      target_transform.py
    models/
      datasets.py
      baselines.py
      neural.py
      train.py
      predict.py
    portfolio/
      signal_restore.py
      candidate_selection.py
      risk_models.py
      allocation.py
      constraints.py
    backtest/
      engine.py
      ledger.py
      metrics.py
      benchmarks.py
    experiments/
      registry.py
      logging.py
      validation.py
  tests/
    test_no_lookahead.py
    test_anonymous_features.py
    test_target_transform.py
    test_portfolio_constraints.py
    test_backtest_accounting.py
```

---

## 9. Version 1 Acceptance Criteria

Version 1 is acceptable only if all conditions below are true:

1. Data pipeline creates event samples without using future prices or future filings.
2. Financial fields are passed to the model as stable anonymous IDs.
3. The model receives no human-readable financial field names.
4. Target values are in `[-1, 1]`.
5. Ex-ante scale is computed using only past data.
6. Predictions are restored outside the model.
7. Portfolio weights are produced outside the model.
8. Long-only constraints are enforced.
9. Transaction costs are included.
10. Backtest reports benchmark-relative results.
11. Tests exist for look-ahead, target construction, feature anonymization, and portfolio constraints.
12. A naive baseline is included.

---

## 10. Required Baselines

Do not evaluate advanced models without these baselines:

1. Equal-weight universe portfolio.
2. Equal-risk universe portfolio.
3. Random signal portfolio with same number of holdings.
4. Ridge regression on anonymous features.
5. LightGBM or XGBoost on anonymous features.
6. Simple inverse-vol weighted portfolio using no model.

If the system cannot beat simple baselines after costs, advanced models are not justified.

---

## 11. Open Decisions

The following must be explicitly chosen before implementation:

1. Exact public price data source.
2. Whether to include delisted securities in Version 1.
3. Whether to use SEC filing dates or earnings announcement dates.
4. Whether to include banks and insurance companies in Version 1.
5. Whether sector data is allowed in portfolio risk module.
6. Whether anonymous company ID is passed to model.
7. Whether price momentum features are excluded from Version 1.

Recommended Version 1 choices:

```yaml
event_date: sec_filing_date
delisted_securities: best_effort_include_if_data_available
include_financials: true
sector_data: portfolio_module_only
company_id_in_model: false
price_momentum_in_model: false
```
