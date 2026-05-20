# RFC-03: Anonymous Feature Encoding

Status: Draft  
Phase: 3  
Purpose: Define how raw financial statement values are converted into stable anonymous model inputs without human-readable financial meaning.

---

## 1. Objective

Create model-ready feature matrices from raw financial facts while enforcing:

1. Stable anonymous feature IDs.
2. No human-readable field names in model input.
3. No manually engineered financial ratios in Version 1.
4. Explicit missing-value handling.
5. Point-in-time feature availability.
6. Consistent scaling and transformation.

---

## 2. Core Rule

The model must not receive:

```text
concept_name
financial field name
statement category
human-readable accounting description
human-built ratio name
```

The model may receive:

```text
anonymous_feature_id
numeric transformed value
missing flag
period offset
filing recency metadata
fiscal period metadata
```

---

## 3. Anonymous Feature Mapping

### 3.1 Mapping Key

Each unique financial field is defined by:

```text
taxonomy
concept_name
unit
```

Example raw key:

```text
taxonomy = "us-gaap"
concept_name = "Revenues"
unit = "USD"
```

Mapping output:

```text
anonymous_feature_id = "feature_000001"
```

---

### 3.2 Mapping Table

Required columns:

```text
anonymous_feature_id
taxonomy
concept_name
unit
mapping_version
first_seen_at
active
```

Example:

```json
{
  "anonymous_feature_id": "feature_000001",
  "taxonomy": "us-gaap",
  "concept_name": "Revenues",
  "unit": "USD",
  "mapping_version": "v1",
  "first_seen_at": "2026-05-20T00:00:00Z",
  "active": true
}
```

This table must be saved for reproducibility but never passed as text to the model.

---

### 3.3 Stable Assignment

Feature IDs must be deterministic.

Recommended:

```text
anonymous_feature_id = "feature_" + zero_padded_rank(sort(unique(taxonomy, concept_name, unit)))
```

Alternative:

```text
anonymous_feature_id = "feature_" + first_12_chars(hash(taxonomy, concept_name, unit))
```

If rank-based IDs are used, freeze mapping after training data generation.

---

## 4. Feature Selection

### 4.1 Initial Universe of Financial Facts

Start with all SEC facts satisfying:

```text
form_type in ["10-Q", "10-K"]
value is numeric
unit is not null
accepted_datetime <= sample accepted_datetime
```

---

### 4.2 Minimum Coverage Filter

To avoid extremely sparse fields, include a feature only if:

```yaml
min_company_count: 100
min_sample_coverage_pct: 1.0
```

This means a feature must appear in at least:

```text
100 companies
and 1% of eligible samples
```

These thresholds are configurable.

---

### 4.3 Maximum Sparsity

A feature with more than:

```yaml
max_missing_pct: 99.0
```

may be excluded from Version 1.

---

### 4.4 No Human Ratio Features

Forbidden in Version 1:

```text
ROE
ROA
gross_margin
operating_margin
debt_to_equity
current_ratio
free_cash_flow_yield
PE ratio
PS ratio
PB ratio
```

Raw components may be included as anonymous fields.

---

## 5. Historical Period Window

For each sample, include raw facts from the current and prior reporting periods.

Default:

```yaml
periods_back: 8
```

The feature tensor has dimensions:

```text
num_samples x periods_back x num_anonymous_features
```

Where:

```text
period_offset = 0 means latest available period at sample event
period_offset = 1 means previous report period
...
period_offset = 7 means eighth latest period
```

---

## 6. Assigning Facts to Period Offsets

For company `i` and sample event `t`:

1. Collect filings with accepted datetime <= current event accepted datetime.
2. Sort by period end date descending.
3. Deduplicate same period and same concept.
4. Assign latest period as offset `0`.
5. Assign previous period as offset `1`.
6. Continue until `periods_back - 1`.

---

## 7. Deduplication Rules

If multiple facts exist for the same:

```text
company
anonymous_feature_id
period_end_date
unit
```

Use the fact with:

1. latest accepted datetime <= sample accepted datetime,
2. preferred form priority:
   - 10-K over 10-Q for annual period,
   - original filing over amendment in Version 1,
3. non-null value.

Record conflicts in diagnostics.

---

## 8. Value Transformation

Raw financial values vary by size. Feeding raw dollar values directly can cause model instability.

Version 1 should create transformed numeric values while preserving anonymity.

### 8.1 Signed Log Transform

For numeric value `x`:

```text
signed_log_value = sign(x) * log1p(abs(x))
```

This handles:

```text
positive values
negative values
zero
large dollar values
```

---

### 8.2 Cross-Sectional Robust Scaling

For each anonymous feature and period offset, compute training-only statistics:

```text
median_f
iqr_f = percentile_75_f - percentile_25_f
```

Then:

```text
scaled_value = (signed_log_value - median_f) / max(iqr_f, epsilon)
```

Default:

```yaml
epsilon: 1e-6
clip_scaled_value: [-10, 10]
```

Statistics must be fit on training data only.

---

### 8.3 Missing Values

For each feature value, store:

```text
value_scaled
missing_flag
```

If missing:

```text
value_scaled = 0
missing_flag = 1
```

If present:

```text
missing_flag = 0
```

---

## 9. Metadata Features

Allowed metadata in Version 1:

```text
period_offset
fiscal_period_index
months_since_period_end
filing_delay_days
is_annual_report
```

These must be numeric or categorical IDs.

Human-readable strings are forbidden.

---

### 9.1 Fiscal Period Encoding

Allowed:

```text
fiscal_period_index:
  FY = 4
  Q1 = 1
  Q2 = 2
  Q3 = 3
  Q4 = 4
```

If using embeddings, pass integer category only.

---

### 9.2 Filing Delay

```text
filing_delay_days = accepted_date - period_end_date
```

Clip:

```yaml
filing_delay_days_clip: [0, 180]
```

---

### 9.3 Months Since Period End

```text
months_since_period_end = floor((entry_date - period_end_date) / 30.4375)
```

Clip:

```yaml
months_since_period_end_clip: [0, 24]
```

---

## 10. Optional Company Identity

Default:

```yaml
include_company_id_embedding: false
```

Reason:

If company ID is included, the model may memorize company-specific behavior instead of learning financial patterns.

Optional experiment:

```yaml
include_company_id_embedding: true
```

If enabled:

```text
company_id must be anonymous integer ID
```

Do not pass ticker or company name.

---

## 11. Optional Sector Identity

Default:

```yaml
include_sector_id_in_model: false
```

If enabled later:

```text
sector_id must be anonymous integer ID
```

Do not pass sector name.

---

## 12. Model Input Formats

### 12.1 Dense Tensor Format

For tree-based models:

```text
sample_id
feature_000001_offset_0_value
feature_000001_offset_0_missing
feature_000001_offset_1_value
feature_000001_offset_1_missing
...
```

Output:

```text
X_train.parquet
X_validation.parquet
X_test.parquet
```

---

### 12.2 Sparse Long Format

For neural set/transformer models:

```text
sample_id
period_offset
anonymous_feature_index
value_scaled
missing_flag
metadata...
```

Output:

```text
features_long.parquet
```

---

### 12.3 Tensor Format

For PyTorch:

```text
values: float32 tensor [num_samples, periods_back, num_features]
missing: float32 tensor [num_samples, periods_back, num_features]
metadata: tensor [num_samples, periods_back, num_metadata_features]
target: float32 tensor [num_samples]
sample_ids: string array
```

Output:

```text
model_inputs/train.pt
model_inputs/validation.pt
model_inputs/test.pt
```

---

## 13. Fit/Transform Split

The feature encoder must have two stages:

### 13.1 Fit

Fit only on training samples:

```python
encoder.fit(train_samples)
```

Creates:

```text
anonymous_feature_map
coverage filters
scaling statistics
feature ordering
metadata encoders
```

---

### 13.2 Transform

Apply to validation/test/backtest samples:

```python
encoder.transform(samples)
```

No new scaling statistics may be learned from validation/test.

If unseen features appear in validation/test:

```text
ignore unseen feature
log unseen feature count
```

Do not change feature ordering after fit.

---

## 14. Feature Encoder Artifact

Save:

```text
artifacts/feature_encoder/{version}/anonymous_feature_map.parquet
artifacts/feature_encoder/{version}/scaling_stats.parquet
artifacts/feature_encoder/{version}/feature_config.yaml
artifacts/feature_encoder/{version}/metadata.json
```

Metadata:

```json
{
  "encoder_version": "feature_encoder_v001",
  "fit_sample_start": "2005-01-01",
  "fit_sample_end": "2015-12-31",
  "num_features": 1843,
  "periods_back": 8,
  "value_transform": "signed_log_robust_scale",
  "created_at": "2026-05-20T00:00:00Z"
}
```

---

## 15. Leakage Tests

### 15.1 Human Name Leakage Test

Scan model input column names.

Forbidden patterns:

```text
Revenue
Income
Asset
Liability
Cash
Debt
Equity
Earnings
Margin
ROE
ROA
```

Allowed pattern:

```text
feature_[0-9]+
```

Test must fail if human financial words appear in model input columns.

---

### 15.2 Future Fact Test

For each sample:

```text
max(fact.accepted_datetime used in features) <= sample.accepted_datetime
```

---

### 15.3 Training-Only Scaler Test

Scaler statistics must be computed from train split only.

Test:

```text
validation/test sample ids not present in scaler fit data
```

---

### 15.4 Stable Feature ID Test

For every mapping version:

```text
one anonymous_feature_id maps to exactly one taxonomy/concept/unit
one taxonomy/concept/unit maps to exactly one anonymous_feature_id
```

---

## 16. Acceptance Criteria

Phase 3 is complete when:

1. Anonymous feature map exists.
2. Model inputs contain no human-readable financial field names.
3. Feature IDs are stable.
4. Missing flags are present.
5. Value scaling uses train-only statistics.
6. Current and prior periods are encoded.
7. Validation/test transformation does not refit scalers.
8. Leakage tests pass.
9. Model input artifacts are saved in dense and/or tensor format.

---

## 17. Example Config

```yaml
features:
  mapping_version: "v1"
  periods_back: 8
  min_company_count: 100
  min_sample_coverage_pct: 1.0
  max_missing_pct: 99.0

  value_transform:
    method: signed_log
    robust_scale: true
    clip_min: -10
    clip_max: 10
    epsilon: 1e-6

  missing_values:
    value_when_missing: 0
    include_missing_flag: true

  metadata:
    include_period_offset: true
    include_fiscal_period_index: true
    include_filing_delay_days: true
    include_months_since_period_end: true
    include_company_id_embedding: false
    include_sector_id_in_model: false

  forbidden_model_input_strings:
    - Revenue
    - Income
    - Assets
    - Liabilities
    - Cash
    - Debt
    - Equity
    - EPS
    - Margin
    - ROE
    - ROA
```

---

## 18. Example Dense Row

```json
{
  "sample_id": "SAMPLE_7f3a2c",
  "feature_000001_o0_value": 3.41,
  "feature_000001_o0_missing": 0,
  "feature_000001_o1_value": 3.27,
  "feature_000001_o1_missing": 0,
  "feature_000002_o0_value": -0.44,
  "feature_000002_o0_missing": 0,
  "feature_000999_o0_value": 0.0,
  "feature_000999_o0_missing": 1,
  "meta_filing_delay_days_o0": 32,
  "meta_is_annual_o0": 0
}
```

This row is acceptable because it contains no human-readable financial meaning.
