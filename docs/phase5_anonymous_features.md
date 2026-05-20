# Phase 5 — Anonymous Feature Encoder

Maps to **RFC-03** and **RFC-09 Milestone 5**.

## Architecture

```
facts (long)                         samples (event-driven)
    │                                       │
    └──► AnonymousFeatureMap.fit ──► table  │
                                            ▼
                                  build_sample_matrices ──► (values, missing, metadata)
                                            │
                                            ▼
                                     RobustScaler.fit (train only)
                                            │
                                            ▼
                              encoder.transform(any split, facts)
                                            │
                       ┌────────────────────┴───────────────────┐
                       ▼                                        ▼
              dense parquet rows                    np tensors (model_inputs/*)
```

## `AnonymousFeatureMap`

- Key: `(taxonomy, concept_name, unit)` → `feature_NNNNNN`.
- IDs are assigned by sorted rank for **stability across re-runs** (RFC-03 §3.3).
- Coverage filters:
  - `min_company_count` (default 100): drops fields seen in too few companies.
  - `min_sample_coverage_pct` and `max_missing_pct` (default 1% / 99%).
- The raw mapping table is metadata only — it is written to
  `artifacts/feature_encoder/<v>/anonymous_feature_map.parquet` but never joined into model input.

## `build_sample_matrices` (RFC-03 §5-§7)

For each sample:

1. Filter facts with `accepted_datetime ≤ sample.accepted_datetime` (point-in-time).
2. Dedup within `(anonymous_feature_id, period_end_date)` preferring 10-K > 10-Q, then latest accepted.
3. Sort the surviving `period_end_date`s descending; assign `period_offset ∈ {0..P-1}`.
4. Fill `values[N,P,F]` from the dedup'd rows; default missing.
5. Per `period_offset`, store metadata: `filing_delay_days`, `fiscal_period_index`,
   `months_since_period_end`, each clipped to its RFC-03 §9 range.

## `RobustScaler` (RFC-03 §8)

```
signed = sign(x) * log1p(|x|)         # signed-log
scaled = (signed - median_pf) / max(IQR_pf, epsilon)
scaled = clip(scaled, [-10, 10])
```

`fit` consumes only training data and stores `(median, iqr)` arrays of shape
`[periods_back, num_features]`. Missing values are kept as NaN during fit and
replaced with zero after transform; the missing flag carries the signal.

## Dense Output

`to_dense_frame` produces rows like:

```
sample_id, feature_000001_o0_value, feature_000001_o0_missing,
           feature_000001_o1_value, ..., meta_filing_delay_days_o0, ...
```

The function `check_no_human_field_names(columns)` returns offending column
names if any of `{Revenue, Income, Asset, Liability, Cash, Debt, Equity, EPS,
Margin, ROE, ROA, Earnings}` appear (case-insensitive, excluding `meta_*`).
The leakage gate in Phase 9 runs this on every materialized model input.

## Persistence

```
artifacts/feature_encoder/<version>/
  anonymous_feature_map.parquet
  scaling_stats.npz       # median, iqr arrays
  metadata.json           # {periods_back, num_features, mapping_version}
```

`AnonymousFeatureEncoder.load(path)` round-trips bit-identically — covered by
the `test_save_and_load_roundtrip` test.

## Acceptance — verified by `tests/test_anonymous_features.py`

1. Encoder fits without error on training samples.
2. Dense column names contain none of the forbidden tokens.
3. Each anonymous feature ID maps to exactly one `(taxonomy, concept, unit)` triple.
4. Polluting facts whose `accepted_datetime > train_cutoff` leaves train-fitted
   scaler stats unchanged.
5. Polluting facts with `accepted_datetime > sample.accepted_datetime` leaves
   transformed values unchanged.
6. Save → load round-trip yields identical transforms.
