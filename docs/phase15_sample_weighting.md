# Phase 15 — Inverse-Frequency Sample Weighting

## Why

Phase 11 confirmed the AAPL/MSFT/NVDA mega-caps each contribute 100+ event
samples — over 25 years of filings × 4 quarters per year. The 1000-CIK
universe has a *median* of 35 samples per company, but the long-tail of
mega-caps dominates the loss surface. Worse, those same mega-caps cluster
heavily in calendar dates (every quarter all big-tech reports in one week),
so calendar-quarter coverage is unbalanced too.

A model trained against unweighted MSE will:

1. memorize mega-cap idiosyncrasies (Apple's Q3 always beats — until it
   doesn't), and
2. underfit smaller-cap signal because those samples are crowded out.

Sample weighting is a cheap regularizer: count-balance the training set so
every company and every quarter contributes ~equal weight.

## Module

`afp.models.train.compute_sample_weights(train_samples, sample_ids,
                                        by_company, by_quarter)`

```
weight_i ∝ 1 / freq(internal_company_id_i)
weight_i ∝ 1 / freq(quarter_i)            # if by_quarter
weights normalized so mean(weights) == 1
```

Passed to `LightGBM.fit(sample_weight=...)`, `Ridge.fit(sample_weight=...)`,
and `HistoricalMean.fit(sample_weight=...)`. `ConstantZero` ignores it.

## Config

```yaml
model:
  sample_weighting:
    by_company: false      # AAPL/MSFT/NVDA don't dominate
    by_quarter: false      # rebalance across calendar quarters
```

Default off so old experiments are unchanged.

## RFC Compliance & Test-Data Discipline

- The model still sees only anonymous feature IDs (unchanged).
- Weights are computed from `internal_company_id` (an opaque token) and
  `entry_date.quarter` (purely calendar). Neither leaks human meaning to
  the model.
- Test data is **never** weighted because it is not used in fitting — `compute_sample_weights` is called only on `train_samples` inside
  `train_and_predict`. Validation rows produce metrics but no gradient.

## Acceptance — `tests/test_sample_weights.py`

1. With 8 AAA + 2 BBB samples, BBB rows get higher weights than AAA.
2. With 10 Q1 + 1 Q2 samples, Q2 row gets a higher weight than Q1.
3. With both flags off, weights = None (the trainer skips sample_weight).
4. Ridge + GBT accept `sample_weight` and still produce valid predictions.
