# Phase 12 — Sample-Level Universe Filter

Maps to **RFC-01 §6.1 / §6.3** applied at the sample-build stage (not at
ticker-master level). Motivated by Phase 11's finding that the 1000-CIK
unfiltered run was dominated by micro-cap / illiquid noise.

## The Problem

Phase 11 ingested 1000 CIKs straight from SEC's ticker file with no liquidity
or price filter at the sample level. The result:

- 950 effective companies, many low-price / low-ADV / short-history
- 26,724 event samples, but only ~2,884 made it past target normalization
  into training — many had no scale because of sparse price history
- LightGBM Sharpe 0.72 vs equal-risk-active 1.48: model gets crushed

The RFC explicitly defines a liquidity-filtered universe (§6.1) but we had
only applied it to the *universe* function, not to the sample stream.

## Module

`afp.features.universe_filter.apply_sample_universe_filter(samples, prices, cfg)`

For each sample, at `entry_date`:

1. Look up the company's price series **strictly before** `entry_date`.
2. Require `len(past) ≥ min_years_price_history * 252`.
3. Require `adj_close[entry_date - 1] ≥ min_adjusted_close_price_usd`.
4. Require `trailing_252d_dollar_volume.shift(1) ≥ min_avg_daily_dollar_volume_usd`.

Failing samples are returned in `dropped` with an explicit reason
(`filter_below_min_price`, `filter_below_min_adv`,
`filter_insufficient_history`, `filter_no_price_series`).

## RFC Compliance

- **No look-ahead**: the rolling dollar-volume series is `.shift(1)`'d
  before the lookup, and only bars `< entry_date` are considered.
- **No future universe membership**: only point-in-time eligibility.
- **No survivor-only bias** *introduced by this filter*: surviving names
  with insufficient history at sample time are correctly dropped at that
  sample; they may be eligible at later samples.

## Wiring

`afp.cli.build_dataset` reads `data.sample_filter` from config:

```yaml
data:
  sample_filter:
    enabled: false                       # default off for backward compat
    min_adjusted_close_price_usd: 5.0
    min_avg_daily_dollar_volume_usd: 10000000.0
    min_years_price_history: 3.0
    avg_dollar_volume_lookback_days: 252
```

When `enabled: true`, the filter runs immediately after `build_event_samples`
and before scale/target/encoder. The encoder will therefore fit only on the
filtered universe, which prevents sparse micro-cap features from setting the
median/IQR baselines.

## Expected Effect (to be measured in Phase 17)

- Sample count drops by ~50-70% on the 1000-CIK universe (many recent IPOs
  or low-volume names fail).
- Feature coverage tightens — encoder retains only features that pass
  `min_company_count` on the filtered set.
- Random-positive and shuffled-signal baselines should improve too, but
  not by as much as the model (signal-to-noise rises).
- Strategy Sharpe expected to move from 0.72 toward equal-risk-active.

## Acceptance — `tests/test_universe_filter.py`

1. `enabled=False` returns input unchanged.
2. Low-price companies (< $5) are dropped with `filter_below_min_price`.
3. Illiquid companies (low ADV) are dropped with `filter_below_min_adv`.
4. Short-history companies are dropped with `filter_insufficient_history`.
5. Poisoning prices on/after `entry_date` does not change which samples are
   kept — verifies the past-only constraint structurally.
