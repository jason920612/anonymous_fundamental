# Phase 4 — Target Normalization

Maps to **RFC-04** and **RFC-09 Milestone 4**.

## Modules

| Module | Role |
|---|---|
| `afp.targets.volatility_scale` | Trailing daily vol, prior-event vol, hybrid scale; `ScaleConfig` |
| `afp.targets.target_transform` | tanh forward, atanh inverse; `TargetConfig` |

## Scale Calculation (RFC-04 §4-7)

```
daily_log_return_i,d = log(adjusted_close_i,d / adjusted_close_i,d-1)

daily_vol_i,t      = std(daily_log_return_i,d  for d < entry_date_i,t, lookback=252)
scale_daily_i,t    = daily_vol_i,t * sqrt(holding_days_i,t)

prior_event_returns_i = [raw_log_return_i,j for j < t]  # strictly prior
event_vol_i,t         = 1.4826 * median(|x - median(x)|) over last 8 prior events
                        # MAD-based robust std, falls back to plain std if MAD=0
                        # returns None if fewer than 4 prior events

if both present:  scale = 0.7 * scale_daily + 0.3 * event_vol
elif daily only:  scale = scale_daily
elif event only:  scale = event_vol
else:             scale = NaN  → sample marked missing_ex_ante_scale

scale clipped to [0.02, 1.00]
```

The accumulator walks samples sorted by `(internal_company_id, entry_date)` and
appends `raw_log_return` to `prior_event_returns` **after** the current scale is
computed, guaranteeing event-vol uses only strictly-prior events.

## Target Transform (RFC-04 §8-9)

```
standardized_movement_i,t = raw_log_return_i,t / ex_ante_scale_i,t
target_normalized_signal_i,t = tanh(standardized_movement_i,t / k)        # k=2.5
```

`apply(samples, TargetConfig)`:
- adds `target_normalized_signal` for valid rows;
- marks rows with missing scale as `eligible_for_training = False` and tags
  `exclusion_reason = missing_ex_ante_scale`;
- defensively clips to `[-1, 1]`.

Also stores `is_extreme_target_event` (`|raw_log_return| > 3 * ex_ante_scale`)
for diagnostics — outliers are kept (RFC-04 §13), tanh bounds them naturally.

## Inverse Restoration (RFC-04 §10-12)

`restore(predicted_signal, ex_ante_scale, cfg)`:

```
clipped = clip(predicted_signal, -0.99, 0.99)
restored_expected_return = atanh(clipped) * k * ex_ante_scale
```

`linear` mode is provided as an option but the default is `atanh` (true inverse).

## Leakage Controls

- `trailing_daily_vol` uses `daily_returns.loc[idx < cutoff]` — strict less-than.
- `trailing_event_vol` is fed only samples with index `< t`.
- Cross-sectional fallbacks are not used in Version 1; missing scale ⇒
  sample dropped from training set.

## Acceptance — verified by `tests/test_target_transform.py`

1. `trailing_daily_vol` is invariant to future values written into the series.
2. `trailing_event_vol` returns `None` when fewer than `min_event_history` prior
   events exist.
3. End-to-end: poisoning prices **strictly after** the latest entry_date does
   not change any computed `ex_ante_scale`. (Direct test of "uses only past data".)
4. `target_normalized_signal ∈ [-1, 1]` for every eligible sample.
5. Samples with no scale are flagged with `missing_ex_ante_scale` and excluded
   from training.
6. `atanh ∘ tanh` round-trips raw return through restoration for randomly chosen
   triples `(raw, scale, k)`.
