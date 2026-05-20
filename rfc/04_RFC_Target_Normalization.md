# RFC-04: Target Normalization

Status: Draft  
Phase: 4  
Purpose: Define raw return calculation, ex-ante volatility scale, normalized bounded target, inverse restoration, and leakage controls.

---

## 1. Objective

Create a model target that avoids making the model simply learn that some stocks are always more volatile than others.

The model should predict a normalized directional movement value:

```text
target ∈ [-1, 1]
```

The portfolio module later restores this into expected directional movement using ex-ante scale.

---

## 2. Raw Return

For company `i`, event `t`:

```text
entry_price_i,t = adjusted close on first trading day after report_event_date_i,t
exit_price_i,t = adjusted close on trading day before report_event_date_i,t+1
```

Raw log return:

```text
raw_return_i,t = log(exit_price_i,t / entry_price_i,t)
```

Raw simple return for reporting:

```text
simple_return_i,t = exit_price_i,t / entry_price_i,t - 1
```

The model target uses log return.

---

## 3. Why Normalize

Without normalization, the model may learn:

```text
high-volatility stocks have large future moves
low-volatility stocks have small future moves
```

This is not the desired signal.

The desired signal is:

```text
given this company's own expected movement scale, is the next event-to-event movement directionally strong or weak?
```

---

## 4. Ex-Ante Scale Requirements

`ex_ante_scale_i,t` must satisfy:

1. Known at or before `entry_date_i,t`.
2. Does not use `raw_return_i,t`.
3. Does not use any price after `entry_date_i,t`.
4. Is positive.
5. Is robust to outliers.
6. Has fallback if company history is short.

---

## 5. Scale Method A: Daily Volatility Scaled to Holding Period

### 5.1 Daily Log Returns

For stock `i`:

```text
daily_log_return_i,d = log(adjusted_close_i,d / adjusted_close_i,d-1)
```

Use only dates:

```text
d < entry_date_i,t
```

---

### 5.2 Trailing Daily Volatility

Default lookback:

```yaml
daily_vol_lookback_days: 252
```

Formula:

```text
daily_vol_i,t = std(daily_log_return_i,d for d in trailing 252 trading days before entry_date)
```

---

### 5.3 Holding Period Length

```text
holding_days_i,t = number of trading days from entry_date_i,t to exit_date_i,t inclusive
```

---

### 5.4 Scale to Holding Period

```text
ex_ante_scale_i,t = daily_vol_i,t * sqrt(holding_days_i,t)
```

This approximates expected log return volatility over the holding period.

---

### 5.5 Minimum and Maximum Bounds

Apply:

```yaml
min_ex_ante_scale: 0.02
max_ex_ante_scale: 1.00
```

Meaning:

```text
minimum expected event movement scale = 2%
maximum expected event movement scale = 100%
```

Values outside bounds are clipped.

---

## 6. Scale Method B: Prior Event-to-Event Volatility

Alternative method:

```text
historical_event_return_i,j = log(exit_price_i,j / entry_price_i,j)
```

Use only prior completed events:

```text
j < t
```

Default lookback:

```yaml
event_vol_lookback_events: 8
min_event_history: 4
```

Formula:

```text
event_vol_i,t = robust_std(historical_event_return_i,t-8 ... historical_event_return_i,t-1)
```

Recommended robust standard deviation:

```text
robust_std = 1.4826 * median(abs(x - median(x)))
```

If fewer than 4 prior events exist, fallback to Method A.

---

## 7. Scale Method C: Hybrid Scale

Recommended Version 1:

```text
scale_daily = daily_vol_scaled_to_holding_period
scale_event = prior_event_vol if available
ex_ante_scale = 0.7 * scale_daily + 0.3 * scale_event
```

If `scale_event` unavailable:

```text
ex_ante_scale = scale_daily
```

Config:

```yaml
scale_method: hybrid_daily_event
daily_weight: 0.7
event_weight: 0.3
```

---

## 8. Target Transform

Given:

```text
raw_return_i,t
ex_ante_scale_i,t
k
```

Compute:

```text
standardized_movement_i,t = raw_return_i,t / ex_ante_scale_i,t
```

Then:

```text
target_i,t = tanh(standardized_movement_i,t / k)
```

Equivalent:

```text
target_i,t = tanh(raw_return_i,t / (k * ex_ante_scale_i,t))
```

Default:

```yaml
k: 2.5
```

---

## 9. Interpretation

| Target | Meaning |
|---:|---|
| near +1 | very strong positive movement relative to ex-ante scale |
| near +0.5 | moderately positive movement |
| near 0 | flat or no clear movement |
| near -0.5 | moderately negative movement |
| near -1 | very strong negative movement relative to ex-ante scale |

The model is trained to predict this target, not the raw return.

---

## 10. Inverse Restoration

For portfolio construction:

```text
predicted_signal_i,t = model output in [-1, 1]
```

Clip:

```text
clipped_signal_i,t = clip(predicted_signal_i,t, -0.99, 0.99)
```

Restore:

```text
restored_expected_return_i,t = atanh(clipped_signal_i,t) * k * ex_ante_scale_i,t
```

Where:

```text
atanh(x) = 0.5 * ln((1 + x) / (1 - x))
```

---

## 11. Why Clip Before atanh

`atanh(1)` and `atanh(-1)` are infinite.

Required:

```yaml
restore_clip_abs: 0.99
```

Optional stricter:

```yaml
restore_clip_abs: 0.95
```

---

## 12. Alternative Linear Restoration

For a simpler first implementation:

```text
restored_expected_return_i,t = predicted_signal_i,t * k * ex_ante_scale_i,t
```

This is more stable but not mathematically inverse to `tanh`.

Default Version 1 should use true inverse:

```yaml
restore_method: atanh
```

---

## 13. Outlier Handling

Before target transform, raw returns should be checked.

Do not remove outliers by default, because extreme moves are real investment outcomes.

Instead:

1. Keep raw returns.
2. Let tanh bound the target.
3. Flag extreme raw returns for diagnostics.

Extreme flag:

```text
abs(raw_return_i,t) > 3 * ex_ante_scale_i,t
```

Store:

```text
is_extreme_target_event
```

---

## 14. Required Target Table

```sql
CREATE TABLE event_targets (
    sample_id TEXT PRIMARY KEY,
    raw_log_return DOUBLE PRECISION NOT NULL,
    raw_simple_return DOUBLE PRECISION NOT NULL,
    holding_days INTEGER NOT NULL,
    daily_vol DOUBLE PRECISION,
    event_vol DOUBLE PRECISION,
    ex_ante_scale DOUBLE PRECISION NOT NULL,
    standardized_movement DOUBLE PRECISION NOT NULL,
    target_normalized_signal DOUBLE PRECISION NOT NULL,
    k DOUBLE PRECISION NOT NULL,
    scale_method TEXT NOT NULL,
    is_extreme_target_event BOOLEAN,
    created_at TIMESTAMP
);
```

---

## 15. Leakage Rules

### 15.1 Scale Date Rule

For sample `i,t`:

```text
max(price_date used for scale_i,t) < entry_date_i,t
```

Use strictly less than entry date.

---

### 15.2 Event Scale Rule

Prior event volatility may use only prior completed events:

```text
prior_event.exit_date < entry_date_i,t
```

Do not use the current event's future exit date.

---

### 15.3 Cross-Sectional Fallback Rule

If using universe median scale, compute it using only companies and dates available at that date.

Allowed:

```text
median scale among eligible companies as of entry_date
```

Forbidden:

```text
median scale over full dataset including future test period
```

---

## 16. Fallback Scale Logic

Pseudocode:

```python
def compute_ex_ante_scale(company_id, entry_date, exit_date):
    holding_days = count_trading_days(entry_date, exit_date)

    daily_vol = trailing_daily_vol(
        company_id=company_id,
        end_date=previous_trading_day(entry_date),
        lookback_days=252
    )

    if daily_vol is not None:
        scale_daily = daily_vol * sqrt(holding_days)
    else:
        scale_daily = None

    event_vol = trailing_event_vol(
        company_id=company_id,
        before_entry_date=entry_date,
        lookback_events=8,
        min_events=4
    )

    if scale_daily is not None and event_vol is not None:
        scale = 0.7 * scale_daily + 0.3 * event_vol
    elif scale_daily is not None:
        scale = scale_daily
    elif event_vol is not None:
        scale = event_vol
    else:
        scale = cross_sectional_median_scale(entry_date)

    scale = clip(scale, min_scale, max_scale)
    return scale
```

---

## 17. Target Construction Pseudocode

```python
def build_target(sample):
    raw_return = log(sample.exit_price / sample.entry_price)

    scale = compute_ex_ante_scale(
        company_id=sample.internal_company_id,
        entry_date=sample.entry_date,
        exit_date=sample.exit_date
    )

    if scale <= 0 or is_nan(scale):
        return excluded("missing_ex_ante_scale")

    standardized = raw_return / scale
    target = tanh(standardized / config.k)

    assert -1.0 <= target <= 1.0

    return {
        "raw_log_return": raw_return,
        "ex_ante_scale": scale,
        "standardized_movement": standardized,
        "target_normalized_signal": target
    }
```

---

## 18. Prediction Quality Metrics

During model evaluation, report:

```text
MSE on normalized target
MAE on normalized target
direction accuracy
rank correlation between predicted_signal and realized target
rank correlation between restored_expected_return and raw_return
top-decile realized return
bottom-decile realized return
calibration by prediction bucket
```

Prediction buckets:

```text
[-1.0, -0.8)
[-0.8, -0.6)
...
[0.8, 1.0]
```

---

## 19. Acceptance Criteria

Phase 4 is complete when:

1. Every valid sample has raw log return.
2. Every training sample has ex-ante scale.
3. Every target is within `[-1, 1]`.
4. Scale calculation uses only prices before entry date.
5. Event-vol scale uses only prior completed events.
6. Inverse restoration function is implemented and tested.
7. Target table is saved.
8. Unit tests prove no future prices are used in scale.
9. Target distribution report is generated.

---

## 20. Example Config

```yaml
target:
  return_type: log
  transform: tanh
  k: 2.5

  scale:
    method: hybrid_daily_event
    daily_vol_lookback_days: 252
    event_vol_lookback_events: 8
    min_event_history: 4
    daily_weight: 0.7
    event_weight: 0.3
    min_ex_ante_scale: 0.02
    max_ex_ante_scale: 1.00

  restore:
    method: atanh
    clip_abs: 0.99

  diagnostics:
    extreme_threshold_scale_multiple: 3.0
```

---

## 21. Example Calculation

Input:

```text
entry_price = 100
exit_price = 112
raw_log_return = log(112 / 100) = 0.1133
ex_ante_scale = 0.18
k = 2.5
```

Standardized:

```text
standardized_movement = 0.1133 / 0.18 = 0.6294
```

Target:

```text
target = tanh(0.6294 / 2.5)
target = tanh(0.2518)
target = 0.2466
```

If model predicts:

```text
predicted_signal = 0.25
```

Restored:

```text
restored_expected_return = atanh(0.25) * 2.5 * 0.18
restored_expected_return = 0.2554 * 0.45
restored_expected_return = 0.1149
```

Interpreted as approximately:

```text
expected log return = 11.49%
```
