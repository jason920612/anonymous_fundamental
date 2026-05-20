# RFC-06: Portfolio Construction

Status: Draft  
Phase: 6  
Purpose: Define how model predictions are converted into long-only portfolio weights using restored expected directional movement and signal-weighted risk parity.

---

## 1. Objective

The model outputs only:

```text
predicted_signal ∈ [-1, 1]
```

The portfolio module converts predictions into actual positions.

The module must:

1. Restore expected directional movement.
2. Select long-only candidates.
3. Estimate historical risk using only past data.
4. Convert positive signals into risk budgets.
5. Allocate using risk parity or inverse-vol approximation.
6. Apply constraints.
7. Include transaction costs.
8. Allow cash if not enough positive signals exist.

---

## 2. Inputs

Required:

```text
sample_id
internal_company_id
entry_date
exit_date
predicted_signal
ex_ante_scale
adjusted_close price history
historical returns
optional sector metadata
current portfolio holdings
```

Optional:

```text
probability_positive
downside_quantile_05
mean_generated_signal
prediction_uncertainty
liquidity data
```

---

## 3. Signal Restoration

Given:

```text
predicted_signal_i,t
ex_ante_scale_i,t
k
```

Compute:

```text
clipped_signal_i,t = clip(predicted_signal_i,t, -0.99, 0.99)
restored_expected_return_i,t = atanh(clipped_signal_i,t) * k * ex_ante_scale_i,t
```

Store both:

```text
predicted_signal
restored_expected_return
```

The portfolio module may use either signal or restored return depending on configuration.

---

## 4. Long-Only Positive Signal

Version 1 is long-only.

```text
positive_signal_i,t = max(predicted_signal_i,t, 0)
```

If:

```text
positive_signal_i,t <= 0
```

Then the stock is not a buy candidate.

Existing holdings with negative new signal should be sold according to rebalancing rules.

---

## 5. Event-Driven Portfolio Timeline

Because each company has its own report events, the portfolio is event-driven.

At each trading date `d`:

1. Read all new predictions whose `entry_date == d`.
2. Update candidate status for those companies.
3. Remove positions whose `exit_date <= d` or whose new prediction is negative.
4. Recompute target weights if rebalance is triggered.
5. Execute trades at configured execution price.

Default execution price:

```text
adjusted close on rebalance date
```

This is a research simplification.

---

## 6. Position Validity Window

For each prediction:

```text
valid_from = entry_date
valid_until = exit_date
```

A company can be held only while:

```text
valid_from <= current_date <= valid_until
```

After `exit_date`, the signal expires.

---

## 7. Candidate Selection

### 7.1 Basic Candidate Rule

A stock is a candidate if:

```text
positive_signal > 0
valid signal exists
passes liquidity filter
has risk estimate
has current price
```

---

### 7.2 Threshold Rule

Config:

```yaml
min_positive_signal: 0.05
```

Candidate condition:

```text
positive_signal >= min_positive_signal
```

---

### 7.3 Top-N Rule

Default:

```yaml
target_positions: 50
min_positions: 30
max_positions: 100
```

Selection:

1. Sort candidates by `positive_signal` or `restored_expected_return`.
2. Keep top `max_positions`.
3. If fewer than `min_positions`, allow cash for unused capital.

Recommended sort key Version 1:

```text
positive_signal
```

Alternative:

```text
restored_expected_return / historical_volatility
```

---

## 8. Risk Estimation

### 8.1 Daily Return Series

Use adjusted close:

```text
daily_log_return_i,d = log(adjusted_close_i,d / adjusted_close_i,d-1)
```

Use only:

```text
d < rebalance_date
```

---

### 8.2 Volatility Estimate

Default:

```yaml
risk_lookback_days: 252
```

Formula:

```text
vol_i,d = std(daily_log_return_i over trailing lookback)
```

Minimum:

```yaml
min_daily_vol: 0.005
max_daily_vol: 0.10
```

If missing, stock is excluded or assigned conservative high volatility.

---

### 8.3 Covariance Estimate

For full risk parity:

```text
Σ_d = covariance matrix of candidate daily returns over trailing 252 days
```

Shrinkage recommended:

```text
Σ_shrunk = (1 - λ) * sample_cov + λ * diagonal(sample_cov)
```

Default:

```yaml
covariance_shrinkage_lambda: 0.5
```

If covariance matrix is unstable, fallback to inverse-vol allocation.

---

## 9. Signal to Risk Budget

For each candidate:

```text
risk_budget_raw_i = positive_signal_i ^ p
```

Default:

```yaml
signal_power: 1.0
```

Normalize:

```text
risk_budget_i = risk_budget_raw_i / sum(risk_budget_raw)
```

If using restored return:

```text
risk_budget_raw_i = max(restored_expected_return_i, 0) ^ p
```

Optional downside-aware version:

```text
risk_budget_raw_i = probability_positive_i * positive_signal_i / max(abs(downside_quantile_05_i), epsilon)
```

---

## 10. Allocation Method A: Signal-Weighted Inverse Volatility

This is the recommended first implementation.

Formula:

```text
raw_weight_i = risk_budget_i / vol_i
weight_i = raw_weight_i / sum(raw_weight)
```

Then apply caps.

Advantages:

1. simple,
2. stable,
3. easy to debug,
4. avoids covariance optimizer failures.

---

## 11. Allocation Method B: Signal-Weighted Risk Parity

Use covariance matrix `Σ`.

Portfolio volatility:

```text
portfolio_vol = sqrt(w^T Σ w)
```

Marginal risk contribution:

```text
mrc_i = (Σw)_i / portfolio_vol
```

Risk contribution:

```text
rc_i = w_i * mrc_i
```

Target:

```text
rc_i / sum(rc) ≈ risk_budget_i
```

Optimization:

```text
minimize sum_i (rc_i / sum(rc) - risk_budget_i)^2
subject to:
  sum(w) <= 1
  w_i >= 0
  w_i <= max_single_stock_weight
```

If cash allowed:

```text
sum(w) <= 1
cash_weight = 1 - sum(w)
```

If fully invested:

```text
sum(w) = 1
```

Version 1 default:

```yaml
fully_invested: false
allow_cash: true
```

---

## 12. Constraints

### 12.1 Single Stock Cap

Default:

```yaml
max_single_stock_weight: 0.05
```

No stock can exceed 5%.

---

### 12.2 Minimum Weight

To reduce tiny trades:

```yaml
min_position_weight: 0.0025
```

Positions below 0.25% are set to zero and weights are renormalized.

---

### 12.3 Sector Cap

Optional:

```yaml
use_sector_caps: true
max_sector_weight: 0.30
```

Sector names are allowed in the portfolio module, not in the prediction model.

---

### 12.4 Turnover Control

Optional:

```yaml
max_daily_turnover: 0.25
```

If required trades exceed this, scale trades proportionally.

---

### 12.5 Cash Rule

If fewer than `min_positions` candidates:

```text
allocate to selected candidates subject to caps
remaining weight stays in cash
```

Cash return default:

```yaml
cash_return: 0
```

Later versions may use Treasury bill rate.

---

## 13. Transaction Costs

Default:

```yaml
transaction_cost_bps_per_trade: 10
```

For trade from old weight to new weight:

```text
trade_weight_i = abs(target_weight_i - current_weight_i)
cost_i = trade_weight_i * transaction_cost_rate
```

Total cost:

```text
daily_transaction_cost = sum(cost_i)
```

Portfolio return after cost:

```text
net_return_d = gross_return_d - daily_transaction_cost
```

---

## 14. Rebalance Rules

### 14.1 Daily Event-Driven Rebalance

Default:

```text
rebalance on any trading day with new entry signals or expired positions
```

### 14.2 Weekly Rebalance Alternative

To reduce turnover:

```yaml
rebalance_frequency: weekly
rebalance_day: Friday
```

Version 1 default:

```yaml
rebalance_frequency: daily_event_driven
```

---

## 15. Execution Price

Research Version 1:

```text
trades executed at adjusted close on rebalance date
```

This is simple but optimistic if signals are assumed known before close.

More conservative alternative:

```text
execute at next trading day's adjusted close
```

Recommended final backtest should compare both.

Config:

```yaml
execution:
  version_1: same_day_close
  conservative: next_day_close
```

---

## 16. Portfolio Weight Output Schema

```sql
CREATE TABLE portfolio_weights (
    portfolio_id TEXT,
    date DATE,
    internal_company_id TEXT,
    ticker_at_date TEXT,
    target_weight DOUBLE PRECISION,
    current_weight_before_trade DOUBLE PRECISION,
    trade_weight DOUBLE PRECISION,
    predicted_signal DOUBLE PRECISION,
    restored_expected_return DOUBLE PRECISION,
    risk_budget DOUBLE PRECISION,
    vol_estimate DOUBLE PRECISION,
    allocation_method TEXT,
    created_at TIMESTAMP,
    PRIMARY KEY(portfolio_id, date, internal_company_id)
);
```

Cash row:

```text
internal_company_id = "CASH"
```

---

## 17. Allocation Pseudocode

```python
def build_target_portfolio(date, active_predictions, current_holdings):
    candidates = []

    for pred in active_predictions:
        if pred.valid_from <= date <= pred.valid_until:
            signal = max(pred.predicted_signal, 0)
            if signal >= config.min_positive_signal:
                vol = estimate_vol(pred.company_id, date)
                if vol is not None:
                    restored = restore_return(pred.predicted_signal, pred.ex_ante_scale)
                    candidates.append({
                        "company_id": pred.company_id,
                        "signal": signal,
                        "restored_return": restored,
                        "vol": vol
                    })

    candidates = sort_and_keep_top_n(candidates)

    if len(candidates) < config.min_positions:
        allow_cash = True

    risk_budget_raw = [c.signal ** config.signal_power for c in candidates]
    risk_budget = normalize(risk_budget_raw)

    if config.allocation_method == "inverse_vol":
        raw_weights = [rb / c.vol for rb, c in zip(risk_budget, candidates)]
        weights = normalize(raw_weights)
    else:
        cov = estimate_covariance(candidates, date)
        weights = solve_risk_parity(cov, risk_budget, constraints)

    weights = apply_single_stock_cap(weights)
    weights = apply_sector_caps(weights)
    weights = remove_tiny_positions(weights)
    weights = renormalize_or_leave_cash(weights)

    return weights
```

---

## 18. Diagnostics

Report each rebalance:

```text
date
num_candidates
num_positions
cash_weight
largest_position
top_10_weight
sector_weights
expected_portfolio_return
estimated_portfolio_vol
turnover
transaction_cost
```

---

## 19. Acceptance Criteria

Phase 6 is complete when:

1. Predictions are restored outside the model.
2. Negative signals are not bought in long-only mode.
3. Candidate selection is deterministic.
4. Inverse-vol allocation works.
5. Full risk parity either works or falls back safely.
6. Single-stock caps are enforced.
7. Optional sector caps are enforced if enabled.
8. Transaction costs are calculated.
9. Cash is handled explicitly.
10. Portfolio weight files are saved.
11. Unit tests verify weights sum to <= 1 and constraints hold.

---

## 20. Example Config

```yaml
portfolio:
  long_only: true
  allow_cash: true
  leverage: 1.0

  signal:
    min_positive_signal: 0.05
    use_restored_return_for_sorting: false
    signal_power: 1.0

  positions:
    min_positions: 30
    target_positions: 50
    max_positions: 100
    max_single_stock_weight: 0.05
    min_position_weight: 0.0025

  risk:
    allocation_method: inverse_vol
    risk_lookback_days: 252
    min_daily_vol: 0.005
    max_daily_vol: 0.10
    covariance_shrinkage_lambda: 0.5

  sector:
    use_sector_caps: true
    max_sector_weight: 0.30

  rebalance:
    frequency: daily_event_driven
    execution_price: same_day_close

  costs:
    transaction_cost_bps_per_trade: 10
```
