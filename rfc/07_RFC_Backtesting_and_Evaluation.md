# RFC-07: Backtesting and Evaluation

Status: Draft  
Phase: 7  
Purpose: Define the event-driven backtest engine, accounting rules, benchmark comparison, risk metrics, and reporting requirements.

---

## 1. Objective

Evaluate whether the model-driven portfolio improves performance and risk relative to benchmarks after transaction costs.

The backtest must measure:

1. portfolio return,
2. volatility,
3. maximum drawdown,
4. Sharpe ratio,
5. Sortino ratio,
6. Calmar ratio,
7. turnover,
8. transaction-cost-adjusted return,
9. benchmark-relative performance,
10. event-level predictive quality.

---

## 2. Backtest Type

The strategy is event-driven.

Predictions become valid on:

```text
entry_date
```

Predictions expire on:

```text
exit_date
```

Portfolio can rebalance when:

1. new predictions become valid,
2. old predictions expire,
3. configured rebalance schedule occurs,
4. risk constraints force change.

---

## 3. Backtest Timeline

For each trading date `d`:

1. Load current holdings from previous close.
2. Apply corporate-action-adjusted daily returns.
3. Identify new valid signals.
4. Identify expired signals.
5. Generate target weights if rebalance is triggered.
6. Calculate trades.
7. Apply transaction costs.
8. Save portfolio state.
9. Save daily performance.

Version 1 assumes trades execute at adjusted close.

---

## 4. Daily Portfolio Return

For date `d`:

```text
gross_return_d = sum_i weight_i,d-1 * asset_return_i,d
```

Where:

```text
asset_return_i,d = adjusted_close_i,d / adjusted_close_i,d-1 - 1
```

Transaction cost:

```text
cost_d = sum_i abs(target_weight_i,d - pre_trade_weight_i,d) * transaction_cost_rate
```

Net return:

```text
net_return_d = gross_return_d - cost_d
```

Portfolio value:

```text
portfolio_value_d = portfolio_value_d-1 * (1 + net_return_d)
```

---

## 5. Weight Drift

Between rebalances, weights drift with returns.

For each holding:

```text
value_i,d = value_i,d-1 * (1 + asset_return_i,d)
```

Portfolio total:

```text
portfolio_value_d = cash_value_d + sum_i value_i,d
```

Weight:

```text
weight_i,d = value_i,d / portfolio_value_d
```

---

## 6. Cash

Default cash return:

```text
cash_return_d = 0
```

Optional later:

```text
cash_return_d = daily Treasury bill proxy
```

Cash is allowed when:

1. insufficient positive candidates,
2. caps prevent full investment,
3. risk model rejects unstable allocation.

---

## 7. Benchmark Returns

Required benchmarks:

```text
SPY adjusted close
QQQ adjusted close
equal-weight model universe
equal-risk model universe
```

Daily benchmark return:

```text
benchmark_return_d = benchmark_adjusted_close_d / benchmark_adjusted_close_d-1 - 1
```

For equal-weight universe:

1. Use same eligible model universe.
2. Rebalance on same dates as strategy or monthly.
3. Apply same transaction cost assumption if positions change.

For equal-risk universe:

1. Use same eligible model universe.
2. Allocate inverse-vol or risk parity without model signal.
3. Apply same constraints where possible.

---

## 8. Performance Metrics

### 8.1 Annualized Return

Given daily returns:

```text
annualized_return = (final_value / initial_value) ^ (252 / num_trading_days) - 1
```

---

### 8.2 Annualized Volatility

```text
annualized_volatility = std(daily_returns) * sqrt(252)
```

---

### 8.3 Sharpe Ratio

Default risk-free rate:

```text
0
```

Formula:

```text
sharpe = annualized_return / annualized_volatility
```

If risk-free series is added:

```text
sharpe = annualized_excess_return / annualized_volatility
```

---

### 8.4 Sortino Ratio

Downside daily returns:

```text
downside_returns = min(daily_return, 0)
```

Annualized downside deviation:

```text
downside_deviation = std(downside_returns) * sqrt(252)
```

Sortino:

```text
sortino = annualized_return / downside_deviation
```

---

### 8.5 Maximum Drawdown

Portfolio cumulative value:

```text
V_d
```

Running peak:

```text
P_d = max(V_0 ... V_d)
```

Drawdown:

```text
DD_d = V_d / P_d - 1
```

Maximum drawdown:

```text
max_drawdown = min(DD_d)
```

---

### 8.6 Calmar Ratio

```text
calmar = annualized_return / abs(max_drawdown)
```

---

### 8.7 Turnover

Daily turnover:

```text
turnover_d = 0.5 * sum_i abs(target_weight_i,d - pre_trade_weight_i,d)
```

Annualized turnover:

```text
annualized_turnover = mean(daily_turnover) * 252
```

---

### 8.8 Hit Rate vs Benchmark

Monthly or event-period hit rate:

```text
hit_rate_spy = count(strategy_return_period > spy_return_period) / count(periods)
hit_rate_qqq = count(strategy_return_period > qqq_return_period) / count(periods)
```

---

### 8.9 Rolling 12-Month Excess Return

For each date `d`:

```text
strategy_12m_return_d = product(1 + strategy_daily_returns over last 252 days) - 1
benchmark_12m_return_d = product(1 + benchmark_daily_returns over last 252 days) - 1
rolling_excess_return_d = strategy_12m_return_d - benchmark_12m_return_d
```

Report:

```text
mean
median
min
max
percentage positive
```

---

## 9. Event-Level Evaluation

For each prediction sample:

```text
predicted_signal
target_normalized_signal
raw_log_return
restored_expected_return
```

Report:

```text
direction_accuracy = sign(predicted_signal) == sign(raw_log_return)
spearman_rank_correlation
top_decile_realized_return
bottom_decile_realized_return
top_minus_bottom_spread
```

Long-only candidate report:

```text
mean realized return for predicted_signal > 0
mean realized return for predicted_signal <= 0
```

---

## 10. Portfolio Attribution

Daily attribution:

```text
position contribution_i,d = weight_i,d-1 * asset_return_i,d
```

Aggregate by:

```text
stock
sector if available
entry cohort
signal bucket
holding age
```

Required reports:

1. top contributors,
2. worst contributors,
3. sector contribution,
4. contribution by signal decile,
5. drawdown-period attribution.

---

## 11. Drawdown Analysis

For each drawdown episode:

```text
peak_date
trough_date
recovery_date
drawdown_depth
days_to_trough
days_to_recovery
portfolio_return
spy_return
qqq_return
largest_losers
sector_exposures
```

---

## 12. Robustness Tests

Required:

1. transaction cost sensitivity:
   - 0 bps
   - 5 bps
   - 10 bps
   - 25 bps
   - 50 bps
2. execution delay:
   - same-day close
   - next-day close
3. max weight sensitivity:
   - 2%
   - 5%
   - 8%
4. position count:
   - top 30
   - top 50
   - top 100
5. signal threshold:
   - 0.00
   - 0.05
   - 0.10
6. benchmark comparison:
   - SPY
   - QQQ
   - equal-weight universe
   - equal-risk universe

---

## 13. Backtest Output Tables

### 13.1 Daily Returns

```sql
CREATE TABLE backtest_daily_returns (
    portfolio_id TEXT,
    date DATE,
    gross_return DOUBLE PRECISION,
    transaction_cost_return DOUBLE PRECISION,
    net_return DOUBLE PRECISION,
    portfolio_value DOUBLE PRECISION,
    cash_weight DOUBLE PRECISION,
    num_positions INTEGER,
    turnover DOUBLE PRECISION,
    spy_return DOUBLE PRECISION,
    qqq_return DOUBLE PRECISION,
    created_at TIMESTAMP,
    PRIMARY KEY(portfolio_id, date)
);
```

---

### 13.2 Trades

```sql
CREATE TABLE backtest_trades (
    portfolio_id TEXT,
    date DATE,
    internal_company_id TEXT,
    ticker_at_date TEXT,
    old_weight DOUBLE PRECISION,
    new_weight DOUBLE PRECISION,
    trade_weight DOUBLE PRECISION,
    trade_cost DOUBLE PRECISION,
    trade_reason TEXT,
    created_at TIMESTAMP
);
```

Allowed trade reasons:

```text
new_signal
expired_signal
negative_signal
rebalance
risk_cap
sector_cap
turnover_control
```

---

### 13.3 Holdings

```sql
CREATE TABLE backtest_holdings (
    portfolio_id TEXT,
    date DATE,
    internal_company_id TEXT,
    ticker_at_date TEXT,
    weight DOUBLE PRECISION,
    market_value DOUBLE PRECISION,
    predicted_signal DOUBLE PRECISION,
    restored_expected_return DOUBLE PRECISION,
    entry_date DATE,
    exit_date DATE,
    holding_age_days INTEGER,
    created_at TIMESTAMP,
    PRIMARY KEY(portfolio_id, date, internal_company_id)
);
```

---

## 14. Required Charts

Generate:

1. cumulative equity curve vs SPY and QQQ,
2. drawdown chart,
3. rolling 12-month return,
4. rolling 12-month excess return vs SPY,
5. rolling volatility,
6. monthly return heatmap,
7. turnover over time,
8. number of positions over time,
9. cash weight over time,
10. predicted signal bucket realized returns.

---

## 15. Required Report Format

Output:

```text
reports/backtest/{portfolio_id}/summary.md
reports/backtest/{portfolio_id}/metrics.csv
reports/backtest/{portfolio_id}/daily_returns.csv
reports/backtest/{portfolio_id}/trades.csv
reports/backtest/{portfolio_id}/holdings.csv
reports/backtest/{portfolio_id}/charts/
```

---

## 16. Backtest Acceptance Criteria

Phase 7 is complete when:

1. Daily portfolio returns are produced.
2. Transaction costs are included.
3. Portfolio constraints are reflected in holdings.
4. Benchmarks are aligned.
5. Performance metrics are calculated.
6. Drawdown is calculated.
7. Turnover is calculated.
8. Event-level prediction diagnostics are reported.
9. Robustness tests are run.
10. Output report is generated.

---

## 17. Minimum Strategy Approval Rule

A model portfolio should not be considered useful unless it satisfies at least the following on out-of-sample test data:

```text
Sharpe ratio > equal-weight universe Sharpe
max drawdown < equal-weight universe max drawdown
net return after costs > SPY or QQQ over at least one complete test window
rolling 12-month excess return positive in more than 50% of windows
turnover not excessive relative to expected transaction costs
```

This rule does not mean the model is production-ready. It only means the research result is worth deeper study.

---

## 18. Example CLI

```bash
python -m src.backtest.run \
  --predictions artifacts/predictions/lgbm_v001.parquet \
  --samples data/processed/event_samples.parquet \
  --prices data/processed/prices_daily.parquet \
  --config configs/backtest.yaml \
  --output reports/backtest/lgbm_v001_inverse_vol
```

---

## 19. Example Summary

```text
Portfolio: lgbm_v001_signal_inverse_vol
Test period: 2020-01-01 to 2025-12-31
Annualized return: 14.2%
Annualized volatility: 18.5%
Sharpe: 0.77
Max drawdown: -24.8%
Calmar: 0.57
Turnover: 3.1x annualized
Transaction cost drag: -0.9% annualized
SPY annualized return: 12.1%
QQQ annualized return: 16.4%
Rolling 12m excess vs SPY positive: 57%
Rolling 12m excess vs QQQ positive: 43%
```
