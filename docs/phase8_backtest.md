# Phase 8 — Backtest Engine

Maps to **RFC-07** and **RFC-09 Milestone 8**.

## Modules

| Module | Role |
|---|---|
| `afp.backtest.engine` | `run_backtest(predictions, prices, calendar, ...)` — daily loop |
| `afp.backtest.metrics` | Sharpe, Sortino, Calmar, max drawdown, turnover, summary |
| `afp.backtest.benchmarks` | SPY/QQQ daily returns + equal-weight / equal-risk universe series |
| `afp.backtest.report` | Combine strategy + benchmarks into a metrics summary CSV |

## Engine Loop (RFC-07 §3)

Per trading day `d`:

1. **Weight drift** — apply yesterday-to-today simple returns to each held name.
   The portfolio is renormalized so weights + cash sum to 1.
2. **Rebalance trigger** — fires if `d` is in the set of entry/exit dates derived
   from `predictions_with_context`, or if there are no current holdings.
3. **Target weights** — `build_target_portfolio(d)` resolves active predictions,
   pulls vol estimates strictly from past returns, selects long-only candidates,
   and runs `inverse_vol_allocation`.
4. **Trades** — `trade_weight = target - current`. Cost = `|trade_weight| * tcost_rate`.
   Reason classified as `new_signal`, `exit`, or `rebalance`.
5. **Net return** — `gross_return - cost`. Portfolio value compounds at the net rate.
6. **State persisted** — daily row, per-name holding rows, trade rows.

`cash_return_daily` is configurable (default 0, RFC-07 §6).

## Outputs

```
result.daily      date, gross_return, transaction_cost_return, net_return,
                  portfolio_value, cash_weight, n_positions, turnover
result.holdings   date, internal_company_id, weight
result.trades     date, internal_company_id, old_weight, new_weight,
                  trade_weight, trade_cost, reason
```

`build_report` adds:

- `summary` per strategy/benchmark (`annualized_return / volatility / sharpe /
  sortino / max_drawdown / calmar / n_days / annualized_turnover`)
- benchmark series for SPY, QQQ, equal-weight universe, equal-risk universe

`write_report` materializes `reports/backtest/<portfolio_id>/metrics.csv`.

## Benchmarks (RFC-07 §7)

- **SPY/QQQ**: daily pct-change on `adjusted_close`.
- **Equal-weight universe**: pivot prices to a daily matrix, ffill, mean of
  pct-change across all symbols on each day.
- **Equal-risk universe**: same matrix, weighted by `1 / trailing_252d_std`
  shifted one day (past-only).

## Acceptance — verified by `tests/test_backtest.py`

1. Annualized return formula matches `(prod(1+r))^(252/n) - 1`.
2. Max drawdown matches direct `(cumprod / cummax - 1).min()`.
3. Sharpe degenerates to 0 when volatility is 0.
4. `summary` exposes all RFC-07 §8 metrics.
5. End-to-end run on the synthetic world produces strictly positive
   `portfolio_value` and cash_weight ∈ [0, 1].
6. `build_report` exposes SPY, equal-weight, and equal-risk benchmark blocks.
