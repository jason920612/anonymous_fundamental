# Phase 7 — Portfolio Construction

Maps to **RFC-06** and **RFC-09 Milestone 7**.

## Pipeline

```
predictions_with_context           # sample_id, internal_company_id,
       │                           # entry_date, exit_date, predicted_signal, ex_ante_scale
       ▼
add_restored_returns               # restored_expected_return = atanh(clip(sig)) * k * scale
       │
       ▼
active_predictions(as_of)          # keep rows where entry ≤ as_of ≤ exit
       │
       ▼
attach vol_estimate via VolEstimator.vol_at(icid, as_of)   # past-only window
       │
       ▼
select_candidates                  # positive_signal ≥ min_positive_signal, top-N
       │
       ▼
inverse_vol_allocation             # signal-weighted inverse-vol + caps + cash
```

## Risk Model (`afp.portfolio.risk_models.VolEstimator`)

- Indexes daily log returns per company at construction time.
- `vol_at(icid, asof)` uses `daily_returns.loc[idx < asof]` — strict past-only.
- Clips returned vol to `[min_daily_vol, max_daily_vol]` (default 0.5%-10% daily).
- `covariance(ids, asof)` shrinks the sample covariance to its diagonal with
  `(1-λ) * Σ + λ * diag(Σ)`. Used by the optional full risk-parity allocator
  (Milestone 11) — Version 1 uses inverse-vol only.

## Allocation (`inverse_vol_allocation`)

```
risk_budget_i = positive_signal_i ** signal_power
risk_budget_i = risk_budget_i / Σ risk_budget
raw_weight_i  = risk_budget_i / vol_i
weight_i      = raw_weight_i / Σ raw_weight

# Constraints (ordered):
1. drop weight_i < min_position_weight (so cap redistribution doesn't inflate tiny names)
2. renormalize survivors
3. iterative single-stock cap at max_single_stock_weight (excess flows to non-capped)
4. if Σ weight > leverage → scale down
5. cash_weight = leverage - Σ weight   (if allow_cash; otherwise force fully invested
   by scaling up and re-capping)
```

Returns `AllocationResult(weights, cash_weight, n_candidates)`.

## Long-Only Rule

`positive_signal = max(predicted_signal, 0)`. Names with `predicted_signal ≤ 0`
never enter the candidate set. Held positions with a fresh non-positive signal
are exited on the next rebalance — that exit logic lives in the backtest engine
(Phase 8), not here.

## Cash Rule

If fewer than `min_positions` candidates pass the threshold, cash absorbs the
remaining budget. If `allow_cash=False`, the allocator scales weights up to the
leverage target and re-caps, which can leave residual cash only when caps bind.

## Acceptance — verified by `tests/test_portfolio.py`

1. Negative-signal and below-threshold names are excluded from candidates.
2. Weights respect `max_single_stock_weight` and `Σ weight ≤ leverage`.
3. With only one candidate hitting the per-name cap, cash absorbs the rest.
4. Tiny per-name weights are dropped (not inflated by cap redistribution).
5. `add_restored_returns` matches `atanh(signal) * k * scale` exactly.
