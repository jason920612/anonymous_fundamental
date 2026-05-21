# Phase 43 — Tail-aware (Expected Shortfall) allocator (candidate)

> Status: code + tests landed; **not yet evaluated**. Same restraint
> as Phase 42 — wait for Phase 41 leaderboard before spending compute
> on a separate evaluation.

## Why ES, not σ

Standard inverse-vol weighting prices risk as Gaussian standard
deviation. But equity returns are not Gaussian — power-law tails are
well documented (Mandelbrot 1963 cotton prices; Gopikrishnan-Plerou-
Stanley 1998 econophysics on NYSE+Nasdaq returns; degrees-of-freedom
estimates `ν ∈ [3, 4]`). The variance is finite for `ν > 2` but
fourth moment diverges — meaning σ *systematically under-prices the
tail* and inverse-σ weighting therefore over-allocates to heavy-tail
names.

Expected Shortfall (= CVaR) at confidence level `α`:

```
ES_α = -E[ r | r ≤ -VaR_α ]
```

is a coherent risk measure (Artzner-Delbaen-Eber-Heath 1999) and
naturally captures the *magnitude* of left-tail events. Weighting by
`signal / ES` instead of `signal / σ` reallocates risk away from
fat-tailed names without requiring any parametric assumption.

## Implementation

- `src/afp/portfolio/tail_aware_allocator.py`:
  - `expected_shortfall(returns, alpha=0.05)` — empirical CVaR.
  - `tail_risk_weights(es, signal)` — simplex projection of
    `signal / ES`.
  - `build_tail_aware_target_portfolio(...)` — drop-in allocator.

- `tests/test_tail_aware_allocator.py`:
  - ES positive on a centered Gaussian.
  - Student-t (df=3) has larger ES than Gaussian on same scale.
  - Weight ranking respects `signal / ES`.
  - Zero-signal degenerates to equal weight.

## Methodology discipline

- α=0.05 is the textbook ES threshold (Basel III risk-management
  default). Not searched on val/test.
- Lookback 252d (same as VolEstimator default).
- No parametric tail assumption — purely empirical ES on the
  historical return panel.

## Status

Evaluated as part of the v2 run.

## Result

**tail_aware: Sharpe 1.019, MDD -0.427, Calmar 0.646**

Essentially identical to baseline (1.018 / -0.431). The empirical
ES distribution across our universe is too similar to the std
distribution at this lookback (252d, α=0.05) to differentiate
weights meaningfully. The Mandelbrot/Stanley heavy-tail story is
real but the difference is concentrated in extreme single-day
events, not in the bulk of weight-determining risk.

We retire ES weighting for this pipeline. A more aggressive form
(rolling ES with shorter window, ES on event-window returns) might
matter, but tuning that would require val/test peek and we will
not do it.
