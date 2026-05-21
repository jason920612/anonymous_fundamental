# Phase 38 — RMT Covariance Cleaning (Marchenko-Pastur)

> Status: parameter-free; no val/test tuning.

A new portfolio allocator that replaces the simple per-asset
inverse-vol with a Markowitz-style signal-tilted min-variance solve
against a **Marchenko-Pastur denoised covariance matrix**.

## Why this might help

Diagnosis: the production v3 portfolio runs at Sharpe ≈ 1.02 and
MDD ≈ -43%, while the *equal-risk-active-universe* benchmark on the
same prediction set achieves Sharpe 1.44 and MDD -18%. The model has
predictive content but the **risk model is naive**: vol is per-asset
only, so portfolio variance ignores correlation structure entirely.
RMT cleaning is a physics-grade fix that adds correlation structure
without overfitting the sample.

## Theory

For an N×N sample covariance Σ̂ built from T iid Gaussian return
observations, Marchenko-Pastur (1967) shows the eigenvalue spectrum is
asymptotically supported on `[(1-√q)², (1+√q)²]` with `q = N/T`. Any
eigenvalue inside this window is *statistically indistinguishable from
noise*. Laloux/Cizeau/Bouchaud/Potters (1999) showed that on US
equities, only the top few eigenvalues escape this band; the rest are
pure noise that contaminates Σ̂⁻¹ and inflates Markowitz weights into
the noise eigendirections.

**The recipe** (standard form):

1. Compute sample correlation matrix `C` from T×N returns.
2. Eigendecompose: `C = V Λ Vᵀ`.
3. Identify `keep` eigenvalues: those above `λ_+ = (1+√q)²`, **plus**
   the largest eigenvalue (market mode) always.
4. Replace every other eigenvalue with their mean (preserves
   trace(C) = N).
5. Reconstruct, rescale to unit diagonal, lift to covariance via
   `Σ_clean = D · C_clean · D` where `D = diag(σᵢ)`.

This is fully **parameter-free**. `q = N/T` is data-determined; no
threshold to tune.

## Allocator

`afp.portfolio.rmt_allocator.rmt_min_var_allocation` is a drop-in
substitute for `inverse_vol_allocation`:

```python
build_rmt_target_portfolio(
    as_of, predictions, vol_estimator, portfolio_cfg,
    k=2.5, sector_map=...,
    lookback_days=252, risk_aversion=5.0,
)
```

Internally:

1. Pull T=252 daily log-returns for each candidate from the
   `VolEstimator` cache.
2. Build `Σ_clean = clean_covariance_rmt(returns_panel)`.
3. Solve `max wᵀμ - (γ/2) wᵀΣw` subject to `w≥0`, `Σw=1` via
   projected gradient. `γ=5.0` is the textbook unit-scale choice and
   is **not** searched.
4. Apply existing single-stock / sector caps and cash rule.

When there are too few price observations to build a covariance
(early backtest days, new universe additions), the allocator silently
falls back to `inverse_vol_allocation`.

## Methodology discipline

- The M-P bound `λ_+` is a closed-form function of `q = N/T` — there
  is nothing to tune on validation or test data.
- `γ=5.0` was chosen *a priori* from the convex-optimization
  literature; we did not sweep it.
- The keep-market-mode convention is the standard recipe.
- Single evaluation: we register this allocator and run the
  Phase-41 leaderboard once, then stop. No iterative refinement.

## Implementation

- `src/afp/portfolio/rmt_covariance.py` — pure-math primitives:
  M-P bounds, correlation cleaning, min-variance solve.
- `src/afp/portfolio/rmt_allocator.py` — drop-in allocator that
  composes the M-P cleaner with the existing caps + cash rule.
- `tests/test_rmt_covariance.py` — 5 unit tests:
  bounds; pure-noise spectrum collapses; market-factor mode survives;
  weights are simplex-valid; signal tilting respects rank order.

## Integration

Backtest engine accepts a generic `custom_allocator=` callback
(extension added in this phase). Existing dispatch (inverse-vol,
distribution, blended) is untouched.
