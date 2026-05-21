# Phase 40 — Maximum-Entropy Allocator (Jaynes / Bera-Park)

> Status: parameter-free; β set structurally by `target_positions`.

A Boltzmann/Gibbs allocator that maximizes Shannon entropy of the
portfolio weights subject to a soft signal constraint. Naturally
produces well-diversified long-only allocations whose effective
position count matches the operator's pre-declared
`target_positions`.

## Theory

Jaynes (1957) maximum-entropy principle: given partial information
(here: a return constraint), the least-biased distribution consistent
with the constraint is the one with maximum Shannon entropy.

For portfolio weights:

```
maximize   H(w) = -Σ w_i log w_i
subject to Σ w_i = 1,  Σ w_i μ_i ≥ μ_target,  w_i ≥ 0
```

The Lagrange-multiplier solution is closed-form:

```
w_i  =  exp(β · μ_i) / Σ_j exp(β · μ_j)
```

which is exactly the **Boltzmann distribution** with inverse
temperature β. As β→0 the allocation becomes equal-weight (maximum
entropy); as β→∞ the allocation becomes winner-takes-all (minimum
entropy under the simplex constraint).

### Connection to the rest of the system

`signal_tilted_min_variance` (Phase 38) solves `max wᵀμ - γ/2 wᵀΣw`.
That trades return against **variance** but requires a covariance
estimate. The max-entropy formulation trades return against
**concentration** and needs only the signal vector. So Phase 40 is
strictly orthogonal to Phase 38: they reduce different sources of
fragility.

## β by structural calibration

We pick β NOT by val/test fitting but by a structural identity tied
to `PortfolioConfig.target_positions` (default 50):

```
N_eff(w) = 1 / Σ w_i²      ← Herfindahl effective count
β chosen so N_eff = target_positions
```

`target_positions` is a portfolio-design parameter that already
existed in `PortfolioConfig` long before any prediction was made. It
encodes the operator's tolerance for concentration, not a fit to
data. We use binary search at allocation time (40 iterations,
tolerance 0.5 positions).

This means the allocator's *only* numerical knob is one that has
**always** been part of the public config; nothing new is tuned.

## Inverse-vol blend (theoretical mid-point)

There is one optional `inverse_vol_blend ∈ [0, 1]` parameter that
replaces the raw signal with

```
μ_i^eff = μ_i − blend · log(σ_i / σ_median)
```

before the softmax. `blend=0` is pure max-entropy on signal; `blend=1`
is max-entropy on a *vol-adjusted signal*. We pick `blend = 0.5` as
the theoretical mid-point in our default driver — a 50/50 mix is the
"agnostic" choice — and report the `blend=1.0` variant too for
comparison. Again, no sweep.

## Implementation

- `src/afp/portfolio/max_entropy_allocator.py` — softmax weights,
  binary search for β, plus the same caps/sector/cash rules used by
  `inverse_vol_allocation`.
- `tests/test_max_entropy_allocator.py` — 4 unit tests:
  uniform signal → equal weights; higher signal → higher weight;
  N_eff matches target; empty input → 100% cash.

## Methodology discipline

- No val/test peek: β is set by binary search per-allocation to match
  a config parameter, not by sweep.
- `inverse_vol_blend = 0.5` is the theoretical mid-point.
- Single evaluation in Phase 41.

## Predicted benefit

- **MDD**: by structurally limiting concentration to `target_positions`
  effective names, single-stock blowups have a bounded impact.
- **Sharpe**: a near-equal-weight tilt over the top signals is a known
  diversification benefit when the signal has positive cross-sectional
  rank correlation but noisy magnitudes — and our LambdaRank model is
  exactly that.
