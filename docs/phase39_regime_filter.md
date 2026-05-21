# Phase 39 — Thermodynamic Regime Detector

> Status: causal, theory-driven; one knob (kT=2.0) set a priori.

A cross-sectional exposure scaler that treats the **dispersion of
single-day stock returns** as a temperature observable, and damps
gross exposure exponentially when the universe enters a "cold"
(correlated) regime.

## Statistical-mechanics analogy

Treat the universe of N stocks on day `t` as a microcanonical system
whose microstate is the return vector `{r_i(t)}`. Define

```
T(t) = std_i { r_i(t) }              (cross-sectional dispersion)
```

The interpretation is direct:

- **Hot regime** (high T): many independent dimensions of variation
  active; stock-specific information drives prices apart. Selection
  alpha is meaningful.
- **Cold regime** (low T): eigenspectrum collapses onto a single
  market mode (think: 2008 crisis, COVID March 2020). Stock picks
  behave like a leveraged beta-1 with extra idiosyncratic risk.
  Selection breadth → 0.

A long-only strategy whose alpha source is **cross-sectional ranking
of fundamentals** has no edge in a cold regime. It should reduce
exposure proportionally.

## Boltzmann damping

The exposure scale is set by a Boltzmann-style factor with one
theoretical parameter `kT`:

```
z(t)     = (T(t) - μ_window) / σ_window         (causal rolling stats)
scale(t) = clip( exp(z / kT), floor, 1 )
```

With `kT = 2.0`:

- `z = 0` (typical): scale = 1.0
- `z = -1` (1σ cold): scale = exp(-0.5) ≈ 0.61
- `z = -2` (2σ cold): scale = exp(-1.0) ≈ 0.37 (most likely floored)

The rolling window is 252 trading days (≈1 year). This is the
**only** memory; no peek at future dates. The floor of 30% is a
hard-coded safety against fully de-risking on a single panic day.

## Methodology discipline

This phase has exactly one numerical knob: `kT`. We set it from the
following theoretical argument:

- A 1σ regime change is "noteworthy"; should produce a *modest* (~40%)
  exposure reduction, not a total flight to cash.
- A 2σ change is "extreme"; should fall to the floor.
- That uniquely picks `kT ≈ 2.0` given the `exp` damping form.

We do not run a sweep, do not check whether `kT=1.5` or `kT=3.0`
yields a better val/test Sharpe, and do not iterate. The
single-evaluation rule binds the same way as Phase 38.

## Integration

The backtest engine accepts two new optional kwargs:

```python
regime_filter_cfg : RegimeFilterConfig(enabled, lookback_days, kT, ...)
regime_temperature_series : pd.Series indexed by date
```

When enabled, after each rebalance the target weights are multiplied
by the daily `scale(t)`; the remainder rolls to cash. Compatible with
**every** allocator (baseline, distribution, blended, RMT, max-ent).

## Implementation

- `src/afp/portfolio/regime_filter.py` — config + temperature
  computation + causal z-score + clipping.
- `tests/test_regime_filter.py` — 4 unit tests covering:
  hot/cold dispersion ordering; disabled = identity; cold tail
  scaled down vs hot tail; clip respects floor/ceiling.

## Predicted benefit

Both axes simultaneously:

- **Sharpe**: equal exposure during "selection-favorable" regimes,
  reduced during "everything-correlated" regimes → cleaner per-trade
  Sharpe.
- **MDD**: during regime collapses (2008, 2020 March, 2022), the
  filter automatically reduces gross before the model's signal has
  time to react.
