# Phase 13 — Cross-Sectional Rank Target

## Motivation

Phase 11 diagnostics showed Spearman ≈ 0.03 and direction accuracy 50.4% for
the `tanh_scaled` magnitude target — essentially noise. The diagnosis isn't
that the model can't learn; it's that **predicting absolute normalized
magnitude per company is dominated by idiosyncratic noise that no fundamental
signal can capture at single-firm horizons**.

A pragmatic alternative supported by the RFC (§10 of RFC-02 lists optional
target modes — `absolute_log_return`, `excess_vs_spy`, `excess_vs_qqq`) is to
predict the **same-quarter cross-sectional rank** of the realized log return.
This decouples the prediction problem from market-wide noise and reduces
single-firm variance.

## Module

`afp.targets.target_transform.apply(samples, TargetConfig(target_mode=...))`

```python
@dataclass
class TargetConfig:
    k: float = 2.5
    target_mode: str = "tanh_scaled"          # or "cross_sectional_rank"
    rank_period: str = "Q"                    # pandas .dt.to_period code
    ...
```

When `target_mode = cross_sectional_rank`:

```
within entry_date.to_period(rank_period):
    target = 2 * pct_rank(raw_log_return, method="average") - 1   ∈ [-1, 1]
```

The `ex_ante_scale` is still computed and persisted — the portfolio module
still needs it for position sizing via `atanh()` restoration.

## RFC Compliance

- Target is still in `[-1, 1]` (RFC-04 §8 contract).
- No look-ahead: `pct_rank` is computed only across samples with the same
  `entry_date.period`. Each sample's quarter cohort is known at sample
  construction time. (The cohort *itself* is observable in real-time —
  ranking a company against others reporting in the same quarter does not
  use future data.)
- The encoder, splits, and downstream code see no changes; the only
  difference is the values in the `target_normalized_signal` column.

## Why Rank Helps

1. **Noise reduction.** Realized log return at single-firm 60-90d horizons
   has σ ≈ 15-20%. Rank is bounded and (in expectation) cohort-stationary.
2. **Easier learning surface.** LightGBM/Ridge on a quasi-uniform target
   is more stable than on a heavy-tailed regression target.
3. **Directly aligns with the portfolio module**, which only uses the *sign*
   and *order* of the signal — magnitude is restored from `ex_ante_scale`.

## How To Activate

```yaml
target:
  target_mode: cross_sectional_rank
  rank_period: Q                   # Q (default), M, A, W, etc.
```

The default remains `tanh_scaled` (RFC-04 §8) so existing experiments are
unchanged.

## Acceptance — `tests/test_rank_target.py`

- Output is in `[-1, 1]` for every eligible sample.
- Within a quarter, higher `raw_log_return` ⇒ monotonically higher rank target.
- Per-quarter mean rank target ≈ 0 (uniform pct_rank centered at 0).
- `tanh_scaled` default behavior is unchanged (regression-on-tanh test still
  passes).
