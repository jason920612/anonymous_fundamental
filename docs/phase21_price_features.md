# Phase 21 — Anonymous Price-Derived Features

## Why

Phase 17 ablation showed every model variant on pure fundamentals
converged to Sharpe ≤ 0.76. The user authorized lifting V1 constraints
(RFC-00 §11 "price_momentum_in_model: false") as long as model input
remains anonymous. Phase 21 adds **trailing-return features per company at
multiple horizons**, exposed under opaque `price_feature_NNN` IDs.

## Module

`afp.features.price_features.compute_price_features(samples, prices, cfg)`

Default `PriceFeatureConfig` produces 6 signals per sample:

| Index | Calculation | Horizon (trading days) |
|---|---|---|
| 000 | `log_close[entry-1] - log_close[entry-1-21]` | 21 |
| 001 | same with horizon | 63 |
| 002 | same with horizon | 126 |
| 003 | same with horizon | 252 |
| 004 | `momentum_21 - momentum_63` (relative momentum) | — |
| 005 | `rolling_std(log_return, 63)` (volatility) | — |

Everything is computed strictly before `entry_date` (the function uses
`searchsorted(side="left")` so `entry_date` itself is excluded). Per-feature
median/IQR scaling is fit on training samples only — no validation/test
leakage.

## RFC Compliance

- All output column names are `price_feature_NNN` plus matching
  `price_feature_NNN_missing` flags. The model cannot tell which one is
  momentum vs vol vs relative — they're opaque numeric channels alongside
  the 1100+ anonymous fundamentals.
- The `tests/test_price_features.py::test_no_human_readable_words_in_feature_ids`
  case verifies no `momentum`, `return`, `vol`, `trend`, `rsi` token leaks
  into the column names.
- `tests/test_price_features.py::test_price_features_only_use_past_prices`
  is the structural past-only test: poisoning prices on or after
  `entry_date` does not change any feature value.

## Result on v1000

| Model | Sharpe | Δ vs no-price |
|---|---:|---:|
| Ridge fundamentals only | 0.76 | — |
| **Ridge + 6 price features** | **0.79** | **+0.03** |
| XGB GPU fundamentals only | 0.65 | — |
| XGB GPU + 6 price features | 0.59 | -0.06 |

Price features helped Ridge by ~3% absolute Sharpe (also barely beats SPY
at 0.77 for the first time). XGB GPU *hurt* — the boosted-tree model
overfit the extra dimensions on the small (1842-row) training set, same
pattern seen with derived features in Phase 14.

## Validation Metrics (same vs Phase 11)

Ridge + price:
- direction accuracy 50.8%, Spearman -0.001 (worse than train-only Ridge)
- But test Sharpe higher → confirms again that validation IC doesn't
  translate cleanly to portfolio performance on this universe.

## Acceptance — `tests/test_price_features.py`

1. Disabled config returns no columns.
2. Default config produces exactly 6 signals.
3. **Past-only constraint:** poisoning future prices leaves all features
   identical.
4. Scaler round-trip stays in `[-3, 3]` and adds missing flags.
5. **Anonymity:** no human momentum/return/vol tokens appear in feature
   IDs.
