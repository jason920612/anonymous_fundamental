# Phase 14 — Anonymous Derived Features

## Why

Phase 11 showed LightGBM trees split on individual `(feature_id, period_offset)`
columns but never seemed to learn time-series patterns across offsets
(Spearman ≈ 0.03 on the 1000-CIK validation set). The raw-value tensor gives
the model 8 lagged snapshots of every anonymous feature, but to recover
"year-over-year change" or "this period vs the trailing mean" the tree has to
learn an interaction across two columns — which is exactly what tree boosters
are bad at without explicit interaction features.

Phase 14 pre-computes those two transforms per-feature and exposes them as new
anonymous columns. The model sees more numeric features but **the same total
information** — and (importantly) **still no human-readable financial meaning**.

## RFC Compliance

RFC-03 §4.4 forbids human-engineered *ratios* (ROE, ROA, margin, PE) because
each requires knowing the meaning of two distinct concepts in order to compose
them. The derived features here are **single-feature time-series operations** —
they're computed independently per anonymous feature ID without any
cross-concept composition. The model still cannot tell `derived_yoy_feature_001`
apart from `derived_yoy_feature_073` except as opaque numbers.

This is the same rationale that allows the model to see `feature_001_o0_value`
and `feature_001_o4_value` separately; the only difference is that we
pre-compute the subtraction.

## Module

`afp.features.derived.compute_derived(values, missing, yoy_lookback=4)`

Input: `values [N, P, F]`, `missing [N, P, F]`  
Output: `derived [N, 1, 2F]`, `derived_missing [N, 1, 2F]`

```
derived[i, 0, f]      = values[i, 0, f] - values[i, yoy_lookback, f]
                          (NaN if either offset is missing)
derived[i, 0, F + f]  = (values[i, 0, f] - mean(values[i, :, f])) / std(values[i, :, f])
                          (NaN if std ≈ 0 or v0 missing)
```

Padding: derived values are slotted at `period_offset = 0` and the rest of the
periods_back depth is zero with missing_flag = 1, so the dense parquet keeps
its `[N, P, F+2F]` shape.

## Encoder Integration

`AnonymousFeatureEncoder(derived_enabled=True, derived_yoy_lookback=4)`

- During `fit`: the derived block gets its own `RobustScaler` (train-only median/IQR).
- During `transform`: the derived block is appended along the feature axis;
  the dense frame names new columns `derived_yoy_<base_id>_o*_value/missing`
  and `derived_z_<base_id>_o*_value/missing`.
- `save` writes `derived_scaling_stats.npz` and remembers
  `derived_enabled` + `derived_yoy_lookback` in `metadata.json`.
- `load` rehydrates everything.

## Config

```yaml
features:
  derived:
    enabled: false               # default off
    yoy_lookback: 4              # 4 quarters back
```

Set `enabled: true` (in `configs/deployment_v1000.yaml` or any override) to
turn on derived features.

## Effect on Model Input

With 1,103 base anonymous features the v1000 model goes from
`1103 * 8 = 8824` numeric columns to `1103 * 8 + 2*1103 = ~11K` columns — a
~25% increase, well within LightGBM/Ridge capacity for the available training
data.

## Acceptance — `tests/test_derived_features.py`

1. `compute_derived` produces correct YoY values for hand-crafted inputs.
2. z-score block matches `(v0 - mean) / std` for a deterministic series.
3. Missing-offset handling is correct (both-present YoY produces a value;
   missing inputs propagate to flags).
4. Encoder with `derived_enabled=True` exposes `derived_yoy_*` and
   `derived_z_*` columns — and the anonymity leakage check still passes
   (no forbidden tokens leak in).
5. `save`/`load` round-trips derived state bit-identically.
