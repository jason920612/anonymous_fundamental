"""Phase 14: anonymous derived features.

Each base anonymous feature gets two derived counterparts computed as
pure single-feature time-series operations on the period_offset axis:

  derived_yoy_<feature_id>  := value[offset 0] - value[offset 4]
  derived_z_<feature_id>    := (value[offset 0] - mean_0..7) / std_0..7

No cross-feature ratios. The model still sees only anonymous IDs and
numeric values — RFC-03 §4.4 forbids human ratios like ROE / margin /
PE; it does not forbid per-feature temporal transforms because they do
not require knowing what each feature *means*.

The derived features live alongside the originals in the feature matrix
and are only present at `period_offset = 0` (single value per sample).
The encoder transparently handles them as extra columns.
"""

from __future__ import annotations

import numpy as np

from afp.features.anonymous_mapping import AnonymousFeatureMap


def derived_feature_ids(base_ids: list[str]) -> list[str]:
    out = []
    for fid in base_ids:
        out.append(f"derived_yoy_{fid}")
        out.append(f"derived_z_{fid}")
    return out


def compute_derived(values: np.ndarray, missing: np.ndarray,
                    yoy_lookback: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """`values`/`missing` are `[N, P, F]`. Returns `[N, 1, 2F]` derived
    arrays where the first F slots are YoY and the next F are z-score.

    YoY at sample i, feature f = value[i, 0, f] - value[i, yoy_lookback, f]
    z-score at sample i, feature f = (value[i, 0, f] - mean(present 0..P-1, f))
                                      / std(present 0..P-1, f)
    """
    N, P, F = values.shape
    derived = np.full((N, 1, 2 * F), np.nan, dtype=np.float64)
    derived_missing = np.ones((N, 1, 2 * F), dtype=np.int8)

    # YoY block --------------------------------------------------------------
    if yoy_lookback < P:
        v0 = values[:, 0, :]
        v_lag = values[:, yoy_lookback, :]
        m0 = missing[:, 0, :]
        m_lag = missing[:, yoy_lookback, :]
        ok = (m0 == 0) & (m_lag == 0)
        yoy = np.where(ok, v0 - v_lag, np.nan)
        derived[:, 0, :F] = yoy
        derived_missing[:, 0, :F] = np.where(ok, 0, 1)

    # Z-score block -----------------------------------------------------------
    masked = np.where(missing == 0, values, np.nan)         # NaN where missing
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nanmean(masked, axis=1)                       # [N, F]
        sigma = np.nanstd(masked, axis=1, ddof=1)             # [N, F]
        v0 = masked[:, 0, :]
        z = np.where((sigma > 1e-9) & np.isfinite(v0), (v0 - mu) / sigma, np.nan)
    ok = ~np.isnan(z)
    derived[:, 0, F:] = np.where(ok, z, np.nan)
    derived_missing[:, 0, F:] = np.where(ok, 0, 1)

    # NaN → 0 with missing flag set (downstream scaler expects no NaN in values)
    derived = np.where(np.isnan(derived), 0.0, derived)
    return derived, derived_missing


def derived_feature_map(base_map: AnonymousFeatureMap) -> AnonymousFeatureMap:
    """Return a new `AnonymousFeatureMap` whose table covers the derived IDs.

    Used purely as metadata so the encoder save() can round-trip the universe
    of features. Each derived row points back to the underlying anonymous ID.
    """
    import pandas as pd
    base = base_map.table.copy()
    rows = []
    for r in base.itertuples():
        rows.append({"anonymous_feature_id": f"derived_yoy_{r.anonymous_feature_id}",
                     "taxonomy": r.taxonomy, "concept_name": r.concept_name,
                     "unit": r.unit, "mapping_version": base_map.mapping_version,
                     "first_seen_at": r.first_seen_at, "active": True,
                     "derived_kind": "yoy"})
        rows.append({"anonymous_feature_id": f"derived_z_{r.anonymous_feature_id}",
                     "taxonomy": r.taxonomy, "concept_name": r.concept_name,
                     "unit": r.unit, "mapping_version": base_map.mapping_version,
                     "first_seen_at": r.first_seen_at, "active": True,
                     "derived_kind": "z"})
    derived = pd.DataFrame(rows)
    table = pd.concat([base.assign(derived_kind="raw"), derived], ignore_index=True)
    return AnonymousFeatureMap(mapping_version=base_map.mapping_version, table=table)
